# PhoWhisper (VinAI) on faster-whisper / CTranslate2 — Runpod Serverless
# Author: ChatGPT for Minh (2025-10-29)
import os, uuid, time, json, logging, tempfile, subprocess, math
from typing import Dict, Any, List, Optional
import runpod
import requests
from faster_whisper import WhisperModel

# -----------------------------
# ENV
# -----------------------------
MODEL_ID     = os.getenv("MODEL_ID", "kiendt/PhoWhisper-large-ct2")
MODEL_DIR    = os.getenv("MODEL_DIR", "/models")
OUT_DIR      = os.getenv("OUT_DIR", "/runpod-volume/out")
DEVICE       = os.getenv("DEVICE", "cuda")                 # cuda | cpu
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float16")        # float16 | int8_float16 | int8
LANG         = os.getenv("LANG", "vi")                     # "vi" để khóa tiếng Việt; "auto" = auto detect
VAD_FILTER   = os.getenv("VAD_FILTER", "1") == "1"         # bật VAD để cắt lặng
MAX_CHUNK    = float(os.getenv("MAX_CHUNK_LEN", "30"))     # giây; chunk_length cho audio dài
MAKE_SRT     = os.getenv("SRT", "1") == "1"
MAKE_VTT     = os.getenv("VTT", "0") == "1"

os.makedirs(OUT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("PhoWhisper-Serverless")

# Lazy model
_model: Optional[WhisperModel] = None
def get_model() -> WhisperModel:
    global _model
    if _model is None:
        log.info(f"[MODEL] Loading: {MODEL_ID} (device={DEVICE}, compute_type={COMPUTE_TYPE})")
        _model = WhisperModel(
            MODEL_ID,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
            download_root=MODEL_DIR
        )
    return _model

# -----------------------------
# Utilities
# -----------------------------
def _download(url: str, to_path: str, timeout: int = 180):
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(to_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    return to_path

def _to_wav_16k_mono(src_path: str, dst_path: str):
    """Convert any audio to 16kHz mono s16 with ffmpeg"""
    cmd = [
        "ffmpeg", "-y", "-i", src_path,
        "-ac", "1", "-ar", "16000", "-sample_fmt", "s16",
        dst_path
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg convert failed: {res.stderr[-500:]}")
    return dst_path

def _fmt_ts(sec: float) -> str:
    if sec < 0: sec = 0.0
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int((sec - math.floor(sec)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def _write_srt(segments: List[Dict[str, Any]], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n{_fmt_ts(seg['start'])} --> {_fmt_ts(seg['end'])}\n{seg['text'].strip()}\n\n")

def _write_vtt(segments: List[Dict[str, Any]], path: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for seg in segments:
            s = _fmt_ts(seg["start"]).replace(",", ".")
            e = _fmt_ts(seg["end"]).replace(",", ".")
            f.write(f"{s} --> {e}\n{seg['text'].strip()}\n\n")

def _normalize_input(inp: Dict[str, Any]) -> str:
    """
    Accept either:
      - audio_path: local path (mounted volume)
      - audio_url:  http(s) to download then transcribe
    Returns a local path to a 16k mono WAV.
    """
    audio_path = inp.get("audio_path")
    audio_url  = inp.get("audio_url")
    if not audio_path and not audio_url:
        raise ValueError("Provide 'audio_path' or 'audio_url'")

    tmp_dir = tempfile.mkdtemp(prefix="pho_")
    raw_path = os.path.join(tmp_dir, "raw.input")
    if audio_path:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"audio_path not found: {audio_path}")
        raw_path = audio_path
    else:
        _download(audio_url, raw_path)

    fixed = os.path.join(tmp_dir, "fixed_16k.wav")
    return _to_wav_16k_mono(raw_path, fixed)

# -----------------------------
# Handler
# -----------------------------
def handler(event):
    """
    Input JSON:
    {
      "audio_path": "/runpod-volume/audio/test.wav",   # hoặc
      "audio_url": "https://.../sample.mp3",
      "beam_size": 5,
      "temperature": 0.0,
      "word_timestamps": true,        # default: true
      "return": "json|text",          # default: json
      "outfile_prefix": "result_xyz"  # mặc định auto theo job_id
    }
    """
    t0 = time.time()
    inp = event.get("input", {}) if isinstance(event, dict) else {}
    try:
        wav16 = _normalize_input(inp)
    except Exception as e:
        return {"error": f"Input error: {e}"}

    # Options
    beam_size  = int(inp.get("beam_size", 5))
    temperature = float(inp.get("temperature", 0.0))
    word_ts    = bool(inp.get("word_timestamps", True))
    lang = None if str(LANG).lower() == "auto" else LANG

    # Prepare output paths
    job_id = str(uuid.uuid4())
    prefix = inp.get("outfile_prefix") or job_id
    json_path = os.path.join(OUT_DIR, f"{prefix}.json")
    srt_path  = os.path.join(OUT_DIR, f"{prefix}.srt")
    vtt_path  = os.path.join(OUT_DIR, f"{prefix}.vtt")
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load + transcribe
    try:
        model = get_model()
        segments_gen, info = model.transcribe(
            wav16,
            language=lang,
            task="transcribe",
            vad_filter=VAD_FILTER,
            beam_size=beam_size,
            temperature=temperature,
            word_timestamps=word_ts,
            chunk_length=MAX_CHUNK  # xử lý file dài ổn định
        )
    except Exception as e:
        return {"error": f"Transcribe failed: {e}"}

    # Collect
    segments = []
    words_all = []
    texts = []
    for seg in segments_gen:
        item = {
            "id": seg.id,
            "start": float(seg.start),
            "end": float(seg.end),
            "text": seg.text.strip()
        }
        if word_ts and seg.words:
            item["words"] = [
                {"start": float(w.start), "end": float(w.end), "word": w.word}
                for w in seg.words
            ]
            words_all.extend(item["words"])
        segments.append(item)
        texts.append(item["text"])

    result = {
        "job_id": job_id,
        "model_id": MODEL_ID,
        "language": info.language,
        "language_probability": float(info.language_probability),
        "elapsed_sec": round(time.time() - t0, 2),
        "num_segments": len(segments),
        "num_words": len(words_all),
        "text": " ".join(texts).strip(),
        "segments": segments,
        "outputs": {"json": json_path, "srt": None, "vtt": None}
    }

    # Save JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # Optional subtitles
    try:
        if MAKE_SRT:
            _write_srt(segments, srt_path)
            result["outputs"]["srt"] = srt_path
        if MAKE_VTT:
            _write_vtt(segments, vtt_path)
            result["outputs"]["vtt"] = vtt_path
    except Exception as e:
        log.warning(f"Subtitle write failed: {e}")

    # Return mode
    ret_mode = inp.get("return", "json")
    if ret_mode == "text":
        return {
            "job_id": job_id, "elapsed_sec": result["elapsed_sec"],
            "language": result["language"],
            "path_json": json_path, "path_srt": result["outputs"]["srt"],
            "text": result["text"]
        }
    return result

# Start serverless worker
runpod.serverless.start({"handler": handler})
