"""
voice/stt.py
Speech-to-Text using OpenAI Whisper — the input half of this service's voice
loop (the output half is voice/tts.py + voice/rvc_client.py). A client app
posts recorded microphone audio to POST /api/transcribe (see app.py) and gets
back plain text to feed into its own assistant as a normal text query — this
module never touches the client's RAG/LLM pipeline.

Whisper pulls in torch, already a dependency via resemblyzer (see
engine/realism_engine.py) — the model itself is loaded lazily on first use
rather than at app.py startup, so a deployment that never calls
/api/transcribe pays no import/load cost. Decoding non-.wav audio (webm/ogg/
m4a/mp3, the formats a browser's MediaRecorder produces) requires an `ffmpeg`
binary on PATH, same requirement already documented for realism_engine.
"""

import os
import tempfile

_model = None
_MODEL_SIZE = os.getenv("WHISPER_MODEL", "small")  # tiny | base | small | medium

_SUFFIX_BY_MIME = {
    "webm": ".webm",
    "ogg":  ".ogg",
    "mp4":  ".m4a",
    "m4a":  ".m4a",
    "mpeg": ".mp3",
    "mp3":  ".mp3",
    "wav":  ".wav",
}


def _load_model():
    global _model
    if _model is None:
        import whisper
        print(f"[STT] Loading Whisper-{_MODEL_SIZE} …")
        _model = whisper.load_model(_MODEL_SIZE)
        print("[STT] Whisper ready.")
    return _model


def _suffix_for(mime: str) -> str:
    mime = (mime or "").lower()
    for key, suffix in _SUFFIX_BY_MIME.items():
        if key in mime:
            return suffix
    return ".webm"  # MediaRecorder's default container when nothing else matches


def transcribe(audio_bytes: bytes, mime: str = "audio/webm", language: str = None) -> dict:
    """
    Parameters
    ----------
    audio_bytes : bytes  Raw audio, typically from a browser MediaRecorder.
    mime        : str    MIME type hint, used only to pick a temp-file suffix —
                          Whisper shells out to ffmpeg, which sniffs the actual
                          format from file contents.
    language    : str    ISO 639-1 code (e.g. "vi") to force a language, or
                          None to let Whisper auto-detect.

    Returns
    -------
    dict {"text": str, "language": str, "segments": [{"start", "end", "text"}, ...]}
    """
    model = _load_model()

    with tempfile.NamedTemporaryFile(suffix=_suffix_for(mime), delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    try:
        result = model.transcribe(
            tmp_path,
            language=language or None,
            task="transcribe",
            fp16=False,  # CPU-safe; harmless no-op on CUDA where fp16 is autodetected anyway
        )
        return {
            "text":     result["text"].strip(),
            "language": result.get("language", "unknown"),
            "segments": [
                {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
                for s in result.get("segments", [])
            ],
            "engine": f"whisper-local:{_MODEL_SIZE}",
        }
    finally:
        os.unlink(tmp_path)
