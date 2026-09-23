"""
voice/stt_segmented.py
Splits a recording longer than SEGMENT_MS into fixed-length pieces and
transcribes each through the station's normal remote-then-local fallback,
instead of sending the whole recording in one request. Rejects recordings
over MAX_DURATION_MS. Full rationale: ai_change_log.txt.
"""
import io
import logging
import os

from voice import rvc_client, stt

logger = logging.getLogger("voice-station")

SEGMENT_MS = int(os.getenv("STT_SEGMENT_MS", "25000"))                     # under Whisper's 30s single-pass window (see ai_change_log.txt)
MAX_DURATION_MS = int(os.getenv("STT_MAX_DURATION_MS", str(10 * 60_000)))  # 10 minutes


def _segment_wav_bytes(chunk) -> bytes:
    buf = io.BytesIO()
    chunk.export(buf, format="wav")
    return buf.getvalue()


def _transcribe_one(audio_bytes: bytes, mime: str, language: str) -> dict:
    """Remote-then-local fallback for one segment's worth of audio."""
    result = rvc_client.transcribe_remote(audio_bytes, mime, language)
    if result is None:
        result = stt.transcribe(audio_bytes, mime, language)
    return result


def transcribe_long(audio_bytes: bytes, mime: str, language: str = None) -> dict:
    """Transcribes a recording of any length, segmenting it if needed."""
    from pydub import AudioSegment

    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    if len(audio) > MAX_DURATION_MS:
        raise ValueError(
            f"Bản ghi dài {len(audio) / 60_000:.1f} phút, vượt giới hạn "
            f"{MAX_DURATION_MS // 60_000} phút cho một lần phiên âm."
        )

    if len(audio) <= SEGMENT_MS:
        return _transcribe_one(audio_bytes, mime, language)

    texts = []
    for start_ms in range(0, len(audio), SEGMENT_MS):
        chunk_bytes = _segment_wav_bytes(audio[start_ms:start_ms + SEGMENT_MS])
        try:
            result = _transcribe_one(chunk_bytes, "audio/wav", language)
            if result and result.get("text"):
                texts.append(result["text"])
        except Exception as e:
            logger.warning(f"[STT-segmented] Segment at {start_ms}ms failed: {e}")

    if not texts:
        raise RuntimeError("Không đoạn nào trong bản ghi được nhận diện thành công.")

    return {
        "text": " ".join(texts).strip(),
        "language": "vi",
        "segments": [],
        "engine": "segmented",
    }
