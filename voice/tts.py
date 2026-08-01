"""
voice/tts.py
Text-to-Speech using Microsoft edge-tts (Vietnamese + English neural voices), with an
optional F5-TTS-Vietnamese-ViVoice backend (zero-shot, runs on Colab — see
colab/voice_server.ipynb) selected via a "f5tts:" -prefixed voice id.
Produces base audio that is optionally passed on to RVC for voice conversion.
"""

import asyncio
import os
import tempfile

import edge_tts

DEFAULT_VOICE = os.getenv("TTS_VOICE", "vi-VN-HoaiMyNeural")
F5TTS_PREFIX  = "f5tts:"

# Safety cap only — RAG answers are already capped at ~200 words by the prompt
# (engine/rag_engine.py build_prompt). This just guards against a runaway answer
# producing an excessively long spoken clip.
MAX_SPEECH_WORDS = 400


async def synthesize(text: str, voice: str = None) -> tuple[bytes, str]:
    """
    Convert text to speech.

    Parameters
    ----------
    text  : str   Text to speak (Vietnamese or English)
    voice : str   edge-TTS voice id, or "f5tts:<id>" to use the F5-TTS-Vietnamese-ViVoice
                   baseline running on Colab (defaults to DEFAULT_VOICE)

    Returns
    -------
    (bytes, str)   Raw audio bytes and their mime type ("audio/mpeg" or "audio/wav") —
                    the mime reflects what was actually produced, since F5-TTS falls
                    back to edge-TTS when Colab is unreachable.
    """
    voice = voice or DEFAULT_VOICE
    text  = truncate_for_speech(text, MAX_SPEECH_WORDS)

    if voice.startswith(F5TTS_PREFIX):
        wav_bytes = await _synthesize_f5tts(text)
        if wav_bytes is not None:
            return wav_bytes, "audio/wav"
        # Colab unreachable / endpoint not configured — degrade to edge-TTS.

    return await _synthesize_edge_tts(text, DEFAULT_VOICE if voice.startswith(F5TTS_PREFIX) else voice), "audio/mpeg"


async def _synthesize_edge_tts(text: str, voice: str) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        tmp_path = f.name

    try:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(tmp_path)
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


async def _synthesize_f5tts(text: str) -> bytes | None:
    """Returns WAV bytes from the Colab F5-TTS baseline, or None if unavailable."""
    from voice import rvc_client  # local import: keeps rvc_client optional for edge-TTS-only setups
    return await asyncio.to_thread(rvc_client.synthesize_f5tts, text)


def truncate_for_speech(text: str, max_words: int = MAX_SPEECH_WORDS) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " … (xem toàn bộ câu trả lời trong khung chat)"
