"""
voice/stt.py
Speech-to-Text using PhoWhisper (VinAI's Vietnamese fine-tune of Whisper) —
the input half of this service's voice loop (the output half is voice/tts.py
+ voice/rvc_client.py). A client app posts recorded microphone audio to
POST /api/transcribe (see app.py) and gets back plain text to feed into its
own assistant as a normal text query — this module never touches the
client's RAG/LLM pipeline.

This is the LAST-RESORT fallback in /api/transcribe's chain (published/
default Tier 2 adapter -> Colab-hosted PhoWhisper-large -> here), reached
only when both of those are unavailable -- so it deliberately uses the
*small* PhoWhisper checkpoint, not -large: it has to run acceptably on this
host's CPU with no GPU guaranteed, and small is meaningfully lighter while
still being the same Vietnamese-tuned family as the Colab primary path,
instead of generic multilingual openai-whisper (used here until this was
changed) which measurably underperforms PhoWhisper on Vietnamese speech (see
thesis pilot results: PhoWhisper-large 11.7% WER vs local Whisper-small
21.2% WER on identical audio) -- every fallback step should still be
Vietnamese-tuned, just progressively smaller/faster.

Loaded via transformers (WhisperForConditionalGeneration + WhisperProcessor),
not the openai-whisper package -- same HF model class stt_adapter_infer.py
already uses for Tier 2 LoRA inference, so this shares that decoding
approach (beam search + anti-repetition kwargs) instead of introducing a
second, differently-behaved Whisper stack. transformers/torch are already
hard dependencies (Tier 2 adapters, resemblyzer). Decoding non-.wav audio
(webm/ogg/m4a/mp3, the formats a browser's MediaRecorder produces) goes
through librosa, same as stt_adapter_infer.py -- no ffmpeg subprocess shell-
out here, sidestepping the conda-forge ffmpeg DLL crash entirely rather than
working around it.

Only Vietnamese is supported (PhoWhisper is Vietnamese-only) -- the
`language` parameter is accepted for interface parity with the other
transcribe() implementations this can be swapped with, but ignored, same as
clone_voice_client.local_stt's transcribe_with_lora().
"""

import os
import tempfile
import threading

from engine.server_log import get_logger

logger = get_logger()

# Bundled static ffmpeg (see bin/, gitignored) -- librosa's audioread
# fallback still shells out to ffmpeg for non-.wav browser recordings
# (webm/ogg/m4a/mp3), same as stt_adapter_infer.py's own copy of this block.
# The conda-forge ffmpeg package in rag_env fails to launch on this machine
# (STATUS_ENTRYPOINT_NOT_FOUND, a DLL conflict with its dynamically-linked
# build); this static build sidesteps that. Falls back to PATH's ffmpeg if
# the bundled binary isn't present (e.g. a different deployment).
_BUNDLED_FFMPEG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if os.path.isfile(os.path.join(_BUNDLED_FFMPEG_DIR, "ffmpeg.exe")):
    os.environ["PATH"] = _BUNDLED_FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")

_MODEL_NAME = os.getenv("PHOWHISPER_LOCAL_MODEL", "vinai/PhoWhisper-small")
_GENERATE_KWARGS = {"no_repeat_ngram_size": 3, "repetition_penalty": 1.3, "num_beams": 5}

_model = None
_processor = None
_device = None
_model_lock = threading.Lock()

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
    global _model, _processor, _device
    if _model is None:
        # transcribe() runs off the request thread via asyncio.to_thread, so
        # overlapping requests (e.g. live-transcribe's 1s local cadence hitting
        # a cold start) can race this check concurrently -- lock + re-check
        # so only the first one actually loads the checkpoint.
        with _model_lock:
            if _model is None:
                import torch
                from transformers import WhisperForConditionalGeneration, WhisperProcessor

                _device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"[STT] Loading {_MODEL_NAME} ({_device}) …")
                _processor = WhisperProcessor.from_pretrained(_MODEL_NAME, language="vietnamese", task="transcribe")
                _model = WhisperForConditionalGeneration.from_pretrained(_MODEL_NAME)
                _model.generation_config.language = "vietnamese"
                _model.generation_config.task = "transcribe"
                _model.to(_device)
                _model.eval()
                logger.info("[STT] PhoWhisper ready.")
    return _model, _processor, _device


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
    mime        : str    MIME type hint, used only to pick a temp-file suffix.
    language    : str    Accepted for interface parity with other transcribe()
                          implementations; ignored (PhoWhisper is Vietnamese-only).

    Returns
    -------
    dict {"text": str, "language": "vi", "segments": [], "engine": str}
    """
    import librosa
    import torch

    model, processor, device = _load_model()

    with tempfile.NamedTemporaryFile(suffix=_suffix_for(mime), delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name
    try:
        audio, _ = librosa.load(tmp_path, sr=16000, mono=True)
    finally:
        os.unlink(tmp_path)

    input_features = processor.feature_extractor(
        audio, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(device)
    with torch.no_grad():
        predicted_ids = model.generate(input_features, **_GENERATE_KWARGS)
    text = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

    return {
        "text": text,
        "language": "vi",
        "segments": [],
        "engine": f"phowhisper-local:{_MODEL_NAME}",
    }
