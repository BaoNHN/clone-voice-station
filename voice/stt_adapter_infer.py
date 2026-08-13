"""
voice/stt_adapter_infer.py
Tier-2 STT inference: loads a manager-published stt_adapters row's LoRA
adapter on top of its HF Whisper base model and transcribes with it, for
POST /api/transcribe callers whose client app has an adapter published (see
database.get_published_stt_adapter_for_client() / app.py's transcribe_route).
When nothing is published for the calling client, or this raises, the caller
falls back to the existing Colab/local base-Whisper path (voice/stt.py) --
same degrade-gracefully contract that path already has.

Uses transformers' WhisperForConditionalGeneration + peft (matching
voice/stt_local_train.py's training stack, NOT the openai-whisper package
voice/stt.py uses) -- an adapter trained via stt_local_train.py can only be
loaded back with the same model classes. Loaded lazily and cached per
adapter_path so repeated calls to the same published adapter don't reload the
base model + LoRA weights every request; a single global lock serializes both
loading and generate() calls -- this deployment is single-worker (same
thesis-scale assumption already documented in app.py's login-rate-limit
comment), so a real per-model concurrency scheme would be premature.
"""

import os
import tempfile
import threading

from engine.server_log import get_logger

logger = get_logger()

# Bundled static ffmpeg (see voice/stt.py's own copy of this block for why --
# same DLL-conflict workaround, needed here too since librosa's audioread
# fallback shells out to ffmpeg for non-wav browser recordings).
_BUNDLED_FFMPEG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if os.path.isfile(os.path.join(_BUNDLED_FFMPEG_DIR, "ffmpeg.exe")):
    os.environ["PATH"] = _BUNDLED_FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")

# Mirrors voice/stt_local_train.py's own _HF_MODEL_BY_NAME/GENERATE_KWARGS --
# an adapter must be loaded and decoded the same way it was trained/gated.
_HF_MODEL_BY_NAME = {
    "phowhisper-small": "vinai/PhoWhisper-small",
    "whisper-tiny": "openai/whisper-tiny",
    "whisper-base": "openai/whisper-base",
}
GENERATE_KWARGS = {"no_repeat_ngram_size": 3, "repetition_penalty": 1.3, "num_beams": 5}

_SUFFIX_BY_MIME = {
    "webm": ".webm", "ogg": ".ogg", "mp4": ".m4a", "m4a": ".m4a",
    "mpeg": ".mp3", "mp3": ".mp3", "wav": ".wav",
}

_lock = threading.Lock()
_cache = {}  # adapter_path -> (processor, model, device)


def _suffix_for(mime: str) -> str:
    mime = (mime or "").lower()
    for key, suffix in _SUFFIX_BY_MIME.items():
        if key in mime:
            return suffix
    return ".webm"  # MediaRecorder's default container when nothing else matches


def _load(adapter: dict):
    adapter_path = adapter["adapter_path"]
    cached = _cache.get(adapter_path)
    if cached:
        return cached

    import torch
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    hf_model_name = _HF_MODEL_BY_NAME.get(adapter["base_model"])
    if not hf_model_name:
        raise ValueError(f"Unsupported base_model for adapter inference: {adapter['base_model']!r}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[STT-adapter] Đang tải {hf_model_name} + LoRA từ {adapter_path} ({device})…")
    processor = WhisperProcessor.from_pretrained(hf_model_name, language="vietnamese", task="transcribe")
    base = WhisperForConditionalGeneration.from_pretrained(hf_model_name)
    base.generation_config.language = "vietnamese"
    base.generation_config.task = "transcribe"
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
    model.to(device)
    model.eval()
    logger.info(f"[STT-adapter] Sẵn sàng: adapter #{adapter['id']} ({adapter['name']}).")

    entry = (processor, model, device)
    _cache[adapter_path] = entry
    return entry


def transcribe(adapter: dict, audio_bytes: bytes, mime: str = "audio/webm", language: str = None) -> dict:
    """Same return shape as voice/stt.py's transcribe() -- {"text","language","segments",
    "engine"} -- so app.py's transcribe_route can use either interchangeably. `language` is
    accepted for interface parity with the base path but ignored: the adapter was trained
    assuming Vietnamese (see stt_local_train.py's fixed language="vietnamese"), so honoring a
    different requested language here would silently produce nonsense output."""
    import librosa
    import torch

    with tempfile.NamedTemporaryFile(suffix=_suffix_for(mime), delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name
    try:
        audio, _ = librosa.load(tmp_path, sr=16000, mono=True)
    finally:
        os.unlink(tmp_path)

    with _lock:
        processor, model, device = _load(adapter)
        input_features = processor.feature_extractor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device)
        with torch.no_grad():
            predicted_ids = model.generate(input_features, **GENERATE_KWARGS)
        text = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

    return {
        "text": text,
        "language": "vi",
        "segments": [],
        "engine": f"stt-adapter:{adapter['id']}:{adapter['base_model']}",
    }
