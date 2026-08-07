"""
voice/stt_local_train.py
Local (this-machine) LoRA fine-tuning of Whisper for the STT Lab's Tier 2 —
used by engine/stt_train_engine.py when a guest picks "local" (or "auto" falls
back to it because Colab, see voice/stt_client.py, is unset/unreachable).

Uses transformers' WhisperForConditionalGeneration, not the openai-whisper
package voice/stt.py/clone_voice_client's Tier-1 inference path uses — PEFT
only attaches to the HF model class. A manual PyTorch loop, not
transformers.Trainer, keeps this dependency-light (no accelerate/Trainer
config surface) and gives a clean per-step hook for progress_cb, same
convention as voice/rvc_local.py's training function.

Restricted to whisper-tiny/whisper-base (see database.ALLOWED_STT_BASE_MODELS)
-- this machine's GPU has ~4GB VRAM and this is a public, no-API-key-gated
page, so batch_size stays at 1 and there's no attempt at anything heavier.

Everything heavy (torch, transformers, peft, librosa) is imported lazily
inside train(), same pattern as voice/stt.py's Whisper load.
"""

import os

from engine.server_log import get_logger

logger = get_logger()

_HF_MODEL_BY_NAME = {
    "whisper-tiny": "openai/whisper-tiny",
    "whisper-base": "openai/whisper-base",
}

EPOCHS = 3
LEARNING_RATE = 1e-3
LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]


def train(adapter_id: int, samples: list, base_model: str, output_dir: str,
          resume_from_path: str = None, progress_cb=None) -> str:
    """
    Parameters
    ----------
    adapter_id       : int   For logging only.
    samples          : list  [{"audio_path": str, "reference_text": str}, ...]
    base_model       : str   One of database.ALLOWED_STT_BASE_MODELS ("whisper-tiny"/"whisper-base").
    output_dir       : str   Where to save the trained adapter (created if missing).
    resume_from_path : str   Directory of a previously-saved adapter (adapter_model.safetensors +
                              adapter_config.json) to continue training from, or None to start fresh.
    progress_cb       : callable(str)  Called with a human-readable status after every sample.

    Returns
    -------
    str  output_dir, once the adapter is saved there.
    """
    hf_model_name = _HF_MODEL_BY_NAME.get(base_model)
    if not hf_model_name:
        raise ValueError(f"Unsupported base_model for local training: {base_model!r}")

    import librosa
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    def _report(msg: str):
        logger.info(f"[STT-train:{adapter_id}] {msg}")
        if progress_cb:
            progress_cb(msg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"

    _report(f"Đang tải model nền {hf_model_name} ({device})…")
    processor = WhisperProcessor.from_pretrained(hf_model_name, language="vietnamese", task="transcribe")
    base = WhisperForConditionalGeneration.from_pretrained(hf_model_name)
    base.generation_config.language = "vietnamese"
    base.generation_config.task = "transcribe"

    if resume_from_path and os.path.isdir(resume_from_path):
        _report(f"Tiếp tục huấn luyện từ adapter đã có ({resume_from_path})…")
        model = PeftModel.from_pretrained(base, resume_from_path, is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=LORA_TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
        )
        model = get_peft_model(base, lora_config)
    model.to(device)
    model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    n = len(samples)
    for epoch in range(EPOCHS):
        for i, sample in enumerate(samples):
            audio, _ = librosa.load(sample["audio_path"], sr=16000, mono=True)
            input_features = processor.feature_extractor(
                audio, sampling_rate=16000, return_tensors="pt"
            ).input_features.to(device)
            labels = processor.tokenizer(sample["reference_text"], return_tensors="pt").input_ids.to(device)

            optimizer.zero_grad()
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    loss = model(input_features=input_features, labels=labels).loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = model(input_features=input_features, labels=labels).loss
                loss.backward()
                optimizer.step()

            _report(f"Epoch {epoch + 1}/{EPOCHS}, mẫu {i + 1}/{n}, loss={loss.item():.3f}")

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    _report("Đã lưu adapter.")
    return output_dir
