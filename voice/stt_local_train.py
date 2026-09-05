"""
voice/stt_local_train.py
Local (this-machine) LoRA fine-tuning of Whisper for the STT Lab's Tier 2,
used by engine/stt_train_engine.py when a guest picks "local" (or "auto"
falls back to it). Uses transformers' WhisperForConditionalGeneration (PEFT
requirement) with a manual training loop instead of transformers.Trainer.

Ships an adapter only if it beats the untrained base model's WER on a
held-out test split -- never blindly saves the last epoch.
"""

import os
import random

from engine.server_log import get_logger

logger = get_logger()

_HF_MODEL_BY_NAME = {
    "phowhisper-small": "vinai/PhoWhisper-small",
    "whisper-tiny": "openai/whisper-tiny",
    "whisper-base": "openai/whisper-base",
}

# 1e-4 is the standard LoRA-fine-tune-of-Whisper learning rate (1e-3
# drove the adapter into degenerate repeated-token output).
LEARNING_RATE = 1e-4
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
# Broadened from just q_proj/v_proj to give the validation gate below
# something real to catch on higher-capacity adapters.
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj"]

# Training policy scales with upload size: fewer samples get a low epoch
# ceiling and tight patience (fail fast rather than overfit); more samples
# support a higher LoRA rank and more epochs.
_POLICY_BUCKETS = [
    # (min_samples, max_epochs, patience, lora_r)
    (10,  5,  2, 8),
    (50,  10, 3, 8),
    (200, 20, 5, 16),
]


def _training_policy(n_samples: int) -> dict:
    max_epochs, patience, lora_r = _POLICY_BUCKETS[0][1:]
    for min_samples, e, p, r in _POLICY_BUCKETS:
        if n_samples >= min_samples:
            max_epochs, patience, lora_r = e, p, r
    return {"max_epochs": max_epochs, "patience": patience, "lora_r": lora_r}

# TEST_* is the final-gate split (accept/reject decision); VAL_* is the
# early-stopping split, carved from what's left. Fraction-based so both
# scale with upload size.
TEST_FRACTION  = 0.25
TEST_MIN_COUNT = 3
TEST_MAX_COUNT = 60
VAL_FRACTION   = 0.15
VAL_MIN_COUNT  = 3
VAL_MAX_COUNT  = 30
# Decode-quality guard for the bare HF transformers generate() call --
# matters as much as the LoRA training itself for this model class.
GENERATE_KWARGS = {"no_repeat_ngram_size": 3, "repetition_penalty": 1.3, "num_beams": 5}


def _normalize(text: str) -> list:
    import re
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    return text.split()


def _edit_distance(reference: str, hypothesis: str) -> tuple:
    """Word-level Levenshtein distance -- (edits, ref_word_count). Same definition
    tools/eval_stt_wer.py and voice-lab-example/tools/test_medical_lora_wer.py use,
    duplicated here rather than imported (this module has no dependency on tools/)."""
    ref = _normalize(reference)
    hyp = _normalize(hypothesis)
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[n][m], max(n, 1)


def _split_train_val(samples: list) -> tuple:
    """train_samples, val_samples (early-stopping) -- used instead of
    _split_train_val_test() when the caller supplies its own holdout_samples (a
    genuinely independent test set), so there's no need to also carve TEST_* out of
    this pool. Same fixed seed/VAL_* sizing as the three-way split, for consistency."""
    shuffled = samples[:]
    random.Random(42).shuffle(shuffled)
    val_count = min(VAL_MAX_COUNT, max(VAL_MIN_COUNT, round(len(shuffled) * VAL_FRACTION)))
    val_count = min(val_count, len(shuffled) - 1)  # always leave at least 1 for training
    val_samples = shuffled[:val_count]
    train_samples = shuffled[val_count:]
    return train_samples, val_samples


def _split_train_val_test(samples: list) -> tuple:
    """train_samples, val_samples (early-stopping), test_samples (final gate) -- three
    disjoint sets, see module docstring's "three-way split" note for why. Fixed seed:
    reproducible split for the same sample set, not a security boundary."""
    shuffled = samples[:]
    random.Random(42).shuffle(shuffled)

    test_count = min(TEST_MAX_COUNT, max(TEST_MIN_COUNT, round(len(shuffled) * TEST_FRACTION)))
    test_count = min(test_count, len(shuffled) - 1)  # always leave at least 1 for train+val
    test_samples = shuffled[:test_count]
    remainder = shuffled[test_count:]

    val_count = min(VAL_MAX_COUNT, max(VAL_MIN_COUNT, round(len(remainder) * VAL_FRACTION)))
    val_count = min(val_count, len(remainder) - 1)  # always leave at least 1 for training
    val_samples = remainder[:val_count]
    train_samples = remainder[val_count:]

    return train_samples, val_samples, test_samples


def _transcribe_one(model, processor, device, audio_path: str) -> str:
    import librosa
    import torch

    audio, _ = librosa.load(audio_path, sr=16000, mono=True)
    input_features = processor.feature_extractor(
        audio, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(device)
    with torch.no_grad():
        predicted_ids = model.generate(input_features, **GENERATE_KWARGS)
    return processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()


def _evaluate_wer(model, processor, device, eval_samples: list) -> float:
    """Scores `model` against `eval_samples` -- caller passes either the early-stop val
    split or the final-gate test split (see module docstring); this function has no
    opinion on which, it just computes WER on whatever it's given."""
    was_training = model.training
    model.eval()
    total_edits = total_ref_words = 0
    for sample in eval_samples:
        hypothesis = _transcribe_one(model, processor, device, sample["audio_path"])
        edits, ref_words = _edit_distance(sample["reference_text"], hypothesis)
        total_edits += edits
        total_ref_words += ref_words
    if was_training:
        model.train()
    return 100 * total_edits / max(total_ref_words, 1)


def train(adapter_id: int, samples: list, base_model: str, output_dir: str,
          resume_from_path: str = None, progress_cb=None, holdout_samples: list = None) -> str:
    """
    Parameters
    ----------
    adapter_id       : int   For logging only.
    samples          : list  [{"audio_path": str, "reference_text": str}, ...].
    base_model       : str   One of database.ALLOWED_STT_BASE_MODELS.
    output_dir       : str   Where to save the trained adapter (created if missing).
    resume_from_path : str   Directory of a previously-saved adapter to continue from, or None.
    progress_cb       : callable(str)  Called with a human-readable status after every sample.
    holdout_samples   : list  Optional independent test set (see module docstring); when
                              None, a final-gate split is carved from `samples` instead.

    Returns
    -------
    str  output_dir, once the selected adapter is saved there.

    Raises
    ------
    RuntimeError  if the selected epoch doesn't beat the base model's WER on the
                  final-gate split -- nothing is saved in that case.
    """
    hf_model_name = _HF_MODEL_BY_NAME.get(base_model)
    if not hf_model_name:
        raise ValueError(f"Unsupported base_model for local training: {base_model!r}")

    import copy

    import librosa
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
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
    base.to(device)

    if holdout_samples:
        train_samples, val_samples = _split_train_val(samples)
        test_samples = holdout_samples
        test_kind = "mẫu test độc lập (holdout, chưa từng thấy trong lúc train)"
    else:
        train_samples, val_samples, test_samples = _split_train_val_test(samples)
        test_kind = "mẫu kiểm định cuối (trích từ dữ liệu tải lên)"

    policy = _training_policy(len(samples))
    _report(f"Chia {len(train_samples)} mẫu huấn luyện / {len(val_samples)} mẫu early-stop / "
            f"{len(test_samples)} {test_kind} (chính sách: tối đa {policy['max_epochs']} "
            f"epoch, patience {policy['patience']}, LoRA r={policy['lora_r']}, dựa trên "
            f"{len(samples)} mẫu)…")

    # Baseline to beat: scored before get_peft_model() attaches LoRA layers
    # (which monkeypatches base's submodules in place), so this must run first.
    _report(f"Đo WER cơ sở của model gốc (chưa huấn luyện) trên {test_kind}…")
    base_wer = _evaluate_wer(base, processor, device, test_samples)
    _report(f"WER cơ sở (model gốc): {base_wer:.1f}%")

    if resume_from_path and os.path.isdir(resume_from_path):
        _report(f"Tiếp tục huấn luyện từ adapter đã có ({resume_from_path})…")
        model = PeftModel.from_pretrained(base, resume_from_path, is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=policy["lora_r"], lora_alpha=LORA_ALPHA, target_modules=LORA_TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
        )
        model = get_peft_model(base, lora_config)
    model.to(device)
    model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    max_epochs = policy["max_epochs"]
    patience = policy["patience"]
    best_wer = None
    best_state = None
    best_epoch = 0
    epochs_since_improvement = 0
    n = len(train_samples)
    for epoch in range(max_epochs):
        for i, sample in enumerate(train_samples):
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

            _report(f"Epoch {epoch + 1}/{max_epochs}, mẫu {i + 1}/{n}, loss={loss.item():.3f}")

        # Compared only against the best early-stop WER so far, not base_wer
        # (a different, disjoint split) -- see module docstring.
        epoch_wer = _evaluate_wer(model, processor, device, val_samples)
        if best_wer is None or epoch_wer < best_wer:
            _report(f"Epoch {epoch + 1}/{max_epochs} — WER early-stop: {epoch_wer:.1f}% "
                     f"— tốt nhất cho tới nay.")
            best_wer, best_epoch, epochs_since_improvement = epoch_wer, epoch + 1, 0
            best_state = copy.deepcopy(get_peft_model_state_dict(model))
        else:
            epochs_since_improvement += 1
            _report(f"Epoch {epoch + 1}/{max_epochs} — WER early-stop: {epoch_wer:.1f}% "
                     f"— không cải thiện so với epoch {best_epoch} "
                     f"({epochs_since_improvement}/{patience})…")
            if epochs_since_improvement >= patience:
                _report(f"Không cải thiện sau {patience} epoch — dừng sớm ở epoch {epoch + 1} "
                         f"(mục tiêu là {max_epochs}).")
                break

    if best_state is None:
        raise RuntimeError("Adapter bị từ chối: không có epoch nào hoàn tất huấn luyện. Không lưu adapter.")

    # Accept/reject decision: test_samples had zero influence over epoch
    # selection above, so this is the first time the run is judged on data
    # it couldn't have been optimized against. See module docstring.
    _report(f"Epoch tốt nhất theo early-stop là epoch {best_epoch} (WER early-stop {best_wer:.1f}%) "
            f"— khôi phục trọng số và đo lại trên {test_kind}…")
    set_peft_model_state_dict(model, best_state)
    final_wer = _evaluate_wer(model, processor, device, test_samples)
    _report(f"WER trên {test_kind}: {final_wer:.1f}% (cơ sở {base_wer:.1f}%)")

    if final_wer >= base_wer:
        raise RuntimeError(
            f"Adapter bị từ chối: WER trên {test_kind} ({final_wer:.1f}%) không thắng "
            f"được model gốc ({base_wer:.1f}%) trên cùng tập đó. Không lưu adapter để tránh "
            f"làm nhận dạng tệ đi -- thử thêm mẫu huấn luyện đa dạng hơn, hoặc mẫu ngắn hơn."
        )

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    _report(f"Đã lưu adapter (WER trên {test_kind}: {final_wer:.1f}%, cải thiện "
            f"{base_wer - final_wer:.1f}pp so với model gốc).")
    return output_dir
