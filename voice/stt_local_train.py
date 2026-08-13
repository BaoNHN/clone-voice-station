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

base_model must be one of database.ALLOWED_STT_BASE_MODELS (phowhisper-small/
whisper-tiny/whisper-base) -- this machine's GPU has ~4GB VRAM and this is a
public, no-API-key-gated page, so batch_size stays at 1 and there's no
attempt at anything heavier. phowhisper-small (~244M params) is noticeably
bigger than tiny/base and the tightest fit of the three for local training
specifically; a guest whose local run OOMs on this hardware can switch to
the Colab backend instead (see engine/stt_train_engine.py's backend param) --
Colab's GPU has much more headroom.

Everything heavy (torch, transformers, peft, librosa) is imported lazily
inside train(), same pattern as voice/stt.py's Whisper load.

Validation gate (added after a real regression was measured -- see
tools/import_hf_stt_dataset.py + voice-lab-example/tools/test_medical_lora_wer.py):
naively training on whatever a guest uploads and always shipping the last
epoch's weights let a customer's own dataset make transcription measurably
*worse* than plain Whisper (confirmed: a 458-sample medical-consultation import
still landed at 62.8% corpus WER against pretrained whisper-tiny's own 32.0%
on the same held-out audio, despite fixing the learning rate and decoding
separately). A guest's few dozen (or few hundred) clips are nothing next to
the hundreds of thousands of hours Whisper itself was pretrained on, so
"train on it and ship whatever comes out" was never a safe default. This
module holds out part of every upload, scores the untouched pretrained model
on it first as the bar to beat, tracks a candidate epoch's adapter weights
against that bar, and raises instead of saving if it doesn't actually clear
it -- an adapter that would make things worse never reaches STT_MODELS_DIR at
all. Callers (engine/stt_train_engine.py) already turn any exception here
into a "failed" adapter status with the message attached, so this needs no
separate status/plumbing of its own.

Three gates, not one -- naming from here on: Gate 0 picks which epoch to keep (VAL_*,
early-stopping, never an accept/reject decision by itself); Gate 1 is the guest's-own-
domain check (TEST_*, below); Gate 2 is the fixed general-benchmark check (further below).
An adapter must clear both Gate 1 and Gate 2 to ship -- either one alone has already been
proven to let a bad adapter through (see each gate's own history below).

Gate 1: three-way split, not two (added after Gate 1 was itself caught being fooled --
see TEST_*/VAL_* below). An earlier version used a single held-out split both to pick the
best epoch *and* to decide accept/reject against the base model. That's double-dipping --
the same data that already picked the "winning" epoch then gets asked to confirm that
epoch is good, which is structurally biased toward yes. Confirmed for real, twice, on
genuinely independent data (voice-lab-example/tools/test_medical_lora_wer.py against a
30-file held-out eval set the training loop never saw at all): an adapter this gate
accepted as beating the base model by 9-15pp on its held-out split still lost by 20-26pp
on that independent set, for both a 60-sample run and a 250-sample run at a higher LoRA
rank -- i.e. the gate's own "no" would have been the right call both times, but its "yes"
was wrong both times. Splitting the held-out portion into a separate early-stopping set
(VAL_* = Gate 0, only ever used to pick which epoch to keep) and final-gate set (TEST_* =
Gate 1 proper, only ever scored once, after epoch selection is already final) is the
standard fix: whatever number the accept/reject decision is made on now had zero influence
over which epoch got there.

Gate 2: fixed general-benchmark check, on top of Gate 1 above (added after Gate 1's fix
was also confirmed insufficient on its own -- a three-way-split adapter still lost by
26.7pp on the same independent eval set described above, essentially unchanged from the
double-dipping version's 24.4pp loss). The three-way split fixes a real statistical bug
(whichever split makes the accept/reject call must have had no say in picking the epoch),
but it doesn't fix the deeper issue: TEST_* is still carved from the *same upload batch*
as TRAIN_*, so train and test share whatever recording session/speaker/domain quirks that
one guest's one upload happens to have -- no amount of correct splitting turns same-source
data into a real test of generalization. See tools/prepare_general_benchmark.py for the
full reasoning. That script provisions stt_general_benchmark/ once with a fixed slice of
doof-ferb/vlsp2020_vinai_100h -- general spontaneous Vietnamese speech with nothing to do
with any guest's domain, downloaded once and never touched by any guest's training data,
ever. Every adapter, regardless of domain, must also not regress against the untrained
base model's WER on that fixed set (see general_wer/general_base_wer below) -- this is a
*different* question from Gate 1's TEST_*-based check ("did this adapter get better at the
guest's own domain, on the guest's own held-out data", still unreliable per above) versus
("did this adapter get worse at Vietnamese speech in general", which a source completely
disjoint from the guest's upload can actually answer). Confirmed catching a real case, not
just theoretical: adapter #21 (60 medical samples) cleared Gate 1 (52.0% vs base 60.2% on
its own domain test split) but was correctly rejected by Gate 2 (47.6% vs base 44.4% on
the general benchmark -- worse at ordinary Vietnamese despite "improving" on its own
domain). Gate 2 is not a full fix either: a guest-domain-specific held-out check would
still need the guest to supply a second, separately-collected batch, which is a product/UX
change out of scope here. What Gate 2 adds is a real, if partial, backstop -- catching at
least the case where an adapter overfits badly enough to also get worse at ordinary
Vietnamese, which Gate 0/1 have no way to see at all.

UPDATE (2026-08-12): Gate 2 removed, replaced by an optional `holdout_samples` param to
train() -- a genuinely independent test set (e.g. a HuggingFace dataset's own official
"test" split, disjoint recording sessions from whatever "train" split the guest's upload
came from) supplied by the caller instead of carved out of the guest's own upload. When
provided, it replaces Gate 1's TEST_* entirely (no more double-dipping risk to design
around -- the accept/reject decision is now made on data that was never in the same
upload/session as anything the model trained on, not an internal proxy split of it) and
Gate 2 (VLSP2020 general-benchmark) is skipped. Rationale: Gate 1's own history above is
double-edged -- the three-way split *by itself* was independently confirmed wrong in both
directions on this same class of held-out-from-one-upload data (see TEST_FRACTION's
comment below: an adapter Gate 1 rated as beating base by 9-15pp on its own split lost by
20-26pp on a genuinely independent set, twice). A true external test set doesn't have that
same-session/speaker proxy-error problem in the first place, which is a more direct fix
than adding another fixed benchmark on top. This does trade away Gate 2's domain-agnostic
"didn't get worse at Vietnamese in general" backstop -- appropriate when the actual
deployment question is domain adaptation ("did this improve unseen target-domain speech
vs. the base model"), not simultaneous general-capability preservation. When no
holdout_samples are supplied (the common case -- a regular guest via the STT Lab page has
no curated external test set), falls back to the original three-way-split Gate 1 alone
(Gate 2 stays removed either way -- it was never a per-guest concept, and removing it
without a same-quality replacement for the common case was judged better than keeping a
check documented above as not fully reliable for what it needs to catch).
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

# 1e-3 was tried first and rejected: on real long-form audio (20-30s clips, ~500-char
# transcripts -- see tools/import_hf_stt_dataset.py's medical-consultation import) it
# reliably drove the adapter into degenerate repeated-token output at inference, WER far
# worse than the untrained base model. 1e-4 is the standard LoRA-fine-tune-of-Whisper
# learning rate (used by PEFT's own Whisper LoRA examples, ~10x lower than full
# fine-tuning would use) and is far more stable for this small a dataset (this page's
# guests upload as few as MIN_STT_TRAIN_SAMPLES=10 clips).
LEARNING_RATE = 1e-4
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
# Broadened from just q_proj/v_proj: the validation gate (below) exists specifically so a
# target-module set with more capacity -- and therefore more overfitting risk, per the
# same experiment that motivated the gate -- gets caught and rejected automatically
# instead of silently shipping a worse adapter. Still stops short of touching fc1/fc2
# (the feed-forward layers), which would roughly double trainable params again for a
# page whose guests upload at most a few hundred clips.
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj"]

# Training policy scales with how much data was actually uploaded -- a guest with 15
# clips and one with 450 are not the same problem. Few samples next to Whisper's
# pretraining corpus (hundreds of thousands of hours) can only overfit no matter how
# careful the loop is, so they get a low epoch ceiling and tight patience (fail fast,
# don't burn GPU time chasing an adapter the validation gate was always going to
# reject); more data can support more capacity (a higher LoRA rank) and more epochs
# before overfitting risk catches up, so patience widens too. Bucket boundaries and
# per-bucket epoch/patience/rank are a starting policy, not tuned against a held-out
# sweep -- the validation gate is what actually keeps a bad outcome from shipping in
# every bucket, this just shapes how much compute is spent looking for a good one.
_POLICY_BUCKETS = [
    # (min_samples, max_epochs, patience, lora_r)
    (10,  5,  2, 8),
    (50,  10, 3, 8),
    (200, 20, 5, 16),  # top bucket -- covers 200 up through MAX_STT_TRAIN_SAMPLES (500);
                        # a separate 500+ tier was removed 2026-08-12 since no upload can
                        # ever actually exceed that hard per-adapter cap (database.py).
]


def _training_policy(n_samples: int) -> dict:
    max_epochs, patience, lora_r = _POLICY_BUCKETS[0][1:]
    for min_samples, e, p, r in _POLICY_BUCKETS:
        if n_samples >= min_samples:
            max_epochs, patience, lora_r = e, p, r
    return {"max_epochs": max_epochs, "patience": patience, "lora_r": lora_r}

# Held out from training in two disjoint pieces, for Gate 0 (VAL_*) and Gate 1 (TEST_*) --
# see module docstring's "Gate 1" note for why two sets, not one. A flat fraction rather
# than a fixed count so both scale with upload size; floors/ceilings keep each sane at both
# ends of database.MIN_STT_TRAIN_SAMPLES..MAX_STT_TRAIN_SAMPLES.
#
# TEST_* carves out Gate 1's set FIRST (before VAL_* sees the remainder) -- this is
# the split the accept/reject decision is actually made on, so it gets first claim on the
# larger, more reliable size. These exact fraction/floor/ceiling numbers are inherited
# unchanged from what used to be this module's only held-out split (back when it did
# double duty for both epoch selection and the gate): 0.15/40 was tried first for that
# combined role and rejected -- for a 60-sample upload that's only ~9 clips, too few and
# too noisy to reliably predict generalization, confirmed for real (an adapter that split
# rated as beating base 50.7% vs 61.4% still lost 57.2% vs 31.7% on a genuinely independent
# 30-file eval set, see voice-lab-example/tools/test_medical_lora_wer.py's
# medical_wer_tier2.csv run). 0.25/60 replaced it. Keeping the same numbers here for TEST_*
# specifically (not VAL_*) is deliberate: it's the one scale this module has real evidence
# behaved reasonably, and it's now assigned to the role (the actual gate) where that
# evidence matters most.
TEST_FRACTION  = 0.25
TEST_MIN_COUNT = 3
TEST_MAX_COUNT = 60
# VAL_* then carves a separate, smaller early-stopping set from whatever TEST_* left
# behind. This one only ever needs to answer "is this epoch better than the best epoch
# seen so far" -- a relative, within-run comparison -- not produce a statistically tight
# absolute WER estimate the way TEST_* does, so it can afford to be smaller, leaving more
# of each upload for actual training.
VAL_FRACTION   = 0.15
VAL_MIN_COUNT  = 3
VAL_MAX_COUNT  = 30
# Decode-quality guard for the bare HF transformers generate() call -- confirmed for
# real (see tools/import_hf_stt_dataset.py's medical-consultation experiment writeup)
# that this matters as much as the LoRA training itself: the *same untrained pretrained*
# whisper-tiny weights scored 80.6% WER via plain greedy generate() vs only 31.5% via
# the openai-whisper package (voice/stt.py's Tier 1 path) on identical audio -- PEFT can
# only attach to the transformers model class, so Tier 2 inherits transformers' much
# weaker out-of-the-box decoding regardless of how well an adapter is trained. Adding
# beam search closed a third of that gap in the same experiment (80.6% -> 61.7%);
# no_repeat_ngram_size/repetition_penalty guard against the degenerate repeated-token
# loops an under-trained adapter can otherwise produce. Still meaningfully behind
# openai-whisper's fuller decoding pipeline (VAD, temperature-fallback retries,
# condition-on-previous-text) -- closing the rest would mean reimplementing that logic
# on top of transformers, out of scope here.
GENERATE_KWARGS = {"no_repeat_ngram_size": 3, "repetition_penalty": 1.3, "num_beams": 5}

# Gate 2 (fixed VLSP2020 general-Vietnamese benchmark) removed 2026-08-12 -- see module
# docstring's "UPDATE" note. tools/prepare_general_benchmark.py and stt_general_benchmark/
# are left in place (still useful as a standalone informational check, e.g. for a thesis
# writeup), just no longer loaded or gated on here.


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
    samples          : list  [{"audio_path": str, "reference_text": str}, ...] -- the
                              trainable pool (early-stop split carved from this; also the
                              final-gate split too, when holdout_samples is None).
    base_model       : str   One of database.ALLOWED_STT_BASE_MODELS
                              ("phowhisper-small"/"whisper-tiny"/"whisper-base").
    output_dir       : str   Where to save the trained adapter (created if missing).
    resume_from_path : str   Directory of a previously-saved adapter (adapter_model.safetensors +
                              adapter_config.json) to continue training from, or None to start fresh.
    progress_cb       : callable(str)  Called with a human-readable status after every sample.
    holdout_samples   : list  Optional genuinely independent test set (e.g. a dataset's own
                              official "test" split) -- see module docstring's 2026-08-12
                              update. When given, this becomes the final-gate split directly
                              (no TEST_* carved from `samples`); when None, falls back to the
                              original three-way split of `samples` alone.

    Returns
    -------
    str  output_dir, once the selected adapter is saved there -- selected by early-stop
         WER on one held-out split, then validated by beating the untrained base model's
         WER on the final-gate split (see holdout_samples above).

    Raises
    ------
    RuntimeError  if the epoch selected via early-stopping doesn't beat the base model's
                  WER on the final-gate split -- nothing is saved in that case (see module
                  docstring).
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

    # Baseline to beat: the untouched pretrained model, scored on the final-gate test split
    # (never the early-stop val split -- see module docstring), *before* any LoRA layer is
    # attached -- get_peft_model() below monkeypatches base's own target-module submodules
    # in place, so this has to run first to see the model's un-adapted output.
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

        # Compared only against the best early-stop WER seen so far in this run -- not
        # against base_wer, which was scored on test_samples, a disjoint split this loop
        # never touches (see module docstring's "three-way split" note). Mixing the two
        # here would be comparing WER on two different sample sets.
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

    # The actual accept/reject decision happens here, now that epoch selection is already
    # final -- test_samples had zero influence over which epoch won, unlike val_samples
    # above, so this is the first and only time this run's outcome is judged on data that
    # couldn't have been (even accidentally) optimized against. See module docstring.
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
