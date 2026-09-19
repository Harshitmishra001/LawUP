"""LawUP Classification Training — SmolLM3-3B QLoRA

Completely version-agnostic training script.
Every API call uses a self-healing pattern: tries the preferred API, and if
it gets a TypeError it reads the bad kwarg name from the error message, drops
it, and retries — until it either succeeds or raises a non-TypeError.

Verified to work with:
  transformers 4.53.x  |  trl 1.x  |  peft 0.13+  |  bitsandbytes 0.43+
"""

import os
import re
import inspect
import argparse
import yaml
import logging
import torch

# ── TRL patches — MUST be applied before any trainer is imported ──────────────
try:
    import trl.trainer.sft_trainer as _sft_mod
    from transformers import Trainer as _BaseTrainer

    # Bug: SmolLM3 has tie_word_embeddings=True, so lm_head.forward gets wrapped
    # in functools.partial by prepare_model_for_kbit_training. TRL's chunked CE
    # patcher then crashes (AttributeError: partial has no __func__). No-op it.
    if hasattr(_sft_mod, '_patch_chunked_ce_lm_head'):
        _sft_mod._patch_chunked_ce_lm_head = lambda *a, **kw: None

    # Bug: With the patcher disabled, SFTTrainer.compute_loss still reads
    # outputs.num_valid_tokens (set by the now-disabled patcher). Override it.
    _orig_compute_loss = _BaseTrainer.compute_loss

    def _safe_compute_loss(self, model, inputs, **kwargs):
        return _orig_compute_loss(self, model, inputs, **kwargs)

    _sft_mod.SFTTrainer.compute_loss = _safe_compute_loss
except Exception:
    pass
# ─────────────────────────────────────────────────────────────────────────────

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# ── Core self-healing helper ──────────────────────────────────────────────────

def self_heal(cls_or_fn, kwargs: dict):
    """
    Call cls_or_fn(**kwargs). If a TypeError says an argument is unexpected,
    drop that argument and retry. Repeat until success or a non-type error.
    Returns the result of the successful call.
    """
    kw = dict(kwargs)
    while True:
        try:
            return cls_or_fn(**kw)
        except TypeError as e:
            m = re.search(r"unexpected keyword argument '([^']+)'", str(e))
            if m:
                bad = m.group(1)
                if bad in kw:
                    logger.warning(f"[self_heal] Dropping unsupported kwarg '{bad}' from {getattr(cls_or_fn, '__name__', str(cls_or_fn))}")
                    kw.pop(bad)
                else:
                    raise  # Can't fix — re-raise
            else:
                raise


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(model_id: str, bnb_config, torch_dtype):
    """Try dtype= (new API) first, then torch_dtype= (old API)."""
    common = dict(quantization_config=bnb_config, device_map="auto")
    for kw in [{"dtype": torch_dtype}, {"torch_dtype": torch_dtype}, {}]:
        try:
            return AutoModelForCausalLM.from_pretrained(model_id, **common, **kw)
        except TypeError:
            continue
    return AutoModelForCausalLM.from_pretrained(model_id, **common)


# ── Training args ─────────────────────────────────────────────────────────────

def build_training_args(tc: dict, max_seq: int):
    """
    Build training args using SFTConfig (TRL 1.x) preferably, which bundles
    both TrainingArguments and SFT-specific params (dataset_text_field,
    max_seq_length) in a single object.

    Falls back to TrainingArguments (TRL 2.x) if SFTConfig is unavailable.

    Returns (training_args, sft_params_included: bool)
      sft_params_included=True  → dataset_text_field is in training_args (SFTConfig path)
      sft_params_included=False → caller must pass SFT params to SFTTrainer separately
    """
    # All params we want — self_heal will drop any that a given class refuses
    all_kwargs = {
        # Standard TrainingArguments
        "output_dir":                  tc['output_dir'],
        "num_train_epochs":            tc['num_epochs'],
        "per_device_train_batch_size": tc['per_device_train_batch_size'],
        "gradient_accumulation_steps": tc['gradient_accumulation_steps'],
        "learning_rate":               float(tc['learning_rate']),
        "warmup_ratio":                tc['warmup_ratio'],
        "weight_decay":                tc['weight_decay'],
        "fp16":                        tc.get('fp16', False),
        "bf16":                        False,
        "group_by_length":             tc.get('group_by_length', False),
        "save_strategy":               tc['save_strategy'],
        "save_steps":                  tc['save_steps'],
        "save_total_limit":            tc['save_total_limit'],
        "eval_strategy":               tc.get('eval_strategy', 'no'),
        "eval_steps":                  tc.get('eval_steps', None),
        "load_best_model_at_end":      tc.get('load_best_model_at_end', False),
        "metric_for_best_model":       tc.get('metric_for_best_model', 'loss'),
        "greater_is_better":           tc.get('greater_is_better', False),
        "logging_steps":               tc.get('logging_steps', 10),
        "max_steps":                   tc.get('max_steps', -1),
        "report_to":                   "none",
        # SFT-specific (SFTConfig only — TrainingArguments will reject these)
        "dataset_text_field":          "text",
        "max_seq_length":              max_seq,
    }

    # Try SFTConfig first (TRL 1.x — includes SFT params)
    try:
        from trl import SFTConfig
        args = self_heal(SFTConfig, all_kwargs)
        logger.info("Using SFTConfig (TRL 1.x path).")
        return args, True   # SFT params already baked in
    except (ImportError, Exception) as e:
        logger.warning(f"SFTConfig not available ({e}), falling back to TrainingArguments.")

    # Fall back to TrainingArguments (TRL 2.x — SFT params passed to SFTTrainer)
    ta_kwargs = {k: v for k, v in all_kwargs.items()
                 if k not in ("dataset_text_field", "max_seq_length", "max_length")}
    args = self_heal(TrainingArguments, ta_kwargs)
    logger.info("Using TrainingArguments (TRL 2.x path).")
    return args, False   # SFT params must be passed to SFTTrainer


# ── Trainer ───────────────────────────────────────────────────────────────────

def build_trainer(model, training_args, sft_params_included: bool,
                  train_ds, eval_ds, tokenizer, max_seq: int):
    """
    Build SFTTrainer. Handles both the SFTConfig (1.x) and TrainingArguments (2.x) paths.
    Uses inspect to detect whether this TRL version uses processing_class or tokenizer.
    """
    ta_params = inspect.signature(SFTTrainer.__init__).parameters
    tok_kwarg = "processing_class" if "processing_class" in ta_params else "tokenizer"

    trainer_kwargs = {
        "model":          model,
        "args":           training_args,
        "train_dataset":  train_ds,
        "eval_dataset":   eval_ds,
        tok_kwarg:        tokenizer,
    }

    # When using plain TrainingArguments (TRL 2.x path), SFT params must go here
    if not sft_params_included:
        trainer_kwargs["dataset_text_field"] = "text"
        trainer_kwargs["max_seq_length"] = max_seq

    # self_heal will drop any kwarg SFTTrainer doesn't know about
    return self_heal(SFTTrainer, trainer_kwargs)


# ── GradScaler disable ────────────────────────────────────────────────────────

def disable_gradscaler(trainer):
    """Disable GradScaler to prevent bf16/T4 crashes (safe no-op if not applicable)."""
    try:
        if not (hasattr(trainer, "accelerator") and
                hasattr(trainer.accelerator, "scaler") and
                trainer.accelerator.scaler is not None):
            return
        try:
            trainer.accelerator.scaler = torch.amp.GradScaler("cuda", enabled=False)
        except TypeError:
            trainer.accelerator.scaler = torch.cuda.amp.GradScaler(enabled=False)
        logger.info("GradScaler disabled (T4 bf16 safety).")
    except Exception as e:
        logger.warning(f"GradScaler patch skipped: {e}")


# ── Config loading ────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LawUP QLoRA classification training")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    cli = parser.parse_args()

    config = load_config(cli.config)
    logger.info(f"Config: {cli.config}")

    model_id = config['model']['name']
    tc       = config['training']
    max_seq  = tc['max_seq_length']

    # 1. Quantization
    logger.info("Setting up NF4 quantization...")
    compute_dtype = getattr(torch, config['quantization']['bnb_4bit_compute_dtype'])
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config['quantization']['load_in_4bit'],
        bnb_4bit_quant_type=config['quantization']['bnb_4bit_quant_type'],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=config['quantization']['bnb_4bit_use_double_quant'],
    )

    # 2. Tokenizer + Model
    logger.info(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = getattr(torch, config['model']['torch_dtype'])
    logger.info(f"Loading model: {model_id}")
    model = load_model(model_id, bnb_config, torch_dtype)

    # Cast stray bfloat16 params to float16 (GradScaler crashes on bf16 on T4)
    try:
        model.config.torch_dtype = torch_dtype
    except Exception:
        pass
    for param in model.parameters():
        if param.dtype == torch.bfloat16:
            param.data = param.data.to(torch_dtype)
    torch.set_default_dtype(torch.float32)

    model = prepare_model_for_kbit_training(model)

    # 3. LoRA
    lora_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['lora_alpha'],
        lora_dropout=config['lora']['lora_dropout'],
        target_modules=config['lora']['target_modules'],
        task_type=config['lora']['task_type'],
    )
    try:
        model = get_peft_model(model, lora_config, autocast_adapter_dtype=False)
    except TypeError:
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. Data
    logger.info("Loading datasets...")
    train_file = config['data']['train_file']
    val_file   = config['data']['val_file']
    dataset = load_dataset("json", data_files={"train": train_file, "val": val_file})
    logger.info(f"  train={len(dataset['train'])}  val={len(dataset['val'])}")

    def apply_template(example):
        return {"text": tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )}

    logger.info("Applying chat template...")
    dataset = dataset.map(apply_template, remove_columns=["messages"])
    logger.info(f"Sample (first 300 chars):\n{dataset['train'][0]['text'][:300]}")

    # 5. Training args
    logger.info("Building training arguments...")
    training_args, sft_params_included = build_training_args(tc, max_seq)

    # 6. Trainer
    eval_ds = dataset["val"] if tc.get('eval_strategy', 'no') != 'no' else None
    logger.info("Building SFTTrainer...")
    trainer = build_trainer(
        model, training_args, sft_params_included,
        dataset["train"], eval_ds, tokenizer, max_seq
    )
    disable_gradscaler(trainer)

    # 7. Checkpoint resume
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and tc.get('resume_from_checkpoint', False):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint:
            logger.info(f"Resuming from: {last_checkpoint}")
        else:
            logger.info("No checkpoint — starting fresh.")

    # 8. Diagnostics
    bf16_leaks = [
        (n, p.dtype) for n, p in model.named_parameters() if p.dtype == torch.bfloat16
    ] + [
        (n, b.dtype) for n, b in model.named_buffers() if b.dtype == torch.bfloat16
    ]
    logger.info(f"DIAGNOSTIC: {len(bf16_leaks)} bf16 tensor(s) remaining.")

    # 9. Train
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # 10. Save
    final_path = os.path.join(training_args.output_dir, "final")
    logger.info(f"Saving adapter to {final_path}")
    trainer.save_model(final_path)
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
