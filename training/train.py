"""LawUP training entry point.

Usage (from repo root, typically on Kaggle/Colab):
    python -m training.train --config training/configs/smollm3_qlora.yaml

Compatibility:
    - transformers >= 4.53.0  (required for SmolLM3 architecture)
    - trl >= 1.0.0            (SFTConfig uses max_length, not max_seq_length)
    - peft >= 0.7.0
    - bitsandbytes >= 0.41.0

TRL 1.8.0 Bug Workarounds (applied at import time):
    Bug 1 – _patch_chunked_ce_lm_head crash:
        TRL's chunked cross-entropy optimizer inspects lm_head.forward expecting
        a normal bound method. But SmolLM3 has tie_word_embeddings=True, so the
        input embeddings and lm_head share the same module. When
        prepare_model_for_kbit_training() calls enable_input_require_grads(), it
        wraps that shared module's forward in functools.partial, which TRL then
        chokes on (AttributeError: 'functools.partial' has no '__func__').
        Fix: Disable _patch_chunked_ce_lm_head entirely (no-op).

    Bug 2 – num_valid_tokens AttributeError in compute_loss:
        With the chunked CE patch disabled, SFTTrainer.compute_loss still tries
        to read outputs.num_valid_tokens (set by the now-disabled patch).
        Fix: Override compute_loss to use the base Trainer implementation,
        which computes standard cross-entropy via the model's built-in loss.
"""

import os
import argparse
import yaml
import logging
import torch

# ──────────────────────────────────────────────────────────────────────
# TRL 1.8+ defensive patches — applied BEFORE any trainer is created.
# Safe on all TRL versions: guarded by hasattr / try-except.
# ──────────────────────────────────────────────────────────────────────
try:
    import trl.trainer.sft_trainer as _sft_mod
    from transformers import Trainer as _BaseTrainer

    # Bug 1: disable the chunked CE lm_head patcher
    if hasattr(_sft_mod, '_patch_chunked_ce_lm_head'):
        _sft_mod._patch_chunked_ce_lm_head = lambda *args, **kwargs: None

    # Bug 2: bypass SFTTrainer.compute_loss → use base Trainer's version
    #   (our data is pre-formatted text; we don't use packing or
    #    completion-only masking, so the base implementation is equivalent)
    _original_base_compute_loss = _BaseTrainer.compute_loss

    def _safe_compute_loss(self, model, inputs, **kwargs):
        return _original_base_compute_loss(self, model, inputs, **kwargs)

    _sft_mod.SFTTrainer.compute_loss = _safe_compute_loss
except Exception:
    pass  # If patching fails, let the original code run and surface its own errors
# ──────────────────────────────────────────────────────────────────────

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="LawUP model training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")

    model_id = config['model']['name']

    # --- 1. Quantization Config ---
    logger.info("Setting up NF4 Quantization...")
    compute_dtype = getattr(torch, config['quantization']['bnb_4bit_compute_dtype'])
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config['quantization']['load_in_4bit'],
        bnb_4bit_quant_type=config['quantization']['bnb_4bit_quant_type'],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=config['quantization']['bnb_4bit_use_double_quant'],
    )

    # --- 2. Load Tokenizer & Model ---
    logger.info(f"Loading tokenizer and model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = getattr(torch, config['model']['torch_dtype'])
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch_dtype,
        # trust_remote_code is NOT needed for SmolLM3 on transformers >= 4.53.0
    )
    
    # Fix BFloat16 crashes on T4 GPUs:
    # SmolLM3 uses bfloat16 natively. Even with torch_dtype=torch.float16,
    # the config or some parameters (like lm_head) might retain bfloat16.
    # This causes GradScaler to crash with NotImplementedError during fp16 training.
    model.config.torch_dtype = torch_dtype
    for param in model.parameters():
        if param.dtype == torch.bfloat16:
            param.data = param.data.to(torch_dtype)

    # Reset default dtype: from_pretrained can sometimes leak the model's native dtype
    # (bfloat16) into the global PyTorch context. This causes all subsequently created
    # layers (like PEFT LoRA matrices) to initialize in bfloat16. We reset to float32.
    torch.set_default_dtype(torch.float32)

    # Prepare for k-bit training: casts layernorm to fp32, enables gradient
    # checkpointing (critical for fitting 3B model on T4 16GB), and enables
    # input require grads for backprop through quantized layers.
    model = prepare_model_for_kbit_training(model)

    # Apply LoRA adapter
    lora_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['lora_alpha'],
        lora_dropout=config['lora']['lora_dropout'],
        target_modules=config['lora']['target_modules'],
        task_type=config['lora']['task_type']
    )
    # autocast_adapter_dtype=False prevents PEFT from auto-casting adapters to match the 
    # base model's native dtype (bfloat16 for SmolLM3). We want them in float32!
    # (If the PEFT version is too old for this kwarg, the fallback loop catches it).
    try:
        model = get_peft_model(model, lora_config, autocast_adapter_dtype=False)
    except TypeError:
        model = get_peft_model(model, lora_config)
            
    model.print_trainable_parameters()

    # --- 3. Load Data ---
    logger.info("Loading formatted datasets...")
    data_dir = os.path.dirname(config['data']['train_file'])
    train_file = os.path.join(data_dir, "train_qlora.jsonl")
    val_file = os.path.join(data_dir, "val_qlora.jsonl")

    dataset = load_dataset("json", data_files={"train": train_file, "val": val_file})
    logger.info(f"Loaded {len(dataset['train'])} training examples and {len(dataset['val'])} validation examples.")

    # --- 4. Setup Training Arguments ---
    tc = config['training']

    sft_kwargs = dict(
        output_dir=tc['output_dir'],
        num_train_epochs=tc['num_epochs'],
        per_device_train_batch_size=tc['per_device_train_batch_size'],
        gradient_accumulation_steps=tc['gradient_accumulation_steps'],
        learning_rate=float(tc['learning_rate']),
        warmup_ratio=tc['warmup_ratio'],
        weight_decay=tc['weight_decay'],
        fp16=tc.get('fp16', False),
        bf16=False,  # Explicitly disable bf16 to prevent T4 crashes
        group_by_length=tc.get('group_by_length', False),
        save_strategy=tc['save_strategy'],
        save_steps=tc['save_steps'],
        save_total_limit=tc['save_total_limit'],
        eval_strategy=tc.get('eval_strategy', 'no'),
        eval_steps=tc.get('eval_steps', None),
        load_best_model_at_end=tc.get('load_best_model_at_end', False),
        metric_for_best_model=tc.get('metric_for_best_model', 'loss'),
        greater_is_better=tc.get('greater_is_better', False),
        logging_steps=tc.get('logging_steps', 10),
        max_steps=tc.get('max_steps', -1),
        dataset_text_field="text",
    )

    # TRL >= 1.0 renamed max_seq_length -> max_length.
    # Try the new name first; fall back to old name for older TRL.
    max_seq = tc['max_seq_length']
    try:
        training_args = SFTConfig(**sft_kwargs, max_length=max_seq)
    except TypeError:
        training_args = SFTConfig(**sft_kwargs, max_seq_length=max_seq)

    # --- 5. Build Trainer ---
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"] if tc.get('eval_strategy', 'no') != 'no' else None,
        peft_config=None,  # Handled above
        processing_class=tokenizer,
    )
    
    # KAGGLE T4 GRADSCALER FIX:
    # The GradScaler has a fused CUDA kernel that crashes when it touches bfloat16 tensors 
    # on hardware that lacks native bfloat16 support (like the T4 GPU).
    # Since PEFT stubbornly locks LoRA adapters in bfloat16, we disable the GradScaler 
    # completely. Mixed precision (fp16) will still run fast via autocast, but without 
    # the scaling steps.
    if hasattr(trainer, "accelerator") and hasattr(trainer.accelerator, "scaler"):
        if trainer.accelerator.scaler is not None:
            trainer.accelerator.scaler = torch.cuda.amp.GradScaler(enabled=False)
            logger.info("Successfully disabled the GradScaler via Accelerator monkeypatch!")

    # --- 6. Checkpoint Resume Logic ---
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and tc.get('resume_from_checkpoint', False):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None:
            logger.info(f"Checkpoint detected. Resuming training from {last_checkpoint}")
        else:
            logger.info("No valid checkpoint found in output_dir. Starting training from scratch.")

    # --- 7. Train ---
    logger.info("Starting training run...")
    
    # DIAGNOSTIC: Check for bfloat16 leaks before training starts
    bf16_leaks = [(n, p.dtype) for n, p in model.named_parameters() if p.dtype == torch.bfloat16]
    bf16_leaks += [(n, b.dtype) for n, b in model.named_buffers() if b.dtype == torch.bfloat16]
    logger.info(f"DIAGNOSTIC: Found {len(bf16_leaks)} bf16 tensors.")
    if bf16_leaks:
        for name, dtype in bf16_leaks[:20]:
            logger.info(f"  {name}: {dtype}")
            
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # --- 8. Save Final ---
    logger.info(f"Saving final adapter to {training_args.output_dir}/final")
    trainer.save_model(os.path.join(training_args.output_dir, "final"))


if __name__ == "__main__":
    main()
