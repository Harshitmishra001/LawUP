import os
import torch
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

adapter_dir = 'classifier_adapter'
base_model_id = 'HuggingFaceTB/SmolLM3-3B'
merged_dir = 'merged_classifier'

print('Loading base model...')
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map='cpu',
    torch_dtype=torch.float16
)

print('Loading adapter...')
model = PeftModel.from_pretrained(base_model, adapter_dir)

print('Merging weights... (this takes ~1-2 minutes)')
merged_model = model.merge_and_unload()

print(f'Saving merged model to {merged_dir}...')
merged_model.save_pretrained(merged_dir)

print('Saving tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.save_pretrained(merged_dir)

print('Merge complete! Model saved to disk.')
