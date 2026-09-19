# LawUP: GGUF Conversion Script (Google Colab)

Run this directly in a new Google Colab notebook (a free CPU or T4 instance is fine) to merge your trained adapter and quantize it into a GGUF file for LMStudio.

### Step 0: Upload Adapter to Google Drive
Since M2 ran on Kaggle, your adapter is currently on your local laptop inside the `model_checkpoints_backup.zip` file.
1. Unzip `model_checkpoints_backup.zip` on your computer.
2. Open your Google Drive.
3. Create a folder named `LawUP`.
4. Upload the entire `checkpoint-300` folder into that `LawUP` folder.
*(You should end up with: `My Drive/LawUP/checkpoint-300/adapter_model.safetensors`)*

### Step 1: Mount Drive & Install Dependencies
Create a new cell and run:
```python
from google.colab import drive
drive.mount('/content/drive')

# Colab already has torch and transformers. We only need to install peft.
!pip install -q peft
!git clone https://github.com/ggerganov/llama.cpp
!cd llama.cpp && make -j
```

### Step 2: Merge the LoRA Adapter (FP16)
Create a new cell and run:
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_id = "HuggingFaceTB/SmolLM3-3B"
# We strictly use checkpoint-300 to retain native bfloat16 adapter weights
adapter_id = "/content/drive/MyDrive/LawUP/checkpoints/checkpoint-300"
merged_dir = "/content/merged_hf"

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.float16, device_map="cpu")

print("Loading adapter...")
peft_model = PeftModel.from_pretrained(base_model, adapter_id)

print("Merging...")
merged_model = peft_model.merge_and_unload()

print("Verifying dtypes across all parameters...")
dtypes = set()
for name, param in merged_model.named_parameters():
    dtypes.add(param.dtype)

print(f"Found {len(dtypes)} distinct dtypes: {dtypes}")
if len(dtypes) > 1 or list(dtypes)[0] != torch.float16:
    print("CRITICAL ERROR: Mixed or incorrect dtypes detected in merged model!")
    for name, param in merged_model.named_parameters():
        if param.dtype != torch.float16:
            print(f" - {name} is {param.dtype}")
    import sys
    sys.exit(1)
print("Dtype verification passed. Model is entirely FP16.")

print("Saving merged model to temporary folder...")
merged_model.save_pretrained(merged_dir, safe_serialization=True)

tokenizer = AutoTokenizer.from_pretrained(base_id)
tokenizer.save_pretrained(merged_dir)
print("Done merging!")
```

### Step 3: Convert to GGUF and Quantize to Q8
Create a new cell and run:
```bash
# Install llama.cpp requirements here so they don't corrupt the PyTorch merge in Step 2
!pip install -q -r llama.cpp/requirements.txt

# Convert Hugging Face format to FP16 GGUF
!python llama.cpp/convert_hf_to_gguf.py /content/merged_hf --outfile /content/LawUP-3B-F16.gguf --outtype f16

# Quantize the FP16 GGUF down to an optimized 8-bit Q8_0 GGUF directly into your Google Drive
!./llama.cpp/llama-quantize /content/LawUP-3B-F16.gguf /content/drive/MyDrive/LawUP/LawUP-3B-Q8.gguf q8_0

print("Quantization complete! You can now download LawUP-3B-Q8.gguf from your Google Drive.")
```
