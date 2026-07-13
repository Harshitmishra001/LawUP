# Colab Launcher Workflow

The actual training code lives in `training/*.py` — **not** in notebook cells.

The Colab notebook is a **thin launcher** that:
1. Clones the repo
2. Installs dependencies
3. Runs the versioned training script

This keeps all training logic version-controlled and reproducible.

## Instructions

1. Create a new Google Colab notebook
2. Connect to a **T4 GPU** runtime (free tier)
3. Run the following cells:

### Cell 1 — Setup

```python
# Clone the repo and install dependencies
!git clone https://github.com/<your-username>/LawUP.git
%cd LawUP
!pip install -r backend/requirements.txt
!pip install pyyaml
```

### Cell 2 — Authenticate (if using private models/data)

```python
from huggingface_hub import login
from google.colab import userdata

login(token=userdata.get("HF_TOKEN"))
```

### Cell 3 — Train

```python
!python -m training.train --config training/configs/smollm3_qlora.yaml
```

### Cell 4 — Evaluate (after training is implemented)

```python
# TODO: Add evaluation cell once M2 is complete
```

## Why not put code in the notebook?

- Notebook cells are hard to diff and review
- Version control works better with `.py` files
- The same training script runs locally, on Colab, or on any cloud GPU
- Configs are in YAML, not scattered across cells
