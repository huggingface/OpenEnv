# Email Triage Final Showcase Playbook

This is the end-to-end plan for your final demo:
1. train RL on Google Colab Free Tier (`T4`)
2. push model to Hugging Face Hub
3. deploy the Gradio demo UI on Hugging Face Spaces
4. present a clean "problem -> training -> results -> live demo" story

## 0) Fast Checklist

- [ ] tests pass locally
- [ ] smoke training works in Colab
- [ ] full training checkpoint uploaded to Hub
- [ ] Space is public and stable
- [ ] 2-3 minute demo script rehearsed

## 1) Local Validation Before Training

From repo root (`OpenEnv`):

```powershell
$env:PYTHONPATH='src;envs'
.venv\Scripts\python -m pytest tests/envs/test_email_triage_env.py tests/envs/test_email_triage_http.py -v --tb=short
```

If green, your environment + server are ready for training/demo.

## 2) Colab T4 RL Training (Reliable Path)

### 2.1 Colab setup

1. Runtime -> Change runtime type -> `T4 GPU`
2. Run:

```bash
!git clone https://github.com/<your-username>/OpenEnv.git
%cd OpenEnv
!pip install -U pip
!pip install "torch>=2.3" "transformers>=4.46" "trl>=0.11.0" "accelerate>=0.34" datasets huggingface_hub bitsandbytes fastmcp
```

If you already cloned before this fix, pull latest first:

```bash
!git -C OpenEnv pull
```

### 2.2 Verify pipeline first (mandatory)

```bash
!PYTHONPATH=src:envs python envs/email_triage_env/train_grpo.py --smoke
```

### 2.3 Main free-tier run

```bash
!PYTHONPATH=src:envs python envs/email_triage_env/train_grpo.py --model Qwen/Qwen2-0.5B --max-steps 50 --dataset-size 64 --output-dir oversight-arena-grpo-t4
```

Important for Colab:
- run the training command in **one cell** exactly as above
- do not add plain `print(...)` after a `!python ...` line in the same cell
- if you want a completion message, use another cell:

```python
print("\nTraining complete. Checkpoint is in oversight-arena-grpo-t4/")
```

### 2.4 Push trained checkpoint to Hub

```bash
!huggingface-cli login
!PYTHONPATH=src:envs python envs/email_triage_env/train_grpo.py --model Qwen/Qwen2-0.5B --max-steps 50 --dataset-size 64 --output-dir oversight-arena-grpo-t4 --push-to-hub --hub-repo YOUR_USERNAME/oversight-arena-grpo-t4
```

### 2.5 Common Colab errors and fixes

- `No module named trl`  
  Run the install cell again, then `Runtime -> Restart runtime`.

- `ModuleNotFoundError: No module named 'fastmcp'` or `No module named 'core'`  
  Pull latest repo code and rerun install cell. This is fixed in the latest `train_grpo.py`.

- `CUDA out of memory`  
  Reduce to `--max-steps 30 --dataset-size 32`.

- `bitsandbytes` / optimizer issues  
  The script now auto-falls back to `adamw_torch` if `bitsandbytes` is unavailable.

- tokenizer/processing class errors  
  The script now explicitly loads tokenizer in non-Unsloth mode.

## 3) Hugging Face Space Deployment (UI)

Your polished UI is in `envs/email_triage_env/server/ui.py`, with a cyber orange hero style inspired by your reference image ("Your Pocket AI Red-Team Agent").

### 3.1 Create Space

```bash
pip install -U huggingface_hub
hf auth login
hf repo create YOUR_USERNAME/oversight-inbox-arena --type space --space-sdk gradio
```

### 3.2 Files to copy into Space repo

- `server/ui.py`
- `server/email_triage_environment.py`
- `server/graders.py`
- `server/scenario_generator.py`
- `server/schema_drift.py`
- `server/stakeholders.py`
- `models.py`
- `server/email_triage_dataset.json`

Also add `app.py` in Space root:

```python
from server.ui import build_ui

demo = build_ui()

if __name__ == "__main__":
    demo.launch()
```

And `requirements.txt`:

```txt
gradio
pydantic
fastapi
numpy
```

If Space logs show a missing package, add it and redeploy.

## 4) Final Project Showcase Flow (What To Say)

Use this exact storyline in your final presentation:

1. **Problem**
   - "Single-agent setups fail in realistic inbox workflows."
2. **What you built**
   - "A coordinator RL agent supervising 4 specialists under schema drift."
3. **How you trained**
   - "GRPO with 5 independent rewards on Colab T4."
4. **Result**
   - "Model learns better triage/oversight behavior than naive specialist-trust baseline."
5. **Live demo**
   - run Space, show one hard/adversarial queue, highlight reward breakdown and drift adaptation.

## 5) Demo Script (2-3 Minutes)

1. Open Space and show hero panel
2. Select `hard` difficulty -> Start Queue
3. Show specialist conflict and your chosen action
4. Submit decisions and point at reward components
5. Trigger/observe drift warning and explain adaptation
6. End with final score + Hub model link

## 6) T4-Safe Defaults (Recommended)

- model: `Qwen/Qwen2-0.5B`
- steps: `30-50`
- dataset size: `32-64`
- keep runs short, save checkpoints often
- do 1 smoke run + 1 full run + optional second tuning run

## 7) What Helps You Win

- clean repo and reproducible commands
- clear metric story (before vs after training)
- stable Space with polished UI text/theme
- confident live walkthrough with no setup surprises

If you have extra time:
- run 2 seeds and report average reward
- upload a short demo clip + Space URL + Model URL together in your submission
