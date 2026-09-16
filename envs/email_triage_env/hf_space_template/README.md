# Hugging Face Space Template (Oversight Inbox Arena)

Use this folder to launch a Gradio Space quickly.

## Files to copy with this template

From `envs/email_triage_env/`, copy these into the Space repo root:

- `hf_space_template/app.py`
- `hf_space_template/requirements.txt`
- `server/ui.py`
- `server/email_triage_environment.py`
- `server/graders.py`
- `server/scenario_generator.py`
- `server/schema_drift.py`
- `server/stakeholders.py`
- `server/email_triage_dataset.json`
- `models.py`

After copying:

- Ensure file layout in Space root is:
  - `app.py`
  - `requirements.txt`
  - `server/`
  - `models.py`

## Create and push a Space

```bash
pip install -U huggingface_hub
hf auth login
hf repo create YOUR_USERNAME/oversight-inbox-arena --type space --space-sdk gradio
```

Then push files to your Space repository.

## Notes

- If build logs show missing package, add it to `requirements.txt`.
- Keep Space public for hackathon judging.
