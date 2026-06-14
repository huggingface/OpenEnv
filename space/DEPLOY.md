# How to deploy this Space

## One-time setup
```bash
pip install -U huggingface_hub
huggingface-cli login
huggingface-cli repo create Rhushya/oversight-inbox-arena --type space --space-sdk gradio
git clone https://huggingface.co/spaces/Rhushya/oversight-inbox-arena
cd oversight-inbox-arena
```

## Copy these files from OpenEnv/space/ into the Space root
```
README.md
app.py
requirements.txt
ui.py                        (from envs/email_triage_env/server/)
email_triage_environment.py (from envs/email_triage_env/server/)
graders.py                   (from envs/email_triage_env/server/)
scenario_generator.py        (from envs/email_triage_env/server/)
schema_drift.py              (from envs/email_triage_env/server/)
stakeholders.py              (from envs/email_triage_env/server/)
models.py                    (from envs/email_triage_env/)
email_triage_dataset.json    (from envs/email_triage_env/server/)
types.py                     (from src/openenv/core/env_server/)
__init__.py                  (empty file)
```

## Push
```bash
git add .
git commit -m 'Deploy oversight inbox arena'
git push
```

Space will be live at: https://huggingface.co/spaces/Rhushya/oversight-inbox-arena
