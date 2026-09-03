---
title: LaTeX OCR Env
emoji: 📐
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
---

# LaTeX OCR Environment

A dataset-backed, single-step (bandit) RL environment for **image → LaTeX**
transcription, served through OpenEnv.

- **Task**: the agent is shown an image of a math/text expression and must
  return its LaTeX source.
- **Dataset**: tasks are served from a Hugging Face dataset (default
  [`unsloth/LaTeX_OCR`](https://huggingface.co/datasets/unsloth/LaTeX_OCR):
  `image` + `text` columns, `train`/`test` splits) via the OpenEnv **Task API**.
- **Reward**: `[(1 - exact_weight) * (1 - CER) + exact_weight * exact_match] * length_factor`,
  where `CER` is the normalized character edit distance over whitespace-stripped
  LaTeX. With the default `exact_weight=0.4` (tunable via `LATEX_OCR_EXACT_WEIGHT`),
  partial answers score in `[0, 0.6]`; only an exact match reaches `1.0`.
  Computed **server-side** against the hidden ground truth — the agent never
  sees the target on `reset`. Dense and smooth, for stable RL training.
- **Length guard**: because the score is whitespace-insensitive, a policy could
  otherwise emit the correct answer followed by unlimited whitespace and still
  score `1.0` — a reward hack that, in RL, shows up as completions drifting to
  the generation-length cap. `length_factor` decays the reward once the *raw*
  prediction grows past `LATEX_OCR_OVERLONG_RATIO`× the target length (default
  `4.0`, floor `LATEX_OCR_OVERLONG_FLOOR=80` chars); normal LaTeX spacing is well
  within the allowance. Set `LATEX_OCR_OVERLONG_RATIO=0` to disable. The rubric
  also strips code fences / `$`-delimiters from raw completions itself.

## Episode

```
reset(split="test", index=0)  -> observation { image_base64, prompt, ... }   # target hidden
step(LatexOCRAction(latex=…)) -> reward, done=True { target_latex, exact_match, char_error_rate }
```

## Task API

| Endpoint | Purpose |
|---|---|
| `GET  /latex_ocr_env/splits` | list splits (`train`, `test`) |
| `POST /latex_ocr_env/num_tasks` | row count for a split |
| `POST /latex_ocr_env/tasks` | all task specs for a split (materialize mode) |
| `POST /latex_ocr_env/task` | one task by `{split, index}` |
| `POST /latex_ocr_env/task_range` | slice `{split, start, stop}` |

Client helpers: `env.list_splits()`, `env.num_tasks(split)`,
`env.get_task(split, index)`, `env.get_task_range(split, start, stop)`.
In stream mode, full task enumeration returns `501`; use `num_tasks` and
bounded `get_task_range` calls instead.

## Run locally

```bash
# From the repo root. LATEX_OCR_SPLITS=test avoids the 380MB train download.
PYTHONPATH=src:envs/latex_ocr_env LATEX_OCR_SPLITS=test \
  uv run --with datasets --with pillow --with fastapi --with uvicorn --with websockets \
  uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Then drive it (needs `HF_TOKEN` for the real VLM policy):

```bash
HF_TOKEN=hf_xxx PYTHONPATH=src \
  uv run --with datasets --with pillow --with requests --with websockets --with openai \
  python envs/latex_ocr_env/validate.py --split test --num 3 \
  --model Qwen/Qwen2.5-VL-7B-Instruct
```

## Configuration (env vars)

| Var | Default | Meaning |
|---|---|---|
| `LATEX_OCR_DATASET` | `unsloth/LaTeX_OCR` | source dataset |
| `LATEX_OCR_IMAGE_COLUMN` | `image` | image column |
| `LATEX_OCR_TEXT_COLUMN` | `text` | ground-truth LaTeX column |
| `LATEX_OCR_SPLITS` | `train,test` | splits to expose |
| `LATEX_OCR_MAX_ROWS` | — | cap rows per split (dev) |
| `LATEX_OCR_EXACT_WEIGHT` | `0.4` | exact-match share of the reward |
| `LATEX_OCR_OVERLONG_RATIO` | `4.0` | raw length allowed as a multiple of the target before the reward decays (`0` disables the length guard) |
| `LATEX_OCR_OVERLONG_FLOOR` | `80` | minimum allowed raw length (chars) for short targets |

Swap in any `(image, latex)` dataset by pointing `LATEX_OCR_DATASET` at it (and
the column vars if they differ).
