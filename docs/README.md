# Building the Docs Locally

The documentation is built with [HF doc-builder](https://github.com/huggingface/doc-builder).

## Prerequisites

- Python 3.11+
- Node.js 18+

## Setup

Install the package and doc-builder:

```bash
pip install -e ".[core,cli]"
pip install hf-doc-builder watchdog
```

## Preview

From the repo root:

```bash
doc-builder preview openenv docs/source
```

Then open the URL printed in the terminal (usually `http://localhost:5173/openenv/main/en/index`).

## Adding an Environment to the Docs

Every environment page is generated from the environment's own `README.md` using an `{include}` directive. There are three steps:

### 1. Write the environment README

Your environment must have a `README.md` at `envs/<name>/README.md`.

### 2. Create the doc page

Create `docs/source/environments/<name>.md` with exactly this content:

````markdown
```{include} ../../../envs/<name>/README.md
```
````

### 3. Add a card and toctree entry

- Add an HTML card in `docs/source/environments.md`
- Add a `local: environments/<name>` entry in `docs/source/_toctree.yml`
