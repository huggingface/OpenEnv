# Building the Docs Locally

## Prerequisites

- Python 3.11+

## Setup

Install OpenEnv with the docs dependencies:

```bash
pip install -e ".[docs]"
```

## Build

From the `docs/` directory:

```bash
cd docs
make html
```

The output will be in `docs/_build/html/`.

## Preview

From the repo root, start a local server:

```bash
cd docs/_build/html
python -m http.server 8000
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.

### Build Variants

| Command | Description |
|---------|-------------|
| `make html` | Full build with Sphinx Gallery execution |
| `make html-noplot` | Skip gallery execution (faster) |
| `make html-stable` | Build as a versioned release |
| `make clean html` | Clean rebuild from scratch |

## How the Getting Started Section Builds

The Getting Started section is powered by [Sphinx Gallery](https://sphinx-gallery.github.io/). Source files live in `docs/source/getting_started/`:

- **`plot_*.py`** — executable Python scripts that Sphinx Gallery converts into rendered notebook pages
- **`*.md`** — static Markdown pages (environment-builder, contributing-envs)
- **`README.rst`** — gallery index template

During the build, Sphinx Gallery processes these sources and writes the output into a generated `auto_getting_started/` directory. A custom `copy_md_pages_to_gallery` hook in `conf.py` copies the static `.md` pages into that same output directory so they appear alongside the gallery notebooks in the left nav.

Because `getting_started/*.md` is in `exclude_patterns` in `conf.py`, Sphinx only generates HTML from the `auto_getting_started/` output — not from the source directory directly. This means all internal links to Getting Started pages must use the `auto_getting_started/` path (e.g. `auto_getting_started/environment-builder.md`). Linking to `getting_started/` will 404.

The copy hook runs on the `builder-inited` event, so static pages are available in every build variant including `make html-noplot`. That flag only skips executing the `plot_*.py` gallery scripts; it does not skip the page copy.

## Adding an Environment to the Docs

Every environment page is generated from the environment's own `README.md` using a Sphinx `{include}` directive. There are three steps:

### 1. Write the environment README

Your environment must have a `README.md` at `envs/<name>/README.md`. This file is the single source of truth — it renders on GitHub and is pulled into the docs site at build time.

Include HuggingFace frontmatter at the top, followed by a `# Title` heading (this becomes the page title and left nav label):

```markdown
---
title: My Environment
emoji: 🎮
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# My Environment

Description, quick start, action/observation docs, etc.
```

### 2. Create the doc page

Create `docs/source/environments/<name>.md` with exactly this content:

````markdown
```{include} ../../../envs/<name>/README.md
```
````

This is the only pattern used — all 29 environment doc pages follow it. Do not add local headings or other content.

### 3. Add a card and toctree entry

Edit `docs/source/environments.md` to add two things:

* **A card** inside the existing `{grid}` block (place alphabetically):

  ````markdown
  ````{grid-item-card} My Environment
  :class-card: sd-border-1

  Short one-line description of the environment.

  +++
  ```{button-link} environments/<name>.html
  :color: primary
  :outline:

  {octicon}`file;1em` Docs
  ```
  ```{button-link} https://huggingface.co/spaces/<org>/<name>
  :color: warning
  :outline:

  🤗 Hugging Face
  ```
  ````

  The Hugging Face button is optional — omit it if the environment isn't deployed to a Space.

* **A toctree entry** in the `{toctree}` block at the bottom of the file (place alphabetically):

  ```
  environments/<name>
  ```

### Verify

Rebuild and check that the environment appears in the left nav and the catalog grid:

```bash
cd docs && make clean html
cd _build/html && python -m http.server 8000
```

## Version Switcher

The docs site includes a version dropdown in the top-left navbar (next to the logo) powered by the `pytorch-sphinx-theme2` theme. It is configured by three pieces:

### `versions.json`

The file `docs/source/_static/versions.json` lists all published versions. Each entry has a `name` (display label), `version` (matching key), `url` (root URL for that version), and optionally `preferred: true` for the default. Currently it contains only the `main` development build:

```json
[
  {
    "name": "main",
    "version": "main",
    "url": "https://meta-pytorch.org/OpenEnv/",
    "preferred": true
  }
]
```

### Build-time version detection

`conf.py` reads the version from `pyproject.toml` and uses the `RELEASE` environment variable to decide which mode to build in:

- **`make html`** (default) — builds as `main`. The switcher highlights the "main" entry.
- **`make html-stable`** — sets `RELEASE=true`. The version is extracted from `pyproject.toml` (e.g. `0.2` from `0.2.2.dev0`) and the switcher highlights that version's entry.

### Publishing a new version

When cutting a release:

1. Ensure `pyproject.toml` has the release version (e.g. `0.2.0`)
2. Build with `make html-stable` — this produces docs tagged as version `0.2`
3. Deploy the output to a versioned path (e.g. `https://meta-pytorch.org/OpenEnv/0.2/`)
4. Add an entry to `docs/source/_static/versions.json`:

```json
{
  "name": "0.2",
  "version": "0.2",
  "url": "https://meta-pytorch.org/OpenEnv/0.2/"
}
```

5. Rebuild and redeploy `main` so its copy of `versions.json` includes the new entry

The switcher on every version of the site fetches `versions.json` at page load, so all versions see the full list once the file is updated.
