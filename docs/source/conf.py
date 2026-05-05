# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "OpenEnv"
copyright = ""
author = ""

# -- Version configuration ---------------------------------------------------
# RELEASE env var controls stable vs dev builds (set by `make html-stable`)
RELEASE = os.environ.get("RELEASE", False)

# Read version from pyproject.toml
import tomli

pyproject_path = os.path.join(os.path.dirname(__file__), "..", "..", "pyproject.toml")
with open(pyproject_path, "rb") as f:
    pyproject_data = tomli.load(f)
openenv_version = pyproject_data["project"]["version"]

if RELEASE:
    version = ".".join(openenv_version.split(".")[:2])
    release = version
    html_title = f"OpenEnv {version} documentation"
    switcher_version = version
else:
    version = "main"
    release = "main"
    html_title = "OpenEnv"
    switcher_version = "main"

# -- Path setup --------------------------------------------------------------
sys.path.insert(0, os.path.abspath("../../src"))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_design",
    "sphinx_sitemap",
    "sphinxcontrib.mermaid",
    "pytorch_sphinx_theme2",
    "sphinxext.opengraph",
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx_gallery.gen_gallery",
]

# -- sphinx-gallery configuration --------------------------------------------
from sphinx_gallery.sorting import FileNameSortKey

sphinx_gallery_conf = {
    "examples_dirs": ["getting_started"],
    "gallery_dirs": ["auto_getting_started"],
    "filename_pattern": r"/plot_",
    "ignore_pattern": r"__init__\.py",
    "download_all_examples": False,
    "show_memory": False,
    "capture_repr": ("_repr_html_", "__repr__"),
    "matplotlib_animations": True,
    "remove_config_comments": True,
    "within_subsection_order": FileNameSortKey,
    "default_thumb_file": None,
    "nested_sections": False,
}

exclude_patterns = ["getting_started/*.md", "getting_started/README.rst"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

import pytorch_sphinx_theme2

html_theme = "pytorch_sphinx_theme2"
html_theme_path = [pytorch_sphinx_theme2.get_html_theme_path()]
html_static_path = ["_static"]

html_theme_options = {
    "navigation_with_keys": False,
    "analytics_id": "GTM-T8XT4PS",
    "header_links_before_dropdown": 7,
    "logo": {
        "text": "OpenEnv",
    },
    "icon_links": [
        {
            "name": "X",
            "url": "https://x.com/PyTorch",
            "icon": "fa-brands fa-x-twitter",
        },
        {
            "name": "GitHub",
            "url": "https://github.com/meta-pytorch/OpenEnv",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "Discourse",
            "url": "https://dev-discuss.pytorch.org/",
            "icon": "fa-brands fa-discourse",
        },
    ],
    "use_edit_page_button": True,
    "navbar_center": "navbar-nav",
    "switcher": {
        "json_url": "_static/versions.json",
        "version_match": switcher_version,
    },
    "check_switcher": False,
    "navbar_align": "left",
    "navbar_start": ["navbar-logo", "version-switcher"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
}

theme_variables = pytorch_sphinx_theme2.get_theme_variables()

# Templates path - local templates override theme templates
templates_path = [
    "_templates",
    os.path.join(os.path.dirname(pytorch_sphinx_theme2.__file__), "templates"),
]

html_context = {
    "theme_variables": theme_variables,
    "display_github": True,
    "github_url": "https://github.com",
    "github_user": "meta-pytorch",
    "github_repo": "OpenEnv",
    "feedback_url": "https://github.com/meta-pytorch/OpenEnv",
    "github_version": "main",
    "doc_path": "docs/source",
    "library_links": theme_variables.get("library_links", []),
    "community_links": theme_variables.get("community_links", []),
    "language_bindings_links": html_theme_options.get("language_bindings_links", []),
}

# Base URL for the site (used by sitemap and canonical URLs)
html_baseurl = "https://meta-pytorch.org/OpenEnv/"
sitemap_locales = [None]
sitemap_excludes = [
    "search.html",
    "genindex.html",
]
sitemap_url_scheme = "{link}"

# -- MyST-Parser configuration -----------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_image",
]


# -- Post-process sphinx-gallery output to fix navigation --------------------
def remove_orphan_and_duplicate_toctree(app, docname, source):
    """Remove :orphan: and duplicate hidden toctree from gallery index."""
    if docname == "auto_getting_started/index":
        content = source[0]
        # Remove the :orphan: directive
        if content.startswith(":orphan:"):
            content = content.replace(":orphan:\n\n", "", 1)
            content = content.replace(":orphan:\n", "", 1)

        # Remove the sphinx-gallery generated hidden toctree
        # Find and remove the hidden toctree block
        import re

        # Match: .. toctree::\n   :hidden:\n\n   /auto_getting_started/...
        pattern = r"\.\. toctree::\n\s+:hidden:\n\n(?:\s+/auto_getting_started/plot_\d+_\w+\n)+"
        content = re.sub(pattern, "", content)

        source[0] = content


def copy_md_pages_to_gallery(app):
    """Copy .md pages from getting_started/ to auto_getting_started/.

    Sphinx Gallery only processes .py files and README.rst.  Any extra .md
    pages that live alongside the gallery source must be copied into the
    generated gallery directory so Sphinx can discover them as part of the
    same toctree (important for section-nav context in pydata-sphinx-theme).
    """
    import glob
    import shutil

    srcdir = os.path.join(app.srcdir, "getting_started")
    dstdir = os.path.join(app.srcdir, "auto_getting_started")
    os.makedirs(dstdir, exist_ok=True)
    for md_file in glob.glob(os.path.join(srcdir, "*.md")):
        shutil.copy2(md_file, dstdir)


def setup(app):
    # Copy extra .md pages into the gallery output dir (priority 900 so it
    # runs after sphinx-gallery's builder-inited handler at default priority).
    app.connect("builder-inited", copy_md_pages_to_gallery, priority=900)
    # Hook into source-read to modify content before Sphinx processes it
    app.connect("source-read", remove_orphan_and_duplicate_toctree)
