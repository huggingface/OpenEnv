#!/usr/bin/env python3
"""Sync environment documentation with the envs/ directory.

Ensures every environment in envs/ with a README.md has a corresponding
doc stub in docs/source/environments/<slug>.md, and that existing stubs
stay in sync with their source README.

Also detects orphaned stubs that reference envs which no longer exist.

Modes:
  --check   : Exit non-zero if out of sync (for CI)
  --fix     : Auto-create missing stubs, refresh stale ones, delete orphans
  --dry-run : Preview what --fix would do without writing anything

Note: entries in docs/source/environments.md (HTML catalog) and
docs/source/_toctree.yml are managed manually. This script only
manages the per-environment stub files.
"""

import argparse
import os
import posixpath
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENVS_DIR = os.path.join(ROOT, "envs")
DOCS_ENVS_DIR = os.path.join(ROOT, "docs", "source", "environments")
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/huggingface/OpenEnv/main"
GITHUB_BLOB_BASE = "https://github.com/huggingface/OpenEnv/blob/main"
GITHUB_TREE_BASE = "https://github.com/huggingface/OpenEnv/tree/main"

SKIP_DIRS = {"README.md"}
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------


def get_env_dirs():
    """Return sorted list of environment directory names under envs/."""
    return sorted(
        d
        for d in os.listdir(ENVS_DIR)
        if os.path.isdir(os.path.join(ENVS_DIR, d)) and d not in SKIP_DIRS
    )


def get_existing_stub_mapping():
    """Build slug → env_dir mapping by reading stubs in docs/source/environments/.

    Supports two formats:
    - New: ``<!-- openenv-source: env_dir -->`` comment at top of file
    - Legacy: ``{include} ../../../envs/<env_dir>/README.md`` directive
    """
    mapping = {}
    for fname in os.listdir(DOCS_ENVS_DIR):
        if not fname.endswith(".md"):
            continue
        slug = fname[:-3]
        stub_path = os.path.join(DOCS_ENVS_DIR, fname)
        with open(stub_path) as f:
            content = f.read()
        # New format: <!-- openenv-source: env_dir -->
        match = re.search(r"<!--\s*openenv-source:\s*(\S+)\s*-->", content)
        if match:
            mapping[slug] = match.group(1)
            continue
        # Legacy format: {include}
        match = re.search(
            r"\{include\}\s+\.\./\.\./\.\./envs/([^/]+)/README\.md", content
        )
        if match:
            mapping[slug] = match.group(1)
    return mapping


def env_dir_to_slug(env_dir):
    """Convert an env directory name to a doc slug (best-effort default)."""
    slug = env_dir
    if slug.endswith("_env"):
        slug = slug[:-4]
    return slug


# ---------------------------------------------------------------------------
# README helpers
# ---------------------------------------------------------------------------


def _strip_frontmatter(text):
    """Remove YAML frontmatter (--- ... ---) from the start of text."""
    if text.startswith("---"):
        try:
            end = text.index("---", 3)
            return text[end + 3 :].lstrip("\n")
        except ValueError:
            pass
    return text


def _is_absolute_or_external_url(url):
    """Return True if *url* should be left untouched (http, mailto, anchors, site-root)."""
    url = url.strip()
    if not url:
        return True
    if url.startswith(("#", "/", "http://", "https://", "mailto:")):
        return True
    if "://" in url:
        return True
    return False


def repo_relpath_from_readme_link(env_dir, url):
    """Resolve a README-relative link to a repo-root POSIX path, or None.

    Extra ``../`` segments that walk out of the repository (a common README
    mistake) are stripped so ``envs/git_env/../../../examples/foo.py`` still
    maps to ``examples/foo.py``.
    """
    path = url.split()[0].strip()
    path, hash_sep, anchor = path.partition("#")
    if not path or _is_absolute_or_external_url(path):
        return None
    joined = posixpath.normpath(posixpath.join("envs", env_dir, path))
    while joined.startswith("../"):
        joined = joined[3:]
    if joined in {".", ""} or joined.startswith(".."):
        return None
    if hash_sep:
        return f"{joined}#{anchor}"
    return joined


def _link_escapes_env_dir(env_dir, repo_relpath):
    """True when the resolved path is outside ``envs/<env_dir>/``."""
    path_only = repo_relpath.split("#", 1)[0]
    prefix = f"envs/{env_dir}"
    return path_only != prefix and not path_only.startswith(prefix + "/")


def rewrite_markdown_links(content, env_dir):
    """Rewrite README-relative markdown links that leave the env directory.

    Those links are valid from ``envs/<env>/README.md`` but break once the
    README is inlined at ``docs/source/environments/<slug>.md``. Convert them
    to GitHub blob/tree URLs, matching the convention already used by some
    env READMEs. See https://github.com/huggingface/OpenEnv/issues/1095
    """

    def repl(match):
        text, url = match.group(1), match.group(2)
        raw_url = url.split()[0]
        repo_rel = repo_relpath_from_readme_link(env_dir, raw_url)
        if repo_rel is None or not _link_escapes_env_dir(env_dir, repo_rel):
            return match.group(0)
        path_only, hash_sep, anchor = repo_rel.partition("#")
        abs_fs = os.path.join(ROOT, *path_only.split("/"))
        is_image = match.start() > 0 and content[match.start() - 1] == "!"
        url_path = raw_url.split("#", 1)[0]
        if is_image:
            new_url = f"{GITHUB_RAW_BASE}/{path_only}"
        elif url_path.endswith("/") or os.path.isdir(abs_fs):
            new_url = f"{GITHUB_TREE_BASE}/{path_only}"
        else:
            new_url = f"{GITHUB_BLOB_BASE}/{path_only}"
        if hash_sep:
            new_url = f"{new_url}#{anchor}"
        return f"[{text}]({new_url})"

    return _MD_LINK_RE.sub(repl, content)


def generate_stub(env_dir):
    """Return the doc stub content for an environment.

    Inlines the README content (without HF Spaces YAML frontmatter) and
    prepends a ``<!-- openenv-source: env_dir -->`` comment so the script
    can identify the source env when re-reading the file later.
    """
    readme_path = os.path.join(ENVS_DIR, env_dir, "README.md")
    with open(readme_path) as f:
        content = f.read()
    content = _strip_frontmatter(content)
    # Rewrite relative assets/ paths to absolute GitHub raw URLs so images
    # render correctly when the README is inlined into the doc-builder site.
    base_url = f"{GITHUB_RAW_BASE}/envs/{env_dir}"
    content = re.sub(r'(src=["\'])assets/', rf"\1{base_url}/assets/", content)
    content = rewrite_markdown_links(content, env_dir)
    return f"<!-- openenv-source: {env_dir} -->\n{content}"


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def analyze(env_dirs, stub_mapping):
    """Return (missing, orphaned, stale, no_readme) lists."""
    reverse_map = {v: k for k, v in stub_mapping.items()}
    documented_env_dirs = set(stub_mapping.values())

    missing = []
    stale = []
    no_readme = []

    for env_dir in env_dirs:
        readme = os.path.join(ENVS_DIR, env_dir, "README.md")
        if not os.path.exists(readme):
            no_readme.append(env_dir)
            continue

        if env_dir in documented_env_dirs:
            slug = reverse_map[env_dir]
            # Check if existing stub is stale (content drifted from README)
            stub_path = os.path.join(DOCS_ENVS_DIR, f"{slug}.md")
            if os.path.exists(stub_path):
                expected = generate_stub(env_dir)
                with open(stub_path) as f:
                    actual = f.read()
                if actual != expected:
                    stale.append((env_dir, slug))
        else:
            slug = env_dir_to_slug(env_dir)
            missing.append((env_dir, slug))

    orphaned = []
    env_dir_set = set(env_dirs)
    for slug, env_dir in stub_mapping.items():
        if env_dir not in env_dir_set:
            orphaned.append((env_dir, slug))

    return missing, orphaned, stale, no_readme


# ---------------------------------------------------------------------------
# Reporting and fixing
# ---------------------------------------------------------------------------


def run_check(missing, orphaned, stale, no_readme):
    ok = True

    if no_readme:
        print(
            "⚠️  The following environments have no README.md and will not appear on the docs site:\n"
        )
        for env_dir in no_readme:
            print(f"  envs/{env_dir}/")
        print()
        print("  This is a warning only — it will not block your PR.\n")

    if missing:
        ok = False
        print("❌ Missing stubs for the following environments:\n")
        for env_dir, slug in missing:
            print(f"  envs/{env_dir}/  →  docs/source/environments/{slug}.md")
        print()
        print("  Run:  python scripts/sync_env_docs.py --fix\n")

    if stale:
        ok = False
        print("⚠️  The following stubs are out of date with their source README:\n")
        for env_dir, slug in stale:
            print(f"  docs/source/environments/{slug}.md  ←  envs/{env_dir}/README.md")
        print()
        print("  Run:  python scripts/sync_env_docs.py --fix\n")

    if orphaned:
        ok = False
        print("⚠️  Orphaned stubs (env directory no longer exists):\n")
        for env_dir, slug in orphaned:
            print(f"  docs/source/environments/{slug}.md  (was envs/{env_dir}/)")
        print()
        print("  Run:  python scripts/sync_env_docs.py --fix\n")

    if ok:
        print("✅ All environment stubs are present and up to date.")
    return 0 if ok else 1


def run_fix(missing, orphaned, stale, dry_run=False):
    def label(action):
        return f"[dry-run] Would {action}" if dry_run else action

    for env_dir, slug in missing:
        stub_path = os.path.join(DOCS_ENVS_DIR, f"{slug}.md")
        if dry_run:
            print(f"  {label('create')} {os.path.relpath(stub_path, ROOT)}")
        else:
            with open(stub_path, "w") as f:
                f.write(generate_stub(env_dir))
            print(f"  ✅ Created {os.path.relpath(stub_path, ROOT)}")

    for env_dir, slug in stale:
        stub_path = os.path.join(DOCS_ENVS_DIR, f"{slug}.md")
        if dry_run:
            print(f"  {label('refresh')} {os.path.relpath(stub_path, ROOT)}")
        else:
            with open(stub_path, "w") as f:
                f.write(generate_stub(env_dir))
            print(f"  🔄 Refreshed {os.path.relpath(stub_path, ROOT)}")

    for env_dir, slug in orphaned:
        stub_path = os.path.join(DOCS_ENVS_DIR, f"{slug}.md")
        if os.path.exists(stub_path):
            if dry_run:
                print(f"  {label('delete')} {os.path.relpath(stub_path, ROOT)}")
            else:
                os.remove(stub_path)
                print(f"  🗑️  Deleted {os.path.relpath(stub_path, ROOT)}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--check", action="store_true", help="Check sync status (CI mode)"
    )
    group.add_argument(
        "--fix", action="store_true", help="Fix missing, stale, and orphaned stubs"
    )
    group.add_argument(
        "--dry-run", action="store_true", help="Preview --fix without writing"
    )
    args = parser.parse_args()

    env_dirs = get_env_dirs()
    stub_mapping = get_existing_stub_mapping()
    missing, orphaned, stale, no_readme = analyze(env_dirs, stub_mapping)

    if args.check:
        sys.exit(run_check(missing, orphaned, stale, no_readme))

    # --fix and --dry-run
    if no_readme:
        print("⚠️  Environments without README.md (skipped):\n")
        for env_dir in no_readme:
            print(f"  envs/{env_dir}/")
        print()

    if not missing and not orphaned and not stale:
        print("✅ Everything is already in sync.")
        return

    print(
        "Fixing documentation...\n"
        if not args.dry_run
        else "Dry run — no files will be modified:\n"
    )
    run_fix(missing, orphaned, stale, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
