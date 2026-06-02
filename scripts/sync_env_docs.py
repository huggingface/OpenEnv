#!/usr/bin/env python3
"""Sync environment documentation with the envs/ directory.

Ensures every environment in envs/ has:
  1. A doc stub in docs/source/environments/<slug>.md
  2. A card in docs/source/environments.md
  3. A toctree entry in docs/source/environments.md

Also detects orphaned docs that reference envs which no longer exist.

Modes:
  --check   : Exit non-zero if out of sync (for CI)
  --fix     : Auto-add missing entries and remove orphaned ones
  --dry-run : Preview what --fix would do without writing anything
"""

import argparse
import os
import re
import sys
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENVS_DIR = os.path.join(ROOT, "envs")
DOCS_ENVS_DIR = os.path.join(ROOT, "docs", "source", "environments")
ENVIRONMENTS_MD = os.path.join(ROOT, "docs", "source", "environments.md")

SKIP_ENTRIES = {"README.md"}


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------


def get_env_dirs():
    """Return sorted list of environment directory names under envs/."""
    return sorted(
        d
        for d in os.listdir(ENVS_DIR)
        if os.path.isdir(os.path.join(ENVS_DIR, d)) and d not in SKIP_ENTRIES
    )


def get_existing_stub_mapping():
    """Build slug → env_dir mapping by parsing {include} directives in stubs."""
    mapping = {}
    for fname in os.listdir(DOCS_ENVS_DIR):
        if not fname.endswith(".md"):
            continue
        slug = fname[:-3]
        stub_path = os.path.join(DOCS_ENVS_DIR, fname)
        with open(stub_path) as f:
            content = f.read()
        match = re.search(
            r"\{include\}\s+\.\./\.\./\.\./envs/([^/]+)/README\.md", content
        )
        if match:
            mapping[slug] = match.group(1)
    return mapping


def get_toctree_slugs(content):
    """Extract environment slugs from the toctree block."""
    match = re.search(
        r"```\{toctree\}[^\n]*\n(?:[^\n]*\n)*?\n(.*?)```",
        content,
        re.DOTALL,
    )
    if not match:
        return []
    return [
        line.strip().replace("environments/", "")
        for line in match.group(1).strip().splitlines()
        if line.strip().startswith("environments/")
    ]


def get_card_slugs(content):
    """Extract environment slugs from card button-links in environments.md."""
    return re.findall(r"button-link\}\s*environments/([\w-]+)\.html", content)


# ---------------------------------------------------------------------------
# README parsing
# ---------------------------------------------------------------------------


def parse_frontmatter(env_dir):
    """Return dict of YAML frontmatter from envs/<dir>/README.md, or {}."""
    readme = os.path.join(ENVS_DIR, env_dir, "README.md")
    if not os.path.exists(readme):
        return {}
    with open(readme) as f:
        text = f.read()
    if not text.startswith("---"):
        return {}
    try:
        end = text.index("---", 3)
        import yaml

        return yaml.safe_load(text[3:end]) or {}
    except (ValueError, Exception):
        return {}


def parse_description(env_dir):
    """Extract the first paragraph after the H1 heading in README.md."""
    readme = os.path.join(ENVS_DIR, env_dir, "README.md")
    if not os.path.exists(readme):
        return "TODO: Add description."
    with open(readme) as f:
        text = f.read()
    if text.startswith("---"):
        try:
            end = text.index("---", 3)
            text = text[end + 3 :].strip()
        except ValueError:
            pass
    lines = text.splitlines()
    found_h1 = False
    desc_lines = []
    for line in lines:
        if line.startswith("# "):
            found_h1 = True
            continue
        if found_h1:
            stripped = line.strip()
            if stripped == "":
                if desc_lines:
                    break
                continue
            desc_lines.append(stripped)
    desc = " ".join(desc_lines) if desc_lines else "TODO: Add description."
    desc = re.sub(r"\*\*([^*]+)\*\*", r"\1", desc)
    desc = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", desc)
    if len(desc) > 160:
        desc = desc[:157] + "..."
    return desc


def derive_display_name(env_dir, frontmatter):
    """Derive a human-readable display name for a card title."""
    title = frontmatter.get("title", "")
    if title:
        name = re.sub(r"\s*(Environment\s*)?Server\s*$", "", title, flags=re.I).strip()
        name = re.sub(
            r"\s*Environment(\s+for\s+OpenEnv)?\s*$", "", name, flags=re.I
        ).strip()
        if name:
            return name
    slug = env_dir_to_slug(env_dir)
    return slug.replace("_", " ").title()


def env_dir_to_slug(env_dir):
    """Convert an env directory name to a doc slug (best-effort default)."""
    slug = env_dir
    if slug.endswith("_env"):
        slug = slug[:-4]
    return slug


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


def generate_stub(env_dir):
    return f"```{{include}} ../../../envs/{env_dir}/README.md\n```\n"


def generate_card(display_name, slug, description):
    return textwrap.dedent(
        f"""\
        ````{{grid-item-card}} {display_name}
        :class-card: sd-border-1

        {description}

        +++
        ```{{button-link}} environments/{slug}.html
        :color: primary
        :outline:

        {{octicon}}`file;1em` Docs
        ```
        ````"""
    )


def generate_toctree_entry(slug):
    return f"environments/{slug}"


# ---------------------------------------------------------------------------
# Insertion / removal helpers
# ---------------------------------------------------------------------------


def insert_card(content, card_block):
    """Insert a card block before the closing ````` of the first grid."""
    marker = "\n`````\n\n```{tip}"
    if marker not in content:
        marker = "\n`````\n"
    if marker not in content:
        raise RuntimeError(
            "Could not find grid closing marker in environments.md. "
            "The file structure may have changed — add the card manually."
        )
    return content.replace(marker, f"\n{card_block}\n{marker}", 1)


def insert_toctree_entry(content, entry):
    """Append a toctree entry at the end of the toctree block."""
    match = re.search(
        r"(```\{toctree\}[^\n]*\n(?:[^\n]*\n)*?\n)(.*?)(```)",
        content,
        re.DOTALL,
    )
    if not match:
        raise RuntimeError(
            "Could not find toctree block in environments.md. "
            "The file structure may have changed — add the entry manually."
        )
    before = content[: match.end(2)]
    after = content[match.end(2) :]
    return before + entry + "\n" + after


def remove_card(content, slug):
    """Remove a card block matching the given slug."""
    pattern = re.compile(
        rf"````\{{grid-item-card\}}[^\n]*\n"
        rf"(?:(?!````\{{grid-item-card\}}).)*?"
        rf"```\{{button-link\}}\s*environments/{re.escape(slug)}\.html.*?"
        rf"````\n?",
        re.DOTALL,
    )
    return pattern.sub("", content)


def remove_toctree_entry(content, slug):
    """Remove a toctree entry for the given slug."""
    return re.sub(rf"\n\s*environments/{re.escape(slug)}\s*(?=\n)", "", content)


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------


def analyze(env_dirs, stub_mapping, content):
    """Return (missing, orphaned, no_readme) lists."""
    toctree_slugs = set(get_toctree_slugs(content))
    card_slugs = set(get_card_slugs(content))
    stub_slugs = set(stub_mapping.keys())

    reverse_map = {v: k for k, v in stub_mapping.items()}

    documented_env_dirs = set(stub_mapping.values())

    missing = []
    no_readme = []
    for env_dir in env_dirs:
        readme = os.path.join(ENVS_DIR, env_dir, "README.md")
        if not os.path.exists(readme):
            no_readme.append(env_dir)
            continue
        if env_dir in documented_env_dirs:
            slug = reverse_map[env_dir]
        else:
            slug = env_dir_to_slug(env_dir)

        issues = []
        if slug not in stub_slugs:
            issues.append("stub")
        if slug not in card_slugs:
            issues.append("card")
        if slug not in toctree_slugs:
            issues.append("toctree")
        if issues:
            missing.append((env_dir, slug, issues))

    orphaned = []
    env_dir_set = set(env_dirs)
    for slug, env_dir in stub_mapping.items():
        if env_dir not in env_dir_set:
            issues = []
            if slug in stub_slugs:
                issues.append("stub")
            if slug in card_slugs:
                issues.append("card")
            if slug in toctree_slugs:
                issues.append("toctree")
            if issues:
                orphaned.append((env_dir, slug, issues))

    return missing, orphaned, no_readme


def run_check(missing, orphaned, no_readme):
    """Print report and return exit code."""
    ok = True

    if no_readme:
        print(
            "⚠️  The following environments have no README.md and will not appear on the docs site:\n"
        )
        for env_dir in no_readme:
            print(f"  envs/{env_dir}/")
        print()
        print("  Add a README.md to include them in the documentation.")
        print("  This is a warning only — it will not block your PR.\n")

    if missing:
        ok = False
        print("❌ Missing documentation for the following environments:\n")
        for env_dir, slug, issues in missing:
            print(f"  envs/{env_dir}/  →  slug '{slug}'  missing: {', '.join(issues)}")
        print()
        print("  Every environment in envs/ with a README.md must have:")
        print("    1. A doc stub at docs/source/environments/<slug>.md")
        print("    2. A card in the grid in docs/source/environments.md")
        print("    3. A toctree entry in docs/source/environments.md")
        print()
        print("  To fix automatically, run:")
        print("    python scripts/sync_env_docs.py --fix")
        print()
        print("  Or to preview what would change first:")
        print("    python scripts/sync_env_docs.py --dry-run")
        print()
        print(
            "  See docs/README.md § 'Adding an Environment to the Docs' for manual steps."
        )
        print()

    if orphaned:
        ok = False
        print("⚠️  Orphaned documentation (env directory no longer exists):\n")
        for env_dir, slug, issues in orphaned:
            print(f"  slug '{slug}'  →  envs/{env_dir}/  orphaned: {', '.join(issues)}")
        print()
        print("  These doc entries reference environments that have been removed.")
        print("  To clean up automatically, run:")
        print("    python scripts/sync_env_docs.py --fix")
        print()

    if ok:
        print("✅ All environments are documented and no orphans found.")
    return 0 if ok else 1


def run_fix(missing, orphaned, content, dry_run=False):
    """Apply fixes. Returns (new_content, files_created, files_deleted)."""
    files_created = []
    files_deleted = []
    new_content = content

    for env_dir, slug, issues in missing:
        fm = parse_frontmatter(env_dir)
        display_name = derive_display_name(env_dir, fm)
        description = parse_description(env_dir)

        if "stub" in issues:
            stub_path = os.path.join(DOCS_ENVS_DIR, f"{slug}.md")
            stub_content = generate_stub(env_dir)
            if dry_run:
                print(f"  [dry-run] Would create {os.path.relpath(stub_path, ROOT)}")
            else:
                with open(stub_path, "w") as f:
                    f.write(stub_content)
                print(f"  ✅ Created {os.path.relpath(stub_path, ROOT)}")
            files_created.append(stub_path)

        if "card" in issues:
            card_block = generate_card(display_name, slug, description)
            new_content = insert_card(new_content, card_block)
            action = "[dry-run] Would add" if dry_run else "✅ Added"
            print(f"  {action} card for '{display_name}' (slug: {slug})")

        if "toctree" in issues:
            entry = generate_toctree_entry(slug)
            new_content = insert_toctree_entry(new_content, entry)
            action = "[dry-run] Would add" if dry_run else "✅ Added"
            print(f"  {action} toctree entry for '{slug}'")

    for env_dir, slug, issues in orphaned:
        if "stub" in issues:
            stub_path = os.path.join(DOCS_ENVS_DIR, f"{slug}.md")
            if os.path.exists(stub_path):
                if dry_run:
                    print(
                        f"  [dry-run] Would delete {os.path.relpath(stub_path, ROOT)}"
                    )
                else:
                    os.remove(stub_path)
                    print(f"  🗑️  Deleted {os.path.relpath(stub_path, ROOT)}")
                files_deleted.append(stub_path)

        if "card" in issues:
            new_content = remove_card(new_content, slug)
            action = "[dry-run] Would remove" if dry_run else "🗑️  Removed"
            print(f"  {action} card for slug '{slug}'")

        if "toctree" in issues:
            new_content = remove_toctree_entry(new_content, slug)
            action = "[dry-run] Would remove" if dry_run else "🗑️  Removed"
            print(f"  {action} toctree entry for '{slug}'")

    if not dry_run and new_content != content:
        with open(ENVIRONMENTS_MD, "w") as f:
            f.write(new_content)
        print(f"\n  ✅ Updated {os.path.relpath(ENVIRONMENTS_MD, ROOT)}")

    return new_content, files_created, files_deleted


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--check", action="store_true", help="Check sync status (CI mode)"
    )
    group.add_argument(
        "--fix", action="store_true", help="Auto-fix missing and orphaned docs"
    )
    group.add_argument(
        "--dry-run", action="store_true", help="Preview --fix without writing"
    )
    args = parser.parse_args()

    env_dirs = get_env_dirs()
    stub_mapping = get_existing_stub_mapping()

    with open(ENVIRONMENTS_MD) as f:
        content = f.read()

    missing, orphaned, no_readme = analyze(env_dirs, stub_mapping, content)

    if args.check:
        sys.exit(run_check(missing, orphaned, no_readme))
    elif args.fix:
        if no_readme:
            print(
                "⚠️  The following environments have no README.md and will not appear on the docs site:\n"
            )
            for env_dir in no_readme:
                print(f"  envs/{env_dir}/")
            print()
            print("  Add a README.md to include them in the documentation.")
            print()
        if not missing and not orphaned:
            print("✅ Everything is already in sync.")
            return
        print("Fixing documentation...\n")
        run_fix(missing, orphaned, content, dry_run=False)
    elif args.dry_run:
        if no_readme:
            print(
                "⚠️  The following environments have no README.md and will not appear on the docs site:\n"
            )
            for env_dir in no_readme:
                print(f"  envs/{env_dir}/")
            print()
            print("  Add a README.md to include them in the documentation.")
            print()
        if not missing and not orphaned:
            print("✅ Everything is already in sync. Nothing to do.")
            return
        print("Dry run — no files will be modified:\n")
        run_fix(missing, orphaned, content, dry_run=True)


if __name__ == "__main__":
    main()
