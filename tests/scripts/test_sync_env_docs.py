"""Tests for scripts/sync_env_docs.py README inlining and link rewriting."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "sync_env_docs.py"
SPEC = importlib.util.spec_from_file_location("sync_env_docs", SCRIPT_PATH)
assert SPEC is not None
sync_env_docs = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules["sync_env_docs"] = sync_env_docs
SPEC.loader.exec_module(sync_env_docs)

repo_relpath_from_readme_link = sync_env_docs.repo_relpath_from_readme_link
rewrite_markdown_links = sync_env_docs.rewrite_markdown_links
generate_stub = sync_env_docs.generate_stub
GITHUB_BLOB_BASE = sync_env_docs.GITHUB_BLOB_BASE
GITHUB_TREE_BASE = sync_env_docs.GITHUB_TREE_BASE


def test_repo_relpath_strips_extra_parent_segments() -> None:
    # git_env README uses ../../../examples/... which walks out of the repo.
    rel = repo_relpath_from_readme_link("git_env", "../../../examples/local_git_env.py")
    assert rel == "examples/local_git_env.py"


def test_repo_relpath_from_env_readme() -> None:
    rel = repo_relpath_from_readme_link(
        "agent_world_model_env", "../../examples/agent_world_model/example_usage.py"
    )
    assert rel == "examples/agent_world_model/example_usage.py"


def test_rewrite_escapes_to_github_blob() -> None:
    src = "See [example](../../examples/repl_with_llm.py) please."
    out = rewrite_markdown_links(src, "repl_env")
    assert out == f"See [example]({GITHUB_BLOB_BASE}/examples/repl_with_llm.py) please."


def test_rewrite_directory_link_uses_tree() -> None:
    src = "The [`examples/carla_env/`](../../examples/carla_env/) directory."
    out = rewrite_markdown_links(src, "carla_env")
    assert f"]({GITHUB_TREE_BASE}/examples/carla_env/)" in out or (
        f"]({GITHUB_TREE_BASE}/examples/carla_env)" in out
    )


def test_rewrite_leaves_in_env_and_absolute_links() -> None:
    src = (
        "See [local](server/app.py) and "
        "[docs](https://github.com/huggingface/OpenEnv/blob/main/examples/x.py) "
        "and [section](#overview)."
    )
    assert rewrite_markdown_links(src, "git_env") == src


def test_generate_stub_rewrites_git_env_example_link() -> None:
    stub = generate_stub("git_env")
    assert f"]({GITHUB_BLOB_BASE}/examples/local_git_env.py)" in stub
    assert "](../../../examples/local_git_env.py)" not in stub


def test_generate_stub_rewrites_agent_world_model_example_link() -> None:
    stub = generate_stub("agent_world_model_env")
    assert f"]({GITHUB_BLOB_BASE}/examples/agent_world_model/example_usage.py)" in stub
    assert "](../../examples/agent_world_model/example_usage.py)" not in stub


def test_generate_stub_rewrites_carla_and_rfc_links() -> None:
    stub = generate_stub("carla_env")
    assert f"]({GITHUB_TREE_BASE}/examples/carla_env)" in stub or (
        f"]({GITHUB_TREE_BASE}/examples/carla_env/)" in stub
    )
    assert f"]({GITHUB_BLOB_BASE}/rfcs/004-rubrics.md)" in stub
    assert "](../../examples/carla_env/)" not in stub
    assert "](../../rfcs/004-rubrics.md)" not in stub


def test_generate_stub_rewrites_opencode_dead_relative_link() -> None:
    stub = generate_stub("opencode_env")
    assert "](../../../DOCS/HF/hf_inference_providers_logprobs.md)" not in stub
    assert f"]({GITHUB_BLOB_BASE}/DOCS/HF/hf_inference_providers_logprobs.md)" in stub
