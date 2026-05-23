#!/usr/bin/env python3
"""Gate 0.3 — Version compatibility smoke test for TRL + vLLM + Qwen3.5-4B.

Validates that the installed combination of TRL, vLLM, and transformers can:
  1. Import TRL's AsyncGRPOTrainer and AsyncGRPOConfig without errors.
  2. Import vLLM's NCCL weight-transfer modules.
  3. Load a Qwen3.5-4B tokenizer via transformers (proves native model support).
  4. (If vLLM server is running) Send a /v1/completions request and get a response.
  5. (If vLLM server is running) Validate weight-transfer endpoints exist.

Usage:
    # Minimal (no GPU / no running vLLM server):
    python examples/mini_swe_env/test_version_compat.py

    # Full (requires vLLM serving Qwen3.5-4B on localhost:8000):
    VLLM_URL=http://localhost:8000 python examples/mini_swe_env/test_version_compat.py

Environment variables:
    VLLM_URL          Base URL for vLLM server (default: http://localhost:8000)
    VLLM_API_KEY      API key for vLLM server (default: token)
    STUDENT_MODEL     Model ID for Qwen3.5-4B (default: unsloth/Qwen3.5-4B)
"""

from __future__ import annotations

import importlib
import os
import warnings

# ── Helpers ────────────────────────────────────────────────────────────


def _check(label: str, passed: bool, detail: str = "") -> bool:
    status = "✅ PASS" if passed else "❌ FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"  {status}  {label}{suffix}")
    return passed


def _version(pkg: str) -> str:
    try:
        mod = importlib.import_module(pkg)
        return getattr(mod, "__version__", "unknown")
    except ImportError:
        return "NOT INSTALLED"


# ── Gate checks ────────────────────────────────────────────────────────


def check_imports() -> bool:
    """Check 1: TRL AsyncGRPO imports."""
    print("\n[1/5] TRL AsyncGRPO imports")
    ok = True
    try:
        from trl.experimental.async_grpo import (  # noqa: F401
            AsyncGRPOConfig,
            AsyncGRPOTrainer,
        )

        ok = _check("AsyncGRPOConfig import", True) and ok
        ok = _check("AsyncGRPOTrainer import", True) and ok
    except Exception as exc:
        ok = _check("TRL AsyncGRPO imports", False, str(exc)) and ok
    return ok


def check_vllm_modules() -> bool:
    """Check 2: vLLM NCCL weight-transfer modules."""
    print("\n[2/5] vLLM weight-transfer modules")
    ok = True
    try:
        from vllm.distributed.weight_transfer.nccl_engine import (  # noqa: F401
            NCCLTrainerSendWeightsArgs,
            NCCLWeightTransferEngine,
        )

        ok = _check("NCCLWeightTransferEngine import", True) and ok
        ok = _check("NCCLTrainerSendWeightsArgs import", True) and ok
    except Exception as exc:
        ok = _check("NCCL engine imports", False, str(exc)) and ok

    try:
        from vllm.utils.network_utils import get_ip, get_open_port  # noqa: F401

        ok = _check("network_utils import", True) and ok
    except Exception as exc:
        ok = _check("network_utils import", False, str(exc)) and ok
    return ok


def check_tokenizer() -> bool:
    """Check 3: Qwen3.5-4B tokenizer loads via transformers."""
    print("\n[3/5] Qwen3.5-4B tokenizer (transformers native support)")
    model_id = os.environ.get("STUDENT_MODEL", "unsloth/Qwen3.5-4B")
    try:
        from transformers import AutoTokenizer

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
        return _check(
            f"AutoTokenizer.from_pretrained('{model_id}')",
            True,
            f"vocab_size={tok.vocab_size}",
        )
    except Exception as exc:
        return _check(f"tokenizer load for {model_id}", False, str(exc))


def check_vllm_completions() -> bool:
    """Check 4: vLLM /v1/completions responds (requires running server)."""
    print("\n[4/5] vLLM /v1/completions endpoint")
    vllm_url = os.environ.get("VLLM_URL", "http://localhost:8000").rstrip("/")
    api_key = os.environ.get("VLLM_API_KEY", "token")
    model_id = os.environ.get("STUDENT_MODEL", "unsloth/Qwen3.5-4B")

    try:
        import requests

        resp = requests.get(f"{vllm_url}/health", timeout=5)
        if resp.status_code != 200:
            return _check(
                "/health",
                False,
                f"status={resp.status_code} — is vLLM running at {vllm_url}?",
            )
    except Exception:
        return _check(
            "vLLM server reachable",
            False,
            f"Cannot connect to {vllm_url} — set VLLM_URL or start vLLM. Skipping.",
        )

    try:
        import requests

        body = {
            "model": model_id,
            "prompt": "Hello",
            "max_tokens": 8,
            "temperature": 0.0,
        }
        resp = requests.post(
            f"{vllm_url}/v1/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=30,
        )
        if resp.status_code != 200:
            return _check(
                "/v1/completions",
                False,
                f"status={resp.status_code}: {resp.text[:200]}",
            )
        data = resp.json()
        text = data["choices"][0].get("text", "")
        return _check("/v1/completions", True, f"generated {len(text)} chars")
    except Exception as exc:
        return _check("/v1/completions", False, str(exc))


def check_weight_transfer_endpoints() -> bool:
    """Check 5: Weight-transfer HTTP endpoints exist (requires running server)."""
    print("\n[5/5] Weight-transfer HTTP endpoints (VLLM_SERVER_DEV_MODE=1 required)")
    vllm_url = os.environ.get("VLLM_URL", "http://localhost:8000").rstrip("/")
    api_key = os.environ.get("VLLM_API_KEY", "token")

    try:
        import requests

        resp = requests.get(f"{vllm_url}/health", timeout=5)
        if resp.status_code != 200:
            return _check("vLLM reachable", False, "skipping endpoint checks")
    except Exception:
        return _check("vLLM server reachable", False, "skipping endpoint checks")

    ok = True
    headers = {"Authorization": f"Bearer {api_key}"}

    # Check /get_world_size
    try:
        resp = requests.get(f"{vllm_url}/get_world_size", headers=headers, timeout=5)
        ws = resp.json().get("world_size", "?") if resp.status_code == 200 else "?"
        ok = (
            _check(
                "GET /get_world_size",
                resp.status_code == 200,
                f"world_size={ws}"
                if resp.status_code == 200
                else f"status={resp.status_code}",
            )
            and ok
        )
    except Exception as exc:
        ok = _check("GET /get_world_size", False, str(exc)) and ok

    # Check that /pause and /resume exist (OPTIONS or try POST)
    for endpoint in ["/pause", "/resume"]:
        try:
            # Use POST with no body — will either work or return 422 (missing params)
            # but NOT 404 if the endpoint exists.
            resp = requests.post(
                f"{vllm_url}{endpoint}",
                headers=headers,
                timeout=5,
                params={"mode": "keep"} if endpoint == "/pause" else None,
            )
            exists = resp.status_code != 404
            ok = (
                _check(
                    f"POST {endpoint}",
                    exists,
                    f"status={resp.status_code}" if exists else "404 Not Found",
                )
                and ok
            )
        except Exception as exc:
            ok = _check(f"POST {endpoint}", False, str(exc)) and ok

    # Check vLLM >= 0.20 lifecycle endpoints
    for endpoint in ["/start_weight_update", "/finish_weight_update"]:
        try:
            resp = requests.post(
                f"{vllm_url}{endpoint}",
                headers=headers,
                json={} if endpoint == "/start_weight_update" else None,
                timeout=5,
            )
            exists = resp.status_code != 404
            ok = (
                _check(
                    f"POST {endpoint} (vLLM >=0.20)",
                    exists,
                    f"status={resp.status_code}" if exists else "404 — pre-0.20 vLLM?",
                )
                and ok
            )
        except Exception as exc:
            ok = _check(f"POST {endpoint}", False, str(exc)) and ok

    return ok


# ── Main ───────────────────────────────────────────────────────────────


def main() -> int:
    print("=" * 60)
    print("Gate 0.3 — TRL + vLLM + Transformers Version Compatibility")
    print("=" * 60)

    # Print installed versions
    print("\nInstalled versions:")
    for pkg in ["trl", "vllm", "transformers", "torch", "accelerate"]:
        print(f"  {pkg:20s} {_version(pkg)}")

    # Run checks
    results: list[bool] = []
    results.append(check_imports())
    results.append(check_vllm_modules())
    results.append(check_tokenizer())
    results.append(check_vllm_completions())
    results.append(check_weight_transfer_endpoints())

    # Summary
    passed = sum(results)
    total = len(results)
    print(f"\n{'=' * 60}")
    print(f"Gate 0.3 result: {passed}/{total} check groups passed")

    if all(results):
        print("✅ Gate 0.3 PASSED — version combination is compatible")
        return 0
    else:
        # Checks 4 and 5 (server-dependent) are soft — they require a running vLLM
        core_ok = all(results[:3])
        if core_ok:
            print(
                "⚠️  Gate 0.3 PARTIAL — core imports OK, "
                "server-dependent checks skipped/failed"
            )
            print(
                "    Start vLLM with VLLM_SERVER_DEV_MODE=1 and re-run for full validation."
            )
            return 0
        else:
            print("❌ Gate 0.3 FAILED — fix import/version errors above")
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
