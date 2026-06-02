#!/usr/bin/env python3
"""Optimize a GPU kernel with a hosted LLM via Hugging Face Inference.

This script demonstrates using an OpenAI-compatible model to iteratively
optimize CUDA/Triton kernels in the kernrl environment. The model receives
a PyTorch reference implementation and feedback from compilation/execution,
then produces optimized kernel code.

Prerequisites
-------------
1. Build the kernrl Docker image (requires NVIDIA GPU)::

       docker build \
           -f envs/kernrl/server/Dockerfile \
           -t kernrl:latest .

2. Set your Hugging Face token or OpenAI API key::

       export HF_TOKEN=your_token_here
       # or
       export OPENAI_API_KEY=your_key_here

3. Run the script::

       python examples/kernrl_inference.py

The script keeps sending kernel code to the environment until it achieves
a speedup > 1.0x or reaches the step limit.
"""

from __future__ import annotations

import os
import re
from typing import List, Tuple

from openai import OpenAI

from kernrl import KernelAction, kernrl_env


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Use HuggingFace router or OpenAI directly
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")

MODEL = os.getenv("MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct")
MAX_STEPS = 5
VERBOSE = True

# Which problem to solve
PROBLEM_ID = os.getenv("PROBLEM_ID", "L1_23_Softmax")

SYSTEM_PROMPT = """You are an expert GPU kernel engineer specializing in CUDA and Triton.

Your task is to optimize PyTorch operations by writing custom GPU kernels.

Guidelines:
1. Analyze the reference PyTorch implementation carefully
2. Identify optimization opportunities (memory access patterns, parallelism, fusion)
3. Write a Triton or CUDA kernel that computes the same result
4. Ensure numerical correctness (outputs must match within tolerance)

Output format:
- Provide a complete Python file
- Include a Model class with the same interface as the reference
- The Model.forward() method should use your optimized kernel
- Include all necessary imports (torch, triton, triton.language)

Focus on:
- Coalesced memory access
- Efficient use of shared memory
- Minimizing thread divergence
- Optimal block/grid dimensions

Always wrap your code in a ```python code block."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def extract_python_code(text: str) -> str:
    """Extract the first Python code block from the model output."""
    code_blocks = re.findall(
        r"```(?:python)?\s*(.*?)```",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if code_blocks:
        return code_blocks[0].strip()
    return text.strip()


def format_feedback(
    step: int,
    compilation_success: bool,
    compilation_error: str | None,
    correctness_pass: bool | None,
    max_diff: float | None,
    speedup: float | None,
) -> str:
    """Generate feedback text describing the kernel evaluation."""
    parts = [f"Evaluation feedback for step {step}:"]

    if compilation_success:
        parts.append("Compilation: SUCCESS")
    else:
        parts.append(f"Compilation: FAILED")
        if compilation_error:
            parts.append(f"Error: {compilation_error[:500]}")
        parts.append("\nFix the compilation error and try again.")
        return "\n".join(parts)

    if correctness_pass is True:
        parts.append("Correctness: PASS")
    elif correctness_pass is False:
        parts.append(f"Correctness: FAIL (max_diff={max_diff})")
        parts.append("The output doesn't match the reference. Check your algorithm.")
        return "\n".join(parts)
    else:
        parts.append("Correctness: Not tested")

    if speedup is not None:
        parts.append(f"Speedup: {speedup:.2f}x")
        if speedup > 1.0:
            parts.append("Great! Your kernel is faster than baseline.")
        else:
            parts.append("Your kernel is slower than baseline. Try optimizing memory access or parallelism.")

    parts.append("\nProvide an improved kernel implementation.")
    return "\n".join(parts)


def build_initial_prompt(problem_description: str, reference_code: str) -> str:
    """Construct the first user prompt for the kernel task."""
    return (
        "Optimize the following PyTorch operation by writing a custom GPU kernel.\n\n"
        f"## Problem Description\n{problem_description}\n\n"
        f"## Reference Implementation\n```python\n{reference_code}\n```\n\n"
        "Write a Triton kernel that computes the same result but faster.\n"
        "Reply with the full implementation in a single ```python code block."
    )


# ---------------------------------------------------------------------------
# Optimization Loop
# ---------------------------------------------------------------------------


def optimize_kernel(
    env: kernrl_env,
    client: OpenAI,
) -> Tuple[bool, float, List[str]]:
    """Iteratively ask the model for kernel code until success."""

    # Reset environment
    result = env.reset(problem_id=PROBLEM_ID)
    obs = result.observation

    history = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_initial_prompt(
            obs.problem_description,
            obs.reference_code,
        )},
    ]

    transcripts: List[str] = []
    best_speedup = 0.0

    for step in range(1, MAX_STEPS + 1):
        if VERBOSE:
            print(f"\n--- Step {step}/{MAX_STEPS} ---")

        response = client.chat.completions.create(
            model=MODEL,
            messages=history,
            max_tokens=4096,
            temperature=0.3,
        )

        assistant_message = response.choices[0].message.content.strip()
        history.append({"role": "assistant", "content": assistant_message})

        code = extract_python_code(assistant_message)

        if VERBOSE:
            print(f"Generated {len(code)} chars of code")
            print(code[:500] + "..." if len(code) > 500 else code)

        # Submit to environment
        result = env.step(KernelAction(code=code))
        obs = result.observation

        transcript = (
            f"Step {step}\n"
            f"  Compilation: {'OK' if obs.compilation_success else 'FAIL'}\n"
            f"  Correctness: {obs.correctness_pass}\n"
            f"  Speedup: {obs.speedup}x\n"
        )
        transcripts.append(transcript)

        if VERBOSE:
            print(f"  Compilation: {'OK' if obs.compilation_success else 'FAIL'}")
            if obs.compilation_error:
                print(f"  Error: {obs.compilation_error[:200]}...")
            print(f"  Correctness: {obs.correctness_pass}")
            print(f"  Speedup: {obs.speedup}x" if obs.speedup else "  Speedup: N/A")

        # Track best speedup
        if obs.speedup and obs.speedup > best_speedup:
            best_speedup = obs.speedup

        # Success condition: correct AND faster than baseline
        if obs.correctness_pass and obs.speedup and obs.speedup > 1.0:
            return True, best_speedup, transcripts

        # Add feedback for next iteration
        feedback = format_feedback(
            step,
            obs.compilation_success,
            obs.compilation_error,
            obs.correctness_pass,
            obs.max_diff,
            obs.speedup,
        )
        history.append({"role": "user", "content": feedback})

        # Keep conversation history compact
        if len(history) > 16:
            history = [history[0]] + history[-15:]

    return False, best_speedup, transcripts


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    if not API_KEY:
        raise SystemExit(
            "Set HF_TOKEN, OPENAI_API_KEY, or API_KEY to query the model."
        )

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    print(f"Connecting to kernrl environment...")
    print(f"Problem: {PROBLEM_ID}")
    print(f"Model: {MODEL}")

    # Connect to local Docker container
    env = kernrl_env.from_docker_image(
        "kernrl:latest",
        ports={8000: 8000},
    )

    try:
        success, best_speedup, transcripts = optimize_kernel(env, client)
    finally:
        env.close()

    print("\n" + "=" * 50)
    if success:
        print(f"SUCCESS! Achieved {best_speedup:.2f}x speedup")
    else:
        print(f"Did not achieve speedup > 1.0x (best: {best_speedup:.2f}x)")

    print("\n--- Execution transcripts ---")
    for entry in transcripts:
        print(entry)


if __name__ == "__main__":
    main()
