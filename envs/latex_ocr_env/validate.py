# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
End-to-end driver for the LaTeX OCR environment.

Connects to a running server, exercises the Task API, resets to a task, runs a
policy over the image, steps, and prints the reward. The policy is either a
real vision-LLM served through the Hugging Face Inference Router
(OpenAI-compatible) or, if no HF_TOKEN is set, a no-op placeholder so the
plumbing can still be verified.

Usage:
    # Start the server first (see README), then:
    python validate.py --base-url http://localhost:8000 \\
        --split test --num 3 --model Qwen/Qwen2.5-VL-7B-Instruct
"""

from __future__ import annotations

import argparse
import os
import sys

# Make `latex_ocr_env` importable when run from the env directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from latex_ocr_env import LatexOCRAction, LatexOCREnv  # noqa: E402


def vlm_transcribe(image_base64: str, prompt: str, model: str) -> str:
    """Run a VLM over the image via the Hugging Face Inference Router."""
    from openai import OpenAI

    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=os.environ["HF_TOKEN"],
    )
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                    },
                ],
            }
        ],
    )
    text = completion.choices[0].message.content or ""
    # Strip common code-fence wrapping so scoring sees raw LaTeX.
    text = text.strip()
    for fence in ("```latex", "```LaTeX", "```"):
        if text.startswith(fence):
            text = text[len(fence) :]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--split", default="test")
    parser.add_argument("--num", type=int, default=3, help="Tasks to run")
    parser.add_argument(
        "--model",
        default=os.environ.get("LATEX_OCR_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct"),
        help="VLM served via the HF router (append :provider to pin, e.g. ':nebius').",
    )
    args = parser.parse_args()

    use_vlm = bool(os.environ.get("HF_TOKEN"))
    if not use_vlm:
        print("HF_TOKEN not set -> running plumbing-only policy (no VLM).\n")

    with LatexOCREnv(base_url=args.base_url).sync() as env:
        # --- Task API ---
        splits = env.list_splits()
        n = env.num_tasks(args.split)
        print(f"splits = {splits}")
        print(f"num_tasks({args.split}) = {n}")
        print(f"get_task({args.split}, 0) = {env.get_task(args.split, 0)}\n")

        rewards = []
        seen_targets = []
        # n <= 0 means unknown count (stream metadata); fall back to --num.
        count = args.num if n <= 0 else min(args.num, n)
        stream_mode = False
        for i in range(count):
            if stream_mode:
                result = env.reset(split=args.split)
            else:
                try:
                    result = env.reset(split=args.split, index=i)
                except Exception:
                    # stream-mode server rejects random index -> pull sequentially
                    stream_mode = True
                    result = env.reset(split=args.split)
            obs = result.observation
            prog = ""
            if obs.total and obs.total > 0:
                prog = (
                    f" | progress: {obs.index}/{obs.total} "
                    f"({obs.pct_done:.4%}), remaining={obs.remaining}"
                )
            print(
                f"[task {obs.task_id}] image bytes(b64)={len(obs.image_base64)}{prog}"
            )

            if use_vlm:
                prediction = vlm_transcribe(obs.image_base64, obs.prompt, args.model)
            else:
                prediction = ""  # plumbing check only

            result = env.step(LatexOCRAction(latex=prediction))
            o = result.observation
            rewards.append(result.reward)
            seen_targets.append(o.target_latex)
            print(f"  predicted : {o.predicted_latex[:80]!r}")
            print(f"  target    : {o.target_latex[:80]!r}")
            print(
                f"  reward={result.reward:.4f} exact={o.exact_match} "
                f"cer={o.char_error_rate:.4f}\n"
            )

        if rewards:
            print(
                f"\nmean reward over {len(rewards)} tasks = {sum(rewards) / len(rewards):.4f}"
            )
            uniq = len(set(seen_targets))
            print(f"no-repeat check: {uniq}/{len(seen_targets)} distinct targets")


if __name__ == "__main__":
    main()
