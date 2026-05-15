#!/usr/bin/env python3
"""Build a deterministic 20-task SWE mini subset.

Examples
--------
Build from the Hugging Face datasets-server API (default):

    uv run python scripts/swe/build_mini_subset.py

Build from a local JSONL dump:

    uv run python scripts/swe/build_mini_subset.py \
      --input-jsonl /path/to/swebench_lite_rows.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

REPO_ROOT = Path(__file__).resolve().parents[2]
ENVS_DIR = REPO_ROOT / "envs"
if str(ENVS_DIR) not in sys.path:
    sys.path.insert(0, str(ENVS_DIR))

from mini_swe_env.task_loader_swebench_lite import (  # noqa: E402
    adapt_swebench_lite_rows,
    deterministic_train_eval_split,
    load_task_file,
    read_jsonl_rows,
    write_tasks_jsonl,
)

DEFAULT_DATASET = "princeton-nlp/SWE-bench_Lite"
DEFAULT_SPLIT = "test"
DEFAULT_TRAIN_OUTPUT = REPO_ROOT / "examples/mini_swe_env/tasks/mini_swe_train.jsonl"
DEFAULT_EVAL_OUTPUT = REPO_ROOT / "examples/mini_swe_env/tasks/mini_swe_eval.jsonl"
DATASET_SERVER_ROWS_URL = "https://datasets-server.huggingface.co/rows"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=None,
        help="Optional local SWE-bench Lite JSONL rows. If omitted, fetches from datasets-server.",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"HF dataset id when fetching remotely (default: {DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help=f"Dataset split when fetching remotely (default: {DEFAULT_SPLIT}).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap for remotely fetched rows (useful for smoke runs).",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--subset-size", type=int, default=20)
    parser.add_argument("--train-size", type=int, default=16)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--train-output",
        type=Path,
        default=DEFAULT_TRAIN_OUTPUT,
    )
    parser.add_argument(
        "--eval-output",
        type=Path,
        default=DEFAULT_EVAL_OUTPUT,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.input_jsonl:
        rows = read_jsonl_rows(args.input_jsonl)
        source_desc = str(args.input_jsonl)
    else:
        rows = fetch_rows_from_dataset_server(
            dataset=args.dataset,
            split=args.split,
            max_rows=args.max_rows,
        )
        source_desc = f"hf://{args.dataset}[{args.split}]"

    tasks, skipped = adapt_swebench_lite_rows(rows, strict=args.strict)
    train_tasks, eval_tasks = deterministic_train_eval_split(
        tasks,
        subset_size=args.subset_size,
        train_size=args.train_size,
        seed=args.seed,
    )

    write_tasks_jsonl(args.train_output, train_tasks)
    write_tasks_jsonl(args.eval_output, eval_tasks)

    # Final safety check: output must parse as valid SWETask rows.
    _ = load_task_file(args.train_output)
    _ = load_task_file(args.eval_output)

    print(
        "Built mini SWE subset:",
        json.dumps(
            {
                "source": source_desc,
                "seed": args.seed,
                "rows_seen": len(rows),
                "valid_tasks": len(tasks),
                "skipped_rows": len(skipped),
                "train_tasks": len(train_tasks),
                "eval_tasks": len(eval_tasks),
                "train_output": str(args.train_output),
                "eval_output": str(args.eval_output),
            },
            indent=2,
            sort_keys=True,
        ),
    )

    if skipped:
        print("Sample skipped rows (up to 5):")
        for skip in skipped[:5]:
            print(
                f"  line={skip.row_index} instance_id={skip.instance_id or '-'} "
                f"reason={skip.reason}"
            )


def fetch_rows_from_dataset_server(
    *, dataset: str, split: str, max_rows: int | None
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    page_size = 100

    while True:
        if max_rows is not None:
            remaining = max_rows - len(rows)
            if remaining <= 0:
                break
            page_length = min(page_size, remaining)
        else:
            page_length = page_size

        payload = _fetch_rows_page(
            dataset=dataset,
            split=split,
            offset=offset,
            length=page_length,
        )
        page_rows = payload.get("rows", [])
        if not page_rows:
            break

        for item in page_rows:
            row = item.get("row") if isinstance(item, dict) else None
            if isinstance(row, dict):
                rows.append(row)

        offset += len(page_rows)

        total = payload.get("num_rows_total")
        if isinstance(total, int) and offset >= total:
            break

        if len(page_rows) < page_length:
            break

    if not rows:
        raise RuntimeError(f"No rows fetched for dataset={dataset!r} split={split!r}.")
    return rows


def _fetch_rows_page(
    *, dataset: str, split: str, offset: int, length: int
) -> dict[str, Any]:
    query = urlencode(
        {
            "dataset": dataset,
            "config": "default",
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    url = f"{DATASET_SERVER_ROWS_URL}?{query}"
    try:
        with urlopen(url, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(
            f"datasets-server request failed ({exc.code}) for {url}: {exc.reason}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(
            f"datasets-server request failed for {url}: {exc.reason}"
        ) from exc

    if not isinstance(data, dict):
        raise RuntimeError(
            f"datasets-server response must be a JSON object, got {type(data).__name__}"
        )
    return data


if __name__ == "__main__":
    main()
