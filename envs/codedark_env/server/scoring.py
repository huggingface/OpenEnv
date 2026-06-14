"""
CodeDark Scoring System

Reward computation with multi-metric scoring:
- 80% correctness (binary: exact match within tolerance)
- 10% efficiency (fewer turns = better)
- 10% token cost (lower usage = better)
"""

from typing import Any, Optional, Tuple
import ast


def normalize_value(val: Any) -> Any:
    """Normalize a value for comparison.

    Handles:
    - String to float/int conversion
    - String to list/dict parsing
    - Float rounding for precision
    - Percentage stripping
    """
    if val is None:
        return None

    # Already a proper type - just normalize floats
    if isinstance(val, float):
        return round(val, 4)
    if isinstance(val, int):
        return float(val)
    if isinstance(val, (list, dict)):
        return val

    # String handling
    if isinstance(val, str):
        val = val.strip()

        # Try to parse as list/dict first
        if val.startswith("[") or val.startswith("{"):
            try:
                return ast.literal_eval(val)
            except (ValueError, SyntaxError):
                pass

        # Try float (strip % if present)
        try:
            return round(float(val.rstrip("%")), 4)
        except ValueError:
            pass

        # Return as lowercase string
        return val.lower()

    return val


def parse_markdown_table(text: str) -> Optional[list]:
    """Parse markdown table to list of dicts.

    Handles tables like:
    | job | mean | std |
    |-----|------|-----|
    | retired | 40.97 | 9.74 |
    """
    if not isinstance(text, str):
        return None

    lines = text.strip().split("\n")

    # Find table lines (contain |)
    table_lines = [line for line in lines if "|" in line]
    if len(table_lines) < 3:  # Need header + separator + at least 1 row
        return None

    # Parse header
    header_line = table_lines[0]
    headers = [h.strip().lower() for h in header_line.split("|") if h.strip()]
    if not headers:
        return None

    # Skip separator line (contains ---)
    data_start = 1
    if "---" in table_lines[1] or "--|" in table_lines[1]:
        data_start = 2

    # Parse data rows
    rows = []
    for line in table_lines[data_start:]:
        cells = [c.strip() for c in line.split("|") if c.strip()]
        if len(cells) != len(headers):
            continue

        row = {}
        for h, c in zip(headers, cells):
            # Clean number formatting (commas, currency symbols)
            c_clean = c.replace(",", "").replace("€", "").replace("$", "").strip()
            try:
                if "." in c_clean:
                    row[h] = round(float(c_clean), 4)
                else:
                    row[h] = int(c_clean)
            except ValueError:
                row[h] = c.lower()
        rows.append(row)

    return rows if rows else None


def compare_answers(submitted: Any, expected: Any, tolerance: float = 0.01) -> bool:
    """Compare answers with support for structured data and numeric tolerance.

    Handles:
    - Type mismatches (string "33.53" vs float 33.53)
    - Floating point precision (rounds to 4 decimals)
    - Nested structures (lists, dicts)
    - String parsing for lists/dicts
    """
    # Normalize both values
    submitted_n = normalize_value(submitted)
    expected_n = normalize_value(expected)

    # Null checks
    if submitted_n is None and expected_n is None:
        return True
    if submitted_n is None or expected_n is None:
        return False

    # Same type comparison after normalization
    if type(submitted_n) == type(expected_n):
        if isinstance(expected_n, list):
            if len(submitted_n) != len(expected_n):
                return False
            # Check if list contains dicts (structured data) - use order-sensitive
            if expected_n and isinstance(expected_n[0], dict):
                return all(
                    compare_answers(s, e, tolerance)
                    for s, e in zip(submitted_n, expected_n)
                )
            # Simple values list - order-insensitive (we don't tell models to sort)
            submitted_sorted = sorted([str(x).lower().strip() for x in submitted_n])
            expected_sorted = sorted([str(x).lower().strip() for x in expected_n])
            return submitted_sorted == expected_sorted

        if isinstance(expected_n, dict):
            if set(submitted_n.keys()) != set(expected_n.keys()):
                return False
            return all(
                compare_answers(submitted_n[k], expected_n[k], tolerance)
                for k in expected_n
            )

        if isinstance(expected_n, float):
            return abs(submitted_n - expected_n) <= tolerance

        # String comparison
        return str(submitted_n) == str(expected_n)

    # Type mismatch after normalization - try numeric comparison
    try:
        sub_f = (
            float(submitted_n) if not isinstance(submitted_n, (list, dict)) else None
        )
        exp_f = float(expected_n) if not isinstance(expected_n, (list, dict)) else None
        if sub_f is not None and exp_f is not None:
            return abs(sub_f - exp_f) <= tolerance
    except (ValueError, TypeError):
        pass

    # Try markdown table parsing if expected is list and submitted is string
    if isinstance(expected_n, list) and isinstance(submitted_n, str):
        parsed = parse_markdown_table(submitted)  # Use original, not normalized
        if parsed is not None:
            return compare_answers(parsed, expected, tolerance)

    # Fallback: string comparison
    return str(submitted_n).lower() == str(expected_n).lower()


def score_correctness(submitted: Any, expected: Any, tolerance: float = 0.01) -> float:
    """Score the submitted answer correctness. Weight: 0.80

    Scoring:
    - 0.80: Exact match
    - 0.20: Almost there (rounding or 100x scale error)
    - 0.00: Wrong

    Args:
        submitted: Submitted answer
        expected: Expected answer
        tolerance: Numeric tolerance for comparison

    Returns:
        Correctness score (0.0, 0.20, or 0.80)
    """
    if submitted is None:
        return 0.0

    try:
        # Try numeric comparison first
        submitted_f = float(submitted)
        expected_f = float(expected)

        # Exact match (within tolerance)
        if abs(submitted_f - expected_f) < tolerance:
            return 0.80

        if expected_f != 0:
            ratio = submitted_f / expected_f

            # 100x scale error (decimal vs percentage)
            # e.g., 0.0959 vs 9.59 or 9.59 vs 0.0959
            if 0.009 < ratio < 0.011 or 99 < ratio < 101:
                return 0.20

            # Rounding error (within 1% of expected)
            if 0.99 < ratio < 1.01:
                return 0.20

    except (ValueError, TypeError):
        # Structured data comparison (lists, dicts)
        if compare_answers(submitted, expected, tolerance=tolerance):
            return 0.80

    return 0.0


def score_efficiency(turns: int, max_turns: int, is_correct: bool) -> float:
    """Score based on turns used (fewer = better). Weight: 0.10

    Only applies if answer is correct.

    Args:
        turns: Number of turns used
        max_turns: Maximum turns allowed
        is_correct: Whether the answer was correct

    Returns:
        Efficiency score (0.0 to 0.10)
    """
    if not is_correct:
        return 0.0

    # Scale: 1 turn = 0.10, max_turns = 0.01
    efficiency = max(0.0, 1.0 - (turns / max_turns))
    return 0.10 * efficiency


def score_token_cost(
    input_tokens: int,
    output_tokens: int,
    is_correct: bool,
    input_price: float = 1.0,
    output_price: float = 5.0,
    target_cost: float = 0.01,
    max_cost: float = 0.10,
) -> Tuple[float, float]:
    """Score based on token cost (lower = better). Weight: 0.10

    Only applies if answer is correct.

    Args:
        input_tokens: Number of input tokens used
        output_tokens: Number of output tokens used
        is_correct: Whether the answer was correct
        input_price: Price per 1M input tokens (default $1)
        output_price: Price per 1M output tokens (default $5)
        target_cost: Cost for full score (default $0.01)
        max_cost: Cost for zero score (default $0.10)

    Returns:
        Tuple of (token_score, cost_usd)
    """
    if not is_correct:
        return 0.0, 0.0

    # Calculate cost in dollars
    cost = (input_tokens * input_price / 1_000_000) + (
        output_tokens * output_price / 1_000_000
    )

    # Scale: <$0.01 = full score, >$0.10 = 0
    if cost <= target_cost:
        efficiency = 1.0
    elif cost >= max_cost:
        efficiency = 0.0
    else:
        efficiency = 1.0 - ((cost - target_cost) / (max_cost - target_cost))

    return 0.10 * efficiency, cost


def compute_reward(
    submitted: Any,
    expected: Any,
    tolerance: float,
    turns: int,
    max_turns: int,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> Tuple[float, float, float, float]:
    """Compute total reward from all components.

    Args:
        submitted: Submitted answer
        expected: Expected answer
        tolerance: Numeric tolerance
        turns: Number of turns used
        max_turns: Maximum turns allowed
        input_tokens: Number of input tokens (optional)
        output_tokens: Number of output tokens (optional)

    Returns:
        Tuple of (total_reward, correctness, efficiency, token_cost_usd)
    """
    # Correctness (0.80 weight)
    correctness = score_correctness(submitted, expected, tolerance)
    is_correct = correctness > 0

    # Efficiency (0.10 weight)
    efficiency = score_efficiency(turns, max_turns, is_correct)

    # Token cost (0.10 weight)
    # If tokens not tracked, estimate from turns
    if input_tokens == 0 and output_tokens == 0:
        input_tokens = turns * 1000
        output_tokens = turns * 500

    token_score, cost_usd = score_token_cost(input_tokens, output_tokens, is_correct)

    total_reward = correctness + efficiency + token_score

    return total_reward, correctness, efficiency, cost_usd
