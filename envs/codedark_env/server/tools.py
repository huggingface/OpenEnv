"""
CodeDark Tool Implementations

Tools available to agents:
- run_python: Execute Python/pandas code in sandboxed environment
- read_notes: Read all saved notes from current episode
- save_note: Save a note for later recall
- clarify: Ask clarifying question (max 2 per episode)
- submit_answer: Submit final answer (ends episode)
"""

import re
import ast
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np


# Safe builtins for sandboxed code execution
SAFE_BUILTINS = {
    "len": len,
    "sum": sum,
    "min": min,
    "max": max,
    "abs": abs,
    "round": round,
    "sorted": sorted,
    "range": range,
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "enumerate": enumerate,
    "zip": zip,
    "True": True,
    "False": False,
    "None": None,
    "print": print,
    "type": type,
    "isinstance": isinstance,
    "map": map,
    "filter": filter,
    "any": any,
    "all": all,
    "hasattr": hasattr,
    "getattr": getattr,
    "repr": repr,
    "reversed": reversed,
    "slice": slice,
    "format": format,
    "Exception": Exception,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "AttributeError": AttributeError,
}


def run_python(
    code: str, df: pd.DataFrame, max_output_chars: int = 200
) -> Tuple[str, str, int]:
    """Execute Python code in sandboxed environment.

    Args:
        code: Python code to execute
        df: DataFrame available as 'df' in execution context
        max_output_chars: Maximum characters for output truncation

    Returns:
        Tuple of (stdout, stderr, exit_code)
    """
    if df is None:
        return "", "Error: No dataframe loaded", 1

    local_vars = {
        "pd": pd,
        "np": np,
        "df": df.copy(),
    }

    try:
        exec(code, {"__builtins__": SAFE_BUILTINS}, local_vars)
        result = local_vars.get("result")

        if result is None:
            return (
                "",
                "Error: No 'result' variable set. Store your result in 'result'.",
                1,
            )

        # Format output with truncation
        if isinstance(result, pd.DataFrame):
            preview = result.head(3).to_string()
        elif isinstance(result, pd.Series):
            preview = result.head(5).to_string()
        else:
            preview = str(result)

        # Truncate if needed
        if len(preview) > max_output_chars:
            preview = preview[:max_output_chars] + "..."

        return f"run_python Result:\n{preview}", "", 0

    except Exception as e:
        return "", f"run_python Error: {e}", 1


def read_notes(notes: List[str]) -> Tuple[str, str, int]:
    """Read all saved notes.

    Args:
        notes: List of saved notes

    Returns:
        Tuple of (stdout, stderr, exit_code)
    """
    if not notes:
        return "No notes saved yet.", "", 0

    notes_list = "\n".join(f"- {n}" for n in notes)
    return f"Saved notes:\n{notes_list}", "", 0


def save_note(content: str, notes: List[str]) -> Tuple[str, str, int]:
    """Save a note to persistent memory.

    Args:
        content: Note content to save
        notes: List to append note to (modified in place)

    Returns:
        Tuple of (stdout, stderr, exit_code)
    """
    content = content.strip()
    if not content:
        return "", "Error: Empty note content", 1

    notes.append(content)
    notes_list = "\n".join(f"- {n}" for n in notes)
    return f"Note saved.\n\nAll notes:\n{notes_list}", "", 0


def clarify(
    question: str,
    clarify_count: int,
    max_clarifications: int,
    ambiguities: Optional[List[str]] = None,
    answer_type: str = "scalar",
) -> Tuple[str, str, int, int]:
    """Ask a clarifying question about the task.

    Args:
        question: The clarifying question
        clarify_count: Current number of clarifications used
        max_clarifications: Maximum allowed clarifications
        ambiguities: List of known ambiguities from task metadata
        answer_type: Expected answer type ("scalar", "list", etc.)

    Returns:
        Tuple of (stdout, stderr, exit_code, new_clarify_count)
    """
    if clarify_count >= max_clarifications:
        return (
            "",
            f"Error: Maximum {max_clarifications} clarifications per episode. Please proceed with your best interpretation.",
            1,
            clarify_count,
        )

    question_lower = question.lower()
    ambiguities = ambiguities or []

    # Build clarification responses from task metadata
    clarifications = {}

    for amb in ambiguities:
        amb_lower = amb.lower()
        if (
            "percentile" in amb_lower
            or "inclusive" in amb_lower
            or "exclusive" in amb_lower
        ):
            clarifications["percentile"] = (
                "Use >= for 'top X%' (inclusive of threshold) and <= for 'bottom X%'."
            )
        if "rate" in amb_lower or "percentage" in amb_lower:
            clarifications["rate"] = (
                "Express rates as percentages 0-100, rounded to 2 decimal places."
            )
        if (
            "positive" in amb_lower
            or "success" in amb_lower
            or "target" in amb_lower
            or "y=1" in amb_lower
        ):
            clarifications["target"] = "Subscription/success means y=1 in the dataset."
        if "boundary" in amb_lower:
            clarifications["boundary"] = (
                "Include boundary values (>=, <=) when filtering."
            )

    # Add format clarifications from answer type
    if answer_type == "scalar":
        clarifications["format"] = (
            "Return a single numeric value, rounded to 2 decimal places."
        )
    elif answer_type == "list":
        clarifications["format"] = (
            "Return as a list/DataFrame with the specified columns."
        )

    # Try to match question to a clarification
    response = None
    for key, value in clarifications.items():
        if key in question_lower or any(word in question_lower for word in key.split()):
            response = value
            break

    if response:
        return f"Clarification: {response}", "", 0, clarify_count + 1
    else:
        return (
            "Clarification: Please proceed with your best interpretation based on standard data analysis conventions.",
            "",
            0,
            clarify_count + 1,
        )


def submit_answer(answer_str: str) -> Tuple[str, str, int, Any]:
    """Submit final answer.

    Args:
        answer_str: Answer string to parse and submit

    Returns:
        Tuple of (stdout, stderr, exit_code, parsed_answer)
    """
    answer_str = answer_str.strip().rstrip("%").strip()

    if not answer_str:
        return "", "Error: Empty answer", 1, None

    # Try to parse as structured data first (list/dict)
    try:
        answer = ast.literal_eval(answer_str)
    except (ValueError, SyntaxError):
        # Fall back to numeric parsing
        try:
            answer = float(answer_str)
        except ValueError:
            answer = answer_str

    return "[SUBMITTED]", "", 0, answer


def parse_tool_call(args: str, tool_name: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse tool-specific arguments from args string.

    Args:
        args: Raw args string
        tool_name: Name of the tool being called

    Returns:
        Tuple of (parsed_content, error_message)
    """
    if tool_name == "run_python":
        # Extract code from <code></code> tags
        match = re.search(r"<code>(.*?)</code>", args, re.DOTALL)
        if not match:
            return None, "No <code> tag found. Use: <code>your_code</code>"
        return match.group(1).strip(), None

    elif tool_name == "clarify":
        # Extract question from <question></question> tags
        match = re.search(r"<question>(.*?)</question>", args, re.DOTALL)
        if not match:
            return (
                None,
                "No <question> tag found. Use: <question>your question</question>",
            )
        return match.group(1).strip(), None

    elif tool_name == "submit_answer":
        # Extract answer from <answer></answer> tags
        match = re.search(r"<answer>(.*?)</answer>", args, re.DOTALL)
        if not match:
            return None, "No <answer> tag found. Use: <answer>value</answer>"
        return match.group(1).strip(), None

    elif tool_name in ("read_notes", "save_note"):
        # These take raw args
        return args.strip(), None

    else:
        return None, f"Unknown tool: {tool_name}"
