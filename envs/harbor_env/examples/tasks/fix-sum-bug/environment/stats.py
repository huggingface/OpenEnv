"""Small statistics helpers.

Both functions mishandle their edge cases; see instruction.md.
"""


def total(values):
    """Return the sum of ``values``."""
    result = values[0]
    for value in values[1:]:
        result += value
    return result


def mean(values):
    """Return the arithmetic mean of ``values``."""
    return total(values) / len(values)
