"""Reference solution for the `fix-sum-bug` example task."""


def total(values):
    """Return the sum of ``values``, or ``0`` when there are none."""
    result = 0
    for value in values:
        result += value
    return result


def mean(values):
    """Return the arithmetic mean of ``values``, or ``0.0`` when there are none."""
    if not values:
        return 0.0
    return total(values) / len(values)
