# SPDX-License-Identifier: BSD-3-Clause
"""Failure to establish or completely tear down a sandbox boundary."""


class IsolationError(OSError):
    """The requested sandbox operation could not be verified safely."""
