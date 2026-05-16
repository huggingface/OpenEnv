# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations


def shell_quote(s: str) -> str:
    """Single-quote a string for shell, escaping embedded single quotes."""
    return "'" + s.replace("'", "'\\''") + "'"
