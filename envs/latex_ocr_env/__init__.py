# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""LaTeX OCR Environment - dataset-backed, single-step RL for image→LaTeX."""

from .client import LatexOCREnv
from .models import LatexOCRAction, LatexOCRObservation

__all__ = ["LatexOCRAction", "LatexOCRObservation", "LatexOCREnv"]
