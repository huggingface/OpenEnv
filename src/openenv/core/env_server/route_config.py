# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Route configuration utilities for declarative FastAPI route registration.

This module provides utilities to reduce boilerplate in route registration
by using configuration objects instead of repeated function calls.
"""

from dataclasses import dataclass
from typing import Callable, List, Type

from fastapi import FastAPI
from pydantic import BaseModel


@dataclass
class GetEndpointConfig:
    """Configuration for a simple GET endpoint."""

    path: str
    handler: Callable[[], BaseModel | dict]
    response_model: Type[BaseModel] | type[dict]
    tag: str
    summary: str
    description: str


def register_get_endpoints(app: FastAPI, configs: List[GetEndpointConfig]) -> None:
    """
    Register multiple GET endpoints from configuration.

    Args:
        app: FastAPI application instance
        configs: List of GET endpoint configurations
    """
    for config in configs:
        # Capture handler in a closure to avoid non-serializable default parameter
        def make_endpoint(
            handler: Callable[[], BaseModel | dict],
        ) -> Callable[[], BaseModel | dict]:
            async def endpoint() -> BaseModel | dict:
                return handler()

            return endpoint

        app.get(
            config.path,
            response_model=config.response_model,
            tags=[config.tag],
            summary=config.summary,
            description=config.description,
        )(make_endpoint(config.handler))
