# Renamed from types.py to avoid shadowing Python stdlib 'types' module
# Copyright (c) Meta Platforms, Inc. and affiliates.

from enum import Enum
from typing import Annotated, Any, Dict, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

Scalar = Union[int, float, bool]


class ServerMode(str, Enum):
    SIMULATION = "simulation"
    PRODUCTION = "production"


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"


class WSErrorCode(str, Enum):
    INVALID_JSON = "INVALID_JSON"
    UNKNOWN_TYPE = "UNKNOWN_TYPE"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    EXECUTION_ERROR = "EXECUTION_ERROR"
    CAPACITY_REACHED = "CAPACITY_REACHED"
    FACTORY_ERROR = "FACTORY_ERROR"
    SESSION_ERROR = "SESSION_ERROR"


class Action(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Observation(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )
    done: bool = Field(default=False)
    reward: bool | int | float | None = Field(default=None)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class State(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )
    episode_id: Optional[str] = Field(default=None)
    step_count: int = Field(default=0, ge=0)


class ResetRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    seed: Optional[int] = Field(default=None, ge=0)
    episode_id: Optional[str] = Field(default=None, max_length=255)


class ResetResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    observation: Dict[str, Any] = Field(...)
    reward: Optional[float] = Field(default=None)
    done: bool = Field(default=False)


class StepRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    action: Dict[str, Any] = Field(...)
    timeout_s: Optional[float] = Field(default=None, gt=0)
    request_id: Optional[str] = Field(default=None, max_length=255)


class StepResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    observation: Dict[str, Any] = Field(...)
    reward: Optional[float] = Field(default=None)
    done: bool = Field(default=False)


class BaseMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class HealthResponse(BaseMessage):
    status: HealthStatus = Field(default=HealthStatus.HEALTHY)


class SchemaResponse(BaseMessage):
    action: Dict[str, Any] = Field(...)
    observation: Dict[str, Any] = Field(...)
    state: Dict[str, Any] = Field(...)


class WSResetMessage(BaseMessage):
    type: Literal["reset"] = Field(default="reset")
    data: Dict[str, Any] = Field(default_factory=dict)


class WSStepMessage(BaseMessage):
    type: Literal["step"] = Field(default="step")
    data: Dict[str, Any] = Field(...)


class WSStateMessage(BaseMessage):
    type: Literal["state"] = Field(default="state")


class WSCloseMessage(BaseMessage):
    type: Literal["close"] = Field(default="close")


WSIncomingMessage = Annotated[
    WSResetMessage | WSStepMessage | WSStateMessage | WSCloseMessage,
    Field(discriminator="type"),
]


class WSObservationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["observation"] = Field(default="observation")
    data: Dict[str, Any] = Field(...)


class WSStateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["state"] = Field(default="state")
    data: Dict[str, Any] = Field(...)


class WSErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["error"] = Field(default="error")
    data: Dict[str, Any] = Field(...)


class ServerCapacityStatus(BaseMessage):
    active_sessions: int = Field(ge=0)
    max_sessions: int = Field(ge=1)

    @model_validator(mode="after")
    def check_capacity_bounds(self) -> "ServerCapacityStatus":
        if self.active_sessions > self.max_sessions:
            raise ValueError("active_sessions cannot exceed max_sessions")
        return self

    @property
    def available_slots(self) -> int:
        return self.max_sessions - self.active_sessions

    @property
    def is_at_capacity(self) -> bool:
        return self.available_slots == 0
