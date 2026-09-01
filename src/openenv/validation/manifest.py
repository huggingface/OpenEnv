"""The normalized manifest: the entire interface between a package format and every grader.

One document, produced only by parsers, consumed only by graders. Capabilities select
contract graders; type tags select domain graders. The manifest's `signature` field is
report provenance only — grader selection never reads it.
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .types import SignatureKind


class RewardDeclaration(BaseModel):
    """
    Author-declared reward contract. Graders normalize measurements against it.

    Declared tolerances are bounded by the severity policy; reports carry the declared
    values so hubs can apply stricter ceilings.

    Attributes:
        range (`tuple[float, float]`):
            Declared reward range, defaults to `(0.0, 1.0)`. Must be strictly increasing.
        oracle_tolerance (`float`):
            The oracle check passes when `reward(oracle) >= max - oracle_tolerance`.
        floor_margin (`float`):
            The gap check requires `max - measured_floor >= floor_margin`.
        variance_tolerance (`float`, *optional*):
            Required iff the environment declares `llm_judged`; bounds reward variance
            for variance-mode determinism checks.
    """

    model_config = ConfigDict(extra="forbid")

    range: tuple[float, float] = (0.0, 1.0)
    oracle_tolerance: float = Field(default=0.0, ge=0.0)
    floor_margin: float = Field(ge=0.0)
    variance_tolerance: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def _range_increasing(self) -> "RewardDeclaration":
        low, high = self.range
        if not low < high:
            raise ValueError(
                f"reward range must be strictly increasing, got {self.range}"
            )
        return self


class JudgePin(BaseModel):
    """
    Pinned judge configuration. Required iff `capabilities.llm_judged`.

    The pin lives in the manifest, not the rubric object: `LLMJudge.state_dict()` does
    not serialize model/version/params, so validation checks the declared pin.
    """

    model_config = ConfigDict(extra="forbid")

    model: str
    version: str
    params: dict[str, Any] = Field(default_factory=dict)


class OracleDeclaration(BaseModel):
    """
    How the package demonstrates max reward.

    Attributes:
        form (`str`):
            `"injected_state"` (preferred — most deterministic; requires the
            `set_state` capability) or `"script"` (Harbor `solution/solve.sh`
            precedent, executed inside the sandbox).
        location (`str`):
            Package-relative path to the oracle artifact; containment-checked.
    """

    model_config = ConfigDict(extra="forbid")

    form: Literal["injected_state", "script"]
    location: str


class VerifierBinding(BaseModel):
    """
    How "evaluate this state" is invoked.

    Attributes:
        kind (`str`):
            `"reward_channel"` (served reward) or `"script"` (Harbor-style
            `tests/test.sh`).
        entry (`str`, *optional*):
            Script path; required when `kind == "script"`, forbidden otherwise.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["reward_channel", "script"]
    entry: str | None = None

    @model_validator(mode="after")
    def _entry_matches_kind(self) -> "VerifierBinding":
        if self.kind == "script" and self.entry is None:
            raise ValueError(
                "verifier.entry is required when verifier.kind is 'script'"
            )
        if self.kind == "reward_channel" and self.entry is not None:
            raise ValueError(
                "verifier.entry is only valid when verifier.kind is 'script'"
            )
        return self


class ResourceDeclaration(BaseModel):
    """
    Declared resource budget; measured usage must fall within it (#778 test #10).

    GPU needs are declarations, not disqualifiers: a package declaring `gpus`
    validates on a sandbox provider with the GPU capability; on providers without
    it, runtime+ checks SKIP with the capability named.
    """

    model_config = ConfigDict(extra="forbid")

    cpu: float = Field(gt=0.0)
    memory_mb: int = Field(gt=0)
    disk_mb: int = Field(gt=0)
    episode_timeout_s: float = Field(gt=0.0)
    gpus: int = Field(default=0, ge=0)
    gpu_types: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _gpu_types_require_gpus(self) -> "ResourceDeclaration":
        if self.gpu_types and self.gpus == 0:
            raise ValueError("gpu_types is declared but gpus is 0")
        return self


class NetworkPolicy(BaseModel):
    """
    Declared network posture, following the Harbor `task.toml` (schema 1.4)
    `network_mode` precedent.

    Attributes:
        mode (`str`, *optional*, defaults to `"public"`):
            `"public"` (egress allowed — the default), `"no-network"`, or
            `"allowlist"`.
        allowed_hosts (`list[str]`, *optional*):
            Permitted destinations (exact hostnames, CIDR ranges, wildcards).
            Only valid with mode `"allowlist"`.
    """

    model_config = ConfigDict(extra="forbid")

    mode: Literal["public", "no-network", "allowlist"] = "public"
    allowed_hosts: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _hosts_require_allowlist(self) -> "NetworkPolicy":
        if self.mode != "allowlist" and self.allowed_hosts:
            raise ValueError("allowed_hosts is only valid when mode is 'allowlist'")
        if self.mode == "allowlist" and not self.allowed_hosts:
            raise ValueError("mode 'allowlist' requires at least one allowed host")
        return self


class TaskDistributionPin(BaseModel):
    """Pins for generated or externally sourced task distributions (#778 test #42)."""

    model_config = ConfigDict(extra="forbid")

    generator_seed: int | None = None
    dataset_version: str | None = None
    generator_model: str | None = None


class CapabilitiesSpec(BaseModel):
    """
    Declared capabilities; these select the contract graders.

    A `None` oracle is a valid manifest — the missing oracle is a graded FAIL on
    `semantic.oracle_max` (it appears in the report with remediation), not a parse
    error.

    Attributes:
        oracle ([`~openenv.validation.manifest.OracleDeclaration`], *optional*):
            The oracle declaration; absence FAILs `semantic.oracle_max`.
        verifier ([`~openenv.validation.manifest.VerifierBinding`]):
            How the verifier is invoked.
        set_state (`bool`, *optional*, defaults to `False`):
            Whether the environment accepts an `injected_state` kwarg on `reset()`.
            `False` with `oracle.form == "injected_state"` is a manifest schema error.
        llm_judged (`bool`, *optional*, defaults to `False`):
            Reward involves an LLM judge; requires a judge pin and variance tolerance.
        rubric_tree (`bool`, *optional*, defaults to `False`):
            OpenEnv format: an RFC 004 `Rubric` is present.
        task_api (`bool`, *optional*, defaults to `False`):
            `TaskProvider` implemented.
        canaries (`str`, *optional*):
            Package-relative path to shipped canary trajectories (#778 test #28).
        declared_tools (`list[str]`, *optional*):
            Checked against runtime discovery (#778 test #34).
        declared_task_count (`dict[str, int]`, *optional*):
            Split name to task count, checked against discovery (#778 test #35).
    """

    model_config = ConfigDict(extra="forbid")

    oracle: OracleDeclaration | None = None
    verifier: VerifierBinding
    set_state: bool = False
    llm_judged: bool = False
    rubric_tree: bool = False
    task_api: bool = False
    canaries: str | None = None
    declared_tools: list[str] = Field(default_factory=list)
    declared_task_count: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _injected_state_requires_set_state(self) -> "CapabilitiesSpec":
        if (
            self.oracle is not None
            and self.oracle.form == "injected_state"
            and not self.set_state
        ):
            raise ValueError(
                "oracle.form 'injected_state' requires the set_state capability; "
                "declare set_state: true or use oracle.form 'script'"
            )
        return self


class TypeSpec(BaseModel):
    """
    Type tags; these select domain graders.

    Bare tags are shared (`swe`); prefixed tags are hub-scoped (`hf:agentic-swe`).
    pass@k comparisons are valid only within a tag.
    """

    model_config = ConfigDict(extra="forbid")

    tags: list[Annotated[str, Field(min_length=1)]] = Field(min_length=1)


class NormalizedManifest(BaseModel):
    """
    The one document every grader reads. Produced only by parsers.

    After a parser returns this, the package format has left the pipeline: `signature`
    is provenance for the report and nothing else.
    """

    model_config = ConfigDict(extra="forbid")

    manifest_schema_version: Literal["1"]
    name: str
    version: str | None = None
    signature: SignatureKind
    reward: RewardDeclaration
    judge: JudgePin | None = None
    resources: ResourceDeclaration
    task_distribution: TaskDistributionPin | None = None
    network: NetworkPolicy = Field(default_factory=NetworkPolicy)
    capabilities: CapabilitiesSpec
    types: TypeSpec

    @model_validator(mode="after")
    def _judge_pin_iff_llm_judged(self) -> "NormalizedManifest":
        if self.capabilities.llm_judged:
            if self.judge is None:
                raise ValueError(
                    "a judge pin is required when capabilities.llm_judged is true"
                )
            if self.reward.variance_tolerance is None:
                raise ValueError(
                    "reward.variance_tolerance is required when capabilities.llm_judged is true"
                )
        elif self.judge is not None:
            raise ValueError(
                "a judge pin is declared but capabilities.llm_judged is false"
            )
        return self
