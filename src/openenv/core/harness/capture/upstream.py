"""The upstream leg: one HTTP client to a vLLM- or SGLang-compatible OpenAI server.

SGLang was excluded here because it genuinely could not do this: its chat route returned token
*text* with no ids (sgl-project/sglang#18378 asked for exactly this, for the same train/inference
consistency reason). That changed with sgl-project/sglang#30917, merged 2026-07-23, which added
`return_token_ids` to the OpenAI-compatible routes. It is on `main` and NOT in v0.5.16 — that
release has only `return_prompt_token_ids`, the prompt ids without the sampled ones — so an SGLang
endpoint is usable here only when built from main.

One shape difference, absorbed in `normalize_response` below: SGLang returns the prompt ids PER
CHOICE (`choices[0].prompt_token_ids`); vLLM returns them at the TOP LEVEL of the response.
Everything downstream reads the top-level field, so normalisation hoists SGLang's.

Two request params do all the work, and both are easy to get subtly wrong:

    return_token_ids=True   makes vLLM emit `response.prompt_token_ids` and `choice.token_ids`.
                            `prompt_token_ids` is the load-bearing one: it is the engine's own
                            tokenisation of the whole conversation so far, which is what lets turn
                            k+1 be matched against turn k by exact token prefix without us ever
                            tokenising locally.
    top_logprobs=0          must be SET, not omitted. vLLM only populates `logprobs.content[]` when
                            `top_logprobs` is not None, even with `logprobs=True`. Zero returns just
                            the sampled token's logprob, which is all training needs.

Neither is standard, so neither can be sent unconditionally. `capture_level` says how much this
particular endpoint tolerates, and it is discovered by probing rather than configured (see
`validate_llm`):

    tokens      prompt ids + sampled ids + aligned logprobs. vLLM with the two serving flags, or
                SGLang built from main. The only level that yields trainable rollouts.
    logprobs    logprobs but no ids. Nothing trainable can be built from these — an unpaired
                logprob has no token to attach to — so they are kept only as an eval diagnostic.
    text        neither. OpenAI's current models reject `logprobs` outright; Anthropic never had it.

Below `tokens` a rollout is an eval rollout: same path, same agents, reward and full trace, no token
fields. What must never happen is *looking* trainable while carrying nothing, which is why the level
travels with every response and no contract is written without it.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx

from .compat import diagnose, MAX_FIXES, ParamFix

logger = logging.getLogger(__name__)

# Ordered weakest-last: `CAPTURE_LEVELS.index` is how callers compare two levels.
CAPTURE_LEVELS = ("tokens", "logprobs", "text")


def auth_headers(api_key: str | None, header: str = "Authorization") -> dict[str, str]:
    """The one header that authenticates us to the upstream, or `{}` when there is no key.

    `Authorization` gets the `Bearer ` prefix the OpenAI spec asks for; any other header name gets
    the raw key, because the providers that use a custom header (`x-api-key`) want it bare. The
    header name is configurable because the OpenResponses compliance suite treats it as
    configuration, and because Anthropic's native route does not accept `Authorization`.

    Args:
        api_key (`str`, *optional*):
            The upstream credential. This is never the agent-facing key: that one is a capture
            session id and is minted per rollout (see `sessions`).
        header (`str`, *optional*, defaults to `"Authorization"`):
            Header to send it under.

    Returns:
        `dict[str, str]`: Headers to merge into the request.
    """
    if not api_key:
        return {}
    name = (header or "Authorization").strip()
    if name.lower() == "authorization":
        return {name: f"Bearer {api_key}"}
    return {name: api_key}


class UpstreamError(RuntimeError):
    """Any failure talking to the engine. Never leaks httpx types to callers."""


class UpstreamHTTPError(UpstreamError):
    """Engine answered with a non-2xx status."""

    def __init__(
        self, status_code: int, body: dict[str, Any] | str | None = None
    ) -> None:
        self.status_code = status_code
        self.body = body
        detail = body
        if isinstance(body, dict):
            error = body.get("error")
            detail = error.get("message") if isinstance(error, dict) else error or body
        super().__init__(f"upstream returned {status_code}: {str(detail)[:400]}")


class UpstreamTimeoutError(UpstreamError):
    """Engine did not answer within the liveness ceiling."""


class UpstreamTransportError(UpstreamError):
    """Connection-level failure: refused, reset, DNS."""


def normalise_engine_base(url: str) -> str:
    """The engine root, with any trailing `/v1` removed.

    Every route this module builds is already `/v1/...`, so a caller who passes the OpenAI-style
    base (`http://host:8000/v1`, which is what most SDKs and most people hand you) would otherwise
    get `/v1/v1/chat/completions` and see a healthy engine reported as unreachable. Accept both
    forms and normalise here rather than making every call site remember which one it holds.
    """
    base = url.rstrip("/")
    return base[: -len("/v1")] if base.endswith("/v1") else base


# Sampling knobs that make a *processed* logprob incomparable to a full-vocab recompute, mapped to the
# value that disables each one.
#
# vLLM's sampler masks the truncated tail to `-inf` and only then takes the log-softmax
# (`v1/sample/ops/topk_topp_sampler.py`: `apply_top_k_top_p` at line 135, `compute_logprobs` at 139,
# which is `logits.log_softmax(...)`). So under `--logprobs-mode processed_logprobs` with `top_p=0.95`,
# every captured logprob is the *renormalised* one — `log p_full(token) - log(kept_mass)` — while a
# trainer recomputing over the full vocabulary gets `log p_full(token)`. The captured number is
# uniformly too high, so GRPO's step-0 importance ratio `exp(recompute - captured)` comes out at
# `kept_mass` rather than 1, and the penalties are worse than a uniform shift because they reorder.
#
# For a training rollout the sampling distribution must BE the policy distribution — that is what makes
# the data on-policy — so truncation is not a feature to preserve here, it is the bug. TRL's own
# GRPOConfig defaults `top_p` to 1.0 for the same reason. At the `tokens` tier these are therefore set
# to their no-op values; at `logprobs`/`text` they are left exactly as the harness sent them, because an
# eval rollout should score the model the harness actually asked for.
#
# What the harness requested is still recorded per turn (`server.SAMPLING_KEYS` ->
# `TurnNode.sampling_params`), read from the caller's dict before this copy is edited, so overriding
# here does not hide the original request from the trace.
NON_POLICY_SAMPLING: dict[str, float | int] = {
    "top_p": 1.0,
    "top_k": -1,
    "min_p": 0.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
}


def truncating_params(request: dict[str, Any]) -> dict[str, Any]:
    """The entries of `request` that would narrow or distort the sampling distribution.

    Args:
        request (`dict[str, Any]`):
            A chat-completions body.

    Returns:
        `dict[str, Any]`: Requested key -> value, for each knob set to something other than its no-op.
    """
    found: dict[str, Any] = {}
    for key, neutral in NON_POLICY_SAMPLING.items():
        value = request.get(key)
        if (
            value is None
            or not isinstance(value, (int, float))
            or isinstance(value, bool)
        ):
            continue
        # `top_k` is disabled by -1 in vLLM and by 0 elsewhere; neither truncates.
        if key == "top_k" and value in (-1, 0):
            continue
        if float(value) != float(neutral):
            found[key] = value
    return found


def prepare_request(
    request: dict[str, Any],
    *,
    served_model: str | None = None,
    capture_level: str = "tokens",
) -> dict[str, Any]:
    """Add the params that make a response capturable. Mutates and returns `request`.

    Only the top level of `request` is mutated. Nested message dicts are copied before they are
    rewritten, because the caller's messages are the same objects the graph stores: rewriting them
    in place would mean a captured turn no longer records what the harness actually sent.

    Args:
        request (`dict[str, Any]`):
            The chat-completions body, already normalised by the dialect transformer.
        served_model (`str`, *optional*):
            Accepted for symmetry with the caller; the model name is set by the server.
        capture_level (`str`, *optional*, defaults to `"tokens"`):
            How much this endpoint tolerates. See the module docstring.

    Returns:
        `dict[str, Any]`: The same object, edited.
    """
    if capture_level == "tokens":
        request["logprobs"] = True
        request["return_token_ids"] = True
        # See `NON_POLICY_SAMPLING`: a processed logprob is taken after these are applied, so leaving
        # them on would silently bias every importance ratio computed from this rollout.
        for key in truncating_params(request):
            request[key] = NON_POLICY_SAMPLING[key]
    elif capture_level == "logprobs":
        request["logprobs"] = True
    # At `text` nothing is injected at all. Current OpenAI models answer `logprobs` with a 400, and
    # a rejected request is worse than a missing diagnostic: the agent's turn is simply lost.

    # Not `setdefault`: that keeps an explicitly-provided `None`, and vLLM only fills
    # `logprobs.content[]` when `top_logprobs` is not None. A harness that sends
    # `"top_logprobs": null` would then get a normal-looking response with no logprobs at all, so
    # every turn it produced would be silently untrainable.
    if capture_level != "text" and request.get("top_logprobs") is None:
        request["top_logprobs"] = 0

    # vLLM reads a prior turn's thinking from `reasoning`, while the dialect transformers emit the
    # canonical `reasoning_content`. Without this rename an earlier turn's interleaved thinking
    # renders as an empty `<think></think>` and the prompt silently differs from what the model
    # actually produced — which breaks prefix matching for the turn after it.
    #
    # That rename is a vLLM accommodation, so at `text` the field is dropped instead: a non-standard
    # key inside a message is the same 400 hazard as a non-standard top-level param, and there is no
    # prefix matching at that level for it to protect.
    messages = request.get("messages")
    if isinstance(messages, list):
        rewritten: list[Any] = []
        for message in messages:
            if (
                isinstance(message, dict)
                and message.get("reasoning_content") is not None
            ):
                message = dict(message)
                if capture_level == "text":
                    message.pop("reasoning_content")
                else:
                    message["reasoning"] = message.pop("reasoning_content")
            rewritten.append(message)
        request["messages"] = rewritten

    # `_served_model` is an internal marker the dialect transformers read; the server sets it before
    # transforming and the transformer strips it. Setting it here would be too late to be read and
    # would leak an unknown field to the engine, so this only guarantees it is gone.
    request.pop("_served_model", None)
    return request


def normalize_response(response: dict[str, Any]) -> dict[str, Any]:
    """Canonicalise the engine's response shape in place, so callers see one shape.

    Two engines, two placements for the same field. vLLM puts the prompt ids at the top level of the
    response; SGLang puts them on each choice (`ChatCompletionResponseChoice.prompt_token_ids`, added
    by sgl-project/sglang#30917). Every reader downstream — `check_upstream_response`, the capture
    server's `_ingest`, the UI — looks only at the top level, so hoist rather than teach each of them
    both spellings.

    Hoisted from `choices[0]` specifically, and only when the top level is empty: the prompt is a
    property of the request, so with n>1 every choice carries the same list, and a top-level value
    that is already present is the engine's own and must win.
    """
    choices = response.get("choices")
    if not isinstance(choices, list):
        return response

    if (
        not response.get("prompt_token_ids")
        and choices
        and isinstance(choices[0], dict)
    ):
        hoisted = choices[0].get("prompt_token_ids")
        if hoisted:
            response["prompt_token_ids"] = hoisted

    for choice in choices:
        if not isinstance(choice, dict):
            continue

        message = choice.get("message")
        if isinstance(message, dict):
            if (
                message.get("reasoning_content") is None
                and message.get("reasoning") is not None
            ):
                message["reasoning_content"] = message.pop("reasoning")

        # Copy each token id onto its logprob entry, and CHECK the pairing while doing it.
        #
        # The ids and the logprobs arrive on two separate channels and are joined by index. Equal
        # length was the only guard, which an equal-length-but-SHIFTED pairing passes — a stop or EOS
        # token present in one channel and not the other is enough — after which every logprob is
        # attributed to its neighbour's token and training proceeds silently on the misattribution.
        #
        # When the engine runs with --return-tokens-as-token-ids the check is free and exact: each
        # entry's `token` field literally reads `token_id:{id}`, so it can be compared against
        # `token_ids[i]` at every position rather than inspected once for a warning. A disagreement
        # drops the logprobs, which is what every other unusable-logprob path already does — the ids
        # stay as real context and `sequence_for` masks the turn out of training.
        token_ids = choice.get("token_ids")
        entries = ((choice.get("logprobs") or {}).get("content")) or []
        if isinstance(token_ids, list) and len(token_ids) == len(entries):
            mismatch = _pairing_mismatch(token_ids, entries)
            if mismatch is not None:
                position, declared, actual = mismatch
                logger.warning(
                    "token id / logprob channels disagree at position %d (ids say %s, logprobs say "
                    "%s); dropping the logprobs for this turn rather than training on a shifted "
                    "pairing",
                    position,
                    declared,
                    actual,
                )
                choice["logprobs"] = None
            else:
                for token_id, entry in zip(token_ids, entries):
                    if isinstance(entry, dict):
                        entry.setdefault("token_id", token_id)
    return response


def _pairing_mismatch(
    token_ids: list[Any], entries: list[Any]
) -> tuple[int, Any, Any] | None:
    """The first position where the two channels disagree, or `None`.

    Only positions whose `token` field is in the `token_id:{id}` form can be checked; a server without
    `--return-tokens-as-token-ids` emits token TEXT and is skipped, since decoding text back to an id
    would need the tokenizer this design deliberately does not load.
    """
    for position, (token_id, entry) in enumerate(zip(token_ids, entries)):
        if not isinstance(entry, dict):
            continue
        token = entry.get("token")
        if not isinstance(token, str) or not token.startswith("token_id:"):
            continue
        declared = token[len("token_id:") :]
        if declared != str(token_id):
            return position, token_id, declared
    return None


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """The `Retry-After` delay a provider asked for, in seconds, if it gave a usable one.

    Only the delta-seconds form is honoured. The HTTP-date form is legal but rare here, and parsing
    it wrong would either sleep for hours or not at all.
    """
    raw = response.headers.get("retry-after")
    if not raw:
        return None
    try:
        return max(0.0, float(raw.strip()))
    except ValueError:
        return None


class InferenceClient:
    """Async client to one engine. One instance per server, shared across sessions."""

    # A high ceiling, not a per-request budget. Callers impose their own deadline; this exists only
    # so a wedged engine cannot pin a connection forever.
    _LIVENESS_TIMEOUT_S = 900.0
    _CONNECT_TIMEOUT_S = 30.0

    # Hosted providers rate-limit; a local engine effectively never does. Without this a single 429
    # became a 502 to the agent, which truncates its trajectory while leaving a graph that looks
    # perfectly well-formed — the failure class ATIF reconciliation exists to catch.
    _RETRY_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504})
    _MAX_ATTEMPTS = 3
    _BACKOFF_S = 2.0
    # A `Retry-After` longer than this is not worth honouring inside one rollout; the sandbox has its
    # own agent timeout and would be killed waiting.
    _MAX_RETRY_AFTER_S = 60.0

    def __init__(
        self,
        base_url: str,
        *,
        served_model: str | None = None,
        api_key: str | None = None,
        auth_header: str = "Authorization",
        capture_level: str = "tokens",
    ) -> None:
        self.base_url = normalise_engine_base(base_url)
        self.served_model = served_model
        self.api_key = api_key or None
        self.auth_header = auth_header or "Authorization"
        self.capture_level = capture_level
        # Fixes discovered from the provider's own 400s, applied to every later request. Cached
        # because they are a property of the endpoint and the model, not of one call: rediscovering
        # them per request would double the call count for the life of the server.
        self.param_fixes: list[ParamFix] = []
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(
                    self._LIVENESS_TIMEOUT_S, connect=self._CONNECT_TIMEOUT_S
                ),
                headers=auth_headers(self.api_key, self.auth_header),
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    async def completion(self, request: dict[str, Any]) -> dict[str, Any]:
        """One non-streaming chat completion, prepared for capture and normalised on the way back."""
        body = prepare_request(
            dict(request),
            served_model=self.served_model,
            capture_level=self.capture_level,
        )
        payload = await self._post("/v1/chat/completions", body)
        return normalize_response(payload)

    async def list_models(self) -> dict[str, Any]:
        client = await self._get_client()
        try:
            response = await client.get("/v1/models")
        except httpx.RequestError as exc:
            raise self._transport_error(exc) from exc
        await self._raise_for_status(response)
        return response.json()

    async def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        """POST with two independent recovery paths: transient status, and a rejected parameter.

        They are separate because they mean different things. A 429 means "the same request, later";
        a 400 naming a parameter means "a different request, now". Conflating them would either sleep
        through a permanent failure or hammer a provider that asked us to slow down.
        """
        for fix in self.param_fixes:
            fix.apply(body)

        while True:
            try:
                return await self._post_with_retries(path, body)
            except UpstreamHTTPError as exc:
                if exc.status_code != 400 or len(self.param_fixes) >= MAX_FIXES:
                    raise
                fix = diagnose(exc.body)
                # `apply` returning False means the parameter it named is not in this body, so
                # retrying would send the identical request and get the identical 400.
                if fix is None or fix in self.param_fixes or not fix.apply(body):
                    raise
                self.param_fixes.append(fix)
                logger.info(
                    "upstream rejected a parameter; %s and retrying (this endpoint now carries "
                    "%d fix(es))",
                    fix,
                    len(self.param_fixes),
                )

    async def _post_with_retries(
        self, path: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        client = await self._get_client()
        for attempt in range(1, self._MAX_ATTEMPTS + 1):
            try:
                response = await client.post(path, json=body)
            except httpx.RequestError as exc:
                raise self._transport_error(exc) from exc
            if response.is_success:
                return response.json()

            retry_after = _retry_after_seconds(response)
            if (
                response.status_code not in self._RETRY_STATUSES
                or attempt == self._MAX_ATTEMPTS
            ):
                await self._raise_for_status(response)
            delay = (
                retry_after
                if retry_after is not None
                else self._BACKOFF_S * (2 ** (attempt - 1))
            )
            logger.info(
                "upstream returned %d; retrying in %.1fs (attempt %d/%d)",
                response.status_code,
                delay,
                attempt,
                self._MAX_ATTEMPTS,
            )
            await response.aclose()
            await asyncio.sleep(min(delay, self._MAX_RETRY_AFTER_S))
        raise AssertionError("unreachable: the final attempt always raises or returns")

    async def _raise_for_status(self, response: httpx.Response) -> None:
        if response.is_success:
            return
        content = await response.aread()
        await response.aclose()
        body: dict[str, Any] | str | None = None
        text = content.decode("utf-8", errors="replace").strip()
        if text:
            try:
                body = json.loads(text)
            except json.JSONDecodeError:
                body = text
        raise UpstreamHTTPError(response.status_code, body)

    @staticmethod
    def _transport_error(exc: httpx.RequestError) -> UpstreamError:
        if isinstance(exc, httpx.TimeoutException):
            return UpstreamTimeoutError(f"engine timed out: {exc}")
        return UpstreamTransportError(f"could not reach engine: {exc}")
