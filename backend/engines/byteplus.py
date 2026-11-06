"""
BytePlus ModelArk Engine
- Text / vision: OpenAI-compatible ``/chat/completions`` (Seed 2.x and DeepSeek V4 models, image inputs as data URIs,
  ``thinking`` on/off, ``response_format`` incl. json_schema on models that support it); long calls (thinking, or a large
  ``max_tokens``) are streamed, so a slow model that is still writing is never mistaken for a dead connection
- Images: ``/images/generations`` with Seedream 5.0 Pro (one image per call, 2–10 reference images,
  explicit ``WxH`` sizes, no negative prompt, no temperature)

Docs: https://ai.byteplus.com/ark/region:ap-southeast-1/docs/image-generation-api.md
"""

import asyncio
import hashlib
import json
import random
import ssl
import time
from typing import Optional, AsyncIterator

import httpx

from .base import (
    BaseEngine,
    Message,
    ImageContent,
    GenerationConfig,
    ImageGenerationConfig,
    TextResponse,
    ImageResponse,
)


DEFAULT_BASE_URL = "https://ark.ap-southeast.bytepluses.com/api/v3"
DEFAULT_TEXT_MODEL = "seed-2-0-pro-260328"
DEFAULT_IMAGE_MODEL = "dola-seedream-5-0-pro-260628"

# Seedream 5.0 Pro / Flash explicit-size limits (image generation scenario)
SEEDREAM_MIN_PIXELS = 1280 * 720            # 921,600
SEEDREAM_MAX_PIXELS = 4_624_220             # 2048x2048x1.1025
SEEDREAM_MAX_REFERENCE_IMAGES = 10

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
STREAM_FROM_TOKENS = 8000   # replies allowed this long (or with thinking) are streamed: DeepSeek V4 Pro thinks for minutes
# Seed keeps a prompt prefix in its implicit cache for 30-60 s only (measured 2026-10-27: a hit 30 s later, none 60 s later),
# so a job's paper-first calls, minutes apart, read the paper at full price (19% of Seed's input was cached). An explicit
# prefix cache (Responses API) serves it for an hour, whatever the gaps; storage costs $0.0083 per million tokens an hour.
PREFIX_CACHE_MODELS = ("seed-", "dola-seed-")
PREFIX_CACHE_SECONDS = 3600
PREFIX_MIN_CHARS = 4000          # about 1,000 tokens: a shorter prefix is not worth a cache
THINKING_HEADROOM = 65536        # a reply's max_output_tokens counts the reasoning; the chat API's max_tokens does not
MAX_OUTPUT_TOKENS = 131072


class BytePlusAPIError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"ModelArk HTTP {status}: {body[:500]}")
        self.status = status
        self.body = body
        self.code, self.detail = _error_code(body)


def _error_code(body: str) -> tuple[str, str]:
    """(code, message) of a ModelArk error body ({"error": {"code": "ModelNotOpen", "message": ...}}); ("", "") if none."""
    try:
        err = json.loads(body).get("error") or {}
    except (ValueError, AttributeError):
        return "", ""
    return (str(err.get("code") or ""), str(err.get("message") or "")) if isinstance(err, dict) else ("", "")


# what the user must do, per ModelArk error code (https://ai.byteplus.com/ark/region:ap-southeast-1/docs/error-code.md)
ACCOUNT_HINTS = {
    "AuthenticationError": ("The ModelArk API key was refused: check providers.byteplus.api_key in config/api_config.yaml "
                            "(or ARK_API_KEY) and restart the backend.",
                            "ModelArk 拒绝了 API key：请检查 config/api_config.yaml 中的 providers.byteplus.api_key（或环境变量 ARK_API_KEY），然后重启后端。"),
    "ModelNotOpen": ("The model {model} is not activated for this account: activate it in the ModelArk console "
                     "(Activation management), then try again.",
                     "模型 {model} 尚未开通：请在 ModelArk 控制台的开通管理中开通后重试。"),
    "InvalidEndpointOrModel": ("The model {model} does not exist for this account or region: check core_engine in config/api_config.yaml.",
                               "当前账号或地域没有模型 {model}：请检查 config/api_config.yaml 中的 core_engine。"),
    "AccountOverdueError": ("The BytePlus account is overdue: top up in the BytePlus Billing Center, then try again.",
                            "BytePlus 账户已欠费：请在费用中心充值后重试。"),
    "QuotaExceeded": ("The free quota of {model} is used up: activate its paid service in the ModelArk console.",
                      "模型 {model} 的免费额度已用完：请在 ModelArk 控制台开通付费服务。"),
    "SetLimitExceeded": ("A limit set on the account stopped the request (Safe Experience Mode or an inference limit): raise or "
                         "turn it off in the ModelArk console.",
                         "账户上设置的限额（安全体验模式或推理限额）拦下了请求：请在 ModelArk 控制台调高或关闭。"),
    "AccessDenied": ("This API key may not use {model}: check the key's permissions in the ModelArk console.",
                     "当前 API key 无权使用 {model}：请在 ModelArk 控制台检查权限。"),
    "OperationDenied": ("ModelArk refused the request for {model} (the service is not enabled or not allowed for this account): "
                        "check the account in the ModelArk console.",
                        "ModelArk 拒绝了对 {model} 的请求（该服务未开通或账号不允许）：请在 ModelArk 控制台检查账号。"),
}
RATE_CODES = ("RateLimitExceeded", "ServerOverloaded", "RequestBurstTooFast", "ModelAccountTpmRateLimitExceeded",
              "ModelAccountRpmRateLimitExceeded")


class AccountError(BytePlusAPIError):
    """The account cannot serve the request (a refused key, a model not activated, an overdue account, a used-up quota, a
    spending limit): waiting and retrying never help, so it is raised at once and stops the job, with what to do."""

    account = True   # best-effort steps (fact-check, quiz, translation fixes) re-raise errors that carry this flag

    def __init__(self, status: int, body: str, model: str = ""):
        super().__init__(status, body)
        self.model = model
        key = next((k for k in ACCOUNT_HINTS if self.code.startswith(k)), "AuthenticationError" if status == 401 else "")
        en, zh = ACCOUNT_HINTS.get(key, ("ModelArk refused the request: {detail}", "ModelArk 拒绝了请求：{detail}"))
        fill = {"model": model or "the model", "detail": self.detail or body[:200]}
        self.hint = f"{en.format(**fill)} / {zh.format(**fill)}"

    def __str__(self) -> str:
        return f"{self.hint} (ModelArk {self.code or self.status})"


class PrefixCacheError(RuntimeError):
    """An explicitly cached call could not be made: the chat API answers it, as before. ``permanent`` (the Responses API
    refused a parameter, or this model on that endpoint) stops explicit caching of the model for the engine's life;
    otherwise (the cache could not be created just now, an unexpected refusal) only this call goes to the chat API."""

    def __init__(self, message: str, permanent: bool = False):
        super().__init__(message)
        self.permanent = permanent


# account-wide refusals stop the job on any endpoint; a refusal of one model or feature may be the Responses endpoint's only
ACCOUNT_WIDE = ("AuthenticationError", "AccountOverdueError", "SetLimitExceeded", "QuotaExceeded")


class ContentFilteredError(BytePlusAPIError):
    """The provider's content filter refused the request and answered nothing (HTTP 200, finish_reason
    "content_filter", 0 tokens): DeepSeek V4 Pro does this for some ordinary paper text (CoT's CommonsenseQA examples)
    that Seed and DeepSeek V4 Flash read fine. Never an empty answer to repair: the caller retries on another model."""

    def __init__(self, model: str):
        super().__init__(200, json.dumps({"error": {"code": "ContentFilter", "message": f"{model} refused the request (content filter)"}}))
        self.model = model


def _api_error(status: int, body: str, model: str = "") -> BytePlusAPIError:
    """The error for a ModelArk reply: an AccountError when the account must change something first."""
    code, _detail = _error_code(body)
    account = (status == 401 or any(code.startswith(k) for k in ACCOUNT_HINTS)) and not any(code.startswith(r) for r in RATE_CODES)
    return AccountError(status, body, model) if account else BytePlusAPIError(status, body)


RETRYABLE_STREAM_CODES = ("Internal", "ServerOverloaded", "ServiceUnavailable", "Timeout", "Busy")


def _stream_error(error) -> BytePlusAPIError:
    """An error event inside a streamed reply. A rate limit or an overloaded server can cut a reply off minutes into it
    (RequestBurstTooFast stopped a story call at event 2869, 2026-10-27, and failed the job): those are marked
    ``retryable`` and asked again with the usual backoff; any other error stops the call."""
    body = json.dumps({"error": error})[:500]
    code = str(error.get("code") or "") if isinstance(error, dict) else ""
    err = BytePlusAPIError(429 if code.startswith(RATE_CODES) else 500, body)
    err.retryable = code.startswith(RATE_CODES) or any(k in code for k in RETRYABLE_STREAM_CODES)
    return err


def _response_text(data: dict) -> str:
    """The answer of a Responses API reply: its message items' output_text (not the reasoning)."""
    return "".join(str(part.get("text") or "") for item in data.get("output") or [] if isinstance(item, dict) and item.get("type") == "message"
                   for part in item.get("content") or [] if isinstance(part, dict) and part.get("type") == "output_text")


def normalize_size(width: int, height: int) -> str:
    """Fit a requested WxH into Seedream's limits: total pixels in [921,600, 4,624,220], aspect ratio in [1/16, 16],
    both sides multiples of 16. Sizes that already satisfy the limits are kept unchanged (e.g. 1024x1536, 2048x1024)."""
    width, height = max(int(width), 16), max(int(height), 16)
    ratio = min(max(width / height, 1 / 16), 16)
    pixels = width * height
    if not (SEEDREAM_MIN_PIXELS <= pixels <= SEEDREAM_MAX_PIXELS) or ratio != width / height:
        target = min(max(pixels, SEEDREAM_MIN_PIXELS * 1.01), SEEDREAM_MAX_PIXELS * 0.99)
        height = (target / ratio) ** 0.5
        width = height * ratio
    w = max(16, int(round(width / 16)) * 16)
    h = max(16, int(round(height / 16)) * 16)
    for _ in range(512):            # nudge in 16 px steps until every limit holds
        p = w * h
        if p > SEEDREAM_MAX_PIXELS:
            if w >= h:
                w -= 16
            else:
                h -= 16
        elif p < SEEDREAM_MIN_PIXELS:
            if w <= h:
                w += 16
            else:
                h += 16
        elif w > 16 * h:
            h += 16
        elif h > 16 * w:
            w += 16
        else:
            break
    return f"{w}x{h}"


class BytePlusEngine(BaseEngine):
    """ModelArk engine: Seed 2.x for text/vision, Seedream 5.0 Pro for images."""

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_TEXT_MODEL,
        image_model: str = DEFAULT_IMAGE_MODEL,
        thinking: bool = False,
        max_retries: int = 6,
        reasoning_effort: Optional[str] = None,
    ):
        if not api_key or api_key.strip() == "" or api_key.startswith("${"):
            raise ValueError(
                "The BytePlus ModelArk API key is not set: put it in config/api_config.yaml (providers.byteplus.api_key) "
                "or the ARK_API_KEY environment variable, then restart the backend. / BytePlus ModelArk API key 未设置："
                "请在 config/api_config.yaml 的 providers.byteplus.api_key 中设置，或设置环境变量 ARK_API_KEY，然后重启后端。"
            )
        super().__init__(api_key, base_url or DEFAULT_BASE_URL, model)
        self.image_model = image_model or DEFAULT_IMAGE_MODEL
        self.default_thinking = thinking
        self.reasoning_effort = reasoning_effort   # how long a thinking call may reason (DeepSeek V4 Pro: low | high, default high)
        self.max_retries = max_retries
        self._client: Optional[httpx.AsyncClient] = None
        self.tokens: dict[str, dict[str, int]] = {}   # model → prompt / completion / cached tokens spent through this engine
        self._prefixes: dict[tuple, tuple[str, float]] = {}   # (model, thinking, prefix hash) → (cache response id, expiry)
        self._prefix_locks: dict[tuple, asyncio.Lock] = {}
        self._prefix_off: set[str] = set()   # models whose explicit cache the API refused: the chat API serves them
        self._prefix_failed: dict[tuple, float] = {}   # key → when creating its cache last failed (the chat API serves it a while)

    # ------------------------------------------------------------------ meta
    @property
    def name(self) -> str:
        return "BytePlus ModelArk"

    @property
    def supports_image_generation(self) -> bool:
        return bool(self.image_model)

    @property
    def supports_vision(self) -> bool:
        return True

    # --------------------------------------------------------------- client
    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(600.0, connect=20.0, read=420.0),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
            )
        return self._client

    async def _post(self, path: str, payload: dict) -> dict:
        """POST with exponential backoff on transient errors (429 / 5xx / network)."""
        client = await self._get_client()
        url = f"{self.base_url}{path}"
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                response = await client.post(url, json=payload)
                if response.status_code < 400:
                    return response.json()
                err = _api_error(response.status_code, response.text, str(payload.get("model") or ""))
                if isinstance(err, AccountError) or response.status_code not in RETRYABLE_STATUS or attempt == self.max_retries:
                    raise err
                last_error = err
            except (httpx.TimeoutException, httpx.TransportError, ssl.SSLError) as e:   # "record layer failure" mid-read too
                if attempt == self.max_retries:
                    raise RuntimeError(f"ModelArk {path} unreachable after {attempt + 1} attempts: {type(e).__name__} {e}") from e
                last_error = f"{type(e).__name__} {e}".strip()
            delay = min(3 * 2 ** attempt + random.uniform(0, 2), 45)
            print(f"[BytePlus] {path} attempt {attempt + 1} failed ({last_error}); retrying in {delay:.1f}s")
            await asyncio.sleep(delay)
        raise RuntimeError(f"unreachable: {last_error}")

    async def _post_stream(self, path: str, payload: dict) -> dict:
        """``_post`` with "stream": true, assembled into the non-streaming reply shape (content, finish_reason, usage).
        The read timeout then counts the gap between chunks, not the whole reply: DeepSeek V4 Pro can think for longer
        than any sane whole-reply timeout. Transient errors, including a stream that breaks off, are retried from the
        start with the same backoff as ``_post``."""
        client = await self._get_client()
        url = f"{self.base_url}{path}"
        responses = path == "/responses"
        body = {**payload, "stream": True, **({} if responses else {"stream_options": {"include_usage": True}})}
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                async with client.stream("POST", url, json=body) as response:
                    if response.status_code < 400:
                        try:
                            return await (self._assemble_response if responses else self._assemble)(response, payload.get("model", self.model))
                        except BytePlusAPIError as e:   # an error event in the stream
                            if not getattr(e, "retryable", False) or attempt == self.max_retries:
                                raise
                            err = e
                    else:
                        err = _api_error(response.status_code, (await response.aread()).decode(errors="replace"), str(payload.get("model") or ""))
                        if isinstance(err, AccountError) or response.status_code not in RETRYABLE_STATUS or attempt == self.max_retries:
                            raise err
                last_error = err
            except (httpx.TimeoutException, httpx.TransportError, ssl.SSLError) as e:   # "record layer failure" mid-read too
                if attempt == self.max_retries:
                    raise RuntimeError(f"ModelArk {path} unreachable after {attempt + 1} attempts: {type(e).__name__} {e}") from e
                last_error = f"{type(e).__name__} {e}".strip()
            delay = min(3 * 2 ** attempt + random.uniform(0, 2), 45)
            print(f"[BytePlus] {path} (stream) attempt {attempt + 1} failed ({last_error}); retrying in {delay:.1f}s")
            await asyncio.sleep(delay)
        raise RuntimeError(f"unreachable: {last_error}")

    @staticmethod
    async def _assemble(response: httpx.Response, model: str) -> dict:
        """The chunks of a streamed chat completion as one reply: the answer (not the reasoning), the finish reason and
        the usage the last chunk carries."""
        content: list[str] = []
        finish, usage = "", {}
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            chunk = line[6:].strip()
            if chunk == "[DONE]":
                break
            try:
                data = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            if data.get("error"):
                raise _stream_error(data["error"])
            model = data.get("model") or model
            usage = data.get("usage") or usage
            for choice in data.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    content.append(delta["content"])
                finish = choice.get("finish_reason") or finish
        return {"model": model, "choices": [{"message": {"content": "".join(content)}, "finish_reason": finish}], "usage": usage}

    @staticmethod
    async def _assemble_response(response: httpx.Response, model: str) -> dict:
        """A streamed Responses API reply as the reply body: the final event's response (status, output, usage), its text
        rebuilt from the deltas when the final event did not come."""
        deltas: list[str] = []
        final: Optional[dict] = None
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue   # "event: ..." lines name the type the data repeats
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                break
            try:
                event = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            kind = event.get("type")
            if kind == "response.output_text.delta":
                deltas.append(str(event.get("delta") or ""))
            elif kind in ("response.completed", "response.incomplete", "response.failed") and isinstance(event.get("response"), dict):
                final = event["response"]
            elif kind == "error" or event.get("error"):
                raise _stream_error(event.get("error") or event)
        if final is None:
            return {"model": model, "status": "incomplete", "incomplete_details": {"reason": "stream ended"},
                    "output": [{"type": "message", "content": [{"type": "output_text", "text": "".join(deltas)}]}], "usage": {}}
        if final.get("status") == "failed":
            raise BytePlusAPIError(500, json.dumps({"error": final.get("error") or {"message": "response failed"}})[:500])
        if not _response_text(final) and deltas:
            final = {**final, "output": [{"type": "message", "content": [{"type": "output_text", "text": "".join(deltas)}]}]}
        return final

    # ----------------------------------------------------------------- text
    @staticmethod
    def _image_part(img: ImageContent) -> dict:
        if img.is_url:
            return {"type": "image_url", "image_url": {"url": img.data}}
        return {"type": "image_url", "image_url": {"url": f"data:{img.mime_type};base64,{img.data}"}}

    def _build_messages(self, messages: list[Message]) -> list[dict]:
        result = []
        for msg in messages:
            if msg.images:
                content = [self._image_part(img) for img in msg.images]
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                result.append({"role": msg.role.value, "content": content})
            else:
                result.append({"role": msg.role.value, "content": msg.content})
        return result

    def _text_payload(self, messages: list[Message], config: Optional[GenerationConfig]) -> dict:
        config = config or GenerationConfig()
        thinking = getattr(config, "thinking", None)
        if thinking is None:
            thinking = self.default_thinking
        payload = {
            "model": getattr(config, "model", None) or self.model,
            "messages": self._build_messages(messages),
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "top_p": config.top_p,
            "thinking": {"type": "enabled" if thinking else "disabled"},
        }
        effort = getattr(config, "reasoning_effort", None) or self.reasoning_effort
        if thinking and effort:
            payload["reasoning_effort"] = effort
        if getattr(config, "prompt_cache_key", None):
            payload["prompt_cache_key"] = config.prompt_cache_key
        if config.stop_sequences:
            payload["stop"] = config.stop_sequences
        response_format = getattr(config, "response_format", None)
        if response_format:
            payload["response_format"] = response_format
        return payload

    async def generate_text(
        self,
        messages: list[Message],
        config: Optional[GenerationConfig] = None,
    ) -> TextResponse:
        payload = self._text_payload(messages, config)
        if self._prefix_cacheable(messages, config, payload):
            try:
                return await self._generate_cached(messages, payload)
            except PrefixCacheError as e:
                print(f"[BytePlus] explicit prefix cache unavailable for {payload['model']} ({str(e)[:200]}); using the chat API"
                      + (" from now on" if e.permanent else " for this call"))
                if e.permanent:
                    self._prefix_off.add(payload["model"])
        long = payload["thinking"]["type"] == "enabled" or int(payload.get("max_tokens") or 0) >= STREAM_FROM_TOKENS
        data = await (self._post_stream if long else self._post)("/chat/completions", payload)
        choice = data["choices"][0]
        message = choice.get("message", {})
        usage = data.get("usage", {}) or {}
        counted = {
            "prompt_tokens": usage.get("prompt_tokens", 0) or 0,
            "completion_tokens": usage.get("completion_tokens", 0) or 0,
            "total_tokens": usage.get("total_tokens", 0) or 0,
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0,
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0,
        }
        self._count(payload["model"], counted)
        if choice.get("finish_reason") == "content_filter" and not (message.get("content") or "").strip():
            raise ContentFilteredError(payload["model"])
        return TextResponse(
            content=message.get("content") or "",
            model=data.get("model", payload["model"]),
            usage=counted,
            finish_reason=choice.get("finish_reason", ""),
        )

    def _prefix_cacheable(self, messages: list[Message], config: Optional[GenerationConfig], payload: dict) -> bool:
        """A call whose first message is a long shared prefix (``config.cache_prefix``: the paper) on a Seed model, with
        text only and nothing the Responses API cannot take after a cached turn (json_schema, stop sequences)."""
        model = str(payload.get("model") or "")
        fmt = (payload.get("response_format") or {}).get("type")
        return (bool(getattr(config, "cache_prefix", False)) and model.startswith(PREFIX_CACHE_MODELS) and model not in self._prefix_off
                and len(messages) >= 2 and messages[0].role.value == "system" and len(messages[0].content or "") >= PREFIX_MIN_CHARS
                and not any(m.images for m in messages) and not payload.get("stop") and fmt in (None, "json_object"))

    async def _prefix_id(self, key: tuple, model: str, thinking: dict, prefix: str) -> str:
        """The id of the explicit cache of ``prefix`` for this model and thinking mode (a cached head serves only calls with
        the same thinking setting), created on first use and renewed before its hour runs out."""
        async with self._prefix_locks.setdefault(key, asyncio.Lock()):
            held = self._prefixes.get(key)
            if held and held[1] - time.time() > 300:   # never start a call on a cache about to expire
                return held[0]
            if time.time() - self._prefix_failed.get(key, 0.0) < 300:   # the callers queued behind a failed creation
                raise PrefixCacheError("creating the prefix cache failed a moment ago")
            expire_at = int(time.time()) + PREFIX_CACHE_SECONDS
            try:
                data = await self._post("/responses", {"model": model, "input": [{"role": "system", "content": prefix}],
                                                       "caching": {"type": "enabled", "prefix": True}, "thinking": thinking,
                                                       "expire_at": expire_at})
            except AccountError:
                raise
            except (BytePlusAPIError, RuntimeError) as e:   # after _post's retries: the chat API serves these calls
                self._prefix_failed[key] = time.time()
                if isinstance(e, BytePlusAPIError) and e.code.startswith("InvalidParameter"):
                    raise PrefixCacheError(str(e), permanent=True) from e
                raise PrefixCacheError(f"could not create the prefix cache: {e}") from e
            usage = data.get("usage") or {}
            self._count(model, {"prompt_tokens": usage.get("input_tokens", 0),
                                "cached_tokens": (usage.get("input_tokens_details") or {}).get("cached_tokens", 0)})
            if not data.get("id"):
                raise PrefixCacheError(f"no cache id in the reply: {str(data)[:200]}")
            self._prefixes[key] = (str(data["id"]), float(expire_at))   # the expiry sent, not the time the reply came
            return self._prefixes[key][0]

    async def _generate_cached(self, messages: list[Message], payload: dict) -> TextResponse:
        """``generate_text`` through the Responses API on the explicit cache of the first message: the rest of the messages
        follow the cached turn, with the same model, thinking, sampling and JSON mode as the chat payload."""
        model, thinking = str(payload["model"]), payload["thinking"]
        prefix = messages[0].content
        key = (model, thinking["type"], hashlib.sha256(prefix.encode("utf-8", "surrogatepass")).hexdigest())
        body = {"model": model, "store": False, "thinking": thinking, "temperature": payload["temperature"], "top_p": payload["top_p"],
                "input": [{"role": m.role.value, "content": m.content} for m in messages[1:]],
                "max_output_tokens": min(MAX_OUTPUT_TOKENS, int(payload.get("max_tokens") or 4096) + (THINKING_HEADROOM if thinking["type"] == "enabled" else 0))}
        if payload.get("reasoning_effort"):
            body["reasoning"] = {"effort": payload["reasoning_effort"]}
        if payload.get("response_format"):
            body["text"] = {"format": {"type": "json_object"}}
        long = thinking["type"] == "enabled" or int(payload.get("max_tokens") or 0) >= STREAM_FROM_TOKENS
        for attempt in range(2):
            try:
                body["previous_response_id"] = await self._prefix_id(key, model, thinking, prefix)
                data = await (self._post_stream if long else self._post)("/responses", body)
                break
            except AccountError as e:
                if e.status == 401 or e.code.startswith(ACCOUNT_WIDE):
                    raise
                raise PrefixCacheError(str(e), permanent=True) from e   # the chat API raises it again if the model is refused there too
            except BytePlusAPIError as e:
                if "PreviousResponseNotFound" in e.code and attempt == 0:   # deleted or expired early: cache it again
                    if self._prefixes.get(key, ("",))[0] == body["previous_response_id"]:   # not one another call just made
                        self._prefixes.pop(key, None)
                    continue
                if e.code.startswith("InvalidParameter"):
                    raise PrefixCacheError(str(e), permanent=True) from e
                if 400 <= e.status < 500 and e.status not in RETRYABLE_STATUS and not isinstance(e, ContentFilteredError) \
                        and "Sensitive" not in e.code:
                    raise PrefixCacheError(str(e)) from e
                raise   # rate limits after every retry, content refusals: the chat API would fail the same way
        usage = data.get("usage") or {}
        counted = {
            "prompt_tokens": usage.get("input_tokens", 0) or 0,
            "completion_tokens": usage.get("output_tokens", 0) or 0,
            "total_tokens": usage.get("total_tokens", 0) or 0,
            "reasoning_tokens": (usage.get("output_tokens_details") or {}).get("reasoning_tokens", 0) or 0,
            "cached_tokens": (usage.get("input_tokens_details") or {}).get("cached_tokens", 0) or 0,
        }
        self._count(model, counted)
        text = _response_text(data)
        details = data.get("incomplete_details") or {}
        reason = str(details.get("reason") or "")
        if data.get("status") == "completed":
            finish = "stop"
        elif reason in ("length", "max_output_tokens"):
            finish = "length"
        else:
            finish = "content_filter" if ("content_filter" in reason or details.get("content_filter")) else (reason or str(data.get("status") or ""))
        if finish == "content_filter" and not text.strip():
            raise ContentFilteredError(model)
        return TextResponse(content=text, model=data.get("model", model), usage=counted, finish_reason=finish)

    def _count(self, model: str, usage: dict) -> None:
        mine = self.tokens.setdefault(model, {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0})
        for key in mine:
            mine[key] += int(usage.get(key, 0) or 0)

    async def generate_text_stream(
        self,
        messages: list[Message],
        config: Optional[GenerationConfig] = None,
    ) -> AsyncIterator[str]:
        payload = self._text_payload(messages, config)
        payload["stream"] = True
        client = await self._get_client()
        async with client.stream("POST", f"{self.base_url}/chat/completions", json=payload) as response:
            if response.status_code >= 400:
                body = await response.aread()
                raise BytePlusAPIError(response.status_code, body.decode(errors="replace"))
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                chunk = line[6:].strip()
                if chunk == "[DONE]":
                    break
                try:
                    delta = json.loads(chunk)["choices"][0].get("delta", {})
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                text = delta.get("content")
                if text:
                    yield text

    # ---------------------------------------------------------------- image
    async def generate_image(
        self,
        prompt: str,
        config: Optional[ImageGenerationConfig] = None,
        reference_images: list[ImageContent] = None,
    ) -> ImageResponse:
        """Generate exactly one image with Seedream.

        ``config.width/height`` are clamped into Seedream's pixel range. Optional per-call knobs are read
        from ``config.extra`` (dict): ``fast`` (bool), ``output_format`` ("png"|"jpeg"), ``watermark`` (bool),
        ``size`` (explicit override such as "2K").
        """
        config = config or ImageGenerationConfig()
        extra = getattr(config, "extra", None) or {}

        fmt = str(extra.get("output_format") or "jpeg").lower().strip(".")
        payload = {
            "model": self.image_model,
            "prompt": prompt,
            "size": extra.get("size") or normalize_size(config.width, config.height),
            "response_format": "b64_json",
            "watermark": bool(extra.get("watermark", False)),
            "output_format": "png" if fmt == "png" else "jpeg",
        }
        if extra.get("fast"):
            payload["optimize_prompt_options"] = {"mode": "fast"}

        refs = []
        for img in (reference_images or [])[:SEEDREAM_MAX_REFERENCE_IMAGES]:
            if img.is_url:
                refs.append(img.data)
            else:
                refs.append(f"data:{img.mime_type};base64,{img.data}")
        if refs:
            payload["image"] = refs if len(refs) > 1 else refs[0]

        data = await self._post("/images/generations", payload)
        if data.get("error"):
            raise RuntimeError(f"Seedream error: {data['error']}")

        images = []
        for item in data.get("data", []):
            fmt = item.get("output_format") or payload["output_format"]
            mime = "image/png" if fmt == "png" else "image/jpeg"
            if item.get("b64_json"):
                images.append(ImageContent.from_base64(item["b64_json"], mime))
            elif item.get("url"):
                images.append(ImageContent.from_url(item["url"], mime))
        if not images:
            raise ValueError(f"Seedream returned no image: {str(data)[:300]}")

        response = ImageResponse(images=images, model=data.get("model", self.image_model), prompt=prompt)
        # expose usage for cost accounting without changing the dataclass contract
        setattr(response, "usage", data.get("usage", {}))
        setattr(response, "size", data["data"][0].get("size"))
        return response

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            for rid, _expiry in list(self._prefixes.values()):   # stop the caches' storage (they would expire within the hour)
                try:
                    await self._client.delete(f"{self.base_url}/responses/{rid}", timeout=10.0)
                except Exception:  # noqa: BLE001
                    pass
            self._prefixes.clear()
            await self._client.aclose()
            self._client = None
