"""engines.byteplus: size clamping, payload building and HTTP retry logic (``_post`` is always faked)."""

import json
from types import SimpleNamespace

import httpx
import pytest

from engines import byteplus
from engines.base import GenerationConfig, ImageContent, ImageGenerationConfig, Message
from engines.byteplus import (
    SEEDREAM_MAX_PIXELS,
    SEEDREAM_MIN_PIXELS,
    BytePlusAPIError,
    BytePlusEngine,
    normalize_size,
)

from .fakes import TINY_PNG_B64

FAKE_KEY = "test-key-not-real"


@pytest.fixture
def engine():
    return BytePlusEngine(api_key=FAKE_KEY)


@pytest.fixture
def seedream_calls(engine, monkeypatch):
    calls = []

    async def fake_post(path, payload):
        calls.append((path, payload))
        return {"data": [{"b64_json": TINY_PNG_B64, "size": "1024x1536", "output_format": "jpeg"}], "usage": {"output_tokens": 6144}}

    monkeypatch.setattr(engine, "_post", fake_post)
    return calls


# ----------------------------------------------------------------------------- sizes

@pytest.mark.parametrize("width, height", [(1024, 1536), (2048, 1024)])
def test_normalize_size_keeps_supported_sizes(width, height):
    assert normalize_size(width, height) == f"{width}x{height}"


@pytest.mark.parametrize("width, height", [(800, 800), (3000, 3000), (16, 16), (100, 9000), (5000, 1200), (961, 961)])
def test_normalize_size_clamps_into_the_seedream_pixel_range(width, height):
    w, h = map(int, normalize_size(width, height).split("x"))
    assert SEEDREAM_MIN_PIXELS <= w * h <= SEEDREAM_MAX_PIXELS
    assert w % 16 == 0 and h % 16 == 0
    if 1 / 16 <= width / height <= 16:
        assert abs((w / h) / (width / height) - 1) < 0.2  # aspect ratio roughly kept
    else:
        assert 1 / 16 <= w / h <= 16  # extreme ratios are clamped into Seedream's allowed range
    if width * height < SEEDREAM_MIN_PIXELS:
        assert w * h >= 921_600
    if width * height > SEEDREAM_MAX_PIXELS:
        assert w * h <= 4_624_220


@pytest.mark.parametrize("key", ["", "   ", "${ARK_API_KEY}", None])
def test_engine_requires_an_api_key(key):
    with pytest.raises(ValueError, match="API key"):
        BytePlusEngine(api_key=key)


# ----------------------------------------------------------------------------- text

def test_text_payload_defaults_to_thinking_disabled(engine):
    payload = engine._text_payload([Message.system("sys"), Message.user("hi")], None)
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["model"] == byteplus.DEFAULT_TEXT_MODEL
    assert payload["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    assert (payload["temperature"], payload["max_tokens"], payload["top_p"]) == (0.7, 4096, 0.95)
    assert "response_format" not in payload and "stop" not in payload


def test_text_payload_passes_overrides(engine):
    config = GenerationConfig(temperature=0.2, max_tokens=8000, stop_sequences=["END"])
    config.thinking = True
    config.model = "dola-seed-2-1-turbo-260628"
    config.response_format = {"type": "json_schema", "json_schema": {"name": "out", "schema": {"type": "object"}, "strict": True}}
    payload = engine._text_payload([Message.user("hi")], config)
    assert payload["thinking"] == {"type": "enabled"} and payload["model"] == "dola-seed-2-1-turbo-260628"
    assert payload["response_format"] == config.response_format and payload["stop"] == ["END"]
    # an engine-level default applies when the call leaves thinking unset
    assert BytePlusEngine(api_key=FAKE_KEY, thinking=True)._text_payload([Message.user("x")], GenerationConfig())["thinking"] == {"type": "enabled"}


def test_vision_messages_become_image_url_parts(engine):
    message = Message.user("Judge this.", images=[ImageContent.from_base64("QUJD", "image/png"), ImageContent.from_url("https://example.invalid/a.jpg")])
    assert engine._build_messages([message]) == [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}},
        {"type": "image_url", "image_url": {"url": "https://example.invalid/a.jpg"}},
        {"type": "text", "text": "Judge this."},
    ]}]


async def test_generate_text_maps_the_response_and_usage(engine, monkeypatch):
    seen = []

    async def fake_post(path, payload):
        seen.append((path, payload))
        return {"model": "seed-2-0-pro-260328", "choices": [{"message": {"content": '{"ok": true}'}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18, "completion_tokens_details": {"reasoning_tokens": 3}}}

    monkeypatch.setattr(engine, "_post", fake_post)
    response = await engine.generate_text([Message.user("hi")])
    assert seen[0][0] == "/chat/completions"
    assert (response.content, response.finish_reason, response.model) == ('{"ok": true}', "stop", "seed-2-0-pro-260328")
    assert response.usage == {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18, "reasoning_tokens": 3, "cached_tokens": 0}
    # the engine counts what each model spent (the job report prices models apart); cached prompt tokens are read too
    async def cached_post(path, payload):
        return {"model": "x", "choices": [{"message": {"content": "{}"}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 5, "prompt_tokens_details": {"cached_tokens": 60}}}

    monkeypatch.setattr(engine, "_post", cached_post)
    await engine.generate_text([Message.user("again")])
    assert engine.tokens == {engine.model: {"prompt_tokens": 111, "completion_tokens": 12, "cached_tokens": 60}}


# ----------------------------------------------------------------------------- images

async def test_generate_image_payload_with_two_references(engine, seedream_calls):
    refs = [ImageContent.from_base64("QUFB", "image/png"), ImageContent.from_base64("QkJC", "image/png")]
    response = await engine.generate_image("draw papi", ImageGenerationConfig(width=1024, height=1536), reference_images=refs)
    (path, payload), = seedream_calls
    assert path == "/images/generations"
    assert payload == {
        "model": byteplus.DEFAULT_IMAGE_MODEL, "prompt": "draw papi", "size": "1024x1536", "response_format": "b64_json",
        "watermark": False, "output_format": "jpeg", "image": ["data:image/png;base64,QUFB", "data:image/png;base64,QkJC"],
    }
    assert response.images[0].data == TINY_PNG_B64 and response.images[0].mime_type == "image/jpeg"
    assert response.usage == {"output_tokens": 6144} and response.size == "1024x1536"


async def test_generate_image_payload_with_one_reference_in_fast_mode(engine, seedream_calls):
    config = ImageGenerationConfig(width=2048, height=1024)
    config.extra = {"fast": True, "output_format": "png", "watermark": False}
    await engine.generate_image("strip", config, reference_images=[ImageContent.from_url("https://example.invalid/sheet.png")])
    payload = seedream_calls[0][1]
    assert payload["image"] == "https://example.invalid/sheet.png"  # a single reference is sent as a string
    assert payload["optimize_prompt_options"] == {"mode": "fast"}
    assert (payload["size"], payload["output_format"], payload["watermark"], payload["response_format"]) == ("2048x1024", "png", False, "b64_json")


async def test_generate_image_without_refs_explicit_size_and_reference_cap(engine, seedream_calls):
    config = ImageGenerationConfig(width=800, height=800)
    config.extra = {"size": "2K"}
    await engine.generate_image("no refs", config)
    payload = seedream_calls[0][1]
    assert payload["size"] == "2K" and "image" not in payload and "optimize_prompt_options" not in payload

    await engine.generate_image("many refs", reference_images=[ImageContent.from_url(f"https://example.invalid/{i}.png") for i in range(12)])
    assert len(seedream_calls[1][1]["image"]) == byteplus.SEEDREAM_MAX_REFERENCE_IMAGES == 10


async def test_generate_image_surfaces_api_errors(engine, monkeypatch):
    async def error_reply(path, payload):
        return {"error": {"code": "InvalidParameter", "message": "bad size"}}

    monkeypatch.setattr(engine, "_post", error_reply)
    with pytest.raises(RuntimeError, match="Seedream error"):
        await engine.generate_image("x")

    async def empty_reply(path, payload):
        return {"data": []}

    monkeypatch.setattr(engine, "_post", empty_reply)
    with pytest.raises(ValueError, match="no image"):
        await engine.generate_image("x")


# ----------------------------------------------------------------------------- HTTP (mock transport only)

async def test_post_retries_transient_errors_and_fails_fast_on_client_errors(engine, monkeypatch):
    statuses, requests, sleeps = [429, 503, 200], [], []

    def handler(request):
        requests.append(request)
        status = statuses.pop(0)
        return httpx.Response(status, json={"ok": True} if status == 200 else {"error": "busy"})

    async def no_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(byteplus, "asyncio", SimpleNamespace(sleep=no_sleep))
    engine._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    assert await engine._post("/chat/completions", {"x": 1}) == {"ok": True}
    assert len(requests) == 3 and len(sleeps) == 2
    assert 3 <= sleeps[0] <= 5 and 6 <= sleeps[1] <= 8  # exponential backoff with jitter
    assert str(requests[0].url) == f"{byteplus.DEFAULT_BASE_URL}/chat/completions"
    assert json.loads(requests[0].content) == {"x": 1}

    statuses[:] = [400]
    with pytest.raises(BytePlusAPIError) as excinfo:
        await engine._post("/chat/completions", {})
    assert excinfo.value.status == 400 and len(requests) == 4 and len(sleeps) == 2  # no retry
    await engine.close()


async def test_an_unpatched_engine_cannot_reach_the_network():
    engine = BytePlusEngine(api_key=FAKE_KEY, max_retries=0)
    with pytest.raises(RuntimeError, match="Network access is disabled"):
        await engine.generate_text([Message.user("hi")])
    await engine.close()


async def test_long_calls_stream_and_assemble_the_reply(engine, monkeypatch):
    """A thinking call (or one allowed a long reply) is streamed; the chunks become one reply with the usage of the last
    chunk; a stream that breaks off is retried from the start."""
    from engines.base import GenerationConfig
    sse = "\n".join([
        'data: {"model": "deepseek-v4-pro-ga-260813", "choices": [{"delta": {"role": "assistant", "reasoning_content": "hmm"}}]}',
        'data: {"choices": [{"delta": {"content": "{\\"ok\\""}}]}',
        'data: {"choices": [{"delta": {"content": ": true}"}, "finish_reason": "stop"}]}',
        'data: {"choices": [], "usage": {"prompt_tokens": 40, "completion_tokens": 9, "prompt_tokens_details": {"cached_tokens": 30},'
        ' "completion_tokens_details": {"reasoning_tokens": 4}}}',
        "data: [DONE]", ""])
    calls, sleeps = [], []

    def handler(request):
        calls.append(json.loads(request.content))
        if len(calls) == 1:
            return httpx.Response(503, json={"error": "busy"})
        return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})

    async def no_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(byteplus, "asyncio", SimpleNamespace(sleep=no_sleep))
    engine._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    config = GenerationConfig(max_tokens=24000)
    config.thinking = True
    response = await engine.generate_text([Message.user("think")], config)
    assert (response.content, response.finish_reason, response.model) == ('{"ok": true}', "stop", "deepseek-v4-pro-ga-260813")
    assert response.usage["cached_tokens"] == 30 and response.usage["reasoning_tokens"] == 4
    assert calls[1]["stream"] is True and calls[1]["stream_options"] == {"include_usage": True} and len(sleeps) == 1
    await engine.close()


def test_reasoning_effort_goes_with_thinking_only(engine):
    from engines.base import GenerationConfig
    config = GenerationConfig()
    config.thinking = True
    engine.reasoning_effort = "low"
    assert engine._text_payload([Message.user("x")], config)["reasoning_effort"] == "low"
    config.reasoning_effort = "high"                                   # a call may ask for more
    assert engine._text_payload([Message.user("x")], config)["reasoning_effort"] == "high"
    config.thinking = False                                           # no thinking: the API refuses an effort
    assert "reasoning_effort" not in engine._text_payload([Message.user("x")], config)


@pytest.mark.parametrize("status,code,account", [
    (404, "ModelNotOpen", True), (403, "AccountOverdueError", True), (429, "QuotaExceeded", True), (429, "SetLimitExceeded", True),
    (401, "AuthenticationError", True), (403, "AccessDenied", True), (429, "RateLimitExceeded", False),
    (429, "ModelAccountTpmRateLimitExceeded", False), (503, "ServerOverloaded", False), (400, "InvalidParameter", False)])
async def test_account_errors_are_raised_at_once_with_what_to_do(engine, monkeypatch, status, code, account):
    """A refused key, an inactive model, an overdue account, a used-up quota or a spending limit never get better by
    waiting: one request, then an AccountError that says what to do. Rate limits are still retried."""
    requests, sleeps = [], []

    def handler(request):
        requests.append(request)
        return httpx.Response(status, json={"error": {"code": code, "message": f"{code} message"}})

    async def no_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(byteplus, "asyncio", SimpleNamespace(sleep=no_sleep))
    engine._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with pytest.raises(BytePlusAPIError) as excinfo:
        await engine._post("/images/generations", {"model": "dola-seedream-5-0-pro-260628"})
    assert isinstance(excinfo.value, byteplus.AccountError) is account
    if account:
        assert len(requests) == 1 and excinfo.value.code == code
        assert "ModelArk" in str(excinfo.value) and " / " in excinfo.value.hint   # English and Chinese
        if code == "ModelNotOpen":
            assert "dola-seedream-5-0-pro-260628" in excinfo.value.hint and "activate" in excinfo.value.hint
    else:
        assert len(requests) == (engine.max_retries + 1 if status in byteplus.RETRYABLE_STATUS else 1)
    await engine.close()


async def test_a_content_filter_refusal_is_an_error_not_an_empty_answer(engine, monkeypatch):
    async def filtered(path, payload):
        return {"model": payload["model"], "choices": [{"message": {"content": ""}, "finish_reason": "content_filter"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0}}

    monkeypatch.setattr(engine, "_post", filtered)
    with pytest.raises(byteplus.ContentFilteredError) as excinfo:
        await engine.generate_text([Message.user("a paper that trips the filter")])
    assert excinfo.value.code == "ContentFilter" and not getattr(excinfo.value, "account", False)


async def test_an_ssl_error_mid_reply_is_retried(engine, monkeypatch):
    import ssl
    calls, sleeps = [], []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            raise ssl.SSLError("[SSL] record layer failure")
        return httpx.Response(200, json={"ok": True})

    async def no_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(byteplus, "asyncio", SimpleNamespace(sleep=no_sleep))
    engine._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    assert await engine._post("/chat/completions", {"x": 1}) == {"ok": True}
    assert len(calls) == 2 and len(sleeps) == 1
    await engine.close()


# ------------------------------------------------------------------ explicit prefix cache (Responses API)

PAPER_HEAD = "You work on one research paper. PAPER TEXT:\n" + "The Transformer uses attention only. " * 200


def cached_config(thinking=False, **kw):
    config = GenerationConfig(temperature=0.3, max_tokens=12000, **kw)
    config.thinking = thinking
    config.cache_prefix = True
    return config


def paper_messages(task="Fix panel 3."):
    return [Message.system(PAPER_HEAD), Message.system("You write manga."), Message.user(task)]


@pytest.fixture
def responses_api(engine, monkeypatch):
    """Fakes _post and _post_stream: the cache creations, the cached calls and the chat calls each engine call makes."""
    log = {"create": [], "call": [], "chat": []}

    async def fake(path, payload):
        if path == "/chat/completions":
            log["chat"].append(payload)
            return {"model": payload["model"], "choices": [{"message": {"content": "chat"}, "finish_reason": "stop"}], "usage": {}}
        assert path == "/responses"
        if payload.get("caching"):
            log["create"].append(payload)
            return {"id": f"resp_prefix_{len(log['create'])}", "status": "completed", "output": [],
                    "usage": {"input_tokens": 5000, "output_tokens": 0, "input_tokens_details": {"cached_tokens": 0}}}
        log["call"].append(payload)
        return {"id": "resp_x", "model": payload["model"], "status": "completed",
                "output": [{"type": "reasoning", "summary": []},
                           {"type": "message", "content": [{"type": "output_text", "text": '{"panels": []}'}]}],
                "usage": {"input_tokens": 5040, "output_tokens": 30, "total_tokens": 5070,
                          "input_tokens_details": {"cached_tokens": 5000}, "output_tokens_details": {"reasoning_tokens": 0}}}

    monkeypatch.setattr(engine, "_post", fake)
    monkeypatch.setattr(engine, "_post_stream", fake)
    return log


async def test_a_paper_first_seed_call_reads_the_paper_from_an_explicit_cache(engine, responses_api):
    # Seed's implicit cache keeps a prefix 30-60 s: the paper-first calls of a job, minutes apart, missed it
    first = await engine.generate_text(paper_messages(), cached_config(response_format={"type": "json_object"}))
    await engine.generate_text(paper_messages("Fix panel 5."), cached_config())
    assert (first.content, first.finish_reason, first.usage["cached_tokens"], first.usage["prompt_tokens"]) == ('{"panels": []}', "stop", 5000, 5040)
    [create] = responses_api["create"]                                   # one cache for both calls
    assert create["input"] == [{"role": "system", "content": PAPER_HEAD}] and create["caching"] == {"type": "enabled", "prefix": True}
    assert create["thinking"] == {"type": "disabled"} and create["expire_at"] > 0
    call = responses_api["call"][0]
    assert call["previous_response_id"] == "resp_prefix_1" and call["store"] is False and call["max_output_tokens"] == 12000
    assert call["input"] == [{"role": "system", "content": "You write manga."}, {"role": "user", "content": "Fix panel 3."}]
    assert call["text"] == {"format": {"type": "json_object"}} and call["temperature"] == 0.3
    assert responses_api["chat"] == []
    assert engine.tokens["seed-2-0-pro-260328"] == {"prompt_tokens": 5000 + 2 * 5040, "completion_tokens": 60, "cached_tokens": 10000}


async def test_a_thinking_call_gets_its_own_cache_and_room_to_reason(engine, responses_api):
    # a cached head serves only calls with its thinking setting; max_output_tokens counts the reasoning, max_tokens did not
    await engine.generate_text(paper_messages(), cached_config())
    await engine.generate_text(paper_messages(), cached_config(thinking=True))
    assert [c["thinking"]["type"] for c in responses_api["create"]] == ["disabled", "enabled"]
    assert responses_api["call"][1]["previous_response_id"] == "resp_prefix_2"
    assert responses_api["call"][1]["max_output_tokens"] == 12000 + byteplus.THINKING_HEADROOM


@pytest.mark.parametrize("change", ["deepseek", "no_flag", "json_schema", "image", "short", "single"])
async def test_calls_the_explicit_cache_cannot_serve_go_to_the_chat_api(engine, responses_api, change):
    messages, config = paper_messages(), cached_config()
    if change == "deepseek":
        config.model = "deepseek-v4-pro-ga-260813"                        # its implicit cache already hits 93%
    elif change == "no_flag":
        config.cache_prefix = False
    elif change == "json_schema":
        config.response_format = {"type": "json_schema", "json_schema": {"name": "out", "schema": {"type": "object"}}}
    elif change == "image":
        messages[2] = Message.user("Judge this.", images=[ImageContent.from_base64(TINY_PNG_B64, "image/png")])
    elif change == "short":
        messages[0] = Message.system("A short system prompt.")
    else:
        messages = messages[:1]
    await engine.generate_text(messages, config)
    assert len(responses_api["chat"]) == 1 and responses_api["create"] == [] and responses_api["call"] == []


async def test_an_expired_cache_is_made_again_once(engine, responses_api, monkeypatch):
    real = engine._post
    gone = []

    async def post(path, payload):
        if path == "/responses" and payload.get("previous_response_id") == "resp_prefix_1":
            gone.append(1)
            raise BytePlusAPIError(400, json.dumps({"error": {"code": "InvalidParameter.PreviousResponseNotFound", "message": "not found"}}))
        return await real(path, payload)

    monkeypatch.setattr(engine, "_post", post)
    monkeypatch.setattr(engine, "_post_stream", post)   # a 12,000-token reply is streamed
    response = await engine.generate_text(paper_messages(), cached_config())
    assert response.content == '{"panels": []}' and len(gone) == 1 and len(responses_api["create"]) == 2
    assert responses_api["call"][0]["previous_response_id"] == "resp_prefix_2"


async def test_a_refused_cached_call_falls_back_to_the_chat_api_for_good(engine, responses_api, monkeypatch):
    real = engine._post

    async def post(path, payload):
        if path == "/responses" and payload.get("previous_response_id"):
            raise BytePlusAPIError(400, json.dumps({"error": {"code": "InvalidParameter", "message": "top_p is not supported"}}))
        return await real(path, payload)

    monkeypatch.setattr(engine, "_post", post)
    monkeypatch.setattr(engine, "_post_stream", post)
    assert (await engine.generate_text(paper_messages(), cached_config())).content == "chat"
    await engine.generate_text(paper_messages(), cached_config())
    assert len(responses_api["chat"]) == 2 and len(responses_api["create"]) == 1   # the second call skips the cache


async def test_account_errors_and_filter_refusals_on_the_cached_path_are_not_swallowed(engine, responses_api, monkeypatch):
    async def overdue(path, payload):
        raise byteplus.AccountError(403, json.dumps({"error": {"code": "AccountOverdueError", "message": "overdue"}}))

    monkeypatch.setattr(engine, "_post", overdue)
    monkeypatch.setattr(engine, "_post_stream", overdue)
    with pytest.raises(byteplus.AccountError):
        await engine.generate_text(paper_messages(), cached_config())

    async def filtered(path, payload):
        if payload.get("caching"):
            return {"id": "resp_prefix", "usage": {}}
        return {"status": "incomplete", "incomplete_details": {"reason": "content_filter"}, "output": [], "usage": {}}

    monkeypatch.setattr(engine, "_post", filtered)
    monkeypatch.setattr(engine, "_post_stream", filtered)
    with pytest.raises(byteplus.ContentFilteredError):
        await engine.generate_text(paper_messages(), cached_config())
    assert responses_api["chat"] == []


async def test_a_streamed_cached_reply_is_assembled_and_close_deletes_the_caches(engine):
    sse = "\n".join([
        "event: response.created", 'data: {"type": "response.created", "response": {"id": "resp_1", "status": "in_progress"}}',
        'data: {"type": "response.reasoning_summary_text.delta", "delta": "hmm"}',
        'data: {"type": "response.output_text.delta", "delta": "{\\"ok\\""}',
        'data: {"type": "response.output_text.delta", "delta": ": true}"}',
        'data: {"type": "response.completed", "response": {"id": "resp_1", "model": "seed-2-0-pro-260328", "status": "completed", '
        '"output": [{"type": "message", "content": [{"type": "output_text", "text": "{\\"ok\\": true}"}]}], '
        '"usage": {"input_tokens": 900, "output_tokens": 20, "input_tokens_details": {"cached_tokens": 880}, '
        '"output_tokens_details": {"reasoning_tokens": 12}}}}',
        "data: [DONE]", ""])
    requests = []

    def handler(request):
        requests.append((request.method, request.url.path, json.loads(request.content) if request.content else None))
        if request.method == "DELETE":
            return httpx.Response(200, json={})
        if requests[-1][2].get("caching"):
            return httpx.Response(200, json={"id": "resp_prefix_9", "usage": {"input_tokens": 880}})
        return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})

    engine._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    response = await engine.generate_text(paper_messages(), cached_config(thinking=True))
    assert (response.content, response.finish_reason, response.usage["cached_tokens"], response.usage["reasoning_tokens"]) == \
        ('{"ok": true}', "stop", 880, 12)
    assert requests[1][2]["stream"] is True and "stream_options" not in requests[1][2]
    await engine.close()
    assert requests[-1][:2] == ("DELETE", "/api/v3/responses/resp_prefix_9")


async def test_a_rate_limit_on_a_cached_call_is_raised_and_keeps_the_cache_for_later_calls(engine, responses_api, monkeypatch):
    # a 429 after every retry fails the chat API alike; it must not switch the model's cache off for the rest of the job
    real, limited = engine._post, []

    async def post(path, payload):
        if path == "/responses" and payload.get("previous_response_id") and not limited:
            limited.append(1)
            raise BytePlusAPIError(429, json.dumps({"error": {"code": "ModelAccountTpmRateLimitExceeded", "message": "slow down"}}))
        return await real(path, payload)

    monkeypatch.setattr(engine, "_post", post)
    monkeypatch.setattr(engine, "_post_stream", post)
    with pytest.raises(BytePlusAPIError):
        await engine.generate_text(paper_messages(), cached_config())
    await engine.generate_text(paper_messages(), cached_config())
    assert engine._prefix_off == set() and len(responses_api["call"]) == 1 and responses_api["chat"] == []


async def test_a_cache_that_cannot_be_created_sends_calls_to_the_chat_api_for_a_while(engine, responses_api, monkeypatch):
    real, creations = engine._post, []

    async def post(path, payload):
        if payload.get("caching"):
            creations.append(1)
            raise RuntimeError("ModelArk /responses unreachable after 7 attempts")
        return await real(path, payload)

    monkeypatch.setattr(engine, "_post", post)
    monkeypatch.setattr(engine, "_post_stream", post)
    for _ in range(3):
        assert (await engine.generate_text(paper_messages(), cached_config())).content == "chat"
    assert len(creations) == 1 and len(responses_api["chat"]) == 3 and engine._prefix_off == set()


async def test_a_late_not_found_never_drops_a_cache_another_call_just_made(engine, responses_api, monkeypatch):
    real = engine._post
    engine._prefixes[("seed-2-0-pro-260328", "disabled", __import__("hashlib").sha256(PAPER_HEAD.encode()).hexdigest())] = \
        ("resp_stale", __import__("time").time() + 3000)

    async def post(path, payload):
        if payload.get("previous_response_id") == "resp_stale":
            if payload["input"][-1]["content"] == "slow":
                await __import__("asyncio").sleep(0.05)   # its 400 comes back after the other call made a new cache
            raise BytePlusAPIError(400, json.dumps({"error": {"code": "InvalidParameter.PreviousResponseNotFound", "message": "gone"}}))
        return await real(path, payload)

    monkeypatch.setattr(engine, "_post", post)
    monkeypatch.setattr(engine, "_post_stream", post)
    import asyncio as _asyncio
    await _asyncio.gather(engine.generate_text(paper_messages("fast"), cached_config()),
                          engine.generate_text(paper_messages("slow"), cached_config()))
    assert len(responses_api["create"]) == 1 and {c["previous_response_id"] for c in responses_api["call"]} == {"resp_prefix_1"}


async def test_a_model_refused_on_the_responses_endpoint_falls_back_to_chat(engine, responses_api, monkeypatch):
    real = engine._post

    async def post(path, payload):
        if path == "/responses" and payload.get("previous_response_id"):
            raise byteplus.AccountError(403, json.dumps({"error": {"code": "AccessDenied", "message": "not on this endpoint"}}), payload["model"])
        return await real(path, payload)

    monkeypatch.setattr(engine, "_post", post)
    monkeypatch.setattr(engine, "_post_stream", post)
    assert (await engine.generate_text(paper_messages(), cached_config())).content == "chat"
    assert engine._prefix_off == {"seed-2-0-pro-260328"}


@pytest.mark.parametrize("path", ["/responses", "/chat/completions"])
async def test_a_rate_limit_inside_a_stream_is_asked_again(engine, monkeypatch, path):
    # RequestBurstTooFast cut a story call off at event 2869 and failed the whole job (2026-10-27)
    burst = ('data: {"type": "error", "code": "RequestBurstTooFast", "message": "slow down"}' if path == "/responses"
             else 'data: {"error": {"code": "RequestBurstTooFast", "message": "slow down"}}')
    good = ('data: {"type": "response.completed", "response": {"status": "completed", "output": [{"type": "message", '
            '"content": [{"type": "output_text", "text": "ok"}]}], "usage": {}}}' if path == "/responses"
            else 'data: {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}')
    sent, sleeps = [], []

    def handler(request):
        sent.append(1)
        lines = ['data: {"type": "response.created", "response": {}}', burst] if len(sent) == 1 else [good]
        return httpx.Response(200, text="\n".join(lines + ["data: [DONE]", ""]), headers={"content-type": "text/event-stream"})

    async def no_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(byteplus, "asyncio", SimpleNamespace(sleep=no_sleep))
    engine._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    data = await engine._post_stream(path, {"model": "seed-2-0-pro-260328"})
    text = byteplus._response_text(data) if path == "/responses" else data["choices"][0]["message"]["content"]
    assert text == "ok" and len(sent) == 2 and len(sleeps) == 1
    await engine.close()


async def test_an_error_event_that_retrying_cannot_fix_stops_the_call(engine, monkeypatch):
    sent = []

    def handler(request):
        sent.append(1)
        return httpx.Response(200, text='data: {"type": "error", "code": "OutputTextSensitiveContentDetected"}\n', headers={"content-type": "text/event-stream"})

    engine._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with pytest.raises(BytePlusAPIError, match="Sensitive"):
        await engine._post_stream("/responses", {"model": "seed-2-0-pro-260328"})
    assert len(sent) == 1
    await engine.close()
