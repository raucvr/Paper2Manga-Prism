"""Regression tests for review fixes W6, W7, W9 and W10 (config loading, engine selection, generation routes,
Seedream request settings).

Safety: every engine is a fake or a ``BytePlusEngine`` whose ``_post`` / HTTP transport is replaced, so nothing
reaches ModelArk (conftest's autouse ``_no_network`` also blocks real transports and sockets). Config tests build
private ``ConfigManager`` instances over ``tmp_path``. The real ``config/`` directory is never written, and the
module singleton is left alone.

Tests marked ``xfail(strict=True)`` describe the correct behaviour for problems that are still open. Once the
source is fixed they XPASS and fail the run: delete the marker then, and the test stays as a regression test.
"""

from __future__ import annotations

import asyncio
import itertools

import httpx
import pytest
import yaml
from fastapi.testclient import TestClient
from watchdog.events import FileModifiedEvent

import config_loader
import engines as engines_pkg
import main
import routes.config as config_routes
import routes.generation as generation
import services.jobs as jobs
from config_loader import ConfigChangeHandler, ConfigManager, ProviderConfig, as_bool, as_int
from engines.base import ImageGenerationConfig, Message
from engines.byteplus import (
    DEFAULT_IMAGE_MODEL,
    DEFAULT_TEXT_MODEL,
    SEEDREAM_MAX_PIXELS,
    SEEDREAM_MIN_PIXELS,
    BytePlusEngine,
    normalize_size,
)
from services import engine_select
from services.engine_select import Engines

from .fakes import TINY_PNG_B64, FakeConfig, FakeEngine, b64, image_bytes

REAL_SETTINGS = jobs._settings  # job_env swaps it for a fixed dict; W6 needs the real one
FAKE_KEY = "test-key-not-real"


# ============================================================================= W6: config values and hot reload

GOOD_YAML = """\
core_engine:
  text_model: "byteplus/seed-2-0-pro-260328"
providers:
  byteplus:
    enabled: true
    api_key: "test-key-not-real"
manga_settings:
  concurrency: 3
  qa_enabled: true
output:
  image_format: "jpeg"
"""

BAD_VALUES_YAML = """\
providers:
  byteplus:
    enabled: "true"
    api_key: 12345
  openrouter:
    enabled: "false"
    api_key: "${PRISM_TEST_UNSET_VAR}"
manga_settings:
  concurrency: "3"
  qa_enabled: "false"
  max_attempts: "0"
  panel_width: ""
  panel_height: "${PRISM_TEST_UNSET_VAR}"
  fast: "no"
  style_anchor: "yes"
  min_panels: "8"
  max_panels: 12.0
  strip_width: "1536"
output:
  image_format: "png"
  image_quality: "high"
"""

HALF_WRITTEN_YAML = "providers:\n  byteplus: {enabled: tr"  # what a reader sees in the middle of a save


@pytest.fixture
def config_file(monkeypatch, tmp_path):
    """A private config dir. ``ConfigManager.reload`` reads ``API_CONFIG_PATH`` / ``CONFIG_DIR`` at call time."""
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    path = cfg_dir / "api_config.yaml"
    monkeypatch.setattr(config_loader, "CONFIG_DIR", cfg_dir)
    monkeypatch.setattr(config_loader, "API_CONFIG_PATH", path)
    monkeypatch.delenv("PRISM_TEST_UNSET_VAR", raising=False)
    return path


def load_manager(path, text=None) -> ConfigManager:
    """A fresh ConfigManager loaded from ``path``. ``__new__`` is bypassed, so the module singleton is untouched."""
    if text is not None:
        path.write_text(text, encoding="utf-8")
    manager = object.__new__(ConfigManager)
    manager._initialized = False
    manager.__init__()
    return manager


def file_saved(manager: ConfigManager, path, text: str) -> None:
    """Write the file and deliver the event the way watchdog's observer thread does."""
    path.write_text(text, encoding="utf-8")
    ConfigChangeHandler(manager.reload).on_modified(FileModifiedEvent(str(path)))


@pytest.mark.parametrize("raw, expected", [
    ("5", 5), (" 7 ", 7), (5.9, 5), ("1e3", 1000), ("", 42), (None, 42), ("${UNSET}", 42), ("five", 42),
    ([1], 42), ({"a": 1}, 42), ("nan", 42),
])
def test_w6_as_int_tolerates_yaml_values(raw, expected):
    assert as_int(raw, 42) == expected


@pytest.mark.parametrize("raw, expected", [
    ("false", False), ("False", False), (" no ", False), ("off", False), ("0", False), (0, False),
    ("true", True), ("YES", True), ("on", True), ("1", True), (2, True),
])
def test_w6_as_bool_parses_strings_instead_of_using_truthiness(raw, expected):
    assert as_bool(raw, not expected) is expected  # the default alone would give the opposite answer


@pytest.mark.parametrize("raw", ["", None, "maybe", "${UNSET}"])
def test_w6_as_bool_falls_back_to_the_default(raw):
    assert as_bool(raw, True) is True and as_bool(raw, False) is False


def test_w6_bad_values_do_not_crash_startup_and_are_read_correctly(config_file):
    manager = load_manager(config_file, BAD_VALUES_YAML)
    byteplus, openrouter = manager.get_provider("byteplus"), manager.get_provider("openrouter")
    assert (byteplus.enabled, byteplus.api_key) == (True, "12345")
    assert (openrouter.enabled, openrouter.api_key) == (False, "")  # bool("false") used to be True
    ms = manager.manga_settings
    assert (ms.concurrency, ms.qa_enabled, ms.max_attempts, ms.fast, ms.style_anchor) == (3, False, 1, False, True)
    assert (ms.panel_width, ms.panel_height, ms.min_panels, ms.max_panels) == (1024, 1536, 8, 12)
    assert (manager.output_settings.image_format, manager.output_settings.image_quality) == ("png", 95)


def test_w6_job_settings_come_from_the_parsed_manga_settings(config_file, monkeypatch):
    manager = load_manager(config_file, BAD_VALUES_YAML)
    monkeypatch.setattr(jobs, "get_config", lambda: manager)
    assert jobs._settings() == {
        "layout": "single", "panel_width": 1024, "panel_height": 1536, "strip_width": 1536, "strip_height": 1120,
        "concurrency": 3, "qa_enabled": False, "qa_blind_check": True, "max_attempts": 1, "min_panels": 8, "max_panels": 12,
        "style_anchor": True, "fast": False, "output_format": "png", "pipeline": "v3", "bilingual": True,
        "read_figures": True,
    }


JOB_YAML = """\
providers:
  byteplus: {enabled: "true", api_key: "test-key-not-real"}
manga_settings:
  pipeline: "v2"
  layout: "single"
  concurrency: "2"
  qa_enabled: "false"
  max_attempts: "1"
  min_panels: "3"
  max_panels: "6"
  fast: "off"
output:
  image_format: "PNG"
"""


async def test_w6_string_flags_in_the_config_reach_the_job(job_env, config_file, monkeypatch):
    manager = load_manager(config_file, JOB_YAML)
    monkeypatch.setattr(jobs, "get_config", lambda: manager)
    monkeypatch.setattr(jobs, "_settings", REAL_SETTINGS)
    state = await job_env.run(qa=None, fast=None)  # None means "use the configured default"
    assert state["status"] == "completed"
    assert (state["params"]["qa"], state["params"]["fast"]) == (False, False)
    assert job_env.vision.calls == []  # QA really is off
    assert job_env.image.calls[0]["config"].extra == {"fast": False, "output_format": "PNG", "watermark": False}


def test_w6_hot_reload_with_invalid_yaml_keeps_the_previous_config(config_file):
    manager = load_manager(config_file, GOOD_YAML)
    file_saved(manager, config_file, "manga_settings:\n  concurrency: [unclosed\n")  # must not raise in the watcher
    assert manager.manga_settings.concurrency == 3 and manager.get_provider("byteplus").enabled is True
    file_saved(manager, config_file, GOOD_YAML.replace("concurrency: 3", "concurrency: 7"))
    assert manager.manga_settings.concurrency == 7  # a later valid edit is still applied


def test_w6_reload_swaps_every_section_at_once(config_file):
    manager = load_manager(config_file, GOOD_YAML)
    # manga_settings parses fine, then the output section is malformed: nothing may be half-applied
    broken = GOOD_YAML.replace("concurrency: 3", "concurrency: 9").replace('output:\n  image_format: "jpeg"\n', 'output: "jpeg"\n')
    file_saved(manager, config_file, broken)
    assert manager.manga_settings.concurrency == 3 and manager.output_settings.image_format == "jpeg"


def test_w6_a_failing_on_change_callback_does_not_abort_the_reload(config_file):
    manager = load_manager(config_file, GOOD_YAML)
    seen = []

    def broken_callback(_config):
        raise RuntimeError("callback bug")

    manager.on_change(broken_callback)
    manager.on_change(lambda cfg: seen.append(cfg.manga_settings.concurrency))
    file_saved(manager, config_file, GOOD_YAML.replace("concurrency: 3", "concurrency: 4"))
    assert seen == [4] and manager.manga_settings.concurrency == 4


def test_w6_first_load_of_a_broken_file_still_fails_loudly(config_file):
    config_file.write_text("providers: {byteplus: [", encoding="utf-8")
    with pytest.raises(yaml.YAMLError):
        load_manager(config_file)


def test_w6_truncated_file_during_hot_reload_keeps_the_previous_config(config_file):
    manager = load_manager(config_file, GOOD_YAML)
    file_saved(manager, config_file, "")
    assert manager.get_provider("byteplus") is not None and manager.get_provider("byteplus").enabled
    assert manager.manga_settings.concurrency == 3


@pytest.mark.parametrize("start", ["file-truncated-mid-edit", "comments-only-at-startup"])
def test_w6_reload_after_a_successful_load_never_raises_into_the_watchdog_thread(config_file, start):
    if start == "file-truncated-mid-edit":
        manager = load_manager(config_file, GOOD_YAML)
        file_saved(manager, config_file, "")
    else:
        try:
            manager = load_manager(config_file, "# filled in later\n")
        except Exception:  # rejecting an empty config at startup also closes this path
            return
    file_saved(manager, config_file, HALF_WRITTEN_YAML)  # raises today
    file_saved(manager, config_file, GOOD_YAML)
    assert manager.get_provider("byteplus").enabled and manager.manga_settings.concurrency == 3


def test_w6_infinite_numbers_fall_back_to_the_default(config_file):
    assert as_int(float("inf"), 5) == 5 and as_int("1e400", 5) == 5
    manager = load_manager(config_file, GOOD_YAML.replace("concurrency: 3", "concurrency: .inf"))
    assert manager.manga_settings.concurrency == 5


def test_w6_reload_endpoint_reports_a_failed_reload(config_file, monkeypatch):
    manager = load_manager(config_file, GOOD_YAML)

    async def no_reset():
        return None

    monkeypatch.setattr(config_routes, "get_config", lambda: manager)
    monkeypatch.setattr(config_routes, "reset_client", no_reset)
    config_file.write_text("manga_settings: {concurrency: [", encoding="utf-8")
    client = TestClient(main.app)  # no lifespan: no watcher, no engines
    try:
        response = client.post("/api/config/reload")
    finally:
        client.close()
    assert manager.manga_settings.concurrency == 3
    assert response.status_code >= 400 or response.json().get("status") != "success"


# ============================================================================= W7: legacy model ids vs ModelArk

BYTEPLUS = ProviderConfig(name="byteplus", enabled=True, api_key=FAKE_KEY, base_url="https://ark.example.invalid/api/v3")
OPENROUTER = ProviderConfig(name="openrouter", enabled=True, api_key=FAKE_KEY, base_url="https://openrouter.example.invalid/api/v1")


def use_config(monkeypatch, core=None, **providers):
    monkeypatch.setattr(engine_select, "get_config", lambda: FakeConfig(core or {}, providers))


@pytest.mark.parametrize("image", [False, True], ids=["text", "image"])
@pytest.mark.parametrize("spec", [
    "google/gemini-3-pro-image-preview", "nano-banana-pro", "gemini-2.5-flash", "gemini-3-pro-image-preview",
    "gpt-4o", "byteplus/google/gemini-x",
])
def test_w7_ids_modelark_cannot_serve_fall_back_to_the_defaults(monkeypatch, capsys, spec, image):
    use_config(monkeypatch, byteplus=BYTEPLUS)
    engine = engine_select._engine_for(spec, image=image)
    assert isinstance(engine, BytePlusEngine)
    assert (engine.model, engine.image_model) == (DEFAULT_TEXT_MODEL, DEFAULT_IMAGE_MODEL)
    out = capsys.readouterr().out
    assert "is not a ModelArk model id" in out and spec in out


@pytest.mark.parametrize("spec, image, text_model, image_model", [
    ("seed-2-0-pro-260328", False, "seed-2-0-pro-260328", DEFAULT_IMAGE_MODEL),
    ("dola-seed-2-1-turbo-260628", False, "dola-seed-2-1-turbo-260628", DEFAULT_IMAGE_MODEL),
    ("ep-20260101123456-abcde", False, "ep-20260101123456-abcde", DEFAULT_IMAGE_MODEL),
    ("deepseek-v3-1-250821", False, "deepseek-v3-1-250821", DEFAULT_IMAGE_MODEL),
    ("dola-seedream-5-0-flash-260915", True, DEFAULT_TEXT_MODEL, "dola-seedream-5-0-flash-260915"),
    ("seedream-4-5-251128", True, DEFAULT_TEXT_MODEL, "seedream-4-5-251128"),
    ("byteplus/dola-seedream-5-0-pro-260628", True, DEFAULT_TEXT_MODEL, "dola-seedream-5-0-pro-260628"),
    ("byteplus/seed-2-0-lite-260428", False, "seed-2-0-lite-260428", DEFAULT_IMAGE_MODEL),
])
def test_w7_modelark_ids_are_kept_without_a_warning(monkeypatch, capsys, spec, image, text_model, image_model):
    use_config(monkeypatch, byteplus=BYTEPLUS)
    engine = engine_select._engine_for(spec, image=image)
    assert (engine.model, engine.image_model) == (text_model, image_model)
    assert "WARNING" not in capsys.readouterr().out


async def test_w7_legacy_core_engine_sends_only_modelark_ids(monkeypatch, capsys):
    core = {"default_model": "google/gemini-3-pro-image-preview", "image_model": "google/gemini-3-pro-image-preview"}
    use_config(monkeypatch, core, byteplus=BYTEPLUS)
    built = engine_select.build_engines()
    assert all(isinstance(e, BytePlusEngine) for e in (built.text, built.fast_text, built.vision, built.image))
    assert "google/gemini-3-pro-image-preview" in capsys.readouterr().out

    sent = []

    async def fake_post(path, payload):
        sent.append((path, payload["model"]))
        if path == "/chat/completions":
            return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
        return {"data": [{"b64_json": TINY_PNG_B64}]}

    for engine in {id(e): e for e in (built.text, built.fast_text, built.vision, built.image)}.values():
        monkeypatch.setattr(engine, "_post", fake_post)
    await built.text.generate_text([Message.user("hi")])
    await built.vision.generate_text([Message.user("judge")])
    await built.image.generate_image("draw")
    assert sent == [("/chat/completions", DEFAULT_TEXT_MODEL), ("/chat/completions", DEFAULT_TEXT_MODEL),
                    ("/images/generations", DEFAULT_IMAGE_MODEL)]


def test_w7_legacy_ids_still_go_to_openrouter_when_it_is_enabled(monkeypatch):
    calls = []
    monkeypatch.setattr(engine_select, "create_engine", lambda provider, model=None, **kw: calls.append((provider, model)) or FakeEngine())
    core = {"default_model": "google/gemini-3-pro-image-preview", "image_model": "google/gemini-3-pro-image-preview"}
    use_config(monkeypatch, core, byteplus=BYTEPLUS, openrouter=OPENROUTER)
    built = engine_select.build_engines()
    assert calls == [("openrouter", "google/gemini-3-pro-image-preview")] * 2  # one text engine (shared), one image engine
    assert not any(isinstance(e, BytePlusEngine) for e in (built.text, built.fast_text, built.vision, built.image))


# ============================================================================= W9: engine / HTTP-client leaks

class StreamingFake(FakeEngine):
    """Streams ``chunks``. With ``fail_at=i`` the upstream raises instead of yielding chunk ``i``."""

    def __init__(self, chunks, fail_at=None):
        super().__init__()
        self.chunks, self.fail_at = list(chunks), fail_at

    async def generate_text_stream(self, messages, config=None):
        for i, chunk in enumerate(self.chunks):
            if i == self.fail_at:
                raise RuntimeError("upstream stream broke")
            await asyncio.sleep(0)
            yield chunk


@pytest.fixture
def generate_api_on(monkeypatch):
    """/api/generate/* is off unless server.enable_generate_api is set; these route tests need it on."""
    monkeypatch.setitem(main.app.dependency_overrides, generation.require_generate_api, lambda: None)


async def stream_body(monkeypatch, engine):
    monkeypatch.setattr(generation, "get_text_engine", lambda: engine)
    response = await generation.generate_text_stream(generation.TextGenerationRequest(prompt="hi"))
    return response.body_iterator


def test_w9_stream_route_closes_its_engine_after_a_full_stream(monkeypatch, generate_api_on):
    engine = StreamingFake(["Hello", " world"])
    monkeypatch.setattr(generation, "get_text_engine", lambda: engine)
    client = TestClient(main.app)
    try:
        response = client.post("/api/generate/text/stream", json={"prompt": "hi"})
    finally:
        client.close()
    assert response.status_code == 200
    assert response.text == "data: Hello\n\ndata:  world\n\ndata: [DONE]\n\n"
    assert engine.closed == 1


@pytest.mark.parametrize("fail_at", [0, 1])
async def test_w9_stream_route_closes_its_engine_when_the_upstream_fails(monkeypatch, fail_at):
    engine = StreamingFake(["a", "b", "c"], fail_at=fail_at)
    body = await stream_body(monkeypatch, engine)
    received = []
    with pytest.raises(RuntimeError, match="upstream stream broke"):
        async for chunk in body:
            received.append(chunk)
    assert received == ["data: a\n\n"][:fail_at] and engine.closed == 1


async def test_w9_stream_route_closes_its_engine_when_the_consumer_stops_early(monkeypatch):
    engine = StreamingFake(["a", "b", "c"])
    body = await stream_body(monkeypatch, engine)
    assert await body.__anext__() == "data: a\n\n"
    await body.aclose()
    assert engine.closed == 1


async def test_w9_client_disconnect_mid_stream_closes_the_engines_http_client(monkeypatch):
    stalled = asyncio.Event()  # never set: the upstream stalls after its first SSE line

    async def upstream():
        yield b'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n'
        await stalled.wait()

    engine = BytePlusEngine(api_key=FAKE_KEY)
    engine._client = http = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=upstream())))
    monkeypatch.setattr(generation, "get_text_engine", lambda: engine)
    response = await generation.generate_text_stream(generation.TextGenerationRequest(prompt="hi"))

    sent, first_chunk = [], asyncio.Event()

    async def send(message):
        sent.append(message)
        if message.get("body"):
            first_chunk.set()

    async def receive():  # the browser goes away once it has the first chunk
        await first_chunk.wait()
        return {"type": "http.disconnect"}

    # uvicorn reports ASGI spec 2.3, so Starlette cancels the streaming task group on disconnect
    scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}, "method": "POST",
             "path": "/api/generate/text/stream", "headers": []}
    await asyncio.wait_for(response(scope, receive, send), timeout=5)
    assert [m["body"] for m in sent if m["type"] == "http.response.body" and m["body"]] == [b"data: Hello\n\n"]
    assert http.is_closed and engine._client is None


async def test_w9_model_client_initialize_survives_a_missing_api_key(monkeypatch):
    def no_key():
        raise ValueError("BytePlus ModelArk API key is not set")

    monkeypatch.setattr(engines_pkg, "get_default_engine", no_key)
    monkeypatch.setattr(engines_pkg, "get_image_engine", no_key)
    model_client = engines_pkg.ModelClient()
    await model_client.initialize()  # only RuntimeError used to be caught
    assert model_client._default_engine is None and model_client._image_engine is None


def test_w9_text_route_answers_500_instead_of_crashing_without_a_key(monkeypatch, generate_api_on):
    def no_key():
        raise ValueError("BytePlus ModelArk API key is not set")

    monkeypatch.setattr(engines_pkg, "get_default_engine", no_key)
    monkeypatch.setattr(engines_pkg, "get_image_engine", no_key)
    monkeypatch.setattr(engines_pkg, "_client", None)
    client = TestClient(main.app)
    try:
        response = client.post("/api/generate/text", json={"prompt": "hi"})
    finally:
        client.close()
    assert response.status_code == 500 and "API key" in response.json()["detail"]


async def test_w9_config_reload_drops_the_cached_model_client(monkeypatch, config_file):
    assert engines_pkg._invalidate_client_on_config_change in config_loader.get_config()._callbacks
    manager = load_manager(config_file, GOOD_YAML)
    manager.on_change(engines_pkg._invalidate_client_on_config_change)
    stale = engines_pkg.ModelClient()
    monkeypatch.setattr(engines_pkg, "_client", stale)
    file_saved(manager, config_file, GOOD_YAML.replace("concurrency: 3", "concurrency: 4"))
    assert engines_pkg._client is None

    fresh_text, fresh_image = FakeEngine(), FakeEngine()
    monkeypatch.setattr(engines_pkg, "get_default_engine", lambda: fresh_text)
    monkeypatch.setattr(engines_pkg, "get_image_engine", lambda: fresh_image)
    rebuilt = await engines_pkg.get_client()
    assert rebuilt is not stale and (rebuilt._default_engine, rebuilt._image_engine) == (fresh_text, fresh_image)


# ============================================================================= W10: Seedream size and format

def as_pair(size: str) -> tuple[int, int]:
    w, h = size.split("x")
    return int(w), int(h)


def within_seedream_limits(w: int, h: int) -> bool:
    return (w % 16 == 0 and h % 16 == 0 and SEEDREAM_MIN_PIXELS <= w * h <= SEEDREAM_MAX_PIXELS
            and w <= 16 * h and h <= 16 * w)


def test_w10_normalize_size_always_lands_inside_seedream_limits():
    values = list(range(16, 6000, 61)) + [1, 0, -40, 720, 960, 1280, 2048, 4096, 30000, 10**6]
    for width, height in itertools.product(values, values):
        w, h = as_pair(normalize_size(width, height))
        assert within_seedream_limits(w, h), (width, height, w, h)
        if within_seedream_limits(width, height):
            assert (w, h) == (width, height)  # valid sizes are sent unchanged
        if width >= 16 and height >= 16 and 1 / 16 <= width / height <= 16:
            assert abs((w / h) / (width / height) - 1) < 0.07, (width, height, w, h)  # aspect ratio kept


@pytest.mark.parametrize("width, height", [
    (1024, 1536), (2048, 1024), (1280, 720), (960, 960), (2048, 2256), (4096, 256), (256, 4096),
])
def test_w10_valid_sizes_are_sent_unchanged(width, height):
    assert normalize_size(width, height) == f"{width}x{height}"


@pytest.mark.parametrize("width, height", [(100000, 1), (1, 100000), (9000, 100), (100, 9000)])
def test_w10_extreme_aspect_ratios_are_clamped_to_16(width, height):
    w, h = as_pair(normalize_size(width, height))
    assert within_seedream_limits(w, h)
    assert max(w / h, h / w) > 14 and (w > h) == (width > height)  # as extreme as allowed, orientation kept


@pytest.mark.parametrize("fmt, payload_format, mime", [
    ("png", "png", "image/png"), ("PNG", "png", "image/png"), (".png", "png", "image/png"),
    ("jpeg", "jpeg", "image/jpeg"), ("jpg", "jpeg", "image/jpeg"), ("JPEG", "jpeg", "image/jpeg"),
    ("webp", "jpeg", "image/jpeg"), ("", "jpeg", "image/jpeg"), (None, "jpeg", "image/jpeg"),
])
async def test_w10_output_format_is_mapped_to_png_or_jpeg(monkeypatch, fmt, payload_format, mime):
    engine = BytePlusEngine(api_key=FAKE_KEY)
    sent = []

    async def fake_post(path, payload):
        sent.append(payload)
        return {"data": [{"b64_json": TINY_PNG_B64, "size": payload["size"]}]}  # no output_format echoed back

    monkeypatch.setattr(engine, "_post", fake_post)
    config = ImageGenerationConfig(width=1024, height=1536)
    config.extra = {"output_format": fmt}
    response = await engine.generate_image("draw", config)
    assert sent[0]["output_format"] == payload_format and response.images[0].mime_type == mime


async def test_w10_job_sends_a_normalised_size_and_mapped_format_to_seedream(job_env, monkeypatch):
    sent = []

    async def fake_post(path, payload):
        sent.append(payload)
        return {"data": [{"b64_json": b64(image_bytes(96, 96, fmt="PNG")), "size": payload["size"]}]}

    seedream = BytePlusEngine(api_key=FAKE_KEY)
    monkeypatch.setattr(seedream, "_post", fake_post)
    monkeypatch.setattr(jobs, "build_engines",
                        lambda: Engines(text=job_env.text, fast_text=job_env.text, vision=job_env.vision, image=seedream))
    job_env.paper.num_panels = 1
    job_env.settings.update(panel_width=800, panel_height=800, output_format="PNG")  # 800x800 is below Seedream's minimum
    state = await job_env.run(qa=False)
    assert state["status"] == "completed"
    assert [(p["size"], p["output_format"]) for p in sent] == [("960x960", "png")]
    assert state["panels"][0]["image_url"].endswith("/panel_01.png")
