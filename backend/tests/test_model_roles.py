"""Model roles: a writer, a checker and a translator that may be different models, and costs priced per model."""

from __future__ import annotations

import pytest

from services import engine_select, pricing
from services.engine_select import Engines
from services.storyboarder_v2 import StoryboarderV2

from .fakes import FakeEngine, PaperModel, make_cast
from .test_story_v3 import v3_paper


def test_prices_by_model_prefix():
    assert pricing.text_price("deepseek-v4-pro-ga-260813") == (1.32, 0.044, 3.96)
    assert pricing.text_price("byteplus/seed-2-0-pro-260328") == (0.50, 0.10, 3.00)
    assert pricing.text_price("deepseek-v4-1-flash-261017") == pricing.TEXT_PRICES["deepseek-v4-1-flash"]   # the longest prefix wins
    assert pricing.text_price("some-new-model") == pricing.DEFAULT_TEXT_PRICE
    # cached prompt tokens are cheaper
    assert pricing.text_usd("deepseek-v4-pro-ga-260813", 1_000_000, 0) == pytest.approx(1.32)
    assert pricing.text_usd("deepseek-v4-pro-ga-260813", 1_000_000, 0, cached_tokens=1_000_000) == pytest.approx(0.044)
    assert pricing.usage_usd({"seed-2-0-pro-260328": {"prompt_tokens": 2_000_000, "completion_tokens": 1_000_000}}) == pytest.approx(4.0)


def test_engines_default_the_checker_and_translator_to_the_writer_and_count_tokens_per_model():
    writer, other = FakeEngine(), FakeEngine()
    writer.tokens = {"seed": {"prompt_tokens": 10, "completion_tokens": 2, "cached_tokens": 0}}
    other.tokens = {"deepseek": {"prompt_tokens": 5, "completion_tokens": 1, "cached_tokens": 3}}
    engines = Engines(text=writer, fast_text=writer, vision=writer, image=writer)
    assert engines.check is writer and engines.translate is writer
    assert engines.token_usage() == {"seed": {"prompt_tokens": 10, "completion_tokens": 2, "cached_tokens": 0}}
    engines = Engines(text=writer, fast_text=writer, vision=writer, image=writer, check=other)
    assert engines.token_usage()["deepseek"] == {"prompt_tokens": 5, "completion_tokens": 1, "cached_tokens": 3}


def test_build_engines_reads_check_and_translate_models_and_takes_overrides(monkeypatch):
    built = []

    def fake_engine_for(spec, image=False):
        built.append(spec)
        e = FakeEngine()
        e.model = spec
        return e

    class Config:
        def get_raw_config(self):
            return {"core_engine": {"text_model": "byteplus/seed-2-0-pro-260328", "check_model": "byteplus/deepseek-v4-pro-ga-260813",
                                    "image_model": "byteplus/dola-seedream-5-0-pro-260628"}}

    monkeypatch.setattr(engine_select, "_engine_for", fake_engine_for)
    monkeypatch.setattr(engine_select, "get_config", lambda: Config())
    engines = engine_select.build_engines()
    assert (engines.text.model, engines.check.model, engines.translate.model) == (
        "byteplus/seed-2-0-pro-260328", "byteplus/deepseek-v4-pro-ga-260813", "byteplus/seed-2-0-pro-260328")
    assert engines.translate is engines.text                                   # one engine per distinct spec
    engines = engine_select.build_engines({"text_model": "byteplus/deepseek-v4-pro-ga-260813", "check_model": None})
    assert engines.text.model == "byteplus/deepseek-v4-pro-ga-260813" and engines.check is engines.text


async def test_the_checker_checks_and_the_translator_translates():
    writer_model, checker_model = PaperModel(), PaperModel()
    writer, checker, translator = FakeEngine(router=writer_model), FakeEngine(router=checker_model), FakeEngine(router=PaperModel())
    sb = await StoryboarderV2(writer, check_engine=checker, translate_engine=translator).build_v3(
        v3_paper(), make_cast(), "zh-CN", (16, 32))
    assert "story" in writer_model.seen and "factcheck" not in writer_model.seen and "briefcheck" not in writer_model.seen
    assert {"briefcheck", "factcheck"} <= set(checker_model.seen) and "story" not in checker_model.seen
    assert {"quiz", "grade"} <= set(writer_model.seen) and "grade" not in checker_model.seen   # the quiz stays on the writer
    assert any(p.startswith("Translate this manga script") for p in translator.prompts)
    assert not any(p.startswith("Translate this manga script") for p in writer.prompts + checker.prompts)
    assert sb.language == "zh-CN"


def test_a_role_spec_can_set_the_reasoning_effort():
    assert engine_select._options("byteplus/deepseek-v4-pro-ga-260813?effort=low") == ("byteplus/deepseek-v4-pro-ga-260813", {"effort": "low"})
    assert engine_select._options("byteplus/seed-2-0-pro-260328") == ("byteplus/seed-2-0-pro-260328", {})


def test_paper_goes_first_and_identical_in_every_call():
    from services.storyboarder_v2 import PAPER_PREFIX, _messages
    body = "The Transformer achieves 28.4 BLEU.\n[T2.r9]   Transformer (big)   c1: 28.4"
    story = _messages("You write manga.", "Write the story.\n\nPAPER TEXT:\n" + body, body)
    check = _messages("", "Check these lines.\n\nPAPER TEXT:\n" + body, body + "\n")   # a builder that strips the body
    assert story[0].content == check[0].content == PAPER_PREFIX + "PAPER TEXT:\n" + body.strip()
    assert [m.role.value for m in story] == ["system", "system", "user"] and story[2].content == "Write the story."
    assert [m.role.value for m in check] == ["system", "user"] and check[1].content == "Check these lines."
    # a prompt that does not end with the paper goes as it is
    plain = _messages("sys", "No paper here.", body)
    assert [m.content for m in plain] == ["sys", "No paper here."]


async def test_build_v3_sends_the_paper_first_with_one_cache_key():
    engine = FakeEngine(router=PaperModel())
    await StoryboarderV2(engine, cache_key="prism-job-1").build_v3(v3_paper(), make_cast(), "en-US", (16, 32))
    paper_calls = [(m, c) for m, c in engine.calls if m[0].content.startswith("You work on one research paper")]
    assert len(paper_calls) >= 4                                   # brief, brief check, story, fact-check
    assert len({m[0].content for m, _ in paper_calls}) == 1        # byte-identical prefix
    assert {c.prompt_cache_key for _, c in paper_calls} == {"prism-job-1"}
    # the engine may cache the paper explicitly (Seed holds an implicit prefix 30-60 s); no other call asks for that
    assert all(c.cache_prefix for _, c in paper_calls)
    assert not any(c.cache_prefix for m, c in engine.calls if not m[0].content.startswith("You work on one research paper"))


async def test_the_warm_up_thinks_like_the_checks_it_serves():
    # a cached head serves only calls with the same thinking setting: a warm-up without thinking left the three
    # brief-check parts of a real job uncached
    engine = FakeEngine([{"ok": True}])
    await StoryboarderV2(engine, cache_key="prism-job-1")._warm(engine, "The paper.\n" * 100)
    [(messages, config)] = engine.calls
    assert config.thinking is True and config.cache_prefix and config.max_tokens == 1
    assert messages[0].content.startswith("You work on one research paper")



async def test_preflight_probes_each_modelark_text_model_once_and_raises_account_errors(monkeypatch):
    import json as _json
    from engines.byteplus import AccountError, BytePlusEngine
    engine_select._ANSWERING.clear()
    calls = []

    def make(model, refuse=False):
        e = BytePlusEngine(api_key="test-key-not-real", model=model)

        async def post(path, payload):
            calls.append(payload["model"])
            if refuse:
                raise AccountError(404, _json.dumps({"error": {"code": "ModelNotOpen", "message": "x"}}), payload["model"])
            return {"model": payload["model"], "choices": [{"message": {"content": "OK"}}], "usage": {}}

        e._post = post
        return e

    seed = make("seed-2-0-pro-260328")
    fake = FakeEngine()                                   # not a ModelArk engine: never probed
    engines = Engines(text=seed, fast_text=fake, vision=seed, image=fake)
    await engine_select.preflight(engines)
    await engine_select.preflight(engines)
    assert calls == ["seed-2-0-pro-260328"]               # once per model and process
    blocked = Engines(text=seed, fast_text=fake, vision=seed, image=fake, check=make("deepseek-v4-pro-ga-260813", refuse=True))
    with pytest.raises(AccountError):
        await engine_select.preflight(blocked)
    engine_select._ANSWERING.clear()


async def test_a_role_falls_back_to_the_writer_when_its_model_refuses_or_is_not_activated():
    import json as _json
    from engines.base import Message
    from engines.byteplus import AccountError, ContentFilteredError
    from services.engine_select import FallbackEngine

    class Refusing(FakeEngine):
        def __init__(self, error):
            super().__init__(model="deepseek-v4-pro-ga-260813")
            self.error, self.asked = error, 0

        async def generate_text(self, messages, config=None):
            self.asked += 1
            raise self.error

    writer = FakeEngine(["from the writer", "again the writer"])
    filtered = FallbackEngine(Refusing(ContentFilteredError("deepseek-v4-pro-ga-260813")), writer)
    assert (await filtered.generate_text([Message.user("x")])).content == "from the writer"
    assert "content filter" in filtered.fallbacks[0] and not filtered._inactive     # the next text may pass the filter

    inactive_model = Refusing(AccountError(404, _json.dumps({"error": {"code": "ModelNotOpen", "message": "x"}}), "deepseek-v4-pro-ga-260813"))
    inactive = FallbackEngine(inactive_model, FakeEngine(["w1", "w2"]))
    assert (await inactive.generate_text([Message.user("x")])).content == "w1"
    assert (await inactive.generate_text([Message.user("y")])).content == "w2"
    assert inactive_model.asked == 1 and "ModelNotOpen" in inactive.fallbacks[0]   # not asked again

    overdue = FallbackEngine(Refusing(AccountError(403, _json.dumps({"error": {"code": "AccountOverdueError", "message": "x"}}))), FakeEngine())
    with pytest.raises(AccountError):   # account-wide: the writer would fail too, the job must stop
        await overdue.generate_text([Message.user("x")])


async def test_an_empty_reply_is_an_error_not_an_empty_result():
    from engines.base import TextResponse
    from services.storyboarder_v2 import _ask_json
    engine = FakeEngine([TextResponse(content="", model="m", usage={}, finish_reason="content_filter")])
    with pytest.raises(ValueError, match="empty reply"):
        await _ask_json(engine, "", "Check this.", thinking=True, temperature=0.1, key="problems")
    assert len(engine.calls) == 1   # no "repair" call that would invent {"problems": []}


async def test_a_separate_checker_reads_the_script_in_parallel_parts_after_one_warm_up():
    from services.storyboarder_v2 import factcheck_parts
    writer_model, checker_model = PaperModel(), PaperModel()
    writer, checker = FakeEngine(router=writer_model), FakeEngine(router=checker_model)
    sb = await StoryboarderV2(writer, check_engine=checker).build_v3(v3_paper(), make_cast(), "en-US", (16, 32))
    parts = factcheck_parts(sb, v3_paper(), "en-US", size=5)
    assert len(parts) > 1 and sum(card for _, card in parts) == 1
    assert sorted(n for panels, _ in parts for n in panels) == [p.n for p in sb.panels]
    warmups = [m for m, _ in checker.calls if m[-1].content == "Reply OK."]
    checks = [m for m, _ in checker.calls if m[-1].content.startswith("You are a meticulous fact-checker")]
    assert len(warmups) >= 1 and len(checks) >= 2 * len(warmups)          # each fact-check pass: one warm-up, then its parts
    assert all(m[0].content == warmups[0][0].content for m in checks)   # the same cached paper prefix


async def test_a_check_part_with_an_empty_reply_is_asked_once_more():
    from engines.base import TextResponse
    from services.storyboarder_v2 import _ask_check
    engine = FakeEngine([TextResponse(content="", model="m", usage={}, finish_reason="stop"), '{"problems": [{"id": "3.d0"}]}'])
    data, _ = await _ask_check(engine, "Check this.", thinking=True, temperature=0.1, key="problems")
    assert data == {"problems": [{"id": "3.d0"}]} and len(engine.calls) == 2
    engine = FakeEngine([TextResponse(content="", model="m", usage={}, finish_reason="stop")] * 2)
    with pytest.raises(ValueError, match="empty reply"):   # twice empty: the part is lost (and recorded by the caller)
        await _ask_check(engine, "Check this.", thinking=True, temperature=0.1, key="problems")
