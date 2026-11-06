"""Final-language reviews are reused only while the actual visible words still match."""
import copy
from types import SimpleNamespace

from services.jobs import _public_reader_reviews, _review_final_scripts
from services.storyboarder_v2 import StoryboardV2, PanelV2


def script(language="en-US", novice=True):
    sb = StoryboardV2(title="Test", summary="", theme="kumomo", language=language, chapters=[], panels=[PanelV2(n=1, chapter="hook", title="Start", characters=[], scene="", visual="", narration="A model writes a solution.")])
    if novice:
        sb.brief["reader_contract"] = "novice_v1"
    return sb


def saved(sb, status="reviewed"):
    from services.reader_review import visible_fingerprint
    return {"status": status, "language": sb.language, "fingerprint": visible_fingerprint(sb),
            "answers": [], "problems": [], "reason": ""}


async def test_review_runs_after_localization_and_backport_and_counts_usage(monkeypatch):
    from services import reader_review
    english, chinese = script(), script("zh-CN")
    english.brief["reader_review"] = saved(english)
    chinese.brief["reader_review"] = copy.deepcopy(english.brief["reader_review"])
    english.panels[0].narration = "The translated fact check changed this explanation."
    chinese.panels[0].narration = "模型正在写解题步骤。"
    calls, usage = [], []

    async def review(sb, engine, *, on_usage=None):
        calls.append((sb.language, sb.panels[0].narration))
        on_usage({"prompt_tokens": 11, "completion_tokens": 7})
        sb.brief["reader_review"] = saved(sb, "needs_revision")
        return sb.brief["reader_review"]

    monkeypatch.setattr(reader_review, "review_story", review)
    writer = SimpleNamespace(check_engine=object(), _add_usage=usage.append)
    scripts = {"en-US": english, "zh-CN": chinese}
    await _review_final_scripts(writer, scripts)
    assert calls == [("en-US", english.panels[0].narration), ("zh-CN", chinese.panels[0].narration)]
    assert len(usage) == 2
    await _review_final_scripts(writer, scripts)
    assert len(calls) == 2, "unchanged diagnostics, including unresolved ones, are reused"
    assert set(_public_reader_reviews(scripts)) == {"en-US", "zh-CN"}


async def test_legacy_scripts_are_not_migrated_or_reviewed(monkeypatch):
    from services import reader_review

    async def unexpected(*args, **kwargs):
        raise AssertionError("legacy script triggered a model call")

    monkeypatch.setattr(reader_review, "review_story", unexpected)
    scripts = {"en-US": script(novice=False)}
    await _review_final_scripts(SimpleNamespace(), scripts)
    assert _public_reader_reviews(scripts) == {}
    assert "reader_contract" not in scripts["en-US"].brief


def test_public_review_detects_changes_and_does_not_mutate_cached_evidence():
    sb = script()
    sb.brief["reader_review"] = saved(sb)
    before = copy.deepcopy(sb.brief["reader_review"])
    assert _public_reader_reviews({sb.language: sb})[sb.language]["status"] == "reviewed"
    sb.panels[0].narration += " Changed after relayout."
    public = _public_reader_reviews({sb.language: sb})[sb.language]
    assert public["status"] == "unassessed" and public["answers"] == []
    assert public["fingerprint"] != before["fingerprint"]
    assert sb.brief["reader_review"] == before


async def test_review_counts_tokens_even_when_reply_is_malformed():
    from services.reader_review import review_story
    from engines.base import TextResponse
    class Engine:
        async def generate_text(self, messages, config=None):
            return TextResponse("not json", "fake", {"prompt_tokens": 12, "completion_tokens": 8})
    usage = []
    result = await review_story(script(), Engine(), on_usage=usage.append)
    assert result["status"] == "unassessed"
    assert usage == [{"prompt_tokens": 12, "completion_tokens": 8}]


def test_novice_adaptation_does_not_rewrite_paper_source_or_legacy_brief():
    from services.storyboarder_v2 import novice_prompt, brief_prompt, brief_prompt_v3
    from .test_story_v3 import v3_paper
    paper = "- A panel that teaches a claim says all of it: a quoted source line.\n7. payoff (this is a source heading)"
    result = novice_prompt("Write a story.\n7. payoff (old instruction)\nPAPER TEXT:\n" + paper)
    assert result.split("PAPER TEXT:\n", 1)[1] == paper
    assert "Plan for a novice" not in brief_prompt(v3_paper())
    assert "Plan for a novice" in brief_prompt_v3(v3_paper(), "An abstract")
