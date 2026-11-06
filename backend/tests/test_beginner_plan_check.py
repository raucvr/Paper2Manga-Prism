"""Offline routing and role contracts; no claim that fake replies prove factual accuracy."""
from copy import deepcopy
import json

import pytest

from services.paper_extractor import PaperText
from services.storyboarder_v2 import (
    StoryboarderV2, _brief_items, brief_check_prompt, brief_prompt_v3,
)
from .fakes import FakeEngine, make_cast

PAPER = PaperText("Task study", "The method aims to preserve answer quality. References are model outputs.", [])


def novice():
    return {"reader_contract": "novice_v1", "beginner_plan": {
        "goal": "Always preserve all quality.",
        "foundations": ["First unsupported benefit.", "Second unsupported benefit.", "Explain input."],
        "opening_questions": ["Why are references perfect?", "Why does it always fail?", "What is the task?"],
        "steps": [{"what": "Use references.", "why": "Makes all text coherent.", "source": "keep metadata"},
                  {"what": "Solve the task.", "why": "Preserves all quality."}],
        "extra": "Unchecked metadata.",
    }}


def finding(pid, corrected, quote="The method aims to preserve answer quality."):
    return {"id": "beginner_plan." + pid, "corrected": corrected, "quote": quote}


async def test_every_supported_plan_leaf_is_sent_to_checker():
    brief = novice()
    expected = {"beginner_plan.goal"}
    expected |= {f"beginner_plan.{field}.{i}" for field in ("foundations", "opening_questions") for i in range(3)}
    expected |= {f"beginner_plan.steps.{i}.{field}" for i in range(2) for field in ("what", "why")}
    engine = FakeEngine([{"problems": []}])
    await StoryboarderV2(engine).check_brief(brief, PAPER)
    prompt = engine.prompts[0]
    entries = json.loads(prompt.split("ITEMS:\n")[1].split("\n\nPAPER TEXT:")[0])
    assert {entry["id"] for entry in entries} == expected


async def test_nested_leaf_corrections_and_deletes_preserve_siblings_and_original_indices():
    brief = novice()
    engine = FakeEngine([{"problems": [
        finding("goal", "Aim to preserve answer quality."),
        finding("foundations.0", "DELETE"), finding("foundations.1", "DELETE"),
        finding("foundations.2", "An input is the question to answer."),
        finding("opening_questions.0", "DELETE"), finding("opening_questions.1", "DELETE"),
        finding("opening_questions.2", "What answer are we trying to produce?"),
        finding("steps.0.why", "DELETE"), finding("steps.1.what", "Answer the question."),
        finding("steps.1.why", "DELETE"),
    ]}])
    assert await StoryboarderV2(engine).check_brief(brief, PAPER) == 10
    plan = brief["beginner_plan"]
    assert plan["foundations"] == ["An input is the question to answer."]
    assert plan["opening_questions"] == ["What answer are we trying to produce?"]
    assert plan["steps"] == [{"what": "Use references.", "source": "keep metadata"}, {"what": "Answer the question."}]
    assert plan["goal"] == "Aim to preserve answer quality."
    assert plan["extra"] == "Unchecked metadata."


async def test_unchecked_or_arbitrary_paths_cannot_be_modified():
    brief = novice()
    brief["beginner_plan"]["steps"][0]["blank"] = ""
    before = deepcopy(brief)
    engine = FakeEngine([{"problems": [finding(path, "Injected text") for path in (
        "extra", "steps.0.source", "steps.0.blank", "steps.0", "steps.9.what", "steps.-1.what",
        "foundations.01", "foundations.-1", "goal.nested", "__class__",
    )]}])
    assert await StoryboarderV2(engine).check_brief(brief, PAPER) == 0
    assert brief["beginner_plan"] == before["beginner_plan"]


async def test_finding_for_another_checker_chunk_is_not_authorized(monkeypatch):
    import services.storyboarder_v2 as mod
    brief = novice()
    monkeypatch.setattr(mod, "FACTCHECK_CHUNK", 2)
    async def warm(*args):
        pass
    async def check(engine, prompt, **kwargs):
        entries = json.loads(prompt.split("ITEMS:\n")[1].split("\n\nPAPER TEXT:")[0])
        return {"problems": [finding("goal", "Wrong chunk correction")] if entries[0]["id"] != "beginner_plan.goal" else []}, {}
    monkeypatch.setattr(mod, "_ask_check", check)
    story = StoryboarderV2(FakeEngine(), check_engine=FakeEngine())
    monkeypatch.setattr(story, "_warm", warm)
    assert await story.check_brief(brief, PAPER) == 0
    assert brief["beginner_plan"]["goal"] == "Always preserve all quality."


async def test_quoted_evidence_is_not_used_as_question_and_correction_retains_role():
    brief = novice()
    engine = FakeEngine([{"problems": [
        finding("opening_questions.0", '"How are reference answers produced?"', "References are model outputs."),
        finding("opening_questions.1", "References are model outputs."),
        finding("goal", "Aim to preserve answer quality."),
    ]}])
    assert await StoryboarderV2(engine).check_brief(brief, PAPER) == 3
    assert brief["beginner_plan"]["opening_questions"] == ["How are reference answers produced?", "What is the task?"]
    assert brief["beginner_plan"]["goal"] == "Aim to preserve answer quality."
    prompt = engine.prompts[0]
    assert "quote field is source evidence, not replacement text" in prompt
    assert "never turn an aim into a measured achievement" in prompt
    assert "check the factual presuppositions of questions" in prompt


@pytest.mark.parametrize("contract", [None, "legacy", "novice_v2"])
async def test_legacy_plan_is_neither_checked_nor_mutated(contract):
    brief = novice()
    brief["reader_contract"] = contract
    before = deepcopy(brief["beginner_plan"])
    assert not _brief_items(brief)
    engine = FakeEngine([{"problems": [finding("goal", "DELETE"), finding("steps.0.why", "changed")]}])
    assert await StoryboarderV2(engine).check_brief(brief, PAPER) == 0
    assert brief["beginner_plan"] == before
    assert "For beginner_plan" not in brief_check_prompt(brief, PAPER)


async def test_writer_receives_the_corrected_plan():
    brief = novice()
    engine = FakeEngine([
        {"problems": [finding("goal", "Aim to preserve answer quality."), finding("steps.0.why", "DELETE")]},
        {"title": "Task", "panels": [{"n": 1, "beat": "hook", "kind": "scene", "characters": ["papi"],
                                      "visual": "Papi holds a blank card.", "dialogue": []}]},
    ])
    story = StoryboarderV2(engine)
    await story.check_brief(brief, PAPER)
    sb = await story.write_v3(brief, [], make_cast(), (1, 1), PAPER)
    assert '"goal": "Aim to preserve answer quality."' in engine.prompts[-1]
    assert "Makes all text coherent." not in engine.prompts[-1]
    assert sb.brief["beginner_plan"] == brief["beginner_plan"]


def test_brief_prompt_requests_concrete_tasks_and_unbiased_questions():
    prompt = brief_prompt_v3(PAPER, PAPER.body)
    for requirement in ("everyday task with a concrete input and desired output", "research aim from a measured achievement",
                        "no unexplained abbreviations", "perfect reference answers", "inevitable failures",
                        "complete recovery", "speedups unsupported by the paper"):
        assert requirement in prompt
