"""Information completeness: the brief lists the paper's key points, panels say which points they cover, missing core
points get panels (or a sentence in a detail), and every panel carries a "detail" plus a paper card for the reader.
Every engine is a fake; conftest blocks the network."""

import json

import pytest

from services.paper_extractor import PaperText
from services.storyboarder_v2 import (
    PanelV2,
    StoryboardV2,
    StoryboarderV2,
    brief_prompt,
    coverage,
    paper_card,
    storyboard_prompt,
    translate_prompt,
    uncovered,
)

from .fakes import BRIEF, FakeEngine, PaperModel, make_cast, make_panel, storyboard_payload

POINTS = [
    {"id": "P1", "kind": "problem", "priority": "core", "text": "Recurrent models read one token at a time."},
    {"id": "P2", "kind": "mechanism", "priority": "core", "text": "Self-attention links every pair of positions in one step."},
    {"id": "P3", "kind": "result", "priority": "core", "text": "28.4 BLEU on WMT 2014 English-German."},
    {"id": "P4", "kind": "setup", "priority": "important", "text": "Trained on 8 P100 GPUs."},
    {"id": "P5", "kind": "limitation", "priority": "detail", "text": "Cost grows with the square of the length."},
]
BRIEF_WITH_POINTS = {**BRIEF, "contributions": ["The Transformer, built only from attention."], "points": POINTS}


def sb_with(covers: list[list[str]], brief=BRIEF_WITH_POINTS) -> StoryboardV2:
    panels = [make_panel(i, chapter="how") for i in range(1, len(covers) + 1)]
    for p, c in zip(panels, covers):
        p.covers, p.detail = list(c), f"Detail of panel {p.n}."
    return StoryboardV2(title="T", summary="S", theme="kumomo", language="en-US", chapters=[], panels=panels, brief=dict(brief),
                        card=paper_card(brief))


# ----------------------------------------------------------------------------- prompts and data model

def test_the_brief_asks_for_the_whole_paper_as_prioritised_points():
    prompt = brief_prompt(PaperText("T", "body", []))
    assert '"points"' in prompt and "core" in prompt and "every main result" in prompt and '"contributions"' in prompt
    assert "3-6 that matter most" not in prompt and "3-5 strongest" not in prompt


def test_the_script_must_cover_every_core_point_and_give_each_panel_a_detail():
    prompt = storyboard_prompt(BRIEF_WITH_POINTS, make_cast(), "en-US", (12, 18))
    assert 'Every "core" point must be covered by at least one panel' in prompt
    assert '"covers": ["P1"], "detail": "..."' in prompt and "never drawn" in prompt


def test_covers_and_detail_survive_from_dict_and_to_dict():
    p = PanelV2.from_dict({"n": 2, "chapter": "how", "characters": ["papi"], "dialogue": [], "covers": ["P2", None, " P3 "],
                           "detail": " Self-attention costs O(n²·d). "}, 1)
    assert (p.covers, p.detail) == (["P2", "P3"], "Self-attention costs O(n²·d).")
    assert PanelV2.from_dict(p.to_dict(), 1) == p


def test_uncovered_and_coverage_count_points_by_priority():
    sb = sb_with([["P1"], ["P2", "P4"]])
    assert [pt["id"] for pt in uncovered(sb)] == ["P3"]
    assert coverage(sb) == {"core": [2, 3], "important": [1, 1]}


def test_the_paper_card_summarises_the_brief():
    card = paper_card(BRIEF_WITH_POINTS)
    assert card["problem"].startswith("Recurrent networks") and card["key_idea"] == BRIEF["key_idea"]
    assert card["contributions"] == ["The Transformer, built only from attention."]
    assert card["how"] == ["self-attention: Words vote on which other words matter."]
    assert card["results"] == ["Better translation (28.4 BLEU on WMT 2014 EN-DE)"]
    assert card["limitations"] == BRIEF["limitations"] and card["takeaway"] == BRIEF["takeaway"]


# ----------------------------------------------------------------------------- coverage repair

def cover_reply(**fields):
    new = {"after": 1, "chapter": "evidence", "title": "Results", "characters": ["papi"], "scene": "Lab", "visual": "papi holds a trophy",
           "dialogue": [{"speaker": "papi", "text": "It scored 28.4 BLEU."}], "narration": "", "on_image_labels": [],
           "emotions": {"papi": "proud"}, "covers": ["P3"], "detail": "28.4 BLEU on WMT 2014 English-German."}
    new.update(fields)
    return {"new_panels": [new], "details": []}


async def test_a_missing_core_point_gets_a_new_panel_in_reading_order():
    sb = sb_with([["P1"], ["P2"]])
    engine = FakeEngine([cover_reply()])
    assert await StoryboarderV2(engine).cover(sb, make_cast(), (2, 6)) == 1
    # an evidence panel goes after the how panels (chapters stay contiguous), whatever "after" says
    assert [(p.n, p.chapter, p.covers) for p in sb.panels] == [(1, "how", ["P1"]), (2, "how", ["P2"]), (3, "evidence", ["P3"])]
    assert uncovered(sb) == [] and [(c.id, c.panel_numbers) for c in sb.chapters] == [("how", [1, 2]), ("evidence", [3])]
    assert "P3" in engine.prompts[0] and "28.4 BLEU" in engine.prompts[0] and "at most 6 panels" in engine.prompts[0]


async def test_without_room_a_missing_point_goes_into_the_closest_panels_detail():
    sb = sb_with([["P1"], ["P2"]])
    reply = {"new_panels": [cover_reply()["new_panels"][0]], "details": [{"n": 2, "add_detail": "It reached 28.4 BLEU.", "covers": ["P3"]}]}
    assert await StoryboarderV2(FakeEngine([reply])).cover(sb, make_cast(), (2, 2)) == 1
    assert len(sb.panels) == 2 and sb.panels[1].detail == "Detail of panel 2. It reached 28.4 BLEU."
    assert sb.panels[1].covers == ["P2", "P3"]


@pytest.mark.parametrize("reply", [RuntimeError("model down"), "no json", {"new_panels": ["junk"], "details": [{"n": "x"}]}])
async def test_a_failed_coverage_repair_keeps_the_script(reply):
    sb = sb_with([["P1"], ["P2"]])
    before = [p.to_dict() for p in sb.panels]
    assert await StoryboarderV2(FakeEngine([reply, reply])).cover(sb, make_cast(), (2, 6)) == 0
    assert [p.to_dict() for p in sb.panels] == before


async def test_nothing_is_asked_when_every_core_point_is_covered():
    engine = FakeEngine([])
    assert await StoryboarderV2(engine).cover(sb_with([["P1", "P2"], ["P3"]]), make_cast(), (2, 6)) == 0
    assert engine.calls == []


# ----------------------------------------------------------------------------- translation and fact-check of the reader text

async def test_details_and_the_card_are_translated_and_checked_for_the_target_script():
    sb = sb_with([["P1"], ["P2"]])
    payload = json.loads(translate_prompt(sb, "zh-CN").rsplit("\n\n", 1)[1])
    assert payload["panels"][0]["detail"] == "Detail of panel 1." and payload["card"]["key_idea"] == BRIEF["key_idea"]
    reply = {"title": "注意力", "summary": "摘要", "chapters": [], "card": {**{k: v for k, v in sb.card.items()},
             "key_idea": "让每个词同时关注所有其他词。", "limitations": ["Cost grows quadratically."]},
             "panels": [{"n": p.n, "title": "格", "dialogue": [{"speaker": b.speaker, "text": "中文台词"} for b in p.dialogue],
                         "narration": "中文旁白。" if p.narration else None, "labels": [], "detail": d}
                        for p, d in zip(sb.panels, ["第一格的细节。", "Still English detail."])]}
    await StoryboarderV2(FakeEngine([reply])).translate(sb, "zh-CN")
    assert sb.panels[0].detail == "第一格的细节。" and sb.panels[1].detail == "Detail of panel 2."   # English kept, not lost
    assert sb.card["key_idea"] == "让每个词同时关注所有其他词。"
    assert sb.card["limitations"] == BRIEF["limitations"]                                         # untranslated list kept whole


async def test_the_fact_check_corrects_details_and_card_items():
    sb = sb_with([["P1"], ["P2"]])
    engine = FakeEngine([{"problems": [
        {"id": "2.x", "issue": "wrong number", "corrected": "Self-attention links any two positions in one step."},
        {"id": "card.results.0", "issue": "missing setting", "corrected": "28.4 BLEU on WMT 2014 English-German (newstest2014)."},
        {"id": "card.key_idea.0", "issue": "overgeneralised", "corrected": "Attention can replace recurrence for translation."},
    ]}])
    assert await StoryboarderV2(engine).factcheck(sb, PaperText("T", "We propose the Transformer.", [])) == 3
    assert sb.panels[1].detail == "Self-attention links any two positions in one step."
    assert sb.card["results"] == ["28.4 BLEU on WMT 2014 English-German (newstest2014)."]
    assert sb.card["key_idea"] == "Attention can replace recurrence for translation."
    lines = engine.prompts[0].split("LINES:", 1)[1].split("PAPER TEXT:", 1)[0]
    assert '"id": "1.x"' in lines and '"id": "card.problem.0"' in lines


# ----------------------------------------------------------------------------- the pipeline and the job state

class CoveragePaper(PaperModel):
    """The fake writer leaves P3 out; the coverage repair adds a panel for it."""

    def __call__(self, prompt, messages, config):
        user = messages[-1].content
        if user.startswith("Read the paper and write a compact, faithful BRIEF"):
            self.seen.append("brief")
            return BRIEF_WITH_POINTS
        if user.startswith("The manga script below does not cover these CORE points"):
            self.seen.append("cover")
            return cover_reply(after=2)
        return super().__call__(prompt, messages, config)

    async def _storyboard(self) -> dict:
        data = storyboard_payload(self.num_panels)
        for p, covers in zip(data["panels"], (["P1"], ["P2"], ["P4"])):
            p["covers"], p["detail"] = covers, f"Detail {p['n']}."
        return data


async def test_build_repairs_coverage_before_the_fact_check():
    model = CoveragePaper(num_panels=3)
    sb = await StoryboarderV2(FakeEngine(router=model)).build(PaperText("T", "body", []), make_cast(), "en-US", (3, 6))
    assert model.seen == ["brief", "briefcheck", "storyboard", "cover", "factcheck"]
    # hook, problem, idea, then the new evidence panel (chapter order), even though the reply said after=2
    assert [p.covers for p in sb.panels] == [["P1"], ["P2"], ["P4"], ["P3"]] and uncovered(sb) == []
    assert sb.card["contributions"] == ["The Transformer, built only from attention."]


async def test_the_job_state_carries_details_the_card_and_coverage(job_env):
    model = CoveragePaper(num_panels=3)
    job_env.text.router = model
    state = await job_env.run(qa=False)
    assert [p["detail"] for p in state["panels"]] == ["Detail 1.", "Detail 2.", "Detail 3.", "28.4 BLEU on WMT 2014 English-German."]
    assert state["paper_card"]["key_idea"] == BRIEF["key_idea"]
    assert state["coverage"] == {"core": [3, 3], "important": [1, 1]}
    saved = json.loads((job_env.output_dir / state["job_id"] / "storyboard.json").read_text(encoding="utf-8"))
    assert saved["card"] == state["paper_card"] and saved["panels"][3]["covers"] == ["P3"]


def test_the_job_report_prints_key_point_coverage(tmp_path):
    from tools import job_report
    job_dir = tmp_path / "20261022_000000_cov1"
    job_dir.mkdir()
    (job_dir / "job.json").write_text(json.dumps({"job_id": job_dir.name, "title": "T", "status": "completed", "language": "en-US",
                                                  "layout": "single", "panels": [], "cost": {}, "coverage": {"core": [9, 10], "important": [4, 7]}}),
                                      encoding="utf-8")
    assert "Key points covered: core 9/10  important 4/7" in job_report.report(job_dir)


def test_the_script_rules_ask_for_facts_in_every_panel_and_no_text_or_copies_in_visuals():
    prompt = storyboard_prompt(BRIEF_WITH_POINTS, make_cast(), "en-US", (12, 18))
    assert "The narration adds a fact the bubbles do not say" in prompt and "explain it in plain words" in prompt
    assert "Never draw copies or miniature versions of a cast member" in prompt and "sums on a chalkboard" in prompt
    brief = brief_prompt(PaperText("T", "body", []))
    assert "including the appendix" in brief and "never add your own" in brief


async def test_numbers_in_a_scene_description_are_fact_checked_because_they_get_drawn():
    sb = sb_with([["P1"], ["P2"]])
    sb.panels[1].visual = "two cards reading 3.5 days and 168 days"
    engine = FakeEngine([{"problems": [{"id": "2.v", "issue": "168 days is not in the paper", "corrected": "two cards: a short bar and a long bar"}]}])
    assert await StoryboarderV2(engine).factcheck(sb, PaperText("T", "Training took 3.5 days.", [])) == 1
    assert sb.panels[1].visual == "two cards: a short bar and a long bar"
    assert '"id": "2.v"' in engine.prompts[0] and '"id": "1.v"' not in engine.prompts[0]


def test_a_scene_with_numbers_or_sums_asks_for_blank_cards():
    from services.panel_renderer import TEXTY_HINT, build_single_prompt
    sums = make_panel(1, characters=["papi"], visual="papi writes sums on a chalkboard")
    digits = make_panel(2, characters=["papi"], visual="a bar chart with 8 bars")
    plain = make_panel(3, characters=["papi"], visual="papi holds a lantern")
    assert TEXTY_HINT in build_single_prompt(sums, make_cast(), "zh-CN", ["papi"])
    assert TEXTY_HINT in build_single_prompt(digits, make_cast(), "zh-CN", ["papi"])
    assert TEXTY_HINT not in build_single_prompt(plain, make_cast(), "zh-CN", ["papi"])


# ----------------------------------------------------------------------------- review of the coverage repair

CHAPTERED = ["hook", "problem", "idea", "how", "how", "evidence", "limits", "takeaway"]


def chaptered_sb() -> StoryboardV2:
    panels = [make_panel(i, chapter=c) for i, c in enumerate(CHAPTERED, start=1)]
    for p in panels:
        p.covers = []
    panels[0].covers, panels[3].covers = ["P1"], ["P2"]
    return StoryboardV2(title="T", summary="S", theme="kumomo", language="en-US", chapters=[], panels=panels,
                        brief=dict(BRIEF_WITH_POINTS), card={})


@pytest.mark.parametrize("after, chapter, expected", [
    (0, "evidence", 6),       # after <= 0: the start of its chapter, not the end of the manga
    (99, "evidence", 7),      # beyond the chapter: its end, never after the takeaway
    (2, "evidence", 6),       # before the chapter: its start
    ("x", "limits", 8),       # no "after": inside its own chapter
    (8, "", 8),               # unknown chapter after the takeaway: before the takeaway, so the manga still ends on it
    (3, "", 4),               # unknown chapter: right after the panel it follows, taking its chapter
])
async def test_new_panels_are_placed_inside_their_own_chapter(after, chapter, expected):
    sb = chaptered_sb()
    await StoryboarderV2(FakeEngine([cover_reply(after=after, chapter=chapter)])).cover(sb, make_cast(), (8, 12))
    new = next(p for p in sb.panels if p.covers == ["P3"])
    assert new.n == expected
    runs = [p.chapter for p in sb.panels]
    assert all(runs.index(c) + runs.count(c) - 1 == len(runs) - 1 - runs[::-1].index(c) for c in set(runs))   # contiguous chapters


@pytest.mark.parametrize("bad", [
    {"covers": ["P2"]},                                                     # covers nothing that is missing
    {"visual": ""},                                                          # nothing to draw
    {"dialogue": [], "narration": ""},                                        # nothing to say
    {"title": "Panel 4"},                                                                                 # repeats panel 4's title
    {"dialogue": [{"speaker": "papi", "text": "Attention lets every word look at every other word."}]},   # or its lines
])
async def test_contentless_off_target_or_repeated_panels_are_not_inserted(bad):
    sb = chaptered_sb()
    before = len(sb.panels)
    await StoryboarderV2(FakeEngine([cover_reply(**bad)])).cover(sb, make_cast(), (8, 12))
    assert len(sb.panels) == before


async def test_detail_items_single_objects_and_tag_only_items_are_used():
    sb = chaptered_sb()
    reply = {"new_panels": {"n": 6, "add_detail": "It reached 28.4 BLEU.", "covers": ["P3"]},   # a detail item in the wrong list, as an object
             "details": {"n": 4, "covers": "P4"}}                                              # tag-only, as an object
    assert await StoryboarderV2(FakeEngine([reply])).cover(sb, make_cast(), (8, 12)) == 1
    assert len(sb.panels) == 8 and sb.panels[5].detail.endswith("It reached 28.4 BLEU.") and "P3" in sb.panels[5].covers
    assert "P4" in sb.panels[3].covers


async def test_panels_beyond_the_room_leave_their_facts_in_the_detail_they_would_follow():
    sb = chaptered_sb()
    second = cover_reply(covers=["P4"], title="Hardware", detail="Trained on 8 P100 GPUs.",
                         dialogue=[{"speaker": "papi", "text": "Eight GPUs did it."}])["new_panels"][0]
    reply = {"new_panels": [cover_reply(after=6)["new_panels"][0], {**second, "after": 6}], "details": []}
    sb.brief = {**sb.brief, "points": [*POINTS[:3], {**POINTS[3], "priority": "core"}]}
    await StoryboarderV2(FakeEngine([reply])).cover(sb, make_cast(), (8, 9))
    assert len(sb.panels) == 9 and uncovered(sb) == []
    assert any("Trained on 8 P100 GPUs." in p.detail and "P4" in p.covers for p in sb.panels)


def test_points_are_normalised_and_counted_once():
    brief = {"points": [{"id": " P1 ", "priority": " Core "}, {"id": "P1", "priority": "core"}, {"id": "P2", "priority": "core"},
                        {"id": "P3", "priority": "important"}]}
    sb = StoryboardV2(title="T", summary="S", theme="kumomo", language="en-US", chapters=[], brief=brief,
                      panels=[PanelV2.from_dict({"n": 1, "covers": "P1, P3"}, 1)])
    assert sb.panels[0].covers == ["P1", "P3"]
    assert coverage(sb) == {"core": [1, 2], "important": [1, 1]} and [pt["id"] for pt in uncovered(sb)] == ["P2"]


def test_the_card_never_shows_none_python_reprs_or_unclear():
    card = paper_card({"problem": ["RNNs are sequential.", "Long paths are hard."], "key_idea": "unclear",
                       "mechanisms": [{"name": "Scaled attention"}, {"plain": "Several heads look at different things."}, {"name": None}],
                       "evidence": [{"claim": "Better BLEU", "numbers": ["28.4 EN-DE", "41.0 EN-FR"]}, {"claim": {"text": "nested"}},
                                    {"claim": "Faster", "numbers": "unclear"}],
                       "limitations": ["unclear", "Quadratic cost."], "takeaway": {"text": "x"}})
    assert card == {"problem": "RNNs are sequential.; Long paths are hard.",
                    "how": ["Scaled attention", "Several heads look at different things."],
                    "results": ["Better BLEU (28.4 EN-DE; 41.0 EN-FR)", "Faster"], "limitations": ["Quadratic cost."]}


async def test_reader_text_left_in_english_gets_one_small_re_request():
    sb = sb_with([["P1"], ["P2"]])
    reply = {"title": "注意力", "summary": "摘要", "chapters": [], "card": {"key_idea": "Still English."},
             "panels": [{"n": p.n, "title": "格", "dialogue": [{"speaker": b.speaker, "text": "中文台词"} for b in p.dialogue],
                         "narration": "中文旁白。" if p.narration else None, "labels": [], "detail": "Still English detail."} for p in sb.panels]}
    leftovers = {"items": [{"id": "1.x", "text": "第一格的细节。"}, {"id": "card.key_idea.0", "text": "让每个词同时关注所有其他词。"},
                           {"id": "2.x", "text": "Still English."}]}
    engine = FakeEngine([reply, leftovers])
    await StoryboarderV2(engine).translate(sb, "zh-CN")
    assert sb.panels[0].detail == "第一格的细节。" and sb.panels[1].detail == "Detail of panel 2."
    assert sb.card["key_idea"] == "让每个词同时关注所有其他词。" and len(engine.calls) == 2


def test_the_models_leaked_cls_placeholder_is_restored():
    """Seed wrote '<[BOS_never_used_51bce0…]>' for [CLS] in two BERT briefs; it reached a panel detail."""
    from services.storyboarder_v2 import extract_json
    reply = ('{"points": [{"id": "P5", "text": "binary classification is used on the<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> '
             'token output"}], "detail": "该任务使用<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>输出", "x": "a<[EOS_never_used_0a1b2c3d]>b"}')
    data = extract_json(reply)
    assert data["points"][0]["text"] == "binary classification is used on the [CLS] token output"
    assert data["detail"] == "该任务使用 [CLS]输出" and data["x"] == "ab"


# ----------------------------------------------------------------------------- brief check, people in visuals, chart direction

async def test_the_brief_is_checked_against_the_paper_before_anything_is_written_from_it():
    brief = json.loads(json.dumps({**BRIEF_WITH_POINTS, "limitations": ["Cost grows quadratically.", "It needs huge compute."],
                                   "evidence": [{"claim": "Single model wins", "numbers": "93.2 F1"}]}))
    engine = FakeEngine([{"problems": [
        {"id": "evidence.0.claim", "issue": "93.2 is the ensemble", "quote": "BERTLARGE (Ens.+TriviaQA) 93.2", "corrected": "The ensemble wins"},
        {"id": "limitations.1", "issue": "not stated by the authors", "corrected": "DELETE"},
        {"id": "points.P3", "issue": "missing setting", "corrected": "28.4 BLEU on WMT 2014 English-German (big model)."},
        {"id": "key_idea", "issue": "overstated", "corrected": "Attention can replace recurrence for translation."},
        {"id": "points.P99", "corrected": "ignored"}, {"id": "nonsense"}, "junk",
    ]}])
    assert await StoryboarderV2(engine).check_brief(brief, PaperText("T", "We propose the Transformer.", [])) == 4
    assert brief["evidence"][0] == {"claim": "The ensemble wins", "numbers": "93.2 F1"}
    assert brief["limitations"] == ["Cost grows quadratically."] and brief["key_idea"].startswith("Attention can replace")
    assert next(pt for pt in brief["points"] if pt["id"] == "P3")["text"].endswith("(big model).")
    assert [f["id"] for f in brief["brief_check"]] == ["evidence.0.claim", "limitations.1", "points.P3", "key_idea"]
    prompt = engine.prompts[0]
    assert '"id": "evidence.0.numbers"' in prompt and '"id": "points.P1"' in prompt and prompt.startswith("You work on one research paper") and "PAPER TEXT:\nWe propose the Transformer." in prompt


@pytest.mark.parametrize("reply", [RuntimeError("down"), "no json"])
async def test_a_failed_brief_check_keeps_the_brief(reply):
    brief = json.loads(json.dumps(BRIEF_WITH_POINTS))
    assert await StoryboarderV2(FakeEngine([reply, reply])).check_brief(brief, PaperText("T", "body", [])) == 0
    assert {k: v for k, v in brief.items() if k != "brief_check"} == BRIEF_WITH_POINTS


async def test_people_in_a_visual_are_rewritten_away_before_drawing():
    """'eight identical small librarians' and a 'Court' full of spectators were drawn in real runs."""
    sb = sb_with([["P1"], ["P2"]])
    sb.panels[0].scene, sb.panels[0].visual = "Library", "papi watches eight identical small librarians sort cards"
    engine = FakeEngine([{"items": [{"id": "1.v", "text": "papi sorts eight colored cards into a row in an empty library; nobody else is there"}]}])
    assert await StoryboarderV2(engine).fit(sb, "en-US") == 1
    assert (sb.panels[0].scene, sb.panels[0].visual) == ("", "papi sorts eight colored cards into a row in an empty library; nobody else is there")
    assert sb.panels[1].visual.startswith("Papi") or "librarian" not in sb.panels[1].visual


def test_detail_and_chart_rules_are_in_the_script_prompt():
    prompt = storyboard_prompt(BRIEF_WITH_POINTS, make_cast(), "en-US", (12, 18))
    assert "no judgments such as first, only, best, optimal, all, order of magnitude" in prompt
    assert "the larger or better one in the paper must look larger" in prompt


async def test_charts_and_comparisons_in_visuals_are_fact_checked():
    sb = sb_with([["P1"], ["P2"]])
    sb.panels[0].visual, sb.panels[1].visual = "a bar chart where the prior best bar is taller", "papi holds a lantern"
    engine = FakeEngine([{"problems": []}])
    await StoryboarderV2(engine).factcheck(sb, PaperText("T", "body", []))
    lines = engine.prompts[0].split("LINES:", 1)[1].split("PAPER TEXT:", 1)[0]
    assert '"id": "1.v"' in lines and '"id": "2.v"' not in lines


@pytest.mark.parametrize("text, people", [
    ("papi watches eight identical small librarians", True), ("a courtroom full of spectators", True),
    ("an empty library with no people", False), ("papi alone; nobody else is there, no other figures", False),
    ("papi holds eight colored cards", False),
])
def test_mentions_people(text, people):
    from services.storyboarder_v2 import _mentions_people
    assert _mentions_people(text) is people


async def test_brief_check_corrections_to_list_items_stick():
    """Found in a real run: a corrected contribution was recorded in brief_check but the card kept the old sentence."""
    brief = json.loads(json.dumps(BRIEF_WITH_POINTS))
    engine = FakeEngine([{"problems": [{"id": "contributions.0", "issue": "overstated", "corrected": "The Transformer, built from attention and feed-forward layers."},
                                       {"id": "limitations.0", "issue": "hedge", "corrected": "Cost may grow with the square of the length."}]}])
    assert await StoryboarderV2(engine).check_brief(brief, PaperText("T", "body", [])) == 2
    assert brief["contributions"] == ["The Transformer, built from attention and feed-forward layers."]
    assert paper_card(brief)["limitations"] == ["Cost may grow with the square of the length."]


def test_markdown_never_reaches_the_lettering():
    from services.storyboarder_v2 import _normalize_panels
    p = PanelV2.from_dict({"n": 1, "characters": ["papi"], "dialogue": [{"speaker": "papi", "text": "Use *only* attention."}],
                           "narration": "It beats **all** of them.", "labels": ["*key*"]}, 1)
    _normalize_panels([p], make_cast())
    assert (p.dialogue[0].text, p.narration, p.labels) == ("Use only attention.", "It beats all of them.", ["key"])


@pytest.mark.parametrize("text, flagged", [
    ("Eight identical small papis sit around a table", True), ("papi holds 8 tiny copies of himself", True),
    ("a mini kumo floats beside kumo", True), ("papi sorts eight colored cards", False), ("kumo floats over a desk", False),
])
def test_copies_of_a_cast_member_are_flagged_for_a_rewrite(text, flagged):
    from services.storyboarder_v2 import _mentions_people
    assert _mentions_people(text) is flagged


async def test_the_idea_a_picture_must_show_is_fact_checked_and_kept_free_of_people():
    sb = sb_with([["P1"], ["P2"]])
    sb.panels[1].shows = "the old method's stack is taller than the new one's"
    engine = FakeEngine([{"problems": [{"id": "2.s", "issue": "the paper's new method is the taller one",
                                        "corrected": "the new method's stack is taller than the old one's"}]}])
    assert await StoryboarderV2(engine).factcheck(sb, PaperText("T", "The new method scores higher.", [])) == 1
    assert sb.panels[1].shows == "the new method's stack is taller than the old one's" and '"id": "2.s"' in engine.prompts[0]
    from .test_story_v3 import clean_panels, v3_storyboard
    v3 = v3_storyboard(clean_panels())
    v3.panels[1].shows = "two students tie every card to every other card with red yarn"
    fit = FakeEngine([{"items": [{"id": "2.s", "text": "every card is tied to every other card with red yarn; nobody else is there"}]}])
    assert await StoryboarderV2(fit).fit(v3, "en-US") == 1
    assert v3.panels[1].shows.startswith("every card is tied") and '"id": "2.s"' in fit.prompts[0]


async def test_a_picture_whose_shows_alone_the_fact_check_corrected_gets_its_visual_rewritten_to_agree(monkeypatch):
    # 22 scenes of 24 books (rounds 14-15) had only one of the two corrected: the image model draws "visual", the art check
    # judges "shows", and LongZero's weaker-model pile was drawn the size the old "visual" said (scenes with a picture idea:
    # rounds 3-22; round 23's scenes have none)
    from services import storyboarder_v2
    monkeypatch.setattr(storyboarder_v2, "SCENE_SHOWS", True)
    from .test_story_v3 import clean_panels, v3_storyboard
    sb = v3_storyboard(clean_panels())
    sb.panels[1].kind, sb.panels[1].board = "scene", None
    sb.panels[1].visual = "papi sets the old method's tall stack of cards beside the new method's short stack"
    sb.panels[1].shows = "the old method's stack is taller than the new one's"
    story = StoryboarderV2(FakeEngine([
        {"problems": [{"id": "2.s", "issue": "the paper's new method is the taller one", "corrected": "the new method's stack is taller than the old one's"}]},
        {"panels": [{"n": 2, "scene": "a study", "visual": "papi sets the new method's tall stack of cards beside the old method's short stack",
                     "shows": "the tall stack is the new method's"}]}]))   # the reply rewords the corrected field too
    await story.factcheck(sb, PaperText("T", "The new method scores higher.", []))
    await story._sync_pictures(sb, make_cast())
    fix = story.text_engine.prompts[1]
    assert "Fix the pictures of these manga panels" in fix and 'the fact-check corrected \\"shows\\"' in fix and 'rewrite \\"visual\\"' in fix
    assert sb.panels[1].visual.startswith("papi sets the new method's tall stack") and story.last_fixes == []
    # the reply's two fields stand together (putting the checker's words back could split the pair again); the last check
    # reads the panel again
    assert sb.panels[1].shows == "the tall stack is the new method's"
    # both corrected, or neither: nothing to sync
    both = StoryboarderV2(FakeEngine([{"problems": [{"id": "2.s", "issue": "x", "corrected": "the left stack is taller"},
                                                    {"id": "2.v", "issue": "x", "corrected": "papi stacks the left pile taller"}]}]))
    await both.factcheck(sb, PaperText("T", "The new method scores higher.", []))
    await both._sync_pictures(sb, make_cast())
    assert len(both.text_engine.prompts) == 1


async def test_a_translations_fact_check_leaves_the_english_art_directions_alone():
    # a zh-CN check rewrote a shows line in Chinese (adding a teacher figure), which reached the image model and the idea check
    sb = sb_with([["P1"], ["P2"]])
    sb.language = "zh-CN"
    sb.panels[1].visual, sb.panels[1].shows = "papi stacks two piles of cards", "the left stack of cards is taller than the right"
    engine = FakeEngine([{"problems": [{"id": "2.s", "issue": "x", "corrected": "右边那堆更高，一位老师在旁边"},
                                       {"id": "2.v", "issue": "x", "corrected": "爸爸把两堆卡片摆好"}]}])
    await StoryboarderV2(engine).factcheck(sb, PaperText("T", "The new method scores higher.", []), "zh-CN")
    assert sb.panels[1].shows == "the left stack of cards is taller than the right" and sb.panels[1].visual == "papi stacks two piles of cards"
    assert '"id": "2.s"' not in engine.prompts[0] and '"id": "2.v"' not in engine.prompts[0]
