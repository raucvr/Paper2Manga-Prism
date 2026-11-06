"""services.storyboarder_v2, v3 story layer: abstract and its claims, story prompt, deterministic lint, fix rounds,
fact-check of board strings, the abstract-quiz gate, translation of boards, and v2 left as it was.

Fakes only; services.boards, services.evidence and services.abstract_quiz are used as they are."""

from __future__ import annotations

import json
import sys
from typing import Optional

import pytest

from services.paper_extractor import PaperText
from services.storyboarder_v2 import (
    BEAT_CHAPTER,
    _salvage_board,
    _story_board_issues,
    board_issues,
    _words_vs_chart,
    number_lines,
    scrub_unsupported,
    Bubble,
    Cast,
    CastMember,
    PanelV2,
    StoryboarderV2,
    StoryboardV2,
    brief_prompt_v3,
    coverage,
    find_abstract,
    fix_prompt,
    lint_v3,
    paper_examples,
    quote_in,
    split_sentences,
    story_prompt,
    story_range,
    translate_prompt,
    verify_claims,
    visible_text,
)

from .fakes import (
    BRIEF_V3,
    LONG_PAPER_TEXT,
    V3_CLAIMS,
    V3_EXAMPLE,
    V3_PAPER_TEXT,
    V3_RESULT,
    FakeEngine,
    PaperModel,
    fixed_panel,
    make_cast,
)


def v3_paper(text: str = V3_PAPER_TEXT) -> PaperText:
    return PaperText("Attention Is All You Need", text, ["Table 2: BLEU scores on the English-to-German newstest2014 test."])


def config_of(engine: FakeEngine, opening: str):
    return next(cfg for messages, cfg in engine.calls if messages[-1].content.startswith(opening))


def prompt_of(engine: FakeEngine, opening: str) -> str:
    return next(p for p in engine.prompts if opening in p)


def v3_panel(n: int, *, kind: str = "scene", beat: str = "how", dialogue=(("papi", "Attention looks at every word."),),
             narration: str = "", board=None, claims=(), characters=("papi", "kumo"), shows: Optional[str] = None) -> PanelV2:
    # a scene's picture shows its idea (the picture rule of lint_v3)
    shows = ("a row of blank cards, each tied to every other card by red yarn" if kind == "scene" else "") if shows is None else shows
    # ...and the visual draws the objects that "shows" names, somewhere new on each panel (a repeated picture is linted)
    place = ("", " on the desk", " on the floor", " on a shelf", " by the window", " on a bench", " on a rug", " on a tray")[(n - 1) % 8]
    visual = f"papi lays out a row of blank cards{place} and ties each card to every other card with red yarn." if shows else "papi points."
    return PanelV2(n=n, chapter=BEAT_CHAPTER[beat], title=f"Panel {n}", characters=list(characters), scene="A study.",
                   visual=visual, dialogue=[Bubble(s, t) for s, t in dialogue], narration=narration, kind=kind,
                   board=board, beat=beat, claims=list(claims), shows=shows)


def v3_storyboard(panels, claims=V3_CLAIMS, brief=None) -> StoryboardV2:
    return StoryboardV2(title="T", summary="S", theme="kumomo", language="en-US", chapters=[], panels=panels, brief=dict(brief or {}),
                        version=3, abstract_claims=[dict(c) for c in claims])


# ----------------------------------------------------------------------------- abstract

def test_find_abstract_stops_at_footnotes_venue_and_first_section():
    body = "\n".join([
        "Attention Is All You Need", "Ashish Vaswani∗ Noam Shazeer∗", "Abstract",
        "The dominant sequence transduction models are based on complex recurrent or",
        "convolutional neural networks. Our model achieves 28.4 BLEU on the WMT 2014",
        "8 GPUs",   # a wrapped line that looks like a numbered heading but continues the sentence
        "and generalizes well.",
        "∗Equal contribution. Listing order is random. Jakob proposed replacing RNNs with self-attention and started",
        "the effort to evaluate this idea.",
        "31st Conference on Neural Information Processing Systems (NIPS 2017), Long Beach, CA, USA.",
        "1 Introduction", "Recurrent neural networks ...",
    ])
    abstract = find_abstract(body)
    assert abstract.startswith("The dominant sequence transduction models") and abstract.endswith("and generalizes well.")
    assert "28.4 BLEU on the WMT 2014 8 GPUs" in abstract
    assert "Equal contribution" not in abstract and "Conference" not in abstract and "Introduction" not in abstract


@pytest.mark.parametrize("lines, expected", [
    (["Title", "Abstract—We propose a model. It works.", "1. Introduction", "Text."], "We propose a model. It works."),
    (["Title", "ABSTRACT We propose a model.", "I. INTRODUCTION"], "We propose a model."),
    (["Title", "A B S T R A C T", "We propose a model.", "Keywords: attention"], "We propose a model."),
    # a figure placed in the abstract's column: its heading ends the abstract
    (["Abstract", "We explore prompting.", "It helps.", "Standard Prompting", "Q: Roger has 5 tennis balls."], "We explore prompting. It helps."),
    (["Abstract", "We explore prompting.", "Example sentence shown in a figure: It is in this spirit ."], "We explore prompting."),
    (["Title", "Abstract reasoning is hard for models.", "1 Introduction"], ""),   # a sentence, not a heading
    (["Title", "1 Introduction", "No abstract here."], ""),
])
def test_find_abstract_variants(lines, expected):
    assert find_abstract("\n".join(lines)) == expected


def test_find_abstract_on_real_papers(attention_paper, cot_paper):
    attention = find_abstract(attention_paper.body)
    assert attention.startswith("The dominant sequence transduction models")
    assert attention.endswith("both with large and limited training data.") and "Equal contribution" not in attention
    assert "28.4 BLEU" in attention and "41.8" in attention
    cot = find_abstract(cot_paper.body)
    assert cot.startswith("We explore how generating a chain of thought") and cot.endswith("GPT-3 with a verifier.")
    assert "Standard Prompting" not in cot and "Roger" not in cot


def test_split_sentences_keeps_numbers_and_abbreviations():
    text = "We reach 28.4 BLEU, e.g. on WMT 2014. It costs 3.5 days. Fig. 3 shows heads."
    assert split_sentences(text) == ["We reach 28.4 BLEU, e.g. on WMT 2014.", "It costs 3.5 days.", "Fig. 3 shows heads."]


# ----------------------------------------------------------------------------- claims and examples

ABSTRACT = ("The dominant models are recurrent. We propose the Transformer, based solely on attention mechanisms. "
            "Our model achieves 28.4 BLEU on the WMT 2014 Englishto-German translation task, improving over the existing best "
            "results by over 2 BLEU. It also generalizes to English constituency parsing.")


def test_verify_claims_checks_quotes_reanchors_drops_and_covers_every_sentence():
    raw = [
        {"id": 7, "text": "28.4 BLEU on English-German", "quote": "achieves 28.4 BLEU on the WMT 2014 English-to-German translation task"},  # lost hyphen
        {"id": 0, "text": "Attention only", "quote": "We propose the Transformer, based solely on attention mechanisms"},
        {"id": 1, "text": "Works on parsing", "quote": "It also generalizes very well to English constituency parsing tasks"},  # paraphrased quote
        {"id": 2, "text": "Invented", "quote": "Our model wins every benchmark ever made by a huge margin"},                   # not in the abstract
        "The dominant models are recurrent.",                                                                                   # a bare string
    ]
    claims = verify_claims(raw, ABSTRACT, body="unused when there is an abstract")
    assert [c["id"] for c in claims] == list(range(len(claims)))
    assert [c["quote"] for c in claims] == [   # abstract order, not reply order
        "The dominant models are recurrent.",
        "We propose the Transformer, based solely on attention mechanisms",
        # the results sentence's "over 2 BLEU" was in no quote: the whole sentence becomes a claim of its own
        "Our model achieves 28.4 BLEU on the WMT 2014 Englishto-German translation task, improving over the existing best results by over 2 BLEU.",
        "achieves 28.4 BLEU on the WMT 2014 English-to-German translation task",   # verified despite the lost hyphen
        "It also generalizes to English constituency parsing.",                    # paraphrased quote re-anchored on its sentence
    ]
    assert "Invented" not in [c["text"] for c in claims]    # dropped: nothing in the abstract supports it
    assert [c["text"] for c in claims][3:] == ["28.4 BLEU on English-German", "Works on parsing"]
    assert all(quote_in(c["quote"], ABSTRACT) for c in claims)


def test_verify_claims_without_an_abstract_checks_quotes_against_the_paper():
    claims = verify_claims([{"text": "a", "quote": "the big transformer model outperforms the best previously reported models"},
                            {"text": "b", "quote": "this sentence is nowhere in the paper at all"}], "", LONG_PAPER_TEXT)
    assert [c["text"] for c in claims] == ["a"]


def test_paper_examples_keeps_only_examples_found_in_the_paper():
    examples = paper_examples([
        {"text": "Example sentence shown in a figure: " + V3_EXAMPLE, "source": "Figure 3"},
        {"text": "A made-up example the paper never shows."}, V3_EXAMPLE, 42,
    ], V3_PAPER_TEXT)
    assert examples == [{"text": V3_EXAMPLE, "source": "Figure 3"}]


def test_story_range_keeps_v3_stories_at_16_to_24_panels_unless_asked_for_more():
    assert story_range((16, 32)) == (16, 24)
    assert story_range((12, 20)) == (16, 20)
    assert story_range((3, 6)) == (16, 16)
    assert story_range((28, 40)) == (28, 40)


# ----------------------------------------------------------------------------- prompts

def test_brief_prompt_v3_shows_the_abstract_and_asks_for_claims_examples_and_glossary():
    prompt = brief_prompt_v3(v3_paper(), "THE ABSTRACT TEXT.")
    assert prompt.startswith("Read the paper and its ABSTRACT and write a compact, faithful BRIEF")
    assert "ABSTRACT:\nTHE ABSTRACT TEXT." in prompt and V3_PAPER_TEXT in prompt
    for needle in ('"abstract_claims": [{"id": 0, "text"', '"quote"', "EVERY sentence", '"examples"', "Example sentence shown in a figure:",
                   '"glossary"', "higher or lower is better"):
        assert needle in prompt, needle
    assert '"points"' not in prompt   # v3 measures coverage on the abstract claims
    assert "no abstract heading was found" in brief_prompt_v3(v3_paper(), "")


def test_story_prompt_carries_claims_beats_boards_budgets_and_fidelity_rules():
    brief = {**BRIEF_V3, "examples": [{"text": V3_EXAMPLE, "source": "Figure 3"}]}
    prompt = story_prompt(brief, V3_CLAIMS, make_cast(), (16, 24), v3_paper())
    assert prompt.startswith("Turn this research paper into a SHORT MANGA STORY")
    assert json.dumps(V3_CLAIMS, ensure_ascii=False) in prompt and V3_PAPER_TEXT in prompt
    assert f"- {V3_EXAMPLE} (Figure 3)" in prompt and '"from_paper": true' in prompt
    for beat in ("hook", "problem", "idea", "how", "proof", "limits", "payoff", "recap"):
        assert f"{beat} (" in prompt, beat
    for kind in ("bar", "table", "flow", "compare", "links", "list"):
        assert f'{kind}: {{"kind": "{kind}"' in prompt, kind
    for needle in ("at most 3 speech bubbles of at most 16 words", "narration at most 20 words", "1-2 cast members",
                   "1-2 speech bubbles of at most 12 words", "narration at most 14 words", "16-24 panels",
                   "at most 44 words of bubbles and narration in total", '"baseline": true', '"open": true',
                   "claims (exactly 1 board panel", "bets something small", "No verdict words", "every claim must ALSO be taught",
                   '"paper_label"', "prior best", "single model or ensemble", "Never compute, round, convert",
                   "which direction is better", "Never overgeneralize", "Limitations only as the authors state them",
                   "nezu is a sharp reviewer", "kumo is the reader's stand-in", '"claims"'):
        assert needle in prompt, needle
    assert '"abstract_claims"' not in prompt.split("BRIEF (checked against the paper):", 1)[1].split("PAPER TEXT:", 1)[0]


def test_story_prompt_uses_the_casts_roles_and_labels_an_illustrative_example():
    cast = Cast(theme="custom", style="s", members=[CastMember("mochi", "mentor", "l", "v"), CastMember("pip", "student", "l", "v"),
                                                    CastMember("rook", "skeptic", "l", "v")])
    prompt = story_prompt({**BRIEF_V3, "examples": []}, V3_CLAIMS, cast, (14, 18), v3_paper())
    assert "pip is the reader's stand-in" in prompt and "rook is a sharp reviewer" in prompt and "mochi knows the paper" in prompt
    assert "The paper shows no example input" in prompt and '"from_paper": false' in prompt


# ----------------------------------------------------------------------------- lint

TABLE = {"kind": "table", "title": "WMT 2014 translation (BLEU)", "subtitle": "WMT: a yearly translation contest; BLEU: match with human translations",
         "columns": ["Task", "BLEU"],
         "rows": [{"cells": ["English-German", "28.4"], "quote": "establishing a new state-of-the-art BLEU score of 28.4"},
                  {"cells": ["English-French", "41.0"], "quote": "our big model achieves a BLEU score of 41.0"}], "source": "Section 6"}
RECAP = {"kind": "list", "title": "What the paper showed", "items": ["Attention only", "28.4 BLEU on WMT 2014 English-German"]}
CLAIMS_WITH_RESULT = V3_CLAIMS + [{"id": 3, "text": "28.4 BLEU, over 2 BLEU better", "quote": V3_RESULT}]


def clean_panels() -> list[PanelV2]:
    return [
        v3_panel(1, beat="hook", narration="The best translators were complex recurrent or convolutional networks with an encoder and a decoder.",
                 claims=[0]),
        v3_panel(2, beat="idea", dialogue=[("papi", "The Transformer, a simple architecture: only attention, so no recurrence and no convolutions.")],
                 claims=[1, 2]),
        v3_panel(3, kind="board", beat="proof", board=TABLE, dialogue=[("nezu", "28.4, more than 2.0 above the rest?")], claims=[3],
                 characters=("nezu", "papi")),
        v3_panel(4, kind="board", beat="recap", board=RECAP, dialogue=[("nezu", "Fine. Convinced.")], claims=[0, 1, 2, 3],
                 characters=("nezu",)),
    ]


def test_lint_passes_a_clean_script():
    assert lint_v3(v3_storyboard(clean_panels(), CLAIMS_WITH_RESULT), V3_PAPER_TEXT) == {}


def test_lint_catches_an_unsupported_number_and_an_uncovered_claim():
    panels = clean_panels()
    panels[0].narration = "Recurrent models needed 99.9 hours to translate a sentence."
    for p in panels:
        p.claims = [c for c in p.claims if c != 2]
    sb = v3_storyboard(panels, CLAIMS_WITH_RESULT)
    problems = lint_v3(sb, V3_PAPER_TEXT)
    assert any(x.startswith("evidence: narration:") and "99.9" in x for x in problems[1])
    homes = [n for n, found in problems.items() if any(x.startswith("claim 2 is not taught by any panel") for x in found)]
    assert len(homes) == 1 and sb.panels[homes[0] - 1].beat not in ("claims", "recap")   # a story panel, never the recap
    assert set(problems) == {1, homes[0]}


def test_lint_counts_a_claim_on_the_recap_or_claims_board_as_not_taught():
    panels = clean_panels()
    recap = next(p for p in panels if p.beat == "recap")
    for p in panels:
        p.claims = [c for c in p.claims if c != 2 or p is recap]
    if 2 not in recap.claims:
        recap.claims.append(2)
    problems = lint_v3(v3_storyboard(panels, CLAIMS_WITH_RESULT), V3_PAPER_TEXT)
    assert any(x.startswith("claim 2 is not taught") for found in problems.values() for x in found)
    assert recap.n not in problems


def test_lint_catches_kind_mismatches_budgets_unsupported_board_values_and_missing_key_numbers():
    long_bubble = " ".join(["word"] * 17)
    bad_bar = {"kind": "bar", "title": "Made-up chart", "bars": [{"label": "A", "value": "77.7", "highlight": True}, {"label": "B", "value": "66.6"}]}
    panels = [
        v3_panel(1, beat="hook", dialogue=[("kumo", "a"), ("papi", "b"), ("kumo", "c"), ("papi", "d")], claims=[0, 1, 2], shows=""),
        v3_panel(2, kind="board", beat="proof", board=None, claims=[3]),
        v3_panel(3, kind="board", beat="proof", board=bad_bar, dialogue=[("nezu", long_bubble)], characters=("nezu", "papi", "kumo")),
        v3_panel(4, beat="how", board=RECAP, dialogue=[("papi", "x")]),
        v3_panel(5, kind="board", beat="recap", board={"kind": "list", "title": "Recap", "items": ["Attention only", "Fast"]},
                 dialogue=[], claims=[0, 1, 2, 3], characters=("nezu",)),
    ]
    problems = lint_v3(v3_storyboard(panels, CLAIMS_WITH_RESULT), V3_PAPER_TEXT)
    assert [x for x in problems[1] if not x.startswith("claim ")] == ["4 speech bubbles: at most 3 on a scene panel"]   # "a", "b": nothing taught
    assert problems[2][0].startswith("board panel without a board")
    assert any(x.startswith("evidence:") and "77.7" in x for x in problems[3])
    assert "3 cast members on a board panel: at most 2" in problems[3]
    assert "bubble 0 (nezu) has 17 words: at most 12 on a board panel" in problems[3]
    assert [x for x in problems[4] if not x.startswith(("term:", "why:", "diagram:"))] == [
        'scene panel with a board: make it a board panel ("kind": "board") or remove the board']   # its list names BLEU, unexplained
    assert "board panel without a speech bubble: a cast member reacts to the board" in problems[5]
    # claim 3 is listed by panels 2 and 5; the recap (5) does not count, and the story panel (2) shows neither 28.4 nor "over 2"
    assert any(x.startswith("claim 3") and "28.4, 2" in x for x in problems[2])
    assert not any(x.startswith("claim 3") for x in problems[5])


def test_lint_allows_the_numbers_of_an_illustrative_running_example_only():
    panels = clean_panels()
    panels[0].narration = "Kumo bakes a layer cake with 12 recipe cards and waits 45 minutes."
    sb = v3_storyboard(panels, CLAIMS_WITH_RESULT, brief={"running_example": {"text": "Kumo bakes a 12 layer cake", "from_paper": False}})
    sb.running_example = "Kumo bakes a 12 layer cake"
    problems = lint_v3(sb, V3_PAPER_TEXT)
    assert list(problems) == [1] and not any("12" in x for x in problems[1]) and any("45" in x for x in problems[1])


def test_fix_prompt_lists_flagged_panels_with_problems_and_attaches_the_paper_when_numbers_are_involved():
    sb = v3_storyboard(clean_panels(), CLAIMS_WITH_RESULT)
    prompt = fix_prompt(sb, make_cast(), {3: ["evidence: rows.0.cells.1: value 28.4 is not supported"]}, paper_text=V3_PAPER_TEXT)
    assert prompt.startswith("Fix the flagged panels of this manga script")
    flagged = json.loads(prompt.split("FLAGGED PANELS:\n", 1)[1].split("\n\n", 1)[0])
    assert [f["panel"]["n"] for f in flagged] == [3] and flagged[0]["problems"] == ["evidence: rows.0.cells.1: value 28.4 is not supported"]
    assert flagged[0]["panel"]["board"]["rows"][0]["quote"] == TABLE["rows"][0]["quote"]
    assert prompt.endswith(V3_PAPER_TEXT) and '"kind": "table", "title": "WMT 2014 translation (BLEU)"}' in prompt   # outline: kind + title
    assert "PAPER TEXT:\n" not in fix_prompt(sb, make_cast(), {1: ["narration has 31 words: at most 30 on a scene panel"]})


# ----------------------------------------------------------------------------- build_v3

async def test_build_v3_produces_a_v3_storyboard_with_boards_claims_chapters_and_quiz():
    model = PaperModel()
    engine = FakeEngine(router=model)
    stages = []
    sb = await StoryboarderV2(engine).build_v3(v3_paper(), make_cast(), "en-US", (16, 32), on_stage=lambda s, m: stages.append(s))

    # the 9-panel draft is well under the 16-24 range, so the story is asked for once more (no longer: the first stands);
    # lint, the reader check and the board check found nothing to fix; the idea chapter's diagram is designed once; the
    # fact-check reads the script after them; no scene has a picture idea to look at (round 23: scenes tell, boards explain);
    # the recap evidence reads the final script
    assert model.seen == ["brief_v3", "briefcheck", "story", "story", "diagram", "clarity", "boardcheck", "factcheck", "quiz", "reader",
                          "grade", "grade", "grade", "recap_evidence"]
    assert "A FIRST DRAFT OF THIS STORY HAD ONLY 9 PANELS" in [p for p in engine.prompts if "SHORT MANGA STORY" in p][1]
    assert all(p.shows == "" for p in sb.panels) and sb.panels[2].board["kind"] == "diagram" and sb.panels[2].board["elements"]
    assert list(dict.fromkeys(stages)) == ["analyzing", "storyboarding"]
    assert (sb.version, sb.language) == (3, "en-US") and [p.n for p in sb.panels] == list(range(1, 10))
    # the brief's three claims plus the abstract's results sentence, which no claim covered
    assert [c["id"] for c in sb.abstract_claims] == [0, 1, 2, 3] and sb.abstract_claims[3]["quote"] == V3_RESULT
    assert "abstract_claims" not in sb.brief and sb.brief["abstract"].endswith("by over 2 BLEU.")
    assert sb.brief["examples"] == [{"text": V3_EXAMPLE, "source": "Figure 3"}]
    assert sb.brief["running_example"]["from_paper"] is True and V3_EXAMPLE in sb.running_example
    boards = {p.n: p.board for p in sb.panels if p.kind == "board"}
    assert {n: b["kind"] for n, b in boards.items()} == {3: "diagram", 4: "flow", 6: "table", 9: "list"}
    assert boards[6]["rows"][0] == {"cells": ["English-German", "28.4"], "quote": "establishing a new state-of-the-art BLEU score of 28.4"}
    assert all(p.board is None for p in sb.panels if p.kind == "scene") and all(p.labels == [] for p in sb.panels)
    assert [(c.id, c.panel_numbers) for c in sb.chapters] == [("hook", [1]), ("problem", [2]), ("idea", [3]), ("how", [4]),
                                                             ("evidence", [5, 6]), ("payoff", [8]), ("limits", [7]), ("takeaway", [9])]
    assert [p.chapter for p in sb.panels] == [BEAT_CHAPTER[p.beat] for p in sb.panels]
    assert (sb.quiz["score"], sb.quiz["failed_claim_ids"], len(sb.quiz["questions"])) == (1.0, [], 8) and "usage" not in sb.quiz
    assert coverage(sb) == {"core": [4, 4], "important": [0, 0], "claims": [4, 4]} and sb.brief["lint"] == {}
    # what each step left, so a problem that ships can be traced to the step that brought it
    assert [(t["after"], t["found"]) for t in sb.brief["lint_trail"]] == [
        ("write", {}), ("fix", {}), ("pictures", {}), ("check", {}), ("quiz", {}), ("pictures again", {}), ("last check", {}),
        ("scrub", {})]
    saved = sb.to_dict()
    assert (saved["version"], saved["running_example"], saved["quiz"]["score"]) == (3, sb.running_example, 1.0)
    assert saved["panels"][5]["kind"] == "board" and saved["panels"][5]["claims"] == [3]
    # thinking for the brief and the story; the quiz has its own settings
    assert config_of(engine, "Read the paper and its ABSTRACT").thinking is True
    assert config_of(engine, "Turn this research paper").thinking is True


async def test_build_v3_fix_round_repairs_an_unsupported_number_and_an_uncovered_claim():
    def story(payload):
        payload["panels"][1]["narration"] = "Recurrent models needed 99.9 hours for one long sentence."
        for p in payload["panels"]:
            p["claims"] = [c for c in p["claims"] if c != 2]
        return payload

    model = PaperModel(story=story)
    engine = FakeEngine(router=model)
    writer = StoryboarderV2(engine)
    sb = await writer.build_v3(v3_paper(), make_cast(), "en-US", (16, 32))

    assert model.seen.count("fix") == 1
    fix = prompt_of(engine, "Fix the flagged panels")
    flagged = {f["panel"]["n"]: f["problems"] for f in json.loads(fix.split("FLAGGED PANELS:\n", 1)[1].split("\n\n", 1)[0])}
    homes = [n for n, found in flagged.items() if any(x.startswith("claim 2 is not taught") for x in found)]
    assert len(homes) == 1 and set(flagged) == {2, homes[0]} and homes[0] != 9   # a story panel, not the recap (9)
    assert any("99.9" in x for x in flagged[2])
    assert "PAPER TEXT:\n" in fix and config_of(engine, "Fix the flagged panels").thinking is False
    assert "99.9" not in sb.panels[1].narration and 2 in sb.panels[homes[0] - 1].claims
    assert coverage(sb)["claims"] == [4, 4] and sb.brief["lint"] == {}
    assert writer.usage["prompt_tokens"] > 0


async def test_build_v3_demotes_a_board_that_stays_invalid_after_two_fix_rounds():
    made_up = {"kind": "bar", "title": "Made-up speed chart", "highlight": True,
               "bars": [{"label": "Old", "value": "77.7"}, {"label": "New", "value": "66.6", "highlight": True}]}

    def story(payload):
        payload["panels"][5]["board"] = made_up
        return payload

    model = PaperModel(story=story, fix=lambda panel, problems: panel)   # a writer that never fixes anything
    sb = await StoryboarderV2(FakeEngine(router=model)).build_v3(v3_paper(), make_cast(), "en-US", (16, 32))

    assert model.seen.count("fix") == 2 and model.seen.count("pictures") == 1   # then the demoted panel's picture: it still points at its board
    panel = sb.panels[5]
    assert (panel.kind, panel.board) == ("scene", None)
    assert "Made-up speed chart" not in panel.narration and panel.dialogue   # no caption for a chart nobody sees
    assert all(p.kind == "scene" or p.board for p in sb.panels)


async def test_build_v3_translates_board_strings_and_keeps_numbers_and_quotes():
    model = PaperModel()
    engine = FakeEngine(router=model)
    stages = []
    sb = await StoryboarderV2(engine).build_v3(v3_paper(), make_cast(), "zh-CN", (16, 32), on_stage=lambda s, m: stages.append(s))

    assert model.seen[-2:] == ["translate", "factcheck"] and list(dict.fromkeys(stages))[-1] == "translating"
    assert sb.language == "zh-CN" and sb.version == 3
    table, recap = sb.panels[5].board, sb.panels[8].board
    assert table["title"] == "板书 2014" and table["columns"] == ["板书", "板书"]
    assert table["rows"][0] == {"cells": ["板书", "28.4"], "quote": "establishing a new state-of-the-art BLEU score of 28.4"}   # values and quotes stay
    assert recap["items"][-1] == "板书 28.4 2014"
    assert sb.panels[3].board["steps"] == ["板书"] * 4
    assert [c.title for c in sb.chapters][:2] == ["第1章", "第2章"]   # chapter titles translated, chapters from beats kept
    prompt = prompt_of(engine, "Translate this manga script")
    payload = json.loads(prompt.rsplit("\n\n", 1)[1])
    assert payload["panels"][5]["board"]["rows.0.cells.0"] == "English-German"
    assert "rows.0.cells.1" not in payload["panels"][5]["board"]   # numbers are not sent for translation
    assert "on a panel with a board, each bubble ≤ 22" in prompt and 'every string value of a panel\'s "board"' in prompt
    assert config_of(engine, "Translate this manga script").thinking is False


async def test_build_v3_quiz_gate_revises_failed_claims_and_stores_both_scores():
    model = PaperModel(quiz_misses=({2},))   # the first reader cannot answer the questions about claim 2
    engine = FakeEngine(router=model)
    sb = await StoryboarderV2(engine).build_v3(v3_paper(), make_cast(), "en-US", (16, 32))

    assert model.seen.count("quiz") == 1 and model.seen.count("reader") == 2 and model.seen.count("revise") == 1
    assert sb.quiz["before"] == {"score": 0.75, "failed_claim_ids": [2]}
    assert (sb.quiz["score"], sb.quiz["failed_claim_ids"]) == (1.0, [])
    new = next(p for p in sb.panels if p.title == "Claim 2")
    assert (new.n, new.beat, new.chapter, new.claims) == (7, "proof", "evidence", [2])   # after the last proof panel
    assert [p.n for p in sb.panels] == list(range(1, 11))
    assert dict((c.id, c.panel_numbers) for c in sb.chapters)["evidence"] == [5, 6, 7]
    revise = prompt_of(engine, "A reader who never saw the paper read only the visible text")
    assert '"reader_answered": "not shown"' in revise and "PAPER TEXT:\n" in revise
    assert model.seen.count("factcheck") == model.seen.count("clarity") == 2   # the revised script is checked again


async def test_build_v3_quiz_gate_runs_a_second_round_for_claims_still_missing():
    # Driver Misconduct: one revision lifted the quiz from 18% to 55% and left the background sentences untaught
    model = PaperModel(quiz_misses=({2}, {2}, set()))
    sb = await StoryboarderV2(FakeEngine(router=model)).build_v3(v3_paper(), make_cast(), "en-US", (16, 32))
    assert model.seen.count("revise") == 2 and model.seen.count("reader") == 3 and model.seen.count("quiz") == 1   # same questions
    assert sb.quiz["before"] == {"score": 0.75, "failed_claim_ids": [2]}
    assert sb.quiz["rounds"] == [{"score": 0.75, "failed_claim_ids": [2]}, {"score": 1.0, "failed_claim_ids": []}]
    assert (sb.quiz["score"], sb.quiz["failed_claim_ids"]) == (1.0, []) and "undone" not in sb.quiz
    assert [p.n for p in sb.panels] == list(range(1, len(sb.panels) + 1))


async def test_build_v3_undoes_a_second_round_that_lowers_the_score():
    model = PaperModel(quiz_misses=({2}, {2}, {1, 2}))
    story = StoryboarderV2(FakeEngine(router=model))
    sb = await story.build_v3(v3_paper(), make_cast(), "en-US", (16, 32))
    assert model.seen.count("revise") == 2
    assert sb.quiz["undone"] == 2 and sb.quiz["rounds"][-1] == {"score": 0.5, "failed_claim_ids": [1, 2]}
    assert (sb.quiz["score"], sb.quiz["failed_claim_ids"]) == (0.75, [2])   # the first round's script and result stay
    assert [p.title for p in sb.panels].count("Claim 2") == 1                 # the second round's panel is gone


async def test_build_v3_sends_what_a_newcomer_stumbles_on_to_a_fix_round():
    problems = [{"n": 2, "kind": "term", "problem": "'attention' is never explained", "fix": "say attention lets each word look at the others"},
                {"n": 2, "kind": "pointer", "problem": "'this' points at nothing", "fix": "name the model"},
                {"n": 2, "kind": "order", "problem": "a third problem on one panel", "fix": "x"},   # at most 2 per panel
                {"n": 99, "kind": "term", "problem": "no such panel", "fix": "x"},
                {"n": 1, "kind": "vibes", "problem": "the opening is slow", "fix": ""}]
    model = PaperModel(clarity_problems=problems)
    engine = FakeEngine(router=model)
    sb = await StoryboarderV2(engine).build_v3(v3_paper(), make_cast(), "en-US", (16, 32))
    fix = prompt_of(engine, "Fix the flagged panels of this manga script")
    assert "reader (term): 'attention' is never explained Fix: say attention lets each word look at the others" in fix
    assert "reader (pointer): 'this' points at nothing" in fix and "a third problem" not in fix and "no such panel" not in fix
    assert "reader (clarity): the opening is slow" in fix   # an unknown kind is kept under a neutral name
    assert '"reader" problems: a reader who never saw the paper stumbled here' in fix and "PAPER TEXT:\n" in fix
    assert model.seen.index("fix") < model.seen.index("factcheck")   # the fact-check reads the reader's rewrites
    assert [(c["n"], c["kind"]) for c in sb.brief["clarity"]] == [(2, "term"), (2, "pointer"), (1, "clarity")]
    clarity = prompt_of(engine, "You are a curious reader who never saw this research paper")
    assert '"bubbles": ["papi:' in clarity and "GLOSSARY" in clarity and "PAPER TEXT" not in clarity   # the script alone


async def test_build_v3_sends_boards_that_do_not_back_their_words_to_the_same_fix_round():
    # the audit of 2026-10-28: 7 of 34 boards fully backed their words (a badge against the base model beside "beat both
    # baselines", fractions beside "percentage points")
    problems = [{"n": 6, "kind": "comparison", "problem": "the words say 'above the best earlier models' but no rival is drawn",
                 "fix": "add ConvS2S Ensemble from Table 2 and mark it as the baseline"},
                {"n": 6, "kind": "units", "problem": "BLEU has no scale", "fix": "say BLEU runs from 0 to 100"},
                {"n": 6, "kind": "label", "problem": "a third problem on one board", "fix": "x"},   # at most 2 per board
                {"n": 2, "kind": "label", "problem": "a scene has no board", "fix": "x"}]
    clarity = [{"n": 6, "kind": "term", "problem": "'BLEU' is never explained", "fix": "explain BLEU"}]
    model = PaperModel(board_problems=problems, clarity_problems=clarity)
    engine = FakeEngine(router=model)
    sb = await StoryboarderV2(engine).build_v3(v3_paper(), make_cast(), "en-US", (16, 32))
    fix = prompt_of(engine, "Fix the flagged panels of this manga script")
    assert "board check (comparison): the words say 'above the best earlier models' but no rival is drawn Fix: add ConvS2S" in fix
    assert "board check (comparison): BLEU has no scale" in fix   # an unknown kind is kept under the main one
    assert "reader (term): 'BLEU' is never explained" in fix and "a third problem" not in fix and "a scene has no board" not in fix
    assert '"board check" problems: the board must show exactly what' in fix and "PAPER TEXT:\n" in fix
    assert model.seen.count("fix") >= 1 and model.seen.index("fix") < model.seen.index("factcheck")
    assert [(c["n"], c["kind"]) for c in sb.brief["board_check"]] == [(6, "comparison"), (6, "comparison")]
    check = prompt_of(engine, "Check every board of this manga script")
    assert '"n": 6' in check and check.startswith("You work on one research paper") and "PAPER TEXT:\n" in check   # the paper first
    # Longzero round 13: "add a gap badge for both baselines" came back every round; a chart draws one badge
    assert "at most ONE gap badge" in check and "narrows the words to what the board shows" in check
    assert 'for "beats both baselines" mark the stronger one' in fix


async def test_a_board_the_fix_rewrote_is_checked_once_more_and_keeps_the_abstract_claims_in_its_words():
    # round 17: half the problems the boards kept were on boards the board check had flagged once (LongZero's "2 LLMs"
    # beside one model row was asked for three times); round 18 told that second look to narrow the words to the board,
    # and OnQ's "shorter answers" left the book (its fixed quiz fell from 0.95 to 0.68)
    first = [{"n": 6, "kind": "comparison", "problem": "the words say 'two models' but one row is drawn", "fix": "add the second row"}]
    still = [{"n": 6, "kind": "comparison", "problem": "still one row for 'two models'", "fix": "say one model in the words"}]

    class Twice(PaperModel):
        checks = 0

        def __call__(self, prompt, messages, config):
            if messages[-1].content.startswith("Check every board of this manga script"):
                self.checks += 1
                self.board_problems = first if self.checks == 1 else still if self.checks == 2 else []
            return super().__call__(prompt, messages, config)

    def fix(panel, problems):
        out = fixed_panel(panel, problems)
        if isinstance(out.get("board"), dict) and any(x.startswith("board check") for x in problems):
            out["board"]["subtitle"] = f"{out['board'].get('subtitle') or ''} (rewritten)".strip()
        return out

    model = Twice(fix=fix)
    engine = FakeEngine(router=model)
    sb = await StoryboarderV2(engine).build_v3(v3_paper(), make_cast(), "en-US", (16, 32))
    checks = [p for p in engine.prompts if "Check every board of this manga script" in p]
    assert len(checks) >= 2 and "were just rewritten" not in checks[0]
    assert "were just rewritten" in checks[1] and "adding the missing rows or bars did not work" not in checks[1]
    assert all("The paper also reports shorter answers." in c and "a part of the panel's claims never leaves the words" in c
               for c in checks)
    fixes = [p for p in engine.prompts if "Fix the flagged panels of this manga script" in p]
    assert any("still one row for 'two models'" in f for f in fixes)
    assert all("a part of this panel's abstract claims stays in the words" in f for f in fixes)
    assert sb.brief["board_check"][0]["problem"] == first[0]["problem"]
    assert [c["problem"] for c in sb.brief["board_recheck"]] == [still[0]["problem"]]   # counted apart in the job record


async def test_a_board_check_that_found_nothing_looks_once_even_when_the_reader_check_rewrote_a_board():
    def fix(panel, problems):
        out = fixed_panel(panel, problems)
        if isinstance(out.get("board"), dict) and any(x.startswith("reader") for x in problems):
            out["board"]["subtitle"] = f"{out['board'].get('subtitle') or ''} (rewritten)".strip()
        return out

    model = PaperModel(fix=fix, clarity_problems=[{"n": 6, "kind": "term", "problem": "BLEU is never explained", "fix": "say what BLEU is"}])
    engine = FakeEngine(router=model)
    sb = await StoryboarderV2(engine).build_v3(v3_paper(), make_cast(), "en-US", (16, 32))
    fixes = [p for p in engine.prompts if "Fix the flagged panels of this manga script" in p]
    assert any("BLEU is never explained" in f for f in fixes)   # the board was rewritten for the reader
    assert sb.panels and model.seen.count("boardcheck") == 1 and not sb.brief.get("board_recheck")


async def test_a_board_check_that_cannot_run_is_recorded():
    class LostCheck(PaperModel):
        def __call__(self, prompt, messages, config):
            if messages[-1].content.startswith("Check every board of this manga script"):
                raise RuntimeError("checker down")
            return super().__call__(prompt, messages, config)

    story = StoryboarderV2(FakeEngine(router=LostCheck()))
    sb = await story.build_v3(v3_paper(), make_cast(), "en-US", (16, 32))
    assert sb.panels and any(x["stage"] == "board check" for x in story.skipped)


async def test_pictures_that_cannot_carry_their_point_are_redrawn_on_paper_and_looked_at_once_more(monkeypatch):
    from services import storyboarder_v2   # scenes that keep a picture idea (rounds 3-22)
    monkeypatch.setattr(storyboarder_v2, "SCENE_SHOWS", True)
    # round 15: of the teaching pictures the art check passed, the Claude audits judged 26 of 70 "no" (a still life of
    # symbols, a prop that says the opposite, an earlier panel's picture again): the plan itself was the problem
    class Redraw(PaperModel):
        def __call__(self, prompt, messages, config):
            user = messages[-1].content
            if user.startswith("Check the pictures planned") and self.seen.count("picturecheck") >= 2:
                self.seen.append("picturecheck")   # the look after the quiz finds nothing more
                return {"problems": []}
            if user.startswith("Fix the pictures of these manga panels"):
                self.seen.append("pictures")
                flagged = json.loads(user.split("PANELS TO FIX:\n", 1)[1])
                take = self.seen.count("pictures")   # the first reply redraws panel 8 only: only a rewrite gets the second look
                return {"panels": [{"n": x["n"], "scene": x["scene"], "visual": "Kumo pours a tall jug of beads into a tiny cup, and "
                                    f"most beads spill (take {take})", "shows": "The tiny cup overflows"}
                                   for x in flagged if x["n"] != 2 or take > 1]}
            return super().__call__(prompt, messages, config)

    model = Redraw(picture_problems=[
        {"n": 8, "kind": "still", "problem": "Kumo only holds the cup.", "fix": "Kumo pours beads into a tiny cup that overflows."},
        {"n": 8, "kind": "repeat", "problem": "A second problem on the same panel.", "fix": "..."},
        {"n": 4, "kind": "still", "problem": "A board panel has no picture to check.", "fix": "..."},
        {"n": 2, "kind": "odd", "problem": "Nothing in the picture shows the slow line.", "fix": ""}])
    engine = FakeEngine(router=model)
    story = StoryboarderV2(engine)
    sb = await story.build_v3(v3_paper(), make_cast(), "en-US", (16, 32))
    # one more look and one more fix, no loop; after the quiz every teaching picture gets one more look
    assert model.seen.count("picturecheck") == 3 and model.seen.count("pictures") == 2
    assert model.seen.index("picturecheck") < model.seen.index("clarity")   # before the checks: the fact-check reads the new pictures
    found = [(c["n"], c["kind"]) for c in sb.brief["picture_check"]]
    assert found == [(8, "still"), (2, "still"), (8, "still")]   # one problem per teaching scene, no board panel; then the rewrite
    assert sb.panels[7].visual.endswith("(take 2)") and sb.panels[7].shows == "The tiny cup overflows"
    first, second, after_quiz = [m[-1].content for m, _ in engine.calls if m[-1].content.startswith("Check the pictures planned")]
    assert "just redrawn" not in after_quiz and "new or rewritten" not in after_quiz
    assert [x["n"] for x in json.loads(after_quiz.split("TEACHING PANELS:\n", 1)[1])] == [x["n"] for x in json.loads(first.split("TEACHING PANELS:\n", 1)[1])]
    teaching = json.loads(first.split("TEACHING PANELS:\n", 1)[1])
    assert 4 not in [x["n"] for x in teaching] and {2, 8} <= {x["n"] for x in teaching}
    assert all(set(x) == {"n", "beat", "characters", "words", "visual", "shows"} for x in teaching)
    assert "just redrawn" not in first and "just redrawn" in second
    assert [x["n"] for x in json.loads(second.split("TEACHING PANELS:\n", 1)[1])] == [8]
    fix, again = [m[-1].content for m, _ in engine.calls if m[-1].content.startswith("Fix the pictures")]
    assert "picture check (still): Kumo only holds the cup. Fix: Kumo pours beads into a tiny cup that overflows." in fix
    assert '"dialogue"' not in fix.split("PANELS TO FIX:")[1]   # the words of the panel stay as they are
    # the second look's finding reaches the second fix, for the rewritten panel only
    assert "picture check (still): Kumo only holds the cup." in again
    assert [x["n"] for x in json.loads(again.split("PANELS TO FIX:\n", 1)[1])] == [8]


async def test_the_picture_check_names_a_wrong_contrast_and_a_missing_reference():
    # rounds 20-21: of 61 teaching pictures the audits judged "no", 46 failed in the plan, most of them for a difference
    # other than the one the words compare (23) or a gap with no level drawn to measure it against (7)
    from services.storyboarder_v2 import PICTURE_CHECK_KINDS, picture_check_prompt
    sb = v3_storyboard(clean_panels())
    sb.panels[0].beat, sb.panels[0].shows = "problem", "a tall stack beside a short one"
    sb.panels[1].kind, sb.panels[1].board, sb.panels[1].shows = "scene", None, "two ovens bake the same flour"
    prompt = picture_check_prompt(sb)
    for kind in ('"contrast": the picture shows a difference or a change, but not the one the words are about',
                 '"reference": the point is an error, a gap, too much or too little', '"abstract": the point is a step no object'):
        assert kind in prompt
    assert {"contrast", "reference", "abstract"} <= set(PICTURE_CHECK_KINDS)
    reply = {"problems": [{"n": 1, "kind": "reference", "problem": "Two piles say few and many.", "fix": "A line on each jar."},
                          {"n": 2, "kind": "Contrast", "problem": "The sides differ in the oven, not the flour.", "fix": "..."}]}
    story = StoryboarderV2(FakeEngine([reply, {"problems": [reply["problems"][0]]}, {"problems": []}]))
    found = await story.picture_check(sb)
    assert found == {1: ["picture check (reference): Two piles say few and many. Fix: A line on each jar."],
                     2: ["picture check (contrast): The sides differ in the oven, not the flour. Fix: ..."]}
    await story.picture_check(sb, append=True)   # the look after the quiz adds to the job report's record
    assert [(x["n"], x["kind"]) for x in sb.brief["picture_check"]] == [(1, "reference"), (2, "contrast"), (1, "reference")]
    await story.picture_check(sb)   # a first look starts it afresh
    assert sb.brief["picture_check"] == []


async def test_the_look_at_every_picture_brings_the_lint_of_the_pictures_redrawn_since():
    # a fact-check correction put a clipboard into a picture after the first look; the checker says nothing about props
    # that carry writing, so the look after the quiz must bring the lint's finding for the pictures redrawn since
    from services.storyboarder_v2 import PICTURE_PROBLEM_RE
    sb = v3_storyboard(clean_panels())
    sb.panels[1].kind, sb.panels[1].board = "scene", None
    sb.panels[1].visual = "papi ties blank cards together with red yarn while kumo reads a clipboard"
    sb.panels[1].shows = "the blank cards are tied together"
    fixed = {"panels": [{"n": 2, "scene": sb.panels[1].scene, "visual": "papi ties blank cards together with red yarn",
                         "shows": "the blank cards are tied together"}]}
    for linted, calls in ((None, ["Check the pictures planned"]), ({2}, ["Check the pictures planned", "Fix the pictures", "Check the pictures planned"])):
        sb.panels[1].visual = "papi ties blank cards together with red yarn while kumo reads a clipboard"
        engine = FakeEngine([{"problems": []}, fixed, {"problems": []}, {"panels": []}])
        await StoryboarderV2(engine)._check_pictures(sb, make_cast(), v3_paper(), append=True, linted=linted)
        sent = [m[-1].content for m, _ in engine.calls]
        assert [next(c for c in calls if x.startswith(c)) for x in sent if x.startswith(tuple(calls))] == calls, linted
        if linted:
            flagged = json.loads(sent[1].split("PANELS TO FIX:\n", 1)[1])
            assert any(PICTURE_PROBLEM_RE.match(x) and "clipboard" in x for x in flagged[0]["problems"])
            assert "clipboard" not in sb.panels[1].visual
        else:
            assert "clipboard" in sb.panels[1].visual


async def test_a_rewritten_picture_is_screened_for_people_and_a_possessive_stages_nobody():
    # the picture step runs after the last fit, which is what keeps people out of the art directions ("two customers watch")
    sb = v3_storyboard(clean_panels())
    sb.panels[1].kind, sb.panels[1].board, sb.panels[1].characters = "scene", None, ["papi"]
    reply = {"panels": [{"n": 2, "scene": "A study.", "visual": "Papi pours a jug of beads into a tiny cup while two customers watch, "
                                                                 "and nezu's red cap lies alone on the shelf",
                         "shows": "The tiny cup overflows"}]}
    engine = FakeEngine([reply, {"items": [{"id": "2.v", "text": "Papi pours a jug of beads into a tiny cup while nezu's red cap lies "
                                                                 "alone on the shelf; nobody else is there"}]}])
    story = StoryboarderV2(engine)
    assert await story._fix_pictures(sb, make_cast(), {2: ["picture check (still): ..."]}) == 1
    assert "Rewrite each text so it fits" in engine.prompts[1] and '"id": "2.v"' in engine.prompts[1]
    assert sb.panels[1].visual.startswith("Papi pours a jug of beads into a tiny cup while nezu's red cap")
    assert sb.panels[1].characters == ["papi"]   # "nezu's red cap" puts no nezu in the picture, in the fix and in the fit


async def test_a_symbol_a_rewritten_picture_draws_before_its_key_is_named_in_one_fix_of_the_words():
    # the picture fix cannot change words, and a symbol drawn before any panel says what it stands for explains nothing
    brief = {"picture_symbols": [{"concept": "lost work", "symbol": "a plastic bucket with a small hole leaking water"}]}
    panels = clean_panels()
    for p, beat in zip(panels[1:3], ("problem", "idea")):
        p.kind, p.board, p.beat = "scene", None, beat
    panels[1].visual, panels[1].shows = "papi stacks two piles of cards", "two piles of cards"
    panels[2].visual, panels[2].shows = "papi holds a plastic bucket with a small hole; water leaks out", "water leaks out of the bucket"
    panels[2].narration = "This leaking bucket is the driver's lost work."
    sb = v3_storyboard(panels, CLAIMS_WITH_RESULT, brief=brief)

    class Bucket(PaperModel):
        def __call__(self, prompt, messages, config):
            if messages[-1].content.startswith("Fix the pictures of these manga panels"):
                self.seen.append("pictures")
                return {"panels": [{"n": 2, "scene": "a study", "visual": "papi holds a plastic bucket with a small hole; water leaks onto "
                                    "the cards", "shows": "water leaks out of the bucket onto the cards"}]}
            return super().__call__(prompt, messages, config)

    def name_it(panel: dict, problems: list) -> dict:
        out = fixed_panel(panel, problems)
        if any(x.startswith("symbol:") for x in problems):
            out["narration"] = "This leaking bucket is lost work."
        return out

    model = Bucket(picture_problems=[{"n": 2, "kind": "still", "problem": "Papi only stacks cards.", "fix": "Water leaks from the bucket."}],
                   fix=name_it)
    await StoryboarderV2(FakeEngine(router=model))._check_pictures(sb, make_cast(), v3_paper())
    assert sb.panels[1].visual.startswith("papi holds a plastic bucket")   # the picture that shows the point stays
    assert sb.panels[1].narration == "This leaking bucket is lost work." and model.seen.count("fix") == 1


async def test_a_fix_round_adds_a_panel_only_for_a_chart_its_problems_ask_for():
    # round 23: a fix round no longer splits a panel in two (drafts of 17 panels grew to 24 by seven splits whose second
    # halves restated the first); it adds a board panel only where a result is drawn on no chart, and a words-only round
    # (split=False) adds none
    from services.storyboarder_v2 import CHART_PANEL_ASK
    half = {"after": 2, "kind": "board", "beat": "proof", "title": "Rest", "characters": ["papi"], "scene": "A study.",
            "visual": "papi points at the big whiteboard.", "dialogue": [{"speaker": "papi", "text": "Here is the BLEU table."}],
            "narration": "", "board": TABLE, "claims": [3]}

    class Splitter(PaperModel):
        def __call__(self, prompt, messages, config):
            out = super().__call__(prompt, messages, config)
            if messages[-1].content.startswith("Fix the flagged panels"):
                out["panels"].append(dict(half))
            return out

    chart = f'evidence: the paper\'s result "28.4 BLEU" is never drawn on a chart: {CHART_PANEL_ASK} ({{"after": N}})'
    for problem, split, count in ((chart, True, 5), (chart, False, 4), ("term: name it", True, 4)):
        sb = v3_storyboard(clean_panels(), CLAIMS_WITH_RESULT)
        engine = FakeEngine(router=Splitter())
        await StoryboarderV2(engine).fix_v3(sb, make_cast(), v3_paper(), rounds=1, extra={2: [problem]}, only={2}, split=split)
        assert len(sb.panels) == count, (problem, split)
        fix = next(p for p in engine.prompts if "Fix the flagged panels" in p)
        assert ("ONE new board panel" in fix) == (count == 5) and "New panels: none." in fix or count == 5


async def test_a_picture_fix_that_cannot_run_is_recorded():
    sb = v3_storyboard(clean_panels())

    class Down(FakeEngine):
        async def generate_text(self, messages, config=None):
            raise RuntimeError("writer down")

    story = StoryboarderV2(Down([]))
    assert await story._fix_pictures(sb, make_cast(), {2: ["picture check (still): ..."]}) == 0
    assert [x["stage"] for x in story.skipped] == ["picture fix"]


async def test_a_picture_a_quiz_revision_adds_gets_its_own_look(monkeypatch):
    from services import storyboarder_v2   # scenes that keep a picture idea (rounds 3-22)
    monkeypatch.setattr(storyboarder_v2, "SCENE_SHOWS", True)
    # the main picture check runs before the quiz; a scene the revision adds would ship unchecked, so every teaching
    # picture, the added one among them, gets one more look after the quiz
    class HowRevision(PaperModel):
        def __call__(self, prompt, messages, config):
            out = super().__call__(prompt, messages, config)
            if messages[-1].content.startswith("A reader who never saw the paper read only the visible text"):
                for x in out["panels"]:
                    x["beat"], x["shows"] = "how", "papi holds up one finger"
            return out

    model = HowRevision(quiz_misses=({2}, set()))
    engine = FakeEngine(router=model)
    sb = await StoryboarderV2(engine).build_v3(v3_paper(), make_cast(), "en-US", (16, 32))
    checks = [m[-1].content for m, _ in engine.calls if m[-1].content.startswith("Check the pictures planned")]
    added = [p.n for p in sb.panels if p.title == "Claim 2"]
    assert len(checks) == 2 and len(added) == 1 and "new or rewritten" not in checks[1]
    second = [x["n"] for x in json.loads(checks[1].split("TEACHING PANELS:\n", 1)[1])]
    assert set(added) < set(second) and added[0] not in [x["n"] for x in json.loads(checks[0].split("TEACHING PANELS:\n", 1)[1])]


async def test_a_picture_check_that_cannot_run_is_recorded(monkeypatch):
    from services import storyboarder_v2   # scenes that keep a picture idea (rounds 3-22)
    monkeypatch.setattr(storyboarder_v2, "SCENE_SHOWS", True)
    class LostCheck(PaperModel):
        def __call__(self, prompt, messages, config):
            if messages[-1].content.startswith("Check the pictures planned"):
                raise RuntimeError("checker down")
            return super().__call__(prompt, messages, config)

    story = StoryboarderV2(FakeEngine(router=LostCheck()))
    sb = await story.build_v3(v3_paper(), make_cast(), "en-US", (16, 32))
    assert sb.panels and any(x["stage"] == "picture check" for x in story.skipped)


async def test_a_second_round_whose_quiz_cannot_run_is_undone():
    class LostQuiz(PaperModel):
        def __call__(self, prompt, messages, config):
            if messages[-1].content.startswith("You have never read the research paper") and self.reader_rounds >= 2:
                raise RuntimeError("reader down")   # the quiz after the second round cannot score
            return super().__call__(prompt, messages, config)

    model = LostQuiz(quiz_misses=({2}, {2}, set()))
    story = StoryboarderV2(FakeEngine(router=model))
    sb = await story.build_v3(v3_paper(), make_cast(), "en-US", (16, 32))
    assert model.seen.count("revise") == 2 and sb.quiz["undone"] == 2
    assert (sb.quiz["score"], sb.quiz["failed_claim_ids"]) == (0.75, [2])   # the score of the script that ships
    assert [p.title for p in sb.panels].count("Claim 2") == 1
    assert any(x["stage"] == "abstract quiz after the revision" for x in story.skipped)


def test_the_reader_check_sees_the_gap_badge_the_chart_draws():
    from services.storyboarder_v2 import clarity_prompt
    board = {"kind": "bar", "title": "MultiNLI accuracy", "subtitle": "higher is better",
             "bars": [{"label": "OpenAI GPT", "value": "82.1", "baseline": True}, {"label": "BERT large", "value": "86.7", "highlight": True}]}
    sb = v3_storyboard([v3_panel(1, kind="board", beat="proof", board=board, dialogue=[("nezu", "4.6 points over OpenAI GPT?")])])
    prompt = clarity_prompt(sb)
    assert "[gap to OpenAI GPT: +4.6]" in prompt and "so a gap the words state is on the board" in prompt
    assert "never supply a fact, number or baseline of your own" in prompt


def fact_check_ids(prompt: str) -> set[str]:
    return {item["id"] for item in json.loads(prompt.split("LINES:\n", 1)[1].split("\n\nPAPER TEXT:", 1)[0])}


async def test_a_quiz_revision_re_checks_only_the_panels_it_changed():
    model = PaperModel(quiz_misses=({2},))
    engine = FakeEngine(router=model)
    sb = await StoryboarderV2(engine).build_v3(v3_paper(), make_cast(), "en-US", (16, 32))
    first, second = [fact_check_ids(p) for p in engine.prompts if "You are a meticulous fact-checker" in p]
    new = next(p for p in sb.panels if p.title == "Claim 2")
    assert any(i.startswith("card.") for i in first)
    assert {i.split(".")[0] for i in first if not i.startswith("card.")} == {str(n) for n in range(1, 10)}   # the whole script, before the insert
    assert f"{new.n}.d0" in second and not any(i.startswith("card.") for i in second)
    assert len({i.split(".")[0] for i in second}) < len({i.split(".")[0] for i in first})   # the unchanged panels are not read again
    clarity = [p for p in engine.prompts if "You are a curious reader" in p]
    assert "are new or rewritten" not in clarity[0] and f"Panels {new.n}" in clarity[1]


async def test_a_partial_fact_check_adds_its_fixes_and_remembers_what_it_read():
    panels = [v3_panel(1, beat="hook", narration="Translation turns one sentence into another."),
              v3_panel(2, beat="idea", dialogue=[("papi", "Attention costs nothing.")])]
    sb = v3_storyboard(panels)
    sb.brief["factcheck"] = [{"id": "1.n", "before": "x", "after": "y", "issue": "earlier"}]
    problems = [{"id": "2.d0", "issue": "the paper does not say so", "corrected": "Attention links any two words in one step."}]
    story = StoryboarderV2(FakeEngine(router=PaperModel(factcheck_problems=problems)))
    assert await story.factcheck(sb, v3_paper(), panels={2}) == 1
    assert [f["id"] for f in sb.brief["factcheck"]] == ["1.n", "2.d0"]   # added to the earlier pass, not replacing it
    assert story._unchecked(sb) == {1} and await story.factcheck(sb, v3_paper(), panels=set()) == 0


async def test_lines_changed_after_their_fact_check_get_one_last_check():
    # the fact-check corrects a bubble into a long line, fit shortens it afterwards: the shortened line is read once more
    long_line = " ".join(["Attention lets each word look at every other word in the sentence at the same time"] * 2)
    model = PaperModel(factcheck_problems=[{"id": "2.d0", "issue": "too strong", "corrected": long_line}])
    engine = FakeEngine(router=model)
    story = StoryboarderV2(engine)
    sb = await story.build_v3(v3_paper(), make_cast(), "en-US", (16, 32))
    checks = [fact_check_ids(p) for p in engine.prompts if "You are a meticulous fact-checker" in p]
    assert len(checks) == 2 and {i.split(".")[0] for i in checks[1]} == {"2"}   # the whole script, then only the panel fit changed
    assert model.seen.index("quiz") < len(model.seen) - 1 - model.seen[::-1].index("factcheck")   # after the quiz gate


async def test_a_reader_check_that_cannot_run_is_recorded_and_changes_nothing():
    class Broken(PaperModel):
        def __call__(self, prompt, messages, config):
            if messages[-1].content.startswith("You are a curious reader"):
                raise RuntimeError("checker down")
            return super().__call__(prompt, messages, config)

    story = StoryboarderV2(FakeEngine(router=Broken()))
    sb = await story.build_v3(v3_paper(), make_cast(), "en-US", (16, 32))
    assert any(x["stage"] == "reader check" and "checker down" in x["error"] for x in story.skipped)
    assert "clarity" not in sb.brief and len(sb.panels) == 9


async def test_build_v3_never_fails_on_the_quiz(monkeypatch):
    import services.abstract_quiz as aq

    async def broken(*_args, **_kwargs):
        raise RuntimeError("quiz exploded")

    monkeypatch.setattr(aq, "run_quiz", broken)
    sb = await StoryboarderV2(FakeEngine(router=PaperModel())).build_v3(v3_paper(), make_cast(), "en-US", (16, 32))
    assert sb.quiz["score"] is None and "quiz exploded" in sb.quiz["error"] and len(sb.panels) == 9

    async def no_score(*_args, **_kwargs):   # run_quiz's own failure: no exception, an error in the result
        return {"score": None, "error": "ValueError: no JSON object in model reply", "usage": {}}

    monkeypatch.setattr(aq, "run_quiz", no_score)
    story = StoryboarderV2(FakeEngine(router=PaperModel()))
    sb = await story.build_v3(v3_paper(), make_cast(), "en-US", (16, 32))
    assert sb.quiz["score"] is None and any(x["stage"] == "abstract quiz" and "no JSON" in x["error"] for x in story.skipped)

    calls = []

    async def once_broken(*args, **kwargs):   # an intermittent failure: the second run scores
        calls.append(1)
        return await no_score() if len(calls) == 1 else {"score": 1.0, "failed_claim_ids": [], "questions": [], "usage": {}}

    monkeypatch.setattr(aq, "run_quiz", once_broken)
    story = StoryboarderV2(FakeEngine(router=PaperModel()))
    sb = await story.build_v3(v3_paper(), make_cast(), "en-US", (16, 32))
    assert len(calls) == 2 and sb.quiz["score"] == 1.0 and not story.skipped

    monkeypatch.setitem(sys.modules, "services.abstract_quiz", None)   # the module cannot be imported at all
    sb = await StoryboarderV2(FakeEngine(router=PaperModel())).build_v3(v3_paper(), make_cast(), "en-US", (16, 32))
    assert sb.quiz == {"score": None, "failed_claim_ids": [], "questions": [], "error": "abstract quiz unavailable"}


async def test_build_v3_factcheck_corrects_board_strings_and_sends_whole_board_findings_to_a_fix_round():
    model = PaperModel(factcheck_problems=[
        {"id": "6.b.title", "issue": "the table shows test sets", "corrected": "WMT 2014 test sets (BLEU)"},
        {"id": "6.b.rows.0.cells.1", "issue": "tries to change a value", "corrected": "28.5"},
        {"id": "6.b", "issue": "mixes English-German and English-French in one table", "corrected": "keep only English-German"},
    ])
    engine = FakeEngine(router=model)
    sb = await StoryboarderV2(engine).build_v3(v3_paper(), make_cast(), "en-US", (16, 32))

    assert sb.panels[5].board["title"] == "WMT 2014 test sets (BLEU)"
    assert sb.panels[5].board["rows"][0]["cells"][1] == "28.4"   # a string fix never changes a number
    fixes = {f["id"]: f for f in sb.brief["factcheck"]}
    assert fixes["6.b.title"]["before"] == "WMT 2014 translation (BLEU)" and "6.b.rows.0.cells.1" not in fixes
    check = prompt_of(engine, "You are a meticulous fact-checker")
    items = json.loads(check.split("LINES:\n", 1)[1].split("\n\nPAPER TEXT:", 1)[0])
    ids = {i["id"] for i in items}
    assert {"6.b", "6.b.title", "6.b.rows.0.cells.0", "9.b.items.3", "4.b.steps.0"} <= ids
    assert "mixes different metrics, units, datasets, splits or training setups" in check and '"prior best"' in check
    fix = prompt_of(engine, "Fix the flagged panels")
    assert "fact-check: mixes English-German and English-French in one table Fix: keep only English-German" in fix


async def test_fit_v3_uses_kind_budgets_and_refuses_rewrites_that_change_numbers():
    sb = v3_storyboard([
        v3_panel(1, dialogue=[("papi", " ".join(["word"] * 16) + " 28.4")]),                     # 17 words on a scene panel
        v3_panel(2, kind="board", board=RECAP, dialogue=[("nezu", " ".join(["word"] * 13))], characters=("nezu",)),
        v3_panel(3, dialogue=[("kumo", " ".join(["word"] * 16))]),                                # exactly at the limit
    ])
    fast = FakeEngine([{"items": [{"id": "1.d0", "text": "Short line without the number."}, {"id": "2.d0", "text": "Short board line."}]},
                       {"items": [{"id": "1.d0", "text": "Short line, still 28.4."}]}])
    assert await StoryboarderV2(FakeEngine(), fast_engine=fast).fit(sb, "en-US") == 2
    assert [b.text for p in sb.panels for b in p.dialogue][:2] == ["Short line, still 28.4.", "Short board line."]
    first = json.loads(fast.prompts[0].split("ITEMS:\n", 1)[1])
    assert [(i["id"], i["max"]) for i in first] == [("1.d0", 16), ("2.d0", 12)]


async def test_check_brief_corrects_claim_texts_and_never_drops_a_claim():
    brief = {"key_idea": "k", "abstract_claims": [{"id": 0, "text": "Wrong paraphrase.", "quote": "the abstract's words"},
                                                  {"id": 1, "text": "Unsupported.", "quote": "other abstract words"}]}
    engine = FakeEngine([{"problems": [{"id": "abstract_claims.0", "issue": "overstated", "corrected": "Right paraphrase."},
                                       {"id": "abstract_claims.1", "issue": "unsupported", "corrected": "DELETE"}]}])
    assert await StoryboarderV2(engine).check_brief(brief, PaperText("T", "body", [])) == 2
    assert [c["text"] for c in brief["abstract_claims"]] == ["Right paraphrase.", "other abstract words"]
    assert '"id": "abstract_claims.1"' in engine.prompts[0]


# ----------------------------------------------------------------------------- v2 untouched

async def test_v2_build_is_unaffected_by_the_v3_layer():
    model = PaperModel(num_panels=3)
    engine = FakeEngine(router=model)
    sb = await StoryboarderV2(engine).build(PaperText("T", "body", []), make_cast(), "zh-CN", (3, 6))
    assert model.seen == ["brief", "briefcheck", "storyboard", "factcheck", "translate", "factcheck"]
    assert sb.version == 2 and "version" not in sb.to_dict() and "abstract_claims" not in sb.to_dict()
    assert all(p.kind == "scene" and p.board is None and len(p.dialogue) <= 2 for p in sb.panels)
    assert set(coverage(sb)) == {"core", "important"}
    translate = prompt_of(engine, "Translate this manga script")
    assert "each bubble ≤ 22 characters" in translate and '"board"' not in translate
    check = prompt_of(engine, "You are a meticulous fact-checker")
    assert '"N.b"' not in check and "bubbles ≤ 14 words" in check
    sb2 = StoryboardV2(title="T", summary="S", theme="kumomo", language="en-US", chapters=[], panels=[v3_panel(1)])
    assert '"board"' not in translate_prompt(sb2, "zh-CN")


async def test_build_v3_without_an_abstract_heading_checks_claims_against_the_paper():
    text = "\n".join(line for line in LONG_PAPER_TEXT.splitlines() if line != "Abstract")
    engine = FakeEngine(router=PaperModel())
    sb = await StoryboarderV2(engine).build_v3(v3_paper(text), make_cast(), "en-US", (16, 32))
    assert "no abstract heading was found" in prompt_of(engine, "Read the paper and its ABSTRACT")
    assert [c["quote"] for c in sb.abstract_claims] == [c["quote"] for c in V3_CLAIMS] and sb.brief["abstract"] == ""
    assert sb.brief["examples"] == []   # LONG_PAPER_TEXT has no figure example
    assert sb.brief["running_example"]["from_paper"] is False   # the story claimed Figure 3, which this paper lacks
    assert all(p.claims == [c for c in p.claims if c in (0, 1, 2)] for p in sb.panels)


def test_running_example_from_paper_only_when_a_verified_example_shows_up():
    from services.storyboarder_v2 import _running_example
    brief = {"examples": [{"text": V3_EXAMPLE, "source": "Figure 3"}]}
    shown = [v3_panel(1, dialogue=[("kumo", f"Look: {V3_EXAMPLE}")])]
    assert _running_example({"text": "Kumo reads a sentence.", "from_paper": True}, brief, shown, V3_PAPER_TEXT) == \
        {"text": "Kumo reads a sentence.", "from_paper": True, "source": "Figure 3"}
    other = [v3_panel(1, dialogue=[("kumo", "I bake a cake in twelve steps.")])]
    assert _running_example({"text": "Kumo bakes a cake.", "from_paper": "true"}, brief, other, V3_PAPER_TEXT)["from_paper"] is False
    assert _running_example({"text": V3_EXAMPLE, "from_paper": "false"}, brief, shown, V3_PAPER_TEXT)["from_paper"] is False


# ----------------------------------------------------------------------------- review fixes

COST_PAPER = V3_PAPER_TEXT + "\nThe big model costs 2.3 · 10^19 FLOPs to train on 8 GPUs and won the 1st place."
COST_CLAIM = {"id": 0, "text": "Training costs 2.3 · 10^19 FLOPs on 8 GPUs and won 1st place.",
              "quote": "The big model costs 2.3 · 10^19 FLOPs to train on 8 GPUs and won the 1st place."}


@pytest.mark.parametrize("line, missing", [
    ("Training took 2.3 × 10¹⁹ FLOPs on eight GPUs and won first place.", []),   # superscripts, number words
    ("Training took 2.3e19 FLOPs on 8 GPUs, 1st place.", []),
    ("Training took 2.3 · 10^18 FLOPs on 8 GPUs, 1st place.", ["2.3 · 10^19"]),   # a different figure is not the claim's
    ("Training was cheap on eight GPUs and won first place.", ["2.3 · 10^19"]),
])
def test_lint_reads_a_claims_numbers_the_way_the_boards_do(line, missing):
    panels = [v3_panel(1, dialogue=[("papi", line)], claims=[0]),
              v3_panel(2, kind="board", beat="recap", board={"kind": "list", "title": "Recap", "items": ["Attention only", "Cheap training"]},
                       dialogue=[("nezu", "Fine.")], characters=("nezu",))]
    problems = lint_v3(v3_storyboard(panels, [COST_CLAIM]), COST_PAPER)
    needs = [x for found in problems.values() for x in found if x.startswith("claim 0")]
    assert [x.split("needs its number ")[1].split(" visible")[0] for x in needs] == ([", ".join(missing)] if missing else [])


async def test_build_v3_keeps_a_recap_board_whose_only_problems_the_renderer_copes_with():
    long_item = "28.4 BLEU on WMT 2014 English-German, which is more than 2.0 BLEU above the best earlier models"

    def story(payload):
        payload["panels"][8]["board"]["items"].append(long_item)        # 18 words: over the 14-word limit
        return payload

    model = PaperModel(story=story, fix=lambda panel, problems: panel)   # a writer that never fixes anything
    sb = await StoryboarderV2(FakeEngine(router=model)).build_v3(v3_paper(), make_cast(), "en-US", (16, 32))

    assert model.seen.count("fix") == 2
    recap = sb.panels[8]
    assert recap.kind == "board" and recap.board["items"][-1] == long_item and len(recap.board["items"]) == 5
    assert any("too long" in x for x in sb.brief["lint"]["9"])          # still reported, but the evidence stays drawn


async def test_build_v3_drops_only_the_unsupported_bar_of_a_chart():
    chart = {"kind": "bar", "title": "English-German BLEU", "subtitle": "BLEU: overlap with human translations", "unit": "BLEU", "bars": [
        {"label": "Transformer (big)", "value": "28.4", "quote": "establishing a new state-of-the-art BLEU score of 28.4", "highlight": True},
        {"label": "ConvS2S Ensemble", "value": "26.36", "quote": "ConvS2S Ensemble 26.36"},
        {"label": "Prior best", "value": "27.1"},                        # nothing in the paper supports it
        {"label": "ByteNet", "value": "23.75", "quote": "ByteNet 23.75"}]}

    def story(payload):
        payload["panels"][5]["board"] = chart
        return payload

    model = PaperModel(story=story, fix=lambda panel, problems: panel)
    sb = await StoryboarderV2(FakeEngine(router=model)).build_v3(v3_paper(), make_cast(), "en-US", (16, 32))

    board = sb.panels[5].board
    assert sb.panels[5].kind == "board" and [b["label"] for b in board["bars"]] == ["Transformer (big)", "ConvS2S Ensemble", "ByteNet"]
    assert "27.1" not in visible_text(sb.panels[5]) and board["bars"][0]["highlight"] is True
    assert sb.brief["lint"] == {}                                        # nothing unsupported is left


def test_a_demoted_board_never_moves_an_unsupported_number_into_the_narration():
    bad = {"kind": "bar", "title": "Transformer is 99.9 times faster",
           "bars": [{"label": "Old", "value": "77.7"}, {"label": "New", "value": "66.6", "highlight": True}]}
    sb = v3_storyboard([v3_panel(1, kind="board", beat="proof", board=bad, dialogue=[("nezu", "Show me.")], narration="Speed matters.")])
    StoryboarderV2(FakeEngine())._demote_boards(sb, V3_PAPER_TEXT)
    assert (sb.panels[0].kind, sb.panels[0].board, sb.panels[0].narration) == ("scene", None, "Speed matters.")


def test_salvage_shows_bars_of_different_units_as_a_table_and_follows_lower_is_better():
    mixed = {"kind": "bar", "title": "Quality and cost", "bars": [
        {"label": "BLEU", "value": "41.0", "quote": "our big model achieves a BLEU score of 41.0", "highlight": True},
        {"label": "Training time", "value": "3.5 days", "quote": "after training for 3.5 days on eight GPUs"}]}
    paper = "Top-5 error of the ensemble is 4.9% on the validation set.\nThe previous best top-5 error was 5.1% on it."
    lower = {"kind": "bar", "title": "Top-5 error (lower is better)", "higher_is_better": True, "bars": [
        {"label": "Ensemble", "value": "4.9%", "quote": "Top-5 error of the ensemble is 4.9%", "highlight": True},
        {"label": "Previous best", "value": "5.1%", "quote": "The previous best top-5 error was 5.1%"}]}
    sb = v3_storyboard([v3_panel(1, kind="board", beat="proof", board=mixed, dialogue=[("nezu", "Hm.")]),
                        v3_panel(2, kind="board", beat="proof", board=lower, dialogue=[("nezu", "Hm.")])])
    writer = StoryboarderV2(FakeEngine())
    writer._demote_boards(v3_storyboard(sb.panels[:1]), V3_PAPER_TEXT)
    writer._demote_boards(v3_storyboard(sb.panels[1:]), paper)
    table = sb.panels[0].board
    assert table["kind"] == "table" and [r["cells"] for r in table["rows"]] == [["BLEU", "41.0"], ["Training time", "3.5 days"]]
    assert table["rows"][1]["quote"] == "after training for 3.5 days on eight GPUs" and table["highlight_rows"] == [0]
    assert sb.panels[1].board["kind"] == "bar" and sb.panels[1].board["higher_is_better"] is False


async def test_build_v3_fix_round_keeps_a_valid_board_that_the_reply_left_out():
    def story(payload):
        for p in payload["panels"]:
            p["claims"] = [c for c in p["claims"] if c != 2]
        return payload

    def lazy_fix(panel, problems):   # fixes the claim but leaves out the board it did not need to touch
        out = fixed_panel(panel, problems)
        out.pop("board", None)
        return out

    model = PaperModel(story=story, fix=lazy_fix)
    sb = await StoryboarderV2(FakeEngine(router=model)).build_v3(v3_paper(), make_cast(), "en-US", (16, 32))

    recap = sb.panels[8]
    assert (recap.kind, recap.board["kind"]) == ("board", "list")   # the board the reply left out is kept
    assert any(2 in p.claims for p in sb.panels if p.beat not in ("claims", "recap"))
    assert model.seen.count("fix") == 1 and sb.brief["lint"] == {}


async def test_build_v3_reads_odd_reply_shapes():
    def story(payload):
        panels = payload["panels"]
        panels[0]["dialogue"] = [f"kumo: {panels[0]['dialogue'][0]['text']}"]          # "speaker: text" strings
        panels[1]["dialogue"] = [[b["speaker"], b["text"]] for b in panels[1]["dialogue"]]   # [speaker, text] pairs
        panels[3]["board"] = json.dumps(panels[3]["board"])                             # the board as a JSON string
        payload["panels"] = {str(p["n"]): {k: v for k, v in p.items() if k != "n"} for p in panels}   # keyed by number
        return payload

    sb = await StoryboarderV2(FakeEngine(router=PaperModel(story=story))).build_v3(v3_paper(), make_cast(), "en-US", (16, 32))
    assert [p.n for p in sb.panels] == list(range(1, 10)) and sb.brief["lint"] == {}
    assert [(b.speaker, b.text[:20]) for b in sb.panels[0].dialogue] == [("kumo", "'The animal didn't c")]
    assert [b.speaker for b in sb.panels[1].dialogue] == ["papi", "kumo"]
    assert (sb.panels[3].kind, sb.panels[3].board["kind"]) == ("board", "flow")


async def test_v3_translation_must_keep_the_numbers_of_each_line():
    def storyboard():
        return v3_storyboard([v3_panel(1, dialogue=[("nezu", "28.4 on English-German. Fine.")], claims=[0]),
                              v3_panel(2, dialogue=[("papi", "It trains on 8 GPUs.")], claims=[1])])

    def reply(first: str, second: str = "它在八块 GPU 上训练。") -> dict:
        return {"title": "注意力", "panels": [{"n": 1, "title": "第1格", "dialogue": [{"speaker": "nezu", "text": first}]},
                                              {"n": 2, "title": "第2格", "dialogue": [{"speaker": "papi", "text": second}]}]}

    retry_ok = {"panels": [{"n": 1, "dialogue": [{"speaker": "nezu", "text": "英德翻译拿到 28.4。好吧。"}]}]}
    engine = FakeEngine([reply("英德翻译。好吧。"), retry_ok])          # the first reply drops 28.4
    sb = await StoryboarderV2(engine).translate(storyboard(), "zh-CN")
    assert [b.text for p in sb.panels for b in p.dialogue] == ["英德翻译拿到 28.4。好吧。", "它在八块 GPU 上训练。"]   # 8 → 八 is fine
    assert len(engine.calls) == 2 and '"28.4 on English-German. Fine."' in engine.prompts[1]

    engine = FakeEngine([reply("英德翻译。好吧。"), {"panels": [{"n": 1, "dialogue": [{"speaker": "nezu", "text": "还是没有数字。"}]}]}])
    sb = await StoryboarderV2(engine).translate(storyboard(), "zh-CN")
    assert sb.panels[0].dialogue[0].text == "还是没有数字。"              # twice: Chinese beats an English panel (logged)


def test_a_stray_section_number_is_neither_abstract_text_nor_a_claim():
    body = "\n".join(["Title", "Abstract", "We propose a model. It reaches 28.4 BLEU on the test set.", "1", "Introduction", "Text."])
    assert find_abstract(body) == "We propose a model. It reaches 28.4 BLEU on the test set."
    claims = verify_claims([], "We propose a model that works well. It reaches 28.4 BLEU. 1", body="")
    assert [c["quote"] for c in claims] == ["We propose a model that works well.", "It reaches 28.4 BLEU."]


def test_panel_replies_with_lists_dicts_and_loose_claim_ids_are_read():
    from services.storyboarder_v2 import _claim_ids, _panel_from_reply
    p = _panel_from_reply({"kind": "scene", "narration": ["Fact one.", "Fact two."], "dialogue": {"kumo": "Hi!", "papi": "Hello."},
                           "claims": ["Claim 3", {"id": 4}, 2.0, "C1", True, "Q5"]}, 1)
    assert p.narration == "Fact one. Fact two." and [(b.speaker, b.text) for b in p.dialogue] == [("kumo", "Hi!"), ("papi", "Hello.")]
    assert p.claims == [3, 4, 2, 1] and _claim_ids("-1") == []
    q = _panel_from_reply({"dialogue": ["papi: Note: it works", "Just a line.", "A speaker name that is far too long: text"]}, 2)
    assert [(b.speaker, b.text) for b in q.dialogue] == [("papi", "Note: it works"), ("", "Just a line."),
                                                         ("", "A speaker name that is far too long: text")]


async def test_build_v3_asks_once_more_when_the_story_reply_has_no_panels():
    calls = []

    def story(payload):
        calls.append(1)
        return {"title": "Attention, Please!", "panels": []} if len(calls) == 1 else payload

    model = PaperModel(story=story)
    sb = await StoryboarderV2(FakeEngine(router=model)).build_v3(v3_paper(), make_cast(), "en-US", (16, 32))
    assert model.seen.count("story") == 2 and len(sb.panels) == 9

    model = PaperModel(story=lambda payload: {"panels": {}})
    with pytest.raises(ValueError, match="no panels"):
        await StoryboarderV2(FakeEngine(router=model)).build_v3(v3_paper(), make_cast(), "en-US", (16, 32))
    assert model.seen.count("story") == 2


async def test_stray_reply_shapes_never_fail_translation_fact_check_or_fit():
    from services.storyboarder_v2 import _normalize_panels, extract_json
    sb = v3_storyboard([v3_panel(1, dialogue=[("nezu", "Attention looks at every word.")], narration="Fact one.")])
    engine = FakeEngine([{"title": ["not", "text"], "chapters": 1.5,
                          "panels": {"1": {"dialogue": [{"speaker": "nezu", "text": ["列表"]}], "narration": "事实一。"}}},
                         {"panels": {"1": {"dialogue": [{"speaker": "nezu", "text": "注意力会看每个词。"}], "narration": "事实一。"}}}])
    sb = await StoryboarderV2(engine).translate(sb, "zh-CN")
    assert sb.title == "T" and sb.panels[0].dialogue[0].text == "注意力会看每个词。"   # a list is no line: re-requested
    checker = FakeEngine([{"problems": {"id": "1.n", "issue": "x", "corrected": "事实二。"}}, {"problems": 0},
                          {"problems": [{"id": "1.d0", "issue": "x", "corrected": ["a", "b"]}]}])
    writer = StoryboarderV2(checker)
    assert await writer.factcheck(sb, v3_paper(), "zh-CN") == 1 and sb.panels[0].narration == "事实二。"
    assert await writer.factcheck(sb, v3_paper(), "zh-CN") == 0
    assert await writer.factcheck(sb, v3_paper(), "zh-CN") == 0 and sb.panels[0].dialogue[0].text == "注意力会看每个词。"
    long = v3_storyboard([v3_panel(1, dialogue=[("papi", " ".join(["word"] * 30))])])
    assert await StoryboarderV2(FakeEngine(), fast_engine=FakeEngine([{"items": 7}, {"items": "x"}])).fit(long, "en-US") == 0
    assert extract_json('{"a": NaN, "b": [Infinity, 1.5]}') == {"a": None, "b": [None, 1.5]}
    panels = [PanelV2.from_dict({"characters": ["kumo"], "dialogue": [{"speaker": "kumo", "text": "  "}, {"speaker": "papi", "text": "**"},
                                                                      {"speaker": "kumo", "text": "ok"}]}, 1)]
    _normalize_panels(panels, make_cast(), max_bubbles=None)
    assert [(b.speaker, b.text) for b in panels[0].dialogue] == [("kumo", "ok")]


def test_salvage_removes_tex_marks_but_keeps_quotes_verbatim():
    board = {"kind": "bar", "title": "**WMT 2014** English-German ($BLEU$)", "bars": [
        {"label": "$Transformer$ (big)", "value": "$28.4$", "quote": "establishing a new state-of-the-art BLEU score of 28.4", "highlight": True},
        {"label": "ByteNet", "value": "23.75", "quote": "ByteNet 23.75"}]}
    sb = v3_storyboard([v3_panel(1, kind="board", beat="proof", board=board, dialogue=[("nezu", "Hm.")])])
    StoryboarderV2(FakeEngine())._demote_boards(sb, V3_PAPER_TEXT)
    kept = sb.panels[0].board
    assert kept["title"] == "WMT 2014 English-German (BLEU)" and [(b["label"], b["value"]) for b in kept["bars"]] == \
        [("Transformer (big)", "28.4"), ("ByteNet", "23.75")]
    assert kept["bars"][0]["quote"] == "establishing a new state-of-the-art BLEU score of 28.4"


def test_lint_flags_props_that_carry_writing_and_verdict_words():
    panels = clean_panels()
    panels[0].visual = "nezu leans forward, pen hovering over her clipboard, a wall clock behind her."
    panels[1].dialogue[0].text = "So the paper proved it works perfectly."
    problems = lint_v3(v3_storyboard(panels, CLAIMS_WITH_RESULT), V3_PAPER_TEXT)
    assert any("props that carry writing (clipboard, pen hovering, wall clock)" in x for x in problems[1])
    assert any(x.startswith("verdict words perfectly, proved") for x in problems[2])
    panels[1].dialogue[0].text = "Its example reads 'The Law will never be perfect'."   # the paper's own words are no verdict
    assert not any(x.startswith("verdict words") for x in lint_v3(v3_storyboard(panels, CLAIMS_WITH_RESULT), V3_PAPER_TEXT).get(2, []))


def test_lint_wants_the_running_example_in_the_hook_and_the_payoff():
    panels = clean_panels() + [v3_panel(5, beat="payoff", dialogue=[("kumo", "Now I get it!")])]
    sb = v3_storyboard(panels, CLAIMS_WITH_RESULT)
    sb.running_example = f'Kumo reads one sentence: "{V3_EXAMPLE}"'
    problems = lint_v3(sb, V3_PAPER_TEXT)
    assert any(x.startswith("the hook does not show the running example") for x in problems[1])
    assert any(x.startswith("the payoff does not show the running example") and "solved" in x for x in problems[5])
    panels[0].narration = "The animal didn't cross the street: who was too tired?"
    panels[4].dialogue[0].text = "It looks at the animal: the animal was tired, not the street!"
    assert not any("running example" in x for found in lint_v3(sb, V3_PAPER_TEXT).values() for x in found)


def test_lint_wants_named_bars_one_highlight_and_honest_links():
    chart = {"kind": "bar", "title": "GLUE", "source": "Table 1", "bars": [
        {"label": "Prior SOTA", "paper_label": "OpenAI GPT", "value": "72.8", "baseline": True},
        {"label": "BERT base", "value": "79.6", "highlight": True}, {"label": "BERT large", "value": "80.5", "highlight": True}]}
    found = _story_board_issues(chart)
    assert any(x.startswith('bar "Prior SOTA" names no system') and '"OpenAI GPT"' in x for x in found)
    assert any(x.startswith("2 highlighted bars") for x in found)
    named = {**chart, "bars": [{**chart["bars"][0], "label": "OpenAI GPT"}, chart["bars"][2]]}
    assert _story_board_issues(named) == ["only 2 bars: show the field too: add 1-3 other systems from the same table column "
                                          "(same metric, dataset and setting), each with its quote, when the paper lists them"]
    assert _story_board_issues({**named, "bars": named["bars"] + [{"label": "ELMo", "value": "70.0"}]}) == []
    assert not any("names no system" in x for x in
                   _story_board_issues({"kind": "bar", "bars": [{"label": "MoE prior best", "paper_label": "MoE", "value": "40.56"}]}))
    links = {"kind": "links", "title": "Actual attention", "note": "Exact weights reproduced from Figure 3", "tokens": ["a", "b"], "from": 0, "weights": [0, 1]}
    assert any(x.startswith("links board claims exact or measured weights") for x in _story_board_issues(links))


def test_number_lines_quote_the_paper_lines_that_hold_the_abstracts_numbers():
    text = number_lines(CLAIMS_WITH_RESULT, V3_PAPER_TEXT)
    assert text.startswith("28.4: ") and '"Transformer (big) 28.4"' in text
    assert "2014" not in text.split(":", 1)[0]                       # years are no key numbers
    assert number_lines([], V3_PAPER_TEXT) == ""


def test_words_and_chart_must_show_the_same_comparison():
    chart = {"kind": "bar", "title": "GLUE", "source": "Table 1", "bars": [
        {"label": "OpenAI GPT", "value": "75.1", "baseline": True}, {"label": "BERT LARGE", "value": "82.1", "highlight": True}]}
    panel = v3_panel(3, kind="board", beat="proof", board=chart, dialogue=[("nezu", "A huge 7.7 point gap.")], characters=("nezu",))
    assert _words_vs_chart(panel) and "7.7" in _words_vs_chart(panel)[0]
    panel.dialogue[0].text = "A 7.0 point gap over OpenAI GPT."              # the drawn gap
    assert _words_vs_chart(panel) == []
    panel.dialogue[0].text = "82.1 points on average."                     # a drawn value
    assert _words_vs_chart(panel) == []


def test_scrub_removes_only_the_sentence_with_a_number_the_paper_lacks():
    panels = clean_panels()
    panels[1].dialogue[0].text = "Only attention. Performance drops 10.7 F1 without it."
    panels[1].narration = "It dropped 99.9 points."
    sb = v3_storyboard(panels, CLAIMS_WITH_RESULT)
    removed = scrub_unsupported(sb, V3_PAPER_TEXT)
    assert [b.text for b in sb.panels[1].dialogue] == ["Only attention."] and sb.panels[1].narration == ""
    assert len(removed) == 2 and all(x.startswith("panel 2:") for x in removed)


def test_words_may_say_the_gap_their_own_chart_draws():
    # LongZero: the board check advised "beats the stronger baseline by 0.85", and the number check then removed the
    # sentence because 0.85 (57.42 - 56.57, computed by the program) is nowhere in the paper
    bleu = {"kind": "bar", "title": "EN-DE BLEU", "bars": [
        {"label": "ConvS2S Ensemble", "value": "26.36", "baseline": True}, {"label": "Transformer (big)", "value": "28.4", "highlight": True}]}
    board = v3_panel(3, kind="board", beat="proof", board=bleu, dialogue=[("nezu", "It beats ConvS2S by 2.04 BLEU.")], claims=[3])
    scene = v3_panel(4, beat="payoff", dialogue=[("nezu", "It beats ConvS2S by 2.04 BLEU.")], claims=[3])
    panels = clean_panels()
    panels[2], panels[3] = board, scene
    sb = v3_storyboard(panels, CLAIMS_WITH_RESULT)
    found = lint_v3(sb, V3_PAPER_TEXT)
    assert not any("2.04" in x for x in found.get(3, []))
    assert any("evidence:" in x and "2.04" in x for x in found.get(4, []))   # a scene draws no gap
    removed = scrub_unsupported(sb, V3_PAPER_TEXT)
    assert sb.panels[2].dialogue[0].text == "It beats ConvS2S by 2.04 BLEU." and removed == ["panel 4: It beats ConvS2S by 2.04 BLEU."]


def test_lint_catches_mixed_setups_and_a_weak_baseline():
    parsing = {"kind": "bar", "title": "Parsing F1", "bars": [
        {"label": "Dyer et al. WSJ only", "value": "91.7", "baseline": True}, {"label": "Transformer WSJ only", "value": "91.3"},
        {"label": "Transformer semi-supervised", "value": "92.7", "highlight": True}]}
    assert any(x.startswith("the chart mixes training data (semi-supervised, wsj only)") for x in _story_board_issues(parsing))
    bleu = {"kind": "bar", "title": "EN-DE BLEU", "bars": [
        {"label": "GNMT Ensemble", "value": "26.30", "baseline": True}, {"label": "ConvS2S Ensemble", "value": "26.36"},
        {"label": "Transformer base", "value": "27.3"}, {"label": "Transformer big", "value": "28.4", "highlight": True}]}
    weak = [x for x in _story_board_issues(bleu) if x.startswith('the baseline "GNMT Ensemble" is not the strongest')]
    assert weak and "ConvS2S Ensemble" in weak[0] and "Transformer base" not in weak[0]   # the paper's own variant is no rival
    bleu["bars"][0]["baseline"], bleu["bars"][1]["baseline"] = False, True
    assert not any("not the strongest" in x for x in _story_board_issues(bleu))


def test_an_unsupported_bar_gets_the_paper_lines_that_name_it():
    paper = ("Figure 2 legend\nPrior best\n  60  55  57\nTable 1: GSM8K solve rate\n"
             "Prior best  N/A (finetuning) 55^(a)  57.4^(b)\nChain of thought 56.9 (+39.0) 79.0 (+9.6)\nPrior best numbers are from Cobbe et al.")
    chart = {"kind": "bar", "title": "GSM8K", "source": "Figure 2", "bars": [
        {"label": "Prior best", "value": "55", "quote": "Prior best 55", "baseline": True},
        {"label": "PaLM 540B chain of thought", "value": "57", "quote": "chain-of-thought prompting 57", "highlight": True}]}
    found = [x for x in board_issues(chart, paper) if x.startswith("evidence: bar")]
    assert '"Prior best  N/A (finetuning) 55^(a)  57.4^(b)"' in found[0].split("Paper lines that name it: ", 1)[1].split(" | ")[0]
    assert found[1].split("Paper lines that name it: ", 1)[1].startswith('"Chain of thought 56.9 (+39.0)')   # the table's 56.9 for the figure's 57


def test_lint_flags_word_cards_and_figure_shaped_props():
    panels = clean_panels()
    panels[0].visual = "Papi points to word cards arranged in a line, next to a toy robot."
    problems = lint_v3(v3_storyboard(panels, CLAIMS_WITH_RESULT), V3_PAPER_TEXT)[1]
    assert any("props that carry writing (word cards)" in x for x in problems)
    assert any("shaped like a person or creature (toy robot)" in x or "(robot)" in x for x in problems)


def test_salvage_snaps_figure_read_values_to_the_papers_rows():
    paper = ("Figure 2 legend\nPrior best\n  60  55  57\nTable 1\nPrior best  N/A (finetuning) 55^(a)  57.4^(b)\n"
             "Chain of thought 56.9 (+39.0) 79.0 (+9.6)\nPaLM 540B  Standard  17.9  69.4")
    chart = {"kind": "bar", "title": "GSM8K", "source": "Figure 2", "bars": [
        {"label": "PaLM 540B standard prompting", "value": "18", "quote": "x"},
        {"label": "Finetuned GPT-3 + verifier", "paper_label": "Prior best", "value": "55", "quote": "Prior best 55", "baseline": True},
        {"label": "PaLM 540B chain of thought", "value": "57", "quote": "chain-of-thought prompting 57", "highlight": True}]}
    snapped = _salvage_board(chart, paper)
    assert [(b["value"], b["quote"].split()[0]) for b in snapped["bars"]] == [("17.9", "PaLM"), ("55", "Prior"), ("56.9", "Chain")]
    far = {**chart, "bars": [{**chart["bars"][0], "value": "25"}, chart["bars"][2]]}   # 25 is no reading of 17.9
    lone = _salvage_board(far, paper)                   # one bar left: a one-bar chart compares nothing, its number is evidence
    assert lone["kind"] == "table" and lone["rows"] == [{"cells": ["PaLM 540B chain of thought", "56.9"],
                                                         "quote": "Chain of thought 56.9 (+39.0) 79.0 (+9.6)"}]


async def test_a_demoted_board_takes_its_caption_with_it():
    broken = {"kind": "links", "title": "Attention weights for the verb making", "tokens": ["laws", "making", "difficult"],
              "from": 1, "weights": [0, 0, 0]}

    def story(payload):
        payload["panels"][3]["board"] = broken
        payload["panels"][3]["narration"] = "Attention weights for the verb making."
        return payload

    model = PaperModel(story=story, fix=lambda panel, problems: panel)
    sb = await StoryboarderV2(FakeEngine(router=model)).build_v3(v3_paper(), make_cast(), "en-US", (16, 32))
    panel = sb.panels[3]
    assert (panel.kind, panel.board, panel.narration) == ("scene", None, "") and panel.dialogue


def test_a_near_value_is_never_snapped_from_a_prose_sentence():
    paper = ("the accuracy of GPT-3 on SST-2 ranges from near chance (54.3%) to near SOTA (93.4%)\n"
             "Prior best  N/A (finetuning) 55^(a)  57.4^(b)\nChain of thought 56.9 (+39.0) 79.0 (+9.6)")
    chart = {"kind": "bar", "title": "GSM8K", "bars": [
        {"label": "GPT-3 + verifier", "value": "55", "quote": "x", "baseline": True},
        {"label": "PaLM 540B chain of thought", "value": "57", "quote": "y", "highlight": True}]}
    assert _salvage_board(chart, paper) is None           # 54.3 is not the prior best: the GPT-3 bar goes, one bar is no chart


async def test_build_v3_asks_again_when_the_story_reply_cannot_be_read():
    model = PaperModel()
    broken = {"n": 0}

    def router(prompt, messages, config):
        user = messages[-1].content
        if user.startswith("Turn this research paper") and broken["n"] == 0:
            broken["n"] += 1
            return '{"panels": [{"n": 1, "kind": "scene" "beat": "hook"}]}'   # a missing comma
        if user.startswith("Return the following as strictly valid JSON") and broken["n"] == 1:
            broken["n"] += 1
            return '{"panels": [ still broken'
        return model(prompt, messages, config)

    sb = await StoryboarderV2(FakeEngine(router=router)).build_v3(v3_paper(), make_cast(), "en-US", (16, 32))
    assert broken["n"] == 2 and len(sb.panels) >= 9           # the second story call was read


async def test_quiz_revision_places_new_panels_numbered_past_the_script():
    """A revision reply that numbers its new panel after the last one ("n": 99) instead of placing it ("after") is still
    inserted, before the recap."""
    seen = {"revise": 0}
    model = PaperModel(quiz_misses=[{3}])

    def router(prompt, messages, config):
        user = messages[-1].content
        if user.startswith("A reader who never saw the paper read only the visible text"):
            seen["revise"] += 1
            return {"panels": [{"n": 99, "kind": "scene", "beat": "proof", "title": "New", "characters": ["papi"], "scene": "a lab",
                                "visual": "papi points", "dialogue": [{"speaker": "papi", "text": "Our model achieves 28.4 BLEU."}],
                                "narration": "", "emotions": {}, "board": None, "claims": [3]}]}
        return model(prompt, messages, config)

    sb = await StoryboarderV2(FakeEngine(router=router)).build_v3(v3_paper(), make_cast(), "en-US", (16, 32))
    assert seen["revise"] == 1 and sb.quiz.get("before") is not None          # the revision applied and was re-tested
    assert any(b.text == "Our model achieves 28.4 BLEU." for p in sb.panels for b in p.dialogue)
    assert sb.panels[-1].beat == "recap"


async def test_a_quiz_that_fails_after_the_revision_is_recorded(monkeypatch):
    import services.abstract_quiz as aq
    original, calls = aq.run_quiz, []

    async def flaky(*args, **kwargs):   # the first quiz runs, the one after the revision cannot
        calls.append(1)
        if len(calls) == 1:
            return await original(*args, **kwargs)
        return {"score": None, "error": "ValueError: no JSON object in model reply", "usage": {}}

    monkeypatch.setattr(aq, "run_quiz", flaky)
    story = StoryboarderV2(FakeEngine(router=PaperModel(quiz_misses=[{3}])))
    sb = await story.build_v3(v3_paper(), make_cast(), "en-US", (16, 32))
    assert len(calls) == 2 and sb.quiz.get("undone") == 1        # the unmeasured revision goes: the shipped script is the scored one
    assert sb.quiz["score"] is not None and not any(p.title == "Claim 3" for p in sb.panels)
    assert any(x["stage"] == "abstract quiz after the revision" and "no JSON" in x["error"] for x in story.skipped)


def test_the_brief_asks_for_no_symbols_and_the_story_explains_on_diagrams():
    from services.storyboarder_v2 import brief_prompt_v3
    brief = brief_prompt_v3(v3_paper(), "An abstract.")
    assert '"picture_symbols"' not in brief and '"on_example"' in brief and '"metaphor"' not in brief
    prompt = story_prompt(BRIEF_V3, V3_CLAIMS, make_cast(), (16, 24), v3_paper())
    assert "DIAGRAMS EXPLAIN, SCENES TELL" in prompt and '"kind": "diagram"' in prompt and "PICTURES TEACH" not in prompt
    assert "Never make a character say what a drawn object stands for" in prompt and "picture symbol" not in prompt
    assert "Every result in the BRIEF's \"evidence\" is shown on a chart" in prompt
    assert '"shows": ""' in prompt and '"shows" is "" on every panel' in prompt   # round 23: no picture is asked to explain
    assert "No object stands in for a concept" in prompt and "red yarn, building blocks" not in prompt
    assert "teaching scene (a scene with a \"shows\")" not in prompt and "16-24 panels" in prompt


async def test_a_draft_well_under_the_range_is_written_once_more_and_the_longer_one_is_kept():
    # round 23 split no panels any more, and 12-13 panel drafts stayed that short (Flexibility Trap's lost the abstract's
    # background); a draft one panel short is kept as it is
    from .fakes import story_payload_v3
    short = story_payload_v3()
    longer = story_payload_v3()
    extra = [dict(longer["panels"][1], n=10 + k) for k in range(3)]
    longer["panels"] = longer["panels"] + extra
    drafts = [short, longer]
    model = PaperModel(story=lambda payload: drafts.pop(0))
    sb = await StoryboarderV2(FakeEngine(router=model)).write_v3(dict(BRIEF_V3), [dict(c) for c in V3_CLAIMS], make_cast(), (11, 24),
                                                                  v3_paper())
    assert model.seen.count("story") == 2 and len(sb.panels) == 12
    model = PaperModel()
    sb = await StoryboarderV2(FakeEngine(router=model)).write_v3(dict(BRIEF_V3), [dict(c) for c in V3_CLAIMS], make_cast(), (10, 24),
                                                                  v3_paper())
    assert model.seen.count("story") == 1 and len(sb.panels) == 9


# ----------------------------------------------------------------------------- round 25: one example from the analysis on

SPINE = {"text": f"Kumo reads one sentence: {V3_EXAMPLE}", "from_paper": True, "source": "Figure 3",
         "parts": ["The animal", "it", "too tired"], "old_way": "A recurrent model reads it word by word.",
         "new_way": "Attention links 'it' to 'animal' in one step."}


def test_brief_example_keeps_one_case_with_its_parts():
    # round 23's judges all asked for one concrete case drawn at every step; no step chose it with its real parts
    from services.storyboarder_v2 import EXAMPLE_PARTS, brief_example
    raw = {"text": " Who used the Vitae Extract? ", "from_paper": "true", "source": "Appendix C.1",
           "parts": ["Who used the Vitae Extract?", {"text": "Sofia Bianchi"}, "who used  the vitae Extract?", 7, "", None]
           + [f"document {i}" for i in range(10)],
           "old_way": "A short-context model cannot read the documents.", "new_way": ["not a text"]}
    ex = brief_example(raw)
    assert ex["text"] == "Who used the Vitae Extract?" and ex["from_paper"] is True and ex["source"] == "Appendix C.1"
    assert ex["parts"][:3] == ["Who used the Vitae Extract?", "Sofia Bianchi", "7"] and len(ex["parts"]) == EXAMPLE_PARTS
    assert ex["old_way"].startswith("A short-context") and ex["new_way"] == ""
    assert brief_example({"text": "A made-up queue of calls", "from_paper": False, "source": "Figure 9"})["source"] == ""
    assert brief_example(None) == {} and brief_example({"parts": ["x"]}) == {} and brief_example("a sentence") == {}


async def test_check_brief_reads_the_running_example_against_the_paper():
    from services.storyboarder_v2 import _brief_items
    brief = {"key_idea": "k", "running_example": {"text": "Q: who used it? A: Sofia", "from_paper": True, "source": "App. C",
                                                  "parts": ["Q", "made up", "also made up", "A: Sofia"], "old_way": "old", "new_way": "new"}}
    ids = [i["id"] for i in _brief_items(brief)]
    assert {"running_example.text", "running_example.old_way", "running_example.parts.3"} <= set(ids)
    engine = FakeEngine([{"problems": [
        {"id": "running_example.parts.1", "issue": "not in the paper", "corrected": "DELETE"},
        {"id": "running_example.parts.2", "issue": "not in the paper", "corrected": "DELETE"},   # numbers name the parts as sent
        {"id": "running_example.parts.3", "issue": "partial", "corrected": "A: Sofia Bianchi"},
        {"id": "running_example.old_way", "issue": "vague", "corrected": "Old, as the paper says."}]}])
    assert await StoryboarderV2(engine).check_brief(brief, PaperText("T", "body", [])) == 4
    assert brief["running_example"]["parts"] == ["Q", "A: Sofia Bianchi"]
    assert brief["running_example"]["old_way"] == "Old, as the paper says." and brief["running_example"]["from_paper"] is True
    # a case the paper does not show stays, as an illustration
    engine = FakeEngine([{"problems": [{"id": "running_example.text", "issue": "not in the paper", "corrected": "DELETE"}]}])
    await StoryboarderV2(engine).check_brief(brief, PaperText("T", "body", []))
    assert brief["running_example"]["from_paper"] is False and brief["running_example"]["text"] == "Q: who used it? A: Sofia"
    assert brief["running_example"]["source"] == ""
    assert not [i for i in _brief_items(brief) if i["id"].startswith("running_example")]   # a made-up case is not read


async def test_the_analysis_example_runs_through_the_story():
    model = PaperModel()

    def router(prompt, messages, config):
        reply = model(prompt, messages, config)
        if messages[-1].content.startswith("Read the paper and its ABSTRACT"):
            reply = {**reply, "running_example": dict(SPINE)}
        return reply
    engine = FakeEngine(router=router)
    sb = await StoryboarderV2(engine).build_v3(v3_paper(), make_cast(), "en-US", (16, 32))
    assert sb.running_example == SPINE["text"] and sb.brief["running_example"]["parts"] == SPINE["parts"]
    assert sb.brief["running_example"]["from_paper"] is True and sb.brief["running_example"]["source"] == "Figure 3"
    story = prompt_of(engine, "Turn this research paper into a SHORT MANGA STORY")
    assert f"THE RUNNING EXAMPLE: {SPINE['text']} (Figure 3)" in story and '["The animal", "it", "too tired"]' in story
    assert "What the paper's method does to it: Attention links" in story and "never switch to another one" in story
    assert story.count("A recurrent model reads it word by word.") == 1   # once, not again in the brief's facts
    assert "Other example inputs the paper shows" not in story   # the paper's one example is the running example
    assert '"id": "running_example.parts.2"' in prompt_of(engine, "You verify the BRIEF")
    assert '"mechanisms"' in brief_prompt_v3(v3_paper(), "abstract") and '"running_example": {"text"' in brief_prompt_v3(v3_paper(), "a")


def test_without_an_analysis_example_the_writer_still_picks_one():
    from services.storyboarder_v2 import _examples_block
    assert _examples_block({"examples": [{"text": V3_EXAMPLE, "source": "Figure 3"}]}).startswith("Example inputs the paper itself shows")
    made_up = _examples_block({"running_example": {"text": "Kumo sorts parcels", "from_paper": False, "parts": []}})
    assert "made up" in made_up and "Its parts" not in made_up


def test_the_example_parts_count_for_the_example_rules():
    from services.storyboarder_v2 import _example_tokens, _illustrative_example
    sb = v3_storyboard(clean_panels(), brief={"running_example": {"text": "Kumo sorts parcels", "from_paper": False,
                                                                  "parts": ["Parcel 7 at the depot", "Driver Nezu"]}})
    sb.running_example = "Kumo sorts parcels"
    assert {"parcel", "depot", "driver", "nezu"} <= _example_tokens(sb)
    # a made-up case's numbers are its text's alone: a part "on-time rate 62.5%" passed as a result (round 25's review)
    assert _illustrative_example(sb) == "Kumo sorts parcels"
    sb.brief["running_example"]["from_paper"] = True
    assert _illustrative_example(sb) == ""
    from services.storyboarder_v2 import brief_example
    assert brief_example({"text": "t", "parts": ["+1 step", "-1 step"]})["parts"] == ["+1 step", "-1 step"]   # signs count


def test_a_parts_label_is_no_word_of_the_example():
    # "Who answers the question?" showed the example on the labels' words alone (round 25's review)
    from services.storyboarder_v2 import _example_beats, _example_tokens
    panels = clean_panels()
    panels[1] = v3_panel(2, beat="how", dialogue=[("papi", "Who answers the question?")], claims=[1, 2])
    sb = v3_storyboard(panels, CLAIMS_WITH_RESULT, brief={"running_example": {
        "text": "The Vitae Extract question", "from_paper": True, "parts": ["Vitae Extract question", "answer: Sofia Bianchi"]}})
    sb.running_example = "The Vitae Extract question"
    assert "answer" not in _example_tokens(sb) and {"sofia", "bianchi", "vitae"} <= _example_tokens(sb)
    assert _example_beats(sb)["how"] is False
    panels[1].dialogue = [Bubble("papi", "The answer Sofia Bianchi is fixed first.")]
    assert _example_beats(sb)["how"] is True
    sb.brief["running_example"]["parts"] = ["14:00 delivery slot"]   # a time is no label
    assert "delivery" in _example_tokens(sb)


def test_a_part_never_spares_a_headline_number_from_the_repeat_rule():
    from services.storyboarder_v2 import _repeated_numbers
    claim = {"id": 3, "text": "The Transformer reaches 28.4 BLEU.", "quote": "achieves 28.4 BLEU on the WMT 2014 English-to-German"}
    panels = [v3_panel(n, beat="proof", dialogue=[("papi", "It got 28.4 BLEU, look.")], claims=[]) for n in range(1, 5)]
    sb = v3_storyboard(panels, [claim], brief={"running_example": {"text": "One German sentence", "from_paper": True,
                                                                   "parts": ["the sentence's score: 28.4 BLEU"]}})
    sb.running_example = "One German sentence"
    assert _repeated_numbers(sb)   # the part repeats the headline: the rule still counts it
    sb.brief["running_example"]["parts"] = ["the sentence has 31 words"]
    sb.panels = [v3_panel(n, beat="how", dialogue=[("papi", "All 31 words at once.")]) for n in range(1, 5)]
    assert _repeated_numbers(sb) == {}   # the example's own count may come back on every step


async def test_a_story_that_follows_another_case_keeps_its_own():
    # the writer was told never to switch; when it does, the lints and the designer follow the case the story tells
    other = {"text": "Kumo sorts parcels at a depot before dawn.", "from_paper": False, "parts": ["Parcel 7", "Driver Nezu"],
             "old_way": "x", "new_way": "y"}
    sb = await StoryboarderV2(FakeEngine(router=PaperModel())).write_v3(
        {**BRIEF_V3, "running_example": dict(other)}, [dict(c) for c in V3_CLAIMS], make_cast(), (10, 24), v3_paper())
    assert sb.running_example.startswith("Kumo wants to understand one sentence at once")   # the story's own case
    assert "parts" not in sb.brief["running_example"]
    followed = await StoryboarderV2(FakeEngine(router=PaperModel())).write_v3(
        {**BRIEF_V3, "running_example": dict(SPINE)}, [dict(c) for c in V3_CLAIMS], make_cast(), (10, 24), v3_paper())
    assert followed.running_example == SPINE["text"] and followed.brief["running_example"]["parts"] == SPINE["parts"]


async def test_the_brief_check_reads_that_the_paper_shows_this_one_instance():
    from services.storyboarder_v2 import _brief_items
    brief = {"key_idea": "k", "running_example": {"text": "Baselines put 46.7% of attention on the first token.",
                                                  "from_paper": True, "source": "Figure 2", "parts": [], "old_way": "", "new_way": ""}}
    item = next(i for i in _brief_items(brief) if i["id"] == "running_example.from_paper")
    assert "this one instance (Figure 2), not only averages" in item["text"]
    engine = FakeEngine([{"problems": [{"id": "running_example.from_paper", "issue": "an average", "corrected": "x"}]}])
    assert await StoryboarderV2(engine).check_brief(brief, PaperText("T", "body", [])) == 1
    assert brief["running_example"]["from_paper"] is False and brief["running_example"]["source"] == ""


def test_the_papers_own_wording_of_the_case_is_never_another_example():
    from services.storyboarder_v2 import _examples_block
    block = _examples_block({"examples": [{"text": "180 - ( × 180 - ( - 180) = 180 - 0.30 × 180 = 54 ...", "source": "Appendix D"},
                                          {"text": "Natalia sold clips to 48 of her friends in April.", "source": "Table 9"}],
                             "running_example": {"text": "A broken solution of the app-downloads problem", "from_paper": True,
                                                 "parts": ["180 - ( × 180 - ( - 180)", "= 180 - 0.30 × 180 = 54"]}})
    assert "In the paper's own words: 180 - ( × 180" in block
    others = block.split("Other example inputs", 1)[1]
    assert "Natalia" in others and "180 - (" not in others


# ----------------------------------------------------------------------------- round 26: each part with its role; the figures

def test_each_part_keeps_what_it_is_in_the_case():
    # round 25's designer, given eight candidate tokens of one step as a flat list, drew them as the student's sentence
    from services.storyboarder_v2 import _brief_items, _examples_block, brief_example
    ex = brief_example({"text": "One prefix of the 2-bit student", "source": "Figure 3",
                        "parts": [{"text": "a", "what": "a candidate the teacher favours"},
                                  {"text": "some", "role": "the student's favourite"}, "two"]})
    assert ex["parts"] == ["a", "some", "two"] and ex["roles"] == ["a candidate the teacher favours", "the student's favourite", ""]
    items = {i["id"]: i["text"] for i in _brief_items({"running_example": ex})}
    assert items["running_example.parts.0"] == "a — a candidate the teacher favours" and items["running_example.parts.2"] == "two"
    block = _examples_block({"running_example": ex})
    assert "each with what it is in the case" in block and '"a — a candidate the teacher favours"' in block and '"two"' in block


async def test_the_brief_check_corrects_a_part_together_with_its_role():
    from services.storyboarder_v2 import brief_example
    ex = brief_example({"text": "One prefix of the 2-bit student", "source": "Figure 3",
                        "parts": [{"text": "the", "what": "the teacher's top candidate"}, {"text": "four", "what": "the student's favourite"},
                                  {"text": "two", "what": "a candidate both weigh"}]})
    brief = {"key_idea": "k", "running_example": ex}
    engine = FakeEngine([{"problems": [
        {"id": "running_example.parts.0", "issue": "Figure 3 gives 'the' about zero", "corrected": "a — the teacher's top candidate"},
        {"id": "running_example.parts.1", "issue": "not in the figure", "corrected": "DELETE"},
        {"id": "running_example.parts.2", "issue": "partial", "corrected": "two"}]}])
    assert await StoryboarderV2(engine).check_brief(brief, PaperText("T", "body", [])) == 3
    assert brief["running_example"]["parts"] == ["a", "two"]
    assert brief["running_example"]["roles"] == ["the teacher's top candidate", ""]   # a correction without a role: unknown


def test_the_analysis_looks_at_the_figures_and_the_writer_invents_no_outcome():
    prompt = brief_prompt_v3(v3_paper(), "abstract")
    assert "FIGURES READ FROM THE IMAGES" in prompt and '"what": "what it is in this case"' in prompt
    assert "the one at which the paper shows its key step acting" in prompt
    assert "never an outcome for this case that the paper does not report" in prompt
    story = story_prompt(BRIEF_V3, V3_CLAIMS, make_cast(), (16, 24), v3_paper())
    assert "When the running example comes from the paper, say and draw only what the paper shows" in story
    assert "never invent one (no after-picture the paper does not draw" in story
    assert "never spend a line or a panel saying that the paper does not show it" in story
