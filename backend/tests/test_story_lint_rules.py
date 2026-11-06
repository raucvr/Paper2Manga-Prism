"""Lint rules from the reader reports: no "every claim was verified", no drawing notes on boards, no chart talk on a
scene panel, and a comparison claim drawn as a comparison."""

from __future__ import annotations

import pytest

from services.storyboarder_v2 import Bubble, _story_board_issues, lint_v3

from .fakes import V3_PAPER_TEXT
from .test_story_v3 import CLAIMS_WITH_RESULT, clean_panels, v3_panel, v3_storyboard


def problems_of(panels):
    return lint_v3(v3_storyboard(panels, CLAIMS_WITH_RESULT), V3_PAPER_TEXT)


def test_the_clean_script_still_passes():
    assert problems_of(clean_panels()) == {}


def test_certifying_the_whole_paper_is_flagged_in_text_and_on_boards():
    panels = clean_panels()
    panels[3].dialogue[0].text = "Every claim was verified."
    panels[3].board = {**panels[3].board, "subtitle": "All claims checked against experimental results"}
    found = problems_of(panels)[4]
    assert any("never that every claim was verified" in x and x.startswith('"all claims checked"') for x in found)


def test_a_board_that_describes_its_drawing_is_flagged():
    panels = clean_panels()
    panels[3].board = {**panels[3].board, "subtitle": "Empty checkboxes: we will verify each"}
    assert any("describe the drawing" in x for x in problems_of(panels)[4])


def test_chart_talk_on_a_scene_panel_is_flagged():
    panels = clean_panels()
    panels[1].dialogue[0].text = "This chart shows attention only."
    assert any("scene panel with no chart" in x for x in problems_of(panels)[2])


def test_a_comparison_claim_needs_a_drawn_comparison():
    panels = clean_panels()
    # the results claim (28.4 BLEU, over 2 BLEU better) taught on a scene panel only: its number is there, its comparison is not
    panels[2] = v3_panel(3, beat="proof", dialogue=[("nezu", "28.4 BLEU, over 2.0 above the rest?")], claims=[3],
                         characters=("nezu", "papi"))
    found = problems_of(panels)
    assert any("is a comparison, but no panel that teaches it draws one" in x for x in found.get(3, []))



def test_the_recap_ticks_only_lines_whose_numbers_a_chart_or_table_draws():
    # the legend under the recap says "tick: a chart or table shows it"; a number a bubble said or a list board listed is
    # the paper's report, not drawn evidence (round 19 ticked "Depth up to 1024 layers" beside a chart of 4 and 64 layers,
    # because a bubble said 1024)
    from services.storyboarder_v2 import _honest_recap
    panels = clean_panels()                        # panel 3's table draws 28.4; the recap lists it and an undrawn 41.8
    panels[3].board = {"kind": "list", "title": "What the paper showed", "open": False,
                       "items": ["28.4 BLEU on WMT 2014 English-German", "41.8 BLEU on English-French"]}
    sb = v3_storyboard(panels, CLAIMS_WITH_RESULT)
    _honest_recap(sb)
    assert sb.panels[3].board["checked"] == [0]
    panels[3].board["items"] = panels[3].board["items"][:1]
    _honest_recap(sb)
    assert "checked" not in sb.panels[3].board       # every line drawn: plain ticks
    panels[3].board["items"] = ["28.4 BLEU on WMT 2014 English-German", "26,000 real user queries"]
    panels[1].dialogue[0].text = "And 26,000 real user queries."              # said in a bubble...
    _honest_recap(sb)
    assert sb.panels[3].board["checked"] == [0]
    panels[1].kind, panels[1].board = "board", {"kind": "list", "title": "The dataset", "items": ["26,000 real user queries"]}
    _honest_recap(sb)                                                          # ...or listed on a list board: the dot
    assert sb.panels[3].board["checked"] == [0]
    panels[1].board = {"kind": "table", "title": "The dataset", "columns": ["Queries"], "rows": [{"cells": ["26,000"]}]}
    _honest_recap(sb)                                                          # drawn in a table: the tick
    assert "checked" not in sb.panels[3].board
    # a title that certifies every line goes when a line only gets a bullet
    panels[1].kind, panels[1].board = "scene", None
    panels[3].board = {**panels[3].board, "title": "Verified Results", "subtitle": "Verified results from this work",
                       "items": ["28.4 BLEU", "41.8 BLEU on English-French"]}
    _honest_recap(sb, retitle=False)   # a relayout of a translated script keeps its titles
    assert sb.panels[3].board["checked"] == [0] and sb.panels[3].board["title"] == "Verified Results"
    _honest_recap(sb)
    assert sb.panels[3].board["checked"] == [0] and sb.panels[3].board["title"] == "What the paper showed"
    assert "subtitle" not in sb.panels[3].board
    # a number said beside a chart is not drawn on it (round 19: "Depth up to 1024 layers" beside a chart of 4 and 64)
    panels[2].dialogue[0].text = "28.4 here, and they stacked up to 1024 layers."
    panels[3].board = {**panels[3].board, "items": ["28.4 BLEU", "Up to 1024 layers"]}
    _honest_recap(sb)
    assert sb.panels[3].board["checked"] == [0]


def test_a_chart_in_a_scene_picture_is_flagged():
    panels = clean_panels()
    panels[1].scene, panels[1].visual = "Kumo looks at the bar chart", "Nezu traces the rising bar heights"
    found = problems_of(panels).get(2, [])
    assert any("shows a chart (bar chart, bar heights)" in x for x in found)
    panels[1].scene, panels[1].visual = "A bar at night", "Kumo orders a juice"   # a bar is not a chart
    assert not any("shows a chart" in x for x in problems_of(panels).get(2, []))



def test_the_gap_a_chart_draws_counts_as_shown_in_the_recap():
    from services.storyboarder_v2 import _honest_recap
    panels = clean_panels()
    panels[2].board = {"kind": "bar", "title": "WMT 2014 English-German (BLEU)",
                       "bars": [{"label": "ConvS2S ensemble", "value": "26.36", "baseline": True},
                                {"label": "Transformer (big)", "value": "28.4", "highlight": True}]}
    panels[3].board = {"kind": "list", "title": "What the paper showed",
                       "items": ["28.4 BLEU, +2.04 over the best ensemble"]}   # the badge draws +2.04
    sb = v3_storyboard(panels, CLAIMS_WITH_RESULT)
    _honest_recap(sb)
    assert "checked" not in sb.panels[3].board


def test_the_opening_never_points_at_what_the_reader_has_not_seen():
    # Hivemind opened "This sameness risks ...", LongZero "This homework ..."
    panels = clean_panels()
    panels[0].narration = "This sameness risks making human thought alike."
    found = problems_of(panels).get(1, [])
    assert any('the story opens with "This sameness' in x and '"This" points at nothing' in x for x in found)
    panels[0].narration = ""
    panels[0].dialogue[0].text = "That is odd: every AI wrote the same poem."   # an exclamation, not a pointer
    assert not any("the story opens with" in x for x in problems_of(panels).get(1, []))
    panels[0].dialogue[0].text = "These answers all sound the same."           # no narration: the first bubble opens
    assert any("the story opens with" in x for x in problems_of(panels).get(1, []))


def test_a_chapter_that_goes_back_to_an_earlier_point_is_flagged():
    # Hivemind's panels 8 to 13 alternated "the first finding" and "the second finding"
    panels = clean_panels()
    extra = [v3_panel(n, beat="idea", dialogue=[("papi", text)], claims=[1]) for n, text in
             ((5, "The first finding: attention alone is enough."), (6, "The second finding: it trains in parallel."),
              (7, "Back to the first finding: no recurrence at all."))]
    panels = panels[:2] + extra + panels[2:]
    found = problems_of(panels)
    assert any("goes back to the first point after panel 6 reached the second" in x for x in found.get(7, []))
    assert not any("goes back" in x for n in (5, 6) for x in found.get(n, []))
    extra[2].dialogue[0].text = "A third finding: it also parses sentences."
    assert not any("goes back" in x for found in problems_of(panels).values() for x in found)


def test_a_number_in_a_chart_panels_words_must_be_on_its_board():
    # Flexibility Trap: "only 0.6% on HumanEval (Pass@1024)" beside a chart of solved problems
    panels = clean_panels()
    panels[2].dialogue[0].text = "28.4 here, but only 0.6% on HumanEval."
    found = problems_of(panels).get(3, [])
    assert any("the words say 0.6, which this table does not show" in x for x in found)
    panels[2].dialogue[0].text = "28.4, more than 2.0 above the rest?"   # a gap is not a value on the board
    assert not any("does not show" in x for x in problems_of(panels).get(3, []))
    panels[2].dialogue[0].text = "28.4 with a 4B model, or 1.7 billion parameters."   # model sizes name the setting
    assert not any("does not show" in x for x in problems_of(panels).get(3, []))
    panels[2].dialogue[0].text = "28.4 BLEU, and 14 times fewer training steps."   # a ratio is a gap
    assert not any("does not show" in x for x in problems_of(panels).get(3, []))


def test_a_claims_own_number_may_be_said_beside_a_chart_that_does_not_draw_it():
    # "41.8 BLEU after 3.5 days on eight GPUs" taught on the BLEU chart: the claim check asks for 3.5 right there, so the
    # board check must not send it away again
    claims = CLAIMS_WITH_RESULT + [{"id": 4, "text": "41.0 BLEU after 3.5 days", "quote": "41.0 after training for 3.5 days"}]
    panels = clean_panels()
    panels[2].dialogue[0].text = "28.4 here, and 41.0 after 3.5 days of training."
    panels[2].claims = [3, 4]
    found = lint_v3(v3_storyboard(panels, claims), V3_PAPER_TEXT).get(3, [])
    assert not any("does not show" in x for x in found)
    panels[2].claims = [3]
    assert any("the words say 3.5" in x for x in lint_v3(v3_storyboard(panels, claims), V3_PAPER_TEXT).get(3, []))


def test_chat_windows_in_a_scene_picture_are_flagged():
    panels = clean_panels()
    panels[1].visual = "Kumo slumps forward, empty reply windows float above them"
    assert any("chat windows or text boxes (reply windows)" in x for x in problems_of(panels).get(2, []))
    panels[1].visual = "Kumo opens a window to let the breeze in"
    assert not any("chat windows" in x for x in problems_of(panels).get(2, []))


def test_a_links_board_belongs_to_papers_about_attention():
    import re
    panels = clean_panels()
    panels[1] = v3_panel(2, kind="board", beat="idea", characters=("papi",), claims=[1, 2],
                         dialogue=[("papi", "Each word looks at every other word.")],
                         board={"kind": "links", "title": "Who looks at whom", "tokens": ["The", "animal", "was", "tired"], "from": 3,
                                "weights": [0.1, 0.7, 0.1, 0.1]})
    sb = v3_storyboard(panels, CLAIMS_WITH_RESULT)
    message = "this paper is not about attention between words"
    assert not any(message in x for x in lint_v3(sb, V3_PAPER_TEXT).get(2, []))
    other = re.sub(r"(?i)attention|attend\w*", "focus", V3_PAPER_TEXT)   # a theory paper: no attention anywhere
    assert any(message in x and "flow board" in x for x in lint_v3(sb, other).get(2, []))


def test_a_drop_in_words_matches_the_negative_bar_it_names():
    # Driver Misconduct: "A 1.60% drop?" beside bars of -1.60 and -1.86 was flagged in every fix round
    from services.storyboarder_v2 import _words_vs_chart
    board = {"kind": "bar", "title": "Next day success change", "unit": "%",
             "bars": [{"label": "Successful deliveries", "value": "-1.60", "highlight": True}, {"label": "First try deliveries", "value": "-1.86"}]}
    panel = v3_panel(9, kind="board", beat="proof", board=board, dialogue=[("kumo", "A 1.60% drop? That's huge!")])
    assert _words_vs_chart(panel) == []
    panel.dialogue[0].text = "A 2.5% drop? That's huge!"
    assert _words_vs_chart(panel)


def term_problems(panels, brief=None, paper=V3_PAPER_TEXT):
    from services.storyboarder_v2 import _unexplained_terms
    return _unexplained_terms(v3_storyboard(panels, CLAIMS_WITH_RESULT, brief=brief), paper)


def test_a_glossary_term_is_explained_where_the_reader_first_meets_it():
    # readers' most frequent note: "IV regression is named but never explained" (Driver Misconduct), FKL/DKL (Right On Q)
    glossary = {"glossary": [{"term": "Self-attention (SA)", "plain": "each token scores how relevant all other tokens are"}]}
    panels = clean_panels()
    panels[1].dialogue[0].text = "Only self-attention, no recurrence, no convolutions."
    found = term_problems(panels, glossary)
    assert list(found) == [2] and "Self-attention / SA" in found[2][0] and found[2][0].endswith("how relevant all other tokens are")
    panels[1].narration = "Self-attention: each word looks at the others."             # an explanation right after it
    assert term_problems(panels, glossary) == {}
    panels[1].narration = "Each token scores the other tokens."                        # the plain meaning in other words
    assert term_problems(panels, glossary) == {}
    panels[1].narration = ""
    panels[0].narration = "Translation turns one sentence into another. SA (self-attention) is the new trick."   # explained earlier
    assert term_problems(panels, glossary) == {}


def test_a_term_named_after_it_was_described_counts_as_explained():
    glossary = {"glossary": [{"term": "chain-of-thought prompting", "plain": "showing worked steps in the examples of a prompt"}]}
    panels = clean_panels()
    panels[1].dialogue = [Bubble("kumo", "What if we show the steps first?"), Bubble("papi", "Exactly. That is chain-of-thought prompting.")]
    assert term_problems(panels, glossary) == {}
    panels[1].dialogue[1].text = "Exactly. We use chain-of-thought prompting."
    assert list(term_problems(panels, glossary)) == [2]


def test_an_abbreviation_the_paper_uses_must_be_spelled_out():
    paper = V3_PAPER_TEXT + "\nWe compare with instrumental variable (IV) estimates and large language models (LLMs)."
    panels = clean_panels()
    panels[1].dialogue[0].text = "They used IV regression to find the cause."
    found = term_problems(panels, paper=paper)
    assert list(found) == [2] and 'the abbreviation "IV"' in found[2][0] and "PAPER TEXT" in found[2][0]
    panels[1].dialogue[0].text = "They used IV regression: it separates cause from coincidence."
    assert term_problems(panels, paper=paper) == {}
    panels[1].dialogue[0].text = "Large language models answer. LLMs are everywhere."   # spelled out before (3+ capitals)
    assert term_problems(panels, paper=paper) == {}
    panels[1].dialogue[0].text = "WOW. They used it everywhere."                        # not in the paper: an exclamation
    assert term_problems(panels, paper=paper) == {}


def test_names_on_bars_and_the_claims_board_are_not_a_first_meeting():
    paper = V3_PAPER_TEXT + "\nGNMT + RL 24.6 and the MoE model."
    panels = clean_panels()
    panels[2].board = {**panels[2].board, "rows": panels[2].board["rows"] + [{"cells": ["GNMT + RL", "24.6"]}]}
    assert term_problems(panels, paper=paper) == {}                                    # a compared system's name on a row
    claims = v3_panel(2, kind="board", beat="claims", board={"kind": "list", "title": "Claims", "items": ["The MoE model is slower"]},
                      dialogue=[("nezu", "Prove each line.")], characters=("nezu",))
    panels = [panels[0], claims] + panels[1:]
    for i, p in enumerate(panels):
        p.n = i + 1
    assert term_problems(panels, paper=paper) == {}                                    # listed, never taught: nothing to explain
    panels[2].dialogue[0].text = "Unlike the MoE model, only attention."
    assert list(term_problems(panels, paper=paper)) == [3]                             # the first teaching panel that uses it
    claims.dialogue[0].text = "MoE: a mixture of expert networks. Prove each line."
    assert term_problems(panels, paper=paper) == {}                                    # explained on the claims board


async def test_a_term_problem_brings_the_paper_to_the_fix_round():
    # the writer spells an abbreviation out from the paper, never from memory
    from services.storyboarder_v2 import StoryboarderV2
    from .fakes import FakeEngine
    from .test_story_v3 import make_cast, v3_paper
    panels = clean_panels()
    panels[1].dialogue[0].text = "Only attention, unlike the MoE model."
    paper = v3_paper()
    paper.body += "\nThe MoE model is a mixture of experts."
    engine = FakeEngine([{"panels": []}])
    await StoryboarderV2(engine).fix_v3(v3_storyboard(panels, CLAIMS_WITH_RESULT), make_cast(), paper, rounds=1)
    assert '"term" problems' in engine.prompts[0] and 'the abbreviation \\"MoE\\"' in engine.prompts[0]   # JSON-escaped
    assert "PAPER TEXT:\n" + paper.body.strip() in engine.prompts[0]


def test_the_full_name_before_an_abbreviations_bracket_explains_it():
    # "instrumental variable (IV)" and "forward KL (FKL)": the usual introduction must clear the problem, or it comes back in
    # every lint round
    paper = V3_PAPER_TEXT + "\nWe use IV estimates, forward KL (FKL) and RL fine-tuning."
    panels = clean_panels()
    for line in ("They used instrumental variable (IV) regression.", "Forward KL (FKL) beats the rest.",
                 "Reinforcement learning (RL) trains it."):
        panels[1].dialogue[0].text = line
        assert term_problems(panels, paper=paper) == {}, line
    panels[1].dialogue[0].text = "The results (IV) were strong."                      # a bracket, but no name before it
    assert list(term_problems(panels, paper=paper)) == [2]


def test_numbers_labels_and_generic_words_do_not_explain_a_term():
    glossary = {"glossary": [{"term": "MMLU", "plain": "a standard benchmark of exam questions in 57 subjects; higher is better"}]}
    panels = clean_panels()
    for line in ("MMLU: 89.1%, higher is better.", "MMLU (Table 3) is higher.", "Columns: MMLU and more."):
        panels[1].dialogue[0].text = line
        assert list(term_problems(panels, glossary)) == [2], line
    panels[1].dialogue[0].text = "MMLU asks exam questions in 57 subjects."
    assert term_problems(panels, glossary) == {}


def test_shouted_words_roman_numerals_and_glossary_names_are_not_abbreviations():
    paper = V3_PAPER_TEXT + "\nAnswer ONLY with yes or no. Use only the context. Only then. only. Type II errors. Dynamic KL (DKL)."
    panels = clean_panels()
    panels[1].dialogue[0].text = "ONLY attention! Fewer Type II errors too."
    assert term_problems(panels, paper=paper) == {}
    glossary = {"glossary": [{"term": "Dynamic KL (DKL)", "plain": "a loss that mixes two distances per token"}]}
    panels[1].dialogue[0].text = "Dynamic KL: a loss that mixes two distances per token."
    assert term_problems(panels, glossary, paper) == {}                                # KL inside the explained name


def test_a_compare_boards_sides_count_as_words():
    paper = V3_PAPER_TEXT + "\nPPO and GRPO are compared."
    panels = clean_panels()
    panels[1] = v3_panel(2, kind="board", beat="idea", characters=("papi",), claims=[1, 2], dialogue=[("papi", "Two ways to train.")],
                         board={"kind": "compare", "title": "Before and after", "left": {"label": "PPO", "items": ["one critic"]},
                                "right": {"label": "Ours", "items": ["no critic"]}})
    found = term_problems(panels, paper=paper)
    assert list(found) == [2] and '"PPO"' in found[2][0]


async def test_a_glossary_term_problem_needs_no_paper_text():
    # its meaning travels with the problem; only an abbreviation is spelled out from the paper
    from services.storyboarder_v2 import StoryboarderV2
    from .fakes import FakeEngine
    from .test_story_v3 import make_cast, v3_paper
    panels = clean_panels()
    panels[1].dialogue[0].text = "The Transformer, a simple architecture: only self-attention, so no recurrence or convolutions."
    glossary = {"glossary": [{"term": "self-attention", "plain": "each token scores how relevant all other tokens are"}]}
    engine = FakeEngine([{"panels": []}])
    await StoryboarderV2(engine).fix_v3(v3_storyboard(panels, CLAIMS_WITH_RESULT, brief=glossary), make_cast(), v3_paper(), rounds=1)
    assert "It means: each token scores" in engine.prompts[0] and "PAPER TEXT:" not in engine.prompts[0]


# ----------------------------------------------------------------------------- pictures that teach

SYMBOLS = {"picture_symbols": [{"concept": "a sentence", "symbol": "a row of blank cards"},
                               {"concept": "attention", "symbol": "red yarn tying cards together"}]}


def test_a_scene_no_longer_has_to_teach_with_its_picture():
    # round 23: scenes tell the story literally and diagram boards explain. The rule that every teaching scene show its idea
    # without words asked for metaphor pictures that needed a legend line each, and code-drawn diagrams of the same words
    # were clearer in 22 of 24 panels
    panels = clean_panels()
    for shows, visual in (("", "papi smiles at kumo"), ("every card is tied to every other one at once", "papi smiles at kumo"),
                          ("the blue wooden blocks rise into a tower", "papi stacks blue wooden blocks into a tower"),
                          ("a row of blank cards and some red yarn", "papi holds up a row of blank cards and some red yarn")):
        panels[1].shows, panels[1].visual = shows, visual
        assert not [x for x in problems_of_with(panels, SYMBOLS).get(2, []) if x.startswith("picture:")], shows


def test_the_symbols_a_picture_draws_are_told_apart_by_their_own_words():
    # the render passes a symbol's reference picture only to the pictures that draw that symbol
    from services.storyboarder_v2 import drawn_symbols
    brief = {"picture_symbols": [{"concept": "gate", "symbol": "small round grey dimmer switch"},
                                 {"concept": "hot lead", "symbol": "a glowing red playing card"},
                                 {"concept": "cold lead", "symbol": "a dull grey playing card"}]}
    sb = v3_storyboard(clean_panels(), CLAIMS_WITH_RESULT, brief=brief)
    assert drawn_symbols(sb, "Papi taps one dimmer switch on the counter") == ["small round grey dimmer switch"]
    assert drawn_symbols(sb, "Kumo drops a dull grey playing card into the bin") == ["a dull grey playing card"]
    assert drawn_symbols(sb, "Kumo holds a glowing red playing card and a dull grey playing card") == [
        "a glowing red playing card", "a dull grey playing card"]
    assert drawn_symbols(sb, "Kumo shuffles a deck of playing cards") == []   # no word only one of them has
    assert drawn_symbols(v3_storyboard(clean_panels(), CLAIMS_WITH_RESULT), "a dimmer switch") == []


def problems_of_with(panels, brief):
    return {n: v for n, v in lint_v3(v3_storyboard(panels, CLAIMS_WITH_RESULT, brief=brief), V3_PAPER_TEXT).items()
            if any(x.startswith("picture:") for x in v)}


# ----------------------------------------------------------------------------- boards that back their words

def bar_panel(words: str, baseline: str = "ByteNet", extra=()):
    bars = [{"label": "ByteNet", "value": "23.75", "quote": "ByteNet 23.75"},
            {"label": "ConvS2S Ensemble", "value": "26.36", "quote": "ConvS2S Ensemble 26.36"},
            {"label": "Transformer (big)", "value": "28.4", "quote": "Transformer (big) 28.4", "highlight": True}] + list(extra)
    for b in bars:
        b["baseline"] = b["label"] == baseline
    return v3_panel(3, kind="board", beat="proof", characters=("nezu", "papi"), claims=[3], dialogue=[("nezu", words)],
                    board={"kind": "bar", "title": "BLEU on WMT 2014", "subtitle": "BLEU: match with human translations",
                           "unit": "BLEU", "bars": bars, "source": "Table 2"})


def test_a_win_over_the_field_is_measured_against_the_strongest_rival():
    from services.storyboarder_v2 import _badge_vs_words
    found = _badge_vs_words(bar_panel("It beat both baselines!"))
    assert len(found) == 1 and '"ConvS2S Ensemble" (26.36)' in found[0] and 'gap to "ByteNet"' in found[0]
    assert _badge_vs_words(bar_panel("It beat both baselines!", baseline="ConvS2S Ensemble")) == []
    assert _badge_vs_words(bar_panel("Better than ByteNet.")) == []                    # the words make the badge's comparison
    upper = {"label": "Human reference", "value": "30.1", "quote": "Human reference 30.1"}
    assert _badge_vs_words(bar_panel("The best of all models!", baseline="ConvS2S Ensemble", extra=[upper])) == []   # an upper bound


def test_a_rival_the_words_name_is_the_one_the_badge_measures():
    from services.storyboarder_v2 import _badge_vs_words
    found = _badge_vs_words(bar_panel("Over 2 BLEU above ConvS2S Ensemble."))
    assert len(found) == 1 and found[0].startswith('the words compare with "ConvS2S Ensemble"')


def test_the_bar_the_words_read_out_is_the_one_the_badge_measures():
    # Gated Attention, two rounds: "At 64k tokens: baseline 37.51, gated 66.60" beside a badge of -16.30 against the 4k
    # baseline (82.90); the words, not the labels, say which pair is compared
    from services.storyboarder_v2 import _badge_vs_words
    bars = [{"label": "Baseline 4k", "value": "82.90", "baseline": True}, {"label": "Baseline 64k", "value": "37.51"},
            {"label": "SDPA Gated 4k", "value": "88.13"}, {"label": "SDPA Gated 64k", "value": "66.60", "highlight": True}]
    panel = v3_panel(3, kind="board", beat="proof", claims=[3], dialogue=[("kumo", "At 64k tokens: baseline 37.51, gated 66.60.")],
                     board={"kind": "bar", "title": "RULER score", "unit": "score", "bars": bars})
    found = _badge_vs_words(panel)
    assert len(found) == 1 and 'mark "Baseline 64k" as the baseline' in found[0] and '"Baseline 4k" (82.90)' in found[0]
    bars[0]["baseline"], bars[1]["baseline"] = False, True
    assert _badge_vs_words(panel) == []
    assert not [x for x in _story_board_issues(panel.board) if "strongest earlier system" in x]   # the rival check agrees
    # a size the script compares on purpose: "Mistral 7B beats LLaMA 13B" reads out the marked bar, or only the gap
    bars = [{"label": "LLaMA 7B", "value": "35.1"}, {"label": "LLaMA 13B", "value": "46.9", "baseline": True},
            {"label": "Mistral 7B", "value": "60.1", "highlight": True}]
    for said in ("Mistral 7B gets 60.1, LLaMA 13B only 46.9.", "Mistral 7B beats LLaMA 13B by 13.2 points."):
        panel = v3_panel(3, kind="board", beat="proof", claims=[3], dialogue=[("kumo", said)],
                         board={"kind": "bar", "title": "MMLU", "unit": "score", "bars": bars})
        assert _badge_vs_words(panel) == [], said


def test_the_baseline_in_the_highlights_setting_is_no_weaker_than_itself_in_another():
    # Gated Attention, round 18 dry run: the board check asked for "Baseline 128k" on a chart of RULER at 128k, and the
    # rival check then asked for "Baseline 64k" (37.51 beats 31.65); the same system in an easier setting is no rival
    bars = [{"label": "Baseline 64k", "value": "37.51", "paper_label": "Baseline YaRN 64k"},
            {"label": "Baseline 128k", "value": "31.65", "baseline": True, "paper_label": "Baseline YaRN 128k"},
            {"label": "SDPA gated 128k", "value": "58.82", "highlight": True, "paper_label": "SDPA-Gate YaRN 128k"}]
    chart = {"kind": "bar", "title": "RULER after YaRN extension", "unit": "RULER score", "bars": bars}
    rival = lambda: [x for x in _story_board_issues(chart) if "strongest earlier system" in x]
    assert rival() == []
    bars[0]["label"], bars[0]["paper_label"] = "Mamba 64k", "Mamba YaRN 64k"              # another system is still a rival
    assert rival() and "Mamba 64k" in rival()[0]
    bars[0]["label"], bars[0]["paper_label"] = "Baseline 64k", "Baseline YaRN 64k"
    bars[2]["label"], bars[2]["paper_label"] = "SDPA gated 256k", "SDPA-Gate YaRN 256k"   # the highlight in another setting
    assert rival() and "Baseline 64k" in rival()[0]
    # a size the script compares on purpose keeps the rule: "Mistral 7B" shares no setting with the marked "LLaMA 13B"
    sizes = {"kind": "bar", "title": "MMLU", "unit": "score", "bars": [
        {"label": "LLaMA 34B", "value": "62.6"}, {"label": "LLaMA 13B", "value": "46.9", "baseline": True},
        {"label": "Mistral 7B", "value": "60.1", "highlight": True}]}
    assert [x for x in _story_board_issues(sizes) if "strongest earlier system" in x and "LLaMA 34B" in x]
    # a newer version is another system, though it shares the highlight's size with the baseline ("7B")
    versions = {"kind": "bar", "title": "MMLU", "unit": "score", "bars": [
        {"label": "Llama 3 8B", "value": "66.6"}, {"label": "Llama 2 7B", "value": "45.3", "baseline": True},
        {"label": "Ours 7B", "value": "68.0", "highlight": True}]}
    assert [x for x in _story_board_issues(versions) if "strongest earlier system" in x and "Llama 3 8B" in x]
    # words that claim a win over the field ask for the strongest rival the same way: never the baseline's easier setting
    from services.storyboarder_v2 import _badge_vs_words
    bars[2]["label"], bars[2]["paper_label"] = "SDPA gated 128k", "SDPA-Gate YaRN 128k"
    for words in ("At 128k, gated attention scores the highest.", "It beats both baselines."):
        panel = v3_panel(3, kind="board", beat="proof", claims=[3], dialogue=[("kumo", words)], board=chart)
        assert _badge_vs_words(panel) == [], words
    bars[0]["label"], bars[0]["paper_label"] = "Mamba 64k", "Mamba YaRN 64k"
    assert _badge_vs_words(v3_panel(3, kind="board", beat="proof", claims=[3], dialogue=[("kumo", "It beats both baselines.")],
                                    board=chart))


def test_numbers_in_parentheses_on_a_board_are_explained():
    from services.storyboarder_v2 import _parentheses_unexplained
    table = {"kind": "table", "title": "Hot lead share", "columns": ["Calls", "Change"],
             "rows": [{"cells": ["1-3", "0.234 (0.033)"]}], "source": "Table 2"}
    panel = v3_panel(3, kind="board", beat="proof", board=table, dialogue=[("nezu", "Up by 0.234?")])
    assert _parentheses_unexplained(panel) and '"0.234 (0.033)"' in _parentheses_unexplained(panel)[0]
    panel.board = {**table, "subtitle": "Standard errors in parentheses"}
    assert _parentheses_unexplained(panel) == []
    panel.board = {**table, "subtitle": "Standard deviations after each estimate"}            # another correct note
    assert _parentheses_unexplained(panel) == []
    starred = {**table, "rows": [{"cells": ["1-3", "0.234*** (0.033)"]}]}                   # significance stars
    panel.board = starred
    assert _parentheses_unexplained(panel)
    panel.board = {**table, "rows": [{"cells": ["GPT-2 (2019)", "28.4"]}]}                  # a year beside a name
    assert _parentheses_unexplained(panel) == []



def test_the_badge_rule_settles_on_everyday_words_and_look_alike_labels():
    from services.storyboarder_v2 import _badge_vs_words
    base = [{"label": "Base model", "value": "40.0", "quote": "Base model 40.0", "baseline": True},
            {"label": "DAPO", "value": "47.0", "quote": "DAPO 47.0"},
            {"label": "Ours", "value": "50.0", "quote": "Ours 50.0", "highlight": True}]

    def board(bars, words):
        return v3_panel(3, kind="board", beat="proof", characters=("nezu", "papi"), dialogue=[("nezu", words)],
                        board={"kind": "bar", "title": "Accuracy", "unit": "%", "bars": bars, "source": "Table 2"})

    for words in ("Ours beats the base model on all four benchmarks.", "Top-1 accuracy jumps over the base model.",
                  "At best, the base model guesses.", "Best-of-8 sampling helps."):
        assert _badge_vs_words(board(base, words)) == [], words
    assert _badge_vs_words(board(base, "Ours beats every earlier method."))                  # a real claim on the field
    kl = [{"label": "GRPO", "value": "45.0", "quote": "GRPO 45.0"},
          {"label": "GRPO + KL", "value": "47.0", "quote": "GRPO + KL 47.0", "baseline": True},
          {"label": "Ours", "value": "50.0", "quote": "Ours 50.0", "highlight": True}]
    assert _badge_vs_words(board(kl, "Ours beats GRPO + KL by 3.0 points.")) == []          # not "GRPO" inside "GRPO + KL"
    assert _badge_vs_words(board(kl, "Ours beats GRPO + KL and GRPO.")) == []               # the weaker one named too
    found = _badge_vs_words(board(kl, "Ours beats GRPO."))
    assert found and 'make the words compare with "GRPO + KL"' in found[0]                 # the weaker system named alone
    full = kl + [{"label": "Full precision", "value": "60.0", "quote": "Full precision 60.0"}]
    assert _badge_vs_words(board(full, "Still below Full precision, but ahead of the best rival.")) == []   # an upper bound


def test_a_shared_setting_in_the_table_row_does_not_hide_a_stronger_rival():
    # ledger bars carry the joined table row as paper_label ("BitDistiller | 2/16"): a shared "2/16" is a setting, not a name
    from services.storyboarder_v2 import _badge_vs_words
    bars = [{"label": "EfficientQAT", "value": "51.95", "paper_label": "EfficientQAT | 2/16", "baseline": True},
            {"label": "BitDistiller", "value": "60.42", "paper_label": "BitDistiller | 2/16"},
            {"label": "OnQ", "value": "70.68", "paper_label": "OnQ | 2/16", "highlight": True}]
    panel = v3_panel(3, kind="board", beat="proof", characters=("nezu",), dialogue=[("nezu", "It beats all baselines.")],
                     board={"kind": "bar", "title": "9-benchmark average", "unit": "%", "bars": bars, "source": "Table 1"})
    found = _badge_vs_words(panel)
    assert found and '"BitDistiller" (60.42)' in found[0]


def test_a_shows_line_the_illustrator_cannot_draw_is_flagged():
    panels = clean_panels()
    panels[1].visual = "papi ties red yarn between every pair of blank cards on the desk while kumo watches"
    panels[1].shows = "two students hold a chat window with the LLM's answer"
    found = problems_of_with(panels, SYMBOLS).get(2, [])
    assert any("asks the illustrator for something it cannot draw" in x and 'set \"shows\" to \"\"' in x for x in found)
    panels[1].shows = "papi's red sweater and her waving paw"
    panels[1].visual = "papi, in her red sweater, smiles and waves at kumo"   # round 23: no picture has to use the symbols
    assert not any("picture symbols" in x for x in problems_of_with(panels, SYMBOLS).get(2, []))


async def test_a_scene_a_rewrite_points_at_a_board_gets_one_more_bounded_picture_rewrite():
    from services.storyboarder_v2 import StoryboarderV2
    from .fakes import FakeEngine
    from .test_story_v3 import make_cast, v3_paper
    panels = clean_panels()
    panels[1].dialogue[0].text = "They used IV regression to find the cause."   # a term problem sends panel 2 to a rewrite
    paper = v3_paper()
    paper.body += "\nWe compare with instrumental variable (IV) estimates."
    rewrite = {"n": 2, "kind": "scene", "beat": "idea", "characters": ["papi", "kumo"], "scene": "A study.",
               "visual": "papi points at the numbers on the whiteboard", "claims": [1, 2],
               "dialogue": [{"speaker": "papi", "text": "IV regression: it separates cause from coincidence."}]}
    engine = FakeEngine([{"panels": [rewrite]}, {"panels": [{"n": 2, "scene": "A study.", "visual": "papi waves kumo over to the desk",
                                                              "shows": ""}]}])
    sb = v3_storyboard(panels, CLAIMS_WITH_RESULT)
    await StoryboarderV2(engine).fix_v3(sb, make_cast(), paper, rounds=1)
    assert len(engine.calls) == 2 and "Fix the pictures of these manga panels" in engine.prompts[1]
    assert sb.panels[1].visual == "papi waves kumo over to the desk" and sb.panels[1].dialogue[0].text.startswith("IV regression")


async def test_a_rewrite_that_leaves_shows_out_keeps_the_panels_idea(monkeypatch):
    from services import storyboarder_v2
    from services.storyboarder_v2 import StoryboarderV2
    from .fakes import FakeEngine
    from .test_story_v3 import make_cast, v3_paper
    rewrite = {"n": 2, "kind": "scene", "beat": "idea", "characters": ["papi", "kumo"], "scene": "A study.",
               "visual": "papi ties red yarn between two blank cards", "claims": [1, 2],
               "dialogue": [{"speaker": "papi", "text": "IV regression: it separates cause from coincidence."}]}
    paper = v3_paper()
    paper.body += "\nWe compare with instrumental variable (IV) estimates."
    for keep in (True, False):   # scenes with a picture idea (rounds 3-22) keep it; round 23's scenes have none
        monkeypatch.setattr(storyboarder_v2, "SCENE_SHOWS", keep)
        panels = clean_panels()
        panels[1].dialogue[0].text = "They used IV regression to find the cause."   # a term problem sends panel 2 to a rewrite
        sb = v3_storyboard(panels, CLAIMS_WITH_RESULT)
        before = sb.panels[1].shows
        await StoryboarderV2(FakeEngine([{"panels": [dict(rewrite)]}])).fix_v3(sb, make_cast(), paper, rounds=1)
        assert sb.panels[1].visual == "papi ties red yarn between two blank cards"
        assert sb.panels[1].shows == (before if keep else "") and (before or not keep)


def test_picture_symbols_that_carry_writing_or_figures_never_reach_the_story():
    # Right On Q: the picture rule demanded a "pre-printed paper scroll" that the writing-prop rule banned, in every round
    from services.storyboarder_v2 import drawable_symbols
    symbols = [{"concept": "teacher", "symbol": "Thick solid gold brick"}, {"concept": "reference", "symbol": "Pre-printed paper scroll"},
               {"concept": "trajectory", "symbol": "Hand-drawn scribbled line on blank paper"}, {"concept": "agent", "symbol": "a small robot"},
               {"concept": "result", "symbol": "a bar chart"}, "not a dict", {"concept": "loop", "symbol": "Closed loop of string"}]
    assert [s["symbol"] for s in drawable_symbols(symbols)] == ["Thick solid gold brick", "Closed loop of string"]


def test_significance_stars_survive_the_markdown_cleanup():
    # "* p<0.10, ** p<0.05, *** p<0.01" was drawn as "* p<0.10, p<0.05, * p<0.01" (AI deployment and service, 2026-10-28)
    from services.storyboarder_v2 import _strip_marks
    assert _strip_marks({"subtitle": "* p<0.10, ** p<0.05, *** p<0.01", "rows": [{"cells": ["1-3", "0.234*** (0.033)"]}]}) == {
        "subtitle": "* p<0.10, ** p<0.05, *** p<0.01", "rows": [{"cells": ["1-3", "0.234*** (0.033)"]}]}
    assert _strip_marks("**28.4** BLEU, $x$") == "28.4 BLEU, x"


def test_the_first_panel_that_draws_a_symbol_says_what_it_stands_for():
    from services.storyboarder_v2 import _symbol_keys
    brief = {"picture_symbols": [{"concept": "lost work", "symbol": "a plastic bucket with a small hole leaking water"}]}
    panels = clean_panels()
    panels[1].visual = "papi holds a plastic bucket with a small hole; water leaks out"
    found = _symbol_keys(v3_storyboard(panels, CLAIMS_WITH_RESULT, brief=brief))
    assert list(found) == [2] and found[2][0].startswith('symbol: this panel draws "a plastic bucket')
    assert '("This plastic bucket with a small hole leaking water is lost work, and ...")' in found[2][0]   # no "the a ..."
    panels[1].narration = "This leaking bucket is the driver's lost work."
    assert _symbol_keys(v3_storyboard(panels, CLAIMS_WITH_RESULT, brief=brief)) == {}
    panels[2].visual = "the leaking plastic bucket again, now bigger"                   # only the first drawing is keyed
    assert _symbol_keys(v3_storyboard(panels, CLAIMS_WITH_RESULT, brief=brief)) == {}


def test_an_idea_or_how_panel_says_why():
    from services.storyboarder_v2 import _missing_why
    panels = clean_panels()
    panels[1].dialogue[0].text = "Only attention, no recurrence, no convolutions."
    assert list(_missing_why(v3_storyboard(panels, CLAIMS_WITH_RESULT))) == [2]
    panels[1].dialogue[0].text = "Only attention, because every word must see every other word at once."
    assert _missing_why(v3_storyboard(panels, CLAIMS_WITH_RESULT)) == {}
    panels[0].dialogue[0].text = "No reason given here."                                 # the hook is not asked
    assert _missing_why(v3_storyboard(panels, CLAIMS_WITH_RESULT)) == {}


def test_the_recap_brings_back_the_claims_boards_lines_in_order():
    from services.storyboarder_v2 import _recap_mirrors_claims
    claims = v3_panel(2, kind="board", beat="claims", characters=("nezu",), dialogue=[("nezu", "Prove each line.")],
                      board={"kind": "list", "title": "What the paper claims", "open": True,
                             "items": ["Attention alone translates well", "It trains much faster"]})
    panels = clean_panels()
    panels = [panels[0], claims] + panels[1:]
    for i, p in enumerate(panels):
        p.n = i + 1
    recap = panels[-1]
    recap.board = {"kind": "list", "title": "What the paper showed", "items": ["28.4 BLEU on WMT 2014", "Attention only"]}
    found = _recap_mirrors_claims(v3_storyboard(panels, CLAIMS_WITH_RESULT))
    assert list(found) == [recap.n] and "Attention alone translates well" in found[recap.n][0]
    recap.board["items"] = ["Attention alone translates well: 28.4 BLEU", "It trains much faster: 3.5 days"]
    assert _recap_mirrors_claims(v3_storyboard(panels, CLAIMS_WITH_RESULT)) == {}


# ----------------------------------------------------------------------------- third round (2026-10-28, after the v2 books)

def test_a_board_with_the_papers_significance_stars_is_valid_and_explains_them():
    # econ_ai_service's cited cells "0.234*** (0.033)" were flagged as markdown in every fix round and never passed
    from services.boards import board_problems
    from services.storyboarder_v2 import MD_BOLD_RE, _parentheses_unexplained
    table = {"kind": "table", "title": "Change in hot-lead probability", "subtitle": "standard errors in parentheses",
             "columns": ["Calls", "Change"], "rows": [{"cells": ["1-3", "0.234*** (0.033)"]}, {"cells": ["4-6", "0.157** (0.033)"]}]}
    assert not any("LaTeX or markdown" in x for x in board_problems(table))
    assert not any("LaTeX or markdown" in x for x in board_problems({**table, "note": "* p<0.10, ** p<0.05, *** p<0.01"}))
    assert any("LaTeX or markdown" in x for x in board_problems({**table, "title": "The **best** call"}))
    panel = v3_panel(8, kind="board", beat="proof", board=table, dialogue=[("nezu", "Up at the first calls!")], characters=("nezu",))
    assert [x for x in _parentheses_unexplained(panel) if x.startswith("the board shows significance stars")]
    panel.board["note"] = "*** p<0.01: very unlikely to be chance"
    assert _parentheses_unexplained(panel) == []
    assert MD_BOLD_RE.sub(r"\1", "0.234*** (0.033) and 0.112** (0.041)") == "0.234*** (0.033) and 0.112** (0.041)"
    assert MD_BOLD_RE.sub(r"\1", "the **best** model") == "the best model"


def test_the_bets_settlement_needs_no_picture_idea():
    # every book of 2026-10-28 settled the bet in a scene of its own, and the picture rule forced an idea onto it
    from services.storyboarder_v2 import _finish_v3
    panels = clean_panels()
    bet = v3_panel(5, beat="limits", dialogue=[("nezu", "Fine, you won the bet. The tart is yours.")], characters=("nezu", "kumo"),
                   shows="")
    assert 5 not in problems_of_with(panels + [bet], SYMBOLS)
    bet.visual, bet.shows = "nezu slides half a berry tart across the table to kumo", "half a berry tart slides across the table"
    assert 5 not in problems_of_with(panels + [bet], SYMBOLS)
    sb = v3_storyboard(panels + [bet], CLAIMS_WITH_RESULT, brief=SYMBOLS)
    sb.panels[2].shows = "a leftover idea on a board"
    _finish_v3(sb)
    assert sb.panels[4].shows == "" and sb.panels[2].shows == ""          # no idea check, no extra try for either
    assert sb.panels[1].shows                                               # a teaching scene keeps its idea
    teach = v3_panel(5, beat="limits", dialogue=[("nezu", "It was only tested on English.")], characters=("nezu",), shows="")
    assert 5 not in problems_of_with(panels + [teach], SYMBOLS)   # round 23: no scene has to teach with its picture


def test_symbols_that_differ_in_color_are_told_apart():
    from services.storyboarder_v2 import _picture_problems, _symbol_keys
    brief = {"picture_symbols": [{"concept": "hot lead", "symbol": "a glowing red playing card"},
                                 {"concept": "cold lead", "symbol": "a dull grey playing card"}]}
    panels = clean_panels()
    panels[0].visual = "kumo looks at a row of blank cards"                 # (red yarn and cards read as a red card)
    panels[1].visual = "papi holds up one dull grey playing card"
    panels[1].dialogue[0].text = "This grey card is a cold lead, so nobody calls it first."
    assert _symbol_keys(v3_storyboard(panels, CLAIMS_WITH_RESULT, brief=brief)) == {}   # the red card is not drawn here
    bricks = {"picture_symbols": [{"concept": "teacher model", "symbol": "a thick solid gold brick"},
                                  {"concept": "student model", "symbol": "a thin hollow silver brick"}]}
    panels[1].visual, panels[1].shows = "papi sets the thick gold brick beside the thin silver brick", "the thick gold brick beside the thin silver brick"
    assert 2 not in _picture_problems(v3_storyboard(panels, CLAIMS_WITH_RESULT, brief=bricks))


def test_why_needs_a_real_connective():
    from services.storyboarder_v2 import WHY_RE
    for text, reason in [("This is why the student drifts.", True), ("The teacher grades it, so the student learns.", True),
                         ("It writes its own answers so it can practice.", True), ("Thanks to the lock, the answer stays right.", True),
                         ("That’s why it fails.", True), ("That's so simple!", False), ("So, what is next?", False),
                         ("a so-called agent", False), ("Instead, they compare.", False)]:
        assert bool(WHY_RE.search(text)) is reason, text



def test_a_recap_line_without_numbers_gets_its_tick_from_a_chart_bound_to_its_claim():
    # econ_delivery's recap ticked "e-commerce fueled two decades of growth", which no chart drew; round 19 ticked "Gate
    # adds non-linearity" beside a metaphor. Without the check model's answer, a line without numbers gets its tick when
    # a chart or table is bound to the claim it restates
    from services.storyboarder_v2 import _honest_recap
    panels = clean_panels()
    panels[3].board = {"kind": "list", "title": "What the paper showed", "open": False,
                       "items": ["The Transformer improves over the existing best translation results",   # claim 3: the table
                                 "A simple architecture built only on attention",                           # claim 1: said, not drawn
                                 "E-commerce fueled the growth of delivery firms"]}
    sb = v3_storyboard(panels, CLAIMS_WITH_RESULT)
    _honest_recap(sb)
    assert sb.panels[3].board["checked"] == [0]
    sb.panels[3].board["items"] = ["改进了现有最佳翻译结果", "只用注意力的简单结构", "电商推动了快递公司的增长"]
    _honest_recap(sb, retitle=False)     # a translated recap the rule cannot read keeps the English marks
    assert sb.panels[3].board["checked"] == [0]


def test_a_term_the_abstract_names_must_appear_in_the_story():
    # econ_ai_service's script never named its regression-discontinuity-in-time design, and no reader could
    from services.storyboarder_v2 import _abstract_terms_missing
    brief = {"abstract": "We adopt a regression-discontinuity-in-time design.",
             "glossary": [{"term": "Regression-discontinuity-in-time", "plain": "compares outcomes right before and after an event"},
                          {"term": "Call duration", "plain": "how long a call lasts"}]}
    panels = clean_panels()
    found = [x for v in _abstract_terms_missing(v3_storyboard(panels, CLAIMS_WITH_RESULT, brief=brief)).values() for x in v]
    assert len(found) == 1 and "Regression discontinuity in time" in found[0] and "It means: compares outcomes" in found[0]
    panels[1].narration = "They use a regression discontinuity in time: before and after the launch."
    assert _abstract_terms_missing(v3_storyboard(panels, CLAIMS_WITH_RESULT, brief=brief)) == {}


def test_a_term_named_only_on_the_recap_is_not_named():
    # the recap and the claims board list names, they explain nothing; and a paper without an "Abstract" heading
    # (Flexibility Trap) is held to its claims' quotes
    from services.storyboarder_v2 import _abstract_terms_missing
    brief = {"abstract": "This trajectory divergence grows as responses lengthen.",
             "glossary": [{"term": "trajectory divergence", "plain": "the student's own answers drift away from the training paths"}]}
    panels = clean_panels()
    panels[3].board = {**panels[3].board, "items": panels[3].board["items"] + ["Trajectory divergence grows with length"]}
    assert any("trajectory divergence" in x for v in _abstract_terms_missing(v3_storyboard(panels, CLAIMS_WITH_RESULT, brief=brief)).values() for x in v)
    claims = CLAIMS_WITH_RESULT + [{"id": 4, "text": "The drift grows with length.", "quote": "this trajectory divergence grows as responses lengthen"}]
    found = _abstract_terms_missing(v3_storyboard(panels, claims, brief={**brief, "abstract": ""}))
    assert any("trajectory divergence" in x for v in found.values() for x in v)



def test_the_recap_marks_follow_the_check_models_answer():
    # which chart shows which line is the check model's call (recap_evidence): "Found a strong Artificial Hivemind effect"
    # shares no word with its two charts. Only panels that draw a chart or table count, and a line the answer does not
    # cover is judged by the rule
    from services.storyboarder_v2 import _honest_recap
    panels = clean_panels()
    panels[3].board = {"kind": "list", "title": "What the paper showed", "open": False,
                       "items": ["Found a strong effect", "28.4 BLEU on WMT 2014 English-German", "Attention only",
                                 "41.0 BLEU on English-French"]}
    sb = v3_storyboard(panels, CLAIMS_WITH_RESULT)
    sb.brief["recap_evidence"] = [{"line": "Found a strong effect", "panels": [3], "shown": True},   # a chart its words never name
                                  {"line": "28.4 BLEU on WMT 2014 English-German", "panels": [], "shown": False},  # judged not shown
                                  {"line": "Attention only", "panels": [2], "shown": False}]    # panel 2 draws no chart
    _honest_recap(sb)
    assert sb.panels[3].board["checked"] == [0, 3]   # line 3 was not judged: the table draws its 41.0
    for p in sb.panels:                               # panels numbered afresh after the judgment: the verdicts stand
        p.n += 10
    _honest_recap(sb)
    assert sb.panels[3].board["checked"] == [0, 3]


def test_a_recap_line_reworded_after_the_check_keeps_its_verdict():
    # the translation's fact-check, backported, reworded "Existing QAT trains only on pre-written fixed answers" into
    # "Existing QAT typically trains only on..." after the check model had judged it: a relayout keeps the verdict of the
    # line it replaced (same place, same numbers); a changed number is a new claim, judged by the rule
    from services.storyboarder_v2 import _honest_recap
    panels = clean_panels()
    panels[3].board = {"kind": "list", "title": "What the paper showed", "open": False,
                       "items": ["Found a strongly marked effect", "28.4 BLEU on WMT 2014 English-German", "41.0 BLEU on English-French"]}
    sb = v3_storyboard(panels, CLAIMS_WITH_RESULT)
    sb.brief["recap_evidence"] = [{"line": "Found a strong effect", "panels": [3], "shown": True},
                                  {"line": "28.4 BLEU on WMT 2014 English-German", "panels": [], "shown": False},
                                  {"line": "41.1 BLEU on English-French", "panels": [], "shown": False}]
    _honest_recap(sb)
    assert sb.panels[3].board["checked"] == [0, 2]   # 0 keeps its tick; 2 says 41.0 now, which the table draws
    sb.brief["recap_evidence"].append({"line": "Attention only", "panels": [], "shown": False})
    _honest_recap(sb)   # four answers for three lines: nothing to pair them by, every unjudged line goes by the rule
    assert sb.panels[3].board["checked"] == [2]


def test_an_ending_that_vouches_for_every_claim_is_flagged():
    # "All the tested claims held up. You won the tart." beside a recap that ticked two of six lines (15 of 176 books)
    panels = clean_panels()
    panels[3].dialogue = [Bubble("nezu", "Fine. All the tested claims held up.")]
    found = problems_of(panels).get(4, [])
    assert any(x.startswith('"All the tested claims held up" vouches for every claim') for x in found)
    for vouch in ("All of these held up.", "Fine, all of them checked out.", "All the findings were confirmed."):
        panels[3].dialogue = [Bubble("nezu", vouch)]   # round 23: beside a recap ticking one line of six
        assert any("vouches for every claim" in x for x in problems_of(panels).get(4, [])), vouch
    for fine in ("All of these models use the same data, and the gains held up.", "All of them use attention; one held up.",
                 "Not all of these held up.", "Did all of these hold up?"):
        panels[3].dialogue = [Bubble("nezu", fine)]
        assert not any("vouches for every claim" in x for x in problems_of(panels).get(4, [])), fine
    for fine in ("Fine. The charts back the main result.", "We tested all claims on five benchmarks.", "Every result is in Table 2.",
                 "After all, the results check out.", "In all five tests, the results confirmed the trend.", "Not all claims held up.",
                 "Did all the claims hold up?", "All claims, not just the verified ones, need a chart.",
                 "All these claims will check out, every number."):   # the bet, before any chart
        panels[3].dialogue = [Bubble("nezu", fine)]
        assert not any("vouches for every claim" in x for x in problems_of(panels).get(4, [])), fine


def test_a_web_address_is_never_lettered():
    # 1000 Layer RL's project address in a bubble: a reader cannot click a picture (its Chinese panel also kept English)
    panels = clean_panels()
    panels[1].dialogue = [Bubble("papi", "Our code: https://wang-kevin3290.github.io/scaling-crl/")]
    found = problems_of(panels).get(2, [])
    assert any(x.startswith('a web address in the words ("https://wang-kevin3290.github.io/scaling-crl/")') for x in found)
    panels[1].dialogue = [Bubble("papi", "The code is public on github.com.")]
    assert any(x.startswith('a web address in the words ("github.com")') for x in problems_of(panels).get(2, []))
    for fine in ("The code is public, e.g. for GPT-4.5 users.", "Booking.com ran 1,000 tests on its users.", "Character.AI users chat for hours."):
        panels[1].dialogue = [Bubble("papi", fine)]   # a company's name is no address
        assert not any("web address" in x for x in problems_of(panels).get(2, [])), fine


def test_fractions_under_a_percentage_point_label_are_flagged():
    from services.storyboarder_v2 import _fractions_as_points
    table = {"kind": "table", "title": "Change in hot-lead probability", "subtitle": "Change after AI, percentage points",
             "columns": ["Calls", "Change"], "rows": [{"cells": ["1-3", "0.234 (0.033)"]}, {"cells": ["4-6", "0.157 (0.033)"]}]}
    panel = v3_panel(8, kind="board", beat="proof", board=table, characters=("nezu",), dialogue=[("nezu", "Up 23 percentage points!")])
    found = _fractions_as_points(panel)
    assert len(found) == 1 and "0.234 is +23.4 percentage points" in found[0]   # a change: a signed reading
    panel.board["subtitle"] = "Change in probability"
    assert _fractions_as_points(panel) == []
    panel.board["subtitle"], panel.dialogue[0].text = "Change, percent", "Up 0.234 percent!"   # read as they are
    assert _fractions_as_points(panel) == []


def test_a_teaching_scene_leaves_room_for_its_picture():
    # the teaching pictures of 2026-10-28 with 38-46 words had their key object lettered over
    panels = clean_panels()
    panels[1].dialogue = [Bubble("papi", "Only attention, so no recurrence and no convolutions."), Bubble("kumo", "Every word at once?"),
                          Bubble("papi", "Yes.")]
    found = problems_of(panels).get(2, [])
    assert "3 speech bubbles: at most 2 on a teaching scene (its picture needs the room)" in found
    panels[1].dialogue = [Bubble("papi", "Only attention, so no recurrence and no convolutions anywhere in the whole model.")]
    panels[1].narration = "Every word looks at every other word at once, so the model sees the whole sentence in one step."
    assert any(x.startswith("32 words") for x in problems_of(panels).get(2, [])) is False
    panels[1].narration += " Fast."
    assert any(x.startswith("33 words of bubbles and narration in total: at most 32 on a teaching scene") for x in problems_of(panels)[2])


def test_symbol_keys_ask_at_most_two_of_one_panel():
    from services.storyboarder_v2 import _symbol_keys
    brief = {"picture_symbols": [{"concept": "the model", "symbol": "a gold brick"},
                                 {"concept": "a repeated answer", "symbol": "a closed loop of string"},
                                 {"concept": "a cut answer", "symbol": "a rope cut in half"}]}
    panels = clean_panels()
    panels[0].visual = "kumo stares at a gold brick, a closed loop of string and a rope cut in half"
    panels[1].visual = "papi lifts the rope cut in half"                     # the third symbol's next drawing gets its key
    found = _symbol_keys(v3_storyboard(panels, CLAIMS_WITH_RESULT, brief=brief))
    assert len(found.get(1, [])) == 2 and len(found.get(2, [])) == 1 and "rope cut in half" in found[2][0]


# ----------------------------------------------------------------------------- round 3 review fixes

def test_a_hook_shows_is_checked_only_for_what_no_illustrator_can_draw():
    # 5 of 10 saved hooks with a "shows" were published "idea unclear"; since round 23 no picture is judged by an idea
    panels = clean_panels()
    panels[0].shows = "most incoming leads are low-quality cold leads"
    assert 1 not in problems_of_with(panels, SYMBOLS)
    panels[0].shows = "two students hold a chat window with the LLM's answer"
    assert any('set \"shows\" to \"\"' in x for x in problems_of_with(panels, SYMBOLS).get(1, []))
    panels[0].shows = ""
    assert 1 not in problems_of_with(panels, SYMBOLS)


def test_a_question_mark_in_the_picture_is_writing():
    # 029d p13: "a large question mark is drawn on the table" failed the art check on all 4 attempts
    panels = clean_panels()
    panels[1].visual = "Papi frowns; a large question mark is drawn on the table next to the cards and red yarn."
    assert any(x.startswith("the picture has props that carry writing (question mark)") for x in problems_of(panels)[2])


def test_words_the_picture_is_asked_to_show_are_flagged_and_left_out_of_the_art_prompt():
    from services.panel_renderer import _unquoted
    panels = clean_panels()
    panels[0].visual = "Kumo holds a phone that displays DELIVERY FAILED: Customer unavailable, beside a tall stack of paper."
    assert any(x.startswith('the picture asks the illustrator to draw words ("DELIVERY FAILED")') for x in problems_of(panels)[1])
    assert "DELIVERY" not in _unquoted(panels[0].visual)
    panels[0].visual = "Papi shows AI sorting the cards."   # an acronym is no sign
    assert not any("draw words" in x for x in problems_of(panels).get(1, []))


def test_a_model_named_after_a_role_is_no_person():
    from services.storyboarder_v2 import _mentions_people
    assert not _mentions_people("a gold brick, the full-precision teacher model, beside the silver student model")
    assert not _mentions_people("the student network copies the teacher policy")
    assert _mentions_people("a worker holds one good lead card")


async def test_a_translation_never_refits_the_art_directions():
    from services.storyboarder_v2 import StoryboarderV2
    from .fakes import FakeEngine
    panels = clean_panels()
    panels[1].shows = "a worker holds one good lead card"
    story = StoryboarderV2(FakeEngine([{"items": []}]))
    sb = v3_storyboard(panels, CLAIMS_WITH_RESULT)
    assert any(x["id"] == "2.s" for x in story._violations_v3(sb, "en-US"))
    assert not any(x["id"].endswith((".s", ".v")) for x in story._violations_v3(sb, "zh-CN"))


def test_a_claim_named_but_not_taught_is_flagged_with_its_missing_words():
    # LongZero's "training data that is long, diverse, challenging and verifiable" became "it needs good data"
    claims = CLAIMS_WITH_RESULT[:3] + [{"id": 3, "text": "RL needs training data that is long, diverse, challenging and verifiable.",
                                        "quote": "requires long, diverse, challenging and verifiable data"}]
    panels = clean_panels()
    panels[2].claims = [3]
    panels[2].kind, panels[2].board = "scene", None
    panels[2].dialogue = [Bubble("nezu", "It needs good data.")]
    brief = {"glossary": [{"term": "Reinforcement learning (RL)", "plain": "learning from rewards"}]}
    found = [x for x in lint_v3(v3_storyboard(panels, claims, brief=brief), V3_PAPER_TEXT).get(3, []) if x.startswith("claim 3")]
    assert len(found) == 1 and "named but not taught" in found[0] and "challenging, diverse" in found[0]
    assert "cut the words that repeat or decorate" in found[0] and "split" not in found[0]   # round 23: no panel is split
    panels[2].dialogue = [Bubble("nezu", "Reinforcement learning needs data that is long, diverse, challenging and verifiable.")]
    assert not any(x.startswith("claim 3") for x in lint_v3(v3_storyboard(panels, claims, brief=brief), V3_PAPER_TEXT).get(3, []))
    panels[2].dialogue = [Bubble("nezu", "RL needs long, diverse, challenging, verifiable data.")]   # the glossary's short name
    assert not any(x.startswith("claim 3") for x in lint_v3(v3_storyboard(panels, claims, brief=brief), V3_PAPER_TEXT).get(3, []))


def _points_board(subtitle="", note="", values=("0.234", "0.157"), title="Change in hot lead probability"):
    board = {"kind": "bar", "title": title, "subtitle": subtitle, "note": note,
             "bars": [{"label": f"Calls {i}", "value": v, "highlight": i == 0} for i, v in enumerate(values)]}
    return v3_panel(3, kind="board", beat="proof", board=board, dialogue=[("kumo", "First calls: +23 percentage points hot leads!")],
                    characters=("kumo", "papi"))


def test_the_unit_rule_reads_the_label_not_the_reading_it_asked_for():
    # econ_ai_service wrote "0.234 = +23.4 percentage points" as told, and the rule fired again every round
    from services.storyboarder_v2 import _fractions_as_points
    assert _fractions_as_points(_points_board(subtitle="Change after AI, percentage points"))
    assert _fractions_as_points(_points_board(subtitle="Change after AI launch; 0.234 = +23.4 percentage points")) == []
    assert _fractions_as_points(_points_board(subtitle="Change in probability", note="*** significant at the 1% level")) == []
    assert _fractions_as_points(_points_board(subtitle="percent", values=("0.5%", "0.25%"))) == []        # a percent already
    assert _fractions_as_points(_points_board(title="Log duration change", subtitle="-0.145 ≈ −13% time; %")) == []   # log points


def test_the_words_may_round_a_number_the_board_prints():
    from services.storyboarder_v2 import _words_vs_chart
    assert _words_vs_chart(_points_board(subtitle="Change after AI launch; 0.234 = +23.4 percentage points")) == []
    found = _words_vs_chart(_points_board(subtitle="Change after AI launch"))
    assert len(found) == 1 and "the paper's own reading of one in the subtitle" in found[0] and "0.234 = 23.4 percentage points" in found[0]
    # a log change is no fraction of 100: -0.145 is about 13% shorter, never "14.5 percentage points"
    logged = _points_board(subtitle="Change in log call duration")
    assert not any("percentage points" in x for x in _words_vs_chart(logged))


def test_a_one_row_trend_needs_bins_of_one_quantity():
    from services.storyboarder_v2 import _ledger_bar_issues

    class Cell:
        def __init__(self, col, column):
            self.table, self.row, self.col, self.column, self.table_label, self.id = "T3", 1, col, column, "Table 3", f"T3.r1.c{col}"

    class Ledger:
        def __init__(self, columns):
            self.cells = {f"T3.r1.c{i}": Cell(i, c) for i, c in enumerate(columns, start=1)}

        def cell(self, cid):
            return self.cells.get(cid)

    def issues(columns):
        bars = [{"label": c, "value": "0.1", "cell": f"T3.r1.c{i}"} for i, c in enumerate(columns, start=1)]
        bars[0]["highlight"] = True
        return _ledger_bar_issues({"kind": "bar"}, bars, [bars[0]], Ledger(columns))

    assert issues(["1–3", "4–6", "7–9"]) == []
    assert issues(["Calls 1–3", "Calls 4–6"]) == []
    assert issues(["(1) › Average conversion", "(2) › Profile slope", "(3) › Early–late contrast"])   # three estimands
    assert issues(["GSM8K", "MATH-500", "AIME 24"])


def test_a_significance_legend_keeps_its_stars():
    from services.storyboarder_v2 import _strip_marks
    for legend in ("*p<0.10, **p<0.05, ***p<0.01", "p<0.10 (*), p<0.05 (**), p<0.01 (***)", "0.234*** (0.033)"):
        assert _strip_marks(legend) == legend
    assert _strip_marks("**28.4** BLEU") == "28.4 BLEU" and _strip_marks("the **best** model") == "the best model"


def test_the_stars_legend_must_be_one_the_reader_sees():
    from services.storyboarder_v2 import _parentheses_unexplained
    table = {"kind": "table", "title": "Effect", "subtitle": "Change in the chance a call is a hot lead; standard errors in parentheses",
             "columns": ["Calls", "Effect"], "rows": [{"cells": ["1–3", "0.234*** (0.033)"]}]}
    panel = v3_panel(3, kind="board", beat="proof", board=table, dialogue=[("kumo", "Look at that.")], characters=("kumo", "papi"))
    assert any("significance stars" in x for x in _parentheses_unexplained(panel))   # "the chance a call..." names the outcome
    table["note"] = "*** p<0.01: very unlikely to be chance"
    assert not any("significance stars" in x for x in _parentheses_unexplained(panel))
    bar = {"kind": "bar", "title": "Effect", "note": "*** p<0.01", "bars": [{"label": "Calls 1–3", "value": "0.005^***", "highlight": True},
                                                                          {"label": "Calls 4–6", "value": "0.001"}]}
    panel.board = bar
    assert any("significance stars" in x for x in _parentheses_unexplained(panel))   # a bar chart never draws its note
    bar["subtitle"] = "*** p<0.01: very unlikely to be chance"
    assert not any("significance stars" in x for x in _parentheses_unexplained(panel))


def test_a_glossary_term_is_no_verdict_word():
    panels = clean_panels()
    panels[1].dialogue = [Bubble("papi", "The Transformer, a simple architecture: only attention, so no recurrence and no convolutions."),
                          Bubble("kumo", "Like a perfect Bayesian equilibrium?")]
    brief = {"glossary": [{"term": "Perfect Bayesian equilibrium", "plain": "a stable outcome of a game with hidden information"}]}
    found = lint_v3(v3_storyboard(panels, CLAIMS_WITH_RESULT, brief=brief), V3_PAPER_TEXT).get(2, [])
    assert not any(x.startswith("verdict words") for x in found)
    panels[1].dialogue[1].text = "That is a perfect result!"
    found = lint_v3(v3_storyboard(panels, CLAIMS_WITH_RESULT, brief=brief), V3_PAPER_TEXT).get(2, [])
    assert any(x.startswith("verdict words perfect") for x in found)


def test_the_recap_ticks_evidence_at_the_lines_precision_and_never_background_numbers():
    from services.storyboarder_v2 import _honest_recap
    panels = clean_panels()
    panels[3].board = {"kind": "list", "title": "What the paper showed", "open": False,
                       "items": ["28 BLEU on English-German", "Two decades of growth", "Recurrent models took 3.5 days"]}
    panels[0].narration = "Recurrent models took 3.5 days to train."     # background on the hook: no evidence
    sb = v3_storyboard(panels, CLAIMS_WITH_RESULT)
    _honest_recap(sb)
    # 28 is the table's 28.4 at the line's precision; "two" is a word, not the "2.0" of panel 3; 3.5 days was background
    assert sb.panels[3].board["checked"] == [0]


def test_a_translated_recap_copies_the_english_marks_and_drops_a_certifying_title():
    from services.storyboarder_v2 import copy_recap_marks
    english, chinese = clean_panels(), clean_panels()
    english[3].board = {"kind": "list", "title": "What the paper showed", "items": ["a", "b", "c"], "checked": [0, 2]}
    chinese[3].board = {"kind": "list", "title": "已验证的结论", "subtitle": "全部经过核实", "items": ["甲", "乙", "丙"]}
    zh = v3_storyboard(chinese, CLAIMS_WITH_RESULT)
    copy_recap_marks(v3_storyboard(english, CLAIMS_WITH_RESULT), zh, "zh-CN")
    assert zh.panels[3].board["checked"] == [0, 2] and zh.panels[3].board["title"] == "论文展示了什么" and "subtitle" not in zh.panels[3].board
    english[3].board = {"kind": "list", "title": "What the paper showed", "items": ["a", "b", "c"]}   # every line ticked
    copy_recap_marks(v3_storyboard(english, CLAIMS_WITH_RESULT), zh, "zh-CN")
    assert "checked" not in zh.panels[3].board


async def test_a_fix_round_can_split_a_full_panel_in_two():
    # the lint's advice to give the rest a neighbouring panel was thrown away: a full teaching scene lost its details
    from services.storyboarder_v2 import StoryboarderV2
    from .fakes import FakeEngine
    from .test_story_v3 import make_cast, v3_paper
    panels = clean_panels()
    idea = "The Transformer, a simple architecture: only attention, so no recurrence and no convolutions."
    panels[1].dialogue = [Bubble("papi", idea), Bubble("kumo", "Every word at once?"), Bubble("papi", "Yes, so every word sees the rest.")]
    first = {"n": 2, "kind": "scene", "beat": "idea", "dialogue": [{"speaker": "papi", "text": idea}], "claims": [1, 2],
             "visual": panels[1].visual, "shows": panels[1].shows}
    second = {"after": 2, "n": 2, "kind": "scene", "beat": "payoff", "characters": ["kumo", "papi"], "visual": panels[1].visual,
              "shows": panels[1].shows, "dialogue": [{"speaker": "kumo", "text": "Every word at once?"},
                                                     {"speaker": "papi", "text": "Yes, so every word sees the rest."}]}
    stray = {"after": 3, "kind": "scene", "dialogue": [{"speaker": "kumo", "text": "A panel nobody asked for."}]}
    sb = v3_storyboard(panels, CLAIMS_WITH_RESULT)
    await StoryboarderV2(FakeEngine([{"panels": [second, first, stray]}])).fix_v3(sb, make_cast(), v3_paper(), rounds=1, extra=chart_ask(2))
    assert [p.n for p in sb.panels] == [1, 2, 3, 4, 5]
    assert sb.panels[1].dialogue[0].text == idea and len(sb.panels[1].dialogue) == 1   # "after" never replaces the panel itself
    assert sb.panels[2].dialogue[0].text == "Every word at once?" and sb.panels[2].beat == "idea"   # it keeps its panel's beat
    assert not any("nobody asked" in b.text for p in sb.panels for b in p.dialogue)   # only a flagged panel splits


# ----------------------------------------------------------------------------- round 3b review fixes

def chart_ask(*numbers):
    """A problem that asks for a chart panel after each panel (the only one a fix round may add a panel for, round 23)."""
    from services.storyboarder_v2 import CHART_PANEL_ASK
    return {n: [f'evidence: the paper\'s result "28.4 BLEU" is never drawn on a chart: {CHART_PANEL_ASK} ({{"after": N}})']
            for n in numbers}


def _split_items(panels, at, first_text, rest_text, **rest):
    p = panels[at - 1]
    first = {"n": at, "kind": "scene", "beat": p.beat, "dialogue": [{"speaker": "papi", "text": first_text}], "claims": list(p.claims),
             "visual": p.visual, "shows": p.shows}
    return first, {"after": at, "kind": "scene", "visual": p.visual, "shows": p.shows,
                   "dialogue": [{"speaker": "kumo", "text": rest_text}], **rest}


async def test_a_split_that_does_not_fit_keeps_the_whole_panel():
    # the first half was applied and its rest dropped past the cap: the panel stopped mid-explanation
    from services.storyboarder_v2 import StoryboarderV2
    from .fakes import FakeEngine
    from .test_story_v3 import make_cast, v3_paper
    panels = clean_panels()
    full = ["The Transformer, a simple architecture: only attention, so no recurrence and no convolutions.", "Every word at once?",
            "Yes, so every word sees the rest."]
    panels[1].dialogue = [Bubble("papi", full[0]), Bubble("kumo", full[1]), Bubble("papi", full[2])]
    engine = FakeEngine([{"panels": list(_split_items(panels, 2, full[0], full[1]))}])
    writer = StoryboarderV2(engine)
    writer.max_panels = 2   # 4 panels against a ceiling of 2 + 2: no room
    sb = v3_storyboard(panels, CLAIMS_WITH_RESULT)
    await writer.fix_v3(sb, make_cast(), v3_paper(), rounds=1, extra=chart_ask(2))
    assert len(sb.panels) == 4 and [b.text for b in sb.panels[1].dialogue] == full
    assert "New panels: none." in engine.prompts[0] and '{"after": 3' not in engine.prompts[0]


async def test_split_halves_follow_their_own_panels_and_keep_their_claims():
    # a half numbered in the new numbering ({"after": 3, "n": 4}) was placed before the next panel's own first half
    from services.storyboarder_v2 import StoryboarderV2
    from .fakes import FakeEngine
    from .test_story_v3 import make_cast, v3_paper
    panels = clean_panels()
    three = [("papi", "Attention looks at every word."), ("kumo", "All of them?"), ("papi", "Yes, all of them at once.")]
    panels[1].dialogue = [Bubble(s, t) for s, t in three]
    panels.insert(2, v3_panel(3, beat="how", dialogue=three, claims=[2]))
    for i, p in enumerate(panels, start=1):
        p.n = i
    a_first, a_rest = _split_items(panels, 2, "P2 first.", "P2 rest.", n=3)
    b_first, b_rest = _split_items(panels, 3, "P3 first.", "P3 rest.", n=4)
    sb = v3_storyboard(panels, CLAIMS_WITH_RESULT)
    await StoryboarderV2(FakeEngine([{"panels": [a_first, a_rest, b_first, b_rest]}])).fix_v3(sb, make_cast(), v3_paper(), rounds=1, extra=chart_ask(2, 3))
    assert [p.dialogue[0].text for p in sb.panels[1:5]] == ["P2 first.", "P2 rest.", "P3 first.", "P3 rest."]
    assert [p.n for p in sb.panels] == [1, 2, 3, 4, 5, 6, 7]
    assert sb.panels[2].claims == [1, 2] and sb.panels[4].claims == [2]   # a half without "claims" teaches its panel's claim


def test_a_panel_that_says_the_abstract_sentence_teaches_its_claim():
    # OnQ's panel said the abstract's own words and was flagged for the brief's paraphrase ("barrier, core, fixing")
    from services.storyboarder_v2 import _thin_claim
    claim = {"id": 3, "text": "The authors identified two core barriers to fixing this problem.",
             "quote": "We identify two challenges in closing this accuracy gap."}
    panel = v3_panel(4, beat="problem", dialogue=[("papi", "We identify two challenges in closing this accuracy gap.")], claims=[3])
    sb = v3_storyboard([panel], [claim])
    assert _thin_claim(sb, claim, [panel]) == []
    panel.dialogue[0].text = "Two things go wrong."
    assert _thin_claim(sb, claim, [panel])


def test_people_of_the_paper_need_a_cast_member_to_play_them():
    # the delivery book's "drivers with red caps" passed every check; the art check removed them and the idea check missed
    # them, three draws each. "Kumo the cloud student" and "a brass teacher weight" are no people to draw.
    from services.storyboarder_v2 import _mentions_people
    for text in ("A driver marks a parcel failed without going", "Drivers with red caps stand by the parcels",
                 "An agent holds a good lead card", "rows of agents on calls"):
        assert _mentions_people(text), text
    for text in ("Kumo the cloud student stares at a tall stack", "Nezu, playing the driver, holds a parcel bag",
                 "Kumo wears a driver's red cap", "a solid brass teacher weight beside a student weight", "an RL agent moves a block",
                 "no driver ever arrived at the door", "teacher model and student model"):
        assert not _mentions_people(text), text


def test_a_hook_reply_may_clear_its_shows(monkeypatch):
    # an introduction leaves "shows" empty, as the rules say; the restore put the old claim back every round (scenes with
    # a picture idea, rounds 3-22)
    from services import storyboarder_v2
    from services.storyboarder_v2 import StoryboarderV2
    from .fakes import FakeEngine
    from .test_story_v3 import make_cast
    monkeypatch.setattr(storyboarder_v2, "SCENE_SHOWS", True)
    panels = clean_panels()
    panels[0].shows = "three characters meet to discuss the problem"
    old = panels[1].shows
    sb = v3_storyboard(panels, CLAIMS_WITH_RESULT)
    hook = {"n": 1, "kind": "scene", "beat": "hook", "visual": panels[0].visual, "shows": "", "narration": panels[0].narration}
    idea = {"n": 2, "kind": "scene", "beat": "idea", "visual": panels[1].visual, "shows": "",
            "dialogue": [{"speaker": "papi", "text": "Only attention."}]}
    StoryboarderV2(FakeEngine())._replace_panels(sb, make_cast(), [hook, idea], {1, 2})
    assert sb.panels[0].shows == "" and sb.panels[1].shows == old   # a teaching scene keeps its picture's idea


def test_a_bar_charts_note_is_not_drawn():
    # a fraction's reading placed only in a bar chart's note silenced the words rule, and no reader saw it
    from services.storyboarder_v2 import _words_vs_chart
    board = {"kind": "bar", "title": "Change in hot-lead probability", "note": "0.234 = 23.4 percentage points",
             "bars": [{"label": "Calls 1-3", "value": "0.234", "highlight": True}, {"label": "Calls 4-6", "value": "0.157"}]}
    panel = v3_panel(9, kind="board", beat="proof", board=board, dialogue=[("kumo", "Up 23 points higher!")])
    assert _words_vs_chart(panel)
    board["subtitle"] = board.pop("note")
    assert _words_vs_chart(panel) == []


def test_a_legend_hides_only_its_own_percents():
    from services.storyboarder_v2 import _fractions_as_points
    table = {"kind": "table", "title": "Hot-lead probability", "subtitle": "Standard errors in parentheses, values in percentage points",
             "columns": ["Calls", "Change"], "rows": [{"cells": ["1-3", "0.234 (0.033)"]}, {"cells": ["4-6", "0.157 (0.033)"]}]}
    panel = v3_panel(8, kind="board", beat="proof", board=table, characters=("nezu",), dialogue=[("nezu", "Up 23 percentage points!")])
    assert _fractions_as_points(panel)   # the label is in the same clause as the legend
    table["subtitle"] = "*** significant at the 1% level; 95% confidence intervals"
    assert _fractions_as_points(panel) == []


def test_a_level_reads_as_a_percent_and_a_change_as_points():
    from services.storyboarder_v2 import _fractions_as_points
    table = {"kind": "table", "title": "Share of calls answered", "subtitle": "percent", "columns": ["Calls", "Share"],
             "rows": [{"cells": ["1-3", "0.234"]}, {"cells": ["4-6", "0.157"]}]}
    panel = v3_panel(8, kind="board", beat="proof", board=table, characters=("nezu",), dialogue=[("nezu", "23 percent answered!")])
    found = _fractions_as_points(panel)
    assert len(found) == 1 and "0.234 is 23.4%" in found[0] and '"0.234 = 23.4%"' in found[0]
    table["rows"][1]["cells"][1] = "-0.157"
    assert "0.234 is +23.4 percentage points" in _fractions_as_points(panel)[0]


def test_a_footnote_star_with_its_own_legend_needs_no_chance_legend():
    from services.storyboarder_v2 import _parentheses_unexplained
    table = {"kind": "table", "title": "Delivery firms", "columns": ["Firm", "Share"],
             "rows": [{"cells": ["A", "0.5*"]}, {"cells": ["B", "0.4"]}]}
    panel = v3_panel(8, kind="board", beat="proof", board=table, characters=("nezu",), dialogue=[("nezu", "Firm A leads.")])
    found = _parentheses_unexplained(panel)
    assert len(found) == 1 and "significance stars" in found[0] and found[0].endswith("or drop them")
    table["note"] = "* estimated from 2019 data"
    assert _parentheses_unexplained(panel) == []



def test_the_recap_never_ticks_a_design_line_a_panel_only_teaches():
    # round 19 ticked "Gate adds non-linearity + query-dependent sparsity" beside a metaphor, under a legend that promised
    # evidence
    from services.storyboarder_v2 import _honest_recap
    panels = clean_panels()
    panels[3].board = {"kind": "list", "title": "What the paper showed", "open": False,
                       "items": ["A new network design based solely on attention mechanisms", "28.4 BLEU on WMT 2014 English-German"]}
    sb = v3_storyboard(panels, CLAIMS_WITH_RESULT)
    _honest_recap(sb)
    assert sb.panels[3].board["checked"] == [1]   # claim 1 is taught on the idea panel and drawn on no chart


def test_a_repeated_picture_or_line_is_flagged():
    # the delivery book drew one scooter scene twice and said one line on two panels; readers took it for a misprint
    panels = clean_panels()
    panels[1].visual = panels[0].visual
    panels[1].dialogue.append(Bubble("kumo", panels[0].narration))
    found = problems_of(panels).get(2, [])
    assert any(x.startswith("the picture repeats panel 1's visual word for word") for x in found)
    assert any("repeats panel 1 word for word" in x for x in found)


def test_a_scene_that_points_at_a_board_is_flagged():
    # a board the fix rounds could not save became a scene, and "Kumo nods at the board" was drawn as a blank board
    panels = clean_panels()
    panels[2] = v3_panel(3, beat="proof", dialogue=[("nezu", "That is real. 28.4.")], claims=[3], characters=("nezu", "kumo"), shows="")
    panels[2].visual = "Nezu's eyes widen, Kumo nods at the board."
    found = problems_of(panels).get(3, [])
    assert any(x.startswith("picture: the scene points at a board or its numbers") for x in found)


def test_a_threshold_legend_explains_the_stars():
    from services.storyboarder_v2 import _parentheses_unexplained
    table = {"kind": "table", "title": "Effects", "subtitle": "Stars: * <10%, *** <1% chance of luck", "columns": ["Call", "Effect"],
             "rows": [{"cells": ["1-3", "0.234***"]}, {"cells": ["4-6", "0.081*"]}]}
    panel = v3_panel(8, kind="board", beat="proof", board=table, characters=("nezu",), dialogue=[("nezu", "Early calls moved most.")])
    assert _parentheses_unexplained(panel) == []


# ----------------------------------------------------------------------------- round 4 review fixes

def test_a_split_half_without_after_is_read_as_a_half():
    # {"n": 3} right after panel 2's rewrite was taken for panel 3: the rest was dropped, or replaced panel 3's own rewrite
    from services.storyboarder_v2 import _mark_halves
    items = [{"n": 2, "dialogue": []}, {"n": 3, "dialogue": []}]
    _mark_halves(items, {2})
    assert items[1]["after"] == 2
    items = [{"n": 2}, {"n": 3}, {"n": 3}]      # panel 3's own rewrite comes later
    _mark_halves(items, {2, 3})
    assert items[1]["after"] == 2 and "after" not in items[2]
    items = [{"n": 2}, {"n": 3}]                 # two flagged panels, each rewritten once: no half
    _mark_halves(items, {2, 3})
    assert all("after" not in x for x in items)


def test_two_recap_boards_answer_two_claims_boards():
    # a 12-claim abstract needs two list boards each way; the last recap was compared with the first claims board
    from services.storyboarder_v2 import _recap_mirrors_claims
    panels = clean_panels()
    a, b = ["Attention only", "No recurrence"], ["28.4 BLEU on English-German", "41.0 BLEU on English-French"]
    panels[0] = v3_panel(1, kind="board", beat="claims", board={"kind": "list", "title": "Claims", "open": True, "items": a})
    panels.insert(1, v3_panel(2, kind="board", beat="claims", board={"kind": "list", "title": "Results", "open": True, "items": b}))
    panels[-1].board = {"kind": "list", "title": "What the paper showed", "items": list(a)}
    panels.append(v3_panel(6, kind="board", beat="recap", board={"kind": "list", "title": "Results shown", "items": list(b)}))
    for i, p in enumerate(panels, start=1):
        p.n = i
    assert _recap_mirrors_claims(v3_storyboard(panels, CLAIMS_WITH_RESULT)) == {}
    panels[-1].board["items"] = list(a)          # the second recap repeats the first claims board
    assert list(_recap_mirrors_claims(v3_storyboard(panels, CLAIMS_WITH_RESULT))) == [6]


def test_a_board_that_repeats_an_earlier_board_is_flagged():
    from .test_story_v3 import TABLE
    panels = clean_panels()
    panels.insert(3, v3_panel(4, kind="board", beat="proof", board=dict(TABLE), dialogue=[("papi", "Now the data again.")],
                              claims=[3], characters=("papi",)))
    for i, p in enumerate(panels, start=1):
        p.n = i
    assert any(x.startswith("the board repeats panel 3's board") for x in problems_of(panels).get(4, []))



def test_a_recap_that_calls_its_lines_confirmed_is_flagged_and_retitled():
    # "Confirmed findings" over a recap whose every line had a tick: the story showed evidence, it confirmed nothing
    from services.storyboarder_v2 import _honest_recap
    panels = clean_panels()
    panels[3].board = {**panels[3].board, "title": "Confirmed findings", "items": ["28.4 BLEU on WMT 2014 English-German"]}
    panels[3].dialogue[0].text = "These are the confirmed findings."
    assert any("never that every claim was verified" in x for x in problems_of(panels).get(4, []))
    sb = v3_storyboard(panels, CLAIMS_WITH_RESULT)
    _honest_recap(sb)
    assert "checked" not in sb.panels[3].board and sb.panels[3].board["title"] == "What the paper showed"


def test_a_legend_line_in_the_words_is_flagged():
    panels = clean_panels()
    panels[1].narration = "Red yarn = attention."
    assert any("reads as a legend" in x for x in problems_of(panels).get(2, []))


def test_words_that_read_a_change_bar_but_not_its_badge_are_flagged():
    # econ_ai_service: "+23.4 points" (the 0.234 bar) under a "+0.242 vs calls 81-120" badge, a difference of two changes
    from services.storyboarder_v2 import _words_vs_chart
    board = {"kind": "bar", "title": "Change in hot-lead probability", "subtitle": "0.234 = +23.4 percentage points",
             "bars": [{"label": "Calls 1-3", "value": "0.234", "highlight": True}, {"label": "Calls 4-6", "value": "0.157"},
                      {"label": "Calls 81-120", "value": "-0.008", "baseline": True}]}
    panel = v3_panel(9, kind="board", beat="proof", board=board, dialogue=[("kumo", "First three calls: +23.4 points!")])
    found = _words_vs_chart(panel)
    assert len(found) == 1 and "the badge draws its gap" in found[0]
    del board["bars"][2]["baseline"]             # no badge: the words read the bar
    assert _words_vs_chart(panel) == []


def test_a_setting_in_the_chart_title_is_no_drawn_value():
    # "adds 2 points" passed because the subtitle said "2-bit"
    from services.storyboarder_v2 import _words_vs_chart
    board = {"kind": "bar", "title": "Average score", "subtitle": "2-bit Qwen3-8B",
             "bars": [{"label": "OnQ", "value": "70.68", "highlight": True}, {"label": "EfficientQAT", "value": "60.42", "baseline": True}]}
    panel = v3_panel(9, kind="board", beat="proof", board=board, dialogue=[("kumo", "OnQ adds 2 points over EfficientQAT.")])
    assert _words_vs_chart(panel)


def test_every_translated_recap_board_gets_the_english_marks():
    from services.storyboarder_v2 import copy_recap_marks
    english, chinese = clean_panels(), clean_panels()
    english.append(v3_panel(5, kind="board", beat="recap", board={"kind": "list", "title": "Results", "items": ["x", "y"], "checked": [1]}))
    chinese.append(v3_panel(5, kind="board", beat="recap", board={"kind": "list", "title": "已验证的结果", "items": ["甲", "乙"]}))
    english[3].board = {"kind": "list", "title": "What the paper showed", "items": ["a", "b"]}
    chinese[3].board = {"kind": "list", "title": "论文展示了什么", "items": ["丙", "丁"]}
    zh = v3_storyboard(chinese, CLAIMS_WITH_RESULT)
    copy_recap_marks(v3_storyboard(english, CLAIMS_WITH_RESULT), zh, "zh-CN")
    assert zh.panels[4].board["checked"] == [1] and zh.panels[4].board["title"] == "论文展示了什么"


def test_the_method_walks_the_running_example():
    # LongZero's question appeared only at the start and the end; "the answer is Sofia Bianchi" meant nothing in between
    from services.storyboarder_v2 import lint_v3
    from .fakes import BRIEF_V3, V3_EXAMPLE
    panels = clean_panels()
    panels[0].dialogue[0].text = f"'{V3_EXAMPLE}' Who is 'it'?"
    panels.insert(2, v3_panel(3, beat="how", dialogue=[("papi", "Every word looks at every other word, so the model sees it all.")], claims=[1]))
    panels.insert(3, v3_panel(4, beat="payoff", dialogue=[("kumo", "The animal was too tired: it is the animal!")], claims=[1]))
    for i, p in enumerate(panels, start=1):
        p.n = i
    sb = v3_storyboard(panels, CLAIMS_WITH_RESULT, brief=BRIEF_V3)
    sb.running_example = V3_EXAMPLE
    found = lint_v3(sb, V3_PAPER_TEXT).get(3, [])
    assert any(x.startswith("the method panels never use the running example") for x in found)
    panels[2].dialogue[0].text = "So 'it' looks at animal, street and tired at once."
    assert not any(x.startswith("the method panels never use the running example") for x in lint_v3(sb, V3_PAPER_TEXT).get(3, []))


def test_a_panel_that_says_the_last_one_again_is_flagged():
    # "the bar to keep this lead rises" three times over, in other words each time
    panels = clean_panels()
    panels[1].dialogue = [Bubble("papi", "Because better leads are coming next, the bar to keep this lead rises.")]
    panels.insert(2, v3_panel(3, beat="idea", dialogue=[("papi", "With better leads coming next, the bar to keep the lead rises.")],
                              claims=[1, 2]))
    for i, p in enumerate(panels, start=1):
        p.n = i
    assert any(x.startswith("this panel's words say again what panel 2 said") for x in problems_of(panels).get(3, []))


# ----------------------------------------------------------------------------- round 5 review fixes

async def test_a_refused_ambiguous_half_keeps_both_panels_whole():
    # [3 complete, 4 first half, {n: 4} rest] read as 3's rest; refused for room, it threw away 3's rewrite and left 4 half
    from services.storyboarder_v2 import StoryboarderV2
    from .fakes import FakeEngine
    from .test_story_v3 import make_cast, v3_paper
    panels = clean_panels()
    three = [("papi", "Attention looks at every word."), ("kumo", "All of them?"), ("papi", "Yes, all of them at once.")]
    panels[1].dialogue = [Bubble(s, t) for s, t in three]
    panels.insert(2, v3_panel(3, beat="how", dialogue=three, claims=[2]))
    for i, p in enumerate(panels, start=1):
        p.n = i
    old = [[b.text for b in p.dialogue] for p in panels]
    reply = [{"n": 2, "kind": "scene", "beat": "idea", "dialogue": [{"speaker": "papi", "text": "P2 whole."}], "visual": panels[1].visual,
              "shows": panels[1].shows}, {"n": 3, "kind": "scene", "dialogue": [{"speaker": "papi", "text": "P3 first."}]},
             {"n": 3, "kind": "scene", "dialogue": [{"speaker": "kumo", "text": "P3 rest."}]}]
    writer = StoryboarderV2(FakeEngine([{"panels": reply}]))
    writer.max_panels = 3   # no room
    sb = v3_storyboard(panels, CLAIMS_WITH_RESULT)
    await writer.fix_v3(sb, make_cast(), v3_paper(), rounds=1)
    assert [[b.text for b in p.dialogue] for p in sb.panels[1:3]] == old[1:3]   # both whole: nothing half-applied


async def test_a_split_that_moves_the_board_leaves_the_first_half_without_it():
    from services.storyboarder_v2 import StoryboarderV2
    from .fakes import FakeEngine
    from .test_story_v3 import TABLE, make_cast, v3_paper
    panels = clean_panels()
    panels[2].dialogue = [Bubble("nezu", "28.4, more than 2.0 above the rest?"), Bubble("papi", "Look at the table."),
                          Bubble("nezu", "Show me.")]
    reply = [{"n": 3, "kind": "scene", "beat": "proof", "board": None, "characters": ["nezu"],
              "dialogue": [{"speaker": "nezu", "text": "Guess the score first."}], "visual": "nezu taps her chin."},
             {"after": 3, "kind": "board", "board": dict(TABLE), "claims": [3], "characters": ["nezu", "papi"],
              "dialogue": [{"speaker": "nezu", "text": "28.4, more than 2.0 above the rest?"}]}]
    sb = v3_storyboard(panels, CLAIMS_WITH_RESULT)
    await StoryboarderV2(FakeEngine([{"panels": reply}])).fix_v3(sb, make_cast(), v3_paper(), rounds=1, extra=chart_ask(3))
    assert sb.panels[2].kind == "scene" and not sb.panels[2].board and sb.panels[3].board["title"] == TABLE["title"]



def test_a_supported_subtitle_goes_from_the_recap():
    from services.storyboarder_v2 import _honest_recap
    panels = clean_panels()
    panels[3].board = {"kind": "list", "title": "What the paper showed", "subtitle": "Claims supported by the presented evidence",
                       "items": ["28.4 BLEU on WMT 2014 English-German", "41.8 BLEU on English-French"]}
    sb = v3_storyboard(panels, CLAIMS_WITH_RESULT)
    _honest_recap(sb)
    assert sb.panels[3].board["checked"] == [0] and "subtitle" not in sb.panels[3].board
    panels[3].board = {"kind": "list", "title": "Verified results", "subtitle": "All results verified by the study data",
                       "items": ["28.4 BLEU on WMT 2014 English-German"]}
    _honest_recap(sb)
    assert sb.panels[3].board["title"] == "What the paper showed" and "subtitle" not in sb.panels[3].board   # over ticks too
    # round 19: "All claims tested in this story" over a recap with three dots
    panels[3].board = {"kind": "list", "title": "What the paper showed", "subtitle": "All claims tested in this story",
                       "items": ["28.4 BLEU on WMT 2014 English-German", "41.8 BLEU on English-French"]}
    _honest_recap(sb)
    assert "subtitle" not in sb.panels[3].board
    # "tested" alone states a scope, and stays
    panels[3].board = {**panels[3].board, "title": "Findings on tested benchmarks", "subtitle": "On tested translation tasks"}
    _honest_recap(sb)
    assert (sb.panels[3].board["title"], sb.panels[3].board["subtitle"]) == ("Findings on tested benchmarks", "On tested translation tasks")


def test_the_recap_rule_reads_past_years_and_settings():
    # without the check model's verdict: "WMT 2014" is the test set's year, not a result a chart must draw; "2-bit" is a
    # setting: "Wins every 2-bit benchmark at every size" kept its tick beside one model's 2-bit average because a chart
    # drew a 2 (round 21 review)
    from services.storyboarder_v2 import _honest_recap
    panels = clean_panels()
    panels[2].board = {"kind": "bar", "title": "English-German translation (BLEU)", "subtitle": "best of 2 runs",
                       "bars": [{"label": "ConvS2S ensemble", "value": "26.36", "baseline": True},
                                {"label": "Transformer (big)", "value": "28.4", "highlight": True}]}
    panels[3].board = {"kind": "list", "title": "What the paper showed",
                       "items": ["28.4 BLEU on WMT 2014 English-German", "Wins every 2-bit benchmark at every size"]}
    sb = v3_storyboard(panels, CLAIMS_WITH_RESULT)
    _honest_recap(sb)
    assert sb.panels[3].board["checked"] == [0]


def test_a_translated_recap_follows_the_english_header():
    # a relayout judged "本故事中检验过的所有主张" by its own words and kept it over the dots while the English "All claims
    # tested in this story" went (round 21 review); a scope subtitle stays in both languages
    import copy as _copy
    from services.storyboarder_v2 import _honest_recap, copy_recap_marks
    panels = clean_panels()
    panels[3].board = {"kind": "list", "title": "What the paper showed", "subtitle": "All claims tested in this story",
                       "items": ["28.4 BLEU on WMT 2014 English-German", "41.8 BLEU on English-French"]}
    english = v3_storyboard(panels, CLAIMS_WITH_RESULT)
    chinese = _copy.deepcopy(english)
    chinese.panels[3].board = {"kind": "list", "title": "论文发现", "subtitle": "本故事中检验过的所有主张",
                               "items": ["英德翻译 28.4 BLEU", "英法翻译 41.8 BLEU"]}
    _honest_recap(english)
    copy_recap_marks(english, chinese, "zh-CN")
    assert "subtitle" not in english.panels[3].board and "subtitle" not in chinese.panels[3].board
    assert chinese.panels[3].board["checked"] == english.panels[3].board["checked"] == [0]
    assert chinese.panels[3].board["title"] == "论文展示了什么"   # the English title is the honest one
    english.panels[3].board = {**english.panels[3].board, "title": "Findings on tested benchmarks", "subtitle": "On tested translation tasks"}
    chinese.panels[3].board = {**chinese.panels[3].board, "title": "测试基准上的发现", "subtitle": "在测试的翻译任务上"}
    _honest_recap(english)
    copy_recap_marks(english, chinese, "zh-CN")
    assert english.panels[3].board["subtitle"] == "On tested translation tasks"
    assert (chinese.panels[3].board["title"], chinese.panels[3].board["subtitle"]) == ("测试基准上的发现", "在测试的翻译任务上")


def test_a_symbol_key_in_another_word_form_counts():
    from services.storyboarder_v2 import _symbol_keys
    brief = {"picture_symbols": [{"concept": "lead reallocation", "symbol": "a card dropped into a woven basket"}]}
    panels = clean_panels()
    panels[1].visual = "kumo drops a card into a woven basket"
    panels[1].dialogue[0].text = "Kumo is reallocating this lead: the card goes back into the basket."
    assert _symbol_keys(v3_storyboard(panels, CLAIMS_WITH_RESULT, brief=brief)) == {}


def test_a_score_chart_reads_its_value_with_its_badge():
    # a gain chart or a score chart: the bar's own value and the badge read fine together
    from services.storyboarder_v2 import _words_vs_chart
    board = {"kind": "bar", "title": "Accuracy gain over the base model",
             "bars": [{"label": "LongZero", "value": "2.74", "highlight": True}, {"label": "Teacher QA", "value": "1.31", "baseline": True}]}
    panel = v3_panel(9, kind="board", beat="proof", board=board, dialogue=[("kumo", "LongZero gains 2.74 points!")])
    assert _words_vs_chart(panel) == []


def test_a_threshold_legend_is_no_unit_label():
    from services.storyboarder_v2 import _fractions_as_points
    table = {"kind": "table", "title": "Hot-lead probability", "subtitle": "Stars: * <10%, *** <1% chance of luck",
             "columns": ["Calls", "Effect"], "rows": [{"cells": ["1-3", "0.234***"]}, {"cells": ["4-6", "0.157*"]}]}
    panel = v3_panel(8, kind="board", beat="proof", board=table, characters=("nezu",), dialogue=[("nezu", "Up 23 percentage points!")])
    assert _fractions_as_points(panel) == []


def test_two_charts_of_one_metric_for_two_models_are_no_repeat():
    # LongZero's Qwen3-4B and Qwen3-8B charts share the title "Average Benchmark Score"
    from .test_story_v3 import TABLE
    panels = clean_panels()
    other = {**TABLE, "rows": [{"cells": ["English-German", "27.3"], "quote": "x"}, {"cells": ["English-French", "38.1"], "quote": "y"}]}
    panels.insert(3, v3_panel(4, kind="board", beat="proof", board=other, dialogue=[("papi", "And the base model?")],
                              claims=[3], characters=("papi",)))
    for i, p in enumerate(panels, start=1):
        p.n = i
    assert not any("repeats panel" in x for x in problems_of(panels).get(4, []))


def test_a_reading_note_is_no_legend_line():
    panels = clean_panels()
    panels[1].narration = "Call duration is call length; lower = shorter calls."
    assert not any("reads as a legend" in x for x in problems_of(panels).get(2, []))


async def test_a_picture_problem_alone_never_rewrites_the_words():
    # round 5: picture rules sent whole panels to rewrites; pictures explained more and readers learned less
    from services.storyboarder_v2 import StoryboarderV2
    from .fakes import FakeEngine
    from .test_story_v3 import make_cast, v3_paper
    panels = clean_panels()
    panels[1].visual, panels[1].shows = "papi points at the numbers on the whiteboard", ""   # a scene has no board: drawn blank
    words = [b.text for b in panels[1].dialogue]
    reply = {"n": 2, "visual": "papi waves kumo over to the desk", "shows": "",
             "dialogue": [{"speaker": "papi", "text": "A rewrite nobody asked for."}]}
    engine = FakeEngine([{"panels": [reply]}])
    sb = v3_storyboard(panels, CLAIMS_WITH_RESULT)
    await StoryboarderV2(engine).fix_v3(sb, make_cast(), v3_paper(), rounds=1)
    assert len(engine.calls) == 1 and "Fix the pictures of these manga panels" in engine.prompts[0]
    assert [b.text for b in sb.panels[1].dialogue] == words and sb.panels[1].visual == reply["visual"]


# ----------------------------------------------------------------------------- round 6 review fixes

async def test_a_picture_fix_puts_the_cast_member_it_stages_in_the_panel():
    # "Nezu, in the red delivery cap, rides past Papi" on a panel of papi alone: the art prompt said nezu does not appear
    from services.storyboarder_v2 import StoryboarderV2
    from .fakes import FakeEngine
    from .test_story_v3 import make_cast
    panels = clean_panels()
    panels[1].characters = ["papi"]
    reply = {"n": 2, "visual": "Nezu, in a red cap, ties red yarn between blank cards past Papi", "shows": "the cards get tied together"}
    sb = v3_storyboard(panels, CLAIMS_WITH_RESULT)
    await StoryboarderV2(FakeEngine([{"panels": [reply]}]))._fix_pictures(sb, make_cast(), {2: ["picture: ..."]})
    assert sb.panels[1].characters == ["papi", "nezu"]


def test_a_half_is_only_the_previous_panels_rest():
    # [2 whole, 5 first, {n: 5} rest] put panel 5's first half after panel 2, before panels 3 and 4
    from services.storyboarder_v2 import _mark_halves
    items = [{"n": 2}, {"n": 5}, {"n": 5}]
    _mark_halves(items, {2, 5})
    assert "after" not in items[1] and items[2]["after"] == 5


def test_a_board_without_numbers_repeats_by_its_kind_and_title():
    panels = clean_panels()
    flow = {"kind": "flow", "title": "The Transformer, end to end", "steps": ["Words in", "Encoder", "Decoder", "Words out"]}
    panels[1] = v3_panel(2, kind="board", beat="how", board=dict(flow), dialogue=[("papi", "Words go in, words come out.")], claims=[1, 2])
    panels.insert(2, v3_panel(3, kind="board", beat="how", board={**flow, "steps": ["Input", "Encoder", "Decoder", "Output"]},
                              dialogue=[("papi", "Encoder, then decoder.")], claims=[1, 2]))
    for i, p in enumerate(panels, start=1):
        p.n = i
    assert any(x.startswith("the board repeats panel 2's board") for x in problems_of(panels).get(3, []))


def test_a_symbol_key_needs_the_symbols_own_word():
    # "thin" is not "thick", and a cast name ("papi") never names a symbol ("paper")
    from services.storyboarder_v2 import _symbol_keys
    brief = {"picture_symbols": [{"concept": "teacher model", "symbol": "a thick gold brick"},
                                 {"concept": "student model", "symbol": "a thin silver brick"}]}
    panels = clean_panels()
    panels[1].visual = "papi sets the thick gold brick beside the thin silver brick"
    panels[1].dialogue[0].text = "The thin one is the student model."
    found = _symbol_keys(v3_storyboard(panels, CLAIMS_WITH_RESULT, brief=brief)).get(2, [])
    assert len(found) == 1 and "thick gold brick" in found[0]


async def test_a_rest_with_another_board_leaves_the_first_half_its_board():
    from services.storyboarder_v2 import StoryboarderV2
    from .fakes import FakeEngine
    from .test_story_v3 import TABLE, make_cast, v3_paper
    panels = clean_panels()
    panels[2].dialogue = [Bubble("nezu", "28.4, more than 2.0 above the rest?"), Bubble("papi", "Look at the table."),
                          Bubble("nezu", "Show me.")]
    reply = [{"n": 3, "kind": "board", "beat": "proof", "characters": ["nezu", "papi"], "claims": [3],
              "dialogue": [{"speaker": "nezu", "text": "28.4, more than 2.0 above the rest?"}]},   # board left out by mistake
             {"after": 3, "kind": "board", "characters": ["papi"], "claims": [3],
              "board": {"kind": "list", "title": "Why it matters", "items": ["Faster training", "Better translations"]},
              "dialogue": [{"speaker": "papi", "text": "And it trains faster."}]}]
    sb = v3_storyboard(panels, CLAIMS_WITH_RESULT)
    await StoryboarderV2(FakeEngine([{"panels": reply}])).fix_v3(sb, make_cast(), v3_paper(), rounds=1)
    assert sb.panels[2].board["title"] == TABLE["title"] and sb.panels[3].board["kind"] == "list"


def test_a_significance_legend_clause_is_no_unit_label():
    from services.storyboarder_v2 import _fractions_as_points
    table = {"kind": "table", "title": "Hot-lead probability", "subtitle": "Significance: *** at 1%, ** at 5%, * at 10%",
             "columns": ["Calls", "Effect"], "rows": [{"cells": ["1-3", "0.234***"]}, {"cells": ["4-6", "0.157*"]}]}
    panel = v3_panel(8, kind="board", beat="proof", board=table, characters=("nezu",), dialogue=[("nezu", "Up 23 percentage points!")])
    assert _fractions_as_points(panel) == []
    table["subtitle"] = "Values in percentage points; *** significant at 1%"
    assert _fractions_as_points(panel)


def test_bins_of_one_quantity_are_changes_for_the_badge_rule():
    from services.storyboarder_v2 import _words_vs_chart
    board = {"kind": "bar", "title": "Hot-lead probability", "subtitle": "0.234 = +23.4 percentage points",
             "bars": [{"label": "Calls 1-3", "value": "0.234", "highlight": True}, {"label": "Calls 4-6", "value": "0.157"},
                      {"label": "Calls 81-120", "value": "0.012", "baseline": True}]}
    panel = v3_panel(9, kind="board", beat="proof", board=board, dialogue=[("kumo", "First three calls: +23.4 points!")])
    assert any("unmark the baseline" in x for x in _words_vs_chart(panel))



async def test_a_first_revision_that_lowers_the_quiz_is_undone_and_tried_again(monkeypatch):
    # Right On Q kept a first revision that took the quiz from 0.82 to 0.75; a second try may do better (5137: 0.75 → 0.6 → 0.85)
    import services.abstract_quiz as aq
    from services.storyboarder_v2 import StoryboarderV2
    from .fakes import FakeEngine, PaperModel
    from .test_story_v3 import make_cast, v3_paper
    original, scores = aq.run_quiz, iter([0.75, 0.5, 1.0])

    async def scripted(*args, **kwargs):
        result = await original(*args, **kwargs)
        score = next(scores)
        return {**result, "score": score, "failed_claim_ids": [] if score == 1.0 else [3]}

    monkeypatch.setattr(aq, "run_quiz", scripted)
    sb = await StoryboarderV2(FakeEngine(router=PaperModel(quiz_misses=[{3}]))).build_v3(v3_paper(), make_cast(), "en-US", (16, 32))
    assert sb.quiz["score"] == 1.0 and [r["score"] for r in sb.quiz["rounds"]] == [0.5, 1.0] and sb.quiz["before"]["score"] == 0.75


def test_a_claims_board_line_that_names_a_claim_by_a_stand_in_is_flagged():
    # every LongZero reader answered "good data": the claims board said it, and the recap repeated it
    from services.storyboarder_v2 import _vague_list_lines
    panels = clean_panels()
    panels[0] = v3_panel(1, kind="board", beat="claims", board={"kind": "list", "title": "What the paper claims", "open": True,
                                                               "items": ["RL needs good data", "Two core flaws cause this gap"]})
    found = _vague_list_lines(v3_storyboard(panels, CLAIMS_WITH_RESULT)).get(1, [])
    assert len(found) == 2 and '"good data"' in found[0] and '"Two core flaws"' in found[1]
    panels[0].board["items"] = ["RL needs long, diverse, challenging, verifiable data", "Trajectory divergence and misalignment"]
    assert _vague_list_lines(v3_storyboard(panels, CLAIMS_WITH_RESULT)) == {}


# ----------------------------------------------------------------------------- round 7 review fixes

async def test_a_rest_marked_board_takes_the_board_its_first_half_dropped():
    from services.storyboarder_v2 import StoryboarderV2
    from .fakes import FakeEngine
    from .test_story_v3 import TABLE, make_cast, v3_paper
    panels = clean_panels()
    panels[2].dialogue = [Bubble("nezu", "28.4, more than 2.0 above the rest?"), Bubble("papi", "Look at the table."),
                          Bubble("nezu", "Show me.")]
    reply = [{"n": 3, "kind": "scene", "beat": "proof", "board": None, "characters": ["nezu"], "visual": "nezu taps her chin.",
              "dialogue": [{"speaker": "nezu", "text": "Guess the score first."}]},
             {"after": 3, "kind": "board", "characters": ["nezu", "papi"], "claims": [3],
              "dialogue": [{"speaker": "nezu", "text": "28.4, more than 2.0 above the rest?"}]}]
    sb = v3_storyboard(panels, CLAIMS_WITH_RESULT)
    await StoryboarderV2(FakeEngine([{"panels": reply}])).fix_v3(sb, make_cast(), v3_paper(), rounds=1, extra=chart_ask(3))
    assert sb.panels[2].kind == "scene" and not sb.panels[2].board and sb.panels[3].board["title"] == TABLE["title"]


async def test_a_repeated_board_can_become_a_scene():
    # the repeat rule offers "react to the board before as a scene"; keep_boards put the table back every round
    from services.storyboarder_v2 import StoryboarderV2
    from .fakes import FakeEngine
    from .test_story_v3 import TABLE, make_cast, v3_paper
    panels = clean_panels()
    panels.insert(3, v3_panel(4, kind="board", beat="proof", board=dict(TABLE), dialogue=[("papi", "Now the data again.")],
                              claims=[3], characters=("papi",)))
    for i, p in enumerate(panels, start=1):
        p.n = i
    reply = [{"n": 4, "kind": "scene", "beat": "proof", "board": None, "characters": ["papi", "kumo"], "claims": [3],
              "visual": "papi grins while kumo claps", "dialogue": [{"speaker": "kumo", "text": "So 28.4 really beat them all!"}]}]
    sb = v3_storyboard(panels, CLAIMS_WITH_RESULT)
    await StoryboarderV2(FakeEngine([{"panels": reply}])).fix_v3(sb, make_cast(), v3_paper(), rounds=1)
    assert sb.panels[3].kind == "scene" and not sb.panels[3].board


def test_short_word_forms_name_a_symbol():
    from services.storyboarder_v2 import _key_words
    assert _key_words({"box"}, {"boxe"}) and _key_words({"loss"}, {"lost"}) and not _key_words({"thick"}, {"thin"})


def test_a_line_that_names_its_items_after_the_count_is_specific():
    from services.storyboarder_v2 import _vague_list_lines
    panels = clean_panels()
    panels[0] = v3_panel(1, kind="board", beat="claims", board={"kind": "list", "title": "Claims", "open": True,
                                                               "items": ["Two core flaws: drift and misalignment"]})
    assert _vague_list_lines(v3_storyboard(panels, CLAIMS_WITH_RESULT)) == {}


def test_a_percent_matches_a_cell_at_the_words_precision():
    # "1.6% fewer" is no reading of a 0.0155 cell's 1.55 at the words' precision... but "2%" of 0.0155 is
    from services.storyboarder_v2 import _fractions_as_points
    table = {"kind": "table", "title": "Change in successful deliveries", "subtitle": "percent", "columns": ["Parcel", "Change"],
             "rows": [{"cells": ["Lied-about", "-0.0120"]}, {"cells": ["New", "-0.0682"]}]}
    panel = v3_panel(8, kind="board", beat="proof", board=table, characters=("nezu",), dialogue=[("nezu", "That is 1.86% fewer.")])
    assert _fractions_as_points(panel) == []   # 1.86 is no reading of 1.20 or 6.82
    panel.dialogue[0].text = "That is 6.82% fewer."
    assert _fractions_as_points(panel)


def test_an_effect_against_a_zero_reference_needs_no_field():
    from services.storyboarder_v2 import _story_board_issues
    board = {"kind": "bar", "title": "Next-day successful deliveries, % change",
             "bars": [{"label": "No fake remark", "value": "0", "baseline": True}, {"label": "Per fake remark", "value": "-1.60", "highlight": True}]}
    assert not any(x.startswith("only 2 bars") for x in _story_board_issues(board))


def test_a_promised_dashed_line_is_flagged():
    # "Dashed line = base model score 54.68": the program draws no such line
    from .test_story_v3 import TABLE
    panels = clean_panels()
    panels[2].board = {**TABLE, "subtitle": "Dashed line = base model; higher is better"}
    assert any("never draws" in x for x in problems_of(panels).get(3, []))


def test_a_legend_of_colon_pairs_is_flagged():
    panels = clean_panels()
    panels[1].narration = "Cards: words, yarn: attention."
    assert any("reads as a legend" in x for x in problems_of(panels).get(2, []))


def test_a_symbol_the_words_name_must_be_drawn():
    # "This smooth river stone is training reward" over a desk of paper, string and a needle
    panels = clean_panels()
    brief = {"picture_symbols": [{"concept": "a sentence", "symbol": "a row of blank cards"},
                                 {"concept": "the reward", "symbol": "a smooth river stone"}]}
    panels[1].dialogue.append(Bubble("kumo", "This smooth river stone is the reward?"))
    found = problems_of_with(panels, brief).get(2, [])
    assert any('the words name "a smooth river stone"' in x for x in found)
    panels[1].visual += " A smooth river stone sits beside them."
    assert not any("the words name" in x for x in problems_of_with(panels, brief).get(2, []))


# ----------------------------------------------------------------------------- round 8 review fixes

async def test_a_rest_with_another_flow_leaves_the_first_half_its_flow():
    from services.storyboarder_v2 import StoryboarderV2
    from .fakes import FakeEngine
    from .test_story_v3 import make_cast, v3_paper
    panels = clean_panels()
    flow = {"kind": "flow", "title": "The Transformer, end to end", "steps": ["Words in", "Encoder", "Decoder", "Words out"]}
    panels[1] = v3_panel(2, kind="board", beat="how", board=dict(flow), claims=[1, 2],
                         dialogue=[("papi", "Words go in, the encoder and decoder attend, and words come out the other side.")])
    reply = [{"n": 2, "kind": "board", "beat": "how", "claims": [1, 2], "dialogue": [{"speaker": "papi", "text": "Words in, words out."}]},
             {"after": 2, "kind": "board", "claims": [1, 2], "dialogue": [{"speaker": "papi", "text": "Inside one layer."}],
              "board": {"kind": "flow", "title": "Inside one encoder layer", "steps": ["Self-attention", "Feed-forward"]}}]
    sb = v3_storyboard(panels, CLAIMS_WITH_RESULT)
    await StoryboarderV2(FakeEngine([{"panels": reply}])).fix_v3(sb, make_cast(), v3_paper(), rounds=1, extra=chart_ask(2))
    assert sb.panels[1].board and sb.panels[1].board["title"] == flow["title"] and sb.panels[2].board["title"] == "Inside one encoder layer"


def test_the_symbol_the_words_name_best_is_the_one_checked():
    # "this grey card is a cold lead" with only a red card drawn: the grey one is missing
    panels = clean_panels()
    brief = {"picture_symbols": [{"concept": "hot lead", "symbol": "a glowing red playing card"},
                                 {"concept": "cold lead", "symbol": "a dull grey playing card"}]}
    panels[1].visual = "papi holds up a glowing red playing card and ties red yarn to it"
    panels[1].dialogue.append(Bubble("kumo", "And this dull grey playing card is a cold lead?"))
    found = problems_of_with(panels, brief).get(2, [])
    assert any('the words name "a dull grey playing card"' in x for x in found)


def test_a_hyphenated_percentage_point_label_is_a_percent_label():
    from services.storyboarder_v2 import _fractions_as_points
    table = {"kind": "table", "title": "Hot-lead probability", "unit": "percentage-point change", "columns": ["Calls", "Effect"],
             "rows": [{"cells": ["1-3", "0.234"]}, {"cells": ["4-6", "0.157"]}]}
    panel = v3_panel(8, kind="board", beat="proof", board=table, characters=("nezu",), dialogue=[("nezu", "Up 23 percentage points!")])
    assert _fractions_as_points(panel)


async def test_a_fix_reply_with_broken_json_is_asked_for_once_more():
    from services.storyboarder_v2 import StoryboarderV2
    from .fakes import FakeEngine
    from .test_story_v3 import make_cast, v3_paper
    panels = clean_panels()
    panels[1].dialogue = [Bubble("papi", "The Transformer, a simple architecture: only attention, so no recurrence and no convolutions."),
                          Bubble("kumo", "Every word at once?"), Bubble("papi", "Yes, so every word sees the rest.")]
    fixed = {"n": 2, "kind": "scene", "beat": "idea", "claims": [1, 2], "visual": panels[1].visual, "shows": panels[1].shows,
             "dialogue": [{"speaker": "papi", "text": "The Transformer, a simple architecture: only attention, so no recurrence and no convolutions."}]}
    engine = FakeEngine(['{"panels": [{"n": 2,', '{"panels": [{"n": 2,', {"panels": [fixed]}])   # broken, its repair broken, then good
    sb = v3_storyboard(panels, CLAIMS_WITH_RESULT)
    writer = StoryboarderV2(engine)
    await writer.fix_v3(sb, make_cast(), v3_paper(), rounds=1)
    assert len(sb.panels[1].dialogue) == 1 and not writer.skipped


# ----------------------------------------------------------------------------- round 9 review fixes

def test_a_scene_that_says_there_are_no_boards_points_at_none():
    panels = clean_panels()
    panels[1].visual += " No boards, text or numbers appear anywhere."
    assert not any("drawn blank" in x for x in problems_of(panels).get(2, []))


def test_position_words_draw_no_symbol():
    # "left = start, right = end" in a dot line's description: a picture of a left and a right pile draws no dot line
    from services.storyboarder_v2 import _symbol_keys
    brief = {"picture_symbols": [{"concept": "the workday", "symbol": "horizontal line of small dots, left = start, right = end"}]}
    panels = clean_panels()
    panels[1].visual = "papi moves a card from the left pile to the right pile"
    assert 2 not in _symbol_keys(v3_storyboard(panels, CLAIMS_WITH_RESULT, brief=brief))


def test_statistics_rows_and_bins_are_no_systems():
    from services.ledger import _not_a_system
    assert _not_a_system("Observations") and _not_a_system("1–3") and _not_a_system("Adj. R²") and not _not_a_system("OnQ")


def test_word_for_word_repeats_never_ship():
    # a late quiz revision copied panel 7 as panel 8, and the lint flagged it with no round left
    from services.storyboarder_v2 import scrub_repeats
    panels = clean_panels()
    copy_panel = v3_panel(3, beat="idea", dialogue=[(b.speaker, b.text) for b in panels[1].dialogue], claims=[2])
    copy_panel.visual = panels[1].visual
    panels.insert(2, copy_panel)
    panels[0].narration = panels[0].narration   # the hook keeps its own line
    extra = v3_panel(4, beat="how", dialogue=[("papi", panels[0].narration), ("kumo", "And it still works?")], claims=[1])
    panels.insert(3, extra)
    for i, p in enumerate(panels, start=1):
        p.n = i
    sb = v3_storyboard(panels, CLAIMS_WITH_RESULT)
    removed = scrub_repeats(sb)
    assert any(x.startswith("panel 3: a repeat of panel 2") for x in removed)          # the copy goes as a whole
    assert [b.text for b in sb.panels[2].dialogue] == ["And it still works?"]            # the repeated line goes, the new one stays
    assert 2 in sb.panels[1].claims and 1 in sb.panels[0].claims


def test_the_running_example_called_back_on_a_new_picture_stays():
    # "Who used the Vitae Extract from Aurora Biologica?" asked again just before its answer is a callback, not a copy
    from services.storyboarder_v2 import scrub_repeats
    from .fakes import V3_EXAMPLE
    panels = clean_panels()
    panels[0].dialogue = [Bubble("kumo", f"'{V3_EXAMPLE}' Who is 'it'?")]
    payoff = v3_panel(3, beat="payoff", dialogue=[("kumo", f"'{V3_EXAMPLE}' Who is 'it'?"), ("papi", "The animal, it was too tired.")],
                      claims=[1])
    payoff.visual = "kumo points at the animal card while papi nods"
    panels.insert(2, payoff)
    for i, p in enumerate(panels, start=1):
        p.n = i
    sb = v3_storyboard(panels, CLAIMS_WITH_RESULT)
    sb.running_example = V3_EXAMPLE
    assert scrub_repeats(sb) == [] and len(sb.panels[2].dialogue) == 2


def test_a_copied_scene_with_one_reworded_line_goes():
    from services.storyboarder_v2 import scrub_repeats
    panels = clean_panels()
    copy_panel = v3_panel(3, beat="idea", dialogue=[(panels[1].dialogue[0].speaker, panels[1].dialogue[0].text),
                                                    ("kumo", "So only attention, no recurrence and no convolutions at all?")], claims=[2])
    copy_panel.visual = panels[1].visual
    panels.insert(2, copy_panel)
    for i, p in enumerate(panels, start=1):
        p.n = i
    sb = v3_storyboard(panels, CLAIMS_WITH_RESULT)
    assert any(x.startswith("panel 3: a repeat of panel 2") for x in scrub_repeats(sb)) and len(sb.panels) == 4


def test_a_copy_that_adds_a_number_gives_it_to_its_twin():
    # a3f8: the copy said "...monitoring works perfectly in models"; deleting the copy deleted the qualifier
    from services.storyboarder_v2 import scrub_repeats
    panels = clean_panels()
    twin = panels[1]
    twin.dialogue = [Bubble("papi", "The Transformer uses only attention."), Bubble("kumo", "It drops recurrence entirely.")]
    copy_panel = v3_panel(3, beat="idea", dialogue=[("papi", "The Transformer uses only attention."),
                                                    ("kumo", "It drops recurrence entirely, in all 6 layers.")], claims=[2])
    copy_panel.visual = twin.visual
    panels.insert(2, copy_panel)
    for i, p in enumerate(panels, start=1):
        p.n = i
    sb = v3_storyboard(panels, CLAIMS_WITH_RESULT)
    scrub_repeats(sb)
    assert len(sb.panels) == 4 and sb.panels[1].dialogue[1].text == "It drops recurrence entirely, in all 6 layers."


# ----------------------------------------------------------------------------- round 12 fixes
def test_a_picture_that_rules_out_robots_has_no_figure_props():
    # 1000 Layer RL: "No other figures or robots are drawn" was flagged as a robot prop in every fix round
    panels = clean_panels()
    panels[0].visual = "Papi steps in beside Kumo at the desk. No other figures or robots are drawn."
    problems = lint_v3(v3_storyboard(panels, CLAIMS_WITH_RESULT), V3_PAPER_TEXT).get(1, [])
    assert not any("shaped like a person or creature" in x for x in problems)
    panels[0].visual = "Papi steps in beside Kumo. No robots, a doll on the desk."
    problems = lint_v3(v3_storyboard(panels, CLAIMS_WITH_RESULT), V3_PAPER_TEXT)[1]
    assert any("shaped like a person or creature (doll)" in x for x in problems)


def test_a_board_with_one_number_asks_for_its_comparison():
    # Driver Misconduct: "Compared to what?" got a one-row table (-1.60); 1000 Layer RL drew 762 alone
    from services.storyboarder_v2 import _story_board_issues
    lone = {"kind": "table", "title": "Next day successful deliveries", "columns": ["", "% change"],
            "rows": [{"cells": ["1 extra fake remark", "-1.60"]}]}
    assert any(x.startswith("the board shows one number (-1.60)") for x in _story_board_issues(lone))
    pair = {"kind": "table", "title": "Arm Push Easy", "columns": ["Method", "64 layers", "4 layers"],
            "rows": [{"cells": ["Scaled CRL", "762", "308"]}]}
    assert not any("one number" in x for x in _story_board_issues(pair))
    # cells bound from the ledger keep the paper's printed value: a delta or a standard error is still one cell
    cited = {"kind": "table", "title": "GSM8K", "columns": ["", "Accuracy"],
             "rows": [{"cells": ["Standard prompting", "17.9"]}, {"cells": ["Chain of thought", "56.9 (+39.0)"]}]}
    assert not any("one number" in x for x in _story_board_issues(cited))
    alone = {"kind": "table", "title": "Familiarity effect", "columns": ["", "Coefficient"],
             "rows": [{"cells": ["Familiar area", "-0.625*** (0.210)"]}]}
    assert any(x.startswith("the board shows one number") for x in _story_board_issues(alone))


async def test_a_one_number_board_moved_into_a_bubble_stays_a_scene():
    # the rule's way out ("say the number in a bubble") was undone: the dropped board came back after every round
    from services.storyboarder_v2 import StoryboarderV2
    from .fakes import FakeEngine
    from .test_story_v3 import make_cast, v3_paper
    panels = clean_panels()
    panels[2].board = {"kind": "table", "title": "WMT 2014 English-German", "columns": ["", "BLEU"],
                       "rows": [{"cells": ["Transformer (big)", "28.4"], "quote": "establishing a new state-of-the-art BLEU score of 28.4"}]}
    sb = v3_storyboard(panels, CLAIMS_WITH_RESULT)
    assert any(x.startswith("the board shows one number") for x in lint_v3(sb, V3_PAPER_TEXT)[3])
    rewrite = {"n": 3, "kind": "scene", "beat": "proof", "characters": ["nezu", "papi"], "scene": "A study.", "board": None,
               "visual": "nezu holds up one tall stack of blank cards beside a short one", "claims": [3],
               "dialogue": [{"speaker": "nezu", "text": "28.4 BLEU, more than 2.0 above the rest?"}]}
    await StoryboarderV2(FakeEngine([{"panels": [rewrite]}])).fix_v3(sb, make_cast(), v3_paper(), rounds=1)
    assert sb.panels[2].kind == "scene" and not sb.panels[2].board



def test_the_recap_ticks_a_line_that_rewords_a_drawn_claim_in_near_forms():
    # Better Leads: "Behavioural losses outweigh allocation gains" got the dot beside a chart of the loss and the gain
    from services.storyboarder_v2 import _honest_recap
    panels = clean_panels()
    panels[3].board = {"kind": "list", "title": "What the paper showed", "open": False,
                       "items": ["Improving on the best existing results in translating"]}
    sb = v3_storyboard(panels, CLAIMS_WITH_RESULT)
    _honest_recap(sb)
    assert "checked" not in sb.panels[3].board


def test_a_baseline_the_paper_names_only_baseline_asks_what_it_is():
    # Gated Attention: the lint asked to label the bar "(1) Baseline | -", the paper's row, and then flagged that label
    from services.storyboarder_v2 import _story_board_issues
    bars = [{"label": "(1) Baseline | -", "paper_label": "(1) Baseline | -", "value": "6.026", "baseline": True},
            {"label": "SDPA elementwise gate", "value": "5.761", "highlight": True}, {"label": "Value gate", "value": "5.820"}]
    found = _story_board_issues({"kind": "bar", "title": "Perplexity", "bars": bars})
    assert any("the paper's row is only" in x and "say in a few words what that system is" in x for x in found)
    bars[0]["label"] = "Standard attention (baseline)"
    assert not any("names no system" in x for x in _story_board_issues({"kind": "bar", "title": "Perplexity", "bars": bars}))


def _story_of(panels):
    """clean_panels grown to a story (DIAGRAM_MIN_PANELS or more): limits, payoff and a second proof scene before the recap."""
    filler = [v3_panel(0, beat="proof", dialogue=[("nezu", "Compared to what?")], characters=("nezu",)),
              v3_panel(0, beat="payoff", dialogue=[("kumo", "So 'it' finds 'animal' at once.")], characters=("kumo",)),
              v3_panel(0, beat="limits", dialogue=[("nezu", "Still one word at a time when it writes.")], characters=("nezu",)),
              v3_panel(0, beat="limits", dialogue=[("papi", "The decoder writes one element at a time.")], characters=("papi",))]
    out = panels[:-1] + filler + panels[-1:]
    return _renumbered(out)


def test_the_method_chapters_draw_a_diagram():
    # round 23: in a blind A/B of 24 teaching panels of round 22, the same words beside a code-drawn diagram were clearer
    # than the shipped picture in 22; idea and how scenes that teach a claim are asked to become diagram boards
    from services.storyboarder_v2 import DIAGRAM_MIN_PANELS
    panels = clean_panels()
    how = [v3_panel(3, beat="how", dialogue=[("papi", "Each word looks at every other word at once.")], claims=[2]),
           v3_panel(4, beat="how", dialogue=[("papi", "So no recurrence: all positions in parallel.")], claims=[2])]
    panels = _story_of(panels[:2] + how + panels[2:])
    assert len(panels) >= DIAGRAM_MIN_PANELS
    found = problems_of(panels)
    asked = sorted(n for n, v in found.items() if any(x.startswith("diagram: this scene teaches claim") for x in v))
    assert asked == [2, 3]   # at most two a run, the earliest first
    assert not problems_of(clean_panels()).get(2)   # a script too short for chapters is never asked
    for i in (1, 2, 3):
        panels[i].kind, panels[i].characters = "board", ["papi"]
        panels[i].board = {"kind": "diagram", "title": f"Step {i}", "brief": "the sentence's words with arrows from 'it'"}
    assert not any(x.startswith("diagram:") for v in problems_of(panels).values() for x in v)


# ----------------------------------------------------------------------------- round 13 fixes
def test_list_markers_and_citations_are_no_key_numbers():
    # Gated Attention: "two key factors: (1) introducing non-linearity ... (2) applying query-dependent sparse gating"
    # asked for the numbers 1 and 2, and the mechanism claim was placed on the results pages
    from services.storyboarder_v2 import _key_numbers
    factors = {"quote": "we attribute this effectiveness to two key factors: (1) introducing non-linearity upon the "
                        "low-rank mapping, and (2) applying query-dependent sparse gating scores"}
    assert _key_numbers(factors) == []
    for spelled in ("two key factors: 1) introducing non-linearity, and 2) applying sparse gating scores",
                    "two key factors: （1）introducing non-linearity, and （2）applying sparse gating scores"):
        assert _key_numbers({"quote": spelled}) == [], spelled
    cited = {"quote": "achieves 28.4 BLEU [12] on WMT 2014, 2.0 higher than the best (2.0 BLEU) [3, 4]"}
    assert [w for w, _ in _key_numbers(cited)] == ["28.4", "2.0"]
    assert [w for w, _ in _key_numbers({"quote": "rose from 50 to 62 (from 50 to 62) and 28.4(1) by 2. The"})] == ["50", "62", "28.4", "2"]


def _renumbered(panels):
    for i, p in enumerate(panels, start=1):
        p.n = i
    return panels


def test_a_method_chapter_with_only_a_list_draws_a_diagram():
    # Gated Attention round 13: one scene and a list of two phrases were the whole method; readers could not say how it works.
    # A story whose problem, idea and how chapters draw no diagram gets one (1000 Layers' round 23 dry run drew a flow of
    # three part names)
    panels = clean_panels()
    listed = v3_panel(2, kind="board", beat="how", claims=[1, 2], characters=("papi",),
                      board={"kind": "list", "title": "Core mechanisms", "items": ["Only attention", "No recurrence"]},
                      dialogue=[("papi", "Two parts: attention only, and no recurrence, because order comes from positions.")])
    panels = _story_of([panels[0], listed] + panels[2:])
    found = [x for x in problems_of(panels).get(2, []) if x.startswith("diagram:")]
    assert found and "draw no diagram of the mechanism, so this list has to show it" in found[0]
    panels[1].board = {"kind": "flow", "title": "Self-attention", "steps": ["Words in", "Every pair scored", "Mix by score"]}
    found = [x for x in problems_of(panels).get(2, []) if x.startswith("diagram:")]
    assert found and "so this flow has to show it" in found[0]
    panels[1].board = {"kind": "diagram", "title": "Self-attention", "brief": "every word with an arrow to every other word"}
    assert not any(x.startswith("diagram:") for v in problems_of(panels).values() for x in v)


def test_a_proof_board_of_words_asks_for_the_numbers():
    # Hivemind round 13: "High, reliable / Much lower, unreliable" where the paper reports Spearman correlations
    panels = clean_panels()
    panels[2].board = {"kind": "table", "title": "Translation quality", "columns": ["System", "Result"],
                       "rows": [{"cells": ["Transformer", "Much better"]}, {"cells": ["Recurrent models", "Worse"]}]}
    found = problems_of(panels)
    [words] = [x for x in found.get(3, []) if x.startswith("the proof board draws")]
    assert words.startswith("the proof board draws no numbers") and "make this panel a scene" not in words   # claim 3 has numbers
    assert any(x.startswith("claim 3") and "is a comparison" in x for x in found.get(3, []))   # a table of words compares nothing
    assert not any(x.startswith("the proof board draws") for found in problems_of(clean_panels()).values() for x in found)
    # the first column names systems and settings, whose digits are no result ("Qwen3-8B", "Depth 4"): only values count
    panels[2].board["rows"] = [{"cells": ["Depth 4", "Falls over"]}, {"cells": ["Qwen3-8B", "Vaults the wall"]}]
    assert any(x.startswith("the proof board draws") for x in problems_of(panels).get(3, []))
    panels[2].board["rows"] = [{"cells": ["Transformer", "Best on 8 of 10 tasks"]}]
    assert not any(x.startswith("the proof board draws") for x in problems_of(panels).get(3, []))


async def test_a_proof_board_of_words_turned_into_a_scene_stays_a_scene():
    # only a claim without numbers may leave the evidence chapter as a picture
    from services.storyboarder_v2 import StoryboarderV2
    from .fakes import FakeEngine
    from .test_story_v3 import make_cast, v3_paper
    panels = clean_panels()
    panels[2].board = {"kind": "list", "title": "How it behaves", "items": ["Translates faster", "Reads every word at once"]}
    panels[2].claims = [2]
    sb = v3_storyboard(panels, CLAIMS_WITH_RESULT)
    assert any(x.startswith("the proof board draws only words") and "make this panel a scene" in x
               for x in lint_v3(sb, V3_PAPER_TEXT).get(3, []))
    rewrite = {"n": 3, "kind": "scene", "beat": "proof", "characters": ["nezu", "papi"], "scene": "A study.", "board": None,
               "visual": "nezu holds up one tall stack of blank cards beside a short one", "claims": [2],
               "dialogue": [{"speaker": "nezu", "text": "No recurrence and no convolutions at all, and it still reads every word?"}]}
    await StoryboarderV2(FakeEngine([{"panels": [rewrite]}])).fix_v3(sb, make_cast(), v3_paper(), rounds=1)
    assert sb.panels[2].kind == "scene" and not sb.panels[2].board


def test_a_short_panel_that_adds_one_word_is_a_repeat():
    # Gated Attention round 13: "This is the attention sink pathology." after two panels of the attention sink
    panels = clean_panels()
    sink = [v3_panel(2, beat="problem", dialogue=[("papi", "This heavy anchor is the attention sink: it wastes attention on the first token.")]),
            v3_panel(3, beat="problem", dialogue=[("papi", "Look at the anchor sink grabbing attention away from the first page.")]),
            v3_panel(4, beat="problem", dialogue=[("papi", "This is the attention sink pathology.")])]
    panels = _renumbered(panels[:1] + sink + panels[1:])
    assert any("say again what panel 3 said" in x for x in problems_of(panels).get(4, []))
    panels[3].dialogue[0].text = "Long questions lose their answer pages to it."
    assert not any("say again" in x for x in problems_of(panels).get(4, []))
    panels[3].dialogue[0].text = "The attention sink takes 46.7% of it."   # its number is what it adds
    assert not any("say again" in x for x in problems_of(panels).get(4, []))
    panels[2].dialogue[0].text = "Kumo cannot find the answer pages in the dark stack."
    panels[3].dialogue[0].text = "This is the attention sink pathology."
    assert any("say again what panel 2 said" in x for x in problems_of(panels).get(4, []))   # the panel it repeats



def test_the_recap_never_ticks_background():
    # Delivery round 13: "Thin margins" got the tick "evidence shown in this manga" because a method panel repeated it
    from services.storyboarder_v2 import _honest_recap
    panels = clean_panels()
    panels[1].dialogue[0].text = ("Complex recurrent networks with an encoder and a decoder were the old way. The Transformer: "
                                  "only attention, so no recurrence and no convolutions.")
    panels[3].board = {**panels[3].board, "items": panels[3].board["items"] + [
        "Older models were complex recurrent networks with an encoder and a decoder"]}
    sb = v3_storyboard(panels, CLAIMS_WITH_RESULT)
    _honest_recap(sb)
    assert sb.panels[3].board["checked"] == [1]


async def test_the_recap_evidence_asks_the_check_model_once_and_keeps_its_answer():
    from services.storyboarder_v2 import StoryboarderV2, _honest_recap
    from .fakes import FakeEngine, PaperModel
    seen = []

    def answer(charts, lines):
        seen.append(([c["n"] for c in charts], [x["line"] for x in lines], [c.get("board", {}).get("rows", [{}])[0].get("quote") for c in charts]))
        return {"lines": [{"i": 0, "panels": [3, 2, "x"]}, {"i": 1, "panels": []}, {"i": 7, "panels": [3]}]}
    story = StoryboarderV2(FakeEngine(router=PaperModel(recap_evidence=answer)))
    sb = v3_storyboard(clean_panels(), CLAIMS_WITH_RESULT)
    await story.recap_evidence(sb)
    # the table is the one chart; its quotes stay out of what the checker reads (the reader never sees them)
    assert seen == [([3], ["Attention only", "28.4 BLEU on WMT 2014 English-German"], [None])]
    assert sb.brief["recap_evidence"] == [{"line": "Attention only", "panels": [3], "shown": True},
                                          {"line": "28.4 BLEU on WMT 2014 English-German", "panels": [], "shown": False}]
    _honest_recap(sb)
    assert sb.panels[3].board["checked"] == [0]
    # no chart or table in the story: every line gets the dot, no question asked
    seen.clear()
    plain = v3_storyboard(clean_panels()[:2] + clean_panels()[3:], CLAIMS_WITH_RESULT)
    await story.recap_evidence(plain)
    assert seen == [] and [e["panels"] for e in plain.brief["recap_evidence"]] == [[], []]
    # a failed call keeps no answer (the rule decides) and says the step was skipped
    def broken(charts, lines):
        raise RuntimeError("checker down")
    story = StoryboarderV2(FakeEngine(router=PaperModel(recap_evidence=broken)))
    sb.brief["recap_evidence"] = [{"line": "stale", "panels": [3]}]
    await story.recap_evidence(sb)
    assert "recap_evidence" not in sb.brief and any(x["stage"] == "recap evidence" for x in story.skipped)


def test_a_line_about_what_the_paper_does_not_show_is_flagged():
    # Flexibility Trap rounds 13-14: the payoff said "The paper reports aggregate benchmark gains, not a specific
    # app-download example"; the limits chapter may still say what the authors did not test
    panels = clean_panels()
    panels[1].narration = "The paper reports aggregate benchmark gains, not a specific app-download example."
    assert any("talks about what the paper does not show" in x for x in problems_of(panels).get(2, []))
    panels[1].narration = "This is not a toy example: it is the real WMT 2014 test."
    assert not any("talks about what the paper does not show" in x for x in problems_of(panels).get(2, []))
    for line in ("The paper shows 89.1% on GSM8K, not this specific example.", "The paper only shows a broken example here.",
                 "It helps on this example too, but the paper gives no exact answer.", "The paper doesn’t show this output."):
        panels[1].narration = line
        assert any(x.startswith("meta:") for x in problems_of(panels).get(2, [])), line
    for line in ("This study did not give drivers any warning before the audit.",   # the design, not a disclaimer
                 "The paper shows no accuracy drop at 4 bits."):                    # a result
        panels[1].narration = line
        assert not any(x.startswith("meta:") for x in problems_of(panels).get(2, [])), line
    panels[0].narration = "Our app story is made up: this is not a specific example from the paper."   # the hook's required line
    assert not any(x.startswith("meta:") for x in problems_of(panels).get(1, []))
    panels[0].narration = panels[1].narration = ""
    limits = v3_panel(4, beat="limits", dialogue=[("nezu", "The paper does not show results on other languages, so that stays open.")])
    panels = _renumbered(panels[:3] + [limits] + panels[3:])
    assert not any("talks about what the paper does not show" in x for found in problems_of(panels).values() for x in found)


def test_the_why_rule_asks_for_natural_reasons():
    # the old template "because ..., so ..." came back as "Because fixed paths mismatch, so we supervise where it walks."
    panels = clean_panels()
    panels[1].dialogue[0].text = "The Transformer, a simple architecture: only attention, no recurrence and no convolutions."
    [why] = [x for x in problems_of(panels).get(2, []) if x.startswith("why:")]
    assert 'never "Because X, so Y"' in why and '"because ..., so ..."' not in why
    # natural reasons count, without the comma: "first so Kumo can", ", as fixed paths do not match" (Right On Q's idea
    # panel stayed flagged through every round); "so fast" and ", as shown" are no reasons
    for line, reason in (("We fix the answer first so Kumo can always check it.", True),
                         ("It fixes the mismatch, as fixed paths do not match the student's output.", True),
                         ("The Transformer is so fast, as shown in Table 2.", False),
                         ("Attention alone does it, as the paper shows.", False)):   # a report gives no reason
        panels[1].dialogue[0].text = line
        assert (not any(x.startswith("why:") for x in problems_of(panels).get(2, []))) == reason, line


def test_a_run_on_one_claim_must_not_repeat_itself_in_new_words():
    # Right On Q round 14: idea panels 10-12 all taught claim 6 and said "Dynamic KL weights each token by the
    # teacher-student ratio" in new words each time (64-69% shared words, under the 70% repeat rule)
    panels = clean_panels()
    run = [v3_panel(2, beat="idea", claims=[2], dialogue=[("papi", "One KL direction cannot fix both errors, so Dynamic KL balances them.")]),
           v3_panel(3, beat="idea", claims=[2], dialogue=[("papi", "Dynamic KL weights each token by the teacher student probability ratio.")]),
           v3_panel(4, beat="idea", claims=[2], dialogue=[("papi", "No single KL direction fixes both errors, so Dynamic KL balances both kinds.")])]
    panels = _renumbered(panels[:1] + run + panels[1:])
    found = problems_of(panels)
    assert any(x.startswith("this panel says much of what panel 2 said") and "claim 2" in x for x in found.get(4, []))
    panels[3].dialogue[0].text = "Forward KL fixes underpredicted tokens and reverse KL the overpredicted ones."
    assert not any(x.startswith("this panel says much of") for found in problems_of(panels).values() for x in found)


# ----------------------------------------------------------------------------- round 15 fixes
def test_reasons_given_without_a_connective_count():
    # told to talk plainly, the round-15 writer gave its reasons as "make", "let", "without", "the better ..., the less"
    # and a student's "why ...?"; 8 of its 19 why-flags were such reasons, and the fixer was asked to add one again
    from services.storyboarder_v2 import WHY_RE
    for text, reason in [("Better future leads make this one less worth keeping.", True),
                         ("Residuals, layer norm, and Swish let deep networks train stably.", True),
                         ("Without these, deep networks cannot be stably scaled.", True),
                         ("The better the next leads are, the less this one matters.", True),
                         ("This avoids needing stronger teachers as the model improves.", True),
                         ("Why throw away all the easy and hard ones?", True),
                         ("This gate works for two key reasons.", True),
                         ("This fixes the weak low-rank mapping bottleneck inside attention.", True),
                         ("This cuts the attention sink by blocking wasted attention weight.", True),
                         ("It stops at the first token.", False),
                         ("Let me show you the maze.", False), ("Two things make this work.", False),
                         ("It was made of wood.", False), ("The answer is fixed first.", False)]:
        assert bool(WHY_RE.search(text)) is reason, text


def test_more_reasons_said_plainly_count():
    # round 18: 6 of the 19 why-flags left were reasons in plain words (LongZero, OnQ, Better Leads, Driver Misconduct)
    from services.storyboarder_v2 import WHY_RE
    for text, reason in [("Bad long text can't change a fixed correct answer.", True),
                         ("A single KL cannot correct both error types equally well.", True),
                         ("This routing frees human time for hot leads.", True),
                         ("Hot leads are high value, worth human effort.", True),
                         ("Simple correlations hide the true effect here.", True),
                         ("This method separates cause from coincidence.", True),
                         ("The paper says this graded pattern is hard to explain by other changes.", True),
                         ("Mixed answers mean there is actually something to learn.", True),
                         ("AI and humans work separate leads from the same pool.", False),   # "separate" names, it gives no reason
                         ("It cannot run on one GPU.", False), ("The table is worth a look.", False),
                         ("KL means distribution difference.", False),
                         ("Gating frees up memory.", True), ("Pruning can free up the GPU.", True),
                         ("The method is training-free.", False), ("It uses classifier-free guidance.", False),   # an adjective
                         ("The model is free to pick any token.", False)]:
        assert bool(WHY_RE.search(text)) is reason, text


def test_an_abbreviation_spelled_out_beside_it_or_a_listed_name_described_is_explained():
    # Right On Q ("KL, Kullback-Leibler divergence, ..."), its MATH-500 subtitle and the Flexibility Trap's list of prior
    # methods were flagged as unexplained in round 15
    from services.storyboarder_v2 import _term_explained
    for forms, text, plain, explained in [
            (["KL"], "KL, Kullback-Leibler divergence, measures probability mismatch.", "", True),
            (["LR"], "LR = learning rate, here 8.0e-3.", "", True),
            (["MATH-500"], "MATH-500: 500 math problems. Lower is better.", "Benchmark of 500 university-level math problems", True),
            (["SPG"], "d1, ESPO, SPG are prior diffusion-specific RL methods.", "", True),
            (["ESPO"], "d1, ESPO, SPG are prior diffusion-specific RL methods.", "", True),
            (["LR"], "48 layer 1.7B model, LR 8.0e-3.", "", False),
            (["BLEU"], "BLEU: 28.4 on WMT 2014.", "", False),
            (["GSM8K"], "GSM8K: 89.1% accuracy.", "", False)]:
        assert _term_explained(forms, text, plain) is explained, text


def test_a_rounded_value_shows_a_claim_number_given_as_about():
    # Better Leads round 15: the reallocation chart drew 8.1 percentage points, the claim says "about 8", and the rule
    # sent "8" to the next panel, whose chart was about call length; the board check took it off again every round
    from services.storyboarder_v2 import _shown_about
    claim = {"quote": "workers reallocated their earliest leads about 8 percentage points more often and cut calls by ~13%"}
    assert _shown_about(claim, "8", "8", {"8.1", "0.081"}) and _shown_about(claim, "13%", "13", {"-13.2"})
    assert not _shown_about(claim, "8", "8", {"8.6"})                                   # rounds to 9
    assert not _shown_about({"quote": "it gains 8 points"}, "8", "8", {"8.1"})          # an exact number stays exact


def test_the_system_a_chart_measures_against_is_said_to_be_something():
    # Right On Q and LongZero, rounds 12-15: "beats BitDistiller by 10.26" and "Enhanced HotpotQA" were never explained
    from services.storyboarder_v2 import _unexplained_rivals
    panels = clean_panels()
    bars = [{"label": "OnQ", "value": "70.68", "highlight": True},
            {"label": "BitDistiller (8B, 2-bit)", "value": "60.42", "baseline": True}]
    panels[2].board = {"kind": "bar", "title": "9-benchmark average", "subtitle": "Qwen3-8B 2-bit; higher is better", "bars": bars}
    sb = v3_storyboard(panels, CLAIMS_WITH_RESULT)
    [found] = _unexplained_rivals(sb)[3]
    assert found.startswith('term: the chart measures against "BitDistiller"') and "PAPER TEXT" in found
    panels[2].board["subtitle"] = "BitDistiller: an earlier low-bit training method; higher is better"
    assert _unexplained_rivals(v3_storyboard(panels, CLAIMS_WITH_RESULT)) == {}
    bars[1]["label"] = "Standard attention (baseline)"                      # a description names no system
    panels[2].board["subtitle"] = "Qwen3-8B 2-bit; higher is better"
    assert _unexplained_rivals(v3_storyboard(panels, CLAIMS_WITH_RESULT)) == {}


def test_a_why_question_counts_only_once_it_is_answered():
    # the review of round 15: a "why ...?" alone passed the reason rule, and the narration rule now moves the answer to
    # a later bubble or the next panel, so an idea panel could ask why and never answer
    panels = clean_panels()
    panels[1].dialogue = [b for b in panels[1].dialogue]
    panels[1].dialogue[0].text = "Why throw away recurrence?"
    panels[1].narration = ""
    assert any(x.startswith("why:") for x in problems_of(panels).get(2, []))
    panels[2].dialogue[0].text = "Recurrence is slow, so attention alone trains faster. 28.4, more than 2.0 above the rest?"
    assert not any(x.startswith("why:") for x in problems_of(panels).get(2, []))   # the next panel opens with the answer


def test_a_result_the_paper_reports_but_no_chart_draws_is_flagged_on_the_panel_that_says_it():
    # Flexibility Trap's 21.3% vs 0.6% and 1000 Layers' 8 of 10 tasks were said and never charted (rounds 21-22)
    from services.storyboarder_v2 import _uncharted_evidence
    panels = clean_panels()
    panels[1].dialogue[0].text = "ByteNet and ConvS2S were the strongest before: the Transformer, only attention, beats both."
    evidence = [{"claim": "The Transformer beats ByteNet and the ConvS2S ensemble", "numbers": "ByteNet 23.75, ConvS2S Ensemble 26.36",
                 "source": "Table 2"},
                {"claim": "English-German result", "numbers": "28.4 BLEU", "source": "Section 6"},          # on the table already
                {"claim": "A made-up result", "numbers": "77.7 points", "source": "nowhere"},               # not in the paper
                {"claim": "Hardware", "numbers": "8 GPUs, WMT 2014", "source": "Section 6"}]   # a small count and a year only
    sb = v3_storyboard(panels, CLAIMS_WITH_RESULT, brief={"evidence": evidence})
    found = _uncharted_evidence(sb, V3_PAPER_TEXT)
    assert list(found) == [2] and len(found[2]) == 1
    assert "ByteNet 23.75, ConvS2S Ensemble 26.36" in found[2][0] and '{"after": N}' in found[2][0]
    assert found[2][0] in lint_v3(sb, V3_PAPER_TEXT).get(2, [])
    # once a chart draws one of its numbers, the result counts as shown
    panels[2].board = dict(panels[2].board, rows=panels[2].board["rows"] + [{"cells": ["ByteNet", "23.75"], "quote": "ByteNet 23.75"}])
    assert _uncharted_evidence(sb, V3_PAPER_TEXT) == {}


def test_a_number_said_on_more_than_two_panels_is_flagged_where_it_comes_back():
    from services.storyboarder_v2 import _repeated_numbers
    panels = clean_panels()
    panels.insert(2, v3_panel(3, beat="proof", dialogue=[("nezu", "28.4 BLEU? Show me.")], claims=[3]))
    panels.insert(4, v3_panel(5, beat="payoff", dialogue=[("kumo", "28.4 BLEU, so it really is better.")], claims=[3]))
    for i, p in enumerate(panels, start=1):
        p.n = i
    panels[3].dialogue[0].text = "28.4, more than 2.0 above the rest?"
    sb = v3_storyboard(panels, CLAIMS_WITH_RESULT)
    found = _repeated_numbers(sb)
    assert list(found) == [5] and "28.4 again (panels 3, 4 also say it)" in found[5][0]
    # the running example's own numbers come back on purpose
    sb.running_example = "Translate 28.4 into words"
    assert _repeated_numbers(sb) == {}


def test_the_repeat_rule_never_takes_a_number_from_the_panel_whose_claim_needs_it():
    # the claim rule asks for 28.4 on a panel that teaches claim 3; the repeat rule flagged that very panel, and the two
    # rules undid each other's fix every round (round 23's review): it flags the panels that do not teach the claim
    from services.storyboarder_v2 import _repeated_numbers
    panels = clean_panels()
    panels.insert(1, v3_panel(2, beat="problem", dialogue=[("nezu", "They say 28.4 BLEU. Sure.")], claims=[]))
    panels.insert(2, v3_panel(3, beat="idea", dialogue=[("kumo", "28.4 BLEU would be a lot.")], claims=[]))
    for i, p in enumerate(panels, start=1):
        p.n = i
    sb = v3_storyboard(panels, CLAIMS_WITH_RESULT)
    found = _repeated_numbers(sb)
    assert list(found) == [3] and 5 not in found   # panel 5, the proof board of claim 3, keeps its number


@pytest.mark.parametrize("words", ["Qwen3-8B beats 41.2 pass@64 on AIME 2024", "the 1000 layer net"])
def test_names_sizes_years_and_title_numbers_are_no_repeats(words):
    from services.storyboarder_v2 import _repeated_numbers
    panels = [v3_panel(i, beat="idea", dialogue=[("papi", words)]) for i in range(1, 6)]
    sb = v3_storyboard(panels, CLAIMS_WITH_RESULT)
    sb.title = "1000 Layer Networks"
    said = " ".join(x for v in _repeated_numbers(sb).values() for x in v)
    assert not any(f"says {w} " in said for w in ("8B", "64", "2024", "1000"))
    assert ("says 41.2 " in said) == ("41.2" in words)


def test_a_huge_power_of_ten_never_crashes_the_number_rules():
    # float("1" + "0" * 360) is inf, and int(inf) raised OverflowError out of lint_v3 (round 23's review)
    from services.storyboarder_v2 import _evidence_numbers, _repeated_numbers
    panels = [v3_panel(i, beat="idea", dialogue=[("papi", "about 10^360 games")]) for i in range(1, 6)]
    _repeated_numbers(v3_storyboard(panels, CLAIMS_WITH_RESULT))
    assert _evidence_numbers("about 10^360 games") == []


def test_a_result_is_asked_for_once_and_never_from_the_hook():
    # 1000 Layers' depth-versus-width chart failed the number check every round, became a scene each time, and five one-
    # line scenes about it piled up (round 23's second dry run): each result is asked for once per book
    from services.storyboarder_v2 import _uncharted_evidence
    panels = clean_panels()
    panels[0].narration = "ByteNet and ConvS2S were the strongest translators before the Transformer arrived."
    evidence = [{"claim": "The Transformer beats ByteNet and the ConvS2S ensemble", "numbers": "ByteNet 23.75, ConvS2S Ensemble 26.36",
                 "source": "Table 2"}]
    sb = v3_storyboard(panels, CLAIMS_WITH_RESULT, brief={"evidence": evidence})
    found = _uncharted_evidence(sb, V3_PAPER_TEXT)
    assert list(found) == [3]   # the proof panel, not the hook that names the rivals
    sb.brief["chart_asked"] = ["The Transformer beats ByteNet and the ConvS2S ensemble"]
    assert _uncharted_evidence(sb, V3_PAPER_TEXT) == {}


async def test_a_chart_ask_in_a_round_without_room_asks_the_panel_itself_and_comes_back_later():
    from services.storyboarder_v2 import CHART_PANEL_ASK, StoryboarderV2
    from .fakes import FakeEngine
    from .test_story_v3 import make_cast, v3_paper
    panels = clean_panels()
    engine = FakeEngine([{"panels": []}])
    writer = StoryboarderV2(engine)
    writer.max_panels = 2   # no room
    sb = v3_storyboard(panels, CLAIMS_WITH_RESULT)
    ask = {2: [f'evidence: the paper\'s result "Beats ByteNet" (Table 2: ByteNet 23.75) is never drawn on a chart: {CHART_PANEL_ASK} '
               '({"after": N}) with a bar chart']}
    await writer.fix_v3(sb, make_cast(), v3_paper(), rounds=1, extra=ask)
    assert CHART_PANEL_ASK not in engine.prompts[0] and "no panel can be added in this round" in engine.prompts[0]
    assert "chart_asked" not in sb.brief   # nothing was added for it: a round with room asks again


async def test_a_result_counts_as_asked_once_a_chart_panel_was_added_for_it():
    # the chart may fail the number check later and become a scene: it is never asked for again (no pile-up), while a
    # result the round had no slot for stays open; a claim with '" (' in it is matched whole
    from services.storyboarder_v2 import CHART_PANEL_ASK, StoryboarderV2, _uncharted_evidence
    from .fakes import FakeEngine
    from .test_story_v3 import make_cast, v3_paper
    evidence = [{"claim": 'The "Transformer" (big) beats ByteNet', "numbers": "ByteNet 23.75", "source": "Table 2"},
                {"claim": "It trains in 3.5 days", "numbers": "3.5 days on eight GPUs", "source": "Section 6"}]
    sb = v3_storyboard(clean_panels(), CLAIMS_WITH_RESULT, brief={"evidence": evidence})
    asks = {3: [f'evidence: the paper\'s result "{e["claim"]}" ({e["source"]}: {e["numbers"]}) is never drawn on a chart: '
                f'{CHART_PANEL_ASK} ({{"after": N}}) with a bar chart' for e in evidence]}
    chart = {"after": 3, "kind": "board", "beat": "proof", "dialogue": [{"speaker": "nezu", "text": "And ByteNet?"}],
             "board": {"kind": "bar", "title": "BLEU", "bars": [{"label": "ByteNet", "value": "23.75", "quote": "ByteNet 23.75"},
                                                               {"label": "Transformer (big)", "value": "28.4", "quote": "Transformer (big) 28.4", "highlight": True}]}}
    writer = StoryboarderV2(FakeEngine([{"panels": [chart]}]))
    writer.max_panels = 10
    await writer.fix_v3(sb, make_cast(), v3_paper(), rounds=1, extra=asks)
    assert sb.brief["chart_asked"] == ['The "Transformer" (big) beats ByteNet']
    left = _uncharted_evidence(sb, V3_PAPER_TEXT)
    assert any("3.5 days" in x for v in left.values() for x in v) and not any("ByteNet" in x for v in left.values() for x in v)


async def test_a_refused_board_rest_of_a_panel_without_a_chart_ask_keeps_the_panel_whole():
    # the writer split a flagged panel and sent the rest as a board; no chart was asked there, so the first half alone
    # must not apply (the claim's details went with the refused rest)
    from services.storyboarder_v2 import StoryboarderV2
    from .fakes import FakeEngine
    from .test_story_v3 import make_cast, v3_paper
    panels = clean_panels()
    full = [b.text for b in panels[1].dialogue]
    first = {"n": 2, "kind": "scene", "beat": "idea", "dialogue": [{"speaker": "papi", "text": "The Transformer uses only attention."}],
             "visual": panels[1].visual, "shows": panels[1].shows, "claims": [1, 2]}
    rest = {"after": 2, "kind": "board", "beat": "idea", "dialogue": [{"speaker": "kumo", "text": "No recurrence at all?"}],
            "board": {"kind": "diagram", "title": "Only attention", "brief": "a row of words, every word linked to every other"}}
    writer = StoryboarderV2(FakeEngine([{"panels": [first, rest]}]))
    writer.max_panels = 2   # no room
    sb = v3_storyboard(panels, CLAIMS_WITH_RESULT)
    await writer.fix_v3(sb, make_cast(), v3_paper(), rounds=1, extra={2: ["reader: the panel never says what replaces recurrence"]})
    assert [b.text for b in sb.panels[1].dialogue] == full and len(sb.panels) == 4


def test_the_repeat_rule_reads_numbers_where_find_numbers_found_them():
    # find_numbers' offsets index its prepared copy (thousands commas dropped, "…" unfolded): "pass@64" after "1,000"
    # or an ellipsis counted as the number 64
    from services.storyboarder_v2 import _repeated_numbers
    for words in ("On 1,000 problems, pass@64 went up.", "Hmm… pass@64 went up."):
        panels = [v3_panel(i, beat="idea", dialogue=[("papi", words)]) for i in range(1, 6)]
        said = " ".join(x for v in _repeated_numbers(v3_storyboard(panels, CLAIMS_WITH_RESULT)).values() for x in v)
        assert "says 64 " not in said, words
    # the paper's title exempts its numbers; the manga's own title does not when the brief names the paper
    panels = [v3_panel(i, beat="idea", dialogue=[("papi", "28.4 BLEU again.")]) for i in range(1, 6)]
    sb = v3_storyboard(panels, CLAIMS_WITH_RESULT, brief={"title": "Attention Is All You Need"})
    sb.title = "28.4 BLEU Without Recurrence"
    assert _repeated_numbers(sb)


async def test_a_refused_chart_panel_leaves_the_panels_own_rewrite_standing():
    # asked for a chart the round had no room for, the writer lost panel 2's whole rewrite (round 23's review)
    from services.storyboarder_v2 import StoryboarderV2
    from .fakes import FakeEngine
    from .test_story_v3 import make_cast, v3_paper
    panels = clean_panels()
    rewrite = {"n": 2, "kind": "scene", "beat": "idea", "dialogue": [{"speaker": "papi", "text": "The Transformer: only attention, no recurrence and no convolutions."}],
               "visual": panels[1].visual, "shows": panels[1].shows, "claims": [1, 2]}
    chart = {"after": 2, "kind": "board", "beat": "proof", "dialogue": [{"speaker": "nezu", "text": "Show me."}],
             "board": {"kind": "bar", "title": "BLEU", "bars": [{"label": "Transformer", "value": "28.4", "quote": "28.4 BLEU"}]}}
    sb = v3_storyboard(panels, CLAIMS_WITH_RESULT)
    writer = StoryboarderV2(FakeEngine([{"panels": [rewrite, chart]}, {"panels": []}]))
    writer.max_panels = 2   # the chart cannot be added
    await writer.fix_v3(sb, make_cast(), v3_paper(), rounds=1, extra=chart_ask(2), split=True)
    assert len(sb.panels) == 4 and sb.panels[1].dialogue[0].text.startswith("The Transformer: only attention")


# ----------------------------------------------------------------------------- round 25: the bet is settled

def test_a_bet_the_story_never_settles_is_flagged_on_the_limits():
    # 15 of 16 books of rounds 22 and 23 settled the skeptic's bet in the limits; Flexibility Trap's round 23 left it open
    from services.storyboarder_v2 import _unsettled_bet
    panels = [v3_panel(1, beat="hook", dialogue=[("kumo", "Why does it leave blanks?")]),
              v3_panel(2, beat="claims", dialogue=[("nezu", "I bet arbitrary order is better."), ("kumo", "Deal.")]),
              v3_panel(3, beat="proof", dialogue=[("nezu", "It wins on every benchmark?")]),
              v3_panel(4, beat="limits", dialogue=[("nezu", "It still needs a base model.")]),
              v3_panel(5, beat="recap", dialogue=[("papi", "Here is what the paper showed.")])]
    found = _unsettled_bet(v3_storyboard(panels))
    assert list(found) == [4] and "the bet of panel 2" in found[4][0] and "I bet arbitrary order" in found[4][0]
    for settled in ("Fine, you win.", "You won. But my limits point stands.", "The tart is yours."):
        panels[3].dialogue = [Bubble("nezu", settled)]
        panels[1].dialogue = [Bubble("nezu", "Bet my berry tart this is all hype."), Bubble("kumo", "Deal.")]
        assert _unsettled_bet(v3_storyboard(panels)) == {}, settled
    panels[3].dialogue = [Bubble("nezu", "It still needs a base model.")]
    panels[2].dialogue = [Bubble("nezu", "My tart is in danger.")]   # the stake named during the proof is no settlement
    assert list(_unsettled_bet(v3_storyboard(panels))) == [4]
    panels[1].dialogue = [Bubble("nezu", "Show me the data first.")]   # no bet: nothing to settle
    assert _unsettled_bet(v3_storyboard(panels)) == {}


def test_the_bet_is_the_early_line_with_a_stake_not_an_idiom_before_it():
    from services.storyboarder_v2 import _unsettled_bet
    panels = [v3_panel(1, beat="hook", dialogue=[("kumo", "I bet it skips the hard parts.")]),   # an idiom, no wager
              v3_panel(2, beat="claims", dialogue=[("nezu", "Bet my tart this is all hype."), ("kumo", "Deal.")]),
              v3_panel(3, beat="proof", dialogue=[("nezu", "Compared to what?")]),
              v3_panel(4, beat="limits", dialogue=[("nezu", "Fine. Enjoy the tart. It still needs a base model.")]),
              v3_panel(5, beat="recap", dialogue=[("papi", "Here is what the paper showed.")])]
    assert _unsettled_bet(v3_storyboard(panels)) == {}   # the tart named in the limits settles the claims board's wager
    panels[3].dialogue = [Bubble("nezu", "It still needs a base model.")]
    found = _unsettled_bet(v3_storyboard(panels))
    assert list(found) == [4] and "the bet of panel 2" in found[4][0]   # the flag quotes the real wager
    panels[1].dialogue = [Bubble("nezu", "I better check the evidence first.")]   # "I better" is no bet
    panels[0].dialogue = [Bubble("kumo", "Why does it leave blanks?")]
    assert _unsettled_bet(v3_storyboard(panels)) == {}


def test_a_word_of_the_claims_is_no_stake_that_settles_the_bet():
    # Flexibility Trap, round 25: "I bet this tart arbitrary order wins" passed as settled wherever "arbitrary" came back
    from services.storyboarder_v2 import _unsettled_bet
    panels = [v3_panel(1, beat="hook", dialogue=[("kumo", "Why are there broken operators?")]),
              v3_panel(2, beat="claims", dialogue=[("nezu", "I bet this tart arbitrary order wins."), ("kumo", "Deal.")]),
              v3_panel(3, beat="proof", dialogue=[("nezu", "Arbitrary order finds almost nothing new.")]),
              v3_panel(4, beat="limits", dialogue=[("papi", "Arbitrary order is still good for sudoku.")]),
              v3_panel(5, beat="recap", dialogue=[("papi", "For math, arbitrary order can limit reasoning.")])]
    claims = [{"id": 0, "text": "For math and coding, arbitrary order limits reasoning potential."}]
    assert list(_unsettled_bet(v3_storyboard(panels, claims=claims))) == [4]
    panels[3].dialogue = [Bubble("nezu", "Fine, the tart is yours. Arbitrary order is still good for sudoku.")]
    assert _unsettled_bet(v3_storyboard(panels, claims=claims)) == {}
