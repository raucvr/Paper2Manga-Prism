"""The v3 story layer with the evidence ledger: citations of table cells are bound as soon as a reply arrives, lint
checks one column per chart and its strongest earlier system, and nothing unresolved reaches the reader."""

from __future__ import annotations

from services import ledger as L
from services.storyboarder_v2 import (
    StoryboarderV2,
    _board_check_items,
    _normalize_v3_panels,
    fix_prompt,
    lint_v3,
    number_lines,
    scrub_unsupported,
)

from .fakes import V3_PAPER_TEXT, FakeEngine, PaperModel, make_cast
from .test_story_v3 import CLAIMS_WITH_RESULT, v3_panel, v3_paper, v3_storyboard

TABLE2 = {"label": "Table 2", "page": 8, "caption": "Table 2: BLEU scores on the newstest2014 tests.",
          "columns": [["BLEU", "EN-DE"], ["BLEU", "EN-FR"]],
          "rows": [{"label": "ByteNet [18]", "group": 0, "cells": ["23.75", ""]},
                   {"label": "MoE [32]", "group": 0, "cells": ["26.03", "40.56"]},
                   {"label": "ConvS2S Ensemble [9]", "group": 1, "cells": ["26.36", "41.29"]},
                   {"label": "Transformer (base model)", "group": 2, "cells": ["27.3", "38.1"]},
                   {"label": "Transformer (big)", "group": 2, "cells": ["28.4", "41.8"]}]}
PAPER = L.with_ledger(V3_PAPER_TEXT, L.Ledger.from_tables([TABLE2]))
LEDGER = L.ledger_of(PAPER)


def chart(*bars, subtitle: str = "BLEU: higher is better; ensembles included"):
    return {"kind": "bar", "title": "English-German BLEU", "subtitle": subtitle, "unit": "BLEU", "bars": list(bars)}


def board_panel(board, line="It beats every earlier system here."):
    return v3_panel(1, kind="board", beat="proof", board=board, dialogue=[("nezu", line)], claims=[3], characters=("nezu", "papi"))


def test_replies_are_bound_before_anything_reads_them():
    panel = board_panel(chart({"label": "ConvS2S Ensemble", "value": "@T2.r3.c1", "baseline": True},
                              {"label": "Transformer big", "value": "@T2.r5.c1", "highlight": True},
                              {"label": "ByteNet", "value": "23.75"}),
                        line="@T2.r5.c1 BLEU against @T2.r3.c1!")
    _normalize_v3_panels([panel], make_cast(), CLAIMS_WITH_RESULT, LEDGER)
    bars = panel.board["bars"]
    assert [(b["value"], b["cell"]) for b in bars] == [("26.36", "T2.r3.c1"), ("28.4", "T2.r5.c1"), ("23.75", "T2.r1.c1")]
    assert panel.board["source"] == "Table 2" and panel.dialogue[0].text == "28.4 BLEU against 26.36!"


def test_a_citation_that_ends_a_sentence_is_bound_or_scrubbed():
    # LongZero round 12 drew "Base model scores @T2.r2.c6." in a bubble: the full stop hid the citation from the binder
    # and from the final scrub
    panel = board_panel(chart({"label": "Transformer big", "value": "@T2.r5.c1", "highlight": True},
                              {"label": "ConvS2S Ensemble", "value": "@T2.r3.c1", "baseline": True}),
                        line="The big model scores @T2.r5.c1. Look at @T2.r9.c1.")
    _normalize_v3_panels([panel], make_cast(), CLAIMS_WITH_RESULT, LEDGER)
    assert panel.dialogue[0].text == "The big model scores 28.4. Look at @T2.r9.c1."
    problems = lint_v3(v3_storyboard([panel], CLAIMS_WITH_RESULT), PAPER)[1]
    assert any("@T2.r9.c1" in x and x.startswith("evidence: bubble 0") for x in problems)
    scrub_unsupported(v3_storyboard([panel], CLAIMS_WITH_RESULT), PAPER)
    assert "@" not in panel.dialogue[0].text


def test_lint_reports_citations_that_name_nothing():
    panel = board_panel(chart({"label": "Transformer big", "value": "@T2.r5.c1", "highlight": True},
                              {"label": "ConvS2S Ensemble", "value": "@T2.r3.c1", "baseline": True},
                              {"label": "Ghost", "value": "@T2.r9.c1"}),
                        line="Look at @T2.r1.c2 there.")
    _normalize_v3_panels([panel], make_cast(), CLAIMS_WITH_RESULT, LEDGER)
    problems = lint_v3(v3_storyboard([panel], CLAIMS_WITH_RESULT), PAPER)[1]
    assert any("@T2.r9.c1" in x and x.startswith("evidence: the board cites") for x in problems)
    assert any("@T2.r1.c2" in x and x.startswith("evidence: bubble 0") for x in problems)


def test_lint_wants_one_column_per_chart():
    panel = board_panel(chart({"label": "Transformer big EN-DE", "value": "@T2.r5.c1", "highlight": True},
                              {"label": "ConvS2S Ensemble EN-FR", "value": "@T2.r3.c2", "baseline": True}))
    _normalize_v3_panels([panel], make_cast(), CLAIMS_WITH_RESULT, LEDGER)
    problems = lint_v3(v3_storyboard([panel], CLAIMS_WITH_RESULT), PAPER)[1]
    assert any("different table columns" in x and "c1" in x and "c2" in x for x in problems)


def test_lint_wants_the_strongest_earlier_system_of_the_column_on_the_chart():
    # single models only: MoE (26.03) is the strongest earlier single system and it is missing
    panel = board_panel(chart({"label": "ByteNet", "value": "@T2.r1.c1", "baseline": True},
                              {"label": "Transformer big", "value": "@T2.r5.c1", "highlight": True},
                              {"label": "Transformer base", "value": "@T2.r4.c1"}, subtitle="BLEU: higher is better"))
    _normalize_v3_panels([panel], make_cast(), CLAIMS_WITH_RESULT, LEDGER)
    problems = lint_v3(v3_storyboard([panel], CLAIMS_WITH_RESULT), PAPER).get(1, [])
    rival = [x for x in problems if "strongest earlier system" in x]
    assert len(rival) == 1 and '"MoE"' in rival[0] and "@T2.r2.c1" in rival[0]
    # with MoE on the chart the rule is met, even when the claim keeps ByteNet as its baseline
    panel.board["bars"].append({"label": "MoE", "value": "26.03", "cell": "T2.r2.c1"})
    problems = lint_v3(v3_storyboard([panel], CLAIMS_WITH_RESULT), PAPER).get(1, [])
    assert not [x for x in problems if "strongest earlier system" in x]


def test_scrub_never_lets_a_citation_reach_the_reader():
    panel = board_panel(chart({"label": "Transformer big", "value": "@T2.r5.c1", "highlight": True},
                              {"label": "ConvS2S Ensemble", "value": "@T2.r3.c1", "baseline": True}),
                        line="Wow. It scored @T7.r1.c1 points.")
    _normalize_v3_panels([panel], make_cast(), CLAIMS_WITH_RESULT, LEDGER)
    removed = scrub_unsupported(v3_storyboard([panel], CLAIMS_WITH_RESULT), PAPER)
    assert panel.dialogue[0].text == "Wow." and removed == ["panel 1: It scored @T7.r1.c1 points."]


def test_number_lines_show_the_cells_first():
    lines = number_lines(CLAIMS_WITH_RESULT, PAPER)
    first = next(ln for ln in lines.splitlines() if ln.startswith("28.4:"))
    assert first.startswith('28.4: @T2.r5.c1 = 28.4 (Table 2, "Transformer (big)", BLEU › EN-DE)')
    assert number_lines(CLAIMS_WITH_RESULT, V3_PAPER_TEXT).startswith("28.4:")   # no ledger: the paper lines as before


def test_fix_prompt_carries_the_tables_when_the_paper_is_left_out():
    sb = v3_storyboard([board_panel(chart({"label": "a", "value": "28.4", "highlight": True}))], CLAIMS_WITH_RESULT)
    prompt = fix_prompt(sb, make_cast(), {1: ["2 speech bubbles"]}, "", L.section(PAPER))
    assert L.LEDGER_HEAD in prompt and "[T2.r5]" in prompt and "\nPAPER TEXT:\n" not in prompt


def test_fact_check_sees_where_a_cited_value_comes_from():
    panel = board_panel(chart({"label": "Transformer big", "value": "@T2.r5.c1", "highlight": True},
                              {"label": "ConvS2S Ensemble", "value": "@T2.r3.c1", "baseline": True}))
    _normalize_v3_panels([panel], make_cast(), CLAIMS_WITH_RESULT, LEDGER)
    whole = _board_check_items(panel, whole=True, paper_text=PAPER)[0]["text"]
    assert 'Table 2, column \\"BLEU › EN-DE\\", row \\"Transformer (big)\\"' in whole and '"cell"' not in whole
    # ...and what that table reports: 1000 Layers drew Table 4's training hours as "Extreme Depth Performance"
    assert '(the table: \\"BLEU scores on the newstest2014 tests.\\")' in whole


async def test_build_v3_draws_cited_cells_with_their_exact_numbers():
    cited = chart({"label": "Transformer (big)", "value": "@T2.r5.c1", "highlight": True},
                  {"label": "ConvS2S Ensemble", "value": "@T2.r3.c1", "baseline": True},
                  {"label": "MoE", "value": "@T2.r2.c1"},
                  {"label": "ByteNet", "value": "@T2.r1.c1"})
    cited["subtitle"] = "ConvS2S: an earlier convolutional translation model"   # the system the gap is measured against

    def story(payload):
        payload["panels"][5]["board"] = cited
        return payload

    model = PaperModel(story=story, fix=lambda panel, problems: panel)
    sb = await StoryboarderV2(FakeEngine(router=model)).build_v3(v3_paper(PAPER), make_cast(), "en-US", (16, 32))
    board = sb.panels[5].board
    assert [b["value"] for b in board["bars"]] == ["28.4", "26.36", "26.03", "23.75"]
    assert board["source"] == "Table 2" and "unbound" not in board
    assert "@" not in " ".join(b.text for p in sb.panels for b in p.dialogue)
    assert sb.brief["lint"] == {}


def test_lint_flags_a_bar_from_outside_the_cited_column():
    panel = board_panel(chart({"label": "MoE", "value": "@T2.r2.c1"},
                              {"label": "Leaderboard entry", "value": "27.9", "quote": "x"},
                              {"label": "Transformer big", "value": "@T2.r5.c1", "highlight": True}))
    _normalize_v3_panels([panel], make_cast(), CLAIMS_WITH_RESULT, LEDGER)
    problems = lint_v3(v3_storyboard([panel], CLAIMS_WITH_RESULT), PAPER)[1]
    assert any("are not in the column the other bars cite" in x and "Leaderboard entry" in x for x in problems)
    # a bar whose value is in the column (its name just did not match) is fine
    panel = board_panel(chart({"label": "MoE", "value": "@T2.r2.c1"}, {"label": "ConvS2S Ens.", "value": "26.36"},
                              {"label": "Transformer big", "value": "@T2.r5.c1", "highlight": True}))
    _normalize_v3_panels([panel], make_cast(), CLAIMS_WITH_RESULT, LEDGER)
    problems = lint_v3(v3_storyboard([panel], CLAIMS_WITH_RESULT), PAPER).get(1, [])
    assert not any("are not in the column" in x for x in problems)


TABLE_BINS = {"label": "Table 4", "page": 9, "caption": "Table 4: Effect of the AI expansion by call position.",
              "columns": [["(1)", "1–3"], ["(2)", "4–6"], ["(3)", "7–9"]],
              "rows": [{"label": "After AI Expansion", "group": 0, "cells": ["0.234*** (0.033)", "0.157*** (0.033)", "0.117*** (0.043)"]},
                       {"label": "Observations", "group": 1, "cells": ["23,085", "21,383", "20,504"]}]}


def test_a_trend_over_the_bins_of_one_table_row_is_one_chart():
    # the board check asks for a trend across call positions as a bar chart; one row of bins is one series, not a field
    paper = L.with_ledger(V3_PAPER_TEXT, L.Ledger.from_tables([TABLE2, TABLE_BINS]))
    ledger = L.ledger_of(paper)
    trend = {"kind": "bar", "title": "Hot-lead chance by call position", "subtitle": "change in probability", "unit": "probability",
             "bars": [{"label": "Calls 1-3", "value": "@T4.r1.c1", "highlight": True}, {"label": "Calls 4-6", "value": "@T4.r1.c2"},
                      {"label": "Calls 7-9", "value": "@T4.r1.c3"}]}
    panel = board_panel(trend, "Biggest at the first calls.")
    _normalize_v3_panels([panel], make_cast(), CLAIMS_WITH_RESULT, ledger)
    assert not any("different table columns" in x for x in lint_v3(v3_storyboard([panel], CLAIMS_WITH_RESULT), paper).get(1, []))
    mixed = board_panel(chart({"label": "Transformer big EN-DE", "value": "@T2.r5.c1", "highlight": True},   # two metrics: still one column
                              {"label": "Transformer big EN-FR", "value": "@T2.r5.c2"}))
    _normalize_v3_panels([mixed], make_cast(), CLAIMS_WITH_RESULT, ledger)
    assert any("different table columns" in x for x in lint_v3(v3_storyboard([mixed], CLAIMS_WITH_RESULT), paper)[1])
