"""services.boards: value parsing and display, spec normalization, problems, the delta badge, translatable strings and
rendering of every board kind (any size, en and zh, inside the canvas, deterministic, any BoardFonts)."""

import copy
import re

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

import services.boards as boards
from services.boards import (BOARD_KINDS, board_numbers, board_problems, board_strings, delta_badge, display_value,
                             find_numbers, normalize_board, parse_value, render_board, set_board_strings)

BLEU = {"kind": "bar", "title": "WMT 2014 English→German (BLEU)", "unit": "BLEU", "higher_is_better": True, "source": "Table 2",
        "bars": [{"label": "GNMT+RL", "value": "24.6"}, {"label": "ConvS2S", "value": "25.16"},
                 {"label": "ConvS2S ensemble", "paper_label": "ConvS2S Ensemble", "value": "26.36", "quote": "ConvS2S Ensemble 26.36",
                  "baseline": True},
                 {"label": "Transformer base", "value": "27.3"},
                 {"label": "Transformer big", "value": "28.4", "highlight": True}]}
FLOPS = {"kind": "bar", "title": "Training cost, English→French", "unit": "FLOPs", "higher_is_better": False, "source": "Table 2",
         "bars": [{"label": "GNMT+RL", "value": "1.4 · 10^20"}, {"label": "ConvS2S", "value": "1.5 · 10^20"},
                  {"label": "MoE", "value": "1.2 · 10^20", "baseline": True}, {"label": "Transformer big", "value": "2.3 · 10^19", "highlight": True}]}
TABLE = {"kind": "table", "title": "Linking two words", "columns": ["Layer type", "Sequential steps", "Longest path"],
         "rows": [{"cells": ["Self-Attention", "O(1)", "O(1)"], "quote": "Self-Attention O(n^2 · d) O(1) O(1)"},
                  {"cells": ["Recurrent", "O(n)", "O(n)"]}, {"cells": ["Convolutional", "O(1)", "O(log_k(n))"]}],
         "highlight_rows": [0], "source": "Table 1"}
FLOW = {"kind": "flow", "title": "The Transformer, end to end", "steps": ["Input words + positions", "Encoder: 6 layers",
                                                                          "Decoder: 6 layers", "Output words"],
        "highlight": [1, 2], "note": "each layer: multi-head self-attention + feed-forward"}
COMPARE = {"kind": "compare", "title": "Reading direction", "left": {"label": "GPT", "items": ["Reads left to right only"]},
           "right": {"label": "BERT", "items": ["Reads both directions at once"]}}
LIST = {"kind": "list", "title": "What the paper showed", "items": ["Only attention, no recurrence",
                                                                    "28.4 BLEU English→German", "3.5 days on 8 GPUs"]}
LINKS = {"kind": "links", "title": "Who does 'it' look at?", "tokens": ["The", "animal", "was", "tired", "so", "it", "slept"],
         "from": 5, "weights": [0.02, 0.6, 0.05, 0.2, 0.03, 0.0, 0.1], "illustrative": True}
DIAGRAM = {"kind": "diagram", "title": "Who fills the hard token first?", "brief": "the same tokens filled left to right and in any order",
           "elements": [
               {"type": "text", "x": 300, "y": 60, "text": "Left to right", "tone": "blue", "size": 36, "bold": True},
               {"type": "text", "x": 900, "y": 60, "text": "Any order", "tone": "red", "size": 36, "bold": True},
               {"type": "box", "x": 120, "y": 140, "w": 110, "h": 80, "text": "180", "tone": "ink"},
               {"type": "box", "x": 370, "y": 140, "w": 110, "h": 80, "text": "×", "tone": "red", "fill": "solid"},
               {"type": "arrow", "from": [230, 180], "to": [370, 180], "tone": "ink"},
               {"type": "box", "x": 720, "y": 140, "w": 110, "h": 80, "text": "180", "tone": "ink"},
               {"type": "box", "x": 970, "y": 140, "w": 110, "h": 80, "text": "×", "tone": "red", "dashed": True},
               {"type": "arrow", "points": [[775, 135], [900, 70], [1025, 135]], "curved": True, "tone": "red", "label": "skips it"},
               {"type": "text", "x": 1025, "y": 270, "text": "filled last", "tone": "red"},
               {"type": "mark", "x": 300, "y": 500, "mark": "check", "tone": "blue", "size": 60},
               {"type": "mark", "x": 900, "y": 500, "mark": "cross", "tone": "red", "size": 60}]}
SPECS = {"bar": BLEU, "table": TABLE, "flow": FLOW, "compare": COMPARE, "list": LIST, "links": LINKS, "diagram": DIAGRAM}


# ----------------------------------------------------------------------------- parse_value / display_value / find_numbers

@pytest.mark.parametrize("value, expected", [
    ("28.4", 28.4), ("26.30", 26.3), ("2.3 · 10^19", 2.3e19), ("1.4 × 10²⁰", 1.4e20), ("1.4·10^20", 1.4e20), ("10^19", 1e19),
    ("4.9%", 4.9), ("−0.9", -0.9), ("-0.9", -0.9), ("1,000", 1000.0), ("3.5 days", 3.5), ("5e-5", 5e-5), ("+2.04", 2.04),
    ("~3.5", 3.5), (28.4, 28.4), (7, 7.0),
])
def test_parse_value(value, expected):
    assert parse_value(value) == pytest.approx(expected)


@pytest.mark.parametrize("value", ["n/a", "", None, True, "86.7/85.9", "28.4 (+2.0)", "v1.1", "—", float("nan")])
def test_parse_value_needs_exactly_one_number(value):
    assert parse_value(value) is None


@pytest.mark.parametrize("value, shown", [
    ("10^19", "10¹⁹"), ("1.4 · 10^20", "1.4 × 10²⁰"), ("1.4·10^20", "1.4 × 10²⁰"), ("2.3 x 10^19", "2.3 × 10¹⁹"),
    ("1.4 · 10^{20} FLOPs", "1.4 × 10²⁰ FLOPs"), ("2.3e+19", "2.3 × 10¹⁹"), ("10^-5", "10⁻⁵"), ("10^19.", "10¹⁹."),
    ("28.4", "28.4"), ("26.30", "26.30"), ("4.9%", "4.9%"), ("O(n^2 · d)", "O(n² · d)"), ("O(log_k(n))", "O(logₖ(n))"),
    ("d_model", "d_model"), ("-0.9", "−0.9"), ("0.1-0.4", "0.1-0.4"), ("1.4 × 10²⁰", "1.4 × 10²⁰"), (1.2e21, "1.2 × 10²¹"),
    (26.3, "26.3"), (0, "0"),
])
def test_display_value(value, shown):
    assert display_value(value) == shown


@pytest.mark.parametrize("value", ["28.4", "26.30", "2.3 · 10^19", "1.4·10^20", "10^19", "4.9%", "-0.9", "5e-5", "1,000"])
def test_display_value_never_changes_the_number(value):
    assert parse_value(display_value(value)) == pytest.approx(parse_value(value))
    assert display_value(display_value(value)) == display_value(value)


def test_find_numbers_skips_names_and_formula_exponents():
    assert find_numbers("ConvS2S WMT14 GPT-2 P100 v1.1 BLEU-4 newstest2014 O(n^2) d_k") == []


def test_find_numbers_reads_scientific_notation_as_one_number():
    nums = find_numbers("Costs: 2.3 · 10^19 and 1.4·10^20. Also 10^19, 1.2 × 10²¹ and 5e-5.")
    assert [n.text for n in nums] == ["2.3 · 10^19", "1.4·10^20", "10^19", "1.2 × 10^21", "5e-5"]
    assert [n.exponent for n in nums] == [19, 20, 19, 21, -5]
    assert nums[0].key == "23000000000000000000"


def test_find_numbers_canonical_keys_and_short_flag():
    a, b, c = find_numbers("26.30 26.3 8")
    assert a.key == b.key == "26.3" and a.decimals == 2 and b.decimals == 1
    assert c.short and not a.short
    assert [n.key for n in find_numbers("达到28.4分，用了8张GPU")] == ["28.4", "8"]
    assert [n.key for n in find_numbers("Repeat x6, 3x faster")] == ["6", "3"]


# ----------------------------------------------------------------------------- normalize_board

def test_normalize_coerces_model_shapes():
    raw = {"type": "bar chart", "title": "  BLEU  ", "highlight": "Transformer big",
           "bars": [["GNMT", 24.6], {"name": "Transformer big", "score": 28.4, "quote": " Transformer (big) 28.4 "},
                    {"label": "Cost", "value": 2.3e19, "highlight": "no"}]}
    b = normalize_board(raw)
    assert b["kind"] == "bar" and b["title"] == "BLEU"
    assert [x["value"] for x in b["bars"]] == ["24.6", "28.4", "2.3 · 10^19"]
    assert [x["highlight"] for x in b["bars"]] == [False, True, False]
    assert b["bars"][1]["quote"] == "Transformer (big) 28.4"
    assert b["higher_is_better"] is True


def test_normalize_infers_direction_and_aliases():
    assert normalize_board({**FLOPS, "higher_is_better": None, "title": "Training cost (lower is better)"})["higher_is_better"] is False
    assert normalize_board({**FLOPS, "higher_is_better": None, "lower_is_better": "true"})["higher_is_better"] is False
    assert normalize_board({**FLOPS, "higher_is_better": "false"})["higher_is_better"] is False
    parallel = normalize_board({"kind": "bar", "title": "T", "labels": ["a", "b"], "values": ["1.5", "2.5"], "highlight": [1]})
    assert [(x["label"], x["value"], x["highlight"]) for x in parallel["bars"]] == [("a", "1.5", False), ("b", "2.5", True)]


def test_normalize_drops_bars_without_one_number_and_reports_them():
    raw = {"kind": "bar", "title": "T", "bars": [{"label": "a", "value": "n/a"}, {"label": "b", "value": "41.0/41.8"},
                                                 {"label": "c", "value": "3"}, {"label": "d", "value": "4", "highlight": True}]}
    assert [x["label"] for x in normalize_board(raw)["bars"]] == ["c", "d"]
    problems = board_problems(raw)
    assert any('"a"' in p and "dropped" in p for p in problems) and any('"b"' in p for p in problems)


def test_normalize_table_rows_in_every_shape():
    columns = ["Model", "BLEU"]
    for rows in ([["A", "1.5"], ["B", 2.5]], [{"cells": ["A", "1.5"]}, {"cells": ["B", "2.5"]}],
                 [{"Model": "A", "BLEU": "1.5"}, {"Model": "B", "BLEU": 2.5}]):
        b = normalize_board({"kind": "table", "title": "T", "columns": columns, "rows": rows})
        assert b["columns"] == columns and [r["cells"] for r in b["rows"]] == [["A", "1.5"], ["B", "2.5"]]
    loose = normalize_board({"kind": "table", "title": "T", "columns": columns, "rows": [{"model": "A", "bleu": "1.5"}, {"MODEL ": "B", "Bleu": 2.5}]})
    assert [r["cells"] for r in loose["rows"]] == [["A", "1.5"], ["B", "2.5"]]     # rows used to be dropped as empty
    inferred = normalize_board({"kind": "table", "title": "T", "rows": [{"Model": "A", "BLEU": "1.5", "quote": "A 1.5"}]})
    assert inferred["columns"] == ["Model", "BLEU"] and inferred["rows"][0] == {"cells": ["A", "1.5"], "quote": "A 1.5"}


def test_normalize_table_pads_ragged_rows_and_marks_highlights():
    raw = {"kind": "table", "title": "T", "columns": ["a", "b", "c"], "rows": [["x", "1"], {"cells": ["y", "2", "3"], "highlight": True}],
           "highlight_rows": [0, 9]}
    b = normalize_board(raw)
    assert b["rows"][0]["cells"] == ["x", "1", ""] and b["highlight_rows"] == [0, 1]
    problems = board_problems(raw)
    assert any("row 1 has 2 cells" in p for p in problems)
    assert any("highlight_rows" in p for p in problems)


def test_normalize_flow_list_compare_links_shapes():
    flow = normalize_board({"type": "pipeline", "title": "T", "steps": "Input → Encoder -> Decoder\nOutput", "highlight": "Encoder"})
    assert flow["kind"] == "flow" and flow["steps"] == ["Input", "Encoder", "Decoder", "Output"] and flow["highlight"] == [1]
    numbered = normalize_board({"kind": "flow", "title": "T", "steps": ["1. Pre-train on 3.3B words", "2) Fine-tune", {"text": "Step 3: Test"}]})
    assert numbered["steps"] == ["Pre-train on 3.3B words", "Fine-tune", "Test"]
    recap = normalize_board({"kind": "recap", "title": "T", "items": ["✅  28.4 BLEU on En-De", "• 3.5 days", "- 41.8 BLEU"]})
    assert recap["kind"] == "list" and recap["items"] == ["28.4 BLEU on En-De", "3.5 days", "41.8 BLEU"]
    vs = normalize_board({"kind": "comparison", "title": "T", "before": ["slow", "serial"], "after": {"title": "New", "points": "fast\nparallel"}})
    assert vs["left"] == {"label": "", "items": ["slow", "serial"]} and vs["right"] == {"label": "New", "items": ["fast", "parallel"]}
    links = normalize_board({"kind": "attention", "title": "T", "tokens": "the cat sat", "from": "Sat", "weights": ["10%", "60", "0"]})
    assert links["tokens"] == ["the", "cat", "sat"] and links["from"] == 2
    assert links["weights"] == [0.1667, 1.0, 0.0]        # one scale: the strongest link is 1.0
    assert links["illustrative"] is True and "source" not in links


def test_normalize_links_pads_weights_and_reports():
    raw = {"kind": "links", "title": "T", "tokens": ["a", "b", "c"], "from": 0, "weights": [0, 0.5]}
    assert normalize_board(raw)["weights"] == [0.0, 1.0, 0.0]
    assert any("2 weights for 3 tokens" in p for p in board_problems(raw))


def test_normalize_infers_kind_from_keys_when_type_is_not_a_kind():
    assert normalize_board({"type": "board", "title": "T", "bars": [{"label": "a", "value": "1"}]})["kind"] == "bar"
    assert normalize_board({"title": "T", "steps": ["a", "b"]})["kind"] == "flow"


@pytest.mark.parametrize("raw", [
    None, "a bar chart", [], {"kind": "pie", "title": "T", "items": ["a"]},
    {"kind": "bar", "title": "T", "bars": [{"label": "a", "value": "big"}]},
    {"kind": "table", "title": "T", "columns": ["a"], "rows": []},
    {"kind": "flow", "title": "T", "steps": []},
    {"kind": "compare", "title": "T", "left": {"label": "old", "items": ["x"]}},
    {"kind": "list", "title": "T", "items": []},
    {"kind": "links", "title": "T", "tokens": ["a", "b"], "weights": [0.5, 0.5]},
    {"kind": "links", "title": "T", "tokens": ["a", "b"], "from": 0, "weights": [1, 0]},
])
def test_unusable_specs_are_dropped_with_a_reason(raw):
    assert normalize_board(raw) is None
    assert board_problems(raw)
    with pytest.raises(ValueError):
        render_board(raw, "en", (400, 300))


@pytest.mark.parametrize("kind", BOARD_KINDS)
def test_normalize_is_idempotent_and_does_not_mutate(kind):
    raw = copy.deepcopy(SPECS[kind])
    once = normalize_board(raw)
    assert normalize_board(once) == once
    assert raw == SPECS[kind]


# ----------------------------------------------------------------------------- board_problems

@pytest.mark.parametrize("kind", BOARD_KINDS)
def test_sample_boards_have_no_problems(kind):
    assert board_problems(SPECS[kind]) == []


def test_problems_counts_highlight_units_title():
    many = {**BLEU, "bars": [{"label": f"m{i}", "value": f"{20 + i}.5"} for i in range(8)]}
    problems = board_problems(many)
    assert any("bars: 8" in p and "at most 7" in p for p in problems)
    assert any("no bar is highlighted" in p for p in problems)
    mixed = {**BLEU, "bars": [{"label": "a", "value": "4.9%"}, {"label": "b", "value": "5.2", "highlight": True}]}
    assert any("mix units" in p for p in board_problems(mixed))
    notation = {**BLEU, "bars": [{"label": "a", "value": "120"}, {"label": "b", "value": "2.3 · 10^19", "highlight": True}]}
    assert any("powers of ten" in p for p in board_problems(notation))
    long_title = {**LIST, "title": "This is a very long title that keeps going well past the limit of words"}
    assert any("title" in p and "too long" in p for p in board_problems(long_title))
    assert any("no title" in p for p in board_problems({**LIST, "title": ""}))


def test_problems_text_limits_latex_and_direction_conflict():
    assert any("LaTeX" in p for p in board_problems({**LIST, "items": ["$O(n^2)$ cost", "fine"]}))
    long_zh = {**FLOW, "steps": ["输入", "这是一个特别特别特别长的步骤说明文字超过了限制"]}
    assert any("step 2" in p and "characters" in p for p in board_problems(long_zh))
    conflict = {**FLOPS, "title": "Training cost (lower is better)", "higher_is_better": True}
    assert any("lower is better" in p for p in board_problems(conflict))
    assert any("steps: 8" in p for p in board_problems({**FLOW, "steps": [f"s{i}" for i in range(8)]}))
    assert any("right items: 5" in p for p in board_problems({**COMPARE, "right": {"label": "B", "items": list("abcde")}}))
    assert any("tokens: 15" in p for p in board_problems({**LINKS, "tokens": [f"t{i}" for i in range(15)], "weights": [0.1] * 15}))


# ----------------------------------------------------------------------------- delta badge

def test_delta_badge_best_highlighted_vs_the_marked_baseline():
    d = delta_badge(BLEU)
    assert d == {"text": "+2.04", "best": 4, "rival": 2, "better": True, "kind": "diff"}


def test_delta_badge_needs_a_marked_baseline():
    # Table 4: a semi-supervised Transformer against a WSJ-only parser would be a gap the paper never claims
    parsing = {"kind": "bar", "title": "Parsing F1", "bars": [{"label": "RNN Grammar, WSJ only", "value": "91.7"},
               {"label": "Transformer, WSJ only", "value": "91.3", "highlight": True},
               {"label": "Transformer, semi-supervised", "value": "92.7", "highlight": True}]}
    assert delta_badge(parsing) is None
    parsing["baseline"] = "RNN Grammar, WSJ only"               # named on the board, but two highlights: which one?
    assert delta_badge(parsing) is None
    parsing["bars"][2]["highlight"] = False                     # one result against one baseline
    assert delta_badge(parsing)["rival"] == 0 and delta_badge(parsing)["text"] == "\u22120.4"
    parsing["bars"][1]["baseline"] = True                       # a highlighted bar is never the baseline
    assert delta_badge(parsing)["rival"] == 0


@pytest.mark.parametrize("values, highlighted, higher, text, better", [
    (["40.56", "41.8"], [1], True, "+1.24", True),
    (["91.7", "92.7"], [1], True, "+1.0", True),                 # decimals: the larger of the two inputs
    (["91.7", "92.7", "91.3"], [1], True, "+1.0", True),          # the highlighted bar, not the best other
    (["91.7", "91.3"], [1], True, "−0.4", False),           # the paper's result is worse: shown honestly
    (["0.1", "0.3"], [1], True, "+0.2", True),                  # decimal arithmetic, no 0.19999; scores never become ratios
    (["74", "80.5"], [1], True, "+6.5", True),
    (["14", "3.5"], [1], False, "1/4", True),                   # lower is better and 4 times less: a ratio
    (["14", "10.5"], [1], False, "−3.5", True),                 # lower is better, a small gap: a difference
    (["3.5", "14"], [1], False, "+10.5", False),                # lower is better and worse: the difference, shown honestly
])
def test_delta_badge_math(values, highlighted, higher, text, better):
    spec = {"kind": "bar", "title": "T", "higher_is_better": higher,
            "bars": [{"label": f"b{i}", "value": v, "highlight": i in highlighted, "baseline": i == 0} for i, v in enumerate(values)]}
    d = delta_badge(spec)
    assert (d["text"], d["better"]) == (text, better)


def test_delta_badge_powers_of_ten_are_ratios():
    assert delta_badge(FLOPS) == {"text": "≈1/5.2", "best": 3, "rival": 2, "better": True, "kind": "ratio"}
    worse = {**FLOPS, "higher_is_better": True}                 # the paper's value is the worse one: no flattering ratio
    assert delta_badge(worse)["text"] == "−9.7 × 10¹⁹" and delta_badge(worse)["better"] is False


def test_delta_badge_percentages_fall_in_points():
    # Gated Attention: "It dropped from 46.7% to 4.8%?!" beside "≈1/9.7"; a share falls in points
    sink = {"kind": "bar", "title": "Attention on the first token", "higher_is_better": False, "unit": "% of total attention",
            "bars": [{"label": "Standard", "value": "46.7", "baseline": True}, {"label": "Gated", "value": "4.8", "highlight": True}]}
    assert delta_badge(sink)["text"] == "−41.9" and delta_badge(sink)["kind"] == "diff"
    assert delta_badge({**sink, "unit": ""})["text"] == "≈1/9.7"   # a cost or a time still drops as a ratio
    marked = {**sink, "unit": "", "bars": [{**b, "value": b["value"] + "%"} for b in sink["bars"]]}
    assert delta_badge(marked)["text"] == "−41.9"   # the footnote names the unit
    for unit in ("总注意力百分比", "百分点", "查询占比", "pp", "share of queries"):   # the translation keeps the badge
        assert delta_badge({**sink, "unit": unit})["text"] == "−41.9", unit
    assert delta_badge({**sink, "unit": "", "title": "Attention on first token (%)"})["text"] == "−41.9"
    assert delta_badge({**sink, "unit": "99th percentile latency (ms)"})["text"] == "≈1/9.7"   # a percentile is no share


def test_delta_badge_is_a_ratio_when_a_quote_states_the_gap_as_a_multiple():
    # 1000-layer RL: the quote read "52×" and the words said 52 times, while the badge drew "+636.4"
    depth = {"kind": "bar", "title": "Time at goal", "unit": "steps / 1000", "bars": [
        {"label": "Depth 4", "value": "12.6", "baseline": True, "quote": "Humanoid U-Maze 12.6 ±1.2 649 ±33 52×"},
        {"label": "Depth 64", "value": "649", "highlight": True}]}
    assert delta_badge(depth) == {"text": "≈51.5×", "best": 1, "rival": 0, "better": True, "kind": "ratio"}
    for quote in ("about 50 times more steps at the goal", "a 52-fold gain", "提升52倍"):
        said = {**depth, "bars": [{**depth["bars"][0], "quote": quote}, depth["bars"][1]]}
        assert delta_badge(said)["kind"] == "ratio", quote
    plain = {**depth, "bars": [{k: v for k, v in b.items() if k != "quote"} for b in depth["bars"]]}
    assert delta_badge(plain)["text"] == "+636.4"
    # titles and notes are translated, so they never decide: the Chinese chart draws the English one's badge
    assert delta_badge({**plain, "subtitle": "about 52 times more"})["text"] == "+636.4"
    zh = {**depth, "title": "到达目标的时间", "unit": "步数/1000", "bars": [{**depth["bars"][0], "label": "深度4"},
                                                                    {**depth["bars"][1], "label": "深度64"}]}
    assert delta_badge(zh)["text"] == delta_badge(depth)["text"]
    other = {**depth, "bars": [{**depth["bars"][0], "quote": "3× the width"}, depth["bars"][1]]}
    assert delta_badge(other)["text"] == "+636.4"                                   # a multiple of something else
    assert delta_badge({**depth, "unit": "% success"})["text"] == "+636.4"          # a share still falls in points
    near = {**depth, "bars": [{**depth["bars"][0], "value": "100", "quote": "baseline speed 1.00× (reference)"},
                              {**depth["bars"][1], "value": "104"}]}
    assert delta_badge(near)["text"] == "+4"                                         # a reference "1.00×" is no stated gap


@pytest.mark.parametrize("bars", [
    [("24.6", False), ("28.4", False)],                 # nothing highlighted
    [("24.6", True), ("28.4", True)],                   # nothing to compare against
    [("4.9%", False), ("5.2", True)],                   # different units
    [("120", False), ("2.3 · 10^19", True)],            # different notations
    [("26.36", False), ("26.360", True)],               # a tie
])
def test_delta_badge_omitted(bars):
    spec = {"kind": "bar", "title": "T", "bars": [{"label": f"b{i}", "value": v, "highlight": h, "baseline": i == 0}
                                                  for i, (v, h) in enumerate(bars)]}
    assert delta_badge(spec) is None


def test_delta_badge_only_for_bar_boards():
    assert delta_badge(TABLE) is None and delta_badge(None) is None


# ----------------------------------------------------------------------------- board_strings / set_board_strings / board_numbers

def test_board_strings_cover_texts_never_values():
    assert board_strings(BLEU) == [("title", BLEU["title"]), ("unit", "BLEU")] + \
        [(f"bars.{i}.label", b["label"]) for i, b in enumerate(BLEU["bars"])] + [("source", "Table 2")]
    table = dict(board_strings(TABLE))
    assert table == {"title": "Linking two words", "columns.0": "Layer type", "columns.1": "Sequential steps",
                     "columns.2": "Longest path", "rows.0.cells.0": "Self-Attention", "rows.1.cells.0": "Recurrent",
                     "rows.2.cells.0": "Convolutional", "source": "Table 1"}
    compare = dict(board_strings(COMPARE))
    assert set(compare) == {"title", "left.label", "left.items.0", "right.label", "right.items.0"}
    assert dict(board_strings(FLOW))["note"] == FLOW["note"]


@pytest.mark.parametrize("kind", BOARD_KINDS)
def test_board_strings_are_words_never_values_or_formulas(kind):
    values = {b["value"] for b in SPECS["bar"]["bars"]} | {"O(1)", "O(n)", "O(log_k(n))"}
    for path, text in board_strings(SPECS[kind]):
        assert re.search(r"[A-Za-z]{2,}|[\u4e00-\u9fff]", text) and text not in values, (path, text)
        assert not path.endswith(".value") and "quote" not in path and "paper_label" not in path


def test_set_board_strings_translates_and_guards_numbers():
    zh = set_board_strings(FLOW, {"title": "Transformer 全流程", "steps.1": "编码器：6 层", "steps.2": "解码器：六层",
                                  "steps.3": "  ", "bogus.0": "x", "note": 42})
    assert zh["title"] == "Transformer 全流程" and zh["steps"][1] == "编码器：6 层"
    assert zh["steps"][2] == "Decoder: 6 layers"     # the translation dropped the 6
    assert zh["steps"][3] == "Output words" and zh["note"] == FLOW["note"]
    assert FLOW["steps"][1] == "Encoder: 6 layers"   # input untouched
    bar = set_board_strings(BLEU, {"bars.4.value": "99", "bars.4.label": "Transformer 大模型", "unit": "BLEU 分"})
    assert bar["bars"][4]["value"] == "28.4" and bar["bars"][4]["label"] == "Transformer 大模型" and bar["unit"] == "BLEU 分"
    assert set_board_strings(None, {"title": "x"}) is None


def test_set_board_strings_round_trip():
    for spec in SPECS.values():
        assert set_board_strings(spec, dict(board_strings(spec))) == normalize_board(spec)


def test_board_numbers():
    assert board_numbers(BLEU)[2] == ("bars.2.value", "26.36", "ConvS2S Ensemble 26.36", "ConvS2S Ensemble")
    assert board_numbers(BLEU)[0] == ("bars.0.value", "24.6", "", "GNMT+RL")
    rows = board_numbers({"kind": "table", "title": "T", "columns": ["Model", "BLEU", "Days"],
                          "rows": [{"cells": ["Transformer (big)", "28.4", "3.5 days"], "quote": "q"}]})
    assert rows == [("rows.0.cells.1", "28.4", "q", "Transformer (big) days"), ("rows.0.cells.2", "3.5 days", "q", "Transformer (big) days")]
    assert [p for p, *_ in board_numbers(TABLE)] == ["rows.0.cells.1", "rows.0.cells.2", "rows.2.cells.1"]
    assert board_numbers(LIST) == []


# ----------------------------------------------------------------------------- render_board

SIZES = [(900, 640), (620, 760), (1400, 400), (300, 200), (160, 120)]


def _assert_inside(img, size):
    assert img.mode == "RGB" and img.size == size
    w, h = size
    px = img.load()
    frame = [px[x, y] for x in range(w) for y in (0, 1, h - 2, h - 1)] + [px[x, y] for y in range(h) for x in (0, 1, w - 2, w - 1)]
    assert min(min(p) for p in frame) >= 245, "something was drawn on the board's edge"
    assert img.convert("L").getextrema()[0] < 120, "nothing was drawn"


@pytest.mark.parametrize("kind", BOARD_KINDS)
@pytest.mark.parametrize("lang", ["en-US", "zh-CN"])
def test_every_kind_renders_inside_at_every_size(kind, lang):
    spec = SPECS[kind]
    if lang == "zh-CN":
        spec = set_board_strings(spec, {p: "中文" + t[:6] for p, t in board_strings(spec) if not find_numbers(t)})
    for size in SIZES:
        _assert_inside(render_board(spec, lang, size), size)


def test_render_is_deterministic():
    a, b = render_board(BLEU, "en", (640, 420)), render_board(BLEU, "en", (640, 420))
    assert a.tobytes() == b.tobytes()


def test_render_copes_with_overlong_content():
    crowded = {"kind": "bar", "title": "An extremely long chart title that cannot possibly fit on a single line of this small board",
               "subtitle": "A subtitle that is also far too long for the space it gets, so it has to wrap and shrink",
               "bars": [{"label": f"A rather long model name number {i}", "value": f"{i}.25", "highlight": i == 11} for i in range(12)],
               "source": "Table 99 of the appendix, which is a long source line"}
    for size in [(700, 480), (240, 180)]:
        _assert_inside(render_board(crowded, "en", size), size)
    words = {"kind": "list", "title": "T", "items": ["Supercalifragilisticexpialidocious" * 3, "x"]}
    _assert_inside(render_board(words, "en", (400, 300)), (400, 300))


def test_render_negative_values_and_tiny_canvas():
    neg = {"kind": "bar", "title": "Ablation (BLEU change)", "bars": [{"label": "no heads", "value": "-0.9"},
                                                                     {"label": "8 heads", "value": "0.4", "highlight": True}]}
    _assert_inside(render_board(neg, "en", (600, 400)), (600, 400))
    tiny = render_board(BLEU, "en", (20, 20))
    assert tiny.size == (20, 20) and tiny.convert("L").getextrema()[0] == 255


class _PlainFace:
    """The smallest object satisfying BoardFace."""

    def __init__(self, size):
        self.font = ImageFont.load_default(size)

    def length(self, text):
        return self.font.getlength(text)

    def draw(self, draw, xy, text, fill):
        draw.text(xy, text, font=self.font, fill=fill)


class _PlainFonts:
    def __init__(self):
        self.calls = []

    def face(self, language, size, bold=False):
        self.calls.append((language, size, bold))
        return _PlainFace(size)


@pytest.mark.parametrize("kind", BOARD_KINDS)
def test_render_with_any_board_fonts(kind):
    fonts = _PlainFonts()
    _assert_inside(render_board(SPECS[kind], "zh-CN", (700, 500), fonts), (700, 500))
    assert fonts.calls and {lang for lang, _, _ in fonts.calls} == {"zh"}
    assert any(bold for _, _, bold in fonts.calls)


def test_default_fonts_without_installed_fonts(monkeypatch):
    # the bundled fonts need nothing installed; without them as well (a broken checkout), Pillow's built-in font draws
    from services import fonts
    for f in (boards._chain, boards._face):
        f.cache_clear()
    monkeypatch.setattr(fonts, "system_cjk", lambda: ())
    try:
        face = boards.DefaultFonts().face("zh", 30)
        assert face.chain[0][0].startswith(fonts.FONT_DIR) and face.length("中文 abc") > 0
        _assert_inside(render_board(LIST, "zh", (500, 360), boards.DefaultFonts()), (500, 360))
        for f in (boards._chain, boards._face):
            f.cache_clear()
        monkeypatch.setattr(fonts, "bundled", lambda role: None)
        face = boards.DefaultFonts().face("en", 30)
        assert face.chain == (fonts.PILLOW,) and face.length("abc") > 0
        _assert_inside(render_board(LIST, "en", (500, 360), boards.DefaultFonts()), (500, 360))
    finally:
        monkeypatch.undo()
        for f in (boards._chain, boards._face):
            f.cache_clear()


def test_default_face_falls_back_for_missing_glyphs():
    face = boards.default_fonts().face("en", 40)
    if len(face.chain) < 3:
        pytest.skip("no fallback font installed")
    runs = face._runs("BLEU 中文")
    assert len(runs) >= 2 and runs[0][0] is face.fonts[0]
    img = Image.new("RGB", (400, 100), "white")
    face.draw(ImageDraw.Draw(img), (10, 10), "BLEU 中文", (0, 0, 0))
    assert img.convert("L").getextrema()[0] < 128


# ----------------------------------------------------------------------------- details found in review

def test_bullets_are_stripped_but_signs_kept():
    items = normalize_board({"kind": "list", "title": "T", "items": ["- 41.8 BLEU", "-0.9 BLEU with one head", "1. Pre-train", "3.5 days"]})["items"]
    assert items == ["41.8 BLEU", "-0.9 BLEU with one head", "Pre-train", "3.5 days"]


def test_translation_may_spell_out_a_times_count():
    zh = set_board_strings({"kind": "flow", "title": "T", "steps": ["Attention", "Repeat x6"]}, {"steps.1": "重复 6 次"})
    assert zh["steps"][1] == "重复 6 次"


def test_absurd_powers_of_ten_are_not_values():
    assert parse_value("1.0 · 10^400") is None
    assert normalize_board({"kind": "bar", "title": "T", "bars": [{"label": "a", "value": "10^400"}]}) is None


def _pen(lang):
    return boards._Pen(boards.default_fonts(), lang)


def test_zh_lines_break_at_punctuation_and_follow_kinsoku():
    pen = _pen("zh")
    text = "8 张 GPU 训练 3.5 天：只是以往训练成本的一小部分"
    width = pen.width(text, 40) * 0.62
    lines = boards._balanced(pen, text, width, 40)
    assert lines == ["8 张 GPU 训练 3.5 天：", "只是以往训练成本的一小部分"]
    for text in ("比此前最好结果（含集成）高出 2 BLEU 以上，", "每个词都用上完整上下文。真的！"):
        for w in (0.3, 0.45, 0.6):
            for line in boards._balanced(pen, text, pen.width(text, 40) * w, 40):
                assert line[0] not in "，。！）" and line[-1] not in "（"


def test_zh_lines_never_end_with_a_minus_sign():
    pen = _pen("zh")
    for text in ("费用 −9.7 × 10¹⁹ 次浮点运算", "单头注意力让 BLEU 降低 −0.9，十六头降低 −0.1", "提高 +2.04 分"):
        for frac in (0.25, 0.3, 0.35, 0.4, 0.5):
            for wrap in (boards._wrap, boards._balanced):
                lines = wrap(pen, text, pen.width(text, 40) * frac, 40)
                assert not any(l.rstrip().endswith(("−", "+")) for l in lines), lines


def test_numbers_stay_with_their_units_when_wrapping():
    pen = _pen("en")
    text = "over 2 BLEU above the best, 28.4 BLEU in 3.5 days on 8 GPUs"
    for frac in (0.3, 0.4, 0.5, 0.7):
        lines = boards._wrap(pen, text, pen.width(text, 40) * frac, 40)
        assert not any(l.endswith((" 2", " 28.4", " 3.5", " 8")) for l in lines), lines


# ----------------------------------------------------------------------------- review fixes

@pytest.mark.parametrize("value", ["10^99999999", "1.5 · 10^" + "9" * 5000, "2^10", "2.5^2", "3e" + "9" * 5000])
def test_absurd_exponents_and_other_powers_are_not_numbers(value):
    # 10^99999999 used to read as 1.0 and a 5000-digit exponent raised; 2^10 read as 2 while printed as 2¹⁰
    assert parse_value(value) is None and find_numbers(value) == []
    display_value(value)


def test_a_glued_minus_is_a_sign_as_display_value_prints_it():
    assert display_value("change -0.9") == "change −0.9" and parse_value("change -0.9") == pytest.approx(-0.9)
    assert parse_value("BLEU −0.9") == pytest.approx(-0.9) and parse_value("(-0.9)") == pytest.approx(-0.9)
    assert parse_value("1-0.9") is None                     # a range is two numbers
    spec = {"kind": "bar", "title": "T", "bars": [{"label": "a", "value": "change -0.9", "baseline": True},
                                                  {"label": "b", "value": "change 0.4", "highlight": True}]}
    assert delta_badge(spec)["text"] == "+1.3"


class _RecordingFonts(_PlainFonts):
    """Records every text run a board draws."""

    def __init__(self):
        super().__init__()
        self.drawn = []

    def face(self, language, size, bold=False):
        face, drawn = super().face(language, size, bold), self.drawn

        class Face:
            def length(self, text):
                return face.length(text)

            def draw(self, draw, xy, text, fill):
                drawn.append(text)
                face.draw(draw, xy, text, fill)

        return Face()


@pytest.mark.parametrize("spec", [
    {"kind": "list", "title": "Cost: O(n^2 · d) per layer", "items": ["Big model: 2.3 · 10^19 FLOPs", "Learning rate 1e-4, d_k = 64"]},
    {"kind": "flow", "title": "softmax(QK^T / sqrt(d_k)) V", "steps": ["Scores QK^T", "Scale by 1/sqrt(d_k)"], "note": "O(log_k(n)) hops"},
    {"kind": "table", "title": "Cost per layer", "columns": ["Layer", "Complexity"], "rows": [["Self-attention", "O(n^2 · d)"]]},
    {"kind": "compare", "title": "T", "left": {"label": "RNN", "items": ["O(n) steps"]}, "right": {"label": "Ours", "items": ["2.3 · 10^19 FLOPs"]}},
])
def test_every_drawn_text_is_printed_not_coded(spec):
    # titles, steps, items, notes and word cells used to show "10^19", "n^2", "1e-4" and "d_k" verbatim
    fonts = _RecordingFonts()
    render_board(spec, "en", (900, 600), fonts)
    drawn = " ".join(fonts.drawn)
    assert "^" not in drawn and "_k" not in drawn and "1e-4" not in drawn and "sqrt" not in drawn, drawn


def test_board_texts_lists_everything_a_reader_sees():
    texts = dict(boards.board_texts(BLEU))
    assert texts["bars.4.value"] == "28.4" and texts["bars.4.label"] == "Transformer big" and texts["source"] == "Table 2"
    assert all(not p.endswith(("quote", "paper_label")) for p in texts)
    assert dict(boards.board_texts(LINKS))["tokens.1"] == "animal"
    assert dict(boards.board_texts({"kind": "list", "title": "T", "items": ["+2.04", "x"]}))["items.0"] == "+2.04"
    for spec in SPECS.values():
        assert set(board_strings(spec)) <= set(boards.board_texts(spec))


def test_links_weights_share_one_scale_with_the_quiz_transcript():
    # abstract_quiz tells its reader which links are drawn strong; percentage weights once made every token "strong" there
    from services import abstract_quiz
    raw = {**LINKS, "weights": ["2%", "60%", "5%", "20%", "3%", "0%", "10%"]}
    b = normalize_board(raw)
    assert b["weights"] == [0.0333, 1.0, 0.0833, 0.3333, 0.05, 0.0, 0.1667] and normalize_board(b) == b
    assert normalize_board({**LINKS, "weights": [w * 100 for w in LINKS["weights"]]})["weights"] == b["weights"]
    red = [t for i, (t, w) in enumerate(zip(b["tokens"], b["weights"])) if i != b["from"] and w >= boards.LINK_TIERS[0]]
    assert red == ["animal"] and (boards.LINK_STRONG, boards.LINK_SHOWN) == (boards.LINK_TIERS[0], boards.LINK_TIERS[2])
    from services.storyboarder_v2 import PanelV2, StoryboardV2
    panel = PanelV2(n=1, chapter="how", title="", characters=["papi"], scene="", visual="", kind="board", board=raw)
    sb = StoryboardV2(title="T", summary="", theme="kumomo", language="en-US", chapters=[], panels=[panel], version=3)
    lines = abstract_quiz.transcript(sb).splitlines()
    assert any(line.startswith('- Links from "it": strong to "animal";') for line in lines), lines
    _assert_inside(render_board(raw, "en", (700, 420)), (700, 420))


def test_a_lone_dollar_is_money_not_latex():
    assert not any("LaTeX" in p for p in board_problems({**LIST, "items": ["Training cost $4.6M", "fine"]}))
    assert any("LaTeX" in p for p in board_problems({**LIST, "items": ["$O(n^2)$ cost", "fine"]}))


# ----------------------------------------------------------------------------- lettering's FontSet and letter_panel

@pytest.fixture(scope="module")
def fontset():
    from services.lettering import FontSet
    return FontSet(config={})          # built-in candidates only, independent of config/api_config.yaml


@pytest.mark.parametrize("kind", BOARD_KINDS)
def test_render_with_lettering_fontset(fontset, kind):
    zh = set_board_strings(SPECS[kind], {p: "中文" + t[:6] for p, t in board_strings(SPECS[kind]) if not find_numbers(t)})
    for spec, lang in ((SPECS[kind], "en-US"), (zh, "zh-CN")):
        for size in ((860, 520), (420, 300)):
            _assert_inside(render_board(spec, lang, size, fontset), size)


def test_render_in_worker_threads_with_a_shared_fontset_is_deterministic():
    # jobs letter panels (and so draw boards) in worker threads that share one FontSet
    from concurrent.futures import ThreadPoolExecutor
    from services.lettering import FontSet
    shared = FontSet(config={})
    work = [(SPECS[k], lang) for k in BOARD_KINDS for lang in ("en", "zh")] * 2
    with ThreadPoolExecutor(max_workers=6) as pool:
        out = list(pool.map(lambda job: render_board(job[0], job[1], (640, 400), shared).tobytes(), work))
    fresh = FontSet(config={})
    assert out == [render_board(s, lang, (640, 400), fresh).tobytes() for s, lang in work]


@pytest.mark.parametrize("lang", ["en-US", "zh-CN"])
@pytest.mark.parametrize("bubbles", [0, 3])
def test_letter_panel_draws_the_real_board(fontset, lang, bubbles):
    from services.lettering import Grounding, letter_panel
    from services.storyboarder_v2 import Bubble, PanelV2
    W, H, board = 1024, 1536, (70, 360, 954, 1080)
    art = Image.new("RGB", (W, H), (238, 232, 220))
    ImageDraw.Draw(art).rectangle(board, fill=(252, 252, 252), outline=(70, 70, 70), width=6)
    lines = [("nezu", "28.4 BLEU, a single model?"), ("papi", "Yes, a single model."), ("nezu", "It beats the ensembles!")]
    panel = PanelV2(n=1, chapter="main", title="", characters=["nezu", "papi"], scene="", visual="", kind="board", board=BLEU,
                    dialogue=[Bubble(s, t) for s, t in lines[:bubbles]], narration="BLEU compares a translation with a human one.")
    quad = [(board[0], board[1]), (board[2], board[1]), (board[2], board[3]), (board[0], board[3])]
    img, layout = letter_panel(art, panel, Grounding(size=(W, H), heads={}, board=quad), lang, fontset)
    assert layout["chart_placement"] == "board"
    x1, y1, x2, y2 = layout["chart"]
    chart = img.crop((x1, y1, x2, y2)).convert("RGB")
    assert chart.getextrema()[0][0] < 100                                  # ink: the board was drawn
    px = np.asarray(chart).reshape(-1, 3).astype(int)
    reds = int(((px[:, 0] > 180) & (px[:, 1] < 110) & (px[:, 2] < 100)).sum())
    assert reds > 50                                                       # the paper's bars in accent red


def test_an_open_list_draws_empty_boxes_for_claims_still_to_check():
    spec = {"kind": "list", "title": "What the paper claims", "items": ["Attention only", "28.4 BLEU English→German"], "open": True}
    assert normalize_board(spec)["open"] is True and "open" not in normalize_board({**spec, "open": False})
    opened = np.asarray(render_board(spec, "en", (900, 640)).convert("RGB")).astype(int)
    checked = np.asarray(render_board({**spec, "open": False}, "en", (900, 640)).convert("RGB")).astype(int)
    red = lambda a: int(((a[..., 0] > 170) & (a[..., 1] < 110) & (a[..., 2] < 110)).sum())
    assert red(checked) > 50 and red(opened) == 0          # red check marks only once the story has checked the claims


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_a_recap_with_ticks_and_bullets_draws_a_legend(lang):
    spec = {"kind": "list", "title": "What the paper showed", "items": ["Attention only", "41.8 BLEU English→French"]}
    plain = np.asarray(render_board(spec, lang, (900, 640)).convert("L")).astype(int)
    mixed = np.asarray(render_board({**spec, "checked": [0]}, lang, (900, 640)).convert("L")).astype(int)
    bottom = slice(int(640 * 0.86), 640)
    assert (plain[bottom] < 200).sum() == 0 and (mixed[bottom] < 200).sum() > 200   # the legend's grey words on the bottom line


def test_magnitude_suffixes_read_as_the_same_number():
    # the writer simplifies "31.0 · 10^6" to "31.0M": the same value, so the evidence check and the badge must agree
    assert [n.key for n in find_numbers("31.0M steps, 2.1 million, 40k sentences, 31.0 · 10^6")] == ["31000000", "2100000", "40000", "31000000"]
    assert [n.text for n in find_numbers("16MB of memory, Model B")] == ["16"]                # units and names are not magnitudes
    steps = {"kind": "bar", "title": "Training steps", "higher_is_better": False,
             "bars": [{"label": "Inception", "value": "31.0M", "baseline": True}, {"label": "BN-x5", "value": "2.1M", "highlight": True}]}
    assert delta_badge(steps)["text"] == "≈1/14.8"                                            # "14 times fewer" in the paper


def test_a_footnote_mark_does_not_hide_a_table_value_and_case_does_not_change_magnitudes():
    # CoT Table 1 prints "Prior best N/A (finetuning) 55^(a)": 55 with footnote (a), not 55 to some power
    assert [n.key for n in find_numbers("Prior best N/A (finetuning) 55^(a) 57.4^(b)")] == ["55", "57.4"]
    assert find_numbers("2^10") == [] and [n.key for n in find_numbers("10^6")] == ["1000000"]
    # the evidence check lower-cases the paper ("palm 540b"); a bubble keeps "PaLM 540B": both read as the same number
    assert find_numbers("palm 540b")[0].key == find_numbers("PaLM 540B")[0].key


def test_the_bar_footnote_names_the_unit_and_what_the_badge_measures():
    from services.boards import _bar_caption
    spec = normalize_board({"kind": "bar", "title": "Attention on the first token", "unit": "%", "higher_is_better": False,
                            "bars": [{"label": "Baseline", "value": "46.7", "baseline": True},
                                     {"label": "SDPA Gated", "value": "4.8", "highlight": True}]})
    badge = delta_badge(spec)
    assert _bar_caption(spec, "en", badge) == f"unit: % · lower is better · {badge['text']} vs Baseline"
    assert _bar_caption(spec, "zh", badge) == f"单位：% · 越低越好 · {badge['text']} 相对 Baseline"
    assert _bar_caption({**spec, "unit": "BLEU"}, "en") == "BLEU · lower is better"   # a named metric stays as it is


def test_subtitles_and_footnotes_are_dark_enough_to_read():
    """Readers found the grey small print too faint: subtitles and footnotes reach the 4.5:1 contrast of small text."""
    def luminance(rgb):
        ch = [c / 255 for c in rgb]
        ch = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in ch]
        return 0.2126 * ch[0] + 0.7152 * ch[1] + 0.0722 * ch[2]

    def contrast(rgb):
        return 1.05 / (luminance(rgb) + 0.05)

    assert contrast(boards.NOTE_GREY) >= 6.5 and contrast(boards.GREY) < 4.5   # the bars keep their softer grey
    img = boards.render_board({"kind": "bar", "title": "BLEU", "subtitle": "how close to a human translation; higher is better",
                               "bars": [{"label": "A", "value": "26.4", "baseline": True}, {"label": "B", "value": "28.4", "highlight": True}],
                               "source": "Table 2"}, "en-US", (1000, 700))
    px = np.asarray(img.convert("RGB")).reshape(-1, 3)
    assert (np.abs(px - np.array(boards.NOTE_GREY)).sum(axis=1) < 30).sum() > 200   # the subtitle and source are drawn in it


def test_the_recap_legend_fits_the_footer():
    # the English legend was 1.3x the footer and lost the dot's meaning to an ellipsis
    from PIL import Image
    for lang in ("en", "zh"):
        for w, h in ((808, 464), (896, 408), (864, 616)):
            c = boards._Ctx(Image.new("RGB", (w * boards.SS, h * boards.SS)), boards._Pen(boards.default_fonts(), lang), 9)
            assert c.pen.width(boards._TXT[lang]["checked"], c.px(24 * c.u)) <= c.W - 2 * c.m, (lang, w, h)


def test_significance_legends_are_not_markdown():
    # econ_ai_service's legend "* <10%, *** <1%" was flagged as markdown in every round and stayed on the board
    for legend in ("Stars: * <10%, *** <1% chance of luck", "***: p<0.01", "*** = p<0.01", "p<0.10 (*), p<0.05 (**), p<0.01 (***)",
                   "*** significant at the 1% level", "*** p<0.01: very unlikely to be chance"):
        board = {"kind": "table", "title": "Effects", "subtitle": legend, "columns": ["Call", "Effect"],
                 "rows": [{"cells": ["1-3", "0.234***"]}, {"cells": ["4-6", "0.081*"]}]}
        assert not any("markdown" in x for x in board_problems(board)), legend
    board = {"kind": "table", "title": "Effects", "subtitle": "a **bold** claim", "columns": ["Call", "Effect"],
             "rows": [{"cells": ["1-3", "0.234"]}, {"cells": ["4-6", "0.081"]}]}
    assert any("markdown" in x for x in board_problems(board))


def test_a_board_that_calls_lower_worse_cannot_print_lower_is_better():
    # "lower = less effort (backfire)" over a printed "lower is better", two rounds of econ_ai_service
    board = {"kind": "bar", "title": "Call duration change", "subtitle": "lower = less effort (backfire)", "higher_is_better": False,
             "bars": [{"label": "Calls 1-3", "value": "-0.145", "highlight": True}, {"label": "Calls 7-9", "value": "-0.128"}]}
    assert any("set higher_is_better to true" in x for x in board_problems(board))
    board["higher_is_better"] = True
    assert not any("higher_is_better" in x for x in board_problems(board))


def test_a_board_where_lower_is_good_is_left_alone():
    for subtitle in ("lower error = less harm", "Harmful reply rate; lower = fewer harmful replies", "lower is better, higher is worse"):
        board = {"kind": "bar", "title": "Error", "subtitle": subtitle, "higher_is_better": False,
                 "bars": [{"label": "A", "value": "3.1", "highlight": True}, {"label": "B", "value": "4.2"}]}
        assert not any("set higher_is_better to true" in x for x in board_problems(board)), subtitle


def test_a_chart_of_changes_prints_no_better_direction():
    # "lower is better" under call-duration changes, while the story called shorter calls lost effort (four rounds)
    from services.boards import _bar_caption
    board = boards.normalize_board({"kind": "bar", "title": "Call duration change", "subtitle": "lower = shorter calls", "higher_is_better": False,
                                    "bars": [{"label": "Calls 1-3", "value": "-0.145", "highlight": True}, {"label": "Calls 4-6", "value": "-0.009"},
                                             {"label": "Calls 81-120", "value": "0.003"}]})
    assert "better" not in _bar_caption(board, "en") and "越" not in _bar_caption(board, "zh")
    scores = boards.normalize_board({"kind": "bar", "title": "BLEU", "bars": [{"label": "Transformer", "value": "28.4", "highlight": True},
                                                                              {"label": "ConvS2S", "value": "26.4"}]})
    assert "higher is better" in _bar_caption(scores, "en")


def test_a_trend_over_model_sizes_keeps_its_direction():
    from services.boards import _bar_caption
    board = boards.normalize_board({"kind": "bar", "title": "Perplexity by size", "higher_is_better": False,
                                    "bars": [{"label": "7B", "value": "9.1"}, {"label": "13B", "value": "8.2"},
                                             {"label": "70B", "value": "6.4", "highlight": True}]})
    assert "lower is better" in _bar_caption(board, "en")
