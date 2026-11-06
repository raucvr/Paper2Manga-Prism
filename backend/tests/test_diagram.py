"""services.diagram (the diagram board: spec, layout lint, reading order, drawing) and how the script pipeline designs,
keeps and redraws diagrams (StoryboarderV2.design_diagrams, _kept_drawing, _prompt_board, board_text)."""
from __future__ import annotations

import copy
import json

import pytest

from services import diagram as D
from services.boards import (board_problems, board_strings, board_texts, diagram_problems, normalize_board, render_board,
                             set_board_strings)
from services.storyboarder_v2 import (StoryboarderV2, _diagram_fallback, _diagram_key, _kept_drawing, _prompt_board, board_text,
                                      diagram_prompt)

from .fakes import FakeEngine
from .test_story_v3 import v3_panel, v3_paper, v3_storyboard

TWO_ROWS = {"kind": "diagram", "title": "Answer first, then the long text", "brief": "the answer is fixed before the text",
            "elements": [
                {"type": "box", "x": 80, "y": 300, "w": 380, "h": 140, "text": "Question + answer", "tone": "red"},
                {"type": "arrow", "from": [460, 370], "to": [700, 370], "tone": "ink", "label": "expand"},
                {"type": "box", "x": 700, "y": 220, "w": 360, "h": 300, "text": "Long documents", "tone": "ink"},
                {"type": "text", "x": 600, "y": 80, "text": "LONGZERO", "tone": "red", "size": 40, "bold": True}]}


def norm(spec):
    b = normalize_board(spec)
    assert b is not None
    return b


# ----------------------------------------------------------------------------- spec

def test_elements_are_normalized_from_the_shapes_a_designer_writes():
    b = norm({"kind": "diagram", "title": "T", "elements": [
        {"type": "rect", "x": "10", "y": 20, "width": 100, "height": 50, "label": "A", "color": "#d0342c", "fill": "#fde0de"},
        {"type": "label", "x": 300, "y": 40, "text": "B", "align": "left", "font-size": "28px"},
        {"type": "connector", "from": {"x": 110, "y": 45}, "to": [300, 45], "via": [[200, 45]], "stroke": "#2a6fb0"},
        {"type": "tick", "x": 500, "y": 300},
        {"type": "ellipse", "cx": 600, "cy": 400, "radius": 30, "text": "+", "tone": "highlight", "fill": "solid"}]})
    box, text, arrow, mark, circle = b["elements"]
    assert box == {"type": "box", "tone": "red", "text": "A", "x": 10.0, "y": 20.0, "w": 100.0, "h": 50.0, "fill": "tint", "size": 38.0}
    assert text["anchor"] == "start" and text["size"] == 32.0 and text["tone"] == "ink"   # 28 px is below the readable floor
    assert arrow["points"] == [[110.0, 45.0], [200.0, 45.0], [300.0, 45.0]] and arrow["tone"] == "blue" and arrow["heads"] == "end"
    assert mark["mark"] == "check" and circle["tone"] == "red" and circle["fill"] == "solid" and circle["r"] == 30.0
    assert normalize_board(b) == b   # idempotent


def test_unusable_elements_are_dropped_with_a_reason_and_a_brief_alone_waits_for_its_design():
    notes = []
    out = D.normalize({"elements": [{"type": "box", "x": 1}, {"type": "pie"}, "x", {"type": "text", "x": 1, "y": 2}]}, {}, notes)
    assert out is None and len(notes) == 5   # four bad elements, and nothing left to draw or design
    waiting = norm({"kind": "diagram", "title": "T", "brief": "two token rows"})
    assert waiting["elements"] == [] and waiting["brief"] == "two token rows" and not D.designed(waiting)
    assert D.designed(norm(TWO_ROWS))
    assert board_problems({"kind": "diagram", "title": "T"})   # neither elements nor a brief: the writer is told


def test_a_flow_once_written_as_a_diagram_still_reads_as_a_flow():
    b = norm({"kind": "diagram", "title": "T", "steps": ["a", "b"]})
    assert b["kind"] == "flow" and b["steps"] == ["a", "b"]


def test_the_brief_and_the_geometry_are_never_drawn_text():
    b = norm(TWO_ROWS)
    texts = dict(board_texts(b))
    assert "the answer is fixed before the text" not in texts.values()
    # reading order: the title, then rows top to bottom, left to right
    assert [t for _, t in board_texts(b)] == ["Answer first, then the long text", "LONGZERO", "Question + answer", "expand",
                                              "Long documents"]
    view = board_text(b)
    assert "Question + answer" in view and "box" not in view.split() and "brief" not in view and "300" not in view


def test_translation_replaces_the_labels_and_keeps_the_numbers():
    spec = copy.deepcopy(TWO_ROWS)
    spec["elements"].append({"type": "text", "x": 600, "y": 700, "text": "26,070 open queries", "tone": "ink"})
    paths = dict(board_strings(spec))
    zh = {p: "中文" + t for p, t in paths.items()}
    zh[next(p for p, t in paths.items() if t.startswith("26,070"))] = "26,070 个开放问题"
    out = set_board_strings(spec, zh)
    labels = [e.get("text") or e.get("label") for e in out["elements"]]
    assert "中文Question + answer" in labels and "26,070 个开放问题" in labels
    assert out["elements"][0]["x"] == 80.0   # the drawing stays where it was


# ----------------------------------------------------------------------------- the layout lint

def test_a_clean_diagram_passes_the_lint():
    assert diagram_problems(norm(TWO_ROWS)) == []
    assert diagram_problems(norm({"kind": "diagram", "title": "T", "brief": "b"})) == []   # not designed yet: nothing to measure


@pytest.mark.parametrize("element,expect", [
    ({"type": "text", "x": 230, "y": 370, "text": "Question + answer twice", "tone": "ink"}, "overlap"),   # on the red box's text
    ({"type": "text", "x": 1180, "y": 80, "text": "far too far right", "tone": "ink"}, "runs off"),
    ({"type": "text", "x": 540, "y": 380, "text": "middle", "tone": "ink"}, "runs through the text"),   # the arrow crosses it
    ({"type": "box", "x": 640, "y": 600, "w": 60, "h": 40, "text": "a very long label in a tiny box", "tone": "ink"}, "does not fit"),
    ({"type": "box", "x": 300, "y": 400, "w": 200, "h": 120, "tone": "grey"}, "cut into each other"),
    ({"type": "text", "x": 700, "y": 230, "text": "edge", "tone": "ink"}, "cut by the edge"),
])
def test_the_lint_names_what_to_move(element, expect):
    spec = copy.deepcopy(TWO_ROWS)
    spec["elements"].append(element)
    found = diagram_problems(norm(spec))
    assert any(expect in x for x in found), found
    assert all("element" in x or "diagram" in x for x in found)   # each problem names the elements to change


def test_too_many_texts_and_long_labels_are_flagged():
    many = {"kind": "diagram", "title": "T", "elements": [
        {"type": "text", "x": 60 + 60 * (i % 18), "y": 60 + 60 * (i // 18), "text": str(i), "size": 24} for i in range(20)]}
    many["elements"].append({"type": "text", "x": 600, "y": 700, "text": "one two three four five six seven eight nine", "size": 24})
    found = diagram_problems(norm(many))
    assert any("texts; keep it within" in x for x in found) and any("words; keep each label" in x for x in found)


@pytest.mark.parametrize("lang", ["en-US", "zh-CN"])
def test_a_diagram_fills_its_board_and_stays_inside(lang):
    spec = TWO_ROWS if lang == "en-US" else set_board_strings(TWO_ROWS, {p: "中文" + t[:4] for p, t in board_strings(TWO_ROWS)})
    img = render_board(spec, lang, (900, 560))
    gray = img.convert("L")
    w, h = img.size
    assert gray.getextrema()[0] < 120
    # the content was scaled to the board: ink reaches past the middle third on both sides
    left = gray.crop((0, 0, w // 3, h)).getextrema()[0]
    right = gray.crop((2 * w // 3, 0, w, h)).getextrema()[0]
    assert left < 140 and right < 140
    assert render_board(spec, lang, (900, 560)).tobytes() == img.tobytes()   # deterministic


# ----------------------------------------------------------------------------- the pipeline's diagrams

def diagram_panel(n=2, brief="the answer box first, then an arrow to the long documents", elements=None):
    board = {"kind": "diagram", "title": "Answer first", "brief": brief}
    if elements is not None:
        board["elements"] = elements
    return v3_panel(n, kind="board", beat="how", board=board, dialogue=[("papi", "We fix the answer first, then write the long text.")],
                    characters=("papi",))


GOOD = {"elements": copy.deepcopy(TWO_ROWS["elements"])}
CROWDED = {"elements": [{"type": "text", "x": 600, "y": 300, "text": "answer box", "tone": "red"},
                        {"type": "text", "x": 610, "y": 305, "text": "long text later", "tone": "ink"}]}


async def test_diagrams_are_designed_drawn_again_for_new_words_and_redesigned_on_a_checkers_request():
    prompts = []

    def model(prompt, messages, config):
        prompts.append(prompt)
        return GOOD if "YOUR PREVIOUS DESIGN" in prompt else CROWDED

    story = StoryboarderV2(FakeEngine(router=model))
    sb = v3_storyboard([v3_panel(1, beat="hook", narration="Long texts are hard."), diagram_panel()])
    await story.design_diagrams(sb, v3_paper())
    # the first design overlaps, the lint sends it back once, and the clean redesign is kept
    assert len(prompts) == 2 and "overlap" in prompts[1] and '"answer box"' in prompts[1]
    board = sb.panels[1].board
    assert D.designed(board) and board["designed_for"] == _diagram_key(sb.panels[1])
    assert sb.brief["diagrams"]["2"] == {"tries": 2, "problems": []}
    assert "the answer box first" in prompts[0] and "We fix the answer first" in prompts[0] and "PAPER TEXT" in prompts[0]
    # nothing changed: nothing asked; new words alone (a fact-check's "may") keep the checked drawing
    await story.design_diagrams(sb, v3_paper())
    sb.panels[1].dialogue[0].text = "The answer is locked before any document exists."
    await story.design_diagrams(sb, v3_paper())
    assert len(prompts) == 2
    # a new brief (a rewrite asked for another drawing): drawn again for it
    sb.panels[1].board["brief"] = "the answer box first, then the locked answer feeding the documents"
    changed = await story.design_diagrams(sb, v3_paper())
    assert len(prompts) == 4 and "locked answer feeding" in prompts[2] and changed == {2}
    # a checker's finding: redesigned from the current drawing, with the finding in the prompt
    await story.design_diagrams(sb, v3_paper(), requests={2: ["board check (label): 'expand' is never explained"]})
    assert "'expand' is never explained" in prompts[4] and "YOUR PREVIOUS DESIGN" in prompts[4] and len(prompts) == 5


async def test_a_diagram_that_cannot_be_designed_becomes_a_flow_and_a_failed_redesign_keeps_the_drawing():
    story = StoryboarderV2(FakeEngine(router=lambda p, m, c: RuntimeError("checker down")))
    sb = v3_storyboard([diagram_panel(1, brief="Write the question and answer; then expand the facts into long documents.")])
    await story.design_diagrams(sb, v3_paper())
    board = sb.panels[0].board
    assert board["kind"] == "flow" and board["steps"] == ["Write the question and answer", "then expand the facts into long documents"]
    assert normalize_board(board) is not None and sb.brief["diagrams"]["1"]["fallback"] is True
    drawn = v3_storyboard([diagram_panel(1, elements=copy.deepcopy(GOOD["elements"]))])
    await story.design_diagrams(drawn, v3_paper(), requests={1: ["fix the label"]})
    assert drawn.panels[0].board["elements"] == GOOD["elements"]


def test_the_writer_sees_a_diagram_as_its_brief_and_labels_and_a_rewrite_keeps_the_drawing():
    drawn = dict(norm(TWO_ROWS), designed_for="abc")
    seen = _prompt_board(drawn)
    assert seen == {"kind": "diagram", "title": "Answer first, then the long text", "brief": "the answer is fixed before the text",
                    "labels": ["LONGZERO", "[Question + answer]", "[Long documents]", "Question + answer → Long documents (expand)"]}
    # the rewrite returns what the writer saw: the drawing stays
    kept = _kept_drawing(drawn, dict(seen, title="Answer first"))
    assert kept["elements"] == drawn["elements"] and kept["title"] == "Answer first" and "labels" not in kept
    assert kept["designed_for"] == "abc"
    # a new brief: drawn again
    new = _kept_drawing(drawn, dict(seen, brief="two rows of tokens"))
    assert "elements" not in new and new["brief"] == "two rows of tokens"
    # other boards pass through
    assert _kept_drawing({"kind": "bar"}, {"kind": "list", "items": ["a", "b"]}) == {"kind": "list", "items": ["a", "b"]}


def test_the_designer_prompt_asks_for_the_real_mechanism_on_the_running_example():
    sb = v3_storyboard([v3_panel(1, beat="hook", narration="Long texts are hard."), diagram_panel()])
    sb.running_example = {"text": "Who used the Vitae Extract in 1885?", "from_paper": True}
    text = diagram_prompt(sb, sb.panels[1], paper_text="PAPER")
    assert "Vitae Extract" in text and "No metaphors and no props" in text and "Long texts are hard." in text
    assert text.endswith("PAPER TEXT:\nPAPER") and "YOUR PREVIOUS DESIGN" not in text
    redo = diagram_prompt(sb, sb.panels[1], ["the texts overlap"], {"elements": []}, "PAPER")
    assert "- the texts overlap" in redo and "YOUR PREVIOUS DESIGN" in redo


def test_the_fallback_is_always_a_valid_flow():
    for brief in ("", "one", "a b c d e f g h i j k l m n o p q r", "First this. Then that; and finally the end."):
        out = _diagram_fallback({"kind": "diagram", "title": "T", "brief": brief})
        assert out["kind"] == "flow" and normalize_board(out) is not None, brief


# ----------------------------------------------------------------------------- round 23c: what the review found

def test_flat_corner_and_point_lists_and_box_names_make_real_arrows():
    b = norm({"kind": "diagram", "title": "T", "elements": [
        {"type": "box", "x": 100, "y": 300, "w": 200, "h": 100, "text": "Question"},
        {"type": "box", "x": 500, "y": 300, "w": 200, "h": 100, "text": "Retriever"},
        {"type": "arrow", "from": [200, 300], "via": [600, 150], "to": [1000, 300], "label": "skip"},   # one corner, flat
        {"type": "line", "points": [100, 600, 700, 600]},                                              # a flat point list
        {"type": "arrow", "from": "Question", "to": "Retriever", "label": "ask"}]})                    # ends named by box text
    skip, flat, named = [e for e in b["elements"] if e["type"] in ("arrow", "line")]
    assert skip["points"] == [[200.0, 300.0], [600.0, 150.0], [1000.0, 300.0]]
    assert flat["points"] == [[100.0, 600.0], [700.0, 600.0]]
    assert named["points"][0][0] > 300 and named["points"][-1][0] < 500 and named["points"][0][1] == 350.0   # edge to edge
    notes = []
    D.normalize({"elements": [{"type": "arrow", "from": "Nowhere", "to": [1, 2]},
                              {"type": "arrow", "from": [0, 0], "to": [9, 9], "via": [1, 2, 3]}]}, {}, notes)
    assert any("'Nowhere'" in x for x in notes) and any("drawn straight" in x for x in notes)


def test_a_curve_passes_through_its_corners_and_its_label_sits_at_its_middle():
    e = {"type": "arrow", "points": [[800, 200], [1000, 400], [800, 600]], "curved": True, "label": "back"}
    pts = D.path(e)
    assert max(x for x, _ in pts) >= 999   # it reaches the corner at x=1000 (a Bezier pulled by it turned at 900)
    (ax, ay), (bx, by) = D._label_seg(e)
    assert abs((ay + by) / 2 - 400) < 40


def test_describe_tells_what_belongs_to_what_and_where_the_arrows_go():
    flow = norm({"kind": "diagram", "title": "T", "elements": [
        {"type": "circle", "x": 190, "y": 240, "r": 22, "text": "1"},
        {"type": "box", "x": 100, "y": 300, "w": 180, "h": 100, "text": "Warehouse"},
        {"type": "circle", "x": 590, "y": 240, "r": 22, "text": "2"},
        {"type": "box", "x": 500, "y": 300, "w": 180, "h": 100, "text": "Driver"},
        {"type": "arrow", "from": [280, 350], "to": [500, 350], "label": "parcel"},
        {"type": "box", "x": 900, "y": 300, "w": 180, "h": 100, "text": "Door"},
        {"type": "arrow", "from": [680, 350], "to": [900, 350], "label": "first try"},
        {"type": "mark", "x": 990, "y": 380, "mark": "cross", "size": 30}]})
    assert D.describe(flow) == ["[#1 Warehouse]", "[#2 Driver]", "[Door ✗]", "Warehouse → Driver (parcel)",
                                "Driver → Door (first try)"]
    sides = norm({"kind": "diagram", "title": "T", "elements": [
        {"type": "text", "x": 250, "y": 120, "text": "Most papers"}, {"type": "text", "x": 850, "y": 120, "text": "This work"}]
        + [{"type": "box", "x": 150, "y": 180 + 110 * k, "w": 200, "h": 90, "text": "layer"} for k in range(4)]
        + [{"type": "box", "x": 750, "y": 180 + 60 * k, "w": 200, "h": 50, "text": "layer"} for k in range(9)]})
    assert D.describe(sides) == ["Most papers", "[layer] ×4", "This work", "[layer] ×9"]   # column by column, repeats folded
    group = norm({"kind": "diagram", "title": "T", "elements": [
        {"type": "circle", "x": 600, "y": 400, "r": 300, "fill": "tint"}, {"type": "text", "x": 600, "y": 160, "text": "Any order"},
        {"type": "circle", "x": 600, "y": 450, "r": 120}, {"type": "text", "x": 600, "y": 450, "text": "Left to right"}]})
    assert D.describe(group) == ["[Any order]", "[Left to right (inside Any order)]"]


def test_a_drawing_that_names_nothing_or_writes_a_groups_name_on_its_boxes_is_sent_back():
    blank = norm({"kind": "diagram", "title": "The fix", "elements": [
        {"type": "box", "x": 60 + 140 * k, "y": 360, "w": 110, "h": 120} for k in range(8)]
        + [{"type": "circle", "x": 115 + 140 * k, "y": 300, "r": 22, "text": str(k + 1)} for k in range(8)]})
    found = diagram_problems(blank)
    assert any("names almost nothing" in x for x in found) and any("boxes are empty" in x for x in found)
    # a group box listed first: its name sits in its top band, clear of the box it holds
    group = norm({"kind": "diagram", "title": "T", "elements": [
        {"type": "box", "x": 200, "y": 60, "w": 800, "h": 500, "text": "Block 12"},
        {"type": "box", "x": 400, "y": 330, "w": 400, "h": 140, "text": "attention", "tone": "red"}]})
    placed = D.place(group, *_pen_wrap())
    assert placed[0].rect[3] <= 330 and diagram_problems(group) == []


def _pen_wrap():
    from services.boards import _Pen, _wrap_lines, default_fonts
    return _Pen(default_fonts(), "en"), _wrap_lines


def test_a_group_box_listed_last_never_paints_over_what_it_holds():
    spec = {"kind": "diagram", "title": "T", "elements": [
        {"type": "box", "x": 300, "y": 300, "w": 240, "h": 140, "text": "Extract", "tone": "red", "fill": "tint"},
        {"type": "box", "x": 200, "y": 200, "w": 800, "h": 400, "tone": "grey", "dashed": True, "fill": "none"}]}
    first = render_board(spec, "en-US", (900, 600))
    spec["elements"].reverse()
    # the tinted box keeps its tint whichever order the elements come in
    for img in (first, render_board(spec, "en-US", (900, 600))):
        rgb = img.convert("RGB")
        tinted = sum(1 for x in range(0, rgb.width, 2) for y in range(0, rgb.height, 2)
                     for px in [rgb.getpixel((x, y))] if px[0] > 240 and 200 < px[1] < 235 and 200 < px[2] < 235)
        assert tinted > 120


def test_a_kept_design_past_the_canvas_is_shrunk_in_not_cut():
    spec = {"kind": "diagram", "title": "T", "elements": [
        {"type": "box", "x": 100, "y": 300, "w": 300, "h": 120, "text": "Queries"},
        {"type": "text", "x": 1150, "y": 360, "text": "26,070 queries", "size": 40}]}
    img = render_board(spec, "en-US", (900, 600)).convert("L")
    w, h = img.size
    assert img.crop((w - 30, 0, w, h)).getextrema()[0] > 200   # nothing drawn up to the board's right edge


async def test_a_made_up_examples_numbers_may_be_drawn_and_a_wrong_number_never_wins():
    sb = v3_storyboard([diagram_panel(1)])
    sb.running_example = "a courier with 120 parcels on Monday"
    sb.brief["running_example"] = {"from_paper": False}
    labelled = copy.deepcopy(GOOD)
    labelled["elements"][3]["text"] = "e.g. 120 parcels"
    story = StoryboarderV2(FakeEngine(router=lambda p, m, c: copy.deepcopy(labelled)))
    await story.design_diagrams(sb, v3_paper())
    assert sb.brief["diagrams"]["1"] == {"tries": 1, "problems": []}   # the example's own number, as in the bubbles
    # from the paper: 120 is no number of it; a redesign without it wins even with a layout problem left
    sb = v3_storyboard([diagram_panel(1)])
    crowded = copy.deepcopy(GOOD)
    crowded["elements"].append({"type": "text", "x": 230, "y": 370, "text": "Question + answer twice", "tone": "ink"})
    replies = [labelled, crowded]
    story = StoryboarderV2(FakeEngine(router=lambda p, m, c: copy.deepcopy(replies.pop(0))))
    await story.design_diagrams(sb, v3_paper())
    texts = [t for _, t in D.texts(sb.panels[0].board)]
    assert "e.g. 120 parcels" not in texts and "Question + answer twice" in texts


async def test_what_the_normalizer_drops_is_sent_back_and_a_failed_call_is_asked_again():
    broken = {"elements": copy.deepcopy(GOOD["elements"]) + [{"type": "arrow", "start": [1, 2], "end": [3, 4]}]}
    replies = [RuntimeError("empty reply"), broken, GOOD]
    prompts = []

    def model(prompt, messages, config):
        prompts.append(prompt)
        return copy.deepcopy(replies.pop(0))
    story = StoryboarderV2(FakeEngine(router=model))
    sb = v3_storyboard([diagram_panel(1)])
    await story.design_diagrams(sb, v3_paper())
    assert len(prompts) == 3 and "the design: element 4 (arrow) needs from and to" in prompts[2]
    assert D.designed(sb.panels[0].board) and not story.skipped   # a call that failed once is no skipped stage


async def test_a_layout_request_keeps_every_text_and_refuses_a_redesign_that_changes_one():
    crowded = copy.deepcopy(TWO_ROWS)
    crowded["elements"][3].update({"x": 230, "y": 370})   # LONGZERO on the answer box: the corrected drawing overlaps
    drawn = norm(crowded)
    assert diagram_problems(drawn)
    sb = v3_storyboard([diagram_panel(1, elements=drawn["elements"])])
    sb.panels[0].board["designed_for"] = _diagram_key(sb.panels[0])
    renamed = copy.deepcopy(GOOD)
    renamed["elements"][0]["text"] = "Question and its answer"
    moved = copy.deepcopy(GOOD)   # the same texts, moved apart
    replies = [renamed, moved]
    prompts = []

    def model(prompt, messages, config):
        prompts.append(prompt)
        return copy.deepcopy(replies.pop(0))
    story = StoryboarderV2(FakeEngine(router=model))
    await story.design_diagrams(sb, v3_paper(), requests={1: ["layout: the text of element 3 overlaps element 1"]})
    assert "keep every text exactly" in prompts[0] and "it changed the texts" in prompts[1]
    texts = sorted(t for _, t in D.texts(sb.panels[0].board))
    assert texts == sorted(t for _, t in D.texts(drawn)) and diagram_problems(sb.panels[0].board) == []


def _diagram_story(label="Big model: 41.8 BLEU"):
    elements = copy.deepcopy(TWO_ROWS["elements"]) + [{"type": "text", "x": 600, "y": 550, "text": label, "tone": "ink", "size": 30}]
    panel = diagram_panel(1, elements=norm({"kind": "diagram", "title": "Answer first", "elements": elements})["elements"])
    return v3_storyboard([panel])


async def test_the_fact_check_corrects_a_labels_number_from_the_paper():
    sb = _diagram_story()
    i = next(k for k, e in enumerate(sb.panels[0].board["elements"]) if e.get("text", "").startswith("Big model"))
    fix = {"problems": [{"id": f"1.b.elements.{i}.text", "issue": "the paper reports 41.0", "corrected": "Big model: 41.0 BLEU"}]}
    story = StoryboarderV2(FakeEngine(router=lambda p, m, c: copy.deepcopy(fix)))
    assert await story.factcheck(sb, v3_paper(), board_flags={}) == 1
    assert sb.panels[0].board["elements"][i]["text"] == "Big model: 41.0 BLEU"
    # a number the paper lacks is never written in
    sb = _diagram_story()
    fix["problems"][0]["corrected"] = "Big model: 49.9 BLEU"
    await StoryboarderV2(FakeEngine(router=lambda p, m, c: copy.deepcopy(fix))).factcheck(sb, v3_paper(), board_flags={})
    assert sb.panels[0].board["elements"][i]["text"] == "Big model: 41.8 BLEU"


async def test_a_correction_that_makes_a_label_longer_stands_and_asks_the_designer_to_move_things():
    sb = _diagram_story()
    fix = {"problems": [{"id": "1.b.elements.1.label", "issue": "the paper expands facts", "corrected": "expand the facts into long documents"}]}
    flags: dict = {}
    await StoryboarderV2(FakeEngine(router=lambda p, m, c: copy.deepcopy(fix))).factcheck(sb, v3_paper(), board_flags=flags)
    assert sb.panels[0].board["elements"][1]["label"] == "expand the facts into long documents"
    assert flags[1] and all(x.startswith("layout: ") for x in flags[1])


async def test_a_correction_that_leaves_a_bare_number_on_a_label_is_written_in_without_a_crash():
    # round 23's paid run lost a whole book: a label corrected to a bare number left the translatable texts, and the
    # list of fixes looked its path up there (KeyError: 'elements.2.text')
    for corrected in ("41.0", "41.8"):   # a number the paper has (written in by the number rule), the same number (by words)
        sb = _diagram_story()
        i = next(k for k, e in enumerate(sb.panels[0].board["elements"]) if e.get("text", "").startswith("Big model"))
        fix = {"problems": [{"id": f"1.b.elements.{i}.text", "issue": "the label is the score", "corrected": corrected}]}
        flags: dict = {}
        story = StoryboarderV2(FakeEngine(router=lambda p, m, c: copy.deepcopy(fix)))
        assert await story.factcheck(sb, v3_paper(), board_flags=flags) == 1
        assert sb.panels[0].board["elements"][i]["text"] == corrected and story.last_fixes[0]["after"] == corrected
        assert not any(x.startswith("fact-check") for x in flags.get(1, []))   # written in: never sent on as unwritten


def test_the_check_view_of_a_diagram_is_its_texts_and_board_text_has_its_footnote():
    from services.storyboarder_v2 import _checked_view
    sb = _diagram_story()
    p = sb.panels[0]
    before = _checked_view(p)
    p.board = copy.deepcopy(p.board)
    p.board["elements"][0]["x"] = 40.0   # moved, not reworded: no second read
    assert _checked_view(p) == before
    p.board["note"] = "WMT 2014"
    assert "WMT 2014" in board_text(p.board)


def test_a_diagram_label_number_never_fails_the_board_and_the_scrub_drops_it():
    from services.storyboarder_v2 import board_issues, scrub_unsupported
    sb = _diagram_story("Big model: 49.9 BLEU")
    assert board_issues(sb.panels[0].board, v3_paper().body) == []   # the designer's to fix, never the whole board's end
    removed = scrub_unsupported(sb, v3_paper().body)
    assert any("49.9" in x for x in removed) and "Big model: 49.9 BLEU" not in board_text(sb.panels[0].board)
    assert D.designed(sb.panels[0].board)


# ----------------------------------------------------------------------------- round 23c: when the drawing is redone

async def test_a_board_check_finding_on_a_diagram_is_redrawn_at_once_before_the_fix_round():
    from .fakes import PaperModel, make_cast
    finding = {"n": 3, "kind": "label", "problem": "the arrow never says which word looks", "fix": "label it 'it looks at animal'"}
    model = PaperModel(board_problems=[finding])
    engine = FakeEngine(router=model)
    await StoryboarderV2(engine).build_v3(v3_paper(), make_cast(), "en-US", (16, 32))
    i = model.seen.index("boardcheck")
    assert model.seen[i + 1] == "diagram"   # while the panel numbers hold (no request is kept for later any more)
    redo = next(p for p in engine.prompts if "You design ONE diagram" in p and "YOUR PREVIOUS DESIGN" in p)
    assert "the arrow never says which word looks" in redo


async def test_a_fact_check_finding_on_a_drawing_goes_to_the_designer_and_the_last_drawing_is_read():
    from .fakes import PaperModel, make_cast
    model = PaperModel(factcheck_problems=[{"id": "3.b", "issue": "the arrow points the wrong way", "corrected": "draw it from it to animal"}])
    engine = FakeEngine(router=model)
    story = StoryboarderV2(engine)
    await story.build_v3(v3_paper(), make_cast(), "en-US", (16, 32))
    fixes = [p for p in engine.prompts if p.startswith("Fix the flagged panels") or "Fix the flagged panels" in p[:400]]
    assert not any("the arrow points the wrong way" in p for p in fixes)   # the writer only sees the brief: never asked
    assert any("the arrow points the wrong way" in p for p in engine.prompts if "You design ONE diagram" in p)
    seen = model.seen
    last_design = max(i for i, x in enumerate(seen) if x == "diagram")
    assert any(x == "factcheck" for x in seen[last_design + 1:])   # nothing drawn after the last check goes unread
    assert any("after the last check" in str(x) for x in story.skipped)   # what it still finds is on the record


async def test_a_quiz_revision_draws_its_diagrams_before_its_checks():
    from .fakes import PaperModel, make_cast
    story = StoryboarderV2(FakeEngine(router=PaperModel(quiz_misses=[{1}, set()])))
    order = []
    design, check = story.design_diagrams, story._check_script

    async def designed(sb, paper, requests=None):
        order.append("design")
        return await design(sb, paper, requests)

    async def checked(sb, cast, paper, changed_only=False):
        order.append("check after the revision" if changed_only else "check")
        return await check(sb, cast, paper, changed_only)
    story.design_diagrams, story._check_script = designed, checked
    await story.build_v3(v3_paper(), make_cast(), "en-US", (16, 32))
    i = order.index("check after the revision")
    assert order[i - 1] == "design"


def test_the_quiz_revision_sees_a_diagram_as_the_writer_does_and_is_shown_a_diagram_for_a_mechanism():
    from services import abstract_quiz
    from .fakes import make_cast
    sb = _diagram_story()
    sb.panels[0].board["designed_for"] = "abc"
    text = abstract_quiz.revise_prompt(sb, [0], make_cast())
    script = text.split("\nSCRIPT\n", 1)[1]
    assert '"elements"' not in script and "designed_for" not in script and "[Question + answer]" in script
    assert '"kind": "diagram"' in text and "what the picture alone makes clear" not in text


# ----------------------------------------------------------------------------- round 23c: the second review

async def test_a_fact_checked_title_keeps_the_drawing_and_a_failed_redesign_is_not_retried_unasked():
    sb = _diagram_story()
    sb.panels[0].board["designed_for"] = _diagram_key(sb.panels[0])
    fix = {"problems": [{"id": "1.b.title", "issue": "the paper says documents", "corrected": "Answer first, documents later"}]}
    story = StoryboarderV2(FakeEngine(router=lambda p, m, c: copy.deepcopy(fix)))
    await story.factcheck(sb, v3_paper(), board_flags={})
    assert sb.panels[0].board["title"] == "Answer first, documents later"
    calls = []
    story = StoryboarderV2(FakeEngine(router=lambda p, m, c: calls.append(p) or RuntimeError("down")))
    assert await story.design_diagrams(sb, v3_paper()) == set() and calls == []   # the key is the brief: nothing to redraw
    sb.panels[0].board["brief"] = "the answer box, then three documents"
    await story.design_diagrams(sb, v3_paper())
    assert len(calls) == 2 and D.designed(sb.panels[0].board)   # two failed calls, the old drawing kept...
    await story.design_diagrams(sb, v3_paper())
    assert len(calls) == 2   # ...and marked as tried for this brief: no call again until something asks


async def test_a_correction_the_paper_does_not_back_is_flagged_not_dropped():
    sb = _diagram_story()
    i = next(k for k, e in enumerate(sb.panels[0].board["elements"]) if e.get("text", "").startswith("Big model"))
    fix = {"problems": [{"id": f"1.b.elements.{i}.text", "issue": "the paper reports 41.0", "corrected": "Big model: 40.95 BLEU"}]}
    flags: dict = {}
    story = StoryboarderV2(FakeEngine(router=lambda p, m, c: copy.deepcopy(fix)))
    await story.factcheck(sb, v3_paper(), board_flags=flags)
    assert any(x.startswith("fact-check: the paper reports 41.0") for x in flags[1]) and 1 in story._unchecked(sb)


async def test_a_layout_only_redesign_may_not_change_a_value_in_a_box():
    els = [{"type": "box", "x": 100, "y": 300, "w": 160, "h": 100, "text": "8"}, {"type": "box", "x": 500, "y": 300, "w": 300, "h": 100, "text": "heads"},
           {"type": "text", "x": 180, "y": 350, "text": "attention heads", "size": 30}]
    sb = v3_storyboard([diagram_panel(1, elements=norm({"kind": "diagram", "title": "T", "elements": els})["elements"])])
    sb.panels[0].board["designed_for"] = _diagram_key(sb.panels[0])
    six = copy.deepcopy(els)
    six[0]["text"], six[2]["y"] = "6", 600
    story = StoryboarderV2(FakeEngine(router=lambda p, m, c: {"elements": copy.deepcopy(six)}))
    await story.design_diagrams(sb, v3_paper(), requests={1: ["layout: the text of element 2 overlaps element 0"]})
    assert sb.panels[0].board["elements"][0]["text"] == "8"


def test_a_label_on_a_solid_group_stays_readable():
    spec = {"kind": "diagram", "title": "T", "elements": [
        {"type": "box", "x": 100, "y": 150, "w": 1000, "h": 500, "tone": "blue", "fill": "solid", "text": "Our model"},
        {"type": "box", "x": 200, "y": 330, "w": 260, "h": 140, "tone": "ink", "fill": "tint", "text": "Encoder"},
        {"type": "box", "x": 740, "y": 330, "w": 260, "h": 140, "tone": "ink", "fill": "tint", "text": "Decoder"},
        {"type": "arrow", "from": [460, 400], "to": [740, 400], "label": "states", "tone": "ink"}]}
    b = norm(spec)
    from services.boards import _Pen, _wrap_lines, default_fonts
    label = next(p for p in D.place(b, _Pen(default_fonts(), "en"), _wrap_lines) if p.role == "label")
    img = render_board(spec, "en-US", (900, 600)).convert("L")
    # the label sits on its white halo: some of its pixels are dark ink, never white on white
    x1, y1, x2, y2 = label.rect
    assert img.getextrema()[0] < 100
    dark = 0
    for x in range(img.width):
        for y in range(img.height):
            if img.getpixel((x, y)) < 90:
                dark += 1
    assert dark > 0


def test_a_step_badge_in_a_card_leaves_its_text_centred_and_shapes_in_boxes_name_them():
    cards = norm({"kind": "diagram", "title": "T", "elements": [
        x for k, t in enumerate(("Retrieve", "Read", "Answer")) for x in (
            {"type": "box", "x": 60 + 380 * k, "y": 300, "w": 300, "h": 200, "text": t},
            {"type": "circle", "x": 100 + 380 * k, "y": 340, "r": 36, "text": str(k + 1), "fill": "solid", "tone": "red"})]})
    assert diagram_problems(cards) == []
    queue = norm({"kind": "diagram", "title": "T", "elements": [
        x for k in range(4) for x in ({"type": "box", "x": 100 + 200 * k, "y": 300, "w": 160, "h": 160, "dashed": True},
                                      {"type": "circle", "x": 180 + 200 * k, "y": 380, "r": 50, "text": "call", "fill": "tint"})]
        + [{"type": "text", "x": 500, "y": 200, "text": "9 am queue"}]})
    assert not any("boxes are empty" in x for x in diagram_problems(queue))


def test_describe_names_arrow_ends_by_what_they_touch_and_keeps_steps_on_their_things():
    cache = norm({"kind": "diagram", "title": "T", "elements": [
        {"type": "box", "x": 60, "y": 350, "w": 200, "h": 100, "text": "New token"},
        {"type": "box", "x": 420, "y": 250, "w": 700, "h": 300, "text": "KV cache"},
        *[{"type": "box", "x": 450 + 110 * k, "y": 400, "w": 90, "h": 90} for k in range(6)],
        {"type": "arrow", "from": [260, 400], "to": [420, 400]},
        {"type": "text", "x": 140, "y": 650, "text": "Input sentence"},
        {"type": "arrow", "from": [228, 650], "to": [420, 520]}]})
    lines = D.describe(cache)
    assert "New token → KV cache" in lines and "Input sentence → KV cache" in lines
    loop = norm({"kind": "diagram", "title": "T", "elements": [
        *[{"type": "box", "x": 100 + 350 * k, "y": 330, "w": 220, "h": 100, "text": t} for k, t in enumerate(("Draft", "Critique", "Revise"))],
        *[{"type": "circle", "x": 140 + 350 * k, "y": 270, "r": 22, "text": str(k + 1)} for k in range(3)],   # nearer the loop
        {"type": "arrow", "points": [[910, 330], [910, 220], [210, 220], [210, 330]], "label": "again"}]})
    assert D.describe(loop)[:3] == ["[#1 Draft]", "[#2 Critique]", "[#3 Revise]"]
    stack = norm({"kind": "diagram", "title": "T", "elements": [
        {"type": "box", "x": 300, "y": 100, "w": 600, "h": 600, "text": "Encoder layer"},
        {"type": "box", "x": 400, "y": 200, "w": 400, "h": 100, "text": "Self-attention"},
        {"type": "box", "x": 350, "y": 350, "w": 500, "h": 100, "text": "Add & norm"},
        {"type": "box", "x": 430, "y": 500, "w": 340, "h": 100, "text": "Feed-forward"}]})
    assert [x.split(" (")[0] for x in D.describe(stack)] == ["[Encoder layer]", "[Self-attention", "[Add & norm", "[Feed-forward"]
    values = norm({"kind": "diagram", "title": "T", "elements": [
        {"type": "box", "x": 100, "y": 300, "w": 200, "h": 100, "text": "Monday"}, {"type": "text", "x": 200, "y": 450, "text": "12"}]})
    assert "#12" not in " ".join(D.describe(values))   # a bare number is a value, not a step


def test_an_arrow_named_from_a_box_to_itself_or_into_its_group_is_sent_back():
    notes = []
    D.normalize({"elements": [{"type": "box", "x": 100, "y": 100, "w": 200, "h": 100, "text": "Refine"},
                              {"type": "box", "x": 50, "y": 50, "w": 900, "h": 600, "text": "Loop"},
                              {"type": "arrow", "from": "Refine", "to": "Refine", "label": "again"},
                              {"type": "arrow", "from": "Loop", "to": "Refine"}]}, {}, notes)
    assert any("to itself" in x for x in notes) and any("one inside the other" in x for x in notes)


def test_a_rewrite_that_adds_to_a_diagrams_labels_asks_the_designer_through_its_brief():
    drawn = dict(norm(TWO_ROWS), designed_for="abc")
    seen = _prompt_board(drawn)
    same = _kept_drawing(drawn, dict(seen, labels=[x.strip("[]") for x in seen["labels"]]))   # reformatted: no change
    assert same["elements"] == drawn["elements"]
    more = _kept_drawing(drawn, dict(seen, labels=seen["labels"] + ["it → street (weak weight)"]))
    assert "elements" not in more and "It also shows: it → street (weak weight)" in more["brief"]


async def test_a_redrawn_diagram_gets_the_second_look_even_when_its_words_stay():
    from .fakes import PaperModel, make_cast
    finding = {"n": 3, "kind": "label", "problem": "the arrow never says which word looks", "fix": "label it 'it looks at animal'"}
    model = PaperModel(board_problems=[finding], fix=lambda panel, problems: panel)   # the writer keeps the panel as it is
    from .fakes import DIAGRAM_DESIGN
    redrawn = copy.deepcopy(DIAGRAM_DESIGN)
    redrawn["elements"][1]["label"] = "it looks at animal"

    def router(prompt, messages, config):   # the designer's redesign really changes the drawing
        return copy.deepcopy(redrawn) if "YOUR PREVIOUS DESIGN" in prompt else model(prompt, messages, config)
    engine = FakeEngine(router=router)
    await StoryboarderV2(engine).build_v3(v3_paper(), make_cast(), "en-US", (16, 32))
    looks = [p for p in engine.prompts if "Check every board of this manga script" in p]
    assert model.seen.count("boardcheck") >= 2 and "are new or rewritten" in looks[1]
    fixes = [p for p in engine.prompts if "Fix the flagged panels" in p]
    assert any("the diagram was just redrawn for this finding" in p for p in fixes)


# ----------------------------------------------------------------------------- round 24: readable, real, no made-up data

def test_the_canvas_is_wide_and_its_fonts_start_where_a_phone_can_read_them():
    # round 23's boards left a diagram about 850 x 500 px under its title: on a 1200 x 800 canvas with fonts of 24-30 its
    # smallest text came out at a median of 14 px beside 44 px bubbles
    assert D.W / D.H == 2 and D.SIZE_MIN >= 32 and D.SIZE_DEFAULT >= 38
    small = norm({"kind": "diagram", "title": "T", "elements": [{"type": "text", "x": 600, "y": 300, "text": "tiny", "size": 20}]})
    assert small["elements"][0]["size"] == D.SIZE_MIN
    prompt = diagram_prompt(v3_storyboard([diagram_panel(1)]), v3_storyboard([diagram_panel(1)]).panels[0])
    assert "1200 x 600" in prompt and "font sizes 32 to 56" in prompt and "placeholder" in prompt


def test_a_diagram_brief_asking_for_a_chart_of_numbers_is_sent_to_a_chart_board():
    from services.storyboarder_v2 import lint_v3
    bars = diagram_panel(1, brief="A bar chart with three bars: 11 votes for river, 10 votes for weaver, 4 votes for tie.")
    found = lint_v3(v3_storyboard([bars]), v3_paper().body).get(1, [])
    assert any("asks for a chart of numbers" in x for x in found)
    for fine in ("Side by side bar charts: the teacher's next-token probabilities beside the student's at the same prefix.",
                 "Four rows: 1) Depth 4 collapses, 2) Depth 16 walks, 3) Depth 64 stalls at the wall, 4) Depth 256 vaults it.",
                 "An arrow from the queue to a raised threshold bar above the held lead.",
                 "Bars of the BF16 teacher's next-token probabilities beside the 2-bit Qwen3 student's, top-5 tokens."):
        found = lint_v3(v3_storyboard([diagram_panel(1, brief=fine)]), v3_paper().body).get(1, [])
        assert not any("asks for a chart of numbers" in x for x in found), fine


def test_a_shapes_text_that_fits_only_below_the_readable_floor_is_sent_back():
    cramped = norm({"kind": "diagram", "title": "T", "elements": [
        {"type": "box", "x": 100, "y": 200, "w": 250, "h": 90, "text": "Question and answer"},
        {"type": "box", "x": 700, "y": 200, "w": 400, "h": 160, "text": "Long documents"}]})
    placed = D.place(cramped, *_pen_wrap())
    assert placed[0].size < D.SIZE_MIN and placed[0].fits   # it fits once shrunk...
    assert any("fits only when shrunk" in x for x in diagram_problems(cramped))   # ...below what a phone can read


def test_bars_without_values_are_read_by_their_heights():
    # the designer may draw a model's spread over next tokens as bars without values: a reader of the transcript (the
    # quiz, the board check) must still learn which token the model favours
    spread = norm({"kind": "diagram", "title": "T", "elements": [
        {"type": "box", "x": 100 + 160 * k, "y": 500 - h, "w": 120, "h": h, "text": t}
        for k, (t, h) in enumerate((("mat", 300), ("hat", 90), ("dog", 160)))]})
    lines = " ".join(D.describe(spread))
    assert "[mat · tallest bar]" in lines and "[hat · short bar]" in lines and "[dog · mid bar]" in lines
    level = norm({"kind": "diagram", "title": "T", "elements": [
        {"type": "box", "x": 100 + 300 * k, "y": 200, "w": 200, "h": 120, "text": t} for k, t in enumerate(("A", "B", "C"))]})
    assert "bar" not in " ".join(D.describe(level))   # boxes of one height are boxes


async def test_a_corrected_label_that_grows_past_the_word_limit_is_never_a_relayout_request():
    # a relayout keeps every text, so it can never shorten a label: the fact-check's longer label stands and only the
    # things it now overlaps are moved (round 24's review)
    sb = _diagram_story()
    fix = {"problems": [{"id": "1.b.elements.1.label", "issue": "the paper expands the facts",
                         "corrected": "expand the facts into many long documents"}]}
    flags: dict = {}
    await StoryboarderV2(FakeEngine(router=lambda p, m, c: copy.deepcopy(fix))).factcheck(sb, v3_paper(), board_flags=flags)
    assert sb.panels[0].board["elements"][1]["label"] == "expand the facts into many long documents"
    assert not any("keep each label within" in x for x in flags.get(1, []))


def test_the_designer_gets_the_running_examples_parts():
    # round 25: diagrams designed one by one from a one-sentence example drew placeholders or other details on each board
    sb = v3_storyboard([v3_panel(1, beat="hook", narration="Long texts are hard."), diagram_panel()],
                       brief={"running_example": {"text": "Who used the Vitae Extract in 1885?", "from_paper": True,
                                                  "parts": ["Vitae Extract question", "answer: Sofia Bianchi"],
                                                  "old_way": "a short-context model never sees both documents",
                                                  "new_way": ""}})
    sb.running_example = "Who used the Vitae Extract in 1885?"
    text = diagram_prompt(sb, sb.panels[1], paper_text="PAPER")
    assert 'ITS PARTS IN THEIR REAL WORDS' in text and '"answer: Sofia Bianchi"' in text
    assert "WHAT THE OLD WAY DOES TO IT (draw only the part this panel reaches)" in text
    assert "WHAT THE PAPER'S METHOD DOES TO IT" not in text   # an empty field says nothing
    assert "the method is what the panel is about: red, and the old way blue" in text
    assert "THIS CASE IS MADE UP" not in text
    sb.brief["running_example"]["from_paper"] = False   # a made-up case is marked, and its values are never results
    made_up = diagram_prompt(sb, sb.panels[1], paper_text="PAPER")
    assert 'THIS CASE IS MADE UP (the paper shows no instance): mark it once with the small grey "e.g." note' in made_up
    sb.brief = {}
    assert "ITS PARTS" not in diagram_prompt(sb, sb.panels[1], paper_text="PAPER")


def test_the_designer_gets_each_parts_role_and_the_papers_own_figures():
    # round 26: the vision model's description of the figure that shows the case, and of the figure the brief names
    from services.paper_figures import section
    fig = lambda n, shows: {"figure": f"Figure {n}", "n": n, "page": n, "caption": f"Figure {n}: x.", "words": [],
                            "panels": [{"id": "", "kind": "method diagram", "shows": shows, "instance": "", "elements": [],
                                        "arrows": [], "data": []}]}
    paper = "Body.\n\n" + section([fig(2, "the student's own continuation after So"), fig(3, "probabilities at one prefix"),
                                   fig(4, "accuracy against tokens")])
    sb = v3_storyboard([v3_panel(1, beat="hook", narration="It loops."), diagram_panel(brief="the bypass, as in Figure 3")],
                       brief={"running_example": {"text": "One answer of the 2-bit student", "from_paper": True, "source": "Figure 2",
                                                  "parts": ["Wait", "So"], "roles": ["the student's own next token", ""],
                                                  "old_way": "", "new_way": ""}})
    sb.running_example = "One answer of the 2-bit student"
    text = diagram_prompt(sb, sb.panels[1], paper_text=paper)
    assert '"Wait — the student\'s own next token"' in text and '"So"' in text and "never strung into a sentence" in text
    head = text.split("PAPER TEXT:")[0]
    assert "THE PAPER'S OWN FIGURE OF THIS CASE" in head and "never a written number" in head
    assert head.index("THE PAPER'S OWN FIGURE OF THIS CASE") < head.index("Figure 2 (page 2)")
    assert "A FIGURE OF THE PAPER THAT THE WRITER NAMED FOR THIS STEP (it may show another case" in head
    assert head.index("THE WRITER NAMED") < head.index("Figure 3 (page 3)")
    assert "Figure 4 (page 4)" not in head   # only the figures of this case and this step
    assert "THE PAPER'S OWN FIGURE" not in diagram_prompt(sb, sb.panels[1], paper_text="Body without figures.")
    sb.brief["running_example"]["from_paper"] = False   # a made-up case has no figure of its own
    assert "THE PAPER'S OWN FIGURE OF THIS CASE" not in diagram_prompt(sb, sb.panels[1], paper_text=paper)
    sb.panels[1].board["brief"] = "the bypass, as in Figures 3 and 4"   # plural references name both
    both = diagram_prompt(sb, sb.panels[1], paper_text=paper).split("PAPER TEXT:")[0]
    assert "Figure 3 (page 3)" in both and "Figure 4 (page 4)" in both


def test_a_fallback_board_never_shows_a_figure_reference():
    from services.storyboarder_v2 import _diagram_fallback
    board = _diagram_fallback({"kind": "diagram", "title": "Bypass", "brief": "the fork token first (as in Figure 2), then the easy tokens, then the fork"})
    assert "Figure" not in json.dumps(board) and "fork token first" in json.dumps(board)
