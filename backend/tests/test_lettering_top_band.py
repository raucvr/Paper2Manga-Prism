"""Offline regression for panel 32: a large upper band stranded by head proximity."""
from PIL import Image, ImageDraw
import pytest

from services import lettering as L
from .test_lettering import StubBoard, assert_layout_ok, assert_untouched

HEADS = {"papi": (23, 1190, 325, 1416), "kumo": (384, 1229, 637, 1444),
         "nezu": (686, 1213, 949, 1452)}
BOARD = [(33, 707), (990, 707), (990, 1147), (33, 1147)]
DIALOGUE = [
    {"speaker": "kumo", "text": "写不完的情况少多了。"},
    {"speaker": "papi", "text": "在六种设置里，OnQ的平均截断率都是QAT方法中最低的。2位时，在MATH-500、AIME、LiveCodeBench上，九种情况里有八种，它的平均回答比最准的对比方法更短。"},
    {"speaker": "nezu", "text": "这些都是在论文的评测里测到的，换了别的任务还得重新测。"},
]
NARRATION = "这组比较把回答长度上限设为32768个文字片段。到了上限仍未写完，就被切断，记为“截断”。"


@pytest.fixture(scope="module")
def fonts():
    return L.FontSet(config={})


def fixture():
    art = Image.new("RGB", (1024, 1536), (255, 247, 235))
    draw = ImageDraw.Draw(art)
    draw.rectangle((9, 9, 1014, 1526), outline=(40, 30, 20), width=2)
    draw.rectangle((33, 707, 990, 1147), fill="white", outline=(100, 70, 50), width=5)
    for head in HEADS.values():
        draw.ellipse(head, fill=(180, 120, 80))
    panel = {"kind": "board", "dialogue": DIALOGUE, "narration": NARRATION,
             "board": {"kind": "bar", "title": "截断率", "unit": "%", "higher_is_better": False,
                       "bars": [{"label": "OnQ", "value": 4.39}, {"label": "对比方法", "value": 11.56}]}}
    return art, panel, L.Grounding(size=art.size, heads=HEADS, board=BOARD)


def compact(text):
    return "".join(text.split())


def test_three_dialogues_use_the_upper_band_without_lowering_readability(fonts):
    art, panel, grounding = fixture()
    board = StubBoard()
    out, layout = L.letter_panel(art, panel, grounding, "zh-CN", fonts, board_renderer=board)
    assert not layout["fallback"] and layout["strip"] is None
    assert out.size == art.size
    assert [(b["speaker"], b["text"]) for b in layout["bubbles"]] == [(b["speaker"], b["text"]) for b in DIALOGUE]
    assert all(b["font_size"] >= min(L.BUBBLE_SIZES["zh"]) for b in layout["bubbles"])
    assert layout["bubbles"][0]["rect"][1] < 300
    assert layout["bubbles"][1]["rect"][2] - layout["bubbles"][1]["rect"][0] > 0.75 * art.width
    for bubble in layout["bubbles"]:
        assert compact("".join(bubble["lines"])) == compact(bubble["text"])
        assert bubble["rect"][3] < layout["chart"][1]
    assert compact("".join(layout["narration"]["lines"])) == compact(NARRATION)
    assert_layout_ok(layout, HEADS)
    for head in HEADS.values():
        assert_untouched(out, art, head)
    full = L.clean_rect(art, BOARD, exclude=list(HEADS.values()))
    assert layout["chart"][1] <= full[1] + max(L.CHART_SHRINK) * (full[3] - full[1]) + 1
    assert len(board.calls) == 1


def test_top_band_search_still_falls_back_when_there_is_truly_no_room(fonts):
    art, panel, grounding = fixture()
    heads = {**HEADS, "protected": (20, 20, 1004, 695)}
    grounding = L.Grounding(size=art.size, heads=heads, board=BOARD)
    out, layout = L.letter_panel(art, panel, grounding, "zh-CN", fonts, board_renderer=StubBoard())
    assert layout["fallback"] and layout["strip"]
    items = [(b["speaker"], b["text"]) for b in layout["bubbles"]]
    items += [(b["speaker"], b["text"]) for b in layout["strip"]["items"] if b["speaker"] is not None]
    assert items == [(b["speaker"], b["text"]) for b in DIALOGUE]
    assert all(b["font_size"] >= min(L.BUBBLE_SIZES["zh"]) for b in layout["bubbles"])
    assert_layout_ok(layout, heads)
    assert_untouched(out, art, heads["protected"])


def test_an_existing_successful_layout_does_not_try_wider_fallback_shapes(fonts, monkeypatch):
    ctx = L._Ctx(Image.new("RGB", (1024, 1536), "white"), "zh-CN", fonts, {})
    shapes = ctx.shapes
    def ordinary_only(text, size, wide=False):
        assert not wide, "A successful ordinary placement must retain its existing strategy"
        return shapes(text, size)
    monkeypatch.setattr(ctx, "shapes", ordinary_only)
    result = L._plan(ctx, [("papi", "这句话很短。")], (20, 20, 1004, 650), [])
    assert result is not None
    assert result[0] == max(L.BUBBLE_SIZES["zh"])
