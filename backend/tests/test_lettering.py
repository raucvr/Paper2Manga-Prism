"""services.lettering (v3 typesetting): line breaking, fitting, bubble placement, tails, boards and the fallback strip.

Synthetic arts only (a beige wall, a striped busy lower half, heads as filled ellipses, an optional white board),
mostly with a stub board renderer; the last tests check the seams with the real services.boards and services.art_check.
No network.
"""

from __future__ import annotations

import pytest
from PIL import Image, ImageChops, ImageDraw

from services import lettering as L
from services.lettering import FontSet, Grounding, LayoutError, fit, letter_panel, tokens, wrap
from services.storyboarder_v2 import Bubble, PanelV2

W, H = 1024, 1536
HEADS_SCENE = {"papi": (110.0, 760.0, 420.0, 1010.0), "kumo": (600.0, 820.0, 890.0, 1060.0)}
BOARD = (70, 360, 954, 1080)
HEADS_BOARD = {"nezu": (90.0, 1130.0, 400.0, 1380.0), "papi": (600.0, 1110.0, 930.0, 1390.0)}

EN = {
    "narration": "Recurrent networks read a sentence one word after another, so the steps cannot run in parallel.",
    "dialogue": [("papi", "What if every word could look at every other word at the same time?"),
                 ("kumo", "Wait. So the whole sentence is read at once, not word by word?")],
}
ZH = {
    "narration": "循环网络一个词接一个词地读句子，所以这些步骤没法并行。",
    "dialogue": [("papi", "如果每个词都能同时看到句子里的其他所有词呢？"),
                 ("kumo", "等等……所以整句话是一次读完的，不是一个词一个词地读？")],
}


@pytest.fixture(scope="module")
def fonts() -> FontSet:
    return FontSet(config={})          # the bundled fonts: independent of config/api_config.yaml and of installed fonts


def make_art(heads=(), board=None, size=(W, H), border=True, mode="RGB") -> Image.Image:
    w, h = size
    img = Image.new("RGB", size, (238, 232, 220))
    d = ImageDraw.Draw(img)
    step = max(6, h // 90)
    for y in range(int(h * 0.42), h, step):             # busy lower part, calm upper band
        d.line([(0, y), (w, y + h // 40)], fill=(150, 120, 90), width=max(1, w // 400))
    if board:
        d.rectangle(board, fill=(252, 252, 252), outline=(70, 70, 70), width=max(2, w // 170))
    for x1, y1, x2, y2 in heads:
        d.ellipse([x1, y1, x2, y2], fill=(170, 110, 60), outline=(40, 25, 10), width=max(2, w // 250))
    if border:
        d.rectangle([w * 0.03, h * 0.02, w * 0.97, h * 0.98], outline=(20, 20, 20), width=3)
    return img.convert(mode) if mode != "RGB" else img


def panel(lines: dict, kind: str = "scene", board: dict | None = None, narration: bool = True) -> PanelV2:
    return PanelV2(n=1, chapter="main", title="", characters=[s for s, _ in lines["dialogue"]], scene="", visual="",
                   dialogue=[Bubble(s, t) for s, t in lines["dialogue"]], narration=lines["narration"] if narration else "",
                   kind=kind, board=board)


BAR = {"kind": "bar", "title": "WMT 2014 English-German (BLEU)",
       "bars": [{"label": "ConvS2S", "value": "25.16"}, {"label": "Transformer big", "value": "28.4", "highlight": True}]}


class StubBoard:
    """Draws a red block on white, like a chart, and records its calls."""

    def __init__(self):
        self.calls = []

    def __call__(self, spec, language, size, fonts):
        self.calls.append((dict(spec), language, tuple(size)))
        w, h = size
        im = Image.new("RGB", (w, h), "white")
        ImageDraw.Draw(im).rectangle([w * 0.1, h * 0.2, w * 0.9, h * 0.8], fill=(200, 60, 50), outline=(30, 30, 30), width=4)
        return im


def overlap(a, b, gap=0.0) -> bool:
    return L._overlap(tuple(a), tuple(b), gap)


def assert_layout_ok(layout: dict, heads: dict, s: float = 1.0) -> None:
    """Bubbles never cover heads, the chart, the narration or each other; reading order; one font size."""
    rects = [b["rect"] for b in layout["bubbles"]]
    narr = layout["narration"]
    for r in rects:
        for h in heads.values():
            assert not overlap(r, h), (r, h)
        if layout["chart"] and layout["chart_placement"] in ("board", "card"):
            assert not overlap(r, layout["chart"]), (r, layout["chart"])
        if narr and narr["placement"] == "box":
            assert not overlap(r, narr["rect"]), (r, narr["rect"])
    for i, a in enumerate(rects):
        for b in rects[i + 1:]:
            assert not overlap(a, b), (a, b)
    for a, b in zip(rects, rects[1:]):
        assert L.reading_order_ok(tuple(a), tuple(b), s), (a, b)
    assert len({b["font_size"] for b in layout["bubbles"]}) <= 1
    if narr and narr["placement"] == "box":
        for h in heads.values():
            assert not overlap(narr["rect"], h)
    for b in layout["bubbles"]:
        if b["tail"]:
            tip = b["tail"][1]
            for h in heads.values():
                assert not (h[0] <= tip[0] <= h[2] and h[1] <= tip[1] <= h[3]), (tip, h)
            if layout["chart"] and layout["chart_placement"] in ("board", "card"):
                c = layout["chart"]
                assert not (c[0] <= tip[0] <= c[2] and c[1] <= tip[1] <= c[3])


def assert_untouched(out: Image.Image, art: Image.Image, rect) -> None:
    """Nothing (bubble, tail, box) was drawn inside rect."""
    box = tuple(int(round(v)) for v in rect)
    assert ImageChops.difference(out.crop(box), art.convert("RGB").crop(box)).getbbox() is None, box


# ----------------------------------------------------------------------------- line breaking

def test_en_tokens_keep_numbers_with_units_and_metrics():
    toks = tokens("It scores F1 77.8 and 28.4 BLEU after 2.3 · 10^19 FLOPs, using feed-forward layers.", "en-US")
    assert "F1 77.8" in toks and "28.4 BLEU" in toks and "2.3 · 10^19 FLOPs," in toks and "and" in toks
    assert "feed-" in toks and "forward" in toks          # a break opportunity after the hyphen


def test_en_wrap_never_splits_number_and_unit(fonts):
    text = "The big Transformer reaches 28.4 BLEU on WMT 2014 English-German, trained for 3.5 days on 8 GPUs."
    face = fonts.face("en-US", 36)
    for width in range(150, 900, 23):
        lines = wrap(text, "en-US", face, width)
        assert "".join("".join(lines).split()) == "".join(text.split())     # nothing lost (a hyphen may end a line)
        assert any("28.4 BLEU" in line for line in lines) and any("3.5 days" in line for line in lines)
        assert all(face.length(line) <= width or len(tokens(line, "en-US")) == 1 for line in lines)
    assert "2014 English-" not in tokens(text, "en-US") and "WMT 2014" in tokens(text, "en-US")


def test_zh_wrap_kinsoku_and_latin_runs(fonts):
    text = "“注意力”就是一切！Transformer在WMT 2014英德翻译上拿到28.4 BLEU，训练只花了3.5天（8块GPU），比GNMT+RL的2.3·10^19还省。"
    face = fonts.face("zh-CN", 40)
    for width in range(90, 700, 17):
        lines = wrap(text, "zh-CN", face, width)
        assert "".join(lines).replace(" ", "") == text.replace(" ", "")
        for line in lines[1:]:
            assert line[0] not in L.NO_START, (width, lines)
        for line in lines[:-1]:
            assert line[-1] not in L.NO_END, (width, lines)
        for run in ("Transformer", "28.4 BLEU", "3.5天", "GNMT+RL", "2.3·10^19"):
            assert any(run in line for line in lines), (run, width, lines)


def test_zh_tokens_units_and_brackets():
    toks = [t for t in tokens("成本是1.5 × 10^20（10^19次），只花了3.5天！GNMT+RL 是 1.4 · 10²⁰ 。", "zh-CN") if not t.isspace()]
    assert "1.5 × 10^20" in toks and "（10^19次），" in toks and "3.5天！" in toks and "1.4 · 10²⁰ 。" in toks


def test_fit_balances_lines_and_respects_box(fonts):
    text = "Attention links any two words in one step, while an RNN needs up to n steps."
    block = fit(text, "en-US", 520, 400, (36, 32, 28), fonts)
    assert block is not None and block.face.size == 36
    widths = [block.face.length(line) for line in block.lines]
    assert block.width <= 520 and block.height <= 400 and block.height == block.lh * len(block.lines)
    assert len(widths) == 1 or widths[-1] >= 0.35 * max(widths)        # no orphan on the last line
    face, lh, lines, bw, bh = fit(text, "en-US", 300, 400, (36, 32, 28), fonts)   # unpacks like the prototype
    assert bw <= 300 and bh <= 400
    assert fit(text, "en-US", 520, 30, (36, 32, 28), fonts) is None      # cannot fit: no line is that short


def test_paragraph_fixes_orphan_without_emptying_the_box(fonts):
    face = fonts.face("en-US", 36)
    text = "Parsing = finding a sentence's grammar tree. F1 measures how well the predicted tree matches the correct one."
    lines = L._paragraph(text, "en-US", face, 896)
    widths = [face.length(line) for line in lines]
    assert " ".join(lines).split() == text.split() and max(widths) <= 896
    assert widths[-1] >= 0.3 * 896 and widths[0] >= 0.8 * 896      # no orphan, and the first line still fills the box


def test_face_falls_back_for_missing_glyphs(fonts):
    face = fonts.face("en-US", 32)
    assert not L._has_glyph(face._chain[0], "中") and len(face._chain) >= 2     # Shantell Sans has no CJK glyphs
    runs = face.runs("Hi 中文 →")
    assert len({id(f) for f, _, _ in runs}) >= 2
    assert face.length("中") > 0
    im = Image.new("RGB", (200, 60), "white")
    face.draw(ImageDraw.Draw(im), (5, 5), "Hi 中文", (0, 0, 0))
    assert im.getbbox() is not None


@pytest.mark.parametrize("lang", ["en-US", "zh-CN"])
def test_superscripts_are_raised_small_digits(fonts, lang):
    face = fonts.face(lang, 40)
    runs = face.runs(L._pretty("2.3 · 10^19 FLOPs"))
    assert ("19" in [t for _, t, rise in runs if rise > 0]) and all(t != "¹⁹" for _, t, _ in runs)
    assert face.length("10¹⁹") < face.length("1019")
    assert L._pretty("1.4 · 10^-3 and 10^20") == "1.4 · 10⁻³ and 10²⁰"


def test_fontset_reads_configured_fonts_and_ignores_bad_paths(tmp_path):
    from services.fonts import bundled
    good = bundled("symbols")[0][0]          # DejaVu Sans, configured as the English font
    fs = FontSet(config={"fonts": {"en": {"text": f"{good}#0", "bold": [good, 0]}, "zh": str(tmp_path / "missing.ttf")}})
    assert fs.files["en"]["text"] == (good, 0)
    assert fs.files["zh"]["text"] == bundled("zh")[0]                     # missing file: the bundled font instead
    assert fs.face("en-US", 30).size == 30 and fs.face("en", 30, bold=True).size == 30


def test_sizes_scale_with_width_and_respect_minimums():
    # phone legibility: a 1024 px panel shows ~390 CSS px wide, so 34-36 px is ~13 CSS px
    assert L._scaled(L.BUBBLE_SIZES["zh"], 1.0)[-1] == 36 and L._scaled(L.BUBBLE_SIZES["en"], 1.0)[-1] == 34
    assert L._scaled(L.BUBBLE_SIZES["en"], 2.0) == tuple(2 * v for v in L.BUBBLE_SIZES["en"])
    assert L._scaled(L.BUBBLE_SIZES["zh"], 0.5)[-1] == 18


# ----------------------------------------------------------------------------- picture analysis

def test_free_band_finds_calm_top_clear_of_heads():
    art = make_art(heads=list(HEADS_SCENE.values()))
    band = L.free_band(art, list(HEADS_SCENE.values()))
    assert band[3] <= H * 0.45 and band[2] - band[0] >= W * 0.5
    assert not any(overlap(band, h) for h in HEADS_SCENE.values())


def test_clean_rect_stays_on_blank_board():
    art = make_art(board=BOARD)
    ImageDraw.Draw(art).rectangle([120, 400, 170, 420], fill=(30, 30, 30))   # a stray mark on the board must be avoided
    quad = [(BOARD[0], BOARD[1]), (BOARD[2], BOARD[1]), (BOARD[2], BOARD[3]), (BOARD[0], BOARD[3])]
    rect = L.clean_rect(art, quad)
    assert rect is not None
    assert BOARD[0] < rect[0] and BOARD[1] < rect[1] and rect[2] < BOARD[2] and rect[3] < BOARD[3]
    assert not overlap(rect, (120, 400, 170, 420))
    assert (rect[2] - rect[0]) * (rect[3] - rect[1]) > 0.5 * (BOARD[2] - BOARD[0]) * (BOARD[3] - BOARD[1])
    assert L.clean_rect(art, [(0, 0), (10, 0), (10, 10), (0, 10)]) is None


# ----------------------------------------------------------------------------- scene panels

@pytest.mark.parametrize("lang,lines", [("en-US", EN), ("zh-CN", ZH)])
def test_scene_panel_layout(fonts, lang, lines):
    art = make_art(heads=list(HEADS_SCENE.values()))
    out, layout = letter_panel(art, panel(lines), Grounding(size=(W, H), heads=HEADS_SCENE), lang, fonts)
    assert out.mode == "RGB" and out.size == (W, H) and layout["size"] == [W, H]
    assert layout["version"] == 1 and layout["kind"] == "scene" and layout["chart"] is None
    assert layout["fallback"] is False and layout["strip"] is None
    assert layout["narration"]["placement"] == "box" and layout["narration"]["rect"][1] < 500
    assert [b["speaker"] for b in layout["bubbles"]] == ["papi", "kumo"]
    assert all(b["tail"] for b in layout["bubbles"])
    size = layout["bubbles"][0]["font_size"]
    assert size >= (28 if lang == "zh-CN" else 26)
    assert_layout_ok(layout, HEADS_SCENE)
    for head in HEADS_SCENE.values():
        assert_untouched(out, art, head)


def test_three_bubbles_follow_reading_order(fonts):
    lines = {"narration": "", "dialogue": [("papi", "First, the problem."), ("kumo", "Then my question about it?"),
                                           ("papi", "And finally the answer, with 28.4 BLEU as proof.")]}
    art = make_art(heads=list(HEADS_SCENE.values()))
    out, layout = letter_panel(art, panel(lines, narration=False), Grounding(size=(W, H), heads=HEADS_SCENE), "en-US", fonts)
    assert [b["text"] for b in layout["bubbles"]] == [t for _, t in lines["dialogue"]]
    assert layout["narration"] is None
    assert_layout_ok(layout, HEADS_SCENE)


def test_speaker_without_head_gets_no_tail_and_text_is_kept(fonts):
    lines = {"narration": "", "dialogue": [("ghost", "Nobody can see me, but you can read me.")]}
    out, layout = letter_panel(make_art(), panel(lines, narration=False), Grounding(size=(W, H)), "en-US", fonts)
    assert len(layout["bubbles"]) == 1 and layout["bubbles"][0]["tail"] is None


def test_grounding_in_other_resolution_is_scaled(fonts):
    art = make_art(heads=list(HEADS_SCENE.values()))
    half = {k: tuple(v / 2 for v in box) for k, box in HEADS_SCENE.items()}
    out, layout = letter_panel(art, panel(EN), Grounding(size=(W // 2, H // 2), heads=half), "en-US", fonts)
    assert_layout_ok(layout, HEADS_SCENE)
    for head in HEADS_SCENE.values():
        assert_untouched(out, art, head)


@pytest.mark.parametrize("mode", ["RGBA", "L", "P"])
def test_output_is_rgb_for_any_input_mode(fonts, mode):
    art = make_art(heads=list(HEADS_SCENE.values()), mode=mode)
    out, layout = letter_panel(art, panel(EN), Grounding(size=(W, H), heads=HEADS_SCENE), "en-US", fonts)
    assert out.mode == "RGB" and art.mode == mode          # the input is never modified


def test_small_art_scales_sizes(fonts):
    art = make_art(heads=[tuple(v / 2 for v in h) for h in HEADS_SCENE.values()], size=(W // 2, H // 2))
    heads = {k: tuple(v / 2 for v in box) for k, box in HEADS_SCENE.items()}
    out, layout = letter_panel(art, panel(EN), Grounding(size=(W // 2, H // 2), heads=heads), "en-US", fonts)
    assert out.size == (W // 2, H // 2)
    assert all(17 <= b["font_size"] <= 23 for b in layout["bubbles"])
    assert_layout_ok(layout, heads, s=0.5)


def test_deterministic(fonts):
    art = make_art(heads=list(HEADS_SCENE.values()))
    g = Grounding(size=(W, H), heads=HEADS_SCENE)
    a, la = letter_panel(art, panel(ZH), g, "zh-CN", fonts)
    b, lb = letter_panel(art, panel(ZH), g, "zh-CN", fonts)
    assert la == lb and a.tobytes() == b.tobytes()


def test_bubble_outlines_are_antialiased(fonts):
    art = Image.new("RGB", (W, H), (180, 200, 230))
    out, layout = letter_panel(art, panel({"narration": "", "dialogue": [("papi", "Smooth edges, please.")]}, narration=False),
                               Grounding(size=(W, H)), "en-US", fonts)
    x1, y1, x2, y2 = layout["bubbles"][0]["rect"]
    colors = set(out.crop((x1, y1, x2, y2)).tobytes()[i:i + 3] for i in range(0, (x2 - x1) * (y2 - y1) * 3, 3))
    assert len(colors) > 20          # intermediate edge shades from the supersampled outline, not just 3 flat colours


# ----------------------------------------------------------------------------- board panels

def board_grounding(heads=HEADS_BOARD, board=BOARD) -> Grounding:
    quad = [(board[0], board[1]), (board[2], board[1]), (board[2], board[3]), (board[0], board[3])] if board else None
    return Grounding(size=(W, H), heads=heads, board=quad)


BOARD_LINES_EN = {"narration": "BLEU scores a translation by its overlap with human translations.",
                  "dialogue": [("nezu", "28.4? That beats even the ensembles?"), ("papi", "Yes. Even the base model beats them.")]}
BOARD_LINES_ZH = {"narration": "BLEU 衡量译文和人工译文的重合度，越高越好。",
                  "dialogue": [("nezu", "28.4？这比集成模型还高？"), ("papi", "没错，连基础版都超过了它们。")]}


@pytest.mark.parametrize("lang,lines", [("en-US", BOARD_LINES_EN), ("zh-CN", BOARD_LINES_ZH)])
def test_board_panel_layout(fonts, lang, lines):
    art = make_art(heads=list(HEADS_BOARD.values()), board=BOARD)
    stub = StubBoard()
    out, layout = letter_panel(art, panel(lines, kind="board", board=BAR), board_grounding(), lang, fonts, board_renderer=stub)
    assert layout["kind"] == "board" and layout["chart_placement"] == "board" and not layout["fallback"]
    chart = layout["chart"]
    assert BOARD[0] <= chart[0] and BOARD[1] <= chart[1] and chart[2] <= BOARD[2] and chart[3] <= BOARD[3]
    assert len(stub.calls) == 1
    spec, language, size = stub.calls[0]
    assert language == lang and size == (chart[2] - chart[0], chart[3] - chart[1]) and spec["kind"] == "bar"
    assert len(layout["bubbles"]) == 2 and all(b["tail"] for b in layout["bubbles"])
    assert_layout_ok(layout, HEADS_BOARD)
    # the chart region holds exactly the multiplied chart: no bubble or tail entered it
    expected = ImageChops.multiply(art.crop(tuple(chart)), stub(spec, lang, size, fonts))
    assert ImageChops.difference(out.crop(tuple(chart)), expected).getbbox() is None
    for head in HEADS_BOARD.values():
        assert_untouched(out, art, head)


PARSING = {"narration": "F1 measures how well the predicted grammar tree matches the correct one, on a scale up to 100.",
           "dialogue": [("papi", "Parsing works too, with the small WSJ data and with extra data.")]}


def test_board_narration_becomes_subtitle_when_nothing_else_fits(fonts):
    board = (70, 60, 954, 470)          # no wall above the board, and a low board: even the shrunk chart leaves no room
    art = make_art(heads=list(HEADS_BOARD.values()), board=board)
    stub = StubBoard()
    out, layout = letter_panel(art, panel(PARSING, kind="board", board=BAR), board_grounding(board=board), "en-US", fonts,
                               board_renderer=stub)
    assert layout["narration"]["placement"] == "subtitle"
    assert stub.calls[0][0]["subtitle"] == PARSING["narration"]
    assert_layout_ok(layout, HEADS_BOARD)


@pytest.mark.parametrize("top", [200, 123])
def test_board_narration_stays_a_box_when_the_chart_can_give_up_its_top(fonts, top):
    """A low wall above the board: the chart's top shrinks for the narration box instead of the narration turning into
    the board's small grey subtitle, and gets back whatever the placed words do not use."""
    board = (70, top, 954, 1080)
    art = make_art(heads=list(HEADS_BOARD.values()), board=board)
    stub = StubBoard()
    out, layout = letter_panel(art, panel(PARSING, kind="board", board=BAR), board_grounding(board=board), "en-US", fonts,
                               board_renderer=stub)
    narr, chart = layout["narration"], layout["chart"]
    assert narr["placement"] == "box" and narr["font_size"] >= 26 and "subtitle" not in stub.calls[0][0]
    assert not overlap(narr["rect"], chart) and board[0] <= chart[0] and chart[3] <= board[3]
    full = L.clean_rect(art, board_grounding(board=board).board, exclude=list(HEADS_BOARD.values()))
    assert chart[1] - full[1] <= 0.36 * (full[3] - full[1]) + 1                  # at most the three 12 % steps
    for f in L.CHART_SHRINK:                                                     # no smaller shrink would have done
        bigger = L._shrink_top(full, f)
        if bigger[1] < chart[1] - 1:
            assert overlap(narr["rect"], bigger, 6) or any(overlap(b["rect"], bigger, 20) for b in layout["bubbles"])
    assert_layout_ok(layout, HEADS_BOARD)


def test_board_without_board_in_art_pins_a_card(fonts):
    art = make_art(heads=list(HEADS_BOARD.values()))
    stub = StubBoard()
    out, layout = letter_panel(art, panel(BOARD_LINES_EN, kind="board", board=BAR), board_grounding(board=None), "en-US",
                               fonts, board_renderer=stub)
    assert layout["chart_placement"] == "card" and len(stub.calls) == 1
    assert not any(overlap(layout["chart"], h) for h in HEADS_BOARD.values())
    assert_layout_ok(layout, HEADS_BOARD)
    for head in HEADS_BOARD.values():
        assert_untouched(out, art, head)


def test_scene_kind_never_draws_a_board(fonts):
    stub = StubBoard()
    art = make_art(heads=list(HEADS_BOARD.values()), board=BOARD)
    out, layout = letter_panel(art, panel(BOARD_LINES_EN, kind="scene", board=BAR), board_grounding(), "en-US", fonts,
                               board_renderer=stub)
    assert stub.calls == [] and layout["chart"] is None and layout["kind"] == "scene"


@pytest.mark.parametrize("failure", ["raises", "no image"])
def test_board_renderer_failure_letters_the_panel_without_the_exhibit(fonts, failure):
    """The same spec fails on every retry: the panel keeps its paid art and its words, and the layout says why."""
    def broken(spec, language, size, fonts):
        if failure == "raises":
            raise ValueError("bad spec")
        return None

    art = make_art(heads=list(HEADS_BOARD.values()), board=BOARD)
    out, layout = letter_panel(art, panel(BOARD_LINES_EN, kind="board", board=BAR), board_grounding(), "en-US", fonts,
                               board_renderer=broken)
    assert layout["kind"] == "board" and layout["chart"] is None and layout["chart_placement"] is None
    assert "board renderer" in layout["board_error"] and out.mode == "RGB"
    strip = [i["text"] for i in (layout["strip"] or {"items": []})["items"] if i["speaker"]]
    assert [b["text"] for b in layout["bubbles"]] + strip == [t for _, t in BOARD_LINES_EN["dialogue"]]
    assert layout["narration"]["placement"] in ("box", "strip")
    assert_layout_ok(layout, HEADS_BOARD)


def test_a_spec_the_real_renderer_rejects_still_letters_the_panel(fonts):
    art = make_art(heads=list(HEADS_BOARD.values()), board=BOARD)
    out, layout = letter_panel(art, panel(BOARD_LINES_ZH, kind="board", board={"kind": "pie", "title": "x"}), board_grounding(),
                               "zh-CN", fonts)
    assert "pie" in layout["board_error"] and len(layout["bubbles"]) == 2
    assert_layout_ok(layout, HEADS_BOARD)


def test_tiny_art_is_a_layout_error(fonts):
    with pytest.raises(LayoutError):
        letter_panel(Image.new("RGB", (32, 32)), panel(EN), Grounding(size=(32, 32)), "en-US", fonts)


# ----------------------------------------------------------------------------- fallback strip

CROWDED = {"papi": (40.0, 40.0, 984.0, 720.0), "kumo": (40.0, 760.0, 984.0, 1500.0)}


@pytest.mark.parametrize("lang,lines", [("en-US", EN), ("zh-CN", ZH)])
def test_caption_strip_when_nothing_fits(fonts, lang, lines):
    art = make_art(heads=list(CROWDED.values()))
    out, layout = letter_panel(art, panel(lines), Grounding(size=(W, H), heads=CROWDED), lang, fonts)
    assert layout["fallback"] is True and out.mode == "RGB"
    assert out.width == W and out.height > H and layout["size"] == [W, out.height]
    strip = layout["strip"]
    assert strip["rect"][1] >= H and strip["rect"][3] <= out.height
    items = strip["items"]
    assert items[0]["speaker"] is None and items[0]["text"] == lines["narration"]
    assert [(i["speaker"], i["text"]) for i in items[1:]] == lines["dialogue"]
    assert layout["bubbles"] == [] and layout["narration"]["placement"] == "strip"
    assert_untouched(out, art, (0, 0, W, H))                     # the art itself is left alone
    assert out.crop((0, H, W, out.height)).getbbox() is not None


def test_bubbles_that_do_not_fit_go_to_the_strip_in_order(fonts):
    heads = {"papi": (40.0, 330.0, 500.0, 1500.0), "kumo": (520.0, 330.0, 984.0, 1500.0)}
    lines = {"narration": "", "dialogue": [
        ("papi", "This is a rather long first line of dialogue that needs a lot of room in the bubble."),
        ("kumo", "And this second line is just as long, so both cannot share the thin band at the top."),
        ("papi", "A third one makes sure that the strip below the art is really needed for this panel.")]}
    art = make_art(heads=list(heads.values()))
    out, layout = letter_panel(art, panel(lines, narration=False), Grounding(size=(W, H), heads=heads), "en-US", fonts)
    placed = [(b["speaker"], b["text"]) for b in layout["bubbles"]]
    rest = [(i["speaker"], i["text"]) for i in (layout["strip"] or {"items": []})["items"]]
    assert placed + rest == lines["dialogue"]                    # bubbles first, then the strip, in speaking order
    assert layout["fallback"] is True and rest and out.height > H
    assert_layout_ok(layout, heads)


def test_panel_dict_form_is_accepted(fonts):
    d = {"kind": "scene", "narration": EN["narration"], "dialogue": [{"speaker": "Papi", "text": "Hello there!"}]}
    out, layout = letter_panel(make_art(heads=[HEADS_SCENE["papi"]]), d, Grounding(size=(W, H), heads={"papi": HEADS_SCENE["papi"]}),
                               "en-US", fonts)
    assert layout["bubbles"][0]["speaker"] == "papi" and layout["bubbles"][0]["tail"]


def test_exhibit_goes_to_the_strip_when_the_art_has_no_room(fonts):
    stub = StubBoard()
    art = make_art(heads=list(CROWDED.values()))
    out, layout = letter_panel(art, panel(BOARD_LINES_EN, kind="board", board=BAR), Grounding(size=(W, H), heads=CROWDED),
                               "en-US", fonts, board_renderer=stub)
    assert layout["chart_placement"] == "strip" and layout["fallback"] is True
    chart, strip = layout["chart"], layout["strip"]
    assert strip["rect"][1] < chart[1] and chart[3] < strip["rect"][3] and chart[1] >= H
    assert [i["speaker"] for i in strip["items"]] == [None, "nezu", "papi"]
    assert len(stub.calls) == 1 and out.height > H


def test_layout_is_json_serializable(fonts):
    import json
    art = make_art(heads=list(HEADS_BOARD.values()), board=BOARD)
    _, layout = letter_panel(art, panel(BOARD_LINES_ZH, kind="board", board=BAR), board_grounding(), "zh-CN", fonts,
                             board_renderer=StubBoard())
    assert json.loads(json.dumps(layout, ensure_ascii=False)) == layout


def test_concurrent_lettering_with_a_shared_fontset_matches_sequential():
    from concurrent.futures import ThreadPoolExecutor

    shared = FontSet(config={})          # fresh: its caches fill while the threads race, like a job's first panels
    art = make_art(heads=list(HEADS_SCENE.values()))
    jobs = [(panel(EN), "en-US"), (panel(ZH), "zh-CN")] * 3
    g = Grounding(size=(W, H), heads=HEADS_SCENE)
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda job: letter_panel(art, job[0], g, job[1], shared), jobs))
    for (p, lang), (img, layout) in zip(jobs, results):
        ref_img, ref_layout = letter_panel(art, p, g, lang, FontSet(config={}))
        assert layout == ref_layout and img.tobytes() == ref_img.tobytes()


def test_plan_bubbles_raises_when_nothing_fits(fonts):
    with pytest.raises(LayoutError):
        L.plan_bubbles(W, H, (40, 40, 984, 400), [("papi", "Hello there, this needs some room.")], CROWDED, [], "en-US", fonts)
    placed = L.plan_bubbles(W, H, (40, 40, 984, 600), EN["dialogue"], HEADS_SCENE, [], "en-US", fonts)
    assert [p.speaker for p in placed] == ["papi", "kumo"] and len({p.block.face.size for p in placed}) == 1


# ----------------------------------------------------------------------------- review fixes

def test_zh_latin_runs_break_after_hyphens_and_slashes(fonts):
    toks = tokens("看multi-head-self-attention和github.com/tensorflow/tensor2tensor，比GNMT+RL好", "zh-CN")
    assert toks[:5] == ["看", "multi-", "head-", "self-", "attention"] and "github.com/" in toks and "GNMT+RL" in toks
    # one run far wider than a bubble: it used to push the whole bubble into the caption strip
    lines = {"narration": "", "dialogue": [
        ("papi", "这就是multi-head-scaled-dot-product-self-attention-with-sinusoidal-positional-encodings的全部秘密。")]}
    art = make_art(heads=list(HEADS_SCENE.values()))
    _, layout = letter_panel(art, panel(lines, narration=False), Grounding(size=(W, H), heads=HEADS_SCENE), "zh-CN", fonts)
    assert len(layout["bubbles"]) == 1 and layout["strip"] is None and not layout["fallback"]
    assert_layout_ok(layout, HEADS_SCENE)


def test_signs_and_ranges_stay_with_their_numbers(fonts):
    toks = [t for t in tokens("要1.4 × 10^20（−0.5 BLEU）。约～4.5分，3~5天，花了$5", "zh-CN") if not t.isspace()]
    assert "（−0.5 BLEU）。" in toks and "约～4.5分，" in toks and "3~5天，" in toks and "$5" in toks
    assert "~ 4.5 points." in tokens("It saves ~ 4.5 points.", "en-US") and "10-20 epochs" in tokens("for 10-20 epochs", "en-US")
    face = fonts.face("zh-CN", 40)
    text = "它比之前最好的模型高出+2.04分（误差−0.5），只花了约~3.5天，训练3~5轮就够。"
    for width in range(120, 700, 13):
        lines = wrap(text, "zh-CN", face, width)
        assert "".join(lines) == text and not any(line[-1] in "+−~～" for line in lines), (width, lines)


@pytest.mark.parametrize("lang", ["en-US", "zh-CN"])
def test_strip_cuts_a_unit_wider_than_the_line_instead_of_overflowing(fonts, lang):
    url = "https://example.org/" + "a1b2c3d4e5" * 14             # one 140-character unit after the slash
    text = f"The code is at {url} for everyone." if lang == "en-US" else f"代码在{url}，谁都能用。"
    art = make_art(heads=list(CROWDED.values()))
    out, layout = letter_panel(art, panel({"narration": "", "dialogue": [("papi", text)]}, narration=False),
                               Grounding(size=(W, H), heads=CROWDED), lang, fonts)
    strip = layout["strip"]
    face = fonts.face(lang, strip["font_size"], emphasis=True)
    prefix = fonts.face(lang, strip["font_size"], bold=True).length("Papi：" if lang == "zh-CN" else "Papi: ")
    inner = strip["rect"][2] - strip["rect"][0] - 2 * 22 - prefix
    item = strip["items"][0]
    assert "".join(item["lines"]).replace(" ", "") == text.replace(" ", "")      # nothing lost
    assert all(face.length(line) <= inner + 0.5 for line in item["lines"]), item["lines"]
    assert strip["rect"][2] <= out.width


def test_invisible_and_undrawable_characters_are_dropped(fonts):
    face = fonts.face("en-US", 36, emphasis=True)
    assert L._pretty("Zero\u200bwidth soft\u00adhyphen\ufeff ok 🎉 done", face) == "Zerowidth softhyphen ok done"
    assert L._pretty("好的🎉中文", fonts.face("zh-CN", 40)) == "好的中文"
    assert all("🎉" not in t for _, t, _ in face.runs("a🎉b"))              # never drawn as an empty box
    out, layout = letter_panel(make_art(heads=list(HEADS_SCENE.values())),
                               panel({"narration": "Tab\there 😀", "dialogue": [("papi", "We did it 🎉!")]}),
                               Grounding(size=(W, H), heads=HEADS_SCENE), "en-US", fonts)
    assert layout["bubbles"][0]["lines"] == ["We did it !"] and layout["narration"]["lines"] == ["Tab here"]


def test_only_real_numbers_are_bold():
    def bold(text):
        return "".join(c for c, b in zip(text, L._strong_mask(text)) if b)

    assert bold("8 P100 GPUs, WMT14, ResNet-152, v1.1") == ""
    assert bold("28.4 BLEU, F1 91.3, 4.9%, WMT 2014, 10⁻³") == "28.491.34.9%201410⁻³"


def test_no_gaps_inside_chinese_sentences():
    assert L._pretty("没错， 连 基础版\n都超过了 。 28.4 分，BLEU 越高越好") == "没错，连基础版都超过了。28.4 分，BLEU 越高越好"
    assert L._pretty("It works .  Really,\n  it does") == "It works . Really, it does"


def test_speaker_with_a_description_still_gets_its_tail(fonts):
    lines = {"narration": "", "dialogue": [("Papi (brown dog)", "Look at this!")]}
    heads = {"Papi": HEADS_SCENE["papi"], "Kumo (cloud)": HEADS_SCENE["kumo"]}
    _, layout = letter_panel(make_art(heads=list(HEADS_SCENE.values())), panel(lines, narration=False),
                             Grounding(size=(W, H), heads=heads), "en-US", fonts)
    assert layout["bubbles"][0]["speaker"] == "papi" and layout["bubbles"][0]["tail"]


def test_non_finite_head_boxes_are_ignored(fonts):
    heads = {"papi": (float("nan"), 760.0, 420.0, 1010.0), "kumo": HEADS_SCENE["kumo"]}
    _, layout = letter_panel(make_art(heads=list(HEADS_SCENE.values())), panel(EN), Grounding(size=(W, H), heads=heads), "en-US", fonts)
    assert [b["speaker"] for b in layout["bubbles"]] == ["papi", "kumo"]
    assert layout["bubbles"][0]["tail"] is None and layout["bubbles"][1]["tail"]


def test_fontset_ignores_a_config_that_is_not_a_mapping():
    assert FontSet(config=["not", "a", "mapping"]).files == FontSet(config={}).files


# ----------------------------------------------------------------------------- seams with the other v3 modules

REAL_SPECS = {
    "bar": BAR,
    "table": {"kind": "table", "title": "Linking two words", "columns": ["Layer", "Steps", "Path"],
              "rows": [{"cells": ["Self-Attention", "O(1)", "O(1)"]}, {"cells": ["Recurrent", "O(n)", "O(n)"]}], "highlight_rows": [0]},
    "flow": {"kind": "flow", "title": "The Transformer", "steps": ["Input + positions", "Encoder: 6 layers", "Decoder: 6 layers"],
             "highlight": [1]},
    "compare": {"kind": "compare", "title": "Reading", "left": {"label": "RNN", "items": ["One word after another"]},
                "right": {"label": "Transformer", "items": ["All words at once"]}},
    "list": {"kind": "list", "title": "What the paper showed", "items": ["28.4 BLEU English→German", "3.5 days on 8 GPUs"]},
    "links": {"kind": "links", "title": "Who does 'it' look at?", "tokens": ["The", "animal", "was", "tired", "so", "it"],
              "from": 5, "weights": [0.02, 0.6, 0.05, 0.2, 0.03, 0.1], "illustrative": True},
    "diagram": {"kind": "diagram", "title": "Answer first, then the long text", "brief": "the answer is fixed before the text",
                "elements": [{"type": "box", "x": 80, "y": 300, "w": 300, "h": 140, "text": "Question + answer", "tone": "red"},
                             {"type": "arrow", "from": [380, 370], "to": [700, 370], "tone": "ink", "label": "expand"},
                             {"type": "box", "x": 700, "y": 220, "w": 360, "h": 300, "text": "Long documents", "tone": "ink"}]},
}


@pytest.mark.parametrize("lang", ["en-US", "zh-CN"])
def test_real_exhibits_are_drawn_with_the_lettering_fonts(fonts, lang):
    """services.boards.render_board takes a lettering.FontSet as its BoardFonts, for every board kind."""
    from services.boards import BOARD_KINDS

    art = make_art(heads=list(HEADS_BOARD.values()), board=BOARD)
    lines = BOARD_LINES_EN if lang == "en-US" else BOARD_LINES_ZH
    for kind in BOARD_KINDS:
        out, layout = letter_panel(art, panel(lines, kind="board", board=REAL_SPECS[kind]), board_grounding(), lang, fonts)
        assert layout["chart_placement"] == "board" and "board_error" not in layout, (kind, layout.get("board_error"))
        chart = tuple(layout["chart"])
        assert ImageChops.difference(out.crop(chart), art.crop(chart)).getbbox() is not None, kind   # the exhibit is there
        assert_layout_ok(layout, HEADS_BOARD)


def test_grounding_from_the_vision_reply_letters_a_board_panel(fonts):
    """art_check.Grounding (built from a 0-1000 vision reply) is what letter_panel reads."""
    from services.art_check import grounding_from_json

    reply = {"characters": [{"name": "Nezu (hedgehog)", "head": [88, 736, 390, 898], "matches_reference": True},
                            {"name": "papi", "head": [586, 723, 908, 905]}],
             "boards": [[[68, 234], [932, 234], [932, 703], [68, 703]]],
             "speech_bubbles": [], "text_or_symbols": [], "extra_creatures": []}
    g = grounding_from_json(reply, (W, H), ["nezu", "papi"])
    assert set(g.heads) == {"nezu", "papi"} and g.board
    art = make_art(heads=list(HEADS_BOARD.values()), board=BOARD)
    out, layout = letter_panel(art, panel(BOARD_LINES_EN, kind="board", board=BAR), g, "en-US", fonts)
    assert layout["chart_placement"] == "board" and not layout["fallback"]
    assert [b["speaker"] for b in layout["bubbles"]] == ["nezu", "papi"] and all(b["tail"] for b in layout["bubbles"])
    assert_layout_ok(layout, g.heads)
    assert_layout_ok(layout, HEADS_BOARD)


def test_every_speaker_is_named_once_per_panel_and_wherever_a_tail_cannot_reach(fonts):
    # board panel: the bubbles sit above the board, the speakers below it; a tail can only point at the board
    lines = {"narration": "", "dialogue": [("nezu", "That beat even the ensembles?"), ("nezu", "By how much?"), ("papi", "Two points.")]}
    quad = [(BOARD[0], BOARD[1]), (BOARD[2], BOARD[1]), (BOARD[2], BOARD[3]), (BOARD[0], BOARD[3])]
    art = make_art(heads=list(HEADS_BOARD.values()), board=BOARD)
    _, layout = letter_panel(art, panel(lines, kind="board", board=BAR, narration=False),
                             Grounding(size=(W, H), heads=HEADS_BOARD, board=quad), "en-US", fonts, board_renderer=StubBoard())
    assert [b["chip"] for b in layout["bubbles"]] == [True, True, True]   # nezu's second bubble too: its tail stops at the board
    # a scene: each speaker's first bubble is named (readers had to work speakers out from the tails), a second one whose
    # tail reaches its speaker is not
    lines = {"narration": "", "dialogue": [("papi", "Look at this."), ("kumo", "What is it?"), ("papi", "A tart.")]}
    scene_art, g = make_art(heads=list(HEADS_SCENE.values())), Grounding(size=(W, H), heads=HEADS_SCENE)
    _, scene = letter_panel(scene_art, panel(lines, narration=False), g, "en-US", fonts)
    chips, tails = [b["chip"] for b in scene["bubbles"]], [b["tail"] for b in scene["bubbles"]]
    assert chips[:2] == [True, True]
    assert chips[2] == (not L._tail_reaches(HEADS_SCENE["papi"], (None, None, tuple(tails[2][1])) if tails[2] else None, 1.0))


def test_chinese_digits_come_from_the_digits_font(fonts):
    from services.fonts import bundled
    face = fonts.face("zh-CN", 40)
    assert face._chain[1] == bundled("digits")[0]             # Atkinson Hyperlegible Next: 1 / l / I and 0 / O distinct
    assert face._pick("1", False) is face._fonts[1] and face._pick("额", False) is face._fonts[0]
    assert fonts.face("en-US", 40)._pick("1", False) is fonts.face("en-US", 40)._fonts[0]   # English keeps its own


def test_a_scenes_idea_objects_stay_clear_of_the_words_when_they_fit(fonts):
    # the eight answer cards of a GRPO panel sat under a bubble: the boxes the idea check returns are kept clear
    art = make_art(heads=list(HEADS_SCENE.values()))
    lines = {"narration": "", "dialogue": [("papi", "Eight answers, one grade each."), ("kumo", "And the best ones teach?")]}
    _, plain = letter_panel(art, panel(lines, narration=False), Grounding(size=(W, H), heads=HEADS_SCENE), "en-US", fonts)
    first = plain["bubbles"][0]["rect"]
    held = (first[0], first[1], first[2], first[3])           # an object exactly where the first bubble went
    _, kept = letter_panel(art, panel(lines, narration=False), Grounding(size=(W, H), heads=HEADS_SCENE, objects=[held]), "en-US", fonts)
    assert kept["fallback"] is False and not any(overlap(b["rect"], held) for b in kept["bubbles"])
    whole = (0, 0, W, H)                                       # a box over half the art is the scene: the words go where they went before
    _, ignored = letter_panel(art, panel(lines, narration=False), Grounding(size=(W, H), heads=HEADS_SCENE, objects=[whole]), "en-US", fonts)
    assert [b["rect"] for b in ignored["bubbles"]] == [b["rect"] for b in plain["bubbles"]] and ignored["fallback"] is False


def test_a_box_over_half_the_art_does_not_push_the_real_object_aside(fonts):
    # the idea check sometimes boxes the whole scene: that box is no object to keep clear, the small one beside it is
    art = make_art(heads=list(HEADS_SCENE.values()))
    lines = {"narration": "", "dialogue": [("papi", "Eight answers, one grade each."), ("kumo", "And the best ones teach?")]}
    _, plain = letter_panel(art, panel(lines, narration=False), Grounding(size=(W, H), heads=HEADS_SCENE), "en-US", fonts)
    held = tuple(plain["bubbles"][0]["rect"])                  # a small object exactly where the first bubble went
    whole = (0, 0, W, H)                                       # ...and the whole scene, listed first
    _, kept = letter_panel(art, panel(lines, narration=False), Grounding(size=(W, H), heads=HEADS_SCENE, objects=[whole, held]),
                           "en-US", fonts)
    assert not any(overlap(b["rect"], held) for b in kept["bubbles"]) and not kept["strip"]


def test_the_first_idea_object_stays_clear_even_when_a_line_must_go_to_the_strip(fonts):
    # "These yellow erasers are unstable questions" sat on the erasers: the plan that ignored the objects covered the very
    # prop the words named (54 of 1284 scene layouts of rounds 15-19 covered the first object more than half). The idea
    # check lists its objects most important first; the first stays clear and the words that do not fit beside it go to
    # the caption strip under the art
    heads = {"papi": (0.0, 620.0, 500.0, 1536.0), "kumo": (524.0, 620.0, 1024.0, 1536.0)}   # only the top is free...
    art = make_art(heads=list(heads.values()))
    erasers = (120.0, 140.0, 904.0, 560.0)                                                   # ...and the erasers fill it
    lines = {"narration": "", "dialogue": [
        ("papi", "These yellow erasers are the unstable questions the model keeps getting wrong."),
        ("kumo", "So the model should practice on exactly those wobbly questions until they stop wobbling?")]}
    _, layout = letter_panel(art, panel(lines, narration=False), Grounding(size=(W, H), heads=heads, objects=[erasers]), "en-US", fonts)
    core = L._held(erasers, L._Ctx(art, "en-US", fonts, heads))
    assert not any(overlap(b["rect"], core) for b in layout["bubbles"])
    strip = [i for i in (layout["strip"] or {}).get("items", []) if i.get("speaker")]
    assert strip and len(layout["bubbles"]) + len(strip) == 2           # every line is lettered, on the art or under it
    assert [i["speaker"] for i in strip] == ["papi", "kumo"][len(layout["bubbles"]):]   # reading order kept


def test_a_scene_drops_the_last_idea_objects_first(fonts, monkeypatch):
    # whole objects, then the middle of the big ones, then fewer objects from the end; the first plan with every bubble on
    # the art wins, else the first object alone with the rest in the strip
    small, big, other = (10.0, 10.0, 60.0, 60.0), (0.0, 0.0, 600.0, 600.0), (900.0, 10.0, 950.0, 60.0)
    ctx = L._Ctx(Image.new("RGB", (W, H), "white"), "en-US", fonts, {}, objects=[big, small, other])
    core = L._held(big, ctx)
    assert core != big and L._held(small, ctx) == small
    tried = []

    def fake(ctx_, bubbles, narration_text, objects):
        tried.append(list(objects))
        return L._Plan(kind="scene", leftover=[] if list(objects) == [core, small] else [("papi", "x")])
    monkeypatch.setattr(L, "_scene_plan", fake)
    assert L._scene(ctx, [("papi", "x")], "").leftover == []
    assert tried == [[big, small, other], [core, small, other], [core, small]]
    tried.clear()
    monkeypatch.setattr(L, "_scene_plan", lambda c, b, n, objects: tried.append(list(objects)) or L._Plan(kind="scene", leftover=[("papi", "x")]))
    assert L._scene(ctx, [("papi", "x")], "").leftover and tried[-1] == [core]
    # no try fits every bubble: the most bubbles on the art win, the earliest of equals (it keeps the most objects clear);
    # dropping objects that freed no room covered them for nothing (round 21 review)
    on_art = {3: 1, 2: 2, 1: 2}
    plans = []

    def placed(ctx_, bubbles, narration_text, objects):
        plans.append(L._Plan(kind="scene", bubbles=[object()] * on_art[len(objects)], leftover=[("papi", "x")]))
        return plans[-1]
    monkeypatch.setattr(L, "_scene_plan", placed)
    # tries: whole (1 on the art), middles (1), [core, small] (2), [core] (2): the earliest with 2 keeps "small" clear
    assert L._scene(ctx, [("papi", "x")], "") is plans[2] and len(plans) == 4


def _frame_art(heads, color, size=(1120, 832)):
    img = Image.new("RGB", size, color)
    d = ImageDraw.Draw(img)
    for y in range(int(size[1] * 0.55), size[1], 12):
        d.line([(0, y), (size[0], y + 20)], fill=(150, 120, 90), width=2)
    for box in heads:
        d.ellipse(box, fill=(170, 110, 60), outline=(40, 25, 10), width=4)
    return img


def test_a_characters_second_head_in_two_frames_stays_clear(fonts):
    # two frames with papi and kumo in both: compose_frames gave their bottom heads to the objects, and the tries that
    # drop the last objects first put two bubbles on kumo's second head (round 21 review); a head no speaker owns ("#2
    # kumo") stays clear like every head
    from services.art_check import Grounding as Located, compose_frames
    top = {"papi": (60, 330, 400, 700), "kumo": (720, 340, 1060, 700)}
    bottom = {"papi": (60, 60, 400, 430), "kumo": (720, 70, 1060, 440)}
    art, g = compose_frames([_frame_art(top.values(), (238, 232, 220)), _frame_art(bottom.values(), (225, 235, 225))],
                            [Located(size=(1120, 832), heads=top, objects=[(300, 60, 820, 330)]),
                             Located(size=(1120, 832), heads=bottom, objects=[(300, 460, 820, 800)])], (W, H))
    lines = {"narration": "", "dialogue": [
        ("papi", "On the left the old answers weigh almost nothing at all, see?"),
        ("kumo", "And after training the very same answers tip the whole scale over?"),
        ("papi", "Exactly, the weight moved to the answers the group rewarded."),
        ("kumo", "So the group's average decides which way the scale tips each round?")]}
    _, layout = letter_panel(art, panel(lines, narration=False), g, "en-US", fonts)
    for name in ("#2 papi", "#2 kumo"):
        assert not any(overlap(b["rect"], g.heads[name]) for b in layout["bubbles"]), name
    strip = [i for i in (layout["strip"] or {}).get("items", []) if i.get("speaker")]
    assert len(layout["bubbles"]) + len(strip) == 4


def test_a_tail_cut_short_at_another_head_is_dropped(fonts):
    # The Flexibility Trap p10: Kumo's second bubble sat above Papi; its tail to Kumo stopped at Papi's head and pointed at
    # Papi under a "Kumo" tab. Such a tail is dropped (the tab names the speaker); a tail aimed at its own speaker stays,
    # even when another head is nearer its tip than the speaker's (28 good tails were lost to a nearest-head rule)
    ctx = L._Ctx(Image.new("RGB", (W, H), "white"), "en-US", fonts,
                 {"kumo": (120.0, 900.0, 380.0, 1150.0), "papi": (560.0, 520.0, 900.0, 800.0)})
    blocked = list(ctx.heads.values())
    over_papi = L.PlacedBubble("kumo", "Wait... what if we tie it straight?", None, 730.0, 330.0, 170.0, 110.0, ctx.stroke)
    tails = L._tails(ctx, [over_papi], [L.blob(730.0, 330.0, 170.0, 110.0, 3)], blocked)
    assert tails == [None]
    ctx.heads = {"kumo": (560.0, 900.0, 900.0, 1150.0), "papi": (380.0, 520.0, 520.0, 700.0)}
    above_kumo = L.PlacedBubble("kumo", "Look at this.", None, 730.0, 330.0, 170.0, 110.0, ctx.stroke)
    tails = L._tails(ctx, [above_kumo], [L.blob(730.0, 330.0, 170.0, 110.0, 3)], list(ctx.heads.values()))
    assert tails[0] is not None
