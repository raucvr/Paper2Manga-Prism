"""services.paper_figures: the paper's figures, cropped from the PDF and described by the vision model."""
import json

import pymupdf
import pytest

from engines.base import TextResponse
from services import paper_figures as pf
from services.ledger import LEDGER_HEAD
from services.paper_extractor import PaperText

PARA = "This paragraph explains the method in plenty of plain words so that it reads as the body text of a paper. " * 4


def _pdf(wrap: bool = False) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_textbox(pymupdf.Rect(72, 60, 540, 150), PARA, fontsize=10)
    if wrap:   # a figure in the right column, body text beside it in the left one
        page.insert_textbox(pymupdf.Rect(72, 160, 290, 420), PARA, fontsize=10)
        page.draw_rect(pymupdf.Rect(330, 170, 530, 290), color=(0, 0, 1))
        page.insert_text((350, 230), "Wait", fontsize=14)
        page.insert_textbox(pymupdf.Rect(320, 300, 540, 340), "Figure 2: Bypassing a fork.", fontsize=10)
    else:
        page.draw_rect(pymupdf.Rect(150, 170, 450, 330), color=(0, 0, 1))
        page.insert_text((200, 250), "Wait", fontsize=14)
        page.insert_text((300, 250), "0.83", fontsize=14)
        page.insert_textbox(pymupdf.Rect(72, 345, 540, 380), "Figure 1: A box with the token Wait.", fontsize=10)
    return doc.tobytes()


def test_the_figure_is_the_band_above_its_caption():
    doc = pymupdf.open(stream=_pdf(), filetype="pdf")
    figs = pf.figure_clips(doc)
    assert [f["n"] for f in figs] == [1]
    f = figs[0]
    x0, y0, x1, y1 = f["clip"]
    para_bottom = max(b[3] for b in doc[0].get_text("blocks") if "paragraph" in b[4])
    assert para_bottom <= y0 <= 170 and y1 >= 360 and x0 <= 150 and x1 >= 450   # the drawing and the caption, not the paragraph
    assert "Wait" in f["words"] and "0.83" in f["words"] and f["caption"].startswith("Figure 1: A box")
    assert f["page"] == 1 and pf.render(pymupdf.open(stream=_pdf(), filetype="pdf"), f)[:8] == b"\x89PNG\r\n\x1a\n"


def test_a_figure_beside_body_text_is_cropped_in_its_own_column():
    f = pf.figure_clips(pymupdf.open(stream=_pdf(wrap=True), filetype="pdf"))[0]
    assert f["n"] == 2 and f["clip"][0] > 290 and f["clip"][2] >= 530   # the left column's text stays out
    assert "Wait" in f["words"] and not any(w in f["words"] for w in ("paragraph", "explains"))


def test_the_description_is_cleaned_and_capped():
    fig = {"n": 3, "page": 4, "caption": "Figure 3: Probabilities.", "words": ["a", "two", "0.5"]}
    raw = "```json\n" + json.dumps({"panels": [
        {"id": "a", "kind": "chart", "shows": "teacher and student probabilities at one prefix", "instance": "",
         "elements": ["x " * 200, "", None, "legend: teacher p"], "arrows": "teacher -> student",
         "data": [{"series": "teacher p", "points": ["x: a, y: ~0.44", {"x": "two", "y": "~0.29"}, 7]}]},
        {"id": "b", "kind": "chart"}, "junk"]}) + "\n```"
    got = pf.clean(raw, fig)
    assert got["figure"] == "Figure 3" and got["page"] == 4 and len(got["panels"]) == 1   # an empty panel is dropped
    p = got["panels"][0]
    assert len(p["elements"][0]) <= 160 and p["elements"][1] == "legend: teacher p" and p["arrows"] == ["teacher -> student"]
    assert p["data"][0]["points"] == [{"x": "a", "y": "~0.44"}, {"x": "two", "y": "~0.29"}]
    assert pf.clean("no json here", fig) is None and pf.clean({"panels": []}, fig) is None


def test_values_read_off_an_axis_appear_only_as_an_order():
    # Right On Q, Figure 3: the model dropped the "~" on values it read off the axis of (b) ("99"); the text layer has no 99
    fig = {"figure": "Figure 3", "n": 3, "page": 4, "caption": "Figure 3: Qwen3-1.7B W2. (a) Teacher and student.",
           "words": ["a", "two", "some", "four", "0.25", "0.5", "6.7", "15.6", "21.1", "MATH-500", "1–50", "51–100", "101+",
                     "2–4", "≥20"],
           "panels": [{"id": "a", "kind": "chart", "shows": "probabilities on MATH-500 at about 0.44", "instance": "Qwen3-1.7B W2",
                       "elements": ["y ticks at 0.25, 0.5", "teacher ~0.44 for a"], "arrows": [],
                       "data": [{"series": "teacher p", "points": [{"x": "a", "y": "~0.44"}, {"x": "two", "y": "~0.29"},
                                                                  {"x": "some", "y": "~0.06"}, {"x": "four", "y": "0"}]},
                                {"series": "FKL", "points": [{"x": "2-4", "y": "99"}, {"x": ">20", "y": "99"}]},
                                {"series": "repeated", "points": [{"x": "1-50", "y": "6.7"}, {"x": "51-100", "y": "15.6"},
                                                                  {"x": "101+", "y": "21.1"}]}]}]}
    text = pf.figure_text(fig)
    assert "teacher p (read off the plot, highest first): a, two, some; about zero: four" in text
    assert "repeated: 1-50 6.7, 51-100 15.6, 101+ 21.1" in text   # printed in the figure: kept exactly
    assert "FKL (read off the plot, highest first): 2-4, >20" in text and "2-4 99" not in text   # unprinted: an order
    assert "~0.44" not in text and "0.29" not in text and "teacher (read off the plot) for a" in text   # marked: scrubbed
    assert "MATH-500" in text and "Qwen3-1.7B W2" in text and "y ticks at 0.25, 0.5" in text   # names and printed ticks stay
    raster = {**fig, "words": []}   # a picture has no text layer: its unmarked values are taken as printed
    assert "FKL: 2-4 99, >20 99" in pf.figure_text(raster)


def test_a_line_over_numeric_x_gives_its_trend():
    fig = {"figure": "Figure 1", "n": 1, "page": 2, "caption": "Figure 1: Pass@k.", "words": [],
           "panels": [{"id": "left", "kind": "chart", "shows": "", "instance": "", "elements": [], "arrows": [],
                       "data": [{"series": "AR order", "points": [{"x": "1", "y": "~80"}, {"x": "32", "y": "~93"}, {"x": "128", "y": "~96"}]}]}]}
    assert "AR order (read off the plot): rises from 1 to 128, highest at the end" in pf.figure_text(fig)


def test_the_section_goes_before_the_tables_and_replaces_an_older_one():
    fig = {"figure": "Figure 1", "n": 1, "page": 2, "caption": "Figure 1: Overview.", "words": ["Wait"],
           "panels": [{"id": "whole", "kind": "method diagram", "shows": "the student's own path", "instance": "... Wait , the problem says",
                       "elements": ["Wait: the student's own next token"], "arrows": ["teacher -> student (targets)"], "data": []}]}
    block = pf.section([fig])
    assert block.startswith(pf.FIGURES_HEAD) and "Figure 1 (page 2): Overview." in block
    assert "- method diagram: the student's own path | its case: ... Wait , the problem says" in block   # "whole" is no label
    body = f"Body text.\n\n{LEDGER_HEAD}\nT1 | Table 1 | page 3"
    once = pf.with_figures(body, block)
    assert once.index(pf.FIGURES_HEAD) < once.index(LEDGER_HEAD) and once.startswith("Body text.")
    twice = pf.with_figures(once, block.replace("Overview", "Pipeline"))
    assert twice.count(pf.FIGURES_HEAD) == 1 and "Pipeline" in twice and "Overview" not in twice and LEDGER_HEAD in twice
    assert pf.with_figures("Body.", block).endswith(block) and pf.with_figures(once, "") .count(pf.FIGURES_HEAD) == 0
    assert pf.figure_block(once, "Figure 1.a").startswith("Figure 1 (page 2)") and pf.figure_block(once, "Fig. 1")
    assert pf.figure_block(once, "Figure 7") == "" and pf.figure_block("no section", "Figure 1") == ""
    assert pf.section([]) == ""


def test_the_section_leaves_out_whole_figures_past_its_size():
    fig = lambda n: {"figure": f"Figure {n}", "n": n, "page": 1, "caption": f"Figure {n}: x.", "words": [],
                     "panels": [{"id": "", "kind": "chart", "shows": "y " * 400, "instance": "", "elements": [], "arrows": [], "data": []}]}
    block = pf.section([fig(1), fig(2), fig(3)], max_chars=2500)
    assert "Figure 1 (page 1)" in block and "Figure 3 (page 1)" not in block and len(block) <= 2500


class FakeVision:
    supports_vision = True

    def __init__(self, reply: str):
        self.reply, self.calls = reply, 0

    async def generate_text(self, messages, config=None):
        self.calls += 1
        assert messages[0].images and messages[0].images[0].mime_type == "image/png"
        assert "Figure 1" in messages[0].content and "Wait" in messages[0].content   # the text layer's words help spelling
        return TextResponse(content=self.reply, model="fake-vision")


async def test_figures_are_read_once_per_pdf_and_join_the_paper_text(tmp_path):
    reply = json.dumps({"panels": [{"id": "whole", "kind": "method diagram", "shows": "a box with a token",
                                    "instance": "Wait", "elements": ["Wait: the token"], "arrows": [], "data": []}]})
    engine = FakeVision(reply)
    paper = PaperText("T", f"Body.\n\n{LEDGER_HEAD}\nT1 | Table 1 | page 3", [], stats={})
    pdf = _pdf()
    figs = await pf.add_figures(paper, pdf, engine, cache_dir=tmp_path)
    assert engine.calls == 1 and [f["figure"] for f in figs] == ["Figure 1"] and paper.stats["figures"] == 1
    assert paper.body.index("Figure 1 (page 1)") < paper.body.index(LEDGER_HEAD)
    again = PaperText("T", "Body.", [], stats={})
    assert await pf.add_figures(again, pdf, engine, cache_dir=tmp_path) == figs and engine.calls == 1   # from the cache
    assert "Figure 1 (page 1)" in again.body


async def test_without_a_pdf_or_a_vision_model_the_paper_is_unchanged(tmp_path):
    paper = PaperText("T", "Body.", [], stats={})
    assert await pf.add_figures(paper, None, FakeVision("{}"), cache_dir=tmp_path) == []
    assert await pf.add_figures(paper, _pdf(), object(), cache_dir=tmp_path) == []
    assert await pf.add_figures(paper, b"not a pdf", FakeVision("{}"), cache_dir=tmp_path) == []   # a broken file: no figures
    assert paper.body == "Body."


async def test_a_figure_the_model_cannot_describe_is_left_out(tmp_path):
    engine = FakeVision("I cannot read this image.")
    paper = PaperText("T", "Body.", [], stats={})
    assert await pf.add_figures(paper, _pdf(), engine, cache_dir=tmp_path) == [] and paper.body == "Body."
    assert not list(tmp_path.iterdir()) and engine.calls == 2   # asked once more; nothing cached, so a later run asks again


class SlipVision(FakeVision):
    async def generate_text(self, messages, config=None):
        self.calls += 1
        if self.calls == 1:
            return TextResponse(content='{"panels": [{"id": "a", "shows": "x",}', model="fake")   # cut short
        if self.calls == 3:
            return TextResponse(content='{"panels": [{"id": "a", "kind": "chart", "shows": "bars", "data": [{"series": "PPL", '
                                        '"points": [{"x": "G1", "y": ~5.76}, {"x": "G2", "y": ≈5.84},]}]}]}', model="fake")
        return TextResponse(content='{"panels": [{"id": "a", "kind": "chart", "shows": "a box", "elements": ["Wait",],}],}', model="fake")


async def test_a_malformed_reply_is_asked_again_and_trailing_commas_are_forgiven(tmp_path):
    engine = SlipVision("")
    figs = await pf.read_figures(engine, _pdf(), tmp_path)
    assert engine.calls == 2 and figs[0]["panels"][0]["elements"] == ["Wait"]
    got = pf.clean((await engine.generate_text([])).content, {"n": 1, "page": 1})   # "y": ~5.76 written without quotes
    assert got["panels"][0]["data"][0]["points"] == [{"x": "G1", "y": "~5.76"}, {"x": "G2", "y": "≈5.84"}]


# ----------------------------------------------------------------------------- round 26 review

def test_the_figures_section_is_never_evidence_for_a_number():
    # a reviewer found many ways a read-off value could pass the number checks once the section sat in the paper text
    from services import evidence
    fig = {"figure": "Figure 1", "n": 1, "page": 1, "caption": "Figure 1: Speed.", "words": [],
           "panels": [{"id": "", "kind": "chart", "shows": "ours is 3.7x faster at 45 ms per step", "instance": "", "elements": [],
                       "arrows": [], "data": [{"series": "ours", "points": [{"x": "a", "y": "0.62"}, {"x": "b", "y": "0.71"}]}]}]}
    body = pf.with_figures(f"Our method reaches 81.2 on the test set.\n\n{LEDGER_HEAD}\nT1 | Table 1 | page 3", pf.section([fig]))
    assert "0.62" in body and "3.7x" in body   # the prompts see the figure (a picture: its values as printed)
    assert evidence.verify_text_numbers("It reaches 81.2.", body) == []
    assert evidence.verify_text_numbers("It scores 0.62, 3.7 times faster.", body)   # the section supports nothing
    assert pf.without_figures(body).startswith("Our method reaches 81.2") and pf.FIGURES_HEAD not in pf.without_figures(body)
    assert LEDGER_HEAD in pf.without_figures(body) and pf.without_figures("plain text") == "plain text"


def _two_columns() -> bytes:
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    for x0, x1 in ((54, 300), (312, 558)):
        page.insert_textbox(pymupdf.Rect(x0, 50, x1, 150), PARA, fontsize=9)
    page.draw_rect(pymupdf.Rect(70, 170, 280, 300), color=(0, 0, 1))             # the left column's figure
    page.draw_rect(pymupdf.Rect(330, 160, 550, 330), color=(1, 0, 0))            # another drawing in the right column
    page.insert_textbox(pymupdf.Rect(54, 310, 300, 340), "Figure 4: The left figure.", fontsize=9)
    return doc.tobytes()


def test_a_figure_in_one_column_never_grows_into_the_other():
    f = pf.figure_clips(pymupdf.open(stream=_two_columns(), filetype="pdf"))[0]
    assert f["n"] == 4 and f["clip"][0] <= 70 and 280 <= f["clip"][2] < 312


def test_a_long_text_inside_the_figure_is_no_paragraph_above_it():
    # a worked example drawn in a box (a prompt and its answer) is the figure, not the body text above it
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_textbox(pymupdf.Rect(72, 50, 540, 120), PARA, fontsize=10)
    page.draw_rect(pymupdf.Rect(80, 140, 530, 300), color=(0, 0, 0))
    page.insert_textbox(pymupdf.Rect(90, 150, 520, 290), "Question: " + "the agent reads the long document and answers. " * 6, fontsize=10)
    page.insert_textbox(pymupdf.Rect(72, 310, 540, 340), "Figure 5: A worked example.", fontsize=10)
    f = pf.figure_clips(pymupdf.open(stream=doc.tobytes(), filetype="pdf"))[0]
    assert f["clip"][1] < 140 and "Question:" in f["words"]


class FlakyVision(FakeVision):
    model = "vision-x"

    async def generate_text(self, messages, config=None):
        self.calls += 1
        if self.calls == 1:
            raise TimeoutError("slow")
        return TextResponse(content=self.reply, model="vision-x")


async def test_a_partial_reading_is_not_kept_and_the_model_is_part_of_the_key(tmp_path):
    reply = json.dumps({"panels": [{"id": "whole", "kind": "chart", "shows": "a box", "instance": "", "elements": [], "arrows": [], "data": []}]})
    engine, pdf = FlakyVision(reply), _pdf()
    assert await pf.read_figures(engine, pdf, tmp_path) == [] and not list(tmp_path.iterdir())   # lost to a timeout: asked again
    got = await pf.read_figures(engine, pdf, tmp_path)
    assert len(got) == 1 and [p.name for p in tmp_path.iterdir()][0].split(".")[1] == "vision-x"
    (next(tmp_path.iterdir())).write_bytes(b"\xff\xfe broken")   # a damaged cache file is read again from the PDF
    assert len(await pf.read_figures(engine, pdf, tmp_path)) == 1 and engine.calls == 3


def test_marked_values_never_survive_as_numbers():
    words = "0 1 2 4 8 10 20 1,000 2,000"
    for text in ("ours is ~3.2x faster", "about ~45ms per step", "loss ~1e-3", "range ~0.2-0.45", "~12K tokens", "∼0.45", "～0.45",
                 "≈ 0.45", "accuracy ~60-80%"):
        out = pf._scrub(text, words, raster=False)
        assert not any(n in out for n in ("3.2", "45", "1e-3", "0.2", "12K", "60", "80")), (text, out)
    assert pf._scrub("MATH-500 at 4K tokens, Qwen3-1.7B", words, raster=False) == "MATH-500 at 4K tokens, Qwen3-1.7B"
    assert not pf._printed("0.62 ± ~0.05", "0.62 0.05", raster=True)   # a mark anywhere in the value
    assert not pf._printed("1,250", words, raster=False) and pf._printed("1,000", words, raster=False)
    assert not pf._printed_in("26", "26.36") and not pf._printed_in("500", "MATH-500") and pf._printed_in("26.36", "26.36")
    fig = {"figure": "Figure 2", "n": 2, "page": 1, "caption": "Figure 2: x.", "words": ["0", "1000", "2000"],
           "panels": [{"id": "", "kind": "chart", "shows": "", "instance": "", "elements": [], "arrows": [],
                       "data": [{"series": "Ours (avg ~0.47)", "points": [{"x": "~100", "y": "~0.4"}, {"x": "~1850", "y": "~0.9"},
                                                                         {"x": "~1990", "y": "~0.8"}]}]}]}
    text = pf.figure_text(fig)
    assert "0.47" not in text and "1850" not in text and "1990" not in text and "100" not in text and "highest in between" in text


def test_figure_references_in_the_plural():
    assert pf.figure_refs("as in Figure 3.a and Fig. 5") == [3, 5]
    assert pf.figure_refs("Figures 2 and 3") == [2, 3] and pf.figure_refs("Figs. 2-4") == [2, 3, 4]
    assert pf.figure_refs("no figure here") == []


def test_a_figure_titled_above_itself_is_the_band_below_its_title():
    # economics papers put "Figure 1: Dispatch Lifecycle" above the drawing, and keep each body line a block of its own
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    for i in range(4):
        page.insert_text((72, 60 + 14 * i), "Body text line that runs across the whole width of the text column here.", fontsize=11)
    page.insert_text((240, 130), "Figure 1: Dispatch Lifecycle", fontsize=11)
    page.draw_rect(pymupdf.Rect(100, 150, 500, 330), color=(0, 0, 1))
    for i in range(4):
        page.insert_text((72, 360 + 14 * i), "Body text line that runs across the whole width of the text column here.", fontsize=11)
    f = pf.figure_clips(pymupdf.open(stream=doc.tobytes(), filetype="pdf"))[0]
    x0, y0, x1, y1 = f["clip"]
    assert y0 < 125 and 330 <= y1 < 350 and x0 <= 100 and x1 >= 500


def test_figures_side_by_side_are_cropped_apart():
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_textbox(pymupdf.Rect(72, 40, 540, 120), PARA, fontsize=10)
    page.draw_rect(pymupdf.Rect(80, 140, 290, 280), color=(0, 0, 1))
    page.draw_rect(pymupdf.Rect(320, 140, 530, 280), color=(1, 0, 0))
    page.insert_textbox(pymupdf.Rect(80, 290, 290, 320), "Figure 5: Left.", fontsize=10)
    page.insert_textbox(pymupdf.Rect(320, 290, 530, 320), "Figure 6: Right.", fontsize=10)
    figs = {f["n"]: f for f in pf.figure_clips(pymupdf.open(stream=doc.tobytes(), filetype="pdf"))}
    assert figs[5]["clip"][2] < 320 and figs[6]["clip"][0] > 290


# ----------------------------------------------------------------------------- round 26: second review

def test_a_caption_that_shares_its_block_with_the_figures_labels_is_found():
    # econ_ai_service: "(c) Positions 7-9" and "(d) Positions 81-120", then "Figure 5. RDiT Plots..." in one PDF block
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_textbox(pymupdf.Rect(72, 40, 540, 110), PARA, fontsize=10)
    page.draw_rect(pymupdf.Rect(90, 130, 520, 300), color=(0, 0, 1))
    assert page.insert_textbox(pymupdf.Rect(90, 305, 540, 380), "(c) Positions 7-9\n\nFigure 5. RDiT Plots of Hot-Lead Conversion",
                               fontsize=11) >= 0
    figs = pf.figure_clips(pymupdf.open(stream=doc.tobytes(), filetype="pdf"))
    assert [f["n"] for f in figs] == [5] and figs[0]["clip"][1] < 130 and figs[0]["caption"].startswith("Figure 5. RDiT")


def test_the_last_short_line_of_the_paragraph_above_stays_out():
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    long_line = "Body text line that runs across the whole width of the text column of the page."
    for i in range(3):
        page.insert_text((72, 60 + 13 * i), long_line, fontsize=11)
    page.insert_text((72, 99), "in Appendix C.", fontsize=11)                  # the paragraph's short last line
    page.insert_image(pymupdf.Rect(150, 120, 460, 300), pixmap=pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 30), 1))
    page.insert_textbox(pymupdf.Rect(72, 310, 540, 360), "Figure 5: The heatmap shows the degree of repetition in responses to "
                        "open-ended queries generated by the same models, over many prompts and samples.", fontsize=10)
    f = pf.figure_clips(pymupdf.open(stream=doc.tobytes(), filetype="pdf"))[0]
    assert f["clip"][1] > 99 and "Appendix" not in f["words"] and f["raster"] is True   # an image: values as read


def test_a_caption_under_its_figure_stays_there_when_something_is_drawn_below_it():
    # hivemind Figure 7: a long caption under its histograms, a rule drawn just below the caption
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.draw_rect(pymupdf.Rect(100, 60, 500, 230), color=(0, 0, 1))
    page.insert_textbox(pymupdf.Rect(72, 250, 540, 300), "Figure 7: The left histogram shows the distribution of Shannon "
                        "entropy across the human annotations for each triplet, and the right one their label counts.", fontsize=10)
    page.draw_line((72, 305), (540, 305))
    f = pf.figure_clips(pymupdf.open(stream=doc.tobytes(), filetype="pdf"))[0]
    assert f["clip"][1] < 60 and f["clip"][3] < 305


def test_a_rotated_page_renders_its_figure():
    doc = pymupdf.open(stream=_pdf(), filetype="pdf")
    doc[0].set_rotation(90)
    rotated = pymupdf.open(stream=doc.tobytes(), filetype="pdf")
    f = pf.figure_clips(rotated)[0]
    assert "Wait" in f["words"]
    png = pymupdf.Pixmap(pf.render(rotated, f))
    assert png.width > 100 and png.height > 100 and png.color_count() > 1   # the drawing, not a blank strip
