"""The paper's figures, read from their images.

A PDF's text layer keeps a figure only as scattered words. Right On Q's Figure 3a (teacher and student probabilities of
eight candidate tokens at one prefix) reached the pipeline as "two the 2-4 4-8 ... some four 8-20 other others another",
and round 25's analysis, asked for the paper's own instance, built its running example from that heap with the roles
wrong (it made "the" the teacher's top token; the figure gives "a" about 0.45 and "the" about 0). Papers put their worked
examples, their method overviews and many results in figures, so the pipeline now looks at them: each "Figure N" caption
is found in the PDF, the figure above it cropped within the caption's column, and the vision model writes what it shows
(its panels, the case it uses, its labels, arrows and data). The descriptions become one section of the paper text,
before the tables, so the analysis, the writer, the diagram designer and every model that checks against the paper read
the same account of each figure.

The section is a model's reading, never evidence: services.evidence leaves it out (without_figures), so no number in it
counts as the paper's own; the numbers the figures print that the PDF's text layer holds are in the body text already.
It still gives a value read off an axis only as an order or a trend, so that the writer is not invited to quote one.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Optional

FIGURES_HEAD = "=== FIGURES READ FROM THE IMAGES ==="
HEAD_NOTE = ("(what each figure shows, described from its image by a vision model; it is not the paper's text: quote numbers "
             "only from the paper text above, and a value read off an axis appears here only as an order or a trend)")
CAPTION_RE = re.compile(r"^\s*(?:figure|fig\.?)\s*(\d{1,3})\s*(?:[:.|]|[–—-]\s)", re.I)
TABLE_CAPTION_RE = re.compile(r"^\s*table\s*\d{1,3}\s*(?:[:.|]|[–—-]\s)", re.I)
REF_RE = re.compile(r"\b(?:Figure|Fig\.?)\s*(\d{1,3})", re.I)   # one reference ("Figure 3.a", "Fig. 3")
REFS_RE = re.compile(r"\b(?:Figures?|Figs?\.?)\s*((?:\d{1,3}[a-z]?(?:\.[a-z])?(?:\s*(?:,|and|&|–|-|to)\s*)?)+)", re.I)
NUM_RE = re.compile(r"(?<![\w.])[-+−]?\d+(?:\.\d+)?")
MARK = "~≈∼～≃"
# a number a description marks as read off the plot, with its unit or suffix and the far end of a range ("~0.2-0.45",
# "~3.2x", "~12K", "~1e-3", "∼45ms")
MARKED_RE = re.compile(rf"[{MARK}]\s*[-+−]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:e[-+]?\d+)?[a-zA-Z%]*"
                       r"(?:\s*(?:-|–|/|to)\s*[-+−]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:e[-+]?\d+)?[a-zA-Z%]*)?")
# a free-standing number in a description ("0.45", "21.1%"), never one inside a name ("MATH-500", "Qwen3-1.7B", "4K")
FREE_RE = re.compile(r"(?<![\w.\-])([-+−]?\d+(?:\.\d+)?)%?(?![\w.]\w|\w)")
PURE_NUM_RE = re.compile(rf"^[{MARK}]?\s*[-+−]?\d+(?:\.\d+)?\s*[kKmM]?$")
VERSION = 4              # bump when the prompt, the crop or the cleaning changes, so cached readings are made again
MAX_FIGURES = 12
MAX_SECTION_CHARS = 20_000
MAX_FIGURE_CHARS = 3_200
CHART_MARKS = 8          # a chart panel's labels kept (its data line carries the series; a method diagram keeps all)
DPI = 200
MAX_SIDE = 2000          # pixels on the longer side of a crop sent to the vision model
MAX_WORDS = 400
TIMEOUT = 300

PROMPT = """This image is {label} of a research paper, cropped from the PDF: its caption is at the bottom, and a little page text may show at the edges (ignore it).
Words printed inside the crop, from the PDF's text layer in no particular order (spell labels as these do; a figure embedded as a picture has none): {words}

Describe what the figure SHOWS, so that someone who cannot see it could redraw it faithfully. Return JSON only:
{{"panels": [{{"id": "a, b, left, right... or whole", "kind": "method diagram | worked example | chart | table | picture", "shows": "one sentence: what this panel shows", "instance": "the concrete case it uses, copied exactly (the prompt, prefix, tokens, outputs, names, positions), or empty", "elements": ["each label, token, box text, legend entry, colour code and highlight, copied exactly, with what it marks or where it sits"], "arrows": ["from -> to (its label)"], "data": [{{"series": "name", "points": [{{"x": "category or x value", "y": "value"}}]}}]}}]}}
Rules: copy every word, token and number exactly as printed. Every value is a JSON string; a value you read off an axis (not printed as a number in the figure) starts with "~" inside it ("~5.76"). Write "unreadable" for anything you cannot read; never guess. Describe only the figure, not the paper."""


# ----------------------------------------------------------------------------- finding and cropping

def _page_rect(page):
    """The page's rectangle in the coordinates its text and drawings use (a rotated page's are unrotated)."""
    return page.rect * page.derotation_matrix if page.rotation else page.rect


def _lines(page) -> list[dict]:
    """The page's text lines: {"rect", "text", "size" (the font size most of its characters use), "block", "first"
    (it starts its run of lines)}. Lines, not blocks: a paragraph that wraps around a figure is one block whose box
    reaches down beside the figure, and some PDFs keep every line a block of its own. A new run starts beside the
    previous line (two captions side by side), and at a caption that shares its block with the figure's own labels
    ("(c) Positions 7-9", then "Figure 5. RDiT plots"): the size changes, a gap opens or the line before is short."""
    import pymupdf
    out = []
    for i, b in enumerate(page.get_text("dict").get("blocks", [])):
        if b.get("type") != 0:
            continue
        starts: list[float] = []   # where each run of text side by side in this block begins
        prev = None                # (rect, size) of the block's previous line
        extra = 0
        for ln in b.get("lines", []):
            groups: list[list[dict]] = []   # spans apart by a wide gap are separate lines (two captions on one baseline)
            for sp in ln.get("spans", []):
                if groups and sp["bbox"][0] - groups[-1][-1]["bbox"][2] <= 15:
                    groups[-1].append(sp)
                else:
                    groups.append([sp])
            for spans in groups:
                text = "".join(sp.get("text", "") for sp in spans).strip()
                if not text:
                    continue
                sizes: Counter = Counter()
                for sp in spans:
                    sizes[round(float(sp.get("size", 0)), 1)] += len(sp.get("text", "").strip())
                size = sizes.most_common(1)[0][0] if sizes else 0.0
                rect = pymupdf.Rect(spans[0]["bbox"])
                for sp in spans[1:]:
                    rect |= pymupdf.Rect(sp["bbox"])
                beside = prev is not None and abs(rect.y0 - prev[0].y0) < 3 and rect.x0 > prev[0].x1 + 10
                near = [k for k, x0 in enumerate(starts) if abs(rect.x0 - x0) < 20]
                caption = prev is not None and not beside and bool(CAPTION_RE.match(text)) and (
                    abs(size - prev[1]) > 0.3 or rect.y0 - prev[0].y1 > 0.5 * max(rect.height, 1) or prev[0].width < 0.6 * rect.width)
                if not starts or (beside and not near):
                    starts.append(rect.x0)
                    sub, first = len(starts) - 1, True
                elif caption:
                    extra += 1
                    sub, first = 500 + extra, True
                else:
                    sub = near[0] if near else min(range(len(starts)), key=lambda k: abs(rect.x0 - starts[k]))
                    first = False
                out.append({"rect": rect, "text": text, "size": size, "block": i * 1000 + sub, "first": first})
                prev = (rect, size)
    return out


def _body_size(pages: list) -> float:
    """The font size of the document's body text: the half point most characters of its wide lines use, over every
    page (a figure-heavy page alone takes its figure's labels for the body; pdfTeX spreads the body over 9.9 to 10.1)."""
    sizes: Counter = Counter()
    for prect, lines in pages:
        for x in lines:
            if x["rect"].width > 0.3 * prect.width:
                sizes[round(x["size"] * 2) / 2] += len(x["text"])
    return sizes.most_common(1)[0][0] if sizes else 0.0


def _column(lines: list[dict], prect, cap, body: float = 0.0, skip: frozenset = frozenset()) -> tuple[float, float, float]:
    """The x-range of the text column a caption sits in, and the width of that column's body text. A page whose body
    lines (captions aside: a full-width figure* caption in the body size is no body text) are all narrower than 60% of
    it is set in two columns: a caption on one side of the gutter belongs to that side's column, and one that crosses
    the gutter (a short centred caption under a figure across both columns) to the whole page; otherwise the column is
    the whole page (a figure under a short centred caption may be as wide as the text)."""
    W = prect.width
    mid = prect.x0 + W / 2
    body_lines = [x["rect"] for x in lines if body and abs(x["size"] - body) < 0.6 and x["rect"].width > 0.3 * W
                  and x["block"] not in skip]
    text_width = (max(r.x1 for r in body_lines) - min(r.x0 for r in body_lines)) if body_lines else W
    if not body_lines or any(r.width > 0.6 * W for r in body_lines) or (cap.x0 < mid - 10 and cap.x1 > mid + 10):
        return prect.x0, prect.x1, text_width
    side = [r for r in body_lines if ((r.x0 + r.x1) / 2 < mid) == ((cap.x0 + cap.x1) / 2 < mid)]
    if not side:
        return prect.x0, prect.x1, text_width
    return min(r.x0 for r in side) - 12, max(r.x1 for r in side) + 12, max(r.x1 for r in side) - min(r.x0 for r in side)


def _graphics(page, prect) -> tuple[list, list]:
    """(rectangles of the page's vector drawings and embedded images, those of its images alone), without backgrounds
    that cover nearly the whole page (a full-page figure in a frame keeps its frame)."""
    import pymupdf
    rects = [pymupdf.Rect(d["rect"]) for d in page.get_drawings()]
    images = []
    try:
        images = [pymupdf.Rect(info["bbox"]) for info in page.get_image_info()]
    except Exception:  # noqa: BLE001 - an image the PDF cannot place is simply not counted
        pass
    area = prect.width * prect.height
    keep = lambda rs: [r for r in rs if r.is_valid and not r.is_empty and r.width * r.height < 0.85 * area]
    images = keep(images)
    return keep(rects) + images, images


def _is_stop(x: dict, body: float, graphics: list, width: float) -> bool:
    """A line a figure never reaches past: another caption, or a line of body text (the body's font size, about as
    wide as the column's text (an indented abstract too), and in no box of the figure: a prompt drawn in a box is the
    figure's own)."""
    if x["first"] and (CAPTION_RE.match(x["text"]) or TABLE_CAPTION_RE.match(x["text"])):
        return True
    r = x["rect"]
    if body and abs(x["size"] - body) >= 0.6:
        return False
    if any(g.contains(r) or (g & r).get_area() > 0.8 * r.get_area() for g in graphics):
        return False
    return r.width >= 0.75 * width


def _paragraph_end(stop: dict, lines: list[dict], limit: float) -> float:
    """The bottom of the stop line's paragraph: the lines right under it in its size and at its left edge (the short
    last line of the paragraph above a figure belongs to the text, not to the figure), never past ``limit``."""
    cur, x0 = stop["rect"], stop["rect"].x0
    for y in sorted(lines, key=lambda y: y["rect"].y0):
        r = y["rect"]
        if r.y0 <= cur.y0 + 0.5 * max(cur.height, 1) or r.y1 > limit:
            continue   # the stop line itself or one beside it (line boxes overlap by their ascent and descent)
        if r.y0 - cur.y1 > 0.6 * max(r.height, 1):
            break
        if abs(y["size"] - stop["size"]) < 0.3 and abs(r.x0 - x0) < 3:
            cur = r
    return cur.y1


def _drawn(graphics: list, band, col0: float, col1: float) -> list:
    """The graphics in a band whose middle lies in the column."""
    return [g for g in graphics if g.intersects(band) and col0 <= (g.x0 + g.x1) / 2 <= col1]


def figure_clips(doc) -> list[dict]:
    """Each figure of a pymupdf document, in number order: {"n", "page", "clip" [x0, y0, x1, y1], "caption", "words",
    "raster"}. The caption is the run of lines that starts "Figure N:" (with the lines right below it that continue it).
    The figure is the band above it within the caption's column, up to the nearest line of body text or caption there
    (and the rest of that paragraph; a prompt drawn in a box, or text set smaller than the body, is the figure's own),
    widened to the drawings and images whose middle lies in that column (a figure wider than its caption; never into
    the other column of a two-column page, nor into a figure beside it). A caption whose drawing starts right under it,
    nearer than anything drawn above it, heads its figure (economics papers title a figure above it): the figure is
    then the band below it. A caption with no figure area (a table set as text, numbered as a figure) is left out.
    "words" are the text-layer words inside the crop; "raster": images cover most of it, so its words cannot tell
    printed values from read ones."""
    import pymupdf
    pages = [(page, _page_rect(page), _lines(page)) for page in doc]
    body = _body_size([(prect, lines) for _, prect, lines in pages])
    found: dict[int, dict] = {}
    for page, prect, lines in pages:
        graphics = images = None
        heads = [x for x in lines if x["first"] and CAPTION_RE.match(x["text"])]
        captions = frozenset(x["block"] for x in heads)
        for head in heads:
            n = int(CAPTION_RE.match(head["text"]).group(1))
            if n in found:
                continue
            if graphics is None:
                graphics, images = _graphics(page, prect)
            cap = pymupdf.Rect(head["rect"])
            for x in lines:
                if x["block"] == head["block"]:
                    cap |= x["rect"]
            for x in sorted(lines, key=lambda x: x["rect"].y0):   # a caption that runs on in the lines right below it
                r = x["rect"]
                if x["block"] != head["block"] and r.y0 >= cap.y1 - 1 and r.y0 - cap.y1 < 4 and r.x0 < cap.x1 and r.x1 > cap.x0 \
                        and not (x["first"] and (CAPTION_RE.match(x["text"]) or TABLE_CAPTION_RE.match(x["text"]))):
                    cap |= r
            col0, col1, width = _column(lines, prect, cap, body, captions)
            pick0, pick1 = col0, col1   # whose drawings these are: never those of a figure beside this one
            for other in heads:
                o = other["rect"]
                if other is head or o.intersects(cap) or abs(o.y0 - cap.y0) > 250:
                    continue
                if o.x0 >= cap.x1 - 2:
                    pick1 = min(pick1, (cap.x1 + o.x0) / 2)
                elif o.x1 <= cap.x0 + 2:
                    pick0 = max(pick0, (o.x1 + cap.x0) / 2)
            in_col = [x for x in lines if not (x["rect"].x1 <= cap.x0 or x["rect"].x0 >= cap.x1) and not cap.contains(x["rect"])]
            top = prect.y0 + 18
            for x in sorted(in_col, key=lambda x: -x["rect"].y1):
                if x["rect"].y1 <= cap.y0 + 1 and _is_stop(x, body, graphics, width):
                    ends = [y["rect"].y1 for y in lines if y["block"] == x["block"] and y["rect"].y1 <= cap.y0 + 1] if x["first"] else []
                    top = max([_paragraph_end(x, in_col, cap.y0 + 1)] + ends) + 2
                    break
            bottom = prect.y1 - 18
            for x in sorted(in_col, key=lambda x: x["rect"].y0):
                if x["rect"].y0 >= cap.y1 - 1 and _is_stop(x, body, graphics, width):
                    bottom = x["rect"].y0 - 2
                    break
            above = _drawn(graphics, pymupdf.Rect(prect.x0, top, prect.x1, cap.y0), pick0, pick1)
            below = _drawn(graphics, pymupdf.Rect(prect.x0, cap.y1, prect.x1, bottom), pick0, pick1)
            gap_above = cap.y0 - max((g.y1 for g in above), default=-1e9)
            gap_below = min((g.y0 for g in below), default=1e9) - cap.y1
            line = max(head["rect"].height, 8)
            one_line = cap.height < 1.8 * line   # a title above its figure is one line; a caption under one describes it
            if below and one_line and gap_below < gap_above and gap_below < 3 * line:
                drawn, y0, y1 = below, cap.y0 - 3, bottom
            elif above:
                drawn, y0, y1 = above, top, cap.y1 + 3
            elif below:
                drawn, y0, y1 = below, cap.y0 - 3, bottom
            elif cap.y0 - top >= 20:
                drawn, y0, y1 = [], top, cap.y1 + 3   # text only (a table numbered as a figure)
            else:
                continue   # no figure area: nothing to read
            gx0, gx1 = cap.x0, cap.x1
            for g in drawn:
                gx0, gx1 = min(gx0, max(g.x0, col0)), max(gx1, min(g.x1, col1))
            clip = pymupdf.Rect(max(prect.x0, gx0 - 6), y0, min(prect.x1, gx1 + 6), y1)
            if clip.height < 40 or clip.width < 40:
                continue
            words = [w[4] for w in page.get_text("words", clip=clip) if not cap.contains(pymupdf.Rect(w[:4]))][:MAX_WORDS]
            area = pymupdf.Rect()   # what the figure draws: its images cover most of it in a picture
            for g in drawn:
                area |= g & clip
            covered = sum((im & area).get_area() for im in images if im in drawn)
            caption = " ".join(x["text"] for x in sorted(lines, key=lambda x: (round(x["rect"].y0), x["rect"].x0)) if cap.contains(x["rect"]))
            found[n] = {"n": n, "page": page.number + 1, "clip": [round(v, 1) for v in clip], "caption": _cut(caption, 600),
                        "words": words, "raster": covered >= 0.5 * max(area.get_area(), 1)}
    return [found[n] for n in sorted(found)]


def render(doc, fig: dict) -> bytes:
    """The figure's crop as PNG, at DPI or less so that its longer side stays within MAX_SIDE pixels (on a rotated
    page, the crop's unrotated coordinates are turned to the page's)."""
    import pymupdf
    page = doc[fig["page"] - 1]
    clip = pymupdf.Rect(fig["clip"])
    if page.rotation:
        clip = clip * page.rotation_matrix
    longer = max(clip.width, clip.height) / 72
    dpi = max(72, min(DPI, int(MAX_SIDE / max(longer, 0.1))))
    return page.get_pixmap(clip=clip, dpi=dpi).tobytes("png")


def _prepare(pdf: bytes, limit: int) -> list[tuple[dict, bytes]]:
    """(figure, its PNG) for the first ``limit`` figures, in a worker thread: pymupdf work stays off the event loop,
    and one document is never used by two threads."""
    import pymupdf
    doc = pymupdf.open(stream=pdf, filetype="pdf")
    try:
        out = []
        for fig in figure_clips(doc)[:limit]:
            try:
                out.append((fig, render(doc, fig)))
            except Exception as e:  # noqa: BLE001 - one figure that will not render is left out
                print(f"[paper_figures] Figure {fig['n']} not rendered: {type(e).__name__}: {e}")
        return out
    finally:
        doc.close()


# ----------------------------------------------------------------------------- the vision model's description

def _cut(text: str, limit: int) -> str:
    """At most ``limit`` characters, cut at a space (never inside a word or a number)."""
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    head = text[:limit + 1]
    return (head[:head.rfind(" ")] if " " in head else head[:limit]).rstrip(" ,;:") + "…"


def _s(value, limit: int) -> str:
    return _cut(str(value), limit) if isinstance(value, (str, int, float)) and str(value).strip() else ""


def _list(value) -> list:
    return value if isinstance(value, list) else ([] if value in (None, "") else [value])


def _json_of(text: str):
    t = str(text or "").strip()
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end <= start:
        return None
    blob = t[start:end + 1]
    commas = re.sub(r",\s*([}\]])", r"\1", blob)                                   # a trailing comma
    marked = re.sub(rf'(:\s*)([{MARK}]\s*[-+−]?[\d.,]+%?)(\s*[,}}\]])', r'\1"\2"\3', commas)   # "y": ~5.76 unquoted
    for attempt in (blob, commas, marked):
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            continue
    return None


POINT_RE = re.compile(r"^\s*(?:x\s*[:=]\s*)?(.*?)\s*[,;]\s*(?:y\s*[:=]\s*)(.+)$|^\s*(.*?)\s*:\s*(.+)$", re.I)


def _point(p) -> Optional[dict]:
    if isinstance(p, dict):
        x, y = _s(p.get("x"), 50), _s(p.get("y"), 30)
    elif isinstance(p, str) and (m := POINT_RE.match(p)):
        x, y = _s(m.group(1) if m.group(1) is not None else m.group(3), 50), _s(m.group(2) or m.group(4), 30)
    else:
        return None
    return {"x": x, "y": y} if (x or y) else None


def clean(raw, fig: dict) -> Optional[dict]:
    """The model's JSON as a figure record (None when it gave nothing usable): known fields only, lengths capped."""
    data = _json_of(raw) if isinstance(raw, str) else raw
    if isinstance(data, list):
        data = {"panels": data}
    if not isinstance(data, dict):
        return None
    panels = []
    for p in _list(data.get("panels"))[:8]:
        if not isinstance(p, dict):
            continue
        series = []
        for s in _list(p.get("data"))[:8]:
            if isinstance(s, dict):
                points = [q for q in (_point(x) for x in _list(s.get("points"))[:40]) if q]
                if points:
                    series.append({"series": _s(s.get("series") or s.get("name"), 60), "points": points})
        panel = {"id": _s(p.get("id"), 20), "kind": _s(p.get("kind"), 30), "shows": _s(p.get("shows"), 300),
                 "instance": _s(p.get("instance"), 400),
                 "elements": [e for e in (_s(x, 160) for x in _list(p.get("elements"))[:30]) if e],
                 "arrows": [a for a in (_s(x, 120) for x in _list(p.get("arrows"))[:16]) if a], "data": series}
        if panel["shows"] or panel["instance"] or panel["elements"] or series:
            panels.append(panel)
    if not panels:
        return None
    return {"figure": f"Figure {fig['n']}", "n": fig["n"], "page": fig["page"], "caption": fig.get("caption", ""),
            "words": list(fig.get("words") or []), "raster": bool(fig.get("raster")), "panels": panels}


# ----------------------------------------------------------------------------- the section of the paper text

def _number(text: str) -> Optional[float]:
    m = NUM_RE.search(str(text or "").replace("−", "-").replace(",", ""))
    try:
        return float(m.group(0).replace("−", "-")) if m else None
    except ValueError:
        return None


def _printed_in(num: str, words: str) -> bool:
    """``num`` is one of the printed numbers: never the start of a longer one (26 in 26.36) or digits inside a name."""
    num = num.replace("−", "-").lstrip("+")
    return bool(re.search(rf"(?<![\w.\-]){re.escape(num)}(?!\d|\.\d)", words.replace("−", "-")))


def _printed(y: str, words: str, raster: bool) -> bool:
    """A value is the figure's own when it prints it: never one with a read-off mark anywhere; in a vector figure, only
    when every number in it is among the figure's text-layer words or caption (the model sometimes drops the mark); a
    figure embedded as a picture has no words to check, so its unmarked values are taken as printed."""
    y = str(y or "").strip()
    if not y or any(c in y for c in MARK) or "unreadable" in y.lower():
        return False
    nums = NUM_RE.findall(y.replace(",", ""))
    if not nums:
        return False
    return raster or all(_printed_in(n, words.replace(",", "")) for n in nums)


def _scrub(text: str, words: str = "", raster: bool = False) -> str:
    """A description with the numbers it marks as read off the plot replaced by "(read off the plot)". Its other
    numbers stay: the section is never evidence (services.evidence leaves it out), and replacing every tick label a text
    layer lacks filled round 26's first sections with "(read off the plot)" where a mostly-pictured figure printed its
    ticks."""
    return MARKED_RE.sub("(read off the plot)", str(text or ""))


def _series_line(s: dict, words: str, raster: bool) -> str:
    pts = [(p.get("x", ""), p.get("y", "")) for p in s["points"]]
    name = _scrub(s.get("series") or "values", words, raster)
    xs = [_scrub(x, words, raster) for x, _ in pts]
    if all(_printed(y, words, raster) for _, y in pts):
        return f"{name}: " + ", ".join(f"{x} {y}".strip() for x, (_, y) in zip(xs, pts))
    known = [(x, v, raw) for x, (raw, y) in zip(xs, pts) if (v := _number(y)) is not None and "unreadable" not in str(y).lower()]
    if len(known) < 2:
        return f"{name} (read off the plot): " + ", ".join(x for x in xs if x) if any(xs) else ""
    if all(PURE_NUM_RE.match(str(raw).strip()) for _, _, raw in known):   # a line over numeric x: its trend
        lo, hi = min(v for _, v, _ in known), max(v for _, v, _ in known)
        first, last = known[0][1], known[-1][1]
        trend = "about flat" if hi == lo or abs(last - first) < 0.1 * (hi - lo) else ("rises" if last > first else "falls")
        peak = max(range(len(known)), key=lambda i: known[i][1])
        where = ("at the start" if peak == 0 else "at the end" if peak == len(known) - 1
                 else f"at {known[peak][0]}" if "(read off the plot)" not in known[peak][0] else "in between")
        ends = f" from {known[0][0]} to {known[-1][0]}" if "(read off the plot)" not in known[0][0] + known[-1][0] else ""
        return f"{name} (read off the plot): {trend}{ends}, highest {where}"
    order = [x for x, v, _ in sorted(known, key=lambda t: -t[1])]
    top = max(v for _, v, _ in known)
    if top > 0 and all(v >= 0 for _, v, _ in known):
        zero = [x for x, v, _ in known if v <= 0.03 * top]
        order = [x for x in order if x not in zero]
        return (f"{name} (read off the plot, highest first): " + ", ".join(order)
                + (f"; about zero: {', '.join(zero)}" if zero else ""))
    return f"{name} (read off the plot, highest first): " + ", ".join(order)


def figure_text(f: dict) -> str:
    """One figure as it appears in the section (whole panels only, up to MAX_FIGURE_CHARS)."""
    words = " ".join(f.get("words") or []) + " " + str(f.get("caption") or "")   # what the paper prints of it
    raster = bool(f.get("raster")) or len(f.get("words") or []) < 3   # images, or no text layer: values as the model read them
    caption = re.sub(rf"^\s*(?:Figure|Fig\.?)\s*{f['n']}\s*[:.|]?\s*", "", str(f.get("caption") or ""), flags=re.I)
    lines = [f"{f['figure']} (page {f['page']}): {_cut(caption, 400)}"]
    size, left = len(lines[0]), 0
    for p in f["panels"]:
        bits = [p["shows"]] if p["shows"] else []
        if p["instance"]:
            bits.append("its case: " + p["instance"])
        marks = p["elements"][:CHART_MARKS] if "chart" in p["kind"].lower() else p["elements"]
        if marks:
            bits.append("marks: " + "; ".join(marks))
        if p["arrows"]:
            bits.append("arrows: " + "; ".join(p["arrows"]))
        bits = [_scrub(b, words, raster) for b in bits]
        series = [x for x in (_series_line(s, words, raster) for s in p["data"]) if x]
        if series:
            bits.append("data: " + "; ".join(series))
        head = f"({_scrub(p['id'], words, raster)}) " if p["id"] and p["id"].lower() not in ("whole", "whole figure") else ""
        kind = f"{p['kind']}: " if p["kind"] else ""
        line = f"- {head}{kind}" + " | ".join(bits)
        if size + len(line) + 1 > MAX_FIGURE_CHARS:
            line = _cut(line, max(0, MAX_FIGURE_CHARS - size - 1)) if len(lines) == 1 else ""
            if not line:
                left = len(f["panels"]) - (len(lines) - 1)
                break
        lines.append(line)
        size += len(line) + 1
    if left:
        lines.append(f"- ({left} more panel{'s' if left > 1 else ''} not described here)")
    return "\n".join(lines)


def section(figs: list[dict], max_chars: int = MAX_SECTION_CHARS) -> str:
    """The figures section ("" without figures); whole figures past ``max_chars`` are left out."""
    parts, size = [], len(FIGURES_HEAD) + len(HEAD_NOTE) + 2
    for f in figs:
        t = figure_text(f)
        if size + len(t) + 2 > max_chars:
            break
        parts.append(t)
        size += len(t) + 2
    return f"{FIGURES_HEAD}\n{HEAD_NOTE}\n\n" + "\n\n".join(parts) if parts else ""


def _span(text: str) -> tuple[int, int]:
    """Where the figures section is in a paper text: (start, end), or (-1, -1)."""
    from services.ledger import LEDGER_HEAD
    start = text.find(FIGURES_HEAD)
    if start < 0:
        return -1, -1
    end = text.find(LEDGER_HEAD, start)
    return start, (end if end >= 0 else len(text))


def without_figures(text: str) -> str:
    """The paper text without the figures section: what the paper itself says, for every check of a number or a quote."""
    text = str(text or "")
    start, end = _span(text)
    return text if start < 0 else text[:start].rstrip() + ("\n\n" + text[end:] if end < len(text) else "")


def with_figures(body: str, block: str) -> str:
    """The paper text with the figures section before the tables section (an older figures section replaced)."""
    from services.ledger import LEDGER_HEAD
    text = without_figures(body)
    if not block:
        return text
    at = text.find(LEDGER_HEAD)
    if at >= 0:
        return text[:at].rstrip() + "\n\n" + block + "\n\n" + text[at:]
    return text.rstrip() + "\n\n" + block


def figure_refs(text: str) -> list[int]:
    """The figure numbers a text names, in order: "Figure 3.a", "Fig. 3", "Figures 2 and 3", "Figs. 2-4"."""
    out: list[int] = []
    for m in REFS_RE.finditer(str(text or "")):
        nums = [int(x) for x in re.findall(r"\d{1,3}", m.group(1))]
        if re.search(r"\d\s*(?:–|-|to)\s*\d", m.group(1)) and len(nums) == 2 and 0 < nums[1] - nums[0] <= 5:
            nums = list(range(nums[0], nums[1] + 1))
        out += [n for n in nums if n not in out]
    return out


def figure_block(paper_text: str, ref) -> str:
    """The section's description of the figure that ``ref`` names ("Figure 3.a", "Fig. 3", or the number), or ""."""
    if isinstance(ref, int):
        n = ref
    else:
        m = REF_RE.search(str(ref or ""))
        if not m:
            return ""
        n = int(m.group(1))
    text = str(paper_text or "")
    start, end = _span(text)
    if start < 0:
        return ""
    at = text.find(f"\nFigure {n} (page ", start, end)
    if at < 0:
        return ""
    stop = text.find("\n\n", at + 1, end)
    return text[at + 1:stop if stop >= 0 else end].strip()


# ----------------------------------------------------------------------------- reading

def _cache_path(cache_dir, pdf: bytes, engine) -> Optional[Path]:
    if not cache_dir:
        return None
    model = re.sub(r"[^\w.-]+", "_", str(getattr(engine, "model", "") or "vision"))[:60]
    return Path(cache_dir) / f"{hashlib.sha256(pdf).hexdigest()[:20]}.{model}.v{VERSION}.json"


async def read_figures(engine, pdf: bytes, cache_dir=None, limit: int = MAX_FIGURES, concurrency: int = 4) -> list[dict]:
    """Every figure of the PDF (the first ``limit`` by number) as the vision model describes it; a figure it cannot
    describe is left out. The readings are kept in ``cache_dir`` by the file's hash, the model and VERSION, and only when
    every figure was read (a figure lost to a timeout is asked again next time)."""
    from engines.base import GenerationConfig, ImageContent, Message
    cache = _cache_path(cache_dir, pdf, engine)
    if cache is not None and cache.exists():
        try:
            got = json.loads(cache.read_text(encoding="utf-8"))
            if isinstance(got, list):
                return got
        except Exception:  # noqa: BLE001 - a damaged cache file is read again from the PDF
            pass
    figs = await asyncio.to_thread(_prepare, pdf, limit)
    gate = asyncio.Semaphore(concurrency)

    async def one(fig: dict, png: bytes) -> Optional[dict]:
        async with gate:
            words = ", ".join(dict.fromkeys(fig["words"]))[:1500] or "(none)"
            msg = Message.user(PROMPT.format(label=f"Figure {fig['n']}", words=words),
                               images=[ImageContent.from_base64(base64.b64encode(png).decode(), "image/png")])
            for attempt in range(2):   # an unusable reply (now and then malformed JSON) is asked once more
                try:
                    r = await asyncio.wait_for(engine.generate_text([msg], GenerationConfig(temperature=0.0, max_tokens=8000, thinking=True)),
                                               timeout=TIMEOUT)
                except Exception as e:  # noqa: BLE001 - one unread figure never stops the job
                    print(f"[paper_figures] Figure {fig['n']} not read: {type(e).__name__}: {str(e)[:200]}")
                    return None
                got = clean(r.content, fig)
                if got is not None:
                    return got
                print(f"[paper_figures] Figure {fig['n']}: no usable description" + (", asking again" if attempt == 0 else ""))
            return None

    read = await asyncio.gather(*(one(f, png) for f, png in figs))
    out = [f for f in read if f]
    if cache is not None and out and len(out) == len(figs):
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_suffix(f".tmp{os.getpid()}")
            tmp.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, cache)
        except OSError:
            pass
    return out


async def add_figures(paper, pdf: Optional[bytes], engine, cache_dir=None) -> list[dict]:
    """Reads the figures of ``pdf`` and puts their section into ``paper.body``; [] (and the paper unchanged) without a PDF,
    without a vision model or when reading fails."""
    if not pdf or engine is None or not getattr(engine, "supports_vision", False):
        return []
    try:
        figs = await read_figures(engine, pdf, cache_dir)
    except Exception as e:  # noqa: BLE001 - the figures are an aid: the paper text never depends on them
        print(f"[paper_figures] figures not read: {type(e).__name__}: {e}")
        return []
    block = section(figs)
    if block:
        paper.body = with_figures(paper.body, block)
        paper.stats["figures"] = len(figs)
    return figs
