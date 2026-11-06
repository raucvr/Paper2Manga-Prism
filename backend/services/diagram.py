"""Diagram boards: a teaching panel's mechanism drawn on the running example.

A board can be a diagram instead of a chart: the paper's real objects (the tokens of a sentence, a question and its answer,
documents, models, layers, a teacher's and a student's choices, calls in a day, parcels on two days) as boxes, texts,
arrows, lines, circles and marks on a W x H canvas. The writer only names what it must show ("brief"); the check model
designs the elements from the panel's words and the paper (StoryboarderV2.design_diagrams); code draws them in the boards'
marker style with the book's fonts, so every label is exact in both languages. Why: in a blind comparison of 24 teaching
panels of round 22 (scratchpad g8 -> diag/), the same words beside a diagram were judged clearer than the shipped metaphor
scene in 22 of 24 (clear 2.71 -> 4.04 of 5; the picture's own help 2.25 -> 3.79), and the shipped scene was never the best.

services.boards calls in here for the "diagram" kind: normalize(), problems() (the layout lint: a text that overlaps
another, runs out of its box or off the canvas, a line through a label, boxes cut into each other, a drawing that names
nothing), texts() (translation and fact-check paths), lines() (the labels in reading order) and describe() (what a reader
takes from the drawing, in words: the boxes with what is written in them, the arrows between them; for the checkers, the
writer and the quiz transcript).
"""
from __future__ import annotations

import math
import re
from typing import Optional

# the canvas the designer draws on (x right, y down). Round 23 drew on 1200 x 800 with fonts of 24-30: the board a panel
# leaves for a diagram is short and wide (the cast stand in front of its lower half; a median of 850 x 500 px with the
# title), so the drawing shrank to fit its height and its smallest text came out at 9-19 px (median 14) beside 44 px
# bubbles, and the images-only reader missed three answers that the same panels' text gave. A 2:1 canvas fills that
# board by its width, and the fonts start where a reader on a phone can still read them.
W, H = 1200, 600
TYPES = ("box", "text", "arrow", "line", "circle", "mark")
TONES = ("ink", "red", "blue", "grey")
FILLS = ("none", "tint", "solid")
SIZE_MIN, SIZE_DEFAULT, SIZE_MAX = 32.0, 38.0, 60.0
MAX_ELEMENTS, MAX_TEXTS = 40, 14
TEXT_LIMIT = (6, 14)             # per label: words of Latin text, characters of CJK text
FIT_MIN = 0.72                   # a box's text may shrink to this share of its size before it counts as not fitting
SLACK = 4.0                      # canvas units two texts may touch before they count as overlapping

_TYPE_ALIASES = {"rect": "box", "rectangle": "box", "node": "box", "card": "box", "label": "text", "caption": "text",
                 "connector": "arrow", "edge": "arrow", "polyline": "line", "path": "line", "dot": "circle",
                 "ellipse": "circle", "node_circle": "circle", "check": "mark", "tick": "mark", "cross": "mark", "x": "mark"}
_TONE_ALIASES = {"black": "ink", "dark": "ink", "default": "ink", "text": "ink", "accent": "red", "highlight": "red",
                 "main": "red", "ours": "red", "result": "red", "gray": "grey", "muted": "grey", "light": "grey",
                 "secondary": "blue", "other": "blue", "rival": "blue", "baseline": "blue"}
_CJK_RE = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿ｦ-ﾟ]")


# ----------------------------------------------------------------------------- spec

def _num(v, default: Optional[float] = None) -> Optional[float]:
    if isinstance(v, bool):
        return default
    try:
        x = float(str(v).strip().rstrip("px")) if isinstance(v, str) else float(v)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def _point(v) -> Optional[list]:
    if isinstance(v, dict):
        x, y = _num(v.get("x")), _num(v.get("y"))
    elif isinstance(v, (list, tuple)) and len(v) >= 2:
        x, y = _num(v[0]), _num(v[1])
    else:
        return None
    return None if x is None or y is None else [round(x, 1), round(y, 1)]


def _points(v) -> Optional[list]:
    """A list of points: [[x, y], ...], [{"x": .., "y": ..}, ...] or one flat list [x1, y1, x2, y2, ...] ("via": [600, 200]
    is one corner, not two broken points). None when an entry is no point."""
    if not isinstance(v, (list, tuple)):
        return None
    if v and all(not isinstance(x, (list, tuple, dict)) for x in v):
        nums = [_num(x) for x in v]
        if len(v) % 2 or any(x is None for x in nums):
            return None
        return [[round(nums[k], 1), round(nums[k + 1], 1)] for k in range(0, len(nums), 2)]
    pts = [_point(x) for x in v]
    return None if any(p is None for p in pts) else pts


def _text(v) -> str:
    if isinstance(v, bool) or not isinstance(v, (str, int, float)):
        return ""
    return re.sub(r"\s+", " ", str(v)).strip()


def _hex_tone(s: str) -> Optional[str]:
    m = re.fullmatch(r"#?([0-9a-f]{6}|[0-9a-f]{3})", s)
    if not m:
        return None
    h = m.group(1)
    r, g, b = (int(h[i] * 2, 16) for i in range(3)) if len(h) == 3 else (int(h[i:i + 2], 16) for i in (0, 2, 4))
    if r > g + 50 and r > b + 50:
        return "red"
    if b > r + 40 and b >= g:
        return "blue"
    if max(r, g, b) - min(r, g, b) < 40 and min(r, g, b) > 95:
        return "grey"
    return "ink"


def _tone(v, default: str = "ink") -> str:
    s = str(v or "").strip().lower()
    s = _TONE_ALIASES.get(s, s)
    return s if s in TONES else (_hex_tone(s) or default)


def _bool(v) -> bool:
    return v is True or str(v).strip().lower() in ("true", "yes", "1", "bold", "dashed")


def _size(v) -> float:
    s = _num(v, SIZE_DEFAULT)
    return round(min(SIZE_MAX, max(SIZE_MIN, s)), 1)


def _element(raw, i: int, notes: list) -> Optional[dict]:
    if not isinstance(raw, dict):
        notes.append(f"element {i} is not an object")
        return None
    t = str(raw.get("type") or raw.get("kind") or raw.get("shape") or "").strip().lower()
    mark = t if t in ("check", "tick", "cross", "x") else ""
    t = _TYPE_ALIASES.get(t, t)
    if t not in TYPES:
        notes.append(f"element {i} has an unknown type {t!r}; use one of {', '.join(TYPES)}")
        return None
    out: dict = {"type": t, "tone": _tone(raw.get("tone") or raw.get("color") or raw.get("stroke"))}
    text = _text(raw.get("text") if raw.get("text") is not None else raw.get("label") if t in ("box", "circle", "text") else "")
    if t in ("box", "circle", "text") and text:
        out["text"] = text
    if t in ("box", "circle", "text", "arrow", "line"):
        if _bool(raw.get("bold")):
            out["bold"] = True
    if t == "box":
        x, y, w, h = (_num(raw.get(k)) for k in ("x", "y", "w", "h"))
        w = w if w is not None else _num(raw.get("width"))
        h = h if h is not None else _num(raw.get("height"))
        if None in (x, y, w, h) or w <= 0 or h <= 0:
            notes.append(f"element {i} (box) needs x, y, w and h, with w and h above 0")
            return None
        out.update({"x": round(x, 1), "y": round(y, 1), "w": round(w, 1), "h": round(h, 1)})
    elif t in ("text", "circle", "mark"):
        x, y = _num(raw.get("x", raw.get("cx"))), _num(raw.get("y", raw.get("cy")))
        if x is None or y is None:
            notes.append(f"element {i} ({t}) needs x and y")
            return None
        out.update({"x": round(x, 1), "y": round(y, 1)})
        if t == "text" and not text:
            notes.append(f"element {i} (text) has no text")
            return None
        if t == "circle":
            r = _num(raw.get("r", raw.get("radius")))
            if r is None or r <= 0:
                notes.append(f"element {i} (circle) needs r above 0")
                return None
            out["r"] = round(r, 1)
        if t == "mark":
            kind = str(raw.get("mark") or raw.get("symbol") or mark or "cross").strip().lower()
            out["mark"] = "check" if kind in ("check", "tick", "ok", "yes") else "cross"
            out["size"] = round(min(120.0, max(20.0, _num(raw.get("size"), 44.0))), 1)
    else:   # arrow, line
        pts = _points(raw.get("points")) or []
        if raw.get("points") and len(pts) < 2:
            notes.append(f"element {i} ({t}): its points are not a list of [x, y] points")
        if len(pts) < 2:
            a, b = _point(raw.get("from")), _point(raw.get("to"))
            via = _points(raw.get("via")) if raw.get("via") else []
            if via is None:
                notes.append(f"element {i} ({t}): its via is not a list of [x, y] points, so it is drawn straight")
                via = []
            # an end named by a box's text ("from": "Question") is placed on that box's edge once every box is known
            names = [v.strip() for v in (raw.get("from"), raw.get("to")) if isinstance(v, str) and v.strip()]
            if (a is None or b is None) and len(names) + (a is not None) + (b is not None) == 2:
                out["_ends"] = [raw.get("from") if a is None else a, raw.get("to") if b is None else b]
                out["points"] = via
            else:
                pts = [a] + via + [b] if a and b else []
        if "_ends" not in out:
            if len(pts) < 2:
                notes.append(f"element {i} ({t}) needs from and to (or points)")
                return None
            out["points"] = pts
        if t == "arrow":
            heads = str(raw.get("heads") or raw.get("head") or "end").strip().lower()
            out["heads"] = heads if heads in ("end", "both", "start") else "end"
        if _bool(raw.get("curved")) and (len(out["points"]) >= 3 or "_ends" in out and out["points"]):
            out["curved"] = True
        label = _text(raw.get("label") if raw.get("label") is not None else raw.get("text"))
        if label:
            out["label"] = label
    if t in ("box", "circle"):
        fill = str(raw.get("fill") or "none").strip().lower()
        out["fill"] = fill if fill in FILLS else ("solid" if _hex_tone(fill) in ("red", "blue", "ink") else "tint" if fill.startswith("#") else "none")
    if t in ("box", "arrow", "line", "circle") and _bool(raw.get("dashed")):
        out["dashed"] = True
    if t in ("box", "circle", "text", "arrow", "line") and (out.get("text") or out.get("label")):
        out["size"] = _size(raw.get("size", raw.get("font_size", raw.get("font-size"))))
    if t == "text":
        anchor = str(raw.get("anchor") or raw.get("align") or "middle").strip().lower()
        out["anchor"] = {"left": "start", "center": "middle", "centre": "middle", "right": "end"}.get(anchor, anchor)
        if out["anchor"] not in ("start", "middle", "end"):
            out["anchor"] = "middle"
    return out


def normalize(spec: dict, out: dict, notes: list) -> Optional[dict]:
    """The canonical diagram: elements (each normalized, unusable ones dropped with a note) and the brief (what the
    writer asked it to show; never drawn). A diagram whose elements are not designed yet keeps its brief and an empty
    element list (``designed`` tells them apart); with neither it is unusable."""
    raw = spec.get("elements")
    elements = []
    if isinstance(raw, list):
        for i, e in enumerate(raw):
            el = _element(e, i, notes)
            if el is not None:
                elements.append(el)
    elements = _named_ends(elements, notes)
    # the board draws its title and subtitle above the diagram: a designer's copy inside it says them twice (both
    # diagrams of round 23's first dry run opened with their board's title)
    heads = {_head_key(out.get("title")), _head_key(out.get("subtitle"))} - {""}
    elements = [e for e in elements if not (e["type"] == "text" and _head_key(e.get("text")) in heads)]
    brief = _text(spec.get("brief") or spec.get("draw") or spec.get("shows") or "")
    if not elements and not brief:
        notes.append("the diagram has no elements and no brief (say in \"brief\" what it must draw)")
        return None
    out["elements"] = elements
    if brief:
        out["brief"] = brief
    # what the drawing was designed for (StoryboarderV2._diagram_key): dropped here, every rewrite round's normalization
    # sent every diagram back to the designer (round 23's first dry run drew panel 5 four times)
    key = spec.get("designed_for")
    if isinstance(key, str) and key.strip():
        out["designed_for"] = key.strip()
    return out


def _named_ends(elements: list, notes: list) -> list:
    """Arrows whose ends name a box or circle by its text ("from": "Question", "to": "Retriever") start and end on the
    edges of those shapes, facing each other (a designer wrote them so; they were dropped without a word, and the
    diagram kept three boxes and no arrows). An end that names no shape drops the arrow with a note."""
    shapes = [e for e in elements if e["type"] in ("box", "circle") and e.get("text")]

    def find(name):
        key = _head_key(name)
        hits = [e for e in shapes if _head_key(e["text"]) == key] or [e for e in shapes if _head_key(e["text"]).startswith(key)]
        return hits[0] if len(hits) == 1 else None

    def centre(e):
        return (e["x"] + e["w"] / 2, e["y"] + e["h"] / 2) if e["type"] == "box" else (e["x"], e["y"])

    def edge(e, toward, gap: float = 6.0):
        cx, cy = centre(e)
        dx, dy = toward[0] - cx, toward[1] - cy
        d = math.hypot(dx, dy) or 1.0
        if e["type"] == "circle":
            k = (e["r"] + gap) / d
        else:
            k = min((e["w"] / 2 + gap) / abs(dx) if dx else math.inf, (e["h"] / 2 + gap) / abs(dy) if dy else math.inf)
        return [round(cx + dx * min(k, 1.0), 1), round(cy + dy * min(k, 1.0), 1)]

    out = []
    for i, e in enumerate(elements):
        ends = e.pop("_ends", None)
        if ends is None:
            out.append(e)
            continue
        named = [find(x) if isinstance(x, str) else None for x in ends]
        if any(isinstance(x, str) and m is None for x, m in zip(ends, named)):
            missing = [x for x, m in zip(ends, named) if isinstance(x, str) and m is None]
            many = [x for x in missing if sum(1 for o in shapes if _head_key(o["text"]) == _head_key(x)) > 1]
            notes.append(f"an {e['type']} names {', '.join(map(repr, missing))} as an end, and "
                         + ("several shapes have that text" if many else "no box or circle has that text")
                         + ": give its ends as [x, y] points")
            continue
        via = e.get("points") or []
        a_shape, b_shape = named
        if a_shape is not None and b_shape is not None and not via:
            ra, rb = _rect_of(a_shape), _rect_of(b_shape)
            if a_shape is b_shape:
                notes.append(f"an {e['type']} from {a_shape['text']!r} to itself needs via points around the shape")
                continue
            if _inside(ra, rb) or _inside(rb, ra):
                notes.append(f"an {e['type']} between {a_shape['text']!r} and {b_shape['text']!r}, one inside the other, "
                             "needs its ends as [x, y] points")
                continue
        a = ends[0] if named[0] is None else edge(named[0], via[0] if via else centre(named[1]) if named[1] else ends[1])
        b = ends[1] if named[1] is None else edge(named[1], via[-1] if via else centre(named[0]) if named[0] else ends[0])
        if not via and (math.dist(a, b) < 12 or a_shape is not None and b_shape is not None and (
                (b[0] - a[0]) * (centre(b_shape)[0] - centre(a_shape)[0]) + (b[1] - a[1]) * (centre(b_shape)[1] - centre(a_shape)[1]) <= 0)):
            notes.append(f"an {e['type']} between {', '.join(repr(x) for x in ends if isinstance(x, str))} has no room: the "
                         "shapes are too close for an arrow; move them apart or give its ends as [x, y] points")
            continue
        e["points"] = [a] + via + [b]
        if e.get("curved") and len(e["points"]) < 3:
            e.pop("curved")
        out.append(e)
    return out


def _head_key(text) -> str:
    return " ".join(str(text or "").lower().split()).strip(" .:")


def designed(board) -> bool:
    return isinstance(board, dict) and board.get("kind") == "diagram" and bool(board.get("elements"))


def texts(board: dict) -> list[tuple[str, str]]:
    """Every text of the elements as (path, text), in reading order (top to bottom, then left to right)."""
    out = []
    for i, e in enumerate(board.get("elements") or []):
        if e.get("text"):
            out.append((f"elements.{i}.text", e["text"], _anchor_y(e), _anchor_x(e)))
        if e.get("label"):
            (ax, ay), (bx, by) = _label_seg(e)
            out.append((f"elements.{i}.label", e["label"], (ay + by) / 2, (ax + bx) / 2))
    out.sort(key=lambda t: (round(t[2] / 40), t[3]))   # rows of about 40 canvas units, then left to right
    return [(p, t) for p, t, _, _ in out]


def lines(board: dict) -> list[str]:
    """What a reader reads on the diagram, in reading order (title first)."""
    return [t for _, t in texts(board)]


# ----------------------------------------------------------------------------- geometry

def _anchor_y(e: dict) -> float:
    if e["type"] == "box":
        return e["y"] + e["h"] / 2
    if e["type"] in ("arrow", "line"):
        (_, ay), (_, by) = _label_seg(e)
        return (ay + by) / 2
    return e["y"]


def _anchor_x(e: dict) -> float:
    if e["type"] == "box":
        return e["x"]
    if e["type"] in ("arrow", "line"):
        (ax, _), (bx, _) = _label_seg(e)
        return (ax + bx) / 2
    return e["x"]


def _longest(points: list) -> tuple:
    segs = list(zip(points, points[1:]))
    return max(segs, key=lambda s: math.dist(s[0], s[1]))


def _curve(points: list, steps: int = 12) -> list:
    """A smooth curve through every point, its vias included (Catmull-Rom): the designer is told that "curved bends it
    through them", and a Bezier pulled by the vias never reached them (round 23's review: an arc meant to pass x=1000
    turned at x=900, and middle vias were ignored)."""
    pts = [tuple(p) for p in points]
    ext = [pts[0]] + pts + [pts[-1]]
    out = [pts[0]]
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        for k in range(1, steps + 1):
            t = k / steps
            out.append(tuple(0.5 * (2 * p1[j] + (p2[j] - p0[j]) * t + (2 * p0[j] - 5 * p1[j] + 4 * p2[j] - p3[j]) * t * t
                                    + (3 * p1[j] - p0[j] - 3 * p2[j] + p3[j]) * t ** 3) for j in (0, 1)))
    return out


def path(e: dict) -> list:
    """The points a line or arrow is drawn through (a curve is sampled)."""
    return _curve(e["points"]) if e.get("curved") and len(e["points"]) >= 3 else [tuple(p) for p in e["points"]]


def _label_seg(e: dict) -> tuple:
    """The stretch of a line or arrow its label sits beside: on a curve the stretch around half its length (however
    uneven its legs; a middle sample by index sat at the via), else its longest segment."""
    pts = path(e)
    if not (e.get("curved") and len(pts) > 4):
        return _longest(pts)
    steps = [math.dist(a, b) for a, b in zip(pts, pts[1:])]
    half, run = sum(steps) / 2, 0.0
    for k, d in enumerate(steps):
        if run + d >= half:
            lo, hi = max(0, k - 1), min(len(pts) - 1, k + 2)
            return pts[lo], pts[hi]
        run += d
    return _longest(pts)


class Placed:
    """One text as it sits on the canvas: its element, role (box, circle, text or label), wrapped lines, size and rect."""
    __slots__ = ("i", "role", "lines", "size", "bold", "rect", "fits", "tone")

    def __init__(self, i, role, lines, size, bold, rect, fits, tone):
        self.i, self.role, self.lines, self.size, self.bold, self.rect, self.fits, self.tone = i, role, lines, size, bold, rect, fits, tone


def _fit(e: dict, area: tuple, pen, wrap, bold: bool) -> tuple:
    """A shape's text centred in ``area``, shrunk step by step to FIT_MIN of its size until it fits: (lines, size, width,
    height, fits, rect)."""
    x1, y1, x2, y2 = area
    s = e["size"]
    while True:
        pad = 0.28 * s
        max_w, max_h = max(1.0, x2 - x1 - 2 * pad), max(1.0, y2 - y1 - 0.5 * pad)
        ls = wrap(pen, e["text"], max_w, s, bold) or [e["text"]]
        w = max(pen.width(l, s, bold) for l in ls)
        h = pen.lh(s) * len(ls)
        fits = w <= max_w + 0.5 and h <= max_h + 0.5
        if fits or s * 0.94 < e["size"] * FIT_MIN:
            break
        s *= 0.94
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return ls, s, w, h, fits, (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def place(board: dict, pen, wrap) -> list[Placed]:
    """Where every text of the diagram sits on the canvas, measured with ``pen`` (services.boards._Pen at canvas units)
    and wrapped with ``wrap(pen, text, max_w, size, bold) -> lines``. A box's or circle's text is centred in it and may
    shrink to FIT_MIN of its size (a box drawn around others names them in its top band); a free text is one line at its
    anchor (y is its middle); an arrow's or line's label sits beside the middle of its longest segment, or of the curve."""
    out = []
    for i, e in enumerate(board.get("elements") or []):
        bold = bool(e.get("bold"))
        if e["type"] in ("box", "circle") and e.get("text"):
            if e["type"] == "box":
                x1, y1, x2, y2 = e["x"], e["y"], e["x"] + e["w"], e["y"] + e["h"]
            else:
                k = e["r"] * 0.72
                x1, y1, x2, y2 = e["x"] - k, e["y"] - k, e["x"] + k, e["y"] + k
            own = _rect_of(e)
            inner = [r for j, o in enumerate(board.get("elements") or []) if j != i and o["type"] in ("box", "circle")
                     and not is_step(o) for r in [_rect_of(o)] if _inside(r, own) and not _inside(own, r)]
            band = (x1, y1, x2, max(y1 + 1.0, min(r[1] for r in inner) - 4.0)) if inner else None
            best = _fit(e, (x1, y1, x2, y2), pen, wrap, bold)
            if band and any(_overlap(best[5], r) for r in inner):
                # a box drawn around others names the group in its top band, never on top of the boxes inside (a step
                # badge in a card's corner is no member: the card's text stays centred)
                best = _fit(e, band, pen, wrap, bold)
            out.append(Placed(i, e["type"], best[0], best[1], bold, best[5], best[4], e["tone"]))
            continue
        elif e["type"] == "text":
            s = e["size"]
            w, h = pen.width(e["text"], s, bold), pen.lh(s)
            x = e["x"] - w / 2 if e["anchor"] == "middle" else e["x"] - w if e["anchor"] == "end" else e["x"]
            out.append(Placed(i, "text", [e["text"]], s, bold, (x, e["y"] - h / 2, x + w, e["y"] + h / 2), True, e["tone"]))
        elif e["type"] in ("arrow", "line") and e.get("label"):
            s = e["size"]
            w, h = pen.width(e["label"], s, bold), pen.lh(s)
            (ax, ay), (bx, by) = _label_seg(e)
            mx, my = (ax + bx) / 2, (ay + by) / 2
            if abs(bx - ax) >= abs(by - ay):     # mostly horizontal: the label above the line
                rect = (mx - w / 2, my - 8 - h, mx + w / 2, my - 8)
            else:                                # mostly vertical: the label to its right
                rect = (mx + 10, my - h / 2, mx + 10 + w, my + h / 2)
            out.append(Placed(i, "label", [e["label"]], s, bold, rect, True, e["tone"]))
    return out


def bounds(board: dict, placed: list) -> tuple[float, float, float, float]:
    """The box around everything drawn (elements and placed texts), on the canvas."""
    xs, ys = [], []
    for e in board.get("elements") or []:
        r = _rect_of(e)
        if r:
            xs += [r[0], r[2]]
            ys += [r[1], r[3]]
        for x, y in (path(e) if e.get("points") else []):
            xs.append(x)
            ys.append(y)
    for p in placed:
        xs += [p.rect[0], p.rect[2]]
        ys += [p.rect[1], p.rect[3]]
    if not xs:
        return 0.0, 0.0, float(W), float(H)
    return min(xs), min(ys), max(xs), max(ys)


def _overlap(a, b, slack: float = SLACK) -> bool:
    return min(a[2], b[2]) - max(a[0], b[0]) > slack and min(a[3], b[3]) - max(a[1], b[1]) > slack


def _inside(a, b, slack: float = 2.0) -> bool:
    """a lies within b."""
    return a[0] >= b[0] - slack and a[1] >= b[1] - slack and a[2] <= b[2] + slack and a[3] <= b[3] + slack


def _seg_hits_rect(p, q, r, inset: float = 3.0) -> bool:
    """The segment pq crosses the rect r shrunk by ``inset`` (Liang-Barsky clipping)."""
    x1, y1, x2, y2 = r[0] + inset, r[1] + inset, r[2] - inset, r[3] - inset
    if x2 <= x1 or y2 <= y1:
        return False
    dx, dy = q[0] - p[0], q[1] - p[1]
    t0, t1 = 0.0, 1.0
    for pp, qq in ((-dx, p[0] - x1), (dx, x2 - p[0]), (-dy, p[1] - y1), (dy, y2 - p[1])):
        if pp == 0:
            if qq < 0:
                return False
            continue
        t = qq / pp
        if pp < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1:
            return False
    return True


def _rect_of(e: dict):
    if e["type"] == "box":
        return (e["x"], e["y"], e["x"] + e["w"], e["y"] + e["h"])
    if e["type"] == "circle":
        return (e["x"] - e["r"], e["y"] - e["r"], e["x"] + e["r"], e["y"] + e["r"])
    if e["type"] == "mark":
        s = e["size"] / 2
        return (e["x"] - s, e["y"] - s, e["x"] + s, e["y"] + s)
    return None


def _name(board: dict, i: int) -> str:
    e = board["elements"][i]
    said = e.get("text") or e.get("label") or ""
    return f'element {i} ({e["type"]}{": " + repr(said[:40]) if said else ""})'


def problems(board: dict, pen, wrap) -> list[str]:
    """The layout lint of a designed diagram, in words the designer can act on: too many elements or texts, a label
    too long, a font below SIZE_MIN, anything off the canvas, a box's text that does not fit even when shrunk, two texts
    that overlap, a free text or label cut by a box's edge, a line or arrow through a text that is not its own, two
    boxes that cut into each other. [] for a diagram without elements (it is not designed yet)."""
    els = board.get("elements") or []
    if not els:
        return []
    out = []
    if len(els) > MAX_ELEMENTS:
        out.append(f"the diagram has {len(els)} elements; keep it within {MAX_ELEMENTS} (a reader takes in a few things at once)")
    placed = place(board, pen, wrap)
    if len(placed) > MAX_TEXTS:
        out.append(f"the diagram has {len(placed)} texts; keep it within {MAX_TEXTS}")
    for p in placed:
        e = els[p.i]
        raw = e.get("text") or e.get("label") or ""
        if _CJK_RE.search(raw):
            n = len(_CJK_RE.findall(raw)) + len(re.findall(r"[A-Za-z0-9]+", raw))
            if n > TEXT_LIMIT[1]:
                out.append(f"{_name(board, p.i)} has {n} characters; keep each label within {TEXT_LIMIT[1]}")
        elif len(raw.split()) > TEXT_LIMIT[0]:
            out.append(f"{_name(board, p.i)} has {len(raw.split())} words; keep each label within {TEXT_LIMIT[0]} (the words "
                       "are in the bubbles beside it)")
        if not p.fits:
            out.append(f"the text of {_name(board, p.i)} does not fit in it even at a smaller size: make the "
                       f"{'box' if e['type'] == 'box' else 'circle'} bigger or the text shorter")
        elif p.size < SIZE_MIN - 0.05:   # only a box's or circle's text shrinks: below the floor a phone cannot read it
            out.append(f"the text of {_name(board, p.i)} fits only when shrunk to {p.size:.0f}, below the smallest size "
                       f"{SIZE_MIN:.0f}: make the {'box' if e['type'] == 'box' else 'circle'} bigger or the text shorter")
        r = p.rect
        if r[0] < -2 or r[1] < -2 or r[2] > W + 2 or r[3] > H + 2:
            out.append(f"{_name(board, p.i)} runs off the {W} x {H} canvas")
    for i, e in enumerate(els):
        r = _rect_of(e)
        pts = e.get("points") or []
        xs = [x for x, _ in pts] + ([r[0], r[2]] if r else [])
        ys = [y for _, y in pts] + ([r[1], r[3]] if r else [])
        if e["type"] == "text":
            continue
        if xs and (min(xs) < -2 or min(ys) < -2 or max(xs) > W + 2 or max(ys) > H + 2):
            out.append(f"{_name(board, i)} runs off the {W} x {H} canvas")
    for a in range(len(placed)):
        for b in range(a + 1, len(placed)):
            pa, pb = placed[a], placed[b]
            if _overlap(pa.rect, pb.rect):
                out.append(f"the texts of {_name(board, pa.i)} and {_name(board, pb.i)} overlap: move one of them")
    boxes = [(i, _rect_of(e)) for i, e in enumerate(els) if e["type"] in ("box", "circle")]
    for p in placed:
        if p.role in ("box", "circle"):   # a shape's own text on another shape (a group's name on the boxes it holds)
            own = _rect_of(els[p.i])
            for i, r in boxes:
                if i != p.i and _overlap(p.rect, r, 1.0) and not _inside(own, r):
                    out.append(f"the text of {_name(board, p.i)} lies on {_name(board, i)}: move one of them or make the "
                               f"{els[p.i]['type']} bigger")
            continue
        for i, r in boxes:
            if i != p.i and _overlap(p.rect, r, 1.0) and not _inside(p.rect, r) and not _inside(r, p.rect):
                out.append(f"the text of {_name(board, p.i)} is cut by the edge of {_name(board, i)}: put it inside or clear of it")
    # a drawing that names nothing teaches nothing: Flexibility Trap's dry run drew eight blank boxes with step numbers
    # (its transcript read "1 | 2 | ... | 8"), and its reader learned no step
    named = [t for _, t in texts(board) if re.search(r"[^\W\d_]", t)]
    if len(named) < 2:
        out.append("the diagram names almost nothing (a reader sees shapes and numbers without words): label the things "
                   "it draws with the panel's own terms, a few words each")
    shapes = [e for e in els if e["type"] == "box"]

    def holds(b: dict) -> bool:   # a box named by what is drawn in it: a labelled shape, a mark, a text
        r = _rect_of(b)
        return any(o is not b and _inside(_rect_of(o), r) and (o["type"] == "mark" or re.search(r"[^\W\d_]", o.get("text") or ""))
                   for o in els if o["type"] in ("box", "circle", "mark"))
    if len(shapes) >= 3 and not any(e.get("text") or holds(e) for e in shapes) and not any(
            p.role == "text" and any(_inside(p.rect, _rect_of(b)) for b in shapes) for p in placed):
        out.append(f"all {len(shapes)} boxes are empty: write in each what it stands for (a token, a step, a model), so a "
                   "reader can tell them apart")
    for i, e in enumerate(els):
        if e["type"] not in ("arrow", "line"):
            continue
        segs = list(zip(path(e), path(e)[1:]))
        for p in placed:
            if p.i == i:
                continue
            if any(_seg_hits_rect(a, b, p.rect) for a, b in segs):
                out.append(f"{_name(board, i)} runs through the text of {_name(board, p.i)}: route it around the text or move the text")
    for x in range(len(boxes)):
        for y in range(x + 1, len(boxes)):
            (i, r), (j, q) = boxes[x], boxes[y]
            if _overlap(r, q, 2.0) and not _inside(r, q) and not _inside(q, r):
                out.append(f"{_name(board, i)} and {_name(board, j)} cut into each other: move them apart or put one inside the other")
    seen, uniq = set(), []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


# ----------------------------------------------------------------------------- what a reader takes from it

STEP_RE = re.compile(r"\d{1,2}")


def is_step(e: dict) -> bool:
    """A step badge: a small circle with a one- or two-digit number in it (the designer is told to show an order that
    way). A bare number written as a text or in a box is a value ("12" parcels), never an order."""
    return e.get("type") == "circle" and e.get("r", 99) <= 45 and STEP_RE.fullmatch(e.get("text") or "") is not None


def read_texts(board: dict) -> list[str]:
    """Every text a reader reads on the drawing, sorted, with its check and cross marks; step badges aside (a relayout
    may renumber nothing else: a value "8" turned into "6" by a layout-only redesign shipped unread)."""
    els = board.get("elements") or []
    out = [t for path, t in texts(board) if not is_step(els[int(path.split(".")[1])])]
    out += ["✓" if e["mark"] == "check" else "✗" for e in els if e.get("type") == "mark"]
    return sorted(out)


def _dist_to_rect(pt, r) -> float:
    dx = max(r[0] - pt[0], 0.0, pt[0] - r[2])
    dy = max(r[1] - pt[1], 0.0, pt[1] - r[3])
    return math.hypot(dx, dy)


def _dist_to_path(pt, pts) -> float:
    best = math.inf
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        dx, dy = x2 - x1, y2 - y1
        t = 0.0 if dx == dy == 0 else max(0.0, min(1.0, ((pt[0] - x1) * dx + (pt[1] - y1) * dy) / (dx * dx + dy * dy)))
        best = min(best, math.hypot(pt[0] - (x1 + t * dx), pt[1] - (y1 + t * dy)))
    return best


def _clusters(spans: list) -> list:
    """Indexes grouped by overlapping 1-D spans (lo, hi), in order along the axis."""
    order = sorted(range(len(spans)), key=lambda k: spans[k])
    groups, hi = [], -math.inf
    for k in order:
        lo, top = spans[k]
        if groups and lo < hi - 2.0:
            groups[-1].append(k)
            hi = max(hi, top)
        else:
            groups.append([k])
            hi = top
    return groups


def _text_rect(e: dict) -> tuple:
    """About where a free text is drawn (the designer is told 0.55 x the size per character, to leave room)."""
    w = 0.48 * e["size"] * len(e["text"])
    x1 = e["x"] - w / 2 if e["anchor"] == "middle" else e["x"] - w if e["anchor"] == "end" else e["x"]
    return (x1, e["y"] - e["size"] / 2, x1 + w, e["y"] + e["size"] / 2)


def describe(board: dict) -> list[str]:
    """What a reader takes from the drawing, in words: each box or circle with everything written in it ("[Depth 4 ·
    faceplants]"; the same box again and again once: "[layer] ×8"), a step badge beside a thing as its order ("[#1
    Warehouse]"), each free text and mark, then each arrow between the things it joins ("Warehouse → Customer door (first
    try)"). Column by column when the things stand in fewer columns than rows (two sides of a comparison), else row by
    row (a flow, a row of tokens); a box drawn around others comes just before the first of them. Round 23's transcripts
    read a diagram as a list of its labels row by row ("1 | 2 | 3 | 4 | Depth 4 | Depth 16 | ... | faceplants | walks
    upright"), so the checkers and the quiz reader could not tell what belonged to what, nor where an arrow went."""
    els = board.get("elements") or []
    if not els:
        return []
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    shapes = [i for i, e in enumerate(els) if e["type"] in ("box", "circle") and not is_step(e)]
    rect = {i: _rect_of(els[i]) for i in shapes}
    inside: dict[int, list[int]] = {i: [] for i in shapes}
    free: list[int] = []
    for i, e in enumerate(els):
        if e["type"] != "text":
            continue
        holders = [k for k in shapes if _inside((e["x"], e["y"], e["x"], e["y"]), rect[k], 0.0)]
        if holders:
            inside[min(holders, key=lambda k: area(rect[k]))].append(i)
        else:
            free.append(i)

    def words_of(k: int) -> str:
        parts = [els[k].get("text") or ""] + [els[j]["text"] for j in sorted(inside[k], key=lambda j: (els[j]["y"], els[j]["x"]))]
        return " · ".join(x for x in parts if x)

    arrows = [i for i, e in enumerate(els) if e["type"] in ("arrow", "line")]
    order_of: dict[tuple, list[str]] = {}   # ("shape" | "arrow", index) -> its step numbers
    for i, e in enumerate(els):
        if not is_step(e):
            continue
        pt = (e["x"], e["y"])
        shape = min(((_dist_to_rect(pt, rect[k]), k) for k in shapes), default=(math.inf, None))
        arrow = min(((_dist_to_path(pt, path(els[k])), k) for k in arrows), default=(math.inf, None))
        # a badge belongs to a thing; to an arrow only when it sits on it or is clearly nearer to it (a loop-back arrow
        # above a row of badges took all their numbers)
        if arrow[1] is not None and (arrow[0] <= e["r"] + 6 or arrow[0] < 0.5 * shape[0]) and arrow[0] <= 160:
            order_of.setdefault(("arrow", arrow[1]), []).append(e["text"])
        elif shape[1] is not None and shape[0] <= 160:
            order_of.setdefault(("shape", shape[1]), []).append(e["text"])
        else:
            free.append(i)
    marks: dict[int, list[str]] = {}
    loose_marks = []
    for e in els:
        if e["type"] == "mark":
            holders = [k for k in shapes if _inside((e["x"], e["y"], e["x"], e["y"]), rect[k], 0.0)]
            sign = "✓" if e["mark"] == "check" else "✗"
            if holders:
                marks.setdefault(min(holders, key=lambda k: area(rect[k])), []).append(sign)
            else:
                loose_marks.append((_rect_of(e), sign))

    def shape_words(k: int) -> str:
        words = " ".join(x for x in [words_of(k)] + marks.get(k, []) if x)
        steps = " ".join(f"#{x}" for x in order_of.get(("shape", k), []))
        holders = [j for j in shapes if j != k and words_of(j) and _inside(rect[k], rect[j]) and not _inside(rect[j], rect[k])]
        group = words_of(min(holders, key=lambda j: area(rect[j]))) if holders and words else ""
        outer = "" if not group else f"（在{group}内）" if _CJK_RE.search(group + words) else f" (inside {group})"
        return f"{steps} {words}".strip() + outer

    # bars: three or more boxes on one baseline at different heights (a model's spread over next tokens, drawn without
    # values as the designer may) say what they say by their heights, which a list of their texts loses
    bars = [k for k in shapes if els[k]["type"] == "box" and not any(j != k and _inside(rect[j], rect[k]) for j in shapes)]
    height: dict[int, str] = {}
    for k in bars:
        row = [els[j]["h"] for j in bars if abs(els[j]["y"] + els[j]["h"] - els[k]["y"] - els[k]["h"]) <= 4]
        if len(row) >= 3 and max(row) >= 1.5 * min(row):
            r = els[k]["h"] / max(row)
            height[k] = "tallest" if r >= 0.99 else "tall" if r >= 0.66 else "mid" if r >= 0.33 else "short"
    entries = []   # (rect, words)
    for k in shapes:
        words = " · ".join(x for x in (shape_words(k), f"{height[k]} bar" if k in height else "") if x)
        if words:   # a step number beside an empty box still gives an order
            entries.append((rect[k], f"[{words}]"))
    for i in free:
        e = els[i]
        if e["type"] == "text":
            entries.append((_text_rect(e), e["text"]))
        elif is_step(e):   # a badge beside nothing: its number as written
            entries.append((_rect_of(e), f"#{e['text']}"))
    entries += loose_marks
    # a box around others (a group) would join every row and column into one: order the rest, then put it before its
    # first member
    groups = {k for k, (r, _) in enumerate(entries) if any(j != k and _inside(q, r) and not _inside(r, q)
                                                            for j, (q, _) in enumerate(entries))}
    plain = [k for k in range(len(entries)) if k not in groups]
    cols = _clusters([(entries[k][0][0], entries[k][0][2]) for k in plain])
    rows = _clusters([(entries[k][0][1], entries[k][0][3]) for k in plain])
    order: list[int] = []
    if 2 <= len(cols) < len(rows):
        for group in cols:
            order += [plain[k] for k in sorted(group, key=lambda k: (entries[plain[k]][0][1], entries[plain[k]][0][0]))]
    else:
        for group in rows:
            order += [plain[k] for k in sorted(group, key=lambda k: (entries[plain[k]][0][0], entries[plain[k]][0][1]))]
    for g in sorted(groups, key=lambda k: -area(entries[k][0])):   # the biggest group first, so an inner one follows it
        first = next((pos for pos, k in enumerate(order) if _inside(entries[k][0], entries[g][0])), len(order))
        order.insert(first, g)
    folded: list[str] = []   # the same box again and again: once, with its count
    for x in (entries[k][1] for k in order):
        m = re.fullmatch(r"(.*) ×(\d+)", folded[-1]) if folded else None
        if folded and x.startswith("[") and (folded[-1] == x or m and m.group(1) == x):
            folded[-1] = f"{x} ×{int(m.group(2)) + 1 if m else 2}"
        else:
            folded.append(x)

    named = [k for k in shapes if words_of(k)]

    def end_name(pt) -> str:
        """The named thing an arrow's end touches (the smallest of those touching it: an arrow between two boxes of a
        group joins the boxes), else the nearest named thing or free text within 40 units, else "…"."""
        near = [(_dist_to_rect(pt, rect[k]), area(rect[k]), words_of(k)) for k in named]
        near += [(_dist_to_rect(pt, _text_rect(els[i])), 0.0, els[i]["text"]) for i in free if els[i]["type"] == "text"]
        touching = [x for x in near if x[0] <= 8]
        if touching:
            return min(touching, key=lambda x: (x[1], x[0]))[2]
        close = [x for x in near if x[0] <= 40]
        return min(close)[2] if close else "…"

    for k in arrows:
        e = els[k]
        pts = path(e)
        a, b = end_name(pts[0]), end_name(pts[-1])
        sign = "—" if e["type"] == "line" else {"end": "→", "start": "←", "both": "↔"}[e.get("heads", "end")]
        extra = [e["label"]] if e.get("label") else []
        extra += [f"#{x}" for x in order_of.get(("arrow", k), [])]
        if (a == b and a != "…" or "…" in (a, b)) and not extra:
            continue   # a stroke inside one thing (a drawn icon) or from nowhere, unlabelled: nothing to read
        if a == b == "…":
            folded.append(f"{sign} " + ", ".join(extra))
            continue
        folded.append(f"{a} {sign} {b}" + (f" ({', '.join(extra)})" if extra else ""))
    return folded
