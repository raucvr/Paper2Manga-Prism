"""
Art check (v3) — where things are in a text-free panel, and whether the art can carry the typeset text.

locate(): ONE vision call per image (JSON mode, temperature 0, thinking off; retried only when the reply is not usable
JSON, at most 3 tries). The reference sheets of the panel's characters are sent before the panel, so the model can say
whether each character matches its sheet. The reply (0-1000 coordinates) becomes a pixel ``Grounding`` in
grounding_from_json(): names mapped onto the cast ("papi (brown dog)" -> papi), boards accepted as {"corners": [...]},
a bare list of 4 points or 4 numbers (an axis-aligned box), invalid shapes dropped, a character drawn twice recorded
as an extra ("a second kumo").
art_problems(): human-readable problems, used as the next attempt's feedback. The calm band of a scene panel and the
clean part of a board are measured in pixels: the model's boxes are not precise enough to promise room for the text.
The board is measured with services.lettering.clean_rect, the rectangle lettering will draw the chart into, so a board
that passes here is the board lettering gets.
rank_key(): orders the attempts when none is perfect (characters, extras, text, board, band); HIGHER is better, like
jobs._attempt_rank, so the attempt to publish is max(attempts, key=rank_key).
"""

from __future__ import annotations

import base64
import io
import json
import math
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image, ImageFilter

from engines.base import BaseEngine, GenerationConfig, ImageContent, Message
from services.storyboarder_v2 import Cast, PanelV2, _as_list, _match_cast, extract_json

Box = tuple[int, int, int, int]
Point = tuple[int, int]


@dataclass
class Grounding:
    """What is where in one panel, in pixels of the art. services/lettering.py reads exactly these fields."""
    size: tuple[int, int]                                    # (W, H) of the art
    heads: dict[str, Box] = field(default_factory=dict)      # cast name -> head box (x1, y1, x2, y2)
    board: Optional[list[Point]] = None                      # the largest blank board: 4 corners, clockwise from top-left
    bubbles: list[Box] = field(default_factory=list)         # speech bubbles drawn into the art (v3 art should have none)
    text_found: list[str] = field(default_factory=list)      # letters, digits or symbols drawn anywhere: "text (on where)"
    extras: list[str] = field(default_factory=list)          # figures that are not the listed characters; a clone is "a second kumo"
    matches: dict[str, bool] = field(default_factory=dict)   # cast name -> looks like its sheet (only when the model gave a verdict)
    objects: list[Box] = field(default_factory=list)         # the objects that carry a scene's idea (lettering keeps them clear)


class GroundingError(RuntimeError):
    """The vision model gave no usable grounding JSON in MAX_TRIES tries."""


# ----------------------------------------------------------------------------- the vision call

SCALE = 1000          # the model's coordinates are normalized to 0-1000
SLACK = 10            # a box overshooting the frame by up to 1% is clamped; further out it is not a normalized box
MIN_SIDE = 5          # boxes thinner than 0.5% of the panel are noise
MAX_TRIES = 3
REPLY_KEYS = ("characters", "boards", "speech_bubbles", "text_or_symbols", "extra_creatures")

LOCATE_PROMPT = """{refs}The LAST image is one panel of an educational manga. Find things in that panel only (text on reference sheets does not count).
Coordinates are integers normalized to 0-1000 over that whole image (x from its left edge, y from its top edge).
Characters who should be in the panel: {present}.{others}

Return JSON only:
{{"characters": [{{"name": "{example}", "head": [x1, y1, x2, y2], "matches_reference": true}}],
 "boards": [{{"corners": [[x, y], [x, y], [x, y], [x, y]]}}],
 "speech_bubbles": [[x1, y1, x2, y2]],
 "text_or_symbols": [{{"text": "<what is drawn>", "where": "<on what, e.g. the clock face>"}}],
 "extra_creatures": ["<short description>"]}}
- characters: one entry per drawn figure of a named character, named exactly as written above (a character drawn twice gets two entries). head = the box around the head, ears included. matches_reference = false when the figure's species, colors or face differ from its {reference}.
- boards: blank whiteboards, boards, screens or paper sheets big enough to hold a chart: the 4 corners of the blank surface, clockwise from top-left. [] when there is none.
- speech_bubbles: speech or thought balloons drawn in the panel, and chat windows, message boxes, caption boxes or text boxes, even empty ones (cards, papers or blocks that a character holds or that lie on a surface are objects, not text boxes). [] when there is none.
- text_or_symbols: every letter, digit, word, numeral (clock faces included), symbol or scribbled writing drawn anywhere in the panel, one entry per surface that carries it ("text": at most 5 words of it); arrows and simple shapes drawn as objects are not text, but any mark drawn on a whiteboard or board is. [] when there is none.
- extra_creatures: every person, animal or creature that is not one of the named characters, and any figure you cannot name. [] when there is none."""


def locate_prompt(cast: Cast, present: list[str], ref_names: list[str]) -> str:
    looks = {m.name: m.look for m in cast.members}
    described = lambda names: ", ".join(f"{n} ({looks[n]})" if looks.get(n) else n for n in names)
    refs = "".join(f"Image {i + 1} is the reference sheet of {n}. " for i, n in enumerate(ref_names))
    others = [m.name for m in cast.members if m.name not in present]
    return LOCATE_PROMPT.format(
        refs=refs, present=described(present) or "none",
        others=f" These cast members should NOT be in it; report them under characters if they are drawn anyway: {described(others)}." if others else "",
        example=" | ".join(dict.fromkeys([*present, *others])) or "name",
        reference="reference sheet" if ref_names and set(present) <= set(ref_names) else "reference sheet or description")


def _image_size(image: ImageContent) -> tuple[int, int]:
    if image.is_url:
        raise ValueError("locate needs inline image data, not a URL")
    with Image.open(io.BytesIO(base64.b64decode(image.data))) as im:   # reads the header only
        return im.size


async def locate(vision_engine: BaseEngine, image: ImageContent, cast: Cast, present: list[str], sheets: dict[str, ImageContent],
                 usage: Optional[dict] = None, size: Optional[tuple[int, int]] = None) -> Grounding:
    """One vision call → pixel Grounding. Only a reply that is not usable JSON is asked again (at most MAX_TRIES tries);
    after that GroundingError is raised. Engine errors propagate (the engine already retries transient ones).
    ``usage``, when given, accumulates prompt/completion tokens like PanelQA.usage. ``size``: the art's (W, H) when the
    caller has it (it decoded the image anyway); otherwise it is read from the inline image data."""
    size = tuple(size) if size else _image_size(image)
    present = list(dict.fromkeys(str(n).strip().lower() for n in present if str(n).strip()))
    names = list(dict.fromkeys([*cast.names(), *present]))
    ref_names = [n for n in names if n in present and n in (sheets or {})]   # cast order, sheets before the panel
    message = Message.user(locate_prompt(cast, present, ref_names), images=[sheets[n] for n in ref_names] + [image])
    # a cut-off reply is asked again with the same prompt at temperature 0, so it would be cut off again: leave room
    config = GenerationConfig(temperature=0.0, max_tokens=4000, thinking=False, response_format={"type": "json_object"})
    last = "no reply"
    for _ in range(MAX_TRIES):
        response = await vision_engine.generate_text([message], config)
        if usage is not None:
            for k in ("prompt_tokens", "completion_tokens"):
                usage[k] = int(usage.get(k, 0) or 0) + int((response.usage or {}).get(k, 0) or 0)
        try:
            data = extract_json(response.content or "")
        except (ValueError, json.JSONDecodeError) as e:
            last = f"{type(e).__name__}: {str(e)[:120]}"
            continue
        if isinstance(data, dict) and any(k in data for k in REPLY_KEYS):
            return grounding_from_json(data, size, names)
        last = f"reply without grounding keys: {str(response.content)[:120]}"
    raise GroundingError(f"no usable grounding JSON after {MAX_TRIES} tries ({last})")


IDEA_PROMPT = """The image is one panel of an educational manga, drawn without words (all text is added later). Judge the drawing only.
It is meant to show, so that a reader who reads no words understands it: {shows}

Return JSON only: {{"shown": "yes", "missing": "", "objects": [[x1, y1, x2, y2]]}}
- shown: "yes" when the objects and their arrangement make this clear at a glance; "partly" when the objects are there but small, hidden, cut off, ambiguous, or arranged so the idea does not read; "no" when they are missing or the picture shows something else.
- missing: what a viewer cannot see or make out (objects, their size, position or arrangement; never text, labels, numbers or scores: all words are added later), in at most 15 words; "" when shown is "yes".
- objects: boxes around the objects that carry the idea (not the characters), the most important first, as integers normalized to 0-1000 over the whole image (x from the left edge, y from the top edge); [] when there are none."""
IDEA_LABELS = ("idea not shown:", "idea unclear:")
# a diagnosis that asks for writing would be repeated to the image model as something to add ("no visible scores or labels")
IDEA_TEXTY_RE = re.compile(r"\b(?:labels?|labell?ed|texts?|words?|letters?|lettering|writing|written|numbers?|numerals?|digits?|"
                           r"scores?|values?|captions?|legends?|annotations?|titles?|names?|signs?|ratings?|percent(?:ages?|s)?)\b", re.I)


async def check_idea(vision_engine: BaseEngine, image: ImageContent, shows: str, size: tuple[int, int],
                     usage: Optional[dict] = None) -> tuple[Optional[str], list[Box]]:
    """Does the wordless art show ``shows`` (a teaching scene's idea)? (problem or None, boxes of the objects that carry
    it in pixels). One vision call, asked once more when its reply is not usable JSON; GroundingError after that."""
    W, H = size
    message = Message.user(IDEA_PROMPT.format(shows=" ".join(str(shows).split())), images=[image])
    config = GenerationConfig(temperature=0.0, max_tokens=1500, thinking=False, response_format={"type": "json_object"})
    for _ in range(2):
        response = await vision_engine.generate_text([message], config)
        if usage is not None:
            for k in ("prompt_tokens", "completion_tokens"):
                usage[k] = int(usage.get(k, 0) or 0) + int((response.usage or {}).get(k, 0) or 0)
        try:
            data = extract_json(response.content or "")
        except (ValueError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict) or "shown" not in data:
            continue
        shown = data.get("shown")
        verdict = {True: "yes", False: "no"}.get(shown, str(shown or "")) if isinstance(shown, bool) else str(shown or "")
        verdict = verdict.strip().lower()
        verdict = "partly" if verdict.startswith(("partial", "mostly", "somewhat")) else "yes" if verdict == "true" else verdict
        missing = _plain(" ".join(str(data.get("missing") or "").split()))[:160]
        if IDEA_TEXTY_RE.search(missing):   # the words are lettered later: a redraw must never be asked for them
            missing = ""
        boxes = [(round(b[0] * W / SCALE), round(b[1] * H / SCALE), round(b[2] * W / SCALE), round(b[3] * H / SCALE))
                 for b in map(_box, _shapes(data.get("objects"))) if b]
        want = " ".join(str(shows).split()).rstrip(".")
        if verdict.startswith("yes"):
            return None, boxes
        if verdict.startswith("partly"):
            return (f"idea unclear: the picture must show {want}" + (f"; {missing}" if missing else "")
                    + "; draw those objects larger, in front and uncovered"), boxes
        return f"idea not shown: the picture must show {want}" + (f"; missing: {missing}" if missing else ""), boxes
    raise GroundingError("no usable idea verdict after 2 tries")


# ----------------------------------------------------------------------------- reply → Grounding

def _num(value) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, str):
        try:
            v = float(value.strip())
        except ValueError:
            return None
        return v if math.isfinite(v) else None
    return None


def _numbers(value) -> Optional[list[float]]:
    """Every coordinate of a shape, flattened (nested lists, [x, y] pairs, {"x", "y"} points, {"x1".."y2"} boxes);
    None when anything else is in it."""
    out: list[float] = []

    def walk(v) -> bool:
        if isinstance(v, dict):
            for keys in (("x", "y"), ("x1", "y1", "x2", "y2")):
                if all(k in v for k in keys):
                    return walk([v[k] for k in keys])
            return False
        if isinstance(v, (list, tuple)):
            return all(walk(x) for x in v)
        n = _num(v)
        if n is not None:
            out.append(n)
        return n is not None

    return out if walk(value) and out else None


def _framed(values: list[float]) -> Optional[list[float]]:
    """Coordinates clamped into 0-1000, or None when one is clearly outside (pixels, or garbage)."""
    if any(v < -SLACK or v > SCALE + SLACK for v in values):
        return None
    return [min(max(v, 0.0), float(SCALE)) for v in values]


def _box(value) -> Optional[tuple[float, float, float, float]]:
    if isinstance(value, dict) and not ({"x1", "x"} & value.keys()):
        value = next((value[k] for k in ("box", "bbox", "head") if k in value), None)
    values = _numbers(value)
    if values and len(values) >= 6 and len(values) % 2 == 0:   # corner points instead of a box: their bounding box
        xs, ys = values[0::2], values[1::2]
        values = [min(xs), min(ys), max(xs), max(ys)]
    values = _framed(values) if values and len(values) == 4 else None
    if values is None:
        return None
    (x1, x2), (y1, y2) = sorted(values[0::2]), sorted(values[1::2])
    return (x1, y1, x2, y2) if x2 - x1 >= MIN_SIDE and y2 - y1 >= MIN_SIDE else None


def _area(quad) -> float:
    return abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(quad, quad[1:] + quad[:1]))) / 2


def _is_point(value) -> bool:
    if isinstance(value, dict):
        return {"x", "y"} <= value.keys() and all(_num(value[k]) is not None for k in ("x", "y"))
    return isinstance(value, (list, tuple)) and len(value) == 2 and all(_num(x) is not None for x in value)


def _quad(value) -> Optional[list[tuple[float, float]]]:
    """A board's corners from {"corners": [...]}, a bare list of 4 points, or 4 numbers (an axis-aligned box), also
    {"x1".."y2"} and named corners ({"top_left": [x, y], ...}); clockwise from top-left. A corner the model repeated is a
    slip, not a fifth corner."""
    if isinstance(value, dict):
        inner = next((value[k] for k in ("corners", "points", "box", "bbox") if k in value), None)
        value = value if inner is None else inner
    if isinstance(value, dict) and not {"x1", "y1", "x2", "y2"} <= value.keys():
        corners = list(value.values())   # named corners; {"x", "y", "width", "height"} is not corners: never misread as a box
        value = corners if corners and all(map(_is_point, corners)) else None
    values = _numbers(value)
    if values is not None and len(values) == 4:
        box = _box(values)
        if box is None:
            return None
        x1, y1, x2, y2 = box
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    if values is None or len(values) < 8 or len(values) % 2 or (values := _framed(values)) is None:
        return None
    points = list(dict.fromkeys(zip(values[0::2], values[1::2])))
    if len(points) != 4:
        return None
    cx, cy = sum(x for x, _ in points) / 4, sum(y for _, y in points) / 4
    points.sort(key=lambda p: math.atan2(p[1] - cy, p[0] - cx))   # y points down: increasing angle runs clockwise
    k = min(range(4), key=lambda i: points[i][0] + points[i][1])
    points = points[k:] + points[:k]
    return points if _area(points) >= MIN_SIDE * MIN_SIDE else None


def _is_shape(value) -> bool:
    """A bare shape where a list of shapes belongs: [x1, y1, x2, y2], [[x, y] * 4] or one {...} object."""
    if isinstance(value, dict):
        return True
    if not isinstance(value, list) or not value:
        return False
    return all(_num(v) is not None for v in value) or all(_is_point(v) and not isinstance(v, dict) for v in value)


def _shapes(value) -> list:
    return [value] if _is_shape(value) else _as_list(value)


_SEEN = r"(?: (?:else|other|found|visible|detected|present|seen|drawn|here|at all))*"
NOTHING_RE = re.compile(rf"(?:none|nothing|nobody|no ?one|no|n/?a|null|nil|empty|not applicable|-+|\[\s*\]|\{{\s*\}}|无|没有){_SEEN}"
                        r"|(?:there (?:is|are) )?no (?:\w+ ){0,3}?(?:texts?|letters?|digits?|symbols?|numbers?|words?|writing|creatures?"
                        rf"|figures?|characters?|people|persons?|animals?|extras?){_SEEN}(?: (?:other than|besides|except|apart from) .+)?")


# plain marks are not writing: a ruler's ticks or a beaker's scale lines are what the picture asked for, and a redraw
# for them is paid for nothing (a word, a letter or a digit always counts)
MARKS_RE = re.compile(r"(?:measurement|measuring|scale|tick|graduation|grid|ruler)\s*(?:marks?|markings?|lines?|ticks?)?|"
                      r"(?:tick|scale|graduation)\s+marks?|markings?|lines|stripes|dots|dashes", re.I)


def _just_marks(text: str, shapes: bool = True) -> bool:
    """An entry that names only marks (ticks, scale lines, dots) and quotes no letters or digits; with ``shapes``, also
    one that names only arrows or simple shapes (never on a board, which must stay blank for the code-drawn exhibit)."""
    text = str(text or "").strip()
    if not text or re.search(r"\d", text) or re.search(r"[\"“「『]", text):
        return False
    words = re.sub(r"\((?:on|in|at)\s+[^()]*\)\s*$", "", text).strip()
    return bool(words) and (bool(MARKS_RE.fullmatch(re.sub(r"\s+", " ", words.lower()).strip().rstrip(".!")))
                            or (shapes and _just_arrows(words)))


ARROW_GLYPHS = set("→←↑↓↔↕⇒⇐⟶⟵➜➔➝➞➡⬅⬆⬇↗↘↙↖✓✔✗✘✕✖❌✅☑")
ARROW_WORDS = frozenset("a an the bold big large small tiny thick thin long short curved curving straight wavy dashed dotted drawn "
                        "simple plain single double line lines motion pointing red blue black green yellow orange white gray grey "
                        "brown pencil ink chalk shape solid left right up down upward upwards downward downwards forward backward "
                        "one two three four five six seven eight nine ten several many multiple few some x check tick symbol "
                        "symbols mark marks and + &".split())
# the simple shapes a teaching picture may draw as objects (an arrow from cause to effect, a cross on a slip, a tick sticker)
SHAPE_RE = re.compile(r"\b(?:arrows?|check ?marks?|checkmarks?|ticks?|cross(?:es)?|x[- ]?marks?|circles?|stars?|hearts?|dots?)\b", re.I)
# "arrow pointing to the stack": where a shape points says nothing drawn, unless it names a word ("arrow pointing to EXIT")
SHAPE_PLACE_RE = re.compile(r"\s+(?:pointing|points|leading|leads|going|goes|running|runs|from|to|toward|towards|between|at|into|"
                            r"onto|over|under|across)\b[^A-Z0-9\"“「『]*$")
BOARD_PLACE_RE = re.compile(r"\b(?:white|black|chalk)?boards?\b", re.I)   # not "cardboard" or "clipboard"


def _just_arrows(words: str) -> bool:
    """An entry that names only arrows or simple shapes ("bold arrow", "→ →", "red X cross", "five green check marks",
    "✔️"): a picture's object, not writing."""
    toks = re.findall(r"[^\s,]+", re.sub("[\ufe0e\ufe0f]", "", SHAPE_PLACE_RE.sub("", words)).lower())   # ✔️ ➡️: emoji selectors
    if toks and all(set(t) <= ARROW_GLYPHS for t in toks):
        return True
    text = " ".join(toks)
    rest = SHAPE_RE.sub(" ", text).split()
    return bool(SHAPE_RE.search(text)) and all(t in ARROW_WORDS or set(t) <= ARROW_GLYPHS for t in rest)


def _is_nothing(text) -> bool:
    """True for an entry that only says there is nothing to report ("None", "no text visible", "n/a", "no", "[]")."""
    return bool(NOTHING_RE.fullmatch(re.sub(r"\s+", " ", str(text)).strip().lower().rstrip(".!")))


# what the model wrote about a place or a figure goes into the next art prompt: never its quotes, digits or CJK glyphs
GLYPHS_RE = re.compile(r"\"[^\"]*\"|“[^”]*”|「[^」]*」|『[^』]*』|\d+(?:st|nd|rd|th)?"
                       r"|[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff\uff00-\uffef]+")


def _tidy(text: str) -> str:
    return re.sub(r"\s+([,.;:!?)])", r"\1", re.sub(r"\s+", " ", text)).strip(" ,;:-")


def _plain(text: str) -> str:
    """A model-written place or figure, safe to repeat in the art prompt."""
    return _tidy(re.sub(r"\(\s*\)", " ", GLYPHS_RE.sub(" ", str(text))))


def _without(where: str, text: str) -> str:
    """The place without the drawn words in it ("the sign reading BLEU" -> "the sign reading"); words of one character
    stay, since removing them would eat articles ("A" on "a sign")."""
    for word in dict.fromkeys(text.split()):
        if len(word) >= 2:
            where = re.sub(rf"(?<!\w){re.escape(word)}(?!\w)", " ", where, flags=re.I)
    return _tidy(where)


def _entries(value, keys: tuple[str, ...]) -> list[str]:
    out = []
    for item in _as_list(value):
        if isinstance(item, dict):
            item = next((item[k] for k in keys if isinstance(item.get(k), str) and item[k].strip()), "")
        if isinstance(item, (str, int, float)) and not isinstance(item, bool) and str(item).strip() and not _is_nothing(item):
            out.append(str(item).strip())
    return out


def _texts(value) -> list[str]:
    """Drawn text as "text (on where)": the place is what the next attempt's feedback names, never the glyphs themselves."""
    if isinstance(value, dict) and not ({"text", "content", "where", "location"} & value.keys()):   # {"the clock face": ["12", "3"]}
        value = [{"text": ", ".join(str(v) for v in _as_list(glyphs)), "where": place} for place, glyphs in value.items()]
    out = []
    for item in _as_list(value):
        if isinstance(item, dict):
            text = str(item.get("text") or item.get("content") or "").strip()
            where = re.sub(r"^(?:on|in|at)\s+", "", str(item.get("where") or item.get("location") or "").strip(), flags=re.I)
            where = "" if _is_nothing(where) else _without(where, text)
            if (not text or _is_nothing(text)) and not where:
                continue
            if text and _just_marks(text, shapes=not BOARD_PLACE_RE.search(where)):
                continue
            out.append(f"{text or 'writing'} (on {where})" if where else text)
        elif isinstance(item, (str, int, float)) and not isinstance(item, bool) and str(item).strip() and not _is_nothing(item) \
                and not _just_marks(str(item), shapes=not BOARD_PLACE_RE.search(str(item))):
            out.append(str(item).strip())
    return list(dict.fromkeys(out))


_TRUE, _FALSE = {"true", "yes", "1"}, {"false", "no", "0"}


def _verdict(value) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower() if value is not None else ""
    return True if s in _TRUE else False if s in _FALSE else None


CLONE = "a second {name}"
CLONE_RE = re.compile(r"a second (\S+)")
CLONE_WORDS = r"second|another|extra|duplicated?|duplicates|copy|copies|clones?|cloned|twins?|again"


def _mention(text: str, names) -> tuple[Optional[str], bool]:
    """The cast member an extras entry is about ("a second kumo", "kumo (duplicate)"), and whether it says it is a copy;
    (None, False) for anything else ("a small bird next to kumo")."""
    t = text.strip().lower()
    for name in names:
        if re.match(rf"\W*(?:(?:a|an|the)\s+)?(?:(?:{CLONE_WORDS})\s+(?:of\s+)?(?:(?:a|an|the)\s+)?)?{re.escape(name)}\b", t):
            return name, bool(re.search(rf"\b(?:{CLONE_WORDS})\b", t))
    return None, False


def grounding_from_json(data: dict, size: tuple[int, int], names) -> Grounding:
    """The model's 0-1000 reply as a pixel Grounding (pure). ``names``: the cast names (plus the panel's characters)."""
    W, H = size
    valid = list(dict.fromkeys(str(n).strip().lower() for n in names if str(n).strip()))
    px = lambda b: (round(b[0] * W / SCALE), round(b[1] * H / SCALE), round(b[2] * W / SCALE), round(b[3] * H / SCALE))
    g = Grounding(size=(W, H))
    if not isinstance(data, dict):
        return g
    chars = data.get("characters")
    if isinstance(chars, dict):   # {"papi": [x1, y1, x2, y2]} or {"papi": {"head": [...]}}
        chars = [{"name": k, **v} if isinstance(v, dict) else {"name": k, "head": v} for k, v in chars.items()]
    for entry in _as_list(chars):
        if not isinstance(entry, dict) or _verdict(entry.get("present")) is False or _verdict(entry.get("visible")) is False:
            continue
        head = _box(next((entry[k] for k in ("head", "head_box", "box", "bbox") if entry.get(k) is not None), None))
        label = str(entry.get("name") or "").strip()
        name = _match_cast(label, valid)
        if head is None:   # not located: a missing head is reported by art_problems
            continue
        if name is None:
            g.extras.append(label or "an unnamed figure")
        elif name in g.heads:
            g.extras.append(CLONE.format(name=name))
        else:
            g.heads[name] = px(head)
            verdict = _verdict(entry.get("matches_reference", entry.get("matches")))
            if verdict is not None:
                g.matches[name] = verdict
    quads = [q for q in map(_quad, _shapes(data.get("boards", data.get("board")))) if q]
    if quads:
        g.board = [(round(x * W / SCALE), round(y * H / SCALE)) for x, y in max(quads, key=_area)]
    g.bubbles = [px(b) for b in map(_box, _shapes(data.get("speech_bubbles", data.get("bubbles")))) if b]
    g.text_found = _texts(data.get("text_or_symbols", data.get("text")))
    for extra in _entries(data.get("extra_creatures", data.get("extras", data.get("extra_characters"))), ("description", "name", "what", "text")):
        name, copy = _mention(extra, valid)
        if name is None or name not in g.heads:   # an unknown figure, or a cast member drawn without a box
            g.extras.append(extra)
        elif copy:
            g.extras.append(CLONE.format(name=name))
        # else: the model repeating a located character
    g.extras = list(dict.fromkeys(g.extras))
    return g


# ----------------------------------------------------------------------------- pixel checks

CELL = 8
EDGE_LEVEL = 38       # FIND_EDGES response that counts as an edge (tuned on the watercolor art of the prototype)
CALM_EDGES = 0.03     # a cell is calm when under 3% of its pixels are edges
BOARD_MIN_W, BOARD_MIN_H = 0.55, 0.25   # the clean part of a board, as a share of the panel...
BOARD_MIN_W_TALL = 0.45                 # ...or narrower when it is tall enough to hold the same area (55% x 25%)
BAND_MIN_H = 0.14     # the calm band of a scene panel, as a share of the panel height
BAND_MIN_W = 0.5      # a band narrower than half the panel cannot hold the narration box
BAND_TOP = 0.5        # the band is looked for in the upper half


def _cells(mask: np.ndarray, cell: int = CELL) -> np.ndarray:
    gh, gw = mask.shape[0] // cell, mask.shape[1] // cell
    return mask[: gh * cell, : gw * cell].reshape(gh, cell, gw, cell).mean(axis=(1, 3))


def _largest_rect(grid: np.ndarray, min_w: int = 1) -> Optional[tuple[int, int, int, int]]:
    """Largest-area all-True rectangle of a boolean grid that is at least min_w cells wide: (c1, r1, c2, r2), ends
    exclusive. Histogram-stack scan; every maximal rectangle is visited, so the width limit cannot hide the best one."""
    rows, cols = grid.shape
    heights = np.zeros(cols, dtype=int)
    best_area, best = 0, None
    for r in range(rows):
        heights = np.where(grid[r], heights + 1, 0)
        stack: list[tuple[int, int]] = []   # (start column, height)
        for c in range(cols + 1):
            h = int(heights[c]) if c < cols else 0
            start = c
            while stack and stack[-1][1] >= h:
                start, hh = stack.pop()
                if c - start >= max(1, min_w) and hh * (c - start) > best_area:
                    best_area, best = hh * (c - start), (start, r - hh + 1, c, r + 1)
            stack.append((start, h))
    return best


def calm_band(art: Image.Image, avoid=(), top: float = BAND_TOP, min_w: float = BAND_MIN_W) -> Optional[Box]:
    """The largest calm rectangle (few edges) in the top part of the art, at least ``min_w`` of the width wide and
    clear of the ``avoid`` boxes (heads); None when there is none."""
    W, H = art.size
    edges = np.asarray(art.convert("L").filter(ImageFilter.BoxBlur(1)).filter(ImageFilter.FIND_EDGES))[: int(H * top)] > EDGE_LEVEL
    grid = _cells(edges) < CALM_EDGES
    for x1, y1, x2, y2 in avoid:
        grid[max(0, int(y1) // CELL): max(0, int(y2) // CELL + 1), max(0, int(x1) // CELL): max(0, int(x2) // CELL + 1)] = False
    rect = _largest_rect(grid, math.ceil(min_w * W / CELL)) if grid.size else None
    return None if rect is None else (rect[0] * CELL, rect[1] * CELL, rect[2] * CELL, rect[3] * CELL)


def clean_rect(art: Image.Image, quad, exclude=(), inset: float = 0.03) -> Optional[Box]:
    """Largest axis-aligned rectangle inside the board quad whose pixels are all blank board surface, away from the
    ``exclude`` boxes (heads in front of the board); None when there is none. This is services.lettering.clean_rect,
    the rectangle the chart is drawn into, so the check and the typesetter can never disagree about the board.
    Imported on use: lettering may import this module (for Grounding), and a module-level cycle would break both."""
    from services.lettering import clean_rect as lettering_clean_rect
    rect = lettering_clean_rect(art, quad, inset=inset, exclude=exclude)
    return None if rect is None else tuple(int(round(v)) for v in rect)


# ----------------------------------------------------------------------------- problems and ranking

# every problem starts with one of these labels: rank_key sorts by them, and they read naturally as retry feedback
UNCHECKED_LABELS = ("unchecked:",)   # the caller's note for an attempt whose art could not be located (jobs._render_panel_v3)
CHARACTER_LABELS = ("missing character:", "wrong look:")
EXTRA_LABELS = ("not in this panel:", "clone:", "extra figure:")
TEXT_LABELS = ("drawn text:", "drawn bubbles:")
BOARD_LABELS = ("no board:", "small board:")
BAND_LABELS = ("busy top:",)
BOARD_SIZE_RE = re.compile(r"only (\d+)% x (\d+)%")
BAND_SIZE_RE = re.compile(r"only (\d+)% of the panel height")
WHERE_RE = re.compile(r"\(on ([^()]+)\)\s*$")   # the place _texts appended ("O(n)" alone names no place)


def is_board_panel(panel) -> bool:
    """A board panel has an exhibit to draw: the same rule as lettering (a "board" panel without a spec is lettered as a
    scene, so its art must be judged as one)."""
    return getattr(panel, "kind", "scene") == "board" and isinstance(getattr(panel, "board", None), dict) and bool(panel.board)


def _and(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + f" and {names[-1]}" if names else "the listed characters"


def art_problems(g: Grounding, panel: PanelV2, art: Image.Image) -> list[str]:
    """What is wrong with one attempt's art, worst first; [] when it is ready for lettering (pure, deterministic)."""
    W, H = art.size
    gw, gh = g.size if g.size else (0, 0)
    sx, sy = (W / gw, H / gh) if gw and gh else (1.0, 1.0)   # made on this art; scaled in case it was resized (no size: as is)
    heads = [(x1 * sx, y1 * sy, x2 * sx, y2 * sy) for x1, y1, x2, y2 in g.heads.values()]
    present = list(dict.fromkeys(panel.characters))
    who = _and(present)
    problems = []
    for name in present:
        if name not in g.heads:
            problems.append(f"missing character: {name} must be in the picture, with the whole head visible")
        elif g.matches.get(name) is False:
            problems.append(f"wrong look: {name} must look exactly like its reference (same species, colors and face)")
    for name in g.heads:
        if name not in present:
            problems.append(f"not in this panel: remove {name}; only {who} appear")
    extras = []
    for extra in g.extras:
        clone = CLONE_RE.fullmatch(extra)
        mentioned, _ = _mention(extra, present)
        if clone and clone.group(1) in g.heads:
            if clone.group(1) in present:   # one who should not be there at all is already told to go
                problems.append(f"clone: draw {clone.group(1)} only once")
        elif not (mentioned and mentioned not in g.heads):   # a missing character is already reported above
            extras.append(_plain(extra) or "an extra figure")
    if extras:
        problems.append(f"extra figure: remove {'; '.join(dict.fromkeys(extras))[:160]}; nobody but {who} appears")
    if g.text_found:
        where = list(dict.fromkeys(p for t in g.text_found if (m := WHERE_RE.search(t)) and (p := _plain(m.group(1)))))
        spot = f" (on {', '.join(where[:4])})" if where else ""
        problems.append(f"drawn text: remove every letter, digit and symbol{spot}; every surface stays blank")
    if g.bubbles:
        problems.append("drawn bubbles: remove the speech bubbles, chat windows and text boxes; the picture has no bubbles or captions")
    if is_board_panel(panel):
        if not g.board:
            problems.append("no board: draw a LARGE, COMPLETELY BLANK whiteboard across almost the full width, with nothing in front of it")
        else:
            rect = clean_rect(art, [(x * sx, y * sy) for x, y in g.board], exclude=heads)
            w, h = ((rect[2] - rect[0]) / W, (rect[3] - rect[1]) / H) if rect else (0.0, 0.0)
            roomy = w >= BOARD_MIN_W_TALL and w * h >= BOARD_MIN_W * BOARD_MIN_H   # a chart fits a tall board as well
            if h < BOARD_MIN_H or (w < BOARD_MIN_W and not roomy):
                problems.append(f"small board: the blank part of the whiteboard is only {round(w * 100)}% x {round(h * 100)}% of the panel "
                                f"(needs {round(BOARD_MIN_W * 100)}% x {round(BOARD_MIN_H * 100)}%); make it bigger, blank and unobstructed")
    else:
        band = calm_band(art, avoid=heads)
        h = (band[3] - band[1]) / H if band else 0.0
        if h < BAND_MIN_H:
            problems.append(f"busy top: the calm area at the top is only {round(h * 100)}% of the panel height (needs {round(BAND_MIN_H * 100)}%); "
                            "keep the top 28% plain, empty background with every head below it")
    return problems


def _shortfall(problems) -> int:
    """How far the board / band misses its size, in percentage points (breaks ties between two flawed attempts)."""
    total = 0
    for p in map(str, problems):
        if p.startswith("no board:"):
            total += round((BOARD_MIN_W + BOARD_MIN_H) * 100)
        elif (m := BOARD_SIZE_RE.search(p)) and p.startswith("small board:"):
            total += max(0, round(BOARD_MIN_W * 100) - int(m.group(1))) + max(0, round(BOARD_MIN_H * 100) - int(m.group(2)))
        elif (m := BAND_SIZE_RE.search(p)) and p.startswith("busy top:"):
            total += max(0, round(BAND_MIN_H * 100) - int(m.group(1)))
    return total


def rank_key(problems) -> tuple:
    """Sort key for attempts, HIGHER is better (like jobs._attempt_rank; an attempt without problems ranks highest, at
    all zeros): checked art first ("unchecked: ..." ranks below every checked attempt: a real verdict beats none, and
    lettering gets no head boxes for it), then fewer character problems, extra figures, drawn text, a teaching scene's
    idea not shown, then only unclear, board, band (lexicographic), then less missing board / band size, then fewer
    problems. Publish max(attempts, key=rank_key). Works on the strings alone, so problems read back from job state rank
    the same."""
    problems = [str(p) for p in problems or []]
    count = lambda labels: sum(p.startswith(labels) for p in problems)
    return (-count(UNCHECKED_LABELS), -count(CHARACTER_LABELS), -count(EXTRA_LABELS), -count(TEXT_LABELS), -count(IDEA_LABELS[:1]),
            -count(IDEA_LABELS[1:]), -count(BOARD_LABELS), -count(BAND_LABELS), -_shortfall(problems), -len(problems))


def compose_frames(arts: list[Image.Image], groundings: list[Optional[Grounding]], size: tuple[int, int]) -> tuple[Image.Image, Grounding]:
    """Frames stacked top to bottom into one panel of ``size``, and one Grounding in its pixels (a frame whose art could
    not be located adds none). A character drawn in two frames keeps the head of the first one it is in under its own
    name (its bubbles' tails point there); its head in a later frame goes in as "#2 papi", a head no speaker owns, so
    the lettering keeps it clear like every head instead of dropping it as an object. The objects come most important
    first, as the lettering drops the last ones first: each frame's objects in turn (the idea check lists a frame's
    objects most important first), so the bottom frame's key object is not the first to go under a bubble; a box over
    half its own frame is the frame itself, not an object (as lettering.GIANT_OBJECT for one picture)."""
    from services.lettering import GIANT_OBJECT   # on use, as clean_rect: lettering may import this module
    W, H = int(size[0]), int(size[1])
    h = H // max(1, len(arts))
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    merged = Grounding(size=(W, H))
    per_frame: list[list] = []
    for i, (art, g) in enumerate(zip(arts, groundings)):
        canvas.paste(art.convert("RGB").resize((W, h), Image.LANCZOS), (0, i * h))
        if g is None:
            continue
        gw, gh = g.size if g.size and g.size[0] and g.size[1] else art.size
        sx, sy = W / gw, h / gh
        box = lambda b: (round(b[0] * sx), round(b[1] * sy + i * h), round(b[2] * sx), round(b[3] * sy + i * h))
        for name, b in g.heads.items():
            merged.heads[f"#{i + 1} {name}" if name in merged.heads else name] = box(b)
        per_frame.append([box(b) for b in g.objects if (b[2] - b[0]) * (b[3] - b[1]) <= GIANT_OBJECT * gw * gh])
        merged.text_found += list(g.text_found)
        merged.extras += list(g.extras)
        for name, ok in g.matches.items():
            merged.matches[name] = merged.matches.get(name, True) and ok
    merged.objects = [f[k] for k in range(max(map(len, per_frame), default=0)) for f in per_frame if k < len(f)]
    return canvas, merged
