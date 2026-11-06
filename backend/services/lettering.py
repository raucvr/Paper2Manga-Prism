"""
Lettering (Prism v3): typesets a panel's words onto text-free art.

The image model draws no text at all; this module adds every word in code, deterministically:
- a narration box (cream fill, ink border) at the top of the calm band of the picture;
- speech bubbles: superellipse bodies with an ink outline and a short curved tail aimed at the speaker's head
  (grounding head boxes). One font size per panel, reading order (the next bubble is to the right on the same
  row, or below), never over a head, another bubble, the narration or the chart; a tail is cut short before
  it would enter a head box, the chart or another bubble, leaving a stub that still points at the speaker;
- on board panels, the exhibit drawn by services.boards into the blank part of the board (multiply blend); its top
  gives up to three 12 % steps when the narration box or the bubbles need room (and gets back what they leave unused);
  when the exhibit cannot be drawn at all, the panel is lettered as a scene and layout["board_error"] says why;
- a caption strip below the art (canvas extended downward) when the words cannot fit at the minimum size:
  the narration, then "Name: text" lines for the bubbles that did not fit; layout["fallback"] is True.

All lengths are given for a 1024 px wide panel and scale with the art width. Pure PIL + numpy, no network.
"""

from __future__ import annotations

import math
import random
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, NamedTuple, Optional, Sequence

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

from services.fonts import (FT_LOCK as _FT_LOCK, SYMBOLS_FIRST, FontSpec, bold_stroke, configured as _configured_fonts,
                            digit_scale, has_glyph as _has_glyph, is_emoji, is_han, load as _font, missing as _missing,
                            resolve as _resolve_fonts)

Rect = tuple[float, float, float, float]

LINE = 1.34                         # line height / font size
INK = (22, 22, 22)                  # outlines of bubbles, boxes and cards
TEXT_INK = (25, 25, 25)
PAPER = (255, 255, 255)
CREAM = (255, 251, 238)             # narration boxes and the caption strip
CHIP_TINTS = {"papi": (243, 222, 196), "kumo": (214, 234, 250), "nezu": (232, 220, 202)}   # name tabs, after each sheet's colors
P_EXP = 2.3                         # superellipse exponent of a bubble body (2 = ellipse, larger = boxier)
SUPERSAMPLE = 4                     # bubble shapes are drawn this many times larger, then box-filtered down
# px at 1024 px width; a phone shows the panel about 390 CSS px wide (x0.38), so the smallest bubble text is ~13 CSS px
BUBBLE_SIZES = {"zh": (48, 46, 44, 42, 40, 38, 36), "en": (46, 44, 42, 40, 38, 36, 34)}
NARRATION_SIZES = {"zh": (42, 40, 38, 36, 34), "en": (40, 38, 36, 34, 32)}
CHART_SHRINK = (0.0, 0.12, 0.24, 0.36)   # board panels: the chart gives up its top in these steps for the bubbles
LAYOUT_VERSION = 1


class LayoutError(Exception):
    """The panel cannot be lettered at all (the art is missing or unusable)."""


class _BoardFailed(LayoutError):
    """The exhibit cannot be drawn (the renderer rejected the spec or is missing); the panel is lettered without it."""


@dataclass
class Grounding:
    """Where things are in a panel's art (filled by services.art_check), in pixels of `size` (W, H)."""
    size: tuple[int, int] = (0, 0)
    heads: dict[str, tuple[float, float, float, float]] = field(default_factory=dict)   # name -> (x1, y1, x2, y2)
    board: Optional[list[tuple[float, float]]] = None    # blank board corners, clockwise from top-left
    bubbles: list = field(default_factory=list)
    text_found: list = field(default_factory=list)
    extras: list = field(default_factory=list)
    matches: dict[str, bool] = field(default_factory=dict)
    objects: list = field(default_factory=list)          # boxes of the objects that carry a scene's idea


def _lang(language: str) -> str:
    """'zh' for Chinese (and legacy ja-JP jobs, which wrap the same way), else 'en'."""
    code = str(language or "").strip().lower()
    return "zh" if code.startswith(("zh", "ja", "cn")) else "en"


# ----------------------------------------------------------------------------- fonts
# The files come from services.fonts: bundled with Prism, so every machine letters the same (config may override them).

# superscripts are drawn as smaller, raised ordinary digits: hand fonts rarely have all of ⁰-⁹, and mixing
# glyphs from two fonts looks broken ('10¹⁹' with a Chalkboard ¹ and an Arial ⁹)
_SCRIPT_DIGITS = {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9", "⁻": "-"}
SCRIPT_SIZE, SCRIPT_RISE = 0.62, 0.36


# a number, not the digits of a name: 'P100', 'WMT14' and 'ResNet-152' stay regular
_STRONG = re.compile(r"(?<![A-Za-z0-9.,⁰¹²³⁴⁵⁶⁷⁸⁹])(?<![A-Za-z][-‐])[\d⁰¹²³⁴⁵⁶⁷⁸⁹][\d.,⁰¹²³⁴⁵⁶⁷⁸⁹⁻]*%?")


def _strong_mask(text: str) -> list[bool]:
    """Characters of the numbers (2+ characters: '28.4', '41.8%', '12') that are set in bold for emphasis."""
    mask = [False] * len(text)
    for m in _STRONG.finditer(text):
        if len(m.group(0).rstrip(".,")) >= 2:
            for i in range(m.start(), m.start() + len(m.group(0).rstrip(".,"))):
                mask[i] = True
    return mask


class Face:
    """A sized font that draws the glyphs it lacks with fallback fonts, all on one baseline.

    Pillow has no font fallback of its own; Shantell Sans has no Greek or CJK, ChillRound M no '↔'.
    With `strong` fonts (emphasis), numbers of two or more characters are set in bold. Chinese text takes its digits
    (superscripts too) from the chain's `digits` font, sized to the main font's own figures; √ and × come from the
    `symbols` font; the fonts at the `han_only` positions draw only Han characters; in bold, the fonts at the `synthetic`
    positions (no bold file) get a synthetic bold.
    """

    def __init__(self, main: FontSpec, fallbacks: Sequence[FontSpec], size: int, lang: str = "en",
                 strong: Optional[tuple[FontSpec, Sequence[FontSpec]]] = None, digits: Optional[int] = 1,
                 han_only: frozenset = frozenset(), bold: bool = False, synthetic: frozenset = frozenset(),
                 symbols: Optional[int] = None):
        self.size, self.lang = int(size), lang
        self._chain = (main, *fallbacks)
        self._digits = digits if lang == "zh" and digits and digits < len(self._chain) else None
        self._symbols = symbols if symbols and symbols < len(self._chain) else None
        self._han_only, self._synthetic = han_only, synthetic
        self._fonts = [self._load(self._chain, k, self.size, bold) for k in range(len(self._chain))]
        self._strong_chain = (strong[0], *strong[1]) if strong else None
        self._strong_fonts = ([self._load(self._strong_chain, k, self.size, True) for k in range(len(self._strong_chain))]
                              if strong else None)
        small = max(4, int(round(self.size * SCRIPT_SIZE)))
        k = self._digits if self._digits and _has_glyph(self._chain[self._digits], "0") else 0
        self._script = (self._load(self._chain, k, small, bold),
                        self._load(self._strong_chain, k, small, True) if strong else None)
        self._rise = int(round(self.size * SCRIPT_RISE))
        with _FT_LOCK:
            self.ascent = self._fonts[0].getmetrics()[0]
        self._lengths: dict[str, float] = {}
        self._offsets: dict[int, float] = {}

    def _load(self, chain: Sequence[FontSpec], k: int, size: int, bold: bool):
        """Font k of a chain at a size: the digits font of Chinese text as tall as the main font's figures, and a
        synthetic bold for a font without a bold file."""
        scale = digit_scale(chain[0], chain[k]) if k == self._digits else 1.0
        return _font(chain[k], max(1, int(round(size * scale))), bold and k in self._synthetic)

    def _pick(self, ch: str, strong: bool):
        """The font that draws ch, or None when no font of the chain has it (an emoji is never drawn)."""
        chain, fonts = (self._strong_chain, self._strong_fonts) if strong and self._strong_chain else (self._chain, self._fonts)
        if self._digits and "0" <= ch <= "9" and _has_glyph(chain[self._digits], ch):
            return fonts[self._digits]   # a hand font's 1 can read as a lowercase l ("额外1层"): digits come from the digits font
        if self._symbols and ch in SYMBOLS_FIRST and _has_glyph(chain[self._symbols], ch):
            return fonts[self._symbols]
        if is_emoji(ch):
            return None
        han = is_han(ch)
        for k, (spec, font) in enumerate(zip(chain, fonts)):
            if (han or k not in self._han_only) and _has_glyph(spec, ch):
                return font
        return None

    def missing(self, ch: str) -> str:
        """What is drawn for a character no font of the chain has (services.fonts.missing): an ASCII stand-in for a
        symbol, nothing for an emoji, else the character's box."""
        return _missing(ch, lambda c: self._pick(c, False) is not None)

    def draws(self, ch: str) -> bool:
        """False for a character that is drawn as nothing: an emoji, or an invisible character no font has."""
        return ch in _SCRIPT_DIGITS or self._pick(ch, False) is not None or self.missing(ch) != ""

    def has(self, ch: str) -> bool:
        return ch in _SCRIPT_DIGITS or self._pick(ch, False) is not None

    def runs(self, text: str) -> list[tuple[Any, str, int]]:
        """(font, text, rise) runs: rise > 0 for superscripts, drawn as smaller digits above the baseline. A character
        no font has is drawn as missing() says: its ASCII stand-in, nothing (an emoji), or the main font's box."""
        marks = _strong_mask(text) if self._strong_chain else None
        out: list[list] = []
        for i, ch in enumerate(text):
            strong = bool(marks and marks[i])
            if ch in _SCRIPT_DIGITS:
                pieces = [(self._script[1] if strong and self._script[1] else self._script[0], _SCRIPT_DIGITS[ch], self._rise)]
            else:
                font = self._pick(ch, strong)
                if font is not None:
                    pieces = [(font, ch, 0)]
                else:
                    main = (self._strong_fonts if strong and self._strong_fonts else self._fonts)[0]
                    pieces = [(self._pick(c, strong) or main, c, 0) for c in self.missing(ch)]
            for font, piece, rise in pieces:
                if out and out[-1][0] is font and out[-1][2] == rise:
                    out[-1][1] += piece
                else:
                    out.append([font, piece, rise])
        return [(f, t, r) for f, t, r in out]

    def length(self, text: str) -> float:
        width = self._lengths.get(text)
        if width is None:
            with _FT_LOCK:
                width = float(sum(f.getlength(t) for f, t, _ in self.runs(text)))
            if len(self._lengths) < 50_000:
                self._lengths[text] = width
        return width

    def draw(self, d: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, fill) -> None:
        """Like ImageDraw.text at xy (top-left on the main font's ascender line); fallback runs share its baseline."""
        x, y = xy
        base = y + self.ascent
        with _FT_LOCK:
            for font, run, rise in self.runs(text):
                d.text((x, base - rise), run, font=font, fill=fill, anchor="ls", stroke_width=bold_stroke(font),
                       stroke_fill=fill)
                x += font.getlength(run)

    def line_offset(self, lh: int) -> float:
        """y offset from the top of a line box of height lh to the draw position that centres the text in it."""
        off = self._offsets.get(lh)
        if off is None:
            with _FT_LOCK:
                _, top, _, bottom = self._fonts[0].getbbox("中" if self.lang == "zh" else "H", anchor="ls")
            off = self._offsets[lh] = lh / 2 - (top + bottom) / 2 - self.ascent
        return off


class FontSet:
    """Fonts per language ({text, bold}) plus fallback fonts for the glyphs the main font lacks, as services.fonts
    resolves them: the bundled files (zh ChillRound M, en Shantell Sans; Atkinson Hyperlegible Next for digits,
    DejaVu Sans for symbols and Greek), so every machine letters the same. `lettering.fonts` in config/api_config.yaml
    overrides them, e.g.
        lettering:
          fonts:
            zh: {text: ~/Library/Fonts/LXGWWenKai-Regular.ttf, bold: ~/Library/Fonts/LXGWWenKai-Bold.ttf}
            en: {text: "/System/Library/Fonts/Supplemental/ChalkboardSE.ttc#1", bold: [path, 2]}
            fallback: "/System/Library/Fonts/Hiragino Sans GB.ttc#0"
            preset: macos          # Yuanti SC + Chalkboard SE, the look before the fonts were bundled (a Mac with them)
    Missing or unloadable entries fall back to the bundled fonts.
    """

    ROLES = ("zh", "en", "fallback")

    def __init__(self, config: Optional[dict] = None):
        """config: the `lettering` section ({"fonts": {...}}) or the fonts mapping itself; None reads the app config."""
        if config is None:
            fonts = _configured_fonts()
        elif isinstance(config, dict):
            fonts = config["fonts"] if isinstance(config.get("fonts"), dict) else config
        else:
            fonts = {}
        self._plan = _resolve_fonts(fonts)
        self.files: dict[str, dict[str, FontSpec]] = dict(self._plan.files)
        # every available fallback, in order: Atkinson Hyperlegible Next (digits), DejaVu Sans (symbols), the zh face, ...
        self.fallbacks: list[dict] = self._plan.fallbacks
        self.files["fallback"] = self.fallbacks[0] if self.fallbacks else {"text": ("", 0), "bold": ("", 0)}
        self._faces: dict[tuple, Face] = {}

    def face(self, language: str, size: int, bold: bool = False, emphasis: bool = False) -> Face:
        """The Face for a language ('zh-CN', 'en-US', 'zh', 'en'), pixel size and weight."""
        lang = _lang(language)
        key = (lang, max(1, int(round(size))), bool(bold), bool(emphasis) and not bold)
        face = self._faces.get(key)
        if face is None:
            weight = "bold" if bold else "text"
            main, chain = self.files[lang], self._plan.chain(lang)
            strong = (main["bold"], [f["bold"] for f in chain]) if key[3] else None
            role = {f["role"]: i + 1 for i, f in reversed(list(enumerate(chain)))}       # the first font of each role
            han_only = frozenset(i + 1 for i, f in enumerate(chain) if f["han_only"])
            synthetic = frozenset(i for i, f in enumerate([main, *chain]) if f["bold"] == f["text"])   # no bold file
            face = self._faces[key] = Face(main[weight], [f[weight] for f in chain], key[1], lang, strong, role.get("digits"),
                                           han_only, bold=key[2], synthetic=synthetic, symbols=role.get("symbols"))
        return face


_DEFAULT_FONTS: Optional[FontSet] = None


def default_fonts() -> FontSet:
    """A FontSet resolved once from the app config."""
    global _DEFAULT_FONTS
    if _DEFAULT_FONTS is None:
        _DEFAULT_FONTS = FontSet()
    return _DEFAULT_FONTS


# ----------------------------------------------------------------------------- line breaking

NO_START = frozenset("，。！？、；：”’）」』】》〉〕｝…‥—～·%‰℃,.!?;:)]}>»ー々ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮヵヶ")
NO_END = frozenset("“‘（「『【《〈〔｛([{<«")
_ZH_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9.,+\-×·^%/'’_()\[\]⁰¹²³⁴⁵⁶⁷⁸⁹⁻]*|\s+|.", re.S)
_HYPHEN_BREAK = re.compile(r"(?<=[A-Za-z][\-/–—→])(?=[A-Za-z])")
_NUMBER = re.compile(r"[+\-−~≈<>]?\$?\d[\d.,]*(?:[~～\-–]\d[\d.,]*)?(?:%|x|×|[⁰¹²³⁴⁵⁶⁷⁸⁹⁻]+|\^[\-−]?\d+)?")
_METRIC = re.compile(r"[A-Z][A-Z0-9\-]{0,5}|[A-Za-z]{1,4}\d{1,2}")
_JOINERS = frozenset({"·", "×", "±", "x", "=", "≈"})
_SIGNS = frozenset("+-−±~≈<>≤≥～$€£¥#")          # never at a line end when a number follows: '−0.5', '~4.5'
_UNIT_ACRONYM = re.compile(r"[A-Z][A-Z0-9]*s?|[A-Z][a-z]?[A-Z]+s?|[A-Z]\d+")
# words that follow a number without being its unit ("28.4 and 41.8", "8 of them"): a line may break before them
_FUNCTION_WORDS = frozenset(
    "a an the and or but nor so yet to of in on at for with by from into onto than then vs vs. as if that which while "
    "when where who whom whose it its is are was were be been being has have had do does did not no also only just even "
    "over under after before about up down out more less most least per cent".split())


_OPEN = "([{（【「『“‘"
_TRAIL = ",.;:!?)]}…，。！？、；：）」』】》”’"


def _is_han(ch: str) -> bool:
    o = ord(ch[0])
    return 0x3400 <= o <= 0x9FFF or 0xF900 <= o <= 0xFAFF or 0x20000 <= o <= 0x2FA1F


def _is_number(token: str) -> bool:
    return bool(_NUMBER.fullmatch(token.strip().lstrip(_OPEN).rstrip(")]}）】」』”’")))


def _is_unit(word: str) -> bool:
    """An acronym or symbol (BLEU, GPUs, FLOPs, F1, %) or a short lower-case word (days, points, layers)."""
    if not word:
        return False
    if word in ("%", "x", "×") or _UNIT_ACRONYM.fullmatch(word):
        return True
    return word.islower() and word.isalpha() and len(word) <= 7 and word not in _FUNCTION_WORDS


def _sticky(left: str, right: str) -> bool:
    """True when the space between two words must not break: a number and its unit or metric."""
    words = left.split()
    a, b = (words[-1] if words else ""), right
    if not a or not b:
        return False
    if a in _JOINERS or b in _JOINERS:                         # 2.3 · 10^19
        return True
    if a in _SIGNS and b[:1].isdigit():                        # ~ 4.5
        return True
    if _is_number(a) and _is_unit(b.rstrip(_TRAIL)):             # 28.4 BLEU, 3.5 days, 8 GPUs
        return True
    return bool(_NUMBER.match(b.lstrip(_OPEN))) and bool(_METRIC.fullmatch(a.lstrip(_OPEN)))   # F1 77.8, WMT 2014


def _glue(toks: list[str]) -> list[str]:
    out: list[str] = []
    for t in toks:
        if not t.isspace() and len(out) >= 2 and out[-1].isspace() and not out[-2].isspace() and _sticky(out[-2], t):
            space = out.pop()
            out[-1] += space + t
        else:
            out.append(t)
    return out


def tokens(text: str, language: str) -> list[str]:
    """Unbreakable units in order (spaces are their own tokens).

    en: words, also split after a hyphen, slash or arrow between letters ('feed-' 'forward').
    zh: characters with kinsoku (no line starts with ，。！？ etc., none ends with “（ etc.), Latin/number runs kept
    whole ('28.4', 'BLEU', '10^19', 'GNMT+RL') except after a hyphen or slash between letters as in English, a unit
    character stays with its number ('3.5天'), a sign or range mark with the number after it ('（−0.5', '3~5天').
    Both: a number stays with its unit or metric across a space ('28.4 BLEU', 'F1 77.8', '2.3 · 10^19').
    """
    if _lang(language) == "en":
        out = []
        for t in re.findall(r"\s+|\S+", text):
            out.extend([t] if t.isspace() else [p for p in _HYPHEN_BREAK.split(t) if p])
        return _glue(out)
    out = []
    for run in _ZH_TOKEN.findall(text):
        for t in [run] if run.isspace() else [p for p in _HYPHEN_BREAK.split(run) if p]:
            glued = out and not t.isspace() and not out[-1].isspace()
            if len(out) >= 2 and not t.isspace() and t[0] in NO_START and out[-1].isspace():
                space = out.pop()                  # 'x ，' or '1.4 · 10': no break before the mark either
                out[-1] += space + t
            elif glued and (t[0] in NO_START or out[-1][-1] in NO_END):
                out[-1] += t
            elif glued and ((out[-1][-1] in _SIGNS and t[0].isdigit()) or (t in _SIGNS and out[-1][-1].isdigit())):
                out[-1] += t                       # '−' '0.5' -> '−0.5'; '3' '~' '5天' -> '3~5天'
            elif out and len(t) == 1 and _is_han(t) and (out[-1][-1].isdigit() or _is_number(out[-1])):
                out[-1] += t
            else:
                out.append(t)
    return _glue(out)


def _greedy(toks: Sequence[str], widths: Sequence[float], max_w: float) -> list[tuple[int, int]]:
    """Greedy line breaking on token widths: (start, end) token spans per line, without edge spaces."""
    lines, start, end, w = [], None, 0, 0.0
    for i, (t, tw) in enumerate(zip(toks, widths)):
        if t.isspace():
            if start is not None:
                w += tw
            continue
        if start is None:
            start, end, w = i, i + 1, tw
        elif w + tw > max_w:
            lines.append((start, end))
            start, end, w = i, i + 1, tw
        else:
            end, w = i + 1, w + tw
    if start is not None:
        lines.append((start, end))
    return lines


def _narrowest(toks, widths, n: int, lo: float, hi: float) -> int:
    """Smallest integer width in [lo, hi] at which greedy breaking needs at most n lines (balanced lines)."""
    lo_i, hi_i = int(math.ceil(lo)), int(math.ceil(hi))
    while lo_i < hi_i:
        mid = (lo_i + hi_i) // 2
        if len(_greedy(toks, widths, mid)) <= n:
            hi_i = mid
        else:
            lo_i = mid + 1
    return hi_i


def _measure(text: str, language: str, face: Face):
    toks = tokens(text, language)
    return toks, [face.length(t) for t in toks]


def _split_wide(toks: list[str], widths: list[float], face: Face, max_w: float) -> tuple[list[str], list[float]]:
    """Last resort for a unit wider than the line (a URL, a long formula): it is cut between characters."""
    out_t, out_w = [], []
    for t, w in zip(toks, widths):
        if t.isspace() or w <= max_w:
            out_t.append(t)
            out_w.append(w)
            continue
        piece = ""
        for ch in t:
            if piece and face.length(piece + ch) > max_w:
                out_t.append(piece)
                out_w.append(face.length(piece))
                piece = ""
            piece += ch
        if piece:
            out_t.append(piece)
            out_w.append(face.length(piece))
    return out_t, out_w


def wrap(text: str, language: str, face: Face, max_w: float) -> list[str]:
    """Greedy wrap at max_w (a single unbreakable unit wider than max_w gets a line of its own)."""
    toks, widths = _measure(text, language, face)
    return ["".join(toks[a:b]).strip() for a, b in _greedy(toks, widths, max_w)]


def _balanced_layouts(text: str, language: str, face: Face, max_w: float, extra: int = 4) -> list[list[str]]:
    """Line breaks for the fewest lines that fit max_w and for up to `extra` more lines, each at the narrowest
    width giving that many lines, so the lines come out even (no short orphan)."""
    toks, widths = _measure(text, language, face)
    content = [w for t, w in zip(toks, widths) if not t.isspace()]
    if not content or max(content) > max_w:
        return []
    n0 = len(_greedy(toks, widths, max_w))
    out = []
    for n in range(n0, n0 + extra + 1):
        spans = _greedy(toks, widths, _narrowest(toks, widths, n, max(content), max_w))
        if len(spans) != n:
            break
        out.append(["".join(toks[a:b]).strip() for a, b in spans])
    return out


def _ragged(widths: Sequence[float]) -> float:
    if len(widths) < 2:
        return 0.0
    top = max(widths)
    return (top - min(widths)) / top + (0.6 if widths[-1] < 0.35 * top else 0.0)


class TextBlock(NamedTuple):
    """Lines set in one face; unpacks like the prototype's (face, lh, lines, bw, bh)."""
    face: Face
    lh: int
    lines: list[str]
    width: float      # widest line
    height: int       # lh * number of lines


def _line_height(size: int) -> int:
    return int(round(size * LINE))


def fit(text: str, language: str, max_w: float, max_h: float, sizes: Sequence[int], fonts: Optional[FontSet] = None,
        bold: bool = False, aspect: Optional[float] = None, emphasis: bool = False) -> Optional[TextBlock]:
    """The first (largest) size in `sizes` whose balanced wrap fits max_w x max_h. Among the line counts that fit:
    the most even lines without an orphan, the fewest lines, and nearest to `aspect` (width / height) when given."""
    fonts = fonts or default_fonts()
    for size in sizes:
        face = fonts.face(language, size, bold, emphasis)
        lh = _line_height(face.size)
        best = None
        for lines in _balanced_layouts(text, language, face, max_w):
            widths = [face.length(line) for line in lines]
            bw, bh = max(widths), lh * len(lines)
            if bw > max_w + 0.5 or bh > max_h:
                continue
            score = _ragged(widths) + 0.02 * len(lines) + (abs(math.log((bw / bh) / aspect)) * 0.7 if aspect else 0.0)
            if best is None or score < best[0] - 1e-9:
                best = (score, lines, bw, bh)
        if best:
            return TextBlock(face, lh, best[1], best[2], best[3])
    return None


def draw_lines(d: ImageDraw.ImageDraw, block: TextBlock, x: float, y: float, w: float, align: str = "center",
               fill=TEXT_INK) -> None:
    """Draw a block's lines from (x, y) in a column w wide, each line centred vertically in its line box."""
    off = block.face.line_offset(block.lh)
    for i, line in enumerate(block.lines):
        lw = block.face.length(line)
        lx = x + (w - lw) / 2 if align == "center" else (x + w - lw if align == "right" else x)
        block.face.draw(d, (lx, y + i * block.lh + off), line, fill)


def draw_block(d, face: Face, lh: int, lines: list[str], x: float, y: float, w: float, align: str = "center",
               fill=TEXT_INK) -> None:
    """Prototype-compatible form of draw_lines."""
    draw_lines(d, TextBlock(face, lh, lines, 0.0, lh * len(lines)), x, y, w, align, fill)


def _paragraph(text: str, language: str, face: Face, width: float, split: bool = False) -> Optional[list[str]]:
    """Left-aligned paragraph lines: greedy at full width; a short last line (orphan) takes words from the line
    above, the two balanced, so the box stays filled. None when a unit is wider than the line, unless `split`
    allows cutting it."""
    toks, widths = _measure(text, language, face)
    if split:
        toks, widths = _split_wide(toks, widths, face, width)
    content = [w for t, w in zip(toks, widths) if not t.isspace()]
    if not content or max(content) > width:
        return None
    spans = _greedy(toks, widths, width)
    if len(spans) >= 2 and sum(widths[spans[-1][0]:spans[-1][1]]) < 0.3 * width:
        a, b = spans[-2][0], spans[-1][1]
        tail_toks, tail_w = toks[a:b], widths[a:b]
        tail = _greedy(tail_toks, tail_w, _narrowest(tail_toks, tail_w, 2, max(w for t, w in zip(tail_toks, tail_w) if not t.isspace()), width))
        if len(tail) == 2:
            spans = spans[:-2] + [(a + x, a + y) for x, y in tail]
    return ["".join(toks[a:b]).strip() for a, b in spans]


_SUPERSCRIPT = str.maketrans("0123456789-−", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁻")


_CJK_PUNCT = "\u3000-\u303f\uff01-\uff0f\uff1a-\uff20\uff3b-\uff40\uff5b-\uff65"   # 、。「」【】（），：；？！ ...
_CJK_CHAR = "\u3040-\u30ff\u3400-\u9fff\uf900-\ufaff"
_CJK_GAP = re.compile(rf"\s+(?=[{_CJK_PUNCT}])|(?<=[{_CJK_PUNCT}])\s+|(?<=[{_CJK_CHAR}])\s+(?=[{_CJK_CHAR}])")


def _pretty(text: str, face: Optional[Face] = None) -> str:
    """Comic text, not code: '10^19' -> '10¹⁹' (Face draws superscripts as raised digits), whitespace collapsed,
    invisible format characters dropped (zero-width space, soft hyphen, BOM), and no gap between Chinese characters
    or next to full-width punctuation (a line break in the script is not a space in the sentence). With a face, the
    characters it draws as nothing (emoji) are dropped as well."""
    text = "".join(ch for ch in unicodedata.normalize("NFC", str(text or ""))
                   if ch.isspace() or (unicodedata.category(ch) not in ("Cf", "Cc") and (face is None or face.draws(ch))))
    text = _CJK_GAP.sub("", " ".join(text.split()))
    return re.sub(r"(?<=\d)\^([\-−]?\d+)", lambda m: m.group(1).translate(_SUPERSCRIPT), text)


# ----------------------------------------------------------------------------- shapes

def _spow(v: float) -> float:
    return math.copysign(abs(v) ** (2 / P_EXP), v)


def blob(cx: float, cy: float, a: float, b: float, seed: int, n: int = 180) -> list[tuple[float, float]]:
    """A superellipse bubble body with a faint hand-drawn wobble (deterministic for a seed)."""
    rnd = random.Random(seed)
    ph = [rnd.uniform(0, 6.3) for _ in range(2)]
    pts = []
    for i in range(n):
        t = 2 * math.pi * i / n
        k = 1 + 0.008 * math.sin(3 * t + ph[0]) + 0.005 * math.sin(5 * t + ph[1])
        pts.append((cx + a * k * _spow(math.cos(t)), cy + b * k * _spow(math.sin(t))))
    return pts


def edge_point(cx: float, cy: float, a: float, b: float, t: float, shrink: float = 1.0) -> tuple[float, float]:
    """Point of the superellipse at parameter t (shrink < 1 moves it inside)."""
    return cx + shrink * a * _spow(math.cos(t)), cy + shrink * b * _spow(math.sin(t))


def _param_toward(a: float, b: float, dx: float, dy: float) -> float:
    """Superellipse parameter whose edge point lies in direction (dx, dy) from the centre."""
    return math.atan2(math.copysign(abs(dy / b) ** (P_EXP / 2), dy), math.copysign(abs(dx / a) ** (P_EXP / 2), dx))


def _quad_bezier(p0, p1, p2, n: int = 12) -> list[tuple[float, float]]:
    return [((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0],
             (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1]) for t in (i / n for i in range(n + 1))]


def _tail_geometry(cx, cy, a, b, tx, ty, length, s=1.0):
    """(polygon, base point on the edge, tip) of a curved wedge reaching `length` px past the edge toward (tx, ty)."""
    dx, dy = tx - cx, ty - cy
    if math.hypot(dx, dy) < 1e-6 or length <= 0:
        return None
    ex, ey = edge_point(cx, cy, a, b, _param_toward(a, b, dx, dy))
    dist = math.hypot(tx - ex, ty - ey)
    if dist < 1e-6:
        return None
    ux, uy = (tx - ex) / dist, (ty - ey) / dist
    tip = (ex + ux * length, ey + uy * length)
    # the base narrows for short stubs, so a clipped tail still reads as a pointer, not a bump
    half = min(0.32 * min(a, b), 8 * s + 0.32 * length)
    r = max(1.0, math.hypot(ex - cx, ey - cy))
    ang = math.atan2(ey - cy, ex - cx)
    bases = []
    for da in (-half / r, half / r):
        t = _param_toward(a, b, math.cos(ang + da), math.sin(ang + da))
        bases.append(edge_point(cx, cy, a, b, t, 0.86))
    bend = (-uy * 0.22 * length, ux * 0.22 * length)
    m1 = ((bases[0][0] + tip[0]) / 2 + bend[0], (bases[0][1] + tip[1]) / 2 + bend[1])
    m2 = ((bases[1][0] + tip[0]) / 2 + bend[0] * 0.4, (bases[1][1] + tip[1]) / 2 + bend[1] * 0.4)
    poly = _quad_bezier(bases[0], m1, tip) + _quad_bezier(tip, m2, bases[1])[1:]
    return poly, (ex, ey), tip


def tail(cx: float, cy: float, a: float, b: float, tx: float, ty: float, max_len: float = 85, s: float = 1.0):
    """Tail polygon toward (tx, ty), at most max_len px long and stopping short of the target (None when too close)."""
    ex, ey = edge_point(cx, cy, a, b, _param_toward(a, b, tx - cx, ty - cy))
    dist = math.hypot(tx - ex, ty - ey)
    if dist < 12 * s:
        return None
    geo = _tail_geometry(cx, cy, a, b, tx, ty, min(max_len, dist * 0.75), s)
    return geo[0] if geo else None


def ink(img: Image.Image, polys: Sequence[Sequence[tuple[float, float]]], stroke: float = 5, fill=PAPER, color=INK,
        ss: int = SUPERSAMPLE) -> None:
    """Fill the union of the polygons and outline the union (not each shape): drawn ss times larger, then
    box-filtered down for smooth edges. Ink is laid under the fill so no art shows through at the seam."""
    polys = [p for p in polys if p and len(p) >= 3]
    if not polys:
        return
    pad = stroke + 2
    xs = [x for p in polys for x, _ in p]
    ys = [y for p in polys for _, y in p]
    x0, y0 = max(0, int(min(xs) - pad)), max(0, int(min(ys) - pad))
    x1, y1 = min(img.width, int(math.ceil(max(xs) + pad))), min(img.height, int(math.ceil(max(ys) + pad)))
    if x1 <= x0 or y1 <= y0:
        return
    w, h = x1 - x0, y1 - y0
    inner, outer = Image.new("L", (w * ss, h * ss), 0), Image.new("L", (w * ss, h * ss), 0)
    di, do = ImageDraw.Draw(inner), ImageDraw.Draw(outer)
    for p in polys:
        q = [((x - x0) * ss, (y - y0) * ss) for x, y in p]
        di.polygon(q, fill=255)
        do.polygon(q, fill=255)
        do.line(q + q[:2], fill=255, width=max(1, int(round(2 * stroke * ss))), joint="curve")
    inner, outer = inner.resize((w, h), Image.BOX), outer.resize((w, h), Image.BOX)
    region = img.crop((x0, y0, x1, y1))
    region.paste(color, (0, 0, w, h), outer)
    region.paste(fill, (0, 0, w, h), inner)
    img.paste(region, (x0, y0))


# ----------------------------------------------------------------------------- picture analysis

def _edges(img: Image.Image) -> np.ndarray:
    """Edge map (H x W bool) used to find calm areas."""
    return np.asarray(img.convert("L").filter(ImageFilter.BoxBlur(1)).filter(ImageFilter.FIND_EDGES)) > 38


def _largest_rect(grid: np.ndarray, min_w: int = 1, min_h: int = 1) -> Optional[tuple[int, int, int, int]]:
    """Largest all-True axis-aligned rectangle (c1, r1, c2, r2), exclusive ends, at least min_w x min_h cells."""
    gh, gw = grid.shape
    best, heights = (0, None), np.zeros(gw, int)
    for r in range(gh):
        heights = np.where(grid[r], heights + 1, 0)
        stack: list[tuple[int, int]] = []
        for c in range(gw + 1):
            h = int(heights[c]) if c < gw else 0
            start = c
            while stack and stack[-1][1] >= h:
                start, hh = stack.pop()
                width = c - start
                if width >= min_w and hh >= min_h and hh * width > best[0]:
                    best = (hh * width, (start, r - hh + 1, c, r + 1))
            stack.append((start, h))
    return best[1]


def _frame(img: Image.Image) -> Rect:
    """Inside of the thin panel border the art model draws (the image bounds when there is none)."""
    g = np.asarray(img.convert("L"))
    H, W = g.shape
    dark = g < 110

    def scan(profile: np.ndarray) -> int:
        for i, v in enumerate(profile):
            if v > 0.55:
                while i + 1 < len(profile) and profile[i + 1] > 0.55:
                    i += 1
                return i + 1
        return 0

    rows, cols = slice(int(H * 0.1), int(H * 0.9)), slice(int(W * 0.1), int(W * 0.9))
    lx, ly = max(1, int(W * 0.08)), max(1, int(H * 0.08))
    left = scan(dark[rows, :lx].mean(axis=0))
    right = scan(dark[rows, W - lx:][:, ::-1].mean(axis=0))
    top = scan(dark[:ly, cols].mean(axis=1))
    bottom = scan(dark[H - ly:, cols][::-1].mean(axis=1))
    return (float(left), float(top), float(W - right), float(H - bottom))


def free_band(img: Image.Image, avoid: Sequence[Rect] = (), max_y_frac: float = 0.5, cell: int = 8,
              edges: Optional[np.ndarray] = None, within: Optional[Rect] = None) -> Rect:
    """Largest calm rectangle (low edge density) in the upper part of the picture, clear of `avoid` (heads),
    preferring rectangles at least half the panel wide."""
    W, H = img.size
    s = W / 1024
    e = edges if edges is not None else _edges(img)
    sub = e[: int(H * max_y_frac)]
    gh, gw = sub.shape[0] // cell, sub.shape[1] // cell
    if gh < 1 or gw < 1:
        return within or (0.0, 0.0, float(W), float(H) * 0.25)
    calm = sub[: gh * cell, : gw * cell].reshape(gh, cell, gw, cell).mean(axis=(1, 3)) < 0.03
    if within:
        calm[:, : max(0, int(math.ceil(within[0] / cell)))] = False
        calm[:, max(0, int(within[2] // cell)):] = False
        calm[: max(0, int(math.ceil(within[1] / cell))), :] = False
    for (x1, y1, x2, y2) in avoid:
        r1, r2 = max(0, int((y1 - 8 * s) // cell)), max(0, int(math.ceil((y2 + 8 * s) / cell)))
        c1, c2 = max(0, int((x1 - 8 * s) // cell)), max(0, int(math.ceil((x2 + 8 * s) / cell)))
        calm[r1:r2, c1:c2] = False
    min_h, min_w = max(2, int(math.ceil(64 * s / cell))), max(2, int(math.ceil(160 * s / cell)))   # stays positive after the insets
    rect = _largest_rect(calm, min_w=max(min_w, gw // 2), min_h=min_h) or _largest_rect(calm, min_w=min_w, min_h=min_h)
    if rect is None:
        ux1, uy1, ux2, _ = within or (0.0, 0.0, float(W), float(H))
        return (ux1, uy1, ux2, uy1 + H * 0.22)
    c1, r1, c2, r2 = rect
    return (c1 * cell + 14 * s, r1 * cell + 14 * s, c2 * cell - 14 * s, r2 * cell - 10 * s)


def clean_rect(img: Image.Image, quad: Sequence[Sequence[float]], cell: int = 8, inset: float = 0.03,
               exclude: Sequence[Rect] = ()) -> Optional[Rect]:
    """Largest axis-aligned rectangle inside the board quad whose pixels are all blank board surface
    (light, unsaturated, no edges), clear of `exclude` (heads in front of the board). None when there is none."""
    pts = _quad_points(quad)
    if not pts:
        return None
    W, H = img.size
    xs, ys = sorted(p[0] for p in pts), sorted(p[1] for p in pts)
    bx1, bx2, by1, by2 = xs[1], xs[2], ys[1], ys[2]          # inner box of a slightly slanted quad
    padx, pady = (bx2 - bx1) * inset, (by2 - by1) * inset
    bx1, bx2 = max(0, int(bx1 + padx)), min(W, int(bx2 - padx))
    by1, by2 = max(0, int(by1 + pady)), min(H, int(by2 - pady))
    if bx2 - bx1 < 4 * cell or by2 - by1 < 4 * cell:
        return None
    a = np.asarray(img.convert("RGB"), dtype=np.int16)[by1:by2, bx1:bx2]
    lum, chroma = a.mean(axis=2), a.max(axis=2) - a.min(axis=2)
    thr = max(150.0, min(185.0, float(np.median(lum)) - 30))   # a shaded board still counts, a dark wall does not
    blank = (lum > thr) & (chroma < 40) & ~_edges(img)[by1:by2, bx1:bx2]
    for (x1, y1, x2, y2) in exclude:
        blank[max(0, int(y1) - 12 - by1): max(0, int(y2) + 12 - by1), max(0, int(x1) - 12 - bx1): max(0, int(x2) + 12 - bx1)] = False
    gh, gw = blank.shape[0] // cell, blank.shape[1] // cell
    grid = blank[: gh * cell, : gw * cell].reshape(gh, cell, gw, cell).mean(axis=(1, 3)) > 0.97
    rect = _largest_rect(grid, min_w=2, min_h=2)
    if rect is None:
        return None
    c1, r1, c2, r2 = rect
    return (float(bx1 + c1 * cell), float(by1 + r1 * cell), float(bx1 + c2 * cell), float(by1 + r2 * cell))


def _quad_points(quad) -> Optional[list[tuple[float, float]]]:
    """4 corner points; also accepts a bare [x1, y1, x2, y2] box."""
    try:
        if quad is None:
            return None
        flat = list(quad)
        if len(flat) == 4 and all(isinstance(v, (int, float)) for v in flat):
            x1, y1, x2, y2 = (float(v) for v in flat)
            return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        pts = [(float(p[0]), float(p[1])) for p in flat]
        return pts if len(pts) == 4 else None
    except (TypeError, ValueError, IndexError):
        return None


def onto_rect(art: Image.Image, chart: Image.Image | Callable[[int, int], Image.Image], rect: Rect) -> Image.Image:
    """Multiply a white-background chart onto the art inside rect, so the board's light and texture stay."""
    x1, y1, x2, y2 = (int(round(v)) for v in rect)
    if x2 <= x1 or y2 <= y1:
        return art
    image = chart(x2 - x1, y2 - y1) if callable(chart) else chart
    image = image.convert("RGB")
    if image.size != (x2 - x1, y2 - y1):
        image = image.resize((x2 - x1, y2 - y1), Image.LANCZOS)
    art.paste(ImageChops.multiply(art.crop((x1, y1, x2, y2)), image), (x1, y1))
    return art


# ----------------------------------------------------------------------------- narration

@dataclass
class NarrationBox:
    rect: Rect
    block: TextBlock
    pad: float


def narration_fit(text: str, language: str, fonts: FontSet, box: Rect, max_h: Optional[float] = None,
                  sizes: Optional[Sequence[int]] = None, scale: float = 1.0) -> Optional[NarrationBox]:
    """Measure a narration box across the full width of `box`, at its top, with the largest size that keeps
    it at most max_h tall. None when no size fits."""
    x1, y1, x2, _ = box
    pad = 22 * scale
    for size in sizes or _scaled(NARRATION_SIZES[_lang(language)], scale):
        face = fonts.face(language, size, emphasis=True)
        lines = _paragraph(_pretty(text, face), language, face, x2 - x1 - 2 * pad)
        if not lines:
            continue
        lh = _line_height(face.size)
        h = lh * len(lines) + 2 * pad - 6 * scale
        if max_h is not None and h > max_h:
            continue
        width = max(face.length(line) for line in lines)
        return NarrationBox((x1, y1, x2, y1 + h), TextBlock(face, lh, lines, width, lh * len(lines)), pad)
    return None


def draw_narration(img: Image.Image, nb: NarrationBox, scale: float = 1.0) -> None:
    x1, y1, x2, y2 = nb.rect
    d = ImageDraw.Draw(img)
    d.rectangle([x1, y1, x2, y2], fill=CREAM, outline=INK, width=max(2, int(round(4 * scale))))
    draw_lines(d, nb.block, x1 + nb.pad, y1 + nb.pad - 3 * scale, x2 - x1 - 2 * nb.pad, align="left")


def narration(img: Image.Image, text: str, language: str, fonts: FontSet, box: Rect,
              sizes: Optional[Sequence[int]] = None, scale: float = 1.0) -> Optional[Rect]:
    """Fit and draw a narration box at the top of box; returns its rect."""
    nb = narration_fit(text, language, fonts, box, None, sizes, scale)
    if nb:
        draw_narration(img, nb, scale)
    return nb.rect if nb else None


# ----------------------------------------------------------------------------- bubbles

def _scaled(sizes: Sequence[int], s: float) -> tuple[int, ...]:
    """Font sizes for this panel width (given for 1024 px), largest first, without duplicates."""
    out: list[int] = []
    for v in sizes:
        px = max(9, int(round(v * s)))
        if px not in out:
            out.append(px)
    return tuple(out)


def _overlap(r1: Rect, r2: Rect, gap: float = 0.0) -> bool:
    return not (r1[2] + gap <= r2[0] or r2[2] + gap <= r1[0] or r1[3] + gap <= r2[1] or r2[3] + gap <= r1[1])


def reading_order_ok(prev: Rect, rect: Rect, s: float = 1.0) -> bool:
    """Horizontal reading order (en and zh): the next bubble is to the right on the same row, or below."""
    if rect[0] >= (prev[0] + prev[2]) / 2 and rect[1] >= prev[1] - 8 * s:
        return True
    return rect[1] >= prev[3] - 0.15 * (prev[3] - prev[1])


@dataclass
class PlacedBubble:
    speaker: str
    text: str
    block: TextBlock
    cx: float
    cy: float
    a: float          # semi-axes of the body
    b: float
    pad: float        # outline width around the body
    chip: bool = False   # drawn with a name tab: the speaker's first bubble in the panel, or a tail that cannot reach them

    @property
    def rect(self) -> Rect:
        return (self.cx - self.a - self.pad, self.cy - self.b - self.pad, self.cx + self.a + self.pad, self.cy + self.b + self.pad)


@dataclass
class _Shape:
    block: TextBlock
    a: float
    b: float
    fw: float          # footprint with the outline
    fh: float
    cost: float


def _axes(block: TextBlock, widths: Sequence[float]) -> tuple[float, float]:
    """Semi-axes of the superellipse body that holds every line with a comfortable margin."""
    size, lh, n = block.face.size, block.lh, len(block.lines)
    half = lh * n / 2
    b = half + 0.62 * size
    a = 0.0
    for i, w in enumerate(widths):
        top = -half + i * lh + 0.16 * lh
        y = max(abs(top), abs(top + 0.72 * lh))
        k = (1 - min(0.9, y / b) ** P_EXP) ** (1 / P_EXP)
        a = max(a, (w / 2 + 0.55 * size) / k)
    return max(a, 0.9 * b), b


@dataclass
class _Ctx:
    """One panel being lettered: the art, its analysis and the size-dependent constants."""
    img: Image.Image
    language: str
    fonts: FontSet
    heads: dict[str, Rect]
    quad: Optional[list[tuple[float, float]]] = None
    shapes_cache: dict = field(default_factory=dict)
    objects: list = field(default_factory=list)   # boxes of the objects that carry a scene's idea (kept clear when the words fit)

    def __post_init__(self):
        self.lang = _lang(self.language)
        self.W, self.H = self.img.size
        self.s = self.W / 1024
        self.edges = _edges(self.img)
        self.frame = _frame(self.img)
        m = 12 * self.s
        fx1, fy1, fx2, fy2 = self.frame
        self.usable = (fx1 + m, fy1 + m, fx2 - m, fy2 - m)
        self.stroke = max(2.0, round(5 * self.s))
        self.bubble_gap = 14 * self.s
        self.chart_gap = 20 * self.s
        self.bubble_sizes = _scaled(BUBBLE_SIZES[self.lang], self.s)
        self.narration_sizes = _scaled(NARRATION_SIZES[self.lang], self.s)

    def head_gap(self, head: Rect) -> float:
        """Clearance around a head box: grounding boxes are approximate, and a tail stub needs room."""
        return 16 * self.s + 0.04 * max(head[2] - head[0], head[3] - head[1])

    def shapes(self, text: str, size: int) -> list[_Shape]:
        """Bubble shapes for a text at one size: the balanced wraps from the fewest lines to a few more."""
        key = (text, size)
        if key not in self.shapes_cache:
            face = self.fonts.face(self.language, size, emphasis=True)
            shown = _pretty(text, face)
            lh = _line_height(face.size)
            ux1, uy1, ux2, uy2 = self.usable
            max_w = min(0.62 * self.W, (ux2 - ux1) - 1.4 * size - 2 * self.stroke)
            out = []
            for lines in _balanced_layouts(shown, self.lang, face, max_w):
                widths = [face.length(line) for line in lines]
                block = TextBlock(face, lh, lines, max(widths), lh * len(lines))
                a, b = _axes(block, widths)
                fw, fh = 2 * (a + self.stroke), 2 * (b + self.stroke)
                if fw > ux2 - ux1 or fh > uy2 - uy1:
                    continue
                cost = 70 * self.s * abs(math.log((a / b) / 1.9)) + 40 * self.s * _ragged(widths)
                out.append(_Shape(block, a, b, fw, fh, cost))
            self.shapes_cache[key] = out
        return self.shapes_cache[key]


def _sat(a: np.ndarray) -> np.ndarray:
    out = np.zeros((a.shape[0] + 1, a.shape[1] + 1), np.float64)
    out[1:, 1:] = a.cumsum(0).cumsum(1)
    return out


class _Planner:
    """Places bubble bodies on a coarse occupancy grid. Every free position of every candidate shape is scored at
    once with summed-area tables; a small branch-and-bound then picks the cheapest set in reading order."""

    def __init__(self, ctx: _Ctx, band: Rect, avoid: Sequence[tuple[Rect, float]]):
        self.ctx, self.band = ctx, band
        c = self.cell = max(2, int(round(4 * ctx.s)))
        self.gw, self.gh = -(-ctx.W // c), -(-ctx.H // c)
        occ = np.zeros((self.gh, self.gw), np.float64)
        ux1, uy1, ux2, uy2 = ctx.usable
        occ[:, : max(0, int(math.ceil(ux1 / c)))] = 1
        occ[:, max(0, int(ux2 // c)):] = 1
        occ[: max(0, int(math.ceil(uy1 / c))), :] = 1
        occ[max(0, int(uy2 // c)):, :] = 1
        for head in ctx.heads.values():
            self.block(occ, head, ctx.head_gap(head))
        for rect, gap in avoid:
            self.block(occ, rect, gap)
        self.occ = occ
        e = np.zeros((self.gh * c, self.gw * c), np.float64)
        e[: ctx.H, : ctx.W] = ctx.edges
        self.busy = _sat(e.reshape(self.gh, c, self.gw, c).mean(axis=(1, 3)))

    def block(self, occ: np.ndarray, rect: Rect, gap: float) -> None:
        c = self.cell
        x1, y1 = max(0, int((rect[0] - gap) // c)), max(0, int((rect[1] - gap) // c))
        x2, y2 = min(self.gw, int(math.ceil((rect[2] + gap) / c))), min(self.gh, int(math.ceil((rect[3] + gap) / c)))
        if x2 > x1 and y2 > y1:
            occ[y1:y2, x1:x2] = 1

    def _cost(self, cx, cy, x1, y1, x2, y2, head) -> np.ndarray:
        s = self.ctx.s
        if head is None:
            return 0.3 * np.abs(cx - self.ctx.W / 2) + 0.12 * y1
        hx1, hy1, hx2, hy2 = head
        above, below = hy1 - y2, y1 - hy2          # > 0: the bubble is above / below the head
        vert = np.where(above >= 0, np.maximum(0.0, above - 90 * s) * 0.5,
                        np.where(below >= 0, 160 * s + 0.6 * below, 50 * s))
        return 0.8 * np.abs(cx - (hx1 + hx2) / 2) + vert + 0.12 * y1

    def candidates(self, shapes: Sequence[_Shape], sat: np.ndarray, prev: Optional[Rect], head: Optional[Rect],
                   k: int) -> list[tuple[float, float, float, _Shape]]:
        s, c = self.ctx.s, self.cell
        bx1, by1, bx2, by2 = self.band
        pool = []
        for si, sh in enumerate(shapes):
            kw, kh = int(math.ceil(sh.fw / c)), int(math.ceil(sh.fh / c))
            if kw >= self.gw or kh >= self.gh:
                continue
            win = sat[kh:, kw:] - sat[:-kh, kw:] - sat[kh:, :-kw] + sat[:-kh, :-kw]
            jj, ii = np.nonzero(win[::2, ::2] < 0.5)
            if not len(jj):
                continue
            j, i = jj * 2, ii * 2
            cx, cy = i * c + kw * c / 2.0, j * c + kh * c / 2.0
            x1, y1 = cx - sh.fw / 2, cy - sh.fh / 2
            x2, y2 = x1 + sh.fw, y1 + sh.fh
            if prev is not None:
                px1, py1, px2, py2 = prev
                keep = (y1 >= py2 - 0.15 * (py2 - py1)) | ((x1 >= (px1 + px2) / 2) & (y1 >= py1 - 8 * s))
                if not keep.any():
                    continue
                j, i, cx, cy, x1, y1, x2, y2 = (v[keep] for v in (j, i, cx, cy, x1, y1, x2, y2))
            busy = (self.busy[j + kh, i + kw] - self.busy[j, i + kw] - self.busy[j + kh, i] + self.busy[j, i]) / (kw * kh)
            in_band = (x1 >= bx1 - c) & (y1 >= by1 - c) & (x2 <= bx2 + c) & (y2 <= by2 + c)
            cost = self._cost(cx, cy, x1, y1, x2, y2, head) + busy * 320 * s + np.where(in_band, 0.0, 400 * s) + sh.cost
            for idx in np.argsort(cost, kind="stable")[:300]:
                pool.append((float(cost[idx]), si, float(cy[idx]), float(cx[idx]), sh))
        pool.sort(key=lambda t: t[:4])
        picked: list[tuple[float, float, float, _Shape]] = []
        for cost, _, cy, cx, sh in pool:
            if all(math.hypot(cx - p[1], cy - p[2]) > 80 * s for p in picked):
                picked.append((cost, cx, cy, sh))
                if len(picked) >= k:
                    break
        return picked

    def search(self, items: Sequence[tuple[str, str, Sequence[_Shape]]], nodes: int = 60):
        """(best complete placement or None, deepest partial placement) for the items in order."""
        best: list = [None, math.inf]
        deepest: list = [0, math.inf, []]
        budget = [nodes]
        n = len(items)

        def rec(k: int, occ: np.ndarray, placed: list[PlacedBubble], total: float) -> None:
            if k > deepest[0] or (k == deepest[0] and total < deepest[1]):
                deepest[:] = [k, total, list(placed)]
            if k == n:
                if total < best[1]:
                    best[:] = [list(placed), total]
                return
            if budget[0] <= 0 or total >= best[1]:
                return
            budget[0] -= 1
            speaker, text, shapes = items[k]
            prev = placed[-1].rect if placed else None
            for cost, cx, cy, sh in self.candidates(shapes, _sat(occ), prev, self.ctx.heads.get(speaker), 4 if k == 0 else 3):
                pb = PlacedBubble(speaker, text, sh.block, cx, cy, sh.a, sh.b, self.ctx.stroke)
                nxt = occ.copy()
                self.block(nxt, pb.rect, self.ctx.bubble_gap)
                placed.append(pb)
                rec(k + 1, nxt, placed, total + cost)
                placed.pop()

        rec(0, self.occ, [], 0.0)
        return best[0], deepest[2]


def _plan(ctx: _Ctx, bubbles: Sequence[tuple[str, str]], band: Rect, avoid: Sequence[tuple[Rect, float]],
          planner: Optional[_Planner] = None) -> Optional[tuple[int, list[PlacedBubble]]]:
    """(font size, placements) at the largest size where every bubble fits; None when even the smallest fails."""
    if not bubbles:
        return 0, []
    planner = planner or _Planner(ctx, band, avoid)
    for size in ctx.bubble_sizes:
        items = [(spk, text, ctx.shapes(text, size)) for spk, text in bubbles]
        if any(not shapes for _, _, shapes in items):
            continue
        full, _ = planner.search(items)
        if full:
            return size, full
    return None


def _plan_prefix(ctx: _Ctx, bubbles: Sequence[tuple[str, str]], band: Rect, avoid: Sequence[tuple[Rect, float]]):
    """The longest leading run of bubbles that fits: (size, placements, leftover bubbles)."""
    planner = _Planner(ctx, band, avoid)
    for k in range(len(bubbles), 0, -1):
        res = _plan(ctx, bubbles[:k], band, avoid, planner)
        if res:
            return res[0], res[1], list(bubbles[k:])
    return 0, [], list(bubbles)


def plan_bubbles(W: int, H: int, band: Rect, bubbles: Sequence[tuple[str, str]], heads: dict[str, Rect],
                 avoid: Sequence[Rect], language: str, fonts: Optional[FontSet] = None, img: Optional[Image.Image] = None,
                 gap: float = 20.0) -> list[PlacedBubble]:
    """Place every bubble at one font size (the largest that fits all), preferring the band, never over heads,
    the avoid rects or each other, in reading order. Raises LayoutError when they cannot all be placed."""
    ctx = _Ctx(img if img is not None else Image.new("RGB", (W, H), PAPER), language, fonts or default_fonts(),
               {_name_key(k): v for k, v in heads.items()})
    res = _plan(ctx, bubbles, band, [(r, gap * ctx.s) for r in avoid])
    if res is None:
        raise LayoutError("no room for the bubbles")
    return res[1]


def _tails(ctx: _Ctx, placed: Sequence[PlacedBubble], bodies: Sequence[list], blocked: Sequence[Rect]):
    """Per bubble (polygon, base, tip) or None: aimed at the speaker's head and cut short before it would enter a
    head box, the chart, the narration, another bubble or an earlier tail; at least a short stub when possible."""
    W, H, s, st = ctx.W, ctx.H, ctx.s, ctx.stroke
    base = np.zeros((H, W), bool)
    fx1, fy1, fx2, fy2 = (int(round(v)) for v in ctx.frame)
    e = int(round(4 * s))
    base[:, : max(0, fx1 + e)] = True
    base[:, max(0, fx2 - e):] = True
    base[: max(0, fy1 + e), :] = True
    base[max(0, fy2 - e):, :] = True
    grow = st + 2 * s
    for x1, y1, x2, y2 in blocked:
        base[max(0, int(y1 - grow)): max(0, int(math.ceil(y2 + grow))), max(0, int(x1 - grow)): max(0, int(math.ceil(x2 + grow)))] = True
    max_len, stub = 85 * s, 14 * s
    lengths = sorted({max_len * f for f in (1, 0.85, 0.72, 0.6, 0.5, 0.41, 0.33, 0.26, 0.2) if max_len * f >= stub} | {stub}, reverse=True)
    out, drawn = [], []
    for i, p in enumerate(placed):
        head = ctx.heads.get(p.speaker)
        if head is None:
            out.append(None)
            continue
        layer = Image.new("L", (W, H), 0)
        d = ImageDraw.Draw(layer)
        for j, poly in enumerate(bodies):
            if j != i:
                d.polygon(poly, fill=255)
                d.line(poly + poly[:1], fill=255, width=int(2 * st + 2))
        for poly in drawn:
            d.polygon(poly, fill=255)
            d.line(poly + poly[:1], fill=255, width=int(2 * st + 2))
        mask = base | (np.asarray(layer) > 0)
        hx1, hy1, hx2, hy2 = head
        hx, hy = (hx1 + hx2) / 2, (hy1 + hy2) / 2
        targets = [(hx, hy1 + 0.3 * (hy2 - hy1))]
        targets += sorted([(hx, hy1), (hx1, hy), (hx2, hy), (hx, hy2)], key=lambda t: math.hypot(t[0] - p.cx, t[1] - p.cy))
        found = None
        for tx, ty in targets:
            for length in lengths:
                geo = _tail_geometry(p.cx, p.cy, p.a, p.b, tx, ty, length, s)
                if geo and not _hits(mask, geo[0], st):
                    found = geo
                    break
            if found:
                break
        if found and not _tail_reaches(head, found, s):
            tip = found[2]
            if any(_rect_gap(h, tip) <= 30 * s for name, h in ctx.heads.items() if name != p.speaker):
                found = None   # cut short at another head, the stub points at that character; the name tab says who speaks
        out.append(found)
        if found:
            drawn.append(found[0])
    return out


def _rect_gap(rect: Rect, pt: tuple[float, float]) -> float:
    """Distance from a point to a box (0 inside it)."""
    x1, y1, x2, y2 = rect
    return math.hypot(max(x1 - pt[0], 0.0, pt[0] - x2), max(y1 - pt[1], 0.0, pt[1] - y2))


def _hits(mask: np.ndarray, poly: Sequence[tuple[float, float]], stroke: float) -> bool:
    """Does the polygon, with its outline, touch the mask (or leave the image)?"""
    H, W = mask.shape
    xs, ys = [x for x, _ in poly], [y for _, y in poly]
    pad = stroke + 2
    x0, y0, x1, y1 = int(min(xs) - pad), int(min(ys) - pad), int(math.ceil(max(xs) + pad)), int(math.ceil(max(ys) + pad))
    if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
        return True
    im = Image.new("L", (x1 - x0, y1 - y0), 0)
    d = ImageDraw.Draw(im)
    q = [(x - x0, y - y0) for x, y in poly]
    d.polygon(q, fill=255)
    d.line(q + q[:1], fill=255, width=int(2 * stroke))
    return bool((np.asarray(im) > 0)[mask[y0:y1, x0:x1]].any())


def _tail_reaches(head: Optional[Rect], tail, s: float) -> bool:
    """Does the tail end at (or close to) the speaker's head? A stub that stops at a board does not."""
    if not head or not tail:
        return False
    return _rect_gap(head, tail[2]) <= max(70 * s, 0.6 * (head[3] - head[1]))


def _speaker_chip(img: Image.Image, p: PlacedBubble, ctx: _Ctx, avoid: Sequence[Rect] = ()) -> Rect:
    """A name tab straddling the bubble's outline: who speaks. Top left, else the first corner (top right, bottom left,
    bottom right) that keeps clear of ``avoid`` (heads, the chart, the narration, other bubbles and tabs); returns its
    rectangle."""
    name = p.speaker[:1].upper() + p.speaker[1:]
    size = max(12, int(round(max(30 * ctx.s, 0.78 * p.block.face.size))))
    face = ctx.fonts.face("en-US", size, bold=True)   # cast names are Latin in every language
    padx, h = 0.5 * size, 1.5 * size
    w = face.length(name) + 2 * padx

    def at(left: bool, top: bool) -> Rect:
        x = min(max(ctx.frame[0] + ctx.stroke, p.cx + (-0.55 if left else 0.55) * p.a - w / 2), ctx.frame[2] - ctx.stroke - w)
        y = p.cy - p.b - p.pad - 0.6 * h if top else p.cy + p.b + p.pad - 0.4 * h
        y = min(max(ctx.frame[1] + ctx.stroke, y), ctx.frame[3] - ctx.stroke - h)
        return (x, y, x + w, y + h)

    corners = [at(True, True), at(False, True), at(True, False), at(False, False)]
    x, y, _, _ = next((r for r in corners if not any(_overlap(r, o) for o in avoid)), corners[0])
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((x, y, x + w, y + h), radius=h / 2, fill=CHIP_TINTS.get(p.speaker, CREAM), outline=INK,
                        width=max(2, int(round(ctx.stroke * 0.8))))
    face.draw(d, (x + padx, y + (h - size * 1.2) / 2), name, INK)
    return (x, y, x + w, y + h)


def draw_bubbles(img: Image.Image, placed: Sequence[PlacedBubble], heads: dict[str, Rect], blocked: Sequence[Rect] = (),
                 language: str = "en", fonts: Optional[FontSet] = None, seed: int = 3, ctx: Optional[_Ctx] = None) -> list:
    """Draw bodies, tails (supersampled ink) and text; returns the (polygon, base, tip) tail per bubble or None. Each
    speaker's first bubble in the panel gets a name tab, and so does any bubble whose tail cannot reach its speaker (cut
    short by a board, or a speaker without a head box): every reader of the validation manga had to work out speakers
    from the tails, and met the cast unnamed."""
    ctx = ctx or _Ctx(img, language, fonts or default_fonts(), {_name_key(k): v for k, v in heads.items()})
    bodies = [blob(p.cx, p.cy, p.a, p.b, seed + 11 * i) for i, p in enumerate(placed)]
    tails = _tails(ctx, placed, bodies, blocked)
    for body, tl in zip(bodies, tails):
        ink(img, [body] + ([tl[0]] if tl else []), ctx.stroke)
    d = ImageDraw.Draw(img)
    for p in placed:
        bk = p.block
        draw_lines(d, bk, p.cx - bk.width / 2, p.cy - bk.height / 2, bk.width)
    named: set = set()
    taken = [*ctx.heads.values(), *blocked]
    for p, tl in zip(placed, tails):   # even alone in the panel: a tail that ends on the chart says nobody is speaking
        if p.speaker and (p.speaker not in named or not _tail_reaches(ctx.heads.get(p.speaker), tl, ctx.s)):
            p.chip = True
            taken.append(_speaker_chip(img, p, ctx, taken + [q.rect for q in placed if q is not p]))
        named.add(p.speaker)
    return tails


# ----------------------------------------------------------------------------- panel layout

@dataclass
class _Plan:
    kind: str = "scene"
    spec: Optional[dict] = None
    chart: Optional[Rect] = None
    chart_mode: Optional[str] = None             # board | card | strip
    narration_box: Optional[NarrationBox] = None
    narration_mode: Optional[str] = None         # box | subtitle | strip
    size: int = 0
    bubbles: list = field(default_factory=list)
    leftover: list = field(default_factory=list)   # (speaker, text) moved to the caption strip
    fallback: bool = False


def _name_key(name) -> str:
    """How speakers and head boxes are matched: 'Papi (brown dog)' -> 'papi'."""
    return re.split(r"\s*[(（]", str(name or "").strip().lower())[0].strip()


def _panel_parts(panel) -> tuple[str, Optional[dict], list[tuple[str, str]], str]:
    """kind, board spec, (speaker, text) bubbles and narration of a PanelV2 (or its dict form)."""
    def get(key: str):
        return panel.get(key) if isinstance(panel, dict) else getattr(panel, key, None)

    bubbles = []
    for b in get("dialogue") or []:
        speaker = b.get("speaker") if isinstance(b, dict) else getattr(b, "speaker", "")
        text = b.get("text") if isinstance(b, dict) else getattr(b, "text", "")
        text = " ".join(str(text or "").split())
        if text:
            bubbles.append((_name_key(speaker), text))
    kind = str(get("kind") or get("type") or "scene").strip().lower()
    board = get("board")
    spec = dict(board) if kind == "board" and isinstance(board, dict) and board else None
    return kind, spec, bubbles, " ".join(str(get("narration") or "").split())


def _scaled_grounding(g: Optional[Grounding], W: int, H: int):
    """Head boxes (clamped, normalized order) and board quad in the art's own pixels."""
    if g is None:
        return {}, None
    gw, gh = (g.size or (0, 0))[:2] if g.size else (0, 0)
    sx, sy = (W / gw, H / gh) if gw and gh else (1.0, 1.0)
    heads = {}
    for name, box in (g.heads or {}).items():
        try:
            x1, y1, x2, y2 = (float(v) for v in box)
        except (TypeError, ValueError):
            continue
        if not all(map(math.isfinite, (x1, y1, x2, y2))):
            continue
        x1, x2 = sorted((x1 * sx, x2 * sx))
        y1, y2 = sorted((y1 * sy, y2 * sy))
        x1, x2, y1, y2 = max(0.0, x1), min(float(W), x2), max(0.0, y1), min(float(H), y2)
        if x2 - x1 >= 4 and y2 - y1 >= 4:
            heads[_name_key(name)] = (x1, y1, x2, y2)
    pts = _quad_points(g.board)
    quad = [(x * sx, y * sy) for x, y in pts] if pts else None
    return heads, quad


def _scaled_boxes(boxes, g: Optional[Grounding], W: int, H: int) -> list[Rect]:
    """Boxes made on the grounding's art size, in this art's pixels (clamped; degenerate ones dropped)."""
    gw, gh = (g.size or (0, 0))[:2] if g is not None and g.size else (0, 0)
    sx, sy = (W / gw, H / gh) if gw and gh else (1.0, 1.0)
    out = []
    for box in boxes or []:
        try:
            x1, y1, x2, y2 = (float(v) for v in box)
        except (TypeError, ValueError):
            continue
        if not all(map(math.isfinite, (x1, y1, x2, y2))):
            continue
        x1, x2 = sorted((max(0.0, x1 * sx), min(float(W), x2 * sx)))
        y1, y2 = sorted((max(0.0, y1 * sy), min(float(H), y2 * sy)))
        if x2 - x1 >= 4 and y2 - y1 >= 4:
            out.append((x1, y1, x2, y2))
    return out


def _place_narration(ctx: _Ctx, text: str, boxes: Sequence[Rect], max_h: float, avoid: Sequence[Rect]) -> Optional[NarrationBox]:
    """First box (in order) where the narration fits at some size without touching `avoid` (heads, chart)."""
    for box in boxes:
        if box[2] - box[0] < 200 * ctx.s:
            continue
        for size in ctx.narration_sizes:
            nb = narration_fit(text, ctx.language, ctx.fonts, box, max_h, (size,), ctx.s)
            if nb and nb.rect[3] <= ctx.usable[3] and not any(_overlap(nb.rect, r, 6 * ctx.s) for r in avoid):
                return nb
    return None


def _narration_bottom(ctx: _Ctx, text: str, avoid: Sequence[Rect]) -> Optional[NarrationBox]:
    """A full-width narration box at the bottom of the panel (when the top is taken by heads)."""
    ux1, uy1, ux2, uy2 = ctx.usable
    for size in ctx.narration_sizes:
        nb = narration_fit(text, ctx.language, ctx.fonts, (ux1, uy1, ux2, uy2), 0.3 * ctx.H, (size,), ctx.s)
        if nb:
            h = nb.rect[3] - nb.rect[1]
            nb = NarrationBox((ux1, uy2 - h, ux2, uy2), nb.block, nb.pad)
            if not any(_overlap(nb.rect, r, 6 * ctx.s) for r in avoid):
                return nb
    return None


def _below(nb: Optional[NarrationBox], band: Rect, s: float) -> Rect:
    """The band under a narration box at its top (bubbles prefer it)."""
    if nb is None or nb.rect[1] > band[3]:
        return band
    return (band[0], nb.rect[3] + 14 * s, band[2], max(band[3], nb.rect[3] + 140 * s))


BIG_OBJECT = 0.12   # share of the art from which an idea object still reads with its edges under a bubble
GIANT_OBJECT = 0.5  # a box over half the art is the scene itself, not an object the words could leave clear


def _held(rect: Rect, ctx: _Ctx) -> Rect:
    """The part of an idea object the words must leave clear: all of a small one (a padlocked box under a bubble is
    gone), the middle of a big one (a pile or a long row still reads with its edges covered)."""
    x1, y1, x2, y2 = rect
    if (x2 - x1) * (y2 - y1) < BIG_OBJECT * ctx.W * ctx.H:
        return rect
    dx, dy = 0.2 * (x2 - x1), 0.2 * (y2 - y1)
    return (x1 + dx, y1 + dy, x2 - dx, y2 - dy)


def _scene(ctx: _Ctx, bubbles, narration_text: str) -> _Plan:
    """A scene's plan; the objects that carry its idea stay clear of the words (the eight answer cards of a GRPO panel
    sat under a bubble). The idea check lists them most important first. In order: every object wholly clear with
    every word on the art (the narration may move to the caption strip); the middle of the big ones only; then fewer
    objects, dropping the last one first. When no try fits every bubble, the plan with the most bubbles on the art wins,
    the earliest of equals, which keeps the most objects clear; the first object stays clear in every try, and the
    bubbles that do not fit go to the caption strip under the art. The
    plan that ignored the objects put "These yellow erasers are unstable questions" over the erasers and "These grey
    dimmers..." over the dimmers: 95 of 1284 scene layouts of rounds 15-19 covered an idea object more than half, and a
    smallest-first fallback kept a side prop clear over the key one."""
    whole = [o for o in ctx.objects if (o[2] - o[0]) * (o[3] - o[1]) <= GIANT_OBJECT * ctx.W * ctx.H]
    if not whole:
        return _scene_plan(ctx, bubbles, narration_text, [])
    held = [_held(o, ctx) for o in whole]
    tries = [whole] + [held[:k] for k in range(len(held), 0, -1)]
    best = None
    for i, keep in enumerate(tries):
        if i and keep == tries[i - 1]:
            continue
        plan = _scene_plan(ctx, bubbles, narration_text, keep)
        if not plan.leftover:   # every bubble fits beside these objects
            return plan
        if best is None or len(plan.bubbles) > len(best.bubbles):   # objects dropped for no room gained stay clear
            best = plan
    return best


def _scene_plan(ctx: _Ctx, bubbles, narration_text: str, objects: Sequence[Rect]) -> _Plan:
    plan = _Plan(kind="scene")
    heads = list(ctx.heads.values())
    keep = heads + list(objects)
    held = [(o, ctx.bubble_gap) for o in objects]
    band = free_band(ctx.img, heads, edges=ctx.edges, within=ctx.usable)
    nb = None
    if narration_text:
        ux1, uy1, ux2, uy2 = ctx.usable
        wide = band[2] - band[0] >= 0.5 * (ux2 - ux1)
        top = (ux1, uy1, ux2, uy2)
        boxes = [band, top] if wide else [top, band]
        nb = _place_narration(ctx, narration_text, boxes, 0.42 * ctx.H, keep) or _narration_bottom(ctx, narration_text, keep)
        plan.narration_mode = "box" if nb else "strip"
    res = _plan(ctx, bubbles, _below(nb, band, ctx.s), ([(nb.rect, ctx.bubble_gap)] if nb else []) + held)
    if res is None and nb is not None:
        # first fallback: the narration moves to the caption strip, giving its room to the bubbles
        res = _plan(ctx, bubbles, band, held)
        if res is not None:
            nb, plan.narration_mode = None, "strip"
    if res is None:
        size, placed, leftover = _plan_prefix(ctx, bubbles, _below(nb, band, ctx.s), ([(nb.rect, ctx.bubble_gap)] if nb else []) + held)
        if nb is not None:
            size2, placed2, leftover2 = _plan_prefix(ctx, bubbles, band, held)
            if len(placed2) > len(placed):
                nb, plan.narration_mode, size, placed, leftover = None, "strip", size2, placed2, leftover2
        res, plan.leftover = (size, placed), leftover
    plan.narration_box = nb
    plan.size, plan.bubbles = res
    plan.fallback = plan.narration_mode == "strip" or bool(plan.leftover)
    return plan


def _card_rect(ctx: _Ctx) -> Optional[Rect]:
    """Where to pin a white card for the exhibit when the art has no usable board: the largest head-free area."""
    cell = max(4, int(round(8 * ctx.s)))
    ux1, uy1, ux2, uy2 = ctx.usable
    gw, gh = int((ux2 - ux1) // cell), int((uy2 - uy1) // cell)
    if gw < 4 or gh < 4:
        return None
    free = np.ones((gh, gw), bool)
    for x1, y1, x2, y2 in ctx.heads.values():
        g = ctx.head_gap((x1, y1, x2, y2))
        free[max(0, int((y1 - g - uy1) // cell)): max(0, int(math.ceil((y2 + g - uy1) / cell))),
             max(0, int((x1 - g - ux1) // cell)): max(0, int(math.ceil((x2 + g - ux1) / cell)))] = False
    rect = _largest_rect(free, min_w=int(gw * 0.5), min_h=int(gh * 0.2))
    if rect is None:
        return None
    c1, r1, c2, r2 = rect
    x1, y1, x2, y2 = ux1 + c1 * cell, uy1 + r1 * cell, ux1 + c2 * cell, uy1 + r2 * cell
    # a tall free area keeps a band of at least a fifth of the panel above the card for the words
    y1 = max(y1 + (0.2 * ctx.H if y2 - y1 >= 0.65 * ctx.H else 0.0), y2 - 0.45 * ctx.H)
    inset = 10 * ctx.s
    return (x1 + inset, y1 + inset, x2 - inset, y2 - inset)


def _chart_ok(ctx: _Ctx, rect: Optional[Rect]) -> bool:
    return bool(rect) and rect[2] - rect[0] >= 0.4 * ctx.W and rect[3] - rect[1] >= 0.16 * ctx.H


def _shrink_top(rect: Rect, f: float) -> Rect:
    return (rect[0], rect[1] + (rect[3] - rect[1]) * f, rect[2], rect[3])


DIAGRAM_STEP_COST = 10   # bubble font size (px at scale 1) one 12 % step off a diagram's height must buy


def _board(ctx: _Ctx, spec: dict, bubbles, narration_text: str) -> _Plan:
    plan = _Plan(kind="board", spec=spec)
    s, heads = ctx.s, list(ctx.heads.values())
    full, mode = None, "strip"
    if ctx.quad:
        full = clean_rect(ctx.img, ctx.quad, exclude=heads)
        mode = "board" if _chart_ok(ctx, full) else "strip"
    if mode == "strip":
        full = _card_rect(ctx)
        mode = "card" if _chart_ok(ctx, full) else "strip"
        full = full if mode == "card" else None
    plan.chart_mode = mode
    ux1, uy1, ux2, uy2 = ctx.usable
    top = min(y for _, y in ctx.quad) if mode == "board" else (full[1] if full else uy2)
    if top - 12 * s - uy1 >= 110 * s:
        band = (ux1, uy1, ux2, top - 12 * s)
    else:
        band = free_band(ctx.img, heads + ([full] if full else []), edges=ctx.edges, within=ctx.usable)
    nb, first = None, 0
    if narration_text:
        # the band above the board first; when that wall is too low, the chart gives up its top in the same 12 % steps
        # the bubbles may take: the definition in a narration box reads far better than as the board's small grey
        # subtitle, which stays the last resort
        for first, f in enumerate(CHART_SHRINK if full and mode != "strip" else CHART_SHRINK[:1]):
            chart = _shrink_top(full, f) if full else None
            area = band if not first else (ux1, uy1, ux2, chart[1] - ctx.chart_gap)
            limit = (area[3] - area[1]) * (0.55 if bubbles else 1.0)
            nb = _place_narration(ctx, narration_text, [area], limit, heads + ([chart] if chart else []))
            if nb:
                band = area
                break
        if nb:
            plan.narration_mode = "box"
        else:
            first = 0
            if mode != "strip" and not str(spec.get("subtitle") or "").strip():
                plan.spec = dict(spec, subtitle=narration_text)    # the definition moves onto the board
                plan.narration_mode = "subtitle"
            else:
                plan.narration_mode = "strip"
    plan.narration_box = nb
    region = _below(nb, band, s) if nb else band
    base = [(nb.rect, ctx.bubble_gap)] if nb else []
    levels = [_shrink_top(full, f) for f in (CHART_SHRINK[first:] if bubbles else CHART_SHRINK[first:first + 1])] if full else [full]

    def preferred(chart: Optional[Rect]) -> Rect:
        # everything above the chart is good room for words (the band, the board's top margin, the freed part)
        return (region[0], region[1], region[2], max(region[3], chart[1] - ctx.chart_gap)) if chart and mode != "strip" else region

    best = None
    for step, chart in enumerate(levels):
        res = _plan(ctx, bubbles, preferred(chart), base + ([(chart, ctx.chart_gap)] if chart else []))
        if res is None:
            continue
        # one 12 % step off the chart must buy at least three font sizes of bubble text; six on a diagram, whose labels
        # are the panel's smallest text (round 23: a median of 14 px beside 44 px bubbles, and the bubbles took up to
        # a third of the board's clean height in 7 of 16 diagram panels)
        score = res[0] / s - (DIAGRAM_STEP_COST if spec.get("kind") == "diagram" else 5) * step
        if best is None or score > best[0]:
            best = (score, chart, res)
        if res[0] >= ctx.bubble_sizes[0]:
            break
    if best is not None:
        plan.chart, (plan.size, plan.bubbles) = best[1], best[2]
    else:
        # fallback: the bubbles that do not fit go to the caption strip; the shrink keeping the most bubbles wins
        keep = None
        for chart in levels:
            size, placed, leftover = _plan_prefix(ctx, bubbles, preferred(chart), base + ([(chart, ctx.chart_gap)] if chart else []))
            if keep is None or len(placed) > len(keep[2]):
                keep = (chart, size, placed, leftover)
        plan.chart, plan.size, plan.bubbles, plan.leftover = keep
    if full and plan.chart and plan.chart[1] > full[1]:
        # the shrink was planned before the words were placed: give the chart back what they do not use
        for f in CHART_SHRINK:
            chart = _shrink_top(full, f)
            if chart[1] >= plan.chart[1]:
                break
            if not (nb and _overlap(nb.rect, chart, 6 * s)) and not any(_overlap(p.rect, chart, ctx.chart_gap) for p in plan.bubbles):
                plan.chart = chart
                break
    plan.fallback = plan.narration_mode == "strip" or bool(plan.leftover) or mode == "strip"
    return plan


def _display_name(speaker: str) -> str:
    return speaker[:1].upper() + speaker[1:] if speaker.isascii() else speaker


def _strip(ctx: _Ctx, img: Image.Image, items: list[tuple[Optional[str], str]], chart: Optional[tuple[dict, Callable]]):
    """Extend the canvas downward with a cream caption box: narration (speaker None), then the exhibit when it had
    no room in the art, then 'Name: text' lines. Returns the new image and the strip layout."""
    s, W, H = ctx.s, img.width, img.height
    size = ctx.narration_sizes[min(1, len(ctx.narration_sizes) - 1)]
    face = ctx.fonts.face(ctx.language, size, emphasis=True)
    bold = ctx.fonts.face(ctx.language, size, bold=True)
    lh = _line_height(face.size)
    fx1, _, fx2, _ = ctx.frame
    x1, x2 = max(fx1, 8 * s), min(fx2, W - 8 * s)
    pad, gap = 22 * s, 0.45 * lh
    inner = x2 - x1 - 2 * pad
    blocks = []
    for speaker, text in items:
        prefix = "" if not speaker else _display_name(speaker) + ("：" if ctx.lang == "zh" else ": ")
        pw = bold.length(prefix) if prefix else 0.0
        shown = _pretty(text, face)
        lines = _paragraph(shown, ctx.language, face, inner - pw) or _paragraph(shown, ctx.language, face, inner - pw, True) or []
        blocks.append((speaker, text, prefix, pw, lines))
    chart_h = int(round(min(0.62 * inner, 0.45 * H))) if chart else 0
    text_h = sum(len(b[4]) * lh for b in blocks) + gap * max(0, len(blocks) - 1)
    top_gap = 14 * s
    box_h = 2 * pad + text_h + (chart_h + (gap if blocks else 0) if chart else 0) - 6 * s
    total = int(math.ceil(H + top_gap + box_h + 14 * s))
    # continue the art's own paper margin below it (rarely pure white); art that runs to the edge gets white paper
    edge = np.median(np.asarray(img)[max(0, H - 4 * int(s + 1)):H].reshape(-1, 3), axis=0)
    paper = tuple(int(v) for v in edge) if edge.min() > 200 and edge.max() - edge.min() < 30 else PAPER
    out = Image.new("RGB", (W, total), paper)
    out.paste(img, (0, 0))
    d = ImageDraw.Draw(out)
    box = (x1, H + top_gap, x2, H + top_gap + box_h)
    d.rectangle(box, fill=CREAM, outline=INK, width=max(2, int(round(4 * s))))
    y = box[1] + pad - 3 * s
    layout_items, chart_rect = [], None
    order = [b for b in blocks if b[0] is None] + (["chart"] if chart else []) + [b for b in blocks if b[0] is not None]
    for item in order:
        if item == "chart":
            spec, render = chart
            chart_rect = (x1 + pad, y, x2 - pad, y + chart_h)
            _card(out, chart_rect, spec, render, ctx)
            y += chart_h + gap
            continue
        speaker, text, prefix, pw, lines = item
        if prefix:
            bold.draw(d, (x1 + pad, y + bold.line_offset(lh)), prefix, TEXT_INK)
        draw_lines(d, TextBlock(face, lh, lines, 0.0, lh * len(lines)), x1 + pad + pw, y, inner - pw, align="left")
        layout_items.append({"speaker": speaker, "text": text, "lines": lines})
        y += lh * len(lines) + gap
    layout = {"rect": [int(round(v)) for v in box], "font_size": face.size, "items": layout_items,
              "chart": [int(round(v)) for v in chart_rect] if chart_rect else None}
    return out, layout


def _card(img: Image.Image, rect: Rect, spec: dict, render: Callable, ctx: _Ctx) -> None:
    """A white card with an ink border, holding the exhibit (used when the art has no usable board)."""
    s = ctx.s
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([rect[0], rect[1], rect[2], rect[3]], radius=int(14 * s), fill=PAPER, outline=INK,
                        width=max(2, int(round(4 * s))))
    inset = 10 * s
    onto_rect(img, render((rect[0] + inset, rect[1] + inset, rect[2] - inset, rect[3] - inset)),
              (rect[0] + inset, rect[1] + inset, rect[2] - inset, rect[3] - inset))


def _default_board_renderer() -> Callable:
    try:
        from services.boards import render_board   # lazy: boards imports this module for its fonts
    except ImportError as e:
        raise _BoardFailed(f"services.boards is not available: {e}") from e
    return render_board


def _chart_renderer(spec: dict, language: str, fonts: FontSet, board_renderer: Optional[Callable]) -> Callable[[Rect], Image.Image]:
    """rect -> the exhibit drawn at that rect's pixel size; a renderer failure becomes _BoardFailed."""
    renderer = board_renderer or _default_board_renderer()

    def render(rect: Rect) -> Image.Image:
        w, h = int(round(rect[2])) - int(round(rect[0])), int(round(rect[3])) - int(round(rect[1]))
        try:
            image = renderer(spec, language, (w, h), fonts)
        except Exception as e:  # noqa: BLE001 - the panel is lettered without its exhibit, see letter_panel
            raise _BoardFailed(f"board renderer failed: {type(e).__name__}: {e}") from e
        if not isinstance(image, Image.Image):
            raise _BoardFailed(f"board renderer returned {type(image).__name__}, not an image")
        return image

    return render


def _ints(rect: Optional[Rect]) -> Optional[list[int]]:
    return [int(round(v)) for v in rect] if rect else None


def letter_panel(art: Image.Image, panel, grounding: Optional[Grounding], language: str, fonts: Optional[FontSet] = None,
                 board_renderer: Optional[Callable] = None) -> tuple[Image.Image, dict]:
    """Letter one panel: returns (RGB image, layout dict).

    scene: narration box at the top of the calm band (full width), bubbles below it or anywhere else that is free.
    board: the exhibit goes in the clean rectangle of the board (its top shrinks in 12 % steps, at most 3, when the
    words need room); the narration goes in the band above the exhibit when it takes at most 55 % of it, else it
    becomes the board's subtitle. Without a usable board the exhibit is pinned on a white card in the largest
    head-free area.
    When the words still do not fit at the minimum size, a caption strip is appended below the art (fallback).
    board_renderer(spec, language, (w, h), fonts) -> PIL image; defaults to services.boards.render_board. When it
    fails, the panel is lettered as a scene and layout["board_error"] says why (the same spec would fail on every
    retry, so failing the panel would only throw the paid art away). Raises LayoutError only for unusable art.
    """
    if art is None or art.width < 64 or art.height < 64:
        raise LayoutError("art is missing or too small to letter")
    fonts = fonts or default_fonts()
    base = art.convert("RGB")          # analysed once; every attempt draws on a copy
    heads, quad = _scaled_grounding(grounding, base.width, base.height)
    ctx = _Ctx(base, language, fonts, heads, quad, objects=_scaled_boxes(getattr(grounding, "objects", None), grounding, base.width, base.height))
    _, spec, bubbles, narration_text = _panel_parts(panel)
    if not spec:
        return _compose(ctx, _scene(ctx, bubbles, narration_text), art.size, narration_text, board_renderer)
    try:
        return _compose(ctx, _board(ctx, spec, bubbles, narration_text), art.size, narration_text, board_renderer)
    except _BoardFailed as e:
        print(f"[lettering] WARNING: {e}; the panel is lettered without its exhibit")
        img, layout = _compose(ctx, _scene(ctx, bubbles, narration_text), art.size, narration_text, board_renderer)
        layout.update(kind="board", board_error=str(e)[:300])
        return img, layout


def _compose(ctx: _Ctx, plan: _Plan, art_size: tuple[int, int], narration_text: str,
             board_renderer: Optional[Callable]) -> tuple[Image.Image, dict]:
    """Draw a plan onto a copy of the art: exhibit, narration, bubbles, then the caption strip; returns (image, layout)."""
    img, heads, language, fonts = ctx.img.copy(), ctx.heads, ctx.language, ctx.fonts
    render = _chart_renderer(plan.spec, language, fonts, board_renderer) if plan.spec is not None else None
    if render and plan.chart_mode == "board" and plan.chart:
        onto_rect(img, render(plan.chart), plan.chart)
    elif render and plan.chart_mode == "card" and plan.chart:
        _card(img, plan.chart, plan.spec, render, ctx)

    nb = plan.narration_box
    if nb:
        draw_narration(img, nb, ctx.s)
    blocked = list(heads.values()) + ([plan.chart] if plan.chart else []) + ([nb.rect] if nb else [])
    tails = draw_bubbles(img, plan.bubbles, heads, blocked, ctx=ctx)

    strip_items: list[tuple[Optional[str], str]] = []
    if plan.narration_mode == "strip":
        strip_items.append((None, narration_text))
    strip_items += plan.leftover
    strip_layout = None
    chart_in_strip = plan.spec is not None and plan.chart_mode == "strip"
    if strip_items or chart_in_strip:
        img, strip_layout = _strip(ctx, img, strip_items, (plan.spec, render) if chart_in_strip else None)

    narration_layout = None
    if narration_text:
        if nb:
            narration_layout = {"rect": _ints(nb.rect), "font_size": nb.block.face.size, "lines": nb.block.lines, "placement": "box"}
        elif plan.narration_mode == "subtitle":
            narration_layout = {"rect": _ints(plan.chart), "font_size": None, "lines": [], "placement": "subtitle"}
        else:
            first = strip_layout["items"][0] if strip_layout and strip_layout["items"] else {"lines": []}
            narration_layout = {"rect": strip_layout["rect"] if strip_layout else None,
                                "font_size": strip_layout["font_size"] if strip_layout else None,
                                "lines": first["lines"], "placement": "strip"}
    board = plan.kind == "board"
    layout = {
        "version": LAYOUT_VERSION,
        "size": [img.width, img.height],
        "art_size": list(art_size),
        "kind": plan.kind,
        "narration": narration_layout,
        "bubbles": [{"speaker": p.speaker, "text": p.text, "rect": _ints(p.rect), "chip": p.chip,
                     "tail": [[int(round(tl[1][0])), int(round(tl[1][1]))], [int(round(tl[2][0])), int(round(tl[2][1]))]] if tl else None,
                     "font_size": p.block.face.size, "lines": list(p.block.lines)}
                    for p, tl in zip(plan.bubbles, tails)],
        "chart": _ints(plan.chart) if plan.chart_mode in ("board", "card") else (strip_layout or {}).get("chart"),
        "chart_placement": plan.chart_mode if board else None,
        "strip": strip_layout,
        "fallback": bool(plan.fallback),
    }
    return img, layout
