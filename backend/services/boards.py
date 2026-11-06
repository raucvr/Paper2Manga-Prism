"""
Boards — exhibits (bar charts, tables, flows, comparisons, recaps, attention links) drawn by code on the blank
whiteboard of a v3 board panel.

The art model never draws text or numbers. The script LLM writes a board spec; normalize_board() coerces its shape,
board_problems() lists what the script must fix, services/evidence.py checks every number against the paper, and
render_board() typesets the board as a white, marker-style image that lettering multiplies onto the whiteboard.
Values stay the paper's exact strings end to end; only display_value() changes how they look ("10^19" → "10¹⁹").
Pure functions + Pillow; deterministic; no network.
"""

from __future__ import annotations

import copy
import math
import random
import re
import unicodedata
from decimal import ROUND_HALF_UP, Decimal
from functools import lru_cache
from typing import Any, Callable, NamedTuple, Optional, Protocol

from PIL import Image, ImageDraw

from services.fonts import (FT_LOCK, bold_stroke, has_glyph, is_emoji, is_han, load as load_font,
                            missing as missing_glyph, resolve as resolve_fonts)

BOARD_KINDS = ("bar", "table", "flow", "compare", "list", "links", "diagram")

# limits of the contract; board_problems() flags specs outside them, the renderers still cope
LIMITS = {"bars": (2, 7), "columns": (1, 4), "rows": (1, 6), "steps": (2, 7), "side_items": (1, 4), "items": (2, 6),
          "tokens": (2, 14)}
# (words for Latin text, characters for CJK text)
TEXT_LIMITS = {"title": (10, 24), "subtitle": (30, 54), "label": (5, 12), "column": (4, 10), "cell": (6, 14),
               "step": (8, 16), "side_label": (4, 10), "side_item": (10, 22), "item": (14, 30), "token": (2, 6),
               "note": (16, 30), "source": (8, 16), "unit": (4, 10)}

# links boards: each link is drawn by its share of the strongest one (normalize_board scales the weights so that link
# is 1.0): strong (red) from LINK_TIERS[0], medium (blue) from [1], faint (thin grey) from [2], weaker ones not at all.
# services/abstract_quiz.py tells its reader the same tiers; LINK_STRONG / LINK_SHOWN name the first and the last.
LINK_TIERS = (0.5, 0.2, 0.03)
LINK_STRONG, LINK_SHOWN = LINK_TIERS[0], LINK_TIERS[2]

INK = (35, 45, 62)
BLUE = (38, 92, 175)
RED = (208, 58, 46)
GREY = (120, 128, 140)
VALUE_GREY = (92, 100, 114)   # grey bars keep a darker value label: the numbers are the evidence
# subtitles and footnotes: GREY on white is about 4:1 and readers found the small print too faint; this is about 7:1
NOTE_GREY = (74, 82, 96)

CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff66-\uff9f]")


# ----------------------------------------------------------------------------- numbers

_SUP_BASE = "0123456789+-=()niT"
_SUP_CHARS = "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079\u207a\u207b\u207c\u207d\u207e\u207f\u2071\u1d40"
_SUB_BASE = "0123456789+-=()aeoxhklmnpst"
_SUB_CHARS = ("\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089\u208a\u208b\u208c\u208d\u208e"
              "\u2090\u2091\u2092\u2093\u2095\u2096\u2097\u2098\u2099\u209a\u209b\u209c")
_TO_SUP = dict(zip(_SUP_BASE, _SUP_CHARS)) | {"\u2212": "\u207b", "\u2013": "\u207b"}
_TO_SUB = dict(zip(_SUB_BASE, _SUB_CHARS))
_FROM_SUP = dict(zip(_SUP_CHARS, _SUP_BASE))
_FROM_SUB = dict(zip(_SUB_CHARS, _SUB_BASE))
_SUP_RUN = re.compile(f"[{_SUP_CHARS}]+")
_SUB_RUN = re.compile(f"[{_SUB_CHARS}]+")
_DASHES = str.maketrans({c: "-" for c in "\u2010\u2011\u2012\u2013\u2014\u2015\u2212\ufe58\ufe63\uff0d"})

# One number as a paper writes it: 28.4, .05, 1,000 (commas removed first), 2.3 · 10^19, 1.4 × 10^20, 10^19, 5e-5.
# Digits glued to letters are names, not quantities (ConvS2S, v1.1, GPT-2, P100, WMT14); digits after ^ or _ are
# exponents or indices of a formula (n^2, d_k), and so is a power that is not of ten (2^10).
_NUM_RE = re.compile(
    r"(?:(?<![A-Za-z0-9.^_])|(?<=(?<![A-Za-z])[xX]))(?<![A-Za-z]-)"
    r"(?:10\s*\^\s*\{?\s*(?P<p>[-+]?\d+)(?:\s*\})?(?!\d|\.\d)"
    r"|(?P<m>\d+(?:\.\d+)?|\.\d+)"
    r"(?:\s*[x×*·⋅∙•]\s*10\s*\^\s*\{?\s*(?P<e>[-+]?\d+)(?:\s*\})?(?!\d|\.\d)|[eE](?P<e2>[-+]?\d+)(?![A-Za-z0-9])"
    r"|(?P<k>[kKmMbB])(?![A-Za-z0-9])|\s?(?P<w>thousand|million|billion)\b)?"   # 31.0M = 31.0 million = 31.0 · 10^6 (any case:
    # the evidence check compares lower-cased paper text with the board as written)
    r"(?!\d|\.\d|\s*\^\s*\{?\s*[-+]?\d))"   # "55^(a)" is 55 with a footnote mark; "2^10" is no plain 2
)
_MAGNITUDE = {"k": 3, "K": 3, "m": 6, "M": 6, "b": 9, "B": 9, "thousand": 3, "million": 6, "billion": 9}


class Number(NamedTuple):
    text: str                 # as written (after normalization), e.g. "2.3 · 10^19"
    key: str                  # canonical value for comparisons: "26.30" and "26.3" → "26.3"; 2.3 · 10^19 → its digits
    decimals: int             # digits after the decimal point of the mantissa ("26.30" → 2)
    exponent: Optional[int]   # the power of ten when written in scientific notation, else None
    short: bool               # a lone digit ("8"): the text checks ignore it
    start: int
    end: int


def _unscript(text: str) -> str:
    """Superscripts and subscripts back to plain markup (10¹⁹ → 10^19, logₖ → log_k), before NFKC folds them into digits."""
    text = _SUP_RUN.sub(lambda m: "^" + "".join(_FROM_SUP[c] for c in m.group(0)), text)
    return _SUB_RUN.sub(lambda m: "_" + "".join(_FROM_SUB[c] for c in m.group(0)), text)


def _prep(text: str) -> str:
    s = unicodedata.normalize("NFKC", _unscript(str(text))).translate(_DASHES)
    return re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", s)   # 1,000 → 1000


MAX_EXPONENT = 400   # beyond any quantity a paper reports (and beyond a float): 10^99999 is not read as a number


def _canon(mantissa: str, exponent: int) -> Optional[str]:
    if abs(exponent) > MAX_EXPONENT:
        return None
    try:
        d = Decimal(mantissa).scaleb(exponent).normalize()
    except ArithmeticError:
        return None
    return format(d, "f")


def _scan(s: str) -> list[Number]:
    """Numbers in already-prepared text (positions refer to ``s``)."""
    out = []
    for m in _NUM_RE.finditer(s):
        e = next((m.group(k) for k in ("p", "e", "e2") if m.group(k) is not None), None)
        suffix = m.group("k") or m.group("w")
        if e is None and suffix:
            e = str(_MAGNITUDE[suffix])
        if e is not None and len(e.lstrip("+-")) > 6:   # 10^999…9: not a quantity (and too long for int())
            continue
        mant, exp = ("1" if m.group("p") is not None else m.group("m")), (int(e) if e is not None else None)
        key = _canon(mant, exp or 0)
        if key is None:
            continue
        decimals = len(mant.split(".")[1]) if "." in mant else 0
        out.append(Number(m.group(0), key, decimals, exp, exp is None and len(mant) < 2, m.start(), m.end()))
    return out


def find_numbers(text) -> list[Number]:
    """Every quantity written in a text, scientific notation as one number; names with digits are skipped."""
    return _scan(_prep(text))


def _negative(s: str, start: int) -> bool:
    """A minus sign belongs to the number: "-0.9", "change -0.9" (glued after a space or bracket, as display_value
    prints it with a real minus), "= - 0.9"; not the dash of a range ("1-0.9")."""
    i = start - 1
    while i >= 0 and s[i] == " ":
        i -= 1
    if i < 0 or s[i] != "-":
        return False
    j = i - 1
    if i == start - 1 and (j < 0 or not s[j].isalnum()):
        return True
    while j >= 0 and s[j] == " ":
        j -= 1
    return j < 0 or s[j] in "([{=:~≈,;"


def parse_value(value) -> Optional[float]:
    """The number a board value encodes ("28.4", "2.3 · 10^19", "4.9%", "1.4 × 10²⁰", "−0.9"); None unless it holds
    exactly one number."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(value) else None
    s = _prep(value)
    nums = _scan(s)
    if len(nums) != 1:
        return None
    v = float(Decimal(nums[0].key))
    if not math.isfinite(v):
        return None
    return -v if _negative(s, nums[0].start) else v


def _sup(text: str) -> str:
    text = text.strip()
    text = text[1:] if text.startswith("+") else text
    return "".join(_TO_SUP.get(c, c) for c in text)


_SCI_MUL_RE = re.compile(r"(\d)\s*[x×*·⋅∙•]\s*(?=10\s*\^)")
_E_NOTATION_RE = re.compile(r"(?<![A-Za-z0-9.])(\d+(?:\.\d+)?)[eE]([-+]?\d+)(?![A-Za-z0-9])")
_POWER_RE = re.compile(r"\^\s*\{?\s*([-+\u2212]?\d+)(?:\s*\})?(?!\d|\.\d)")
_LETTER_POWER_RE = re.compile(r"\^\s*\{?\s*([niT])(?:\s*\})?(?![A-Za-z0-9])")   # x^n, QK^T
_SQRT_RE = re.compile(r"\bsqrt\s*(?=\()")
_INDEX_RE = re.compile(r"(?<=[A-Za-z)])_\{?([0-9a-z]{1,3})\}?(?![A-Za-z0-9])")


def display_value(value) -> str:
    """How a value or any board text is printed: powers as superscripts ("10^19" → "10¹⁹", "1.4 · 10^20" →
    "1.4 × 10²⁰", "n^2" → "n²", "QK^T" → "QKᵀ"), e-notation spelled out, simple indices as subscripts (log_k → logₖ),
    sqrt( as √(, a leading minus as a real minus sign. The digits never change."""
    s = _num_str(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else str(value or "").strip()
    s = _unscript(s)
    s = _SCI_MUL_RE.sub(r"\1 × ", s)
    s = _E_NOTATION_RE.sub(lambda m: f"{m.group(1)} × 10^{int(m.group(2))}" if len(m.group(2)) <= 6 else m.group(0), s)
    s = _POWER_RE.sub(lambda m: _sup(m.group(1).replace("\u2212", "-")), s)
    s = _LETTER_POWER_RE.sub(lambda m: _TO_SUP[m.group(1)], s)
    s = _SQRT_RE.sub("√", s)
    s = _INDEX_RE.sub(lambda m: "".join(_TO_SUB[c] for c in m.group(1)) if all(c in _TO_SUB for c in m.group(1)) else m.group(0), s)
    return re.sub(r"(^|[\s(\[])-(?=\.?\d)", "\\1\u2212", s)


def _num_str(x) -> str:
    """A number the model sent as JSON, written the way papers write it (2.3e19 → "2.3 · 10^19")."""
    if isinstance(x, bool):
        return ""
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if not math.isfinite(x):
            return ""
        r = repr(x)
        if "e" in r:
            mant, exp = r.split("e")
            return f"{mant} · 10^{int(exp)}"
        return r
    return str(x)


class _Parts(NamedTuple):
    value: Decimal
    decimals: int
    exponent: Optional[int]
    unit: str


def _value_parts(value) -> Optional[_Parts]:
    s = _prep(value)
    nums = _scan(s)
    if len(nums) != 1:
        return None
    n = nums[0]
    v = Decimal(n.key) * (-1 if _negative(s, n.start) else 1)
    unit = re.sub(r"[\s+\-~≈≥≤<>]+", " ", s[:n.start] + " " + s[n.end:]).strip().lower()
    return _Parts(v, n.decimals, n.exponent, unit)


def _fmt_decimal(d: Decimal, decimals: int) -> str:
    return format(d.quantize(Decimal(1).scaleb(-decimals)), "f")


RATIO_FROM = 3   # a lower-is-better gap of this factor or more (or one between powers of ten) reads better as a ratio
# a percentage or a share in the unit or the title, in both languages ("% of total attention", "总注意力百分比", "百分点", "占比",
# "share", "probability"), never a percentile: those values fall in points, not by a factor
_SHARE_UNIT_RE = re.compile(r"[%％]|\bpercent(?:age)?\b|\bpp\b|\bshare\b|\bprobabilit|百分|占比|比例|概率", re.I)


def _sig2(d: Decimal) -> str:
    """Two significant figures, without trailing zeros: 5.217 -> "5.2", 52.17 -> "52", 523 -> "520", 0.5 -> "0.5"."""
    exp = d.adjusted()
    q = d.scaleb(1 - exp).quantize(Decimal(1), rounding=ROUND_HALF_UP).scaleb(exp - 1)
    return str(int(q)) if q == q.to_integral_value() else format(q.normalize(), "f")


# a gap stated as a multiple in a bar's quote ("52×", "50x", "3 times", "10-fold", "3倍"); ASCII bounds, so the number of a
# Chinese text glued to its characters reads the same
_MULTIPLE_RE = re.compile(r"(?<![A-Za-z0-9_.])(\d+(?:\.\d+)?)\s*(?:[×xX](?![A-Za-z0-9_])|times\b|-?fold\b|倍)")


def _stated_multiple(b: dict, best: int, rival: int, ratio: Decimal) -> bool:
    """The paper states this gap as a multiple in the quote of either bar: rl_1000_layers quoted "12.6 ... 649 ... 52×" and
    its words said "52 times" while the badge drew "+636.4". Only the quotes count: they stay verbatim in every language,
    so the Chinese chart draws the badge the English one draws (titles and notes are translated)."""
    said = " ".join(str(b["bars"][i].get("quote") or "") for i in (best, rival))
    return any(Decimal(m) >= 2 and abs(Decimal(m) - ratio) <= ratio / 20 for m in _MULTIPLE_RE.findall(said))   # never a "1.00×" reference


def delta_badge(spec) -> Optional[dict]:
    """The gap a bar chart shows, computed here and never by the model: the best highlighted bar (the paper's) against the
    bar the script marks as the baseline ("baseline": true, the strongest earlier system the paper compares with). Without
    a marked baseline, or with more than one highlighted bar, there is no badge: the best other bar may belong to another
    setting (a semi-supervised result against a supervised one), and a computed gap would then be a claim the paper never
    makes.
    A difference ("+2.04", the larger number of decimals of the two values) or, between powers of ten, for a lower-is-
    better gap of RATIO_FROM times or more, or for a gap a bar's quote states as a multiple ("52×"), a
    ratio ("≈1/5.2" when lower is better, "5.2×" when higher is better); never between percentages, a share falls in
    points ("dropped from 46.7% to 4.8%" beside "≈1/9.7" read as a second claim).
    {"text", "best": i, "rival": j, "better": bool, "kind": "diff" | "ratio"}, or None when there is no such pair, a value
    does not parse, the two values carry different units or notations, or they are equal."""
    b = normalize_board(spec)
    if not b or b["kind"] != "bar":
        return None
    parts = [_value_parts(x["value"]) for x in b["bars"]]
    hl = [i for i, x in enumerate(b["bars"]) if x.get("highlight")]
    rivals = [i for i, x in enumerate(b["bars"]) if x.get("baseline") and not x.get("highlight")]
    if len(hl) != 1 or not rivals or any(parts[i] is None for i in hl + rivals[:1]):   # one result against one baseline
        return None
    higher = b.get("higher_is_better", True)
    best = (max if higher else min)(hl, key=lambda i: parts[i].value)
    rival = rivals[0]
    a, o = parts[best], parts[rival]
    if a.unit != o.unit or (a.exponent is None) != (o.exponent is None):
        return None
    d = a.value - o.value
    if d == 0:
        return None
    better = (d > 0) == higher
    if better and a.value > 0 and o.value > 0:   # costs, times and powers of ten compare as ratios; scores as differences
        ratio = max(a.value, o.value) / min(a.value, o.value)
        share = "%" in a.unit or bool(_SHARE_UNIT_RE.search(f"{b.get('unit') or ''} {b.get('title') or ''}"))
        if a.exponent is not None or (not share and ((not higher and ratio >= RATIO_FROM) or _stated_multiple(b, best, rival, ratio))):
            n = _sig2(ratio) if ratio < 10 else format(ratio.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP).normalize(), "f")
            exact = Decimal(n) == ratio
            text = (f"{'' if exact else '≈'}1/{n}" if not higher else f"{'' if exact else '≈'}{n}×")
            return {"text": text, "best": best, "rival": rival, "better": True, "kind": "ratio"}
    decimals, mag = max(a.decimals, o.decimals), abs(d)
    if a.exponent is None:
        text = _fmt_decimal(mag, decimals)
    else:
        exp = mag.adjusted()
        mant = mag.scaleb(-exp).quantize(Decimal(1).scaleb(-decimals))
        if mant >= 10:
            mant, exp = (mant / 10).quantize(Decimal(1).scaleb(-decimals)), exp + 1
        text = f"{_fmt_decimal(mant, decimals)} × 10{_sup(str(exp))}"
    return {"text": ("+" if d > 0 else "\u2212") + text, "best": best, "rival": rival, "better": better, "kind": "diff"}


# ----------------------------------------------------------------------------- normalize

_KIND_ALIASES = {
    "bar": "bar", "bars": "bar", "bar_chart": "bar", "barchart": "bar", "chart": "bar", "leaderboard": "bar",
    "histogram": "bar", "column_chart": "bar",
    "table": "table", "grid": "table", "matrix": "table",
    "flow": "flow", "flowchart": "flow", "flow_chart": "flow", "pipeline": "flow", "steps": "flow", "process": "flow",
    "architecture": "flow", "diagram": "diagram", "mechanism": "diagram", "schematic": "diagram",
    "compare": "compare", "comparison": "compare", "versus": "compare", "vs": "compare", "contrast": "compare",
    "before_after": "compare",
    "list": "list", "recap": "list", "summary": "list", "checklist": "list", "bullets": "list",
    "links": "links", "link": "links", "attention": "links", "arcs": "links",
}
_LOWER_BETTER_RE = re.compile(r"\b(?:lower|smaller|less|fewer)\s+is\s+better\b|越(?:低|小|少)越好", re.I)
# a board that calls lower worse: econ_ai_service's call-duration chart said "lower = less effort (backfire)" over a printed
# "lower is better" in two rounds
_LOWER_WORSE_RE = re.compile(r"\b(?:lower|shorter|fewer|less)\s*(?:is|=|means)\s*(?:worse|bad)\b|\bless effort\s*\(?(?:backfire|counterproductive)|"
                             r"越(?:低|短|少)(?:越差|[^，,；;。]{0,8}适得其反)", re.I)
_DIRECTION_RE = re.compile(r"\b(?:higher|lower|smaller|larger|bigger|less|more|fewer)\s+is\s+better\b|越(?:高|低|小|大|少|多)越好", re.I)
# significance stars after a number, or in a legend ("0.234*** (0.033)", "*** p<0.01", "*** <1%", "***: p<0.01",
# "p<0.01 (***)", "*** significant at the 1% level"), are the paper's notation, not markdown bold (econ_ai_service's
# legend "* <10%, *** <1%" was flagged as markdown in every round and left on the board)
SIG_STARS_RE = re.compile(r"(?<=[\d)\]^])\*{1,3}|(?:^|(?<=[\s,;(]))\*{1,3}(?=\s*(?:p\s*[<≤=]|[<≤=:]|\)|significan))", re.I)
_BULLET_RE = re.compile(r"^\s*(?:(?:[✅✓✔☑✗✘❌•·▪►*]|[\-–—](?!\.?\d))+\s*|(?:step\s*)?\d{1,2}\s*[.):：]\s+)", re.I)
_SPLIT_STEPS_RE = re.compile(r"\s*(?:→|->|=>|⇒|\n)\s*")


def _s(v) -> str:
    """Board text from whatever the model sent: numbers as written, lists joined, whitespace collapsed."""
    if v is None or isinstance(v, bool):
        return ""
    if isinstance(v, (int, float)):
        return _num_str(v)
    if isinstance(v, dict):
        for k in ("text", "label", "title", "name", "value"):
            if v.get(k) is not None:
                return _s(v[k])
        return ""
    if isinstance(v, (list, tuple)):
        return " ".join(t for t in (_s(x) for x in v) if t)
    return re.sub(r"\s+", " ", str(v)).strip()


def _bool(v, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return v != 0
    s = str(v).strip().lower()
    if s in ("true", "yes", "y", "1", "on", "ours", "highlight", "highlighted"):
        return True
    if s in ("false", "no", "n", "0", "off", "none"):
        return False
    return default


def _int(v) -> Optional[int]:
    if isinstance(v, bool):
        return None
    try:
        f = float(str(v).strip().lstrip("#"))
    except (TypeError, ValueError):
        return None
    return int(f) if f.is_integer() else None


def _list(v) -> list:
    if v is None:
        return []
    return list(v) if isinstance(v, (list, tuple)) else [v]


def _first(d: dict, *keys):
    for k in keys:
        if d.get(k) is not None:
            return d[k]
    return None


def _strip_bullet(text: str) -> str:
    return _BULLET_RE.sub("", text, count=1).strip() or text


def _indexes(value, labels: list[str]) -> list[int]:
    """Highlight markers as indexes: ints, numeric strings or the labels themselves."""
    lowered = [x.lower() for x in labels]
    out = []
    for v in _list(value):
        i = _int(v)
        if i is None and isinstance(v, str) and v.strip().lower() in lowered:
            i = lowered.index(v.strip().lower())
        if i is not None and 0 <= i < len(labels) and i not in out:
            out.append(i)
    return sorted(out)


def _kind(spec: dict) -> str:
    raw = _s(_first(spec, "kind", "type", "board_type", "chart"))
    kind = _KIND_ALIASES.get(re.sub(r"[\s\-]+", "_", raw.lower()), "")
    if kind == "diagram" and "elements" not in spec and "brief" not in spec and ("steps" in spec or "nodes" in spec):
        return "flow"   # before diagrams were drawn, "diagram" named a flow of steps (scripts saved then still read)
    if kind or raw.lower() not in ("", "board", "exhibit", "figure", "whiteboard", "visual"):
        return kind
    for key, guess in (("bars", "bar"), ("rows", "table"), ("columns", "table"), ("steps", "flow"), ("left", "compare"),
                       ("tokens", "links"), ("items", "list")):
        if key in spec:
            return guess
    return ""


def _normalize(spec) -> tuple[Optional[dict], list[str]]:
    """(normalized spec or None, what had to be dropped or padded on the way)."""
    if not isinstance(spec, dict):
        return None, ["the board spec is not an object"]
    kind = _kind(spec)
    if kind not in BOARD_KINDS:
        raw = _s(_first(spec, "kind", "type"))
        return None, [f"unknown board kind {raw!r}; use one of {', '.join(BOARD_KINDS)}"]
    out: dict = {"kind": kind, "title": _s(_first(spec, "title", "heading", "caption"))}
    if _s(spec.get("subtitle")):
        out["subtitle"] = _s(spec["subtitle"])
    notes: list[str] = []
    body = {"bar": _norm_bar, "table": _norm_table, "flow": _norm_flow, "compare": _norm_compare, "list": _norm_list,
            "links": _norm_links, "diagram": _norm_diagram}[kind](spec, out, notes)
    if body is None:
        return None, notes
    focus_from_source = kind == "links" and _first(spec, "from", "source_index", "focus", "query") is None
    for k in ("note", "source"):
        if _s(spec.get(k)) and not (k == "source" and focus_from_source):
            out[k] = _s(spec[k])
    if isinstance(spec.get("unbound"), list) and spec["unbound"]:
        out["unbound"] = [_s(x) for x in spec["unbound"]]   # ledger citations that named no cell (lint reports them)
    return out, notes


def _norm_bar(spec: dict, out: dict, notes: list) -> Optional[dict]:
    raw = spec.get("bars")
    if raw is None and isinstance(spec.get("labels"), list) and isinstance(spec.get("values"), list):
        raw = [{"label": lab, "value": v} for lab, v in zip(spec["labels"], spec["values"])]
    if raw is None:
        raw = _first(spec, "data", "values", "items")
    if isinstance(raw, dict):
        raw = [{"label": k, "value": v} for k, v in raw.items()]
    bars = []
    for i, x in enumerate(_list(raw)):
        if isinstance(x, (list, tuple)) and len(x) >= 2:
            x = {"label": x[0], "value": x[1]}
        if not isinstance(x, dict):
            notes.append(f"bar {i + 1} is not an object with a label and a value and was dropped")
            continue
        label = _s(_first(x, "label", "name", "model", "x", "category", "paper_label"))
        value = _s(_first(x, "value", "y", "score", "v"))
        if parse_value(value) is None:
            notes.append(f'bar "{label or i + 1}": value {value!r} is not a single number and was dropped')
            continue
        bar = {"label": label, "value": value, "highlight": _bool(_first(x, "highlight", "highlighted", "ours"))}
        if _bool(_first(x, "baseline", "rival")):
            bar["baseline"] = True
        if _s(x.get("quote")):
            bar["quote"] = _s(x["quote"])
        if _s(x.get("paper_label")):
            bar["paper_label"] = _s(x["paper_label"])
        if _s(x.get("cell")):
            bar["cell"] = _s(x["cell"])   # the ledger cell the value was copied from (services.ledger)
        bars.append(bar)
    if not bars:
        notes.append("the bar chart has no bar with a number value")
        return None
    for i in _indexes(spec.get("highlight"), [b["label"] for b in bars]):
        bars[i]["highlight"] = True
    for i in _indexes(spec.get("baseline"), [b["label"] for b in bars]) if not isinstance(spec.get("baseline"), bool) else []:
        bars[i]["baseline"] = True
    out["bars"] = bars
    if _s(spec.get("unit")):
        out["unit"] = _s(spec["unit"])
    words = " ".join([out["title"], out.get("subtitle", ""), out.get("unit", "")])
    if spec.get("higher_is_better") is not None:
        out["higher_is_better"] = _bool(spec["higher_is_better"], True)
    elif spec.get("lower_is_better") is not None:
        out["higher_is_better"] = not _bool(spec["lower_is_better"], False)
    else:
        out["higher_is_better"] = not _LOWER_BETTER_RE.search(words)
    return out


def _norm_table(spec: dict, out: dict, notes: list) -> Optional[dict]:
    columns = [_s(c) for c in _list(_first(spec, "columns", "headers", "header", "cols"))]
    raw_rows = _list(_first(spec, "rows", "data"))
    if not columns and raw_rows and isinstance(raw_rows[0], dict) and "cells" not in raw_rows[0]:
        columns = [str(k) for k in raw_rows[0] if k not in ("quote", "highlight")]
    rows, marked = [], []
    for i, r in enumerate(raw_rows):
        quote = ""
        if isinstance(r, dict):
            quote = _s(r.get("quote"))
            if _bool(r.get("highlight")):
                marked.append(len(rows))
            cells = _first(r, "cells", "values", "row")
            if cells is None:   # a row keyed by column name: "model" or "Model " still names the column "Model"
                keyed = {re.sub(r"\s+", " ", str(k)).strip().lower(): v for k, v in r.items()}
                cells = [keyed.get(c.strip().lower(), "") for c in columns]
        else:
            cells = r
        cells = [_s(c) for c in _list(cells)]
        if not any(cells):
            notes.append(f"table row {i + 1} is empty and was dropped")
            continue
        row = {"cells": cells}
        if quote:
            row["quote"] = quote
        if isinstance(r, dict) and isinstance(r.get("cell_ids"), list):
            row["cell_ids"] = [_s(x) or None for x in r["cell_ids"]][:len(cells)]
        rows.append(row)
    if not rows:
        notes.append("the table has no rows")
        return None
    width = max([len(columns)] + [len(r["cells"]) for r in rows])
    if columns and len(columns) < width:
        notes.append(f"the table has {len(columns)} column headers but rows with {width} cells")
    for i, r in enumerate(rows):
        if columns and len(r["cells"]) < width:
            notes.append(f"table row {i + 1} has {len(r['cells'])} cells for {width} columns")
        r["cells"] += [""] * (width - len(r["cells"]))
        if "cell_ids" in r:
            r["cell_ids"] += [None] * (width - len(r["cell_ids"]))
    out["columns"] = columns + [""] * (width - len(columns))
    out["rows"] = rows
    out["highlight_rows"] = sorted(set(_indexes(_first(spec, "highlight_rows", "highlight", "highlights"),
                                                [r["cells"][0] for r in rows]) + marked))
    return out


def _norm_flow(spec: dict, out: dict, notes: list) -> Optional[dict]:
    raw = _first(spec, "steps", "nodes", "boxes", "stages", "items")
    if isinstance(raw, str):
        raw = _SPLIT_STEPS_RE.split(raw)
    steps = [_strip_bullet(_s(x)) for x in _list(raw)]
    steps = [x for x in steps if x]
    if not steps:
        notes.append("the flow has no steps")
        return None
    out["steps"] = steps
    out["highlight"] = _indexes(spec.get("highlight"), steps)
    return out


def _norm_side(value) -> dict:
    if isinstance(value, dict):
        items = _first(value, "items", "points", "bullets", "lines")
        label = _s(_first(value, "label", "title", "name", "heading"))
    else:
        items, label = value, ""
    if isinstance(items, str):
        items = items.split("\n")
    return {"label": label, "items": [x for x in (_strip_bullet(_s(i)) for i in _list(items)) if x]}


def _norm_compare(spec: dict, out: dict, notes: list) -> Optional[dict]:
    left = _norm_side(_first(spec, "left", "before", "old", "baseline", "a"))
    right = _norm_side(_first(spec, "right", "after", "new", "ours", "b"))
    for name, side in (("left", left), ("right", right)):
        if not side["items"]:
            notes.append(f"the comparison has no {name} items")
    if not left["items"] or not right["items"]:
        return None
    out["left"], out["right"] = left, right
    return out


def _norm_list(spec: dict, out: dict, notes: list) -> Optional[dict]:
    raw = _first(spec, "items", "points", "lines", "bullets")
    if isinstance(raw, str):
        raw = raw.split("\n")
    items = [x for x in (_strip_bullet(_s(i)) for i in _list(raw)) if x]
    if not items:
        notes.append("the list has no items")
        return None
    out["items"] = items
    if _bool(spec.get("open")):
        out["open"] = True
    elif spec.get("checked") is not None:   # a recap ticks only the lines the story showed evidence for; the others get a bullet
        out["checked"] = sorted(set(_indexes(spec.get("checked"), items)))
    return out


def _weight(v) -> Optional[float]:
    if isinstance(v, bool):
        return None
    try:
        f = float(str(v).strip().rstrip("%"))
    except (TypeError, ValueError):
        return None
    return max(0.0, f) if math.isfinite(f) else None


def _norm_links(spec: dict, out: dict, notes: list) -> Optional[dict]:
    spec = {**spec, "illustrative": True}   # always tagged: the weights are drawn by the writer, never read from the paper
    raw = _first(spec, "tokens", "words")
    tokens = [_s(t) for t in (raw.split() if isinstance(raw, str) else _list(raw))]
    tokens = [t for t in tokens if t]
    if not tokens:
        notes.append("the links board has no tokens")
        return None
    src = _first(spec, "from", "source_index", "focus", "query")
    if src is None and spec.get("source") is not None:
        src = spec["source"]
    i = _int(src)
    if i is None and isinstance(src, str):
        lowered = [t.lower() for t in tokens]
        i = lowered.index(src.strip().lower()) if src.strip().lower() in lowered else None
    if i is None or not 0 <= i < len(tokens):
        notes.append("the links board needs 'from': the index of the token whose links are drawn")
        return None
    weights = [_weight(w) for w in _list(_first(spec, "weights", "attention", "scores"))]
    weights = [w if w is not None else 0.0 for w in weights]
    if len(weights) != len(tokens):
        if weights:
            notes.append(f"the links board has {len(weights)} weights for {len(tokens)} tokens")
        weights = (weights + [0.0] * len(tokens))[: len(tokens)]
    top = max((w for k, w in enumerate(weights) if k != i), default=0.0)
    if top <= 0:
        notes.append("the links board needs a positive weight for at least one other token")
        return None
    # percentages, raw scores or probabilities: one scale, the strongest link 1.0, as the board draws them
    weights = [round(w / top, 4) for w in weights]
    out.update(tokens=tokens, **{"from": i}, weights=weights, illustrative=_bool(spec.get("illustrative"), True))
    return out


def _norm_diagram(spec: dict, out: dict, notes: list) -> Optional[dict]:
    from services import diagram   # imports this module
    return diagram.normalize(spec, out, notes)


def normalize_board(spec) -> Optional[dict]:
    """The canonical spec of a board the script LLM wrote (kind from 'type', numbers as strings, rows as {cells},
    highlight flags as booleans, bullets stripped), or None when nothing usable is left. Idempotent."""
    return _normalize(spec)[0]


# ----------------------------------------------------------------------------- problems

def _measure(text: str) -> tuple[int, int]:
    """(Latin words, CJK characters + Latin words)."""
    cjk = len(CJK_RE.findall(text))
    words = len(re.findall(r"[A-Za-z0-9][^\s\u3000-\u9fff]*", text))
    return len(text.split()), cjk + words


def _too_long(text: str, kind: str) -> Optional[str]:
    words, cjk = _measure(text)
    max_words, max_chars = TEXT_LIMITS[kind]
    if CJK_RE.search(text):
        return f"{cjk} characters, keep it within {max_chars}" if cjk > max_chars else None
    return f"{words} words, keep it within {max_words}" if words > max_words else None


def _count(n: int, key: str, what: str) -> Optional[str]:
    lo, hi = LIMITS[key]
    if n < lo:
        return f"{what}: {n} (use at least {lo})"
    if n > hi:
        return f"{what}: {n} (use at most {hi})"
    return None


def board_problems(spec) -> list[str]:
    """What the script must fix in a board (shape, counts, text lengths, highlights, units); [] when it is fine.
    Numbers are checked against the paper separately (evidence.verify_board)."""
    board, notes = _normalize(spec)
    if board is None:
        return notes or ["the board spec is unusable"]
    out = list(notes)

    def text(value: str, kind: str, where: str):
        if value and (msg := _too_long(value, kind)):
            out.append(f'{where} "{value}" is too long ({msg})')
        if value and re.search(r"\$[^$]*\$|\\[a-zA-Z]|\*\*", SIG_STARS_RE.sub("", value)):
            out.append(f'{where} "{value}" contains LaTeX or markdown; write plain text (10^19, O(n^2) are fine)')

    def count(n: int, key: str, what: str):
        if msg := _count(n, key, what):
            out.append(msg)

    if not board["title"]:
        out.append("the board has no title")
    text(board["title"], "title", "title")
    text(board.get("subtitle", ""), "subtitle", "subtitle")
    text(board.get("note", ""), "note", "note")
    text(board.get("source", ""), "source", "source")
    kind = board["kind"]
    if kind == "bar":
        bars = board["bars"]
        count(len(bars), "bars", "bars")
        text(board.get("unit", ""), "unit", "unit")
        for i, b in enumerate(bars):
            if not b["label"]:
                out.append(f"bar {i + 1} has no label")
            text(b["label"], "label", f"bar {i + 1} label")
        if not any(b["highlight"] for b in bars):
            out.append("no bar is highlighted: set highlight true on the paper's own result")
        parts = [_value_parts(b["value"]) for b in bars]
        if len({p.unit for p in parts if p}) > 1:
            out.append("the bars mix units (" + ", ".join(repr(b["value"]) for b in bars) + "); compare one metric in one unit")
        if len({p.exponent is None for p in parts if p}) > 1:
            out.append("the bars mix plain numbers and powers of ten; write every value exactly as the paper does")
        said = " ".join([board["title"], board.get("subtitle", ""), board.get("unit", "")])
        if board["higher_is_better"] and _LOWER_BETTER_RE.search(said):
            out.append("the title says lower is better but higher_is_better is true")
        elif not board["higher_is_better"] and _LOWER_WORSE_RE.search(said) and not _LOWER_BETTER_RE.search(said):
            out.append('the board calls lower worse ("lower = less effort (backfire)") but higher_is_better is false, so the chart '
                       'prints "lower is better": set higher_is_better to true')
    elif kind == "table":
        count(len(board["columns"]), "columns", "columns")
        count(len(board["rows"]), "rows", "rows")
        for j, col in enumerate(board["columns"]):
            text(col, "column", f"column {j + 1}")
        for i, r in enumerate(board["rows"]):
            for j, cell in enumerate(r["cells"]):
                text(cell, "cell", f"row {i + 1} cell {j + 1}")
        raw = _list(spec.get("highlight_rows")) if isinstance(spec, dict) else []
        if any((i := _int(v)) is not None and not 0 <= i < len(board["rows"]) for v in raw):
            out.append("highlight_rows has indexes outside the table (rows count from 0)")
    elif kind == "flow":
        count(len(board["steps"]), "steps", "steps")
        for i, step in enumerate(board["steps"]):
            text(step, "step", f"step {i + 1}")
    elif kind == "compare":
        for name in ("left", "right"):
            side = board[name]
            count(len(side["items"]), "side_items", f"{name} items")
            if not side["label"]:
                out.append(f"the {name} side has no label")
            text(side["label"], "side_label", f"{name} label")
            for i, it in enumerate(side["items"]):
                text(it, "side_item", f"{name} item {i + 1}")
    elif kind == "list":
        count(len(board["items"]), "items", "items")
        for i, it in enumerate(board["items"]):
            text(it, "item", f"item {i + 1}")
    elif kind == "links":
        count(len(board["tokens"]), "tokens", "tokens")
        for i, t in enumerate(board["tokens"]):
            text(t, "token", f"token {i + 1}")
    # a diagram's layout is the designer's to fix (diagram_problems, in StoryboarderV2.design_diagrams), never the writer's
    return out


def diagram_problems(board: dict) -> list[str]:
    """The layout lint of a diagram board (services.diagram.problems), measured with the default board fonts in the
    language of its texts; [] while its elements are not designed."""
    from services import diagram
    lang = "zh" if any(CJK_RE.search(t) for _, t in diagram.texts(board)) else "en"
    return diagram.problems(board, _Pen(default_fonts(), lang), _wrap_lines)


def _wrap_lines(pen, text: str, max_w: float, size: float, bold: bool = False) -> list[str]:
    return _lines(pen, text, max_w, size, bold, True)


# ----------------------------------------------------------------------------- strings and numbers

_FORMULA_WORD_RE = re.compile(r"\b(?:log|ln|exp|sqrt|max|min|softmax)(?:_?[a-z])?(?=\s*\()", re.I)


def _translatable(text: str) -> bool:
    """Words a reader reads (not a number, a formula such as O(n^2 · d) or O(log_k(n)), or a bare symbol)."""
    s = _FORMULA_WORD_RE.sub(" ", _unscript(text))
    return bool(CJK_RE.search(s) or re.search(r"[A-Za-z]{2,}", s))


def board_texts(spec) -> list[tuple[str, str]]:
    """Every text a reader sees on a board as (path, text) in reading order, values and numbers included: title,
    subtitle, unit, bar labels and values, columns and cells, steps, compare sides, items, tokens, note, source. Paths
    address the normalized spec ("bars.2.value", "rows.1.cells.0", "left.items.3"); quotes and paper labels are never
    drawn, so never listed."""
    b = normalize_board(spec)
    if b is None:
        return []
    out: list[tuple[str, str]] = []

    def add(path: str, value: str):
        if value:
            out.append((path, value))

    for k in ("title", "subtitle", "unit"):
        add(k, b.get(k, ""))
    kind = b["kind"]
    if kind == "bar":
        for i, x in enumerate(b["bars"]):
            add(f"bars.{i}.label", x["label"])
            add(f"bars.{i}.value", x["value"])
    elif kind == "table":
        for j, col in enumerate(b["columns"]):
            add(f"columns.{j}", col)
        for i, r in enumerate(b["rows"]):
            for j, cell in enumerate(r["cells"]):
                add(f"rows.{i}.cells.{j}", cell)
    elif kind == "flow":
        for i, x in enumerate(b["steps"]):
            add(f"steps.{i}", x)
    elif kind == "compare":
        for side in ("left", "right"):
            add(f"{side}.label", b[side]["label"])
            for i, x in enumerate(b[side]["items"]):
                add(f"{side}.items.{i}", x)
    elif kind == "list":
        for i, x in enumerate(b["items"]):
            add(f"items.{i}", x)
    elif kind == "links":
        for i, x in enumerate(b["tokens"]):
            add(f"tokens.{i}", x)
    elif kind == "diagram":
        from services import diagram
        for path, x in diagram.texts(b):
            add(path, x)
    for k in ("note", "source"):
        add(k, b.get(k, ""))
    return out


def board_strings(spec) -> list[tuple[str, str]]:
    """Every translatable text of a board as (path, text) in reading order: title, subtitle, unit, labels, columns,
    word cells, steps, items, compare sides, note, source. Values, pure numbers, formulas and the tokens of a links
    board (the running example's own words) are never included. Paths address the normalized spec."""
    return [(path, text) for path, text in board_texts(spec)
            if not path.endswith(".value") and not path.startswith("tokens.") and _translatable(text)]


def _same_numbers(a: str, b: str) -> bool:
    return sorted(n.key for n in find_numbers(a)) == sorted(n.key for n in find_numbers(b))


def _set_path(obj, path: str, value: str) -> None:
    keys = path.split(".")
    for k in keys[:-1]:
        obj = obj[int(k)] if isinstance(obj, list) else obj[k]
    last = keys[-1]
    if isinstance(obj, list):
        obj[int(last)] = value
    else:
        obj[last] = value


def set_board_strings(spec, texts: dict) -> Optional[dict]:
    """A copy of the normalized board with the texts at the given board_strings() paths replaced (translation).
    Unknown paths, empty texts and replacements that change, add or drop a number are ignored."""
    b = normalize_board(spec)
    if b is None:
        return None
    current = dict(board_strings(b))
    out = copy.deepcopy(b)
    for path, value in (texts or {}).items():
        if path not in current or not isinstance(value, str) or not value.strip():
            continue
        value = re.sub(r"\s+", " ", value).strip()
        if _same_numbers(current[path], value):
            _set_path(out, path, value)
    return out


def _row_label(cells: list[str]) -> str:
    """The words of a table row with its numbers removed ("Transformer (big)", "28.4", "3.5 days" → "Transformer (big) days")."""
    parts = []
    for cell in cells:
        s = _prep(cell)
        for n in reversed(_scan(s)):
            s = s[: n.start] + " " + s[n.end:]
        s = re.sub(r"\s+", " ", s).strip()
        if _translatable(s):
            parts.append(s)
    return " ".join(parts)


def board_numbers(spec) -> list[tuple[str, str, str, str]]:
    """Every data value on a bar or table board as (path, value, quote, label) for evidence checks: bar values with the
    bar's quote and paper_label (or label); table cells holding numbers with the row's quote and word cells."""
    b = normalize_board(spec)
    if b is None:
        return []
    out = []
    if b["kind"] == "bar":
        for i, x in enumerate(b["bars"]):
            out.append((f"bars.{i}.value", x["value"], x.get("quote", ""), x.get("paper_label") or x["label"]))
    elif b["kind"] == "table":
        for i, r in enumerate(b["rows"]):
            label = _row_label(r["cells"])
            for j, cell in enumerate(r["cells"]):
                if find_numbers(cell):
                    out.append((f"rows.{i}.cells.{j}", cell, r.get("quote", ""), label))
    return out


# ----------------------------------------------------------------------------- fonts

class BoardFace(Protocol):
    """One font at one size: width of a text, and drawing it with its line box's top-left at xy (glyph fallback inside)."""

    def length(self, text: str) -> float: ...

    def draw(self, draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, fill) -> None: ...


class BoardFonts(Protocol):
    """What render_board needs from a font source; lettering.FontSet satisfies it."""

    def face(self, language: str, size: int, bold: bool = False) -> BoardFace: ...


class _Face:
    """One size of a font chain: each glyph comes from the first font that has it, and every run sits on the main font's
    baseline (drawing mixed fonts from the same top-left would raise or sink the fallback glyphs). The fonts at the
    han_only positions (installed CJK fonts) draw only Han characters, the fonts at the synthetic positions get a
    synthetic bold; a character no font has is drawn as services.fonts.missing decides (an ASCII stand-in for a symbol,
    nothing for an emoji)."""

    def __init__(self, chain: tuple, size: int, han_only: frozenset = frozenset(), synthetic: frozenset = frozenset()):
        self.chain, self.han_only = chain, han_only
        self.fonts = [load_font(spec, size, k in synthetic) for k, spec in enumerate(chain)]
        with FT_LOCK:
            self.ascent = self.fonts[0].getmetrics()[0]

    def _index(self, ch: str) -> Optional[int]:
        """Position of the font that draws ch; None when no font has it (an emoji is never drawn)."""
        if unicodedata.category(ch).startswith("C"):
            return 0
        if is_emoji(ch):
            return None
        han = is_han(ch)
        return next((j for j, spec in enumerate(self.chain)
                     if (han or j not in self.han_only) and has_glyph(spec, ch)), None)

    def _runs(self, text: str) -> list:
        runs: list = []
        for ch in text:
            k = self._index(ch)
            pieces = ([(k, ch)] if k is not None else
                      [(self._index(c) or 0, c) for c in missing_glyph(ch, lambda c: self._index(c) is not None)])
            for k, piece in pieces:
                if runs and runs[-1][0] == k:
                    runs[-1][1] += piece
                else:
                    runs.append([k, piece])
        return [(self.fonts[k], t) for k, t in runs]

    def length(self, text: str) -> float:
        with FT_LOCK:
            return sum(f.getlength(t) for f, t in self._runs(text))

    def draw(self, draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, fill) -> None:
        x, y = xy
        with FT_LOCK:
            for f, t in self._runs(text):
                draw.text((x, y + self.ascent), t, font=f, fill=fill, anchor="ls", stroke_width=bold_stroke(f),
                          stroke_fill=fill)
                x += f.getlength(t)


@lru_cache(maxsize=8)
def _chain(lang: str, bold: bool) -> tuple[tuple, frozenset, frozenset]:
    """The bundled fonts of a language (services.fonts without config): the main face then its fallbacks, the han_only
    positions and, in bold, the synthetic ones (fonts without a bold file)."""
    plan, weight = resolve_fonts({}), "bold" if bold else "text"
    fonts = [plan.files[lang], *plan.chain(lang)]
    han_only = frozenset(j for j, f in enumerate(fonts) if f.get("han_only"))
    synthetic = frozenset(j for j, f in enumerate(fonts) if bold and f["bold"] == f["text"])
    return tuple(f[weight] for f in fonts), han_only, synthetic


@lru_cache(maxsize=256)
def _face(lang: str, bold: bool, size: int) -> _Face:
    chain, han_only, synthetic = _chain(lang, bold)
    return _Face(chain, size, han_only, synthetic)


class DefaultFonts:
    """A minimal BoardFonts used when no FontSet is passed: the bundled fonts lettering uses too (services.fonts, without
    config; ChillRound M for zh, Shantell Sans for en), with their fallbacks for missing glyphs."""

    def face(self, language: str, size: int, bold: bool = False) -> _Face:
        lang = "zh" if str(language).lower().startswith("zh") else "en"
        return _face(lang, bool(bold), max(1, int(size)))


@lru_cache(maxsize=1)
def default_fonts() -> DefaultFonts:
    return DefaultFonts()


# ----------------------------------------------------------------------------- text layout

SS = 2                 # boards are drawn at twice the size and downsampled: smooth marker strokes and glyph edges
SCRIPT_SCALE = 0.62    # superscripts / subscripts relative to the text size
_PIECE_RE = re.compile(f"([{_SUP_CHARS}]+)|([{_SUB_CHARS}]+)|([ \u00a0]?×[ \u00a0]?)|( ?→ ?)|(\u2212)")
_SYMBOL_W = {2: 0.58, 3: 1.0, 4: 0.5}   # widths of the drawn ×, → and − in ems
_NO_START = set("，。！？、；：”’）」』】》…%,.!?;:)]}％·")
_NO_END = set("“‘（「『【《([{")
_RUN_CHARS = "A-Za-z0-9\u00a0" + _SUP_CHARS + _SUB_CHARS
_ZH_TOKEN_RE = re.compile(rf"(?:[\u2212+\-](?=\d))?[{_RUN_CHARS}][{_RUN_CHARS}.,+\-×·^%/'’_()]*|\s+|.", re.S)   # "−0.9" stays whole
_TXT = {"en": {"higher": "higher is better", "lower": "lower is better", "illustrative": "illustration (example weights)",
               "checked": "tick: a chart in this manga shows it", "unit": "unit: {}", "vs": "{} vs {}"},
        "zh": {"higher": "越高越好", "lower": "越低越好", "illustrative": "示意图（权重为举例）",
               "checked": "打勾：漫画里有图表展示", "unit": "单位：{}", "vs": "{} 相对 {}"}}


class _Pen:
    """Measures and draws single lines of board text in one language. Superscripts and subscripts are drawn as small
    raised or lowered text and × / → as strokes, so no font needs those glyphs and every board looks the same."""

    def __init__(self, fonts: BoardFonts, lang: str):
        self.fonts, self.lang = fonts, lang
        self._faces: dict = {}
        self._boxes: dict = {}

    def face(self, size: float, bold: bool = False):
        key = (max(4, int(round(size))), bool(bold))
        if key not in self._faces:
            self._faces[key] = self.fonts.face(self.lang, key[0], key[1])
        return self._faces[key]

    @staticmethod
    def _pieces(text: str) -> list[tuple[str, int]]:
        """(text, kind) pieces: 0 plain, 1 superscript, -1 subscript, 2 a times sign, 3 an arrow, 4 a minus sign."""
        out, pos = [], 0
        for m in _PIECE_RE.finditer(text):
            if m.start() > pos:
                out.append((text[pos:m.start()].replace("\u00a0", " "), 0))
            if m.group(1):
                out.append(("".join(_FROM_SUP[c] for c in m.group(1)), 1))
            elif m.group(2):
                out.append(("".join(_FROM_SUB[c] for c in m.group(2)), -1))
            else:
                out.append(("", 2 if m.group(3) else 3 if m.group(4) else 4))
            pos = m.end()
        if pos < len(text):
            out.append((text[pos:].replace("\u00a0", " "), 0))
        return out

    def width(self, text: str, size: float, bold: bool = False) -> float:
        w = 0.0
        for t, k in self._pieces(text):
            w += size * _SYMBOL_W[k] if k in _SYMBOL_W else self.face(size * (SCRIPT_SCALE if k else 1), bold).length(t)
        return w

    def box(self, size: float, bold: bool = False, sample: str = "0") -> tuple[float, float]:
        """Ink top and bottom of a sample glyph relative to the y a line is drawn at (works with any face)."""
        key = (int(round(size)), bool(bold), sample)
        if key not in self._boxes:
            s = key[0]
            im = Image.new("RGB", (s * 4 + 16, s * 3 + 16), (0, 0, 0))
            self.face(s, bold).draw(ImageDraw.Draw(im), (8, s), sample, (255, 255, 255))
            bb = im.convert("L").getbbox()
            self._boxes[key] = (bb[1] - s, bb[3] - s) if bb else (0.25 * s, 1.0 * s)
        return self._boxes[key]

    def mid(self, size: float, bold: bool = False) -> float:
        top, bottom = self.box(size, bold, "中" if self.lang == "zh" else "0H")
        return (top + bottom) / 2

    def lh(self, size: float) -> float:
        return size * (1.36 if self.lang == "zh" else 1.3)

    def draw(self, d: ImageDraw.ImageDraw, x: float, y: float, text: str, size: float, bold: bool = False, fill=INK) -> None:
        """One line whose line box starts at y."""
        top, bottom = self.box(size, bold)
        for t, k in self._pieces(text):
            if not k:
                f = self.face(size, bold)
                f.draw(d, (x, y), t, fill)
                x += f.length(t)
                continue
            if k in _SYMBOL_W:
                w, cy = size * _SYMBOL_W[k], y + (top + bottom) / 2
                sw = max(1, int(round(size * (0.095 if bold else 0.07))))
                if k == 2:
                    h = size * 0.15
                    d.line([(x + w / 2 - h, cy - h), (x + w / 2 + h, cy + h)], fill=fill, width=sw)
                    d.line([(x + w / 2 - h, cy + h), (x + w / 2 + h, cy - h)], fill=fill, width=sw)
                elif k == 4:
                    d.line([(x + size * 0.07, cy), (x + w - size * 0.07, cy)], fill=fill, width=sw)
                else:
                    x1, x2, h = x + size * 0.14, x + w - size * 0.14, size * 0.17
                    d.line([(x1, cy), (x2, cy)], fill=fill, width=sw)
                    d.line([(x2 - h, cy - h), (x2, cy), (x2 - h, cy + h)], fill=fill, width=sw, joint="curve")
                x += w
                continue
            small = size * SCRIPT_SCALE
            f = self.face(small, bold)
            stop, sbottom = self.box(small, bold)
            y2 = y + top - 0.1 * size - stop if k > 0 else y + bottom + 0.18 * size - sbottom
            f.draw(d, (x, y2), t, fill)
            x += f.length(t)


def _protect(text: str) -> str:
    """Keep "1.4 × 10²⁰", joined names ("GNMT + RL") and a number with its unit or metric ("28.4 BLEU", "F1 93.2") on one line."""
    text = re.sub(r"(\d) ([×x·]) (?=10)", "\\1\u00a0\\2\u00a0", text)
    text = re.sub(r"(?<=[A-Za-z0-9)]) ([+&]) (?=[A-Za-z0-9(])", "\u00a0\\1\u00a0", text)
    text = re.sub(r"(?<![A-Za-z0-9.])(\d+(?:\.\d+)?%?) (?=[A-Za-z])", "\\1\u00a0", text)   # "28.4 BLEU", "8 GPUs"
    return re.sub(r"(?<![A-Za-z0-9])([A-Z][A-Za-z0-9]{0,5}) (?=\d)", "\\1\u00a0", text)   # "F1 77.8", "WMT 2014"


def _tokens(text: str, lang: str) -> list[str]:
    text = _protect(text)
    if lang != "zh":
        return re.findall(r"[^ \t\n]+|[ \t\n]+", text)
    out: list[str] = []
    for t in _ZH_TOKEN_RE.findall(text):
        if out and not t.isspace() and (t[0] in _NO_START or out[-1][-1] in _NO_END):
            out[-1] += t
        else:
            out.append(t)
    return out


def _wrap(pen: _Pen, text: str, max_w: float, size: float, bold: bool = False) -> list[str]:
    """Greedy wrap: en by words, zh by characters with kinsoku and whole Latin/number runs; a word wider than the line is
    split after its hyphens, slashes or arrows."""
    toks = []
    for t in _tokens(text, pen.lang):
        if not t.isspace() and pen.width(t, size, bold) > max_w:
            toks += re.findall(r"[^\-/→]+[\-/→]*|[\-/→]+", t)
        else:
            toks.append(t)
    lines, cur = [], ""
    for t in toks:
        cand = cur + t
        if cur.strip() and pen.width(cand.rstrip(), size, bold) > max_w:
            lines.append(cur.rstrip())
            cur = t.lstrip()
        else:
            cur = cand
    if cur.strip():
        lines.append(cur.rstrip())
    return lines


_BREAK_AFTER = set("，。！？、；：,.!?;:)]}）」』】》…—-/→")


def _break_cost(before: str, after: str) -> float:
    """How bad a line break between two tokens is: free at spaces and after punctuation, cheap where Latin meets CJK,
    dearer between two CJK characters (it may split a word such as 只是)."""
    if before.isspace() or after.isspace() or before[-1] in _BREAK_AFTER:
        return 0.0
    if bool(CJK_RE.match(before[-1])) != bool(CJK_RE.match(after[0])):
        return 0.06
    return 0.3


def _lines(pen: _Pen, text: str, max_w: float, size: float, bold: bool, balance: bool) -> list[str]:
    """The greedy line count, with the breaks chosen by a small dynamic program: even lines (balance) or just no orphan,
    and breaks at good places."""
    greedy = _wrap(pen, text, max_w, size, bold)
    target = len(greedy)
    if target < 2:
        return greedy
    toks = []
    for t in _tokens(text, pen.lang):
        toks += re.findall(r"[^\-/→]+[\-/→]*|[\-/→]+", t) if not t.isspace() and pen.width(t, size, bold) > max_w else [t]
    n, inf = len(toks), float("inf")
    pre = [0.0]
    for t in toks:
        pre.append(pre[-1] + pen.width(t, size, bold))
    first = [n] * (n + 1)    # first non-space token at or after i
    for i in range(n - 1, -1, -1):
        first[i] = i if not toks[i].isspace() else first[i + 1]
    end = [0] * (n + 1)      # end of the last non-space token at or before j
    for j in range(1, n + 1):
        end[j] = j if not toks[j - 1].isspace() else end[j - 1]

    def width(i, j):   # toks[i:j] without leading/trailing spaces
        a, b = first[i], end[j]
        return pre[b] - pre[a] if b > a else 0.0

    cost = [[inf] * (n + 1) for _ in range(target + 1)]
    back = [[0] * (n + 1) for _ in range(target + 1)]
    cost[0][0] = 0.0
    for k in range(1, target + 1):
        for jj in range(1, n + 1):
            last = k == target and jj == n
            for ii in range(jj):
                if cost[k - 1][ii] == inf:
                    continue
                w = width(ii, jj)
                if w > max_w + 0.5 or w == 0:
                    continue
                slack = (max_w - w) / max_w
                c = cost[k - 1][ii] + (slack ** 2 if balance or not last else 0.0)
                if not last:
                    c += _break_cost(toks[jj - 1], toks[jj]) if jj < n else inf
                elif not balance and w < 0.35 * max_w:
                    c += 0.5   # orphan
                if c < cost[k][jj]:
                    cost[k][jj], back[k][jj] = c, ii
    if cost[target][n] == inf:
        return greedy
    cuts, jj = [], n
    for k in range(target, 0, -1):
        cuts.append((back[k][jj], jj))
        jj = back[k][jj]
    lines = ["".join(toks[a:b]).strip() for a, b in reversed(cuts)]
    return lines if all(pen.width(l, size, bold) <= max_w + 0.5 for l in lines) else greedy


def _balanced(pen: _Pen, text: str, max_w: float, size: float, bold: bool = False) -> list[str]:
    """As many lines as a greedy wrap, evened out (no orphan words), broken at good places."""
    return _lines(pen, text, max_w, size, bold, True)


def _nice(pen: _Pen, text: str, max_w: float, size: float, bold: bool = False) -> list[str]:
    """As many lines as a greedy wrap, ragged right but without an orphan, broken at good places."""
    return _lines(pen, text, max_w, size, bold, False)


def _choose(sizes: list[int], attempt: Callable[[int, bool], Optional[tuple[bool, int, Any]]]) -> Any:
    """The plan for the largest size whose attempt fits, or for a size at most 14% smaller that needs fewer lines (fewer
    wraps read faster than a slightly bigger font); the smallest size's plan when nothing fits. attempt(size, final) ->
    (fits, line count, plan) or None; the search wraps greedily, and the chosen size is laid out again with final=True
    for the nicer line breaks (same line count)."""
    best, best_lines, ref, last = None, None, None, None
    for size in sizes:
        res = attempt(size, False)
        if res is None:
            continue
        ok, lines, _ = res
        last = size
        if not ok:
            continue
        if best is None:
            best, best_lines, ref = size, lines, size
        elif size < ref * 0.86:
            break
        elif lines < best_lines:
            best, best_lines = size, lines
    chosen = best if best is not None else last
    return None if chosen is None else attempt(chosen, True)[2]


def _sizes(hi: float, lo: float) -> list[int]:
    """Font sizes to try, largest first, about 4% apart."""
    hi, lo = int(round(hi)), int(round(lo))
    out, s = [], max(hi, lo)
    while s > lo:
        out.append(s)
        s -= max(1, int(s * 0.04))
    return out + [lo]


def _fit(pen: _Pen, text: str, max_w: float, max_h: float, hi: float, lo: float, bold: bool = False,
         max_lines: int = 99, balance: bool = True) -> Optional[tuple[int, list[str]]]:
    """Largest size in [lo, hi] at which text wraps into ≤ max_lines lines inside max_w × max_h: (size, lines) or None."""
    for size in _sizes(hi, lo):
        lines = _wrap(pen, text, max_w, size, bold)
        if (len(lines) <= max_lines and len(lines) * pen.lh(size) <= max_h + 0.5
                and all(pen.width(l, size, bold) <= max_w + 0.5 for l in lines)):
            return size, (_balanced(pen, text, max_w, size, bold) if balance else lines)
    return None


def _join(lines: list[str], lang: str) -> str:
    return ("" if lang == "zh" else " ").join(lines)


def _clip(pen: _Pen, text: str, max_w: float, size: float, bold: bool = False) -> str:
    """The text, shortened with an ellipsis only when even the smallest size cannot hold it."""
    if pen.width(text, size, bold) <= max_w + 0.5:
        return text
    while text and pen.width(text.rstrip() + "…", size, bold) > max_w + 0.5:
        text = text[:-1]
    return text.rstrip() + "…" if text else ""


def _fit_or_clip(pen: _Pen, text: str, max_w: float, max_h: float, hi: float, lo: float, bold: bool = False,
                 max_lines: int = 99) -> tuple[int, list[str]]:
    res = _fit(pen, text, max_w, max_h, hi, lo, bold, max_lines)
    if res:
        return res
    lo = int(round(lo))
    rows = max(1, min(max_lines, int(max_h // pen.lh(lo))))
    lines = _wrap(pen, text, max_w, lo, bold)
    if len(lines) > rows:
        lines = lines[: rows - 1] + [_join(lines[rows - 1:], pen.lang)]
    return lo, [_clip(pen, l, max_w, lo, bold) for l in lines]


# ----------------------------------------------------------------------------- drawing

class _Ctx:
    def __init__(self, img: Image.Image, pen: _Pen, seed: int):
        self.img, self.d, self.pen, self.lang = img, ImageDraw.Draw(img), pen, pen.lang
        self.W, self.H = img.size
        self.u = min(self.W / 900, self.H / 640)    # 1.0 on the 900×640 board the sizes below were tuned for
        self.m = max(3 * SS, round(min(self.W, self.H) * 0.035))
        self.min_px = 9 * SS
        self.rnd = random.Random(seed)

    def px(self, v: float) -> float:
        return max(self.min_px, v)

    def stroke(self, k: float = 4.0) -> int:
        return max(2, int(round(k * self.u)))

    def line(self, text: str, x: float, cy: float, size: float, bold: bool = False, fill=INK, align: str = "left") -> float:
        """One line vertically centred on cy; x is its left edge, centre or right edge."""
        w = self.pen.width(text, size, bold)
        x = x - w / 2 if align == "center" else x - w if align == "right" else x
        self.pen.draw(self.d, x, cy - self.pen.mid(size, bold), text, size, bold, fill)
        return w

    def block(self, lines: list[str], x: float, top: float, size: float, bold: bool = False, fill=INK,
              align: str = "left") -> float:
        """Lines stacked from top; returns the block height."""
        lh = self.pen.lh(size)
        for i, text in enumerate(lines):
            self.line(text, x, top + (i + 0.5) * lh, size, bold, fill, align)
        return lh * len(lines)

    def wob(self, pts: list, width: int, color, amp: Optional[float] = None, closed: bool = False) -> None:
        """A hand-drawn marker stroke through the points."""
        amp = 0.7 * self.u if amp is None else amp
        pts = list(pts) + ([pts[0]] if closed else [])
        out = []
        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            n = max(2, int(math.dist((x1, y1), (x2, y2)) / (16 * self.u + 1)))
            for i in range(n):
                t = i / n
                out.append((x1 + (x2 - x1) * t + self.rnd.uniform(-amp, amp), y1 + (y2 - y1) * t + self.rnd.uniform(-amp, amp)))
        out.append(pts[-1])
        self.d.line(out, fill=color, width=width, joint="curve")

    def box(self, rect, color, width: int, fill=(255, 255, 255), radius: Optional[float] = None) -> None:
        x1, y1, x2, y2 = rect
        r = min(x2 - x1, y2 - y1) * 0.2 if radius is None else radius
        r = max(0.0, min(r, (x2 - x1) / 2, (y2 - y1) / 2))
        self.d.rounded_rectangle(rect, radius=r, fill=fill)
        pts = []
        for cx, cy, a0 in ((x2 - r, y1 + r, -90), (x2 - r, y2 - r, 0), (x1 + r, y2 - r, 90), (x1 + r, y1 + r, 180)):
            pts += [(cx + r * math.cos(math.radians(a0 + 15 * k)), cy + r * math.sin(math.radians(a0 + 15 * k))) for k in range(7)]
        self.wob(pts, width, color, closed=True)

    def arrow(self, p1, p2, color=INK, width: Optional[int] = None) -> None:
        width = width or self.stroke(4)
        self.wob([p1, p2], width, color)
        ang = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        head = min(16 * self.u, math.dist(p1, p2) * 0.45)
        for da in (2.55, -2.55):
            self.d.line([p2, (p2[0] + head * math.cos(ang + da), p2[1] + head * math.sin(ang + da))], fill=color, width=width)

    def mark(self, x: float, top: float, size: float, good: bool) -> None:
        """A hand-drawn check (good, red) or cross (grey) in a size × size square."""
        w = max(2, int(round(size * 0.13)))
        if good:
            self.d.line([(x, top + size * 0.55), (x + size * 0.38, top + size * 0.9), (x + size, top + size * 0.1)],
                        fill=RED, width=w, joint="curve")
        else:
            self.d.line([(x + size * 0.1, top + size * 0.15), (x + size * 0.85, top + size * 0.9)], fill=GREY, width=w)
            self.d.line([(x + size * 0.85, top + size * 0.15), (x + size * 0.1, top + size * 0.9)], fill=GREY, width=w)


def _tint(color, t: float = 0.62) -> tuple:
    return tuple(int(c + (255 - c) * t) for c in color)


def _header(c: _Ctx, b: dict) -> float:
    """Title (bold, shrinks to fit, two lines at most) and subtitle (dark grey, wraps); returns the top of the content."""
    y, avail = c.m, c.W - 2 * c.m
    title = b.get("title", "")
    if title:
        hi = c.px(52 * c.u)
        res = _fit(c.pen, title, avail, c.H * 0.14, hi, c.px(hi * 0.74), True, 1) or \
            _fit_or_clip(c.pen, title, avail, c.H * 0.24, c.px(hi * 0.84), c.min_px, True, 2)
        y += c.block(res[1], c.m, y, res[0], True, INK)
    sub = b.get("subtitle", "")
    if sub:
        hi = c.px(36 * c.u)
        size, lines = _fit_or_clip(c.pen, sub, avail, c.H * 0.3, hi, c.px(hi * 0.72), False, 4)
        y += c.block(lines, c.m, y + 2 * c.u, size, False, NOTE_GREY) + 2 * c.u
    return y + (14 * c.u if title or sub else 0)


def _footer(c: _Ctx, left: str, right: str) -> float:
    """Dark grey footnotes on the bottom line (left: unit or note, right: source); returns the bottom of the content."""
    if not left and not right:
        return c.H - c.m
    avail, gap = c.W - 2 * c.m, 28 * c.u
    lo = c.px(24 * c.u)
    for size in _sizes(c.px(31 * c.u), lo):
        if c.pen.width(left, size) + c.pen.width(right, size) + (gap if left and right else 0) <= avail:
            cy = c.H - c.m - c.pen.lh(size) / 2
            if left:
                c.line(left, c.m, cy, size, False, NOTE_GREY)
            if right:
                c.line(right, c.W - c.m, cy, size, False, NOTE_GREY, "right")
            return c.H - c.m - c.pen.lh(size) - 8 * c.u
    lh = c.pen.lh(lo)
    y = c.H - c.m - lh * (2 if left and right else 1)
    if left:
        c.line(_clip(c.pen, left, avail, lo), c.m, y + lh / 2, lo, False, NOTE_GREY)
    if right:
        c.line(_clip(c.pen, right, avail, lo), c.W - c.m, c.H - c.m - lh / 2, lo, False, NOTE_GREY, "right")
    return y - 8 * c.u


# ----------------------------------------------------------------------------- bar

_BIN_RANGE_RE = re.compile(r"(?<![A-Za-z0-9.])\d+(?:\.\d+)?\s*[-–—~]\s*\d+(?:\.\d+)?(?![A-Za-z0-9.])")


def _changes(b: dict) -> bool:
    """Bars that are changes rather than scores: values on both sides of zero, or three or more bins of one quantity
    ("Calls 1-3", "Calls 4-6", ...)."""
    values = [v for v in (parse_value(x.get("value")) for x in b.get("bars") or []) if v is not None]
    bins = len(b.get("bars") or []) >= 3 and all(_BIN_RANGE_RE.search(str(x.get("label") or "")) for x in b.get("bars") or [])
    return bool(values) and min(values) < 0 < max(values) or bins   # a trend over "7B, 13B, 70B" keeps its direction


def _bar_caption(b: dict, lang: str, badge: Optional[dict] = None) -> str:
    """The bar chart's footnote: unit · direction · what the badge measures. Readers took a lone "%" for a stray mark and
    could not tell which bar a "+2.40" badge was measured against (only a faint dashed line said so)."""
    unit = b.get("unit", "")
    said = " ".join([b.get("title", ""), b.get("subtitle", ""), unit])
    # a chart of changes (both signs, or bins of one quantity) has no better direction of its own: econ_ai_service's call
    # durations printed "lower is better" in four rounds while the story called shorter calls lost effort
    direction = "" if _DIRECTION_RE.search(said) or _changes(b) else _TXT[lang]["higher" if b.get("higher_is_better", True) else "lower"]
    shown = unit if re.search(r"[^\W\d_]", unit) else _TXT[lang]["unit"].format(unit) if unit else ""   # "BLEU" as is, "%" named
    rival = b["bars"][badge["rival"]].get("label", "") if badge else ""
    return " · ".join(x for x in (shown, direction, _TXT[lang]["vs"].format(badge["text"], rival) if rival else "") if x)


def _bar_scale(vals: list[float], extra: list[float], x0: float, x1: float) -> tuple[float, float]:
    """Pixels per unit and the x of zero so that every bar plus the labels at its end fits in [x0, x1] (zero-based)."""
    pos = [(v, e) for v, e in zip(vals, extra) if v >= 0]
    neg = [(-v, e) for v, e in zip(vals, extra) if v < 0]
    span = x1 - x0
    if not neg:
        ks = [(span - e) / v for v, e in pos if v > 0]
        return max(0.0, min(ks)) if ks else 0.0, x0
    if not pos:
        return max(0.0, min((span - e) / v for v, e in neg)), x1
    k = max(0.0, min((span - ep - en) / (vp + vn) for vp, ep in pos for vn, en in neg))
    return k, x0 + max(k * vn + en for vn, en in neg)


def _render_bar(c: _Ctx, b: dict) -> None:
    bars, n = b["bars"], len(b["bars"])
    top = _header(c, b)
    badge = delta_badge(b)
    bottom = _footer(c, _bar_caption(b, c.lang, badge), b.get("source", ""))
    vals = [parse_value(x["value"]) or 0.0 for x in bars]
    texts = [display_value(x["value"]) for x in bars]
    hl = [bool(x.get("highlight")) for x in bars]
    marked = [i for i, x in enumerate(bars) if x.get("baseline") and not x.get("highlight")]
    rival = badge["rival"] if badge else marked[0] if marked else None
    colors = [RED if hl[i] else BLUE if (i == rival or not any(hl)) else GREY for i in range(n)]
    left, right = c.m, c.W - c.m
    area = max(1.0, bottom - top)
    plan = None
    for L in _sizes(c.px(42 * c.u), c.min_px):
        V = L * 1.1
        slot = min(area / n, L * 3.6)
        label_max = (right - left) * 0.42
        labels = [_wrap(c.pen, x["label"], label_max, L) or [""] for x in bars]
        fits = (all(len(w) <= 2 for w in labels) and all(c.pen.width(l, L) <= label_max + 0.5 for w in labels for l in w)
                and max(len(w) for w in labels) * c.pen.lh(L) <= slot * 0.98)
        if not fits and L > c.min_px:
            continue
        if fits:
            labels = [_balanced(c.pen, x["label"], label_max, L) or [""] for x in bars]
        else:
            labels = [[_clip(c.pen, _join(w, c.lang), label_max, L)] for w in labels]
        gap = 0.5 * L
        x_axis = left + max(c.pen.width(l, L) for w in labels for l in w) + gap
        Vb = min(V * 0.9, slot * 0.62)
        badge_w = c.pen.width(badge["text"], Vb, True) + Vb * 0.9 if badge else 0.0
        vw = [c.pen.width(t, V, True) for t in texts]
        extra = [vw[i] + gap + (badge_w + gap if badge and i == badge["best"] else 0) for i in range(n)]
        k, x_zero = _bar_scale(vals, extra, x_axis, right)
        longest = max(abs(v) for v in vals) * k
        plan = dict(L=L, V=V, Vb=Vb, slot=slot, labels=labels, gap=gap, x_axis=x_axis, k=k, x_zero=x_zero, badge_w=badge_w)
        if longest >= 0.25 * (right - x_axis) or not any(vals) or L <= c.min_px:
            break
    p = plan
    L, V, slot, gap, k, x_zero = p["L"], p["V"], p["slot"], p["gap"], p["k"], p["x_zero"]
    y0 = top + (area - slot * n) * 0.35
    thick = min(slot * 0.62, V * 1.75)
    centers = [y0 + slot * (i + 0.5) for i in range(n)]
    ends = [x_zero + k * v for v in vals]
    if badge:   # dashed guide from the rival's bar end to the highlighted bar: the gap the badge measures
        x, ya, yb = ends[badge["rival"]], centers[badge["rival"]], centers[badge["best"]]
        step = 14 * c.u
        y, direction = ya, 1 if yb > ya else -1
        while (yb - y) * direction > 0:
            y2 = y + direction * min(step * 0.6, abs(yb - y))
            c.d.line([(x, y), (x, y2)], fill=_tint(BLUE, 0.25), width=c.stroke(3))
            y += direction * step
    for i, x in enumerate(bars):
        cy, col = centers[i], colors[i]
        lh = c.pen.lh(L)
        lines = p["labels"][i]
        for j, text in enumerate(lines):
            c.line(text, p["x_axis"] - gap, cy + (j - (len(lines) - 1) / 2) * lh, L, False, INK, "right")
        x1, x2 = sorted((x_zero, ends[i]))
        if x2 - x1 >= 1:
            c.d.rectangle([x1, cy - thick / 2, x2, cy + thick / 2], fill=_tint(col))
            c.wob([(x1, cy - thick / 2), (x2, cy - thick / 2), (x2, cy + thick / 2), (x1, cy + thick / 2)], c.stroke(4), col)
        label_col = VALUE_GREY if col == GREY else col
        if vals[i] >= 0:
            end = ends[i] + gap + c.line(texts[i], ends[i] + gap, cy, V, True, label_col)
        else:
            c.line(texts[i], ends[i] - gap, cy, V, True, label_col, "right")
            end = ends[i]
        if badge and i == badge["best"]:
            Vb = p["Vb"]
            bx1, bh = end + gap, Vb * 1.5
            tone = RED if badge["better"] else GREY
            c.box((bx1, cy - bh / 2, bx1 + p["badge_w"], cy + bh / 2), tone, c.stroke(3), _tint(tone, 0.86), bh / 2)
            c.line(badge["text"], bx1 + p["badge_w"] / 2, cy, Vb, True, tone, "center")
    c.wob([(x_zero, y0 + slot * 0.08), (x_zero, y0 + slot * n - slot * 0.08)], c.stroke(5), INK)


# ----------------------------------------------------------------------------- table

def _render_table(c: _Ctx, b: dict) -> None:
    top = _header(c, b)
    bottom = _footer(c, b.get("note", ""), b.get("source", ""))
    cols = b["columns"]
    grid = [cols] + [r["cells"] for r in b["rows"]]
    grid = [[display_value(cell) if cell and not _translatable(cell) else cell for cell in row] for row in grid]
    ncol, hl = len(cols), set(b.get("highlight_rows", []))
    numeric = [[bool(cell) and not _translatable(cell) for cell in row] for row in grid]
    avail_w, avail_h = c.W - 2 * c.m, max(1.0, bottom - top)
    has_header = any(cols)
    def attempt(T, final):
        pad = T * 0.42
        bold = [[ri == 0 or ri - 1 in hl for _ in row] for ri, row in enumerate(grid)]
        wrap = _balanced if final else _wrap
        nat = [max(c.pen.width(grid[r][j], T, bold[r][j]) for r in range(len(grid))) + 2 * pad for j in range(ncol)]
        if sum(nat) <= avail_w:
            widths = nat
        else:
            mins = [max(max((c.pen.width(t, T, bold[r][j]) for t in _tokens(grid[r][j], c.lang) if not t.isspace()), default=0)
                        for r in range(len(grid))) + 2 * pad for j in range(ncol)]
            if sum(mins) > avail_w and T > c.min_px:
                return None
            spare, flex = max(0.0, avail_w - sum(mins)), [max(0.0, a - m) for a, m in zip(nat, mins)]
            widths = [m + spare * f / (sum(flex) or 1) for m, f in zip(mins, flex)]
        cells = [[wrap(c.pen, grid[r][j], max(1.0, widths[j] - 2 * pad), T, bold[r][j]) or [""] for j in range(ncol)]
                 for r in range(len(grid))]
        lh = c.pen.lh(T)
        heights = [max(len(x) for x in row) * lh + 1.1 * pad for row in cells]
        if not has_header:
            heights[0] = 0
        fits = all(len(x) <= 3 for row in cells for x in row) and sum(heights) <= avail_h
        plan = dict(T=T, pad=pad, widths=widths, cells=cells, heights=heights, bold=bold)
        return fits or T <= c.min_px, sum(len(x) for row in cells for x in row), plan

    plan = _choose(_sizes(c.px(44 * c.u), c.min_px), attempt)
    p = plan
    T, pad, widths, heights = p["T"], p["pad"], p["widths"], p["heights"]
    if sum(widths) < avail_w * 0.72:   # a narrow table reads better spread over most of the board
        widths = [w * avail_w * 0.72 / sum(widths) for w in widths]
    elif sum(widths) > avail_w:        # smallest size and still too wide: narrower columns, cells clipped below
        widths = [w * avail_w / sum(widths) for w in widths]
    scale = min(1.0, avail_h / max(1.0, sum(heights)))
    heights = [h * scale for h in heights]
    x0 = c.m + (avail_w - sum(widths)) / 2
    y = top + max(0.0, (avail_h - sum(heights)) * 0.25)
    lh = c.pen.lh(T)
    for r, row in enumerate(p["cells"]):
        h = heights[r]
        if r == 0 and not has_header:
            continue
        if r - 1 in hl:
            c.d.rectangle([x0 - pad * 0.3, y + 3 * c.u, x0 + sum(widths) + pad * 0.3, y + h - 3 * c.u], fill=(252, 226, 220))
        x = x0
        for j, lines in enumerate(row):
            color = RED if r - 1 in hl else INK
            lines = [_clip(c.pen, l, max(1.0, widths[j] - 2 * pad), T, p["bold"][r][j]) for l in lines[: max(1, min(3, int(h // lh)))]]
            block_top = y + (h - lh * len(lines)) / 2
            if numeric[r][j] and j > 0:
                c.block(lines, x + widths[j] / 2, block_top, T, p["bold"][r][j], color, "center")
            else:
                c.block(lines, x + pad, block_top, T, p["bold"][r][j], color)
            x += widths[j]
        y += h
        c.wob([(x0, y), (x0 + sum(widths), y)], c.stroke(5 if r == 0 else 2.5), INK if r == 0 else _tint(INK, 0.45))


# ----------------------------------------------------------------------------- flow

def _flow_fit(c: _Ctx, steps: list[str], bold: list[bool], box_w: float,
              box_h_max: float) -> Optional[tuple[int, list[list[str]], float]]:
    """Largest font (or one ≤14% smaller needing fewer lines) at which every step fits a box_w-wide box ≤ box_h_max tall:
    (size, lines per step, box height). Highlighted steps are measured bold, as they are drawn."""
    if box_w <= 0 or box_h_max <= 0:
        return None

    def attempt(T, final):
        pad = T * 0.55
        wrapped = [(_balanced if final else _wrap)(c.pen, s, box_w - 2 * pad, T, b) for s, b in zip(steps, bold)]
        if any(len(w) > 4 or any(c.pen.width(l, T, b) > box_w - 2 * pad + 0.5 for l in w) for w, b in zip(wrapped, bold)):
            return None
        need = max(len(w) for w in wrapped) * c.pen.lh(T) + 2 * pad * 0.8
        if need > box_h_max:
            return None
        return True, sum(len(w) for w in wrapped), (T, wrapped, min(box_h_max, max(need, T * 2.6)))

    return _choose(_sizes(c.px(48 * c.u), c.min_px), attempt)


def _render_flow(c: _Ctx, b: dict) -> None:
    top = _header(c, b)
    bottom = _footer(c, b.get("note", ""), b.get("source", ""))
    steps, hl = b["steps"], set(b.get("highlight", []))
    n, bold = len(steps), [i in hl for i in range(len(b["steps"]))]
    avail_w, avail_h = c.W - 2 * c.m, max(1.0, bottom - top)
    gx, gy = 52 * c.u, 40 * c.u
    options = []
    row_w = (avail_w - (n - 1) * gx) / n
    wide = avail_w > avail_h * 1.2    # a column wastes a wide board and reads against the left-to-right flow
    if (res := _flow_fit(c, steps, bold, row_w, min(avail_h, row_w * 1.1))):
        options.append((res[0], 1.0, "row", res))
    col_h = (avail_h - (n - 1) * gy) / n
    if (res := _flow_fit(c, steps, bold, avail_w * 0.7, col_h)):
        options.append((res[0], 0.88 if wide else 1.0, "column", res))
    if n >= 4:
        per = math.ceil(n / 2)
        two_w = (avail_w - (per - 1) * gx) / per
        if (res := _flow_fit(c, steps, bold, two_w, min((avail_h - gy * 1.6) / 2, two_w * 1.1))):
            options.append((res[0], 0.96, "two", res))
    if not options:   # tiny board: one column at the smallest size, text clipped
        T = c.min_px
        options.append((T, 1.0, "column", (T, [[_clip(c.pen, s, avail_w * 0.7 - T, T, b)] for s, b in zip(steps, bold)], max(1.0, col_h))))
    # bigger text wins, but each extra line per box costs ~9% and the layout's fit to the board counts too
    best = max(options, key=lambda o: o[0] * o[1] * (1 - 0.09 * (max(len(x) for x in o[3][1]) - 1)))
    layout, (T, lines, bh) = best[2], best[3]
    rects = []
    if layout == "row":
        y = top + (avail_h - bh) / 2
        rects = [(c.m + i * (row_w + gx), y, c.m + i * (row_w + gx) + row_w, y + bh) for i in range(n)]
    elif layout == "column":
        bw = avail_w * 0.7
        x = c.m + (avail_w - bw) / 2
        y0 = top + max(0.0, (avail_h - n * bh - (n - 1) * gy) / 2)
        rects = [(x, y0 + i * (bh + gy), x + bw, y0 + i * (bh + gy) + bh) for i in range(n)]
    else:
        per = math.ceil(n / 2)
        bw = (avail_w - (per - 1) * gx) / per
        y0 = top + max(0.0, (avail_h - 2 * bh - gy * 1.6) / 2)
        for i in range(n):
            r, k = divmod(i, per)
            count = per if r == 0 else n - per
            x0 = c.m + (avail_w - (count * bw + (count - 1) * gx)) / 2
            y = y0 + r * (bh + gy * 1.6)
            rects.append((x0 + k * (bw + gx), y, x0 + k * (bw + gx) + bw, y + bh))
    for i in range(n - 1):
        a, z = rects[i], rects[i + 1]
        if abs(a[1] - z[1]) < 1:      # same row
            c.arrow((a[2] + 6 * c.u, (a[1] + a[3]) / 2), (z[0] - 6 * c.u, (z[1] + z[3]) / 2))
        elif layout == "column":
            c.arrow(((a[0] + a[2]) / 2, a[3] + 4 * c.u), ((z[0] + z[2]) / 2, z[1] - 4 * c.u))
        else:                         # elbow from the end of the first row down to the start of the second
            ax, zx = (a[0] + a[2]) / 2, (z[0] + z[2]) / 2
            ym = (a[3] + z[1]) / 2
            c.wob([(ax, a[3] + 4 * c.u), (ax, ym), (zx, ym)], c.stroke(4), INK)
            c.arrow((zx, ym), (zx, z[1] - 4 * c.u))
    for i, (rect, text) in enumerate(zip(rects, lines)):
        col = RED if i in hl else BLUE
        c.box(rect, col, c.stroke(4.5 if i in hl else 3.5), _tint(col, 0.9) if i in hl else (255, 255, 255))
        lh = c.pen.lh(T)
        text = [_clip(c.pen, l, rect[2] - rect[0] - T * 0.6, T, i in hl) for l in text]
        c.block(text, (rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2 - lh * len(text) / 2, T, i in hl, col, "center")


# ----------------------------------------------------------------------------- compare

def _render_compare(c: _Ctx, b: dict) -> None:
    top = _header(c, b)
    bottom = _footer(c, b.get("note", ""), b.get("source", ""))
    sides = [b["left"], b["right"]]
    tones = [GREY, RED]
    gap = 44 * c.u
    colw = (c.W - 2 * c.m - gap) / 2
    xs = [c.m, c.m + colw + gap]
    avail_h = max(1.0, bottom - top)
    paired = len(sides[0]["items"]) == len(sides[1]["items"])

    def attempt(T, final):
        Ls = min(c.px(50 * c.u), T * 1.2)
        labels = [_fit_or_clip(c.pen, s["label"], colw, Ls * 2.8, Ls, c.px(Ls * 0.7), True, 2) if s["label"] else (Ls, [])
                  for s in sides]
        label_h = max(len(lab[1]) * c.pen.lh(lab[0]) for lab in labels)
        mark_w = T * 1.15
        items = [[(_balanced if final else _wrap)(c.pen, it, colw - mark_w, T) for it in s["items"]] for s in sides]
        lh, gi = c.pen.lh(T), T * 0.5
        if any(c.pen.width(l, T) > colw - mark_w + 0.5 for side in items for it in side for l in it) and T > c.min_px:
            return None
        if paired:
            rows = [max(len(a), len(z)) * lh for a, z in zip(*items)]
            total = sum(rows) + gi * (len(rows) - 1)
        else:
            rows = None
            total = max(sum(len(it) * lh for it in side) + gi * (len(side) - 1) for side in items)
        under = 18 * c.u if label_h else 0
        used = label_h + under + total
        plan = dict(T=T, labels=labels, label_h=label_h, items=items, rows=rows, lh=lh, gi=gi, mark_w=mark_w, under=under, used=used)
        return used <= avail_h or T <= c.min_px, sum(len(it) for side in items for it in side), plan

    p = _choose(_sizes(c.px(48 * c.u), c.min_px), attempt)
    T, lh, gi = p["T"], p["lh"], p["gi"]
    y_top = top + max(0.0, (avail_h - p["used"]) * 0.2)
    for k, side in enumerate(sides):
        x, tone = xs[k], tones[k]
        size, lines = p["labels"][k]
        if lines:
            c.block(lines, x + colw / 2, y_top, size, True, tone, "center")
            yl = y_top + p["label_h"] + 6 * c.u
            c.wob([(x + 10 * c.u, yl), (x + colw - 10 * c.u, yl)], c.stroke(4), tone)
        y = y_top + p["label_h"] + p["under"]
        for i, it in enumerate(p["items"][k]):
            if y + lh * len(it) > bottom + 0.5:
                it = [_clip(c.pen, l, colw - p["mark_w"], T) for l in it[: max(1, int((bottom - y) // lh))]]
            r = T * 0.17   # a neutral bullet: the side's colour tells old from new, a ✗ / ✓ would pass a verdict
            c.d.ellipse((x + T * 0.3 - r, y + lh / 2 - r, x + T * 0.3 + r, y + lh / 2 + r), fill=tone)
            c.block([_clip(c.pen, l, colw - p["mark_w"], T) for l in it], x + p["mark_w"], y, T, False, INK)
            y += (p["rows"][i] if p["rows"] else lh * len(it)) + gi
    c.wob([(c.W / 2, y_top), (c.W / 2, min(bottom, y_top + p["used"] + 20 * c.u))], c.stroke(2.5), _tint(GREY, 0.3))


# ----------------------------------------------------------------------------- list

def _render_list(c: _Ctx, b: dict) -> None:
    top = _header(c, b)
    items, n = b["items"], len(b["items"])
    # ticks and bullets mixed: say what a tick means (readers could not tell which lines counted as shown); never what a
    # bullet lacks: "dot: no chart here" read as the book admitting its gaps (round 23's blind judges, 3 of 7 books)
    legend = _TXT[c.lang]["checked"] if "checked" in b and not b.get("open") and len(b["checked"]) < n else ""
    bottom = _footer(c, " · ".join(x for x in (b.get("note", ""), legend) if x), b.get("source", ""))
    avail_w, avail_h = c.W - 2 * c.m, max(1.0, bottom - top)

    def attempt(T, final):
        mark_w = T * 1.3
        lines = [(_nice if final else _wrap)(c.pen, it, avail_w - mark_w, T) for it in items]
        if any(c.pen.width(l, T) > avail_w - mark_w + 0.5 for it in lines for l in it) and T > c.min_px:
            return None
        lh, gap = c.pen.lh(T), T * 0.42
        count = sum(len(it) for it in lines)
        total = count * lh + gap * (n - 1)
        return total <= avail_h or T <= c.min_px, count, (T, mark_w, lines, lh, gap, total)

    T, mark_w, lines, lh, gap, total = _choose(_sizes(c.px(54 * c.u), c.min_px), attempt)
    if n > 1:   # spread the spare height a little: the recap breathes instead of hugging the title
        extra = min((avail_h - total) / (n - 1), T * 0.55) if avail_h > total else 0
        gap, total = gap + extra, total + extra * (n - 1)
    y = top + max(0.0, (avail_h - total) * 0.3)
    for k, it in enumerate(lines):
        if y + lh * len(it) > bottom + 0.5:
            it = [_clip(c.pen, l, avail_w - mark_w, T) for l in it[: max(1, int((bottom - y) // lh))]]
        if b.get("open"):   # still to be checked: an empty box
            box = T * 0.78
            bx, by = c.m + T * 0.12, y + (lh - box) / 2
            c.box((bx, by, bx + box, by + box), INK, c.stroke(3), (255, 255, 255), box * 0.18)
        elif "checked" in b and k not in b["checked"]:   # stated by the paper, its evidence not shown: a bullet, no tick
            r = T * 0.17
            cx, cy = c.m + T * 0.5, y + lh / 2
            c.d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=GREY)
        else:
            c.mark(c.m + T * 0.1, y + (lh - T * 0.85) / 2, T * 0.85, True)
        c.block([_clip(c.pen, l, avail_w - mark_w, T) for l in it], c.m + mark_w, y, T, False, INK)
        y += lh * len(it) + gap


# ----------------------------------------------------------------------------- links

def _render_links(c: _Ctx, b: dict) -> None:
    top = _header(c, b)
    tag = _TXT[c.lang]["illustrative"] if b.get("illustrative", True) else ""
    bottom = _footer(c, b.get("note", ""), tag)
    toks, weights, src = b["tokens"], b["weights"], b["from"]
    wmax = max(w for i, w in enumerate(weights) if i != src) or 1.0
    rel = [w / wmax if i != src else 0.0 for i, w in enumerate(weights)]
    strong, medium, faint = LINK_TIERS
    avail_w, avail_h = c.W - 2 * c.m, max(1.0, bottom - top)
    plan = None
    for T in _sizes(c.px(46 * c.u), c.min_px):
        padx, bh, gap = T * 0.45, T * 1.7, T * 0.42
        widths = [min(avail_w, c.pen.width(t, T, i == src) + 2 * padx) for i, t in enumerate(toks)]
        rows, cur, cur_w = [], [], 0.0
        for i, w in enumerate(widths):
            if cur and cur_w + gap + w > avail_w:
                rows.append(cur)
                cur, cur_w = [], 0.0
            cur.append(i)
            cur_w += w + (gap if len(cur) > 1 else 0)
        rows.append(cur)
        arc = min(avail_h - bh, T * 5.0) if len(rows) == 1 else T * 2.3
        need = len(rows) * (bh + arc) + (len(rows) - 1) * gap
        plan = dict(T=T, padx=padx, bh=bh, gap=gap, widths=widths, rows=rows, arc=arc, need=need)
        if (need <= avail_h and arc >= T * 1.2) or T <= c.min_px:
            break
    p = plan
    T, bh, gap, arc = p["T"], p["bh"], p["gap"], max(0.0, p["arc"])
    y = top + max(0.0, (avail_h - p["need"]) / 2)
    rects, row_of = {}, {}
    for r, row in enumerate(p["rows"]):
        y += arc
        row_w = sum(p["widths"][i] for i in row) + gap * (len(row) - 1)
        x = c.m + (avail_w - row_w) / 2
        for i in row:
            rects[i], row_of[i] = (x, y, x + p["widths"][i], y + bh), r
            x += p["widths"][i] + gap
        y += bh + gap
    sx1, sy1, sx2, sy2 = rects[src]
    scx = (sx1 + sx2) / 2
    order = sorted((i for i in range(len(toks)) if i != src and rel[i] >= faint), key=lambda i: rel[i])
    span = max((abs((rects[i][0] + rects[i][2]) / 2 - scx) for i in order if row_of[i] == row_of[src]), default=1.0) or 1.0
    for i in order:     # weakest first, so the strong red links stay on top
        x1, y1, x2, y2 = rects[i]
        tcx = (x1 + x2) / 2
        width = max(2, int(round(max(1.5 * SS, rel[i] * T * 0.42))))
        color = RED if rel[i] >= strong else BLUE if rel[i] >= medium else _tint(GREY, 0.35)
        if row_of[i] == row_of[src]:
            h = arc * (0.35 + 0.6 * abs(tcx - scx) / span)
            p0, p2 = (scx + (6 * c.u if tcx > scx else -6 * c.u), sy1), (tcx, y1)
            p1 = ((p0[0] + p2[0]) / 2, sy1 - 2 * h)
            pts = [((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0],
                    (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1]) for t in (j / 40 for j in range(41))]
        else:
            down = row_of[i] > row_of[src]
            p0, p3 = (scx, sy2 if down else sy1), (tcx, y1 if down else y2)
            dy = (p3[1] - p0[1]) * 0.5
            pts = [((1 - t) ** 3 * p0[0] + 3 * (1 - t) ** 2 * t * p0[0] + 3 * (1 - t) * t * t * p3[0] + t ** 3 * p3[0],
                    (1 - t) ** 3 * p0[1] + 3 * (1 - t) ** 2 * t * (p0[1] + dy) + 3 * (1 - t) * t * t * (p3[1] - dy) + t ** 3 * p3[1])
                   for t in (j / 40 for j in range(41))]
        c.d.line(pts, fill=color, width=width, joint="curve")
    for i, t in enumerate(toks):
        rect = rects[i]
        if i == src:
            c.box(rect, RED, c.stroke(5), _tint(RED, 0.88), T * 0.3)
            col = RED
        elif rel[i] >= strong:
            c.box(rect, BLUE, c.stroke(4), (255, 255, 255), T * 0.3)
            col = BLUE
        else:
            c.box(rect, _tint(INK, 0.35), c.stroke(2.5), (255, 255, 255), T * 0.3)
            col = INK
        c.line(_clip(c.pen, t, rect[2] - rect[0] - 2 * p["padx"], T, i == src), (rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2,
               T, i == src, col, "center")


# ----------------------------------------------------------------------------- diagram

_TONE_RGB = {"ink": INK, "red": RED, "blue": BLUE, "grey": GREY}


def _dashed(c: _Ctx, pts: list, width: int, color, dash: float) -> None:
    """A dashed marker stroke through the points (dash and gap of about ``dash`` px)."""
    carry, on = 0.0, True
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        seg = math.dist((x1, y1), (x2, y2))
        pos = 0.0
        while pos < seg:
            step = min(dash - carry, seg - pos)
            if on:
                a, b = pos / seg if seg else 0, (pos + step) / seg if seg else 0
                c.d.line([(x1 + (x2 - x1) * a, y1 + (y2 - y1) * a), (x1 + (x2 - x1) * b, y1 + (y2 - y1) * b)], fill=color, width=width)
            pos += step
            carry += step
            if carry >= dash - 1e-6:
                carry, on = 0.0, not on


def _back(pts: list, length: float):
    """The point ``length`` back along the stroke from its last point (or its first point): an arrowhead on a sampled
    curve points along the curve, at full size, not along its last tiny step."""
    left = length
    for a, b in zip(reversed(pts), list(reversed(pts))[1:]):
        d = math.dist(a, b)
        if d >= left:
            t = left / d if d else 0.0
            return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
        left -= d
    return pts[0]


def _head(c: _Ctx, tip, frm, color, width: int) -> None:
    ang = math.atan2(tip[1] - frm[1], tip[0] - frm[0])
    head = min(18 * c.u, math.dist(tip, frm) * 0.6)
    for da in (2.55, -2.55):
        c.d.line([tip, (tip[0] + head * math.cos(ang + da), tip[1] + head * math.sin(ang + da))], fill=color, width=width)


def _render_diagram(c: _Ctx, b: dict) -> None:
    """The designer's elements on the diagram canvas (diagram.W x diagram.H), scaled into the board under its header: fills first, then
    lines and arrows, marks, outlines and texts (services.diagram places every text, the same way its lint measured)."""
    from services import diagram as D
    top = _header(c, b)
    bottom = _footer(c, b.get("note", ""), b.get("source", ""))
    els = b.get("elements") or []
    if not els:
        return
    avail_w, avail_h = int(c.W - 2 * c.m), int(max(1.0, bottom - top))
    if avail_w < 8 or avail_h < 8:
        return
    placed = D.place(b, _Pen(c.pen.fonts, c.lang), _wrap_lines)
    # the drawn content, not the whole canvas, fills the board (up to 1.5 times the canvas scale); what a kept design
    # draws past the canvas is shrunk in with the rest, never cut at the board's edge ("26,070 qu")
    x1, y1, x2, y2 = D.bounds(b, placed)
    pad = 12.0
    x1, y1, x2, y2 = x1 - pad, y1 - pad, x2 + pad, y2 + pad
    full = min(avail_w / D.W, avail_h / D.H)
    s = min(avail_w / max(1.0, x2 - x1), avail_h / max(1.0, y2 - y1), full * 1.5)
    ox, oy = (avail_w - (x2 - x1) * s) / 2 - x1 * s, (avail_h - (y2 - y1) * s) / 2 - y1 * s
    # drawn on a layer of the content area and pasted: nothing (a wobble, a text at the smallest size) reaches the edge
    outer = c
    c = _Ctx(Image.new("RGB", (avail_w, avail_h), (255, 255, 255)), outer.pen, _SEEDS["diagram"])
    c.u, c.min_px = outer.u, outer.min_px

    def P(x, y):
        return (ox + x * s, oy + y * s)

    # fills and outlines of boxes and circles, the biggest first: a box drawn around others (listed after them) never
    # paints over them, and "none" is no fill at all (it was white, and a group box listed last erased what it held)
    area = lambda e: e["w"] * e["h"] if e["type"] == "box" else math.pi * e["r"] ** 2 if e["type"] == "circle" else 0.0
    for e in sorted((e for e in els if e["type"] in ("box", "circle")), key=area, reverse=True):
        col = _TONE_RGB[e["tone"]]
        fill = _tint(col, 0.86) if e.get("fill") == "tint" else col if e.get("fill") == "solid" else None
        width = c.stroke(4.5 if e["tone"] == "red" else 3.5)
        if e["type"] == "box":
            x1, y1 = P(e["x"], e["y"])
            x2, y2 = P(e["x"] + e["w"], e["y"] + e["h"])
            if e.get("dashed"):
                r = min(x2 - x1, y2 - y1) * 0.12
                c.d.rounded_rectangle((x1, y1, x2, y2), radius=r, fill=fill)
                _dashed(c, [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)], width, col, 14 * c.u)
            else:
                c.box((x1, y1, x2, y2), col, width, fill, radius=min(x2 - x1, y2 - y1) * 0.12)
        elif e["type"] == "circle":
            (cx, cy), r = P(e["x"], e["y"]), e["r"] * s
            c.d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=fill)
            ring = [(cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a))) for a in range(0, 361, 12)]
            (_dashed(c, ring, width, col, 12 * c.u) if e.get("dashed") else c.wob(ring, width, col, closed=False))
    for e in els:   # lines and arrows
        if e["type"] not in ("arrow", "line"):
            continue
        col, width = _TONE_RGB[e["tone"]], c.stroke(4.5 if e["tone"] == "red" else 3.5)
        pts = [P(x, y) for x, y in D.path(e)]
        if e.get("dashed"):
            _dashed(c, pts, width, col, 14 * c.u)
        else:
            c.wob(pts, width, col)
        if e["type"] == "arrow":   # a curve's head points along the curve; a straight leg's along its last leg
            back = (lambda q: _back(q, 24 * c.u)) if e.get("curved") else (lambda q: next((x for x in reversed(q[:-1]) if x != q[-1]), q[0]))
            if e["heads"] in ("end", "both"):
                _head(c, pts[-1], back(pts), col, width)
            if e["heads"] in ("start", "both"):
                _head(c, pts[0], back(pts[::-1]), col, width)
    for e in els:   # marks
        if e["type"] != "mark":
            continue
        (cx, cy), size = P(e["x"], e["y"]), e["size"] * s
        col, w = _TONE_RGB[e["tone"]], max(2, int(round(size * 0.13)))
        x, top_y = cx - size / 2, cy - size / 2
        if e["mark"] == "check":
            c.d.line([(x, top_y + size * 0.55), (x + size * 0.38, top_y + size * 0.9), (x + size, top_y + size * 0.1)],
                     fill=col, width=w, joint="curve")
        else:
            c.d.line([(x + size * 0.1, top_y + size * 0.1), (x + size * 0.9, top_y + size * 0.9)], fill=col, width=w)
            c.d.line([(x + size * 0.9, top_y + size * 0.1), (x + size * 0.1, top_y + size * 0.9)], fill=col, width=w)
    # what lies under a text: the smallest filled shape that holds it (a "none" shape shows what is under it), the
    # text's own shape first; white text only on a solid fill, and never for a label, whose white halo it would vanish on
    filled = sorted(((D._rect_of(e), e.get("fill")) for e in els if e["type"] in ("box", "circle") and e.get("fill") in ("tint", "solid")),
                    key=lambda t: (t[0][2] - t[0][0]) * (t[0][3] - t[0][1]))

    def under(i: int, rect) -> str:
        e = els[i]
        if e["type"] in ("box", "circle") and e.get("fill") in ("tint", "solid"):
            return e["fill"]
        return next((fill for r, fill in filled if D._inside(rect, r)), "none")
    for p in placed:   # texts last, over everything
        e = els[p.i]
        size = max(c.min_px, p.size * s)
        on_solid = p.role != "label" and under(p.i, p.rect if p.role in ("text", "label") else D._rect_of(e)) == "solid"
        col = (255, 255, 255) if on_solid else _TONE_RGB[p.tone]
        x1, y1 = P(p.rect[0], p.rect[1])
        x2, y2 = P(p.rect[2], p.rect[3])
        if p.role in ("box", "circle"):   # clipped to the shape's inside, not to the width measured at another size
            r = D._rect_of(e)
            k = 1.0 if e["type"] == "box" else 0.72
            inner = (r[2] - r[0]) * k * s - 2 * 0.2 * size
            lines = [_clip(c.pen, l, max(1.0, (x2 - x1) * 1.02 + 2, inner), size, p.bold) for l in p.lines]
        else:
            lines = p.lines
        if p.role == "label":   # a white halo keeps a label readable where it sits on a line
            pad = size * 0.12
            c.d.rectangle((x1 - pad, y1, x2 + pad, y2), fill=(255, 255, 255))
        c.block(lines, (x1 + x2) / 2, y1, size, p.bold, col, "center")
    outer.img.paste(c.img, (int(outer.m), int(top)))


_RENDERERS = {"bar": _render_bar, "table": _render_table, "flow": _render_flow, "compare": _render_compare,
              "list": _render_list, "links": _render_links, "diagram": _render_diagram}
_SEEDS = {"bar": 4, "table": 6, "flow": 7, "compare": 8, "list": 9, "links": 5, "diagram": 3}


def render_board(spec, language: str, size: tuple[int, int], fonts: Optional[BoardFonts] = None) -> Image.Image:
    """The board drawn on white at exactly ``size`` (marker style: wobbly ink strokes, red for the paper's result, blue
    for the best rival, grey for the rest). Fonts scale with the size and every text shrinks, wraps or (last resort) is
    ellipsized to stay inside it. ``fonts`` is any BoardFonts (lettering.FontSet); the default needs no config.
    Raises ValueError for a spec normalize_board() cannot use."""
    board = normalize_board(spec)
    if board is None:
        raise ValueError("render_board: unusable board spec: " + "; ".join(board_problems(spec)))
    for path, text in board_texts(board):   # printed, not coded: 10^19 → 10¹⁹, O(n^2) → O(n²), d_k → dₖ, 1e-4 → 1 × 10⁻⁴
        _set_path(board, path, display_value(text))
    W, H = max(1, int(size[0])), max(1, int(size[1]))
    img = Image.new("RGB", (W * SS, H * SS), (255, 255, 255))
    if W >= 24 and H >= 24:
        lang = "zh" if str(language).lower().startswith("zh") else "en"
        c = _Ctx(img, _Pen(fonts or default_fonts(), lang), _SEEDS[board["kind"]])
        _RENDERERS[board["kind"]](c, board)
    return img.resize((W, H), Image.Resampling.LANCZOS)
