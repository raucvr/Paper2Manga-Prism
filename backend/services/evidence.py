"""
Evidence — deterministic checks that every number a reader sees comes from the paper.

verify_board(): each value on a bar or table board must be supported by the paper, either (a) through the ledger cell it
was bound to (services.ledger: a table read from the PDF layout, appended to the paper text), (b) through its quote — a
verbatim snippet of the paper that contains the value ("…" may elide words) — or (c) through a paper line that holds
the value next to most of the words of its label (the row check). Every other number drawn on any board (titles, steps,
list items, a bare "+2.04", ...) must occur in the paper. verify_text_numbers() does the same for bubbles and narration.

Numbers are compared as the paper writes them, not as quantities: 26.30 matches 26.3 and 1.2 × 10²¹ matches
"1.2 · 10^21", but a converted value (120 with a unit of 10^19 FLOPs for 1.2 · 10^21) or a rounded one (26.4 for 26.36)
never does. Pure functions; no network.
"""

from __future__ import annotations

import math
import re
from decimal import Decimal
from functools import lru_cache
from typing import Iterable, NamedTuple, Union

from services.boards import _prep, _scan, board_numbers, board_texts, find_numbers, normalize_board
from services.ledger import ledger_of
from services.paper_figures import without_figures

_QUOTES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'", "\u2032": "'", "\u00b4": "'", "`": "'",
                         "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u2033": '"'})
_WORD_RE = re.compile(r"[a-z0-9]+|[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_ELISION_RE = re.compile(r"\[\s*(?:\.\s*){3}\]|(?:\.\s*){3,}|…")
# words that say nothing about which row a value belongs to (metrics, editorial words like "prior best")
STOP_WORDS = frozenset("""
a an the of and or to in on at for with by from as vs versus via per
model models method methods system systems approach result results score scores value values
test dev set accuracy acc bleu f1 em ppl perplexity metric
prior previous best sota state art baseline baselines ours our new old top
""".split())


def normalize_text(text: str) -> str:
    """Comparable form of paper text and quotes: NFKC, plain dashes and quotes, hyphenated line breaks joined, hyphens
    inside words dropped (the extractor writes "Englishto-German" for "English-to-German"), whitespace collapsed,
    lower case. Numbers keep their characters (find_numbers() canonicalizes them)."""
    s = _prep(str(text)).translate(_QUOTES)
    s = re.sub(r"(?<=[A-Za-z])-[ \t]*\n[ \t]*(?=[A-Za-z])", "", s)
    s = re.sub(r"(?<=[A-Za-z])-(?=[A-Za-z])", "", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def _stream(norm: str) -> list[str]:
    """Words and numbers of normalized text; numbers as "#<canonical value>" so "26.30" equals "26.3"."""
    out, pos = [], 0
    for n in _scan(norm):
        out += _WORD_RE.findall(norm[pos:n.start])
        out.append("#" + n.key)
        pos = n.end
    return out + _WORD_RE.findall(norm[pos:])


class _Line(NamedTuple):
    text: str          # normalized
    words: frozenset
    keys: frozenset    # canonical values of the numbers on the line
    table: bool        # laid out in columns (a table row): its label must be on the same line


class _Paper:
    def __init__(self, text: str):
        self.stream = _stream(normalize_text(text))
        self.starts: dict[str, list[int]] = {}
        for i, tok in enumerate(self.stream):
            self.starts.setdefault(tok, []).append(i)
        self.keys = frozenset(t[1:] for t in self.stream if t.startswith("#"))
        self.lines = []
        for raw in str(text).splitlines():
            if not raw.strip():
                continue
            norm = normalize_text(raw)
            nums = _scan(norm)
            gaps = len(re.findall(r"\S {3,}(?=\S)", raw.strip()))
            self.lines.append(_Line(norm, frozenset(_WORD_RE.findall(norm)), frozenset(n.key for n in nums),
                                    gaps >= 2 or (gaps >= 1 and len(nums) >= 2)))

    def has(self, tokens: list[str]) -> bool:
        """The token sequence occurs contiguously in the paper."""
        if not tokens:
            return False
        k = len(tokens)
        return any(self.stream[i:i + k] == tokens for i in self.starts.get(tokens[0], ()))


@lru_cache(maxsize=4)
def _paper(text: str) -> _Paper:
    """The paper as evidence: its own text, never the figures section (services.paper_figures), a vision model's reading
    whose values read off an axis must never pass as the paper's numbers."""
    return _Paper(without_figures(text))


def _quote_keys(paper: _Paper, quote: str) -> tuple[bool, frozenset]:
    """(the quote occurs in the paper, canonical values of its numbers). A quote must carry words, not only numbers;
    an elided quote ("Our model achieves 28.4 BLEU … by over 2 BLEU") must occur piece by piece, each with words."""
    pieces = [toks for toks in (_stream(normalize_text(x)) for x in _ELISION_RE.split(quote or "")) if toks]
    if not pieces:
        return False, frozenset()
    for toks in pieces:
        if not any(not t.startswith("#") and (len(t) >= 2 or not t.isascii()) for t in toks) or not paper.has(toks):
            return False, frozenset()
    return True, frozenset(t[1:] for toks in pieces for t in toks if t.startswith("#"))


def quote_in_paper(quote: str, paper_text: str) -> bool:
    """True when the quote occurs in the paper (ignoring case, spacing, punctuation, dashes and line-break hyphens)."""
    return _quote_keys(_paper(paper_text), quote)[0]


def _label_words(label: str) -> list[str]:
    norm = normalize_text(label)
    for n in reversed(_scan(norm)):
        norm = norm[:n.start] + " " + norm[n.end:]
    return list(dict.fromkeys(w for w in _WORD_RE.findall(norm) if w not in STOP_WORDS and not w.isdigit()))


def _row_supported(paper: _Paper, key: str, words: list[str], others: list[str]) -> bool:
    """Some paper line holds the value next to ≥60% of the label's words (a table row: on that line; prose: that line ±1).
    A label without words falls back to the row's other numbers standing on the same line."""
    need = math.ceil(0.6 * len(words))
    for i, line in enumerate(paper.lines):
        if key not in line.keys:
            continue
        span = [line] if line.table else paper.lines[max(0, i - 1): i + 2]
        if words:
            text = " ".join(l.text for l in span)
            vocab = frozenset().union(*(l.words for l in span))
            hits = sum(1 for w in words if (w in text if len(w) >= 3 else w in vocab))
            if hits >= need:
                return True
        elif others:
            keys = frozenset().union(*(l.keys for l in span))
            if all(o in keys for o in others):
                return True
    return False


def _where(board: dict, path: str) -> str:
    parts = path.split(".")
    kind = board["kind"]
    if parts[0] == "bars":
        bar = board["bars"][int(parts[1])]
        return f'bar "{bar["label"] or int(parts[1]) + 1}"'
    if parts[0] == "rows":
        i, j = int(parts[1]), int(parts[3])
        return f'table row {i + 1} ("{board["rows"][i]["cells"][0]}") cell {j + 1}'
    if parts[0] in ("steps", "items", "columns", "tokens"):
        return f"{kind} {parts[0][:-1]} {int(parts[1]) + 1}"
    if parts[0] in ("left", "right"):
        return f"{parts[0]} {'label' if parts[1] == 'label' else 'item ' + str(int(parts[2]) + 1)}"
    return parts[0]


MAGNITUDE_RE = re.compile(r"\b(thousands?|millions?|billions?)\b|[x×]\s*10\s*\^\s*\{?(\d{1,2})\}?|\((?:in\s+)?([KMB])\)", re.I)
_MAGNITUDE_WORDS = {"thousand": 3, "million": 6, "billion": 9, "k": 3, "m": 6, "b": 9}


def _board_magnitude(board: dict) -> int:
    """A power of ten the chart states once for all its values ("Training steps (millions)", unit "x10^6 steps"), so a
    bar value of 31.0 can be read as the paper's 31.0 · 10^6; 0 when there is none."""
    said = " ".join(str(board.get(k) or "") for k in ("unit", "title", "subtitle", "note"))
    m = MAGNITUDE_RE.search(said)
    if not m:
        return 0
    word = (m.group(1) or m.group(3) or "").lower().rstrip("s")
    return int(m.group(2)) if m.group(2) else _MAGNITUDE_WORDS.get(word, 0)


def _bound_cell(board: dict, path: str) -> str:
    """The ledger cell a data value was bound to ("" when none): a bar's "cell", a table row's "cell_ids"."""
    parts = path.split(".")
    if parts[0] == "bars":
        return str(board["bars"][int(parts[1])].get("cell") or "")
    if parts[0] == "rows":
        ids = board["rows"][int(parts[1])].get("cell_ids") or []
        j = int(parts[3])
        return str(ids[j] or "") if j < len(ids) else ""
    return ""


def verify_board(spec, paper_text: str) -> list[str]:
    """Problems with the numbers of a board against the paper; [] when every number is supported. Messages say which
    value failed and how to fix it, for the script's fix round."""
    board = normalize_board(spec)
    if board is None:
        return ["the board spec is unusable, so its numbers cannot be checked (see board_problems)"]
    paper = _paper(paper_text)
    ledger = ledger_of(paper_text)
    problems: list[str] = []
    data_paths = set()
    magnitude = _board_magnitude(board) if board["kind"] == "bar" else 0
    for path, value, quote, label in board_numbers(board):
        data_paths.add(path)
        nums = find_numbers(value)
        if not nums:
            continue
        cell = ledger.cell(_bound_cell(board, path)) if ledger else None
        if cell is not None and {n.key for n in nums} <= {n.key for n in find_numbers(cell.value)}:
            continue   # copied by the program from a table cell the PDF layout holds
        quoted, quote_keys = _quote_keys(paper, quote)
        labels = [label]
        if path.startswith("bars."):
            labels.append(board["bars"][int(path.split(".")[1])]["label"])
        if path.startswith("rows."):
            row = board["rows"][int(path.split(".")[1])]["cells"]
            row_keys = [n.key for cell in row for n in find_numbers(cell)]
        else:
            row_keys = [n.key for n in nums]
        candidates = [_label_words(l) for l in dict.fromkeys(l for l in labels if l)] or [[]]   # no label: the row's numbers must stand together
        for n in nums:
            keys = [n.key] + ([_scaled_key(n.key, magnitude)] if magnitude and n.exponent is None else [])
            if any(k in quote_keys for k in keys):
                continue
            others = [k for k in row_keys if k != n.key]
            if any(_row_supported(paper, k, words, others) for k in keys for words in candidates):
                continue
            why = ("its quote is not in the paper" if quote and not quoted else
                   "its quote does not contain it" if quote else "it has no quote")
            beside = f'next to "{label}"' if label else "together with the row's other numbers"
            problems.append(f'{_where(board, path)}: the value {n.text} ("{value}") is not supported by the paper — {why}, '
                            f"and no paper line shows {n.text} {beside}. Copy the value character for character "
                            f"from the paper with a verbatim quote of its table row or sentence, or drop it.")
    # every other drawn text, bare numbers included ("+2.04" on a list is not translatable, but a reader sees it);
    # the tokens of an illustrative links board are the running example's own words, and a diagram's labels are its
    # designer's: StoryboarderV2 checks them when it designs (a made-up example's numbers allowed) and drops what is
    # still unsupported at the end. Here they made a whole drawing invalid over one label the writer cannot edit, and
    # the diagram became a scene (round 23's review)
    illustrative = board["kind"] == "links" and board.get("illustrative", True)
    drawn = board["kind"] == "diagram"
    texts = [(p, t) for p, t in board_texts(board) if p not in data_paths and not (illustrative and p.startswith("tokens."))
             and not (drawn and p.startswith("elements."))]
    for path, text in texts:
        for n in find_numbers(text):
            if magnitude and n.key == _scaled_key("1", magnitude):   # the chart's stated scale ("x10^6 steps")
                continue
            if not n.short and n.key not in paper.keys:
                problems.append(f'{_where(board, path)} "{text}": the number {n.text} is not in the paper. '
                                f"Use only numbers the paper states; never compute, convert or round them.")
    return problems


def _scaled_key(key: str, magnitude: int) -> str:
    return format(Decimal(key).scaleb(magnitude).normalize(), "f")


def verify_text_numbers(text: str, paper_text: str, allowed: Union[str, Iterable[str]] = ()) -> list[str]:
    """Numbers in a bubble or narration that the paper does not contain; single digits are ignored, and so are the numbers
    of ``allowed`` texts (the running example's own sentence when it is labelled illustrative)."""
    if not text:
        return []
    paper = _paper(paper_text)
    if isinstance(allowed, str):
        allowed = [allowed]
    ok = {n.key for a in allowed if a for n in find_numbers(a)}
    out: list[str] = []
    for n in find_numbers(text):
        if n.short or n.key in paper.keys or n.key in ok:
            continue
        msg = f"the number {n.text} is not in the paper"
        if msg not in out:
            out.append(msg)
    return out
