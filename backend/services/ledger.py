"""
Evidence ledger — the paper's tables, read from the PDF layout, as cells a board cites by id.

services.pdf_tables reads every table from the page geometry (header paths, rows, row blocks). The ledger appends them
to the paper text as one machine-readable section, so everything that reads the paper text sees the same tables: the
story and fix prompts, the fact-check, the evidence checks, and paper.txt for later relayouts and translations:

    === TABLES READ FROM THE PDF LAYOUT === (cite a cell as @T2.r9.c1: table T2, row 9, column 1)
    [T2] Table 2 (page 8): The Transformer achieves better BLEU scores than previous state-of-the-art models ...
    [T2] columns: c1 = BLEU › EN-DE | c2 = BLEU › EN-FR | c3 = Training Cost (FLOPs) › EN-DE
    [T2.r1]   ByteNet   c1: 23.75
    [T2] ---
    [T2.r9]   Transformer (big)   c1: 28.4   c2: 41.8   c3: 2.3 · 10^19

A board cites a cell instead of writing its number ("value": "@T2.r9.c1", a table cell "@T2.r9.c2", "@T2.r9.c1" in a
bubble). bind_board() / bind_text() copy the exact value, the row line as the quote and the row name as the paper label,
so a chart is drawn from the PDF's own numbers; a bar written as a plain number is bound to the one cell holding that
value next to its name. strongest_rival() finds the best earlier system of the same column and setting, which a result
chart must show. Pure functions over the paper text; no PDF access, no network.
"""

from __future__ import annotations

import copy
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable, Optional

from services.boards import _negative, _prep, _scan

LEDGER_HEAD = "=== TABLES READ FROM THE PDF LAYOUT ==="
HEAD_NOTE = "(cite a cell as @T2.r9.c1: table T2, row 9, column 1)"
PATH_SEP = " › "
MAX_SECTION_CHARS = 40_000

_TID = r"[A-Z]{1,3}\d{1,3}[a-z]?"
REF_RE = re.compile(rf"@({_TID})\.r(\d+)\.c(\d+)(?!\w|\.\d)")   # a citation inside a text, also before a full stop ("scores @T2.r2.c6.")
CELL_RE = re.compile(rf"^\s*@?\s*({_TID})\s*\.\s*[rR]\s*(\d+)\s*\.\s*[cC]\s*(\d+)\s*$")         # a value that is only a citation
_HEAD_RE = re.compile(rf"^\[({_TID})\] (.+?) \(page (\d+)\)(?:: (.*))?$")
_COLS_RE = re.compile(rf"^\[({_TID})\] columns: (.*)$")
_RULE_RE = re.compile(rf"^\[({_TID})\] ---$")
_ROW_RE = re.compile(rf"^\[({_TID})\.r(\d+)\]\s+(.*)$")
_CELL_SPLIT = re.compile(r"\s{2,}(?=c\d+: )")
_COL_ITEM = re.compile(r"^c(\d+) = (.*)$")
_CITATION_RE = re.compile(r"\s*\[\d+(?:\s*[,–-]\s*\d+)*\]")
_EMPTY = {"", "-", "–", "—", "−", "n/a", "N/A", "na", "NA"}
_WORD_RE = re.compile(r"[a-z0-9]+(?:[.+-][a-z0-9]+)*")

# words that say nothing about which system a row is ("BERT large" and "BERT base" are both BERT)
GENERIC = frozenset("""
a an the of and or with without w o for on in to by from via vs
model models method methods system systems approach baseline baselines ours our full proposed
single ensemble ens base large big small tiny medium huge xl xxl mini
standard prompting prompt finetuned fine tuned tuning pretrained trained training
et al
""".split())
_SIZE_RE = re.compile(r"^\d+(?:\.\d+)?[kmbt]?$")
ENSEMBLE_RE = re.compile(r"\bens(?:emble|embles|\.)?(?![a-z])", re.I)
HUMAN_RE = re.compile(r"\bhumans?\b|\bpeople\b|\bannotators?\b|\bexperts?\b", re.I)
OURS_RE = re.compile(r"\bours\b|\bour (?:model|method|approach|system)\b|\bproposed\b", re.I)
# a section heading row that names a setting ("Semi-supervised", "Zero-shot", "WSJ only"): rows under two different such
# headings are not compared
SETTING_RE = re.compile(r"supervis|\bsemi\b|zero[\s-]?shot|few[\s-]?shot|one[\s-]?shot|\d+[\s-]?shot|fine[\s-]?tun|\bwsj\b|"
                        r"multi[\s-]?task|generative|discriminative|\bdev\b|\btest\b|\bval(?:idation)?\b|in[\s-]domain|"
                        r"out[\s-]of[\s-]domain|pre[\s-]?train|from scratch|frozen|linear probe|extra data|additional data", re.I)
# a table of the paper's own variants (no earlier systems to beat): no rival is required
ABLATION_RE = re.compile(r"\bablations?\b|\bvariations?\b|\bvariants?\b|\beffect of\b|\bimpact of\b|\bsensitivity\b|"
                         r"\bwithout\b|\bw/o\b|\bcomponents?\b", re.I)


@dataclass
class Row:
    n: int                                    # 1-based, in the paper's order
    label: str
    cells: dict[int, str] = field(default_factory=dict)   # column number (1-based) → value as printed; none: a section heading
    group: int = 0                            # row block: blocks are separated by rules in the paper


@dataclass
class Table:
    id: str                                   # "T2", "TA1" (Table A.1), "F4" (a table captioned as Figure 4)
    label: str                                # "Table 2"
    page: int
    caption: str
    columns: list[str]                        # header paths ("BLEU › EN-DE"); column c(j+1) is columns[j]
    rows: list[Row] = field(default_factory=list)


@dataclass(frozen=True)
class Cell:
    table: str
    row: int
    col: int
    label: str                                # the row's name
    column: str                               # the column's header path
    value: str                                # as printed ("56.9 (+39.0)", "92.4 ± 0.3")
    primary: str                              # its first number as written, sign included ("56.9", "−0.9" → "-0.9")
    number: Optional[float]
    table_label: str
    group: int

    @property
    def id(self) -> str:
        return f"{self.table}.r{self.row}.c{self.col}"

    def describe(self) -> str:
        return f'@{self.id} = {self.primary} ({self.table_label}, "{self.label}", {self.column})'


def _clean(text) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _primary(value: str) -> tuple[str, Optional[float]]:
    """The first number of a cell as written ("56.9 (+39.0)" → "56.9", "−0.9" → "-0.9"); ("", None) for a text cell."""
    s = _prep(value)
    nums = _scan(s)
    if not nums:
        return "", None
    n = nums[0]
    try:
        number = float(n.key)
    except ValueError:
        return "", None
    if _negative(s, n.start):
        return "-" + n.text, -number
    return n.text, number


def name_words(text: str) -> set[str]:
    """Lower-case words of a row or bar name without citations ([18]) and generic words."""
    s = _CITATION_RE.sub(" ", _prep(str(text or ""))).lower().replace("_", " ")
    return {w for w in _WORD_RE.findall(s) if w not in GENERIC}


# rows of a regression table that are no systems: statistics ("Observations", "Bandwidth", "Adj. R²") and bins ("1–3")
STAT_ROW_RE = re.compile(r"^\s*(?:observations?|obs\.?|n|bandwidth|clusters?|adj\W*r\W*2|r\W*2|controls?|[\w\s-]*\bfe|"
                         r"(?:pre[\s-]?period|control|baseline|sample|dep(?:endent)?\.?\s*var(?:iable)?\.?)\s+mean)\s*$", re.I)
BIN_ROW_RE = re.compile(r"^\s*[<>≤≥]?\s*\d+(?:\.\d+)?\s*(?:[–—-]\s*\d+(?:\.\d+)?|\+)?\s*%?\s*$")


def _not_a_system(label: str) -> bool:
    text = unicodedata.normalize("NFKC", str(label or "")).replace("²", "2")
    return bool(STAT_ROW_RE.match(text) or BIN_ROW_RE.match(text))


def _distinct(text: str) -> set[str]:
    """The words that name a system: no sizes ("540b", "65b"), no generic words, no single letters ("(B)")."""
    return {w for w in name_words(text) if len(w) > 1 and not _SIZE_RE.match(w)}


def _continued(label: str, previous: str) -> str:
    """A row printed as an addition to the row above ("Codex | + ext. calc" under "Codex | Chain of thought") named in
    full ("Codex | Chain of thought + ext. calc"): it is the paper's method plus a calculator, not another system."""
    parts, before = label.split(" | "), previous.split(" | ")
    if len(parts) > 1 and parts[-1].startswith("+") and len(before) == len(parts) and before[:-1] == parts[:-1]:
        stem = before[-1].split(" + ")[0]
        return " | ".join(parts[:-1] + [f"{stem} {parts[-1]}"])
    return label


def _table_id(label: str, used: set[str]) -> str:
    m = re.match(r"^\s*(Table|Tab\.?|Figure|Fig\.?)\s*([A-Z]?)\s*\.?\s*(\d{1,3})", label or "", re.I)
    base = (("F" if m.group(1).lower().startswith("fig") else "T") + m.group(2).upper() + m.group(3)) if m else f"X{len(used) + 1}"
    tid, k = base, 0
    while tid in used:
        k += 1
        tid = base + "bcdefghijk"[min(k - 1, 9)]
    used.add(tid)
    return tid


class Ledger:
    """The tables of one paper; empty (falsy) when the PDF had none that could be read reliably."""

    def __init__(self, tables: Optional[list[Table]] = None):
        self.tables = list(tables or [])
        self._by_id = {t.id: t for t in self.tables}

    def __bool__(self) -> bool:
        return bool(self.tables)

    # ------------------------------------------------------------------ building

    @classmethod
    def from_tables(cls, raw: Iterable[dict], min_confidence: float = 0.0) -> "Ledger":
        """From services.pdf_tables output ({"label", "page", "caption", "columns": [[header path]], "rows": [{"label",
        "group", "cells"}], "confidence"}): tables below ``min_confidence`` and tables without a number are left out."""
        used: set[str] = set()
        tables = []
        for t in raw or []:
            if not isinstance(t, dict) or float(t.get("confidence", 1.0) or 0.0) < min_confidence:
                continue
            columns = []
            for j, path in enumerate(t.get("columns") or []):
                parts = [_clean(x).replace("|", "/") for x in (path if isinstance(path, (list, tuple)) else [path])]
                columns.append(PATH_SEP.join(x for x in parts if x) or f"column {j + 1}")
            rows = []
            for r in t.get("rows") or []:
                if not isinstance(r, dict):
                    continue
                cells = {}
                for j, v in enumerate(r.get("cells") or []):
                    v = _clean(v)
                    if v not in _EMPTY and j < len(columns):
                        cells[j + 1] = v
                label = _clean(_CITATION_RE.sub("", str(r.get("label") or "")))
                if not cells and not label:
                    continue
                label = _continued(label, rows[-1].label if rows else "")
                # a row with a name and no values is a section heading ("Ours", "Semi-supervised"): kept, it tells
                # rivals from variants and settings apart
                rows.append(Row(n=len(rows) + 1, label=label or f"(row {len(rows) + 1})", cells=cells,
                                group=int(r.get("group") or 0)))
            if not rows or not any(_primary(v)[1] is not None for r in rows for v in r.cells.values()):
                continue
            label = _clean(t.get("label")) or "Table"
            tables.append(Table(id=_table_id(label, used), label=label, page=int(t.get("page") or 0),
                                caption=_clean(t.get("caption"))[:240], columns=columns, rows=rows))
        return cls(tables)

    # ------------------------------------------------------------------ text form

    @staticmethod
    def row_text(t: Table, r: Row) -> str:
        cells = "   ".join(f"c{j}: {v}" for j, v in sorted(r.cells.items()))
        return f"[{t.id}.r{r.n}]   {r.label}" + (f"   {cells}" if cells else "")

    def format(self, max_chars: int = MAX_SECTION_CHARS) -> str:
        """The section appended to the paper text ("" without tables); whole tables only, in the paper's order, until
        ``max_chars``."""
        if not self.tables:
            return ""
        out = [f"{LEDGER_HEAD} {HEAD_NOTE}"]
        size = len(out[0])
        for t in self.tables:
            caption = t.caption if not t.caption.lower().startswith(t.label.lower()) else t.caption[len(t.label):].lstrip(" .:")
            lines = ["", f"[{t.id}] {t.label} (page {t.page})" + (f": {caption}" if caption else ""),
                     f"[{t.id}] columns: " + " | ".join(f"c{j + 1} = {c}" for j, c in enumerate(t.columns))]
            group = t.rows[0].group if t.rows else 0
            for r in t.rows:
                if r.group != group:
                    lines.append(f"[{t.id}] ---")
                    group = r.group
                lines.append(self.row_text(t, r))
            block = "\n".join(lines)
            if size + len(block) > max_chars:
                break
            out.append(block)
            size += len(block)
        return "\n".join(out) if len(out) > 1 else ""

    @classmethod
    def parse(cls, text: str) -> "Ledger":
        return _parse(str(text or ""))

    # ------------------------------------------------------------------ lookups

    def table(self, tid: str) -> Optional[Table]:
        return self._by_id.get(tid)

    def _cell(self, t: Table, r: Row, col: int) -> Optional[Cell]:
        value = r.cells.get(col)
        if value is None or not 1 <= col <= len(t.columns):
            return None
        primary, number = _primary(value)
        return Cell(t.id, r.n, col, r.label, t.columns[col - 1], value, primary, number, t.label, r.group)

    def cell(self, ref) -> Optional[Cell]:
        """The cell a citation names ("@T2.r9.c1", "T2.r9.c1", "t2.R9.C1" all work); None when there is none."""
        m = CELL_RE.match(str(ref or "")) or REF_RE.search(str(ref or ""))
        if not m:
            return None
        tid = m.group(1)
        t = self._by_id.get(tid) or self._by_id.get(tid.upper())
        if t is None:
            return None
        r = next((x for x in t.rows if x.n == int(m.group(2))), None)
        return self._cell(t, r, int(m.group(3))) if r else None

    def row_line(self, tid: str, n: int) -> str:
        t = self._by_id.get(tid)
        r = next((x for x in t.rows if x.n == n), None) if t else None
        return self.row_text(t, r) if r else ""

    def column(self, tid: str, col: int) -> list[Cell]:
        t = self._by_id.get(tid)
        return [c for r in t.rows if (c := self._cell(t, r, col))] if t else []

    def cells(self) -> list[Cell]:
        return [c for t in self.tables for r in t.rows for j in sorted(r.cells) if (c := self._cell(t, r, j))]

    def cells_with(self, key: str) -> list[Cell]:
        """Number cells whose first number is ``key`` (a canonical value: services.boards Number.key)."""
        out = []
        for c in self.cells():
            if c.number is not None and _scan(_prep(c.primary.lstrip("-")))[0].key == key:
                out.append(c)
        return out

    def _settings(self, t: Table, r: Row) -> dict[int, str]:
        """The row's text cells (a "Training" column saying "WSJ only" or "semi-supervised"): its setting."""
        return {j: v.lower() for j, v in r.cells.items() if _primary(v)[1] is None}

    @staticmethod
    def _sections(t: Table) -> dict[int, str]:
        """Row number → the section heading above it ("" when none): the nearest earlier row without values."""
        out, current = {}, ""
        for r in t.rows:
            if not r.cells:
                current = r.label
            out[r.n] = current
        return out

    # ------------------------------------------------------------------ binding

    def match(self, value, labels: Iterable, prefer: Optional[tuple[str, int]] = None) -> Optional[Cell]:
        """The one number cell holding ``value`` whose row is named by one of ``labels`` (≥60% of the label's words):
        how a bar written as a plain number finds its cell. A cell in the ``prefer``red (table, column) wins; None when
        no cell or several equally good cells qualify. Within the preferred column only, a value read off a figure
        ("57" for the table's 56.9) finds the row's cell when it is within max(1, 2%) of it: never across columns."""
        nums = _scan(_prep(str(value or "")))
        if len(nums) != 1:
            return None
        key = nums[0].key
        names = [w for w in (name_words(x) for x in labels if x) if w]
        if not names:
            return None
        def share(c: Cell) -> float:
            row = name_words(c.label)
            return max(len(n & row) / len(n) for n in names)
        scored = [((prefer == (c.table, c.col)), share(c), c) for c in self.cells_with(key)]
        scored = [x for x in scored if x[1] >= 0.6]
        if not scored and prefer is not None:
            target = float(key)
            near = [(abs(c.number - target), -share(c), c) for c in self.column(*prefer)
                    if c.number is not None and share(c) >= 0.6 and abs(c.number - target) <= max(1.0, 0.02 * abs(target))]
            near.sort(key=lambda x: (x[0], x[1]))
            if near and (len(near) == 1 or near[1][:2] != near[0][:2]):
                return near[0][2]
            return None
        if not scored:
            return None
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        top = scored[0]
        if len(scored) > 1 and scored[1][:2] == top[:2] and scored[1][2].id != top[2].id:
            return None
        return top[2]

    def strongest_rival(self, ref, higher_is_better: bool = True, context: str = "") -> Optional[Cell]:
        """The best earlier system in the cited cell's column that the chart must show next to it: same column, same
        setting (text cells of the row), not a variant of the cited system (a row sharing a name word with it, or both
        "ours"), no ensembles unless the cited row is one or ``context`` (the board's words) speaks of ensembles, no
        human rows unless ``context`` speaks of humans. None for a table of the paper's own variants (an ablation) and
        when no row qualifies."""
        me = self.cell(ref)
        if me is None or me.number is None:
            return None
        t = self._by_id[me.table]
        if ABLATION_RE.search(t.caption):
            return None
        # the reader can write the setting into the row name ("Ours | WSJ only, discriminative"): the system is the part
        # before it, and a rival's setting must be the same
        tail = lambda label: label.rsplit(" | ", 1)[1].lower().strip() if " | " in label and SETTING_RE.search(label.rsplit(" | ", 1)[1]) else ""
        system = lambda label: label.rsplit(" | ", 1)[0] if tail(label) else label
        if _not_a_system(me.label):
            return None   # a statistics row ("Observations") or a bin ("1–3") of a regression table
        mine, ours = _distinct(system(me.label)), bool(OURS_RE.search(me.label))
        my_setting = tail(me.label)
        if not mine and not ours:
            return None   # the cited row has no name to tell its variants from rivals ("base", "(A)")
        ensembles = bool(ENSEMBLE_RE.search(me.label) or ENSEMBLE_RE.search(context))
        humans = bool(HUMAN_RE.search(context))
        rows = {r.n: r for r in t.rows}
        setting = self._settings(t, rows[me.row])
        sections = self._sections(t)
        home = sections.get(me.row, "")
        # a table that repeats its systems per block (a section heading and its rule group: one per base model or bit
        # width): the rival comes from the cited row's block (OnQ's Qwen3-8B chart was asked for a Qwen3-1.7B rival)
        block = lambda r: (sections.get(r.n, ""), r.group)
        blocks: dict = {}
        for r in t.rows:
            if r.cells:
                blocks.setdefault(block(r), []).append(r)
        same = lambda r: bool(mine & _distinct(system(r.label))) or (ours and bool(OURS_RE.search(r.label)))   # the cited system again
        names = lambda r: {" ".join(r.label.lower().split())} | _distinct(r.label)
        # the cited system's own words never make two blocks parallel (a table that lists the base model it builds on)
        here = set().union(*(names(r) for r in blocks.get(block(rows[me.row]), []) if r.n != me.row)) - mine
        # a block is parallel only when it repeats the cited system and shares its rivals' names (one shared word is no block)
        parallel = {k for k, rs in blocks.items() if k != block(rows[me.row]) and any(same(r) for r in rs)
                    and here & set().union(*(names(r) for r in rs))}
        # rows under another section heading that repeats the cited system (OnQ's per-model "Full precision | 16/16" groups)
        under = lambda sec: set().union(*(names(r) for r in t.rows if r.cells and r.n != me.row and sections.get(r.n, "") == sec)) - mine
        base = lambda r: _distinct(OURS_RE.sub(" ", system(r.label)))
        plug = lambda r: any(base(r) & _distinct(x.label) for x in t.rows
                             if x.cells and x.n != r.n and not OURS_RE.search(x.label) and sections.get(x.n, "") == sections.get(r.n, ""))
        repeated = {sec for sec in {sections.get(r.n, "") for r in t.rows if r.cells and r.n != me.row and same(r)} - {"", home}
                    if under(home) & under(sec) or (ours and any(OURS_RE.search(r.label) and plug(r) == plug(rows[me.row]) for r in t.rows
                                                                  if r.cells and r.n != me.row and sections.get(r.n, "") == sec))}
        best = None
        for c in self.column(me.table, me.col):
            if c.row == me.row or c.number is None:
                continue
            if c.row in rows and block(rows[c.row]) in parallel or _not_a_system(c.label):
                continue
            if mine & _distinct(system(c.label)) or (ours and OURS_RE.search(c.label)):
                continue
            if my_setting and tail(c.label) and tail(c.label) != my_setting:
                continue   # "X | WSJ only" against "Y | semi-supervised"
            there = sections.get(c.row, "")
            if there in repeated:
                continue
            if home and there == home and OURS_RE.search(home):
                continue   # another row of the paper's own section ("Ours")
            if home != there and home and there and SETTING_RE.search(home) and SETTING_RE.search(there):
                continue   # rows under two different setting headings ("Semi-supervised" vs "WSJ only")
            if (ENSEMBLE_RE.search(c.label) and not ensembles) or (HUMAN_RE.search(c.label) and not humans):
                continue
            other = self._settings(t, rows[c.row])
            if any(setting.get(j) != other.get(j) for j in set(setting) & set(other)):
                continue
            if best is None or (c.number > best.number if higher_is_better else c.number < best.number):
                best = c
        return best


@lru_cache(maxsize=16)
def _parse(text: str) -> Ledger:
    start = text.find(LEDGER_HEAD)
    if start < 0:
        return Ledger()
    tables: dict[str, Table] = {}
    order: list[str] = []
    groups: dict[str, int] = {}
    for raw in text[start:].splitlines()[1:]:
        line = raw.strip()
        if not line:
            continue
        if m := _HEAD_RE.match(line):
            tid = m.group(1)
            if tid not in tables:
                order.append(tid)
            tables[tid] = Table(id=tid, label=m.group(2), page=int(m.group(3)), caption=m.group(4) or "", columns=[])
            groups[tid] = 0
        elif (m := _COLS_RE.match(line)) and m.group(1) in tables:
            cols: dict[int, str] = {}
            for item in m.group(2).split(" | "):
                if c := _COL_ITEM.match(item.strip()):
                    cols[int(c.group(1))] = c.group(2).strip()
            tables[m.group(1)].columns = [cols.get(j, f"column {j}") for j in range(1, max(cols, default=0) + 1)]
        elif (m := _RULE_RE.match(line)) and m.group(1) in tables:
            groups[m.group(1)] += 1
        elif (m := _ROW_RE.match(line)) and m.group(1) in tables:
            parts = _CELL_SPLIT.split(m.group(3).strip())
            cells = {}
            for part in parts[1:]:
                key, _, value = part.partition(": ")
                if key[1:].isdigit():
                    cells[int(key[1:])] = value.strip()
            tables[m.group(1)].rows.append(Row(n=int(m.group(2)), label=parts[0].strip(), cells=cells, group=groups[m.group(1)]))
        else:
            break   # the section ends at the first line that is not part of it
    return Ledger([tables[t] for t in order if tables[t].rows])


def ledger_of(paper_text: str) -> Ledger:
    """The ledger appended to a paper text (empty without one)."""
    return _parse(str(paper_text or ""))


def section(paper_text: str) -> str:
    """The ledger section of a paper text, as written ("" without one): for prompts that leave the paper out."""
    text = str(paper_text or "")
    start = text.find(LEDGER_HEAD)
    return text[start:].strip() if start >= 0 else ""


def with_ledger(body: str, ledger: Ledger, max_chars: int = MAX_SECTION_CHARS) -> str:
    """The paper text with the ledger section appended (an older section replaced)."""
    start = body.find(LEDGER_HEAD)
    if start >= 0:
        body = body[:start].rstrip()
    block = ledger.format(max_chars)
    return f"{body}\n\n{block}" if block else body


# ----------------------------------------------------------------------------- binding boards and texts

def bind_text(text: str, ledger: Ledger) -> str:
    """A bubble or narration with every resolvable citation replaced by the cell's number ("@T2.r9.c1" → "28.4");
    a citation that resolves to nothing stays visible, so lint and the final scrub catch it."""
    if not text or "@" not in text or not ledger:
        return text
    def sub(m: re.Match) -> str:
        c = ledger.cell(m.group(0))
        return c.primary if c is not None and c.number is not None else m.group(0)
    return REF_RE.sub(sub, text)


def unresolved(text: str, ledger: Ledger) -> list[str]:
    """Citations in a text that name no number cell."""
    return [m.group(0) for m in REF_RE.finditer(str(text or "")) if not ((c := ledger.cell(m.group(0))) and c.number is not None)]


def _bind_bar(bar: dict, cell: Cell, ledger: Ledger) -> None:
    bar["cell"] = cell.id
    bar["value"] = cell.primary
    bar["quote"] = ledger.row_line(cell.table, cell.row)
    bar["paper_label"] = cell.label
    if not str(bar.get("label") or "").strip():
        bar["label"] = cell.label


def bind_board(spec, ledger: Ledger):
    """A copy of a board spec (as the script model wrote it, before normalization) with its citations resolved:
    a bar whose "value" or "cell" cites a cell gets the cell's number, the row line as its quote and the row name as its
    paper label; a bar written as a plain number is bound to the one cell holding it next to its name (preferring the
    column most bars cite); a table cell that is a citation becomes the cell's text (the row remembers which cells are
    bound in "cell_ids"); other texts get their citations replaced. Citations that name nothing are listed in the
    board's "unbound" (lint reports them) and the bar or cell is left out. The source names the table when every bound
    value comes from one table. No ledger: the spec unchanged."""
    if not isinstance(spec, dict) or not ledger:
        return spec
    spec = copy.deepcopy(spec)
    unbound: list[str] = []
    tables: set[str] = set()
    bars = spec.get("bars")
    if isinstance(bars, list):
        kept = []
        for b in bars:
            if not isinstance(b, dict):
                kept.append(b)
                continue
            ref = b.get("cell") or (b.get("value") if isinstance(b.get("value"), str) and CELL_RE.match(b["value"]) else None)
            if ref:
                c = ledger.cell(ref)
                if c is None or c.number is None:
                    unbound.append(f'{str(ref).strip()} (bar "{_clean(b.get("label"))}")')
                    continue
                _bind_bar(b, c, ledger)
            kept.append(b)
        bound = [ledger.cell(b["cell"]) for b in kept if isinstance(b, dict) and b.get("cell")]
        common = Counter((c.table, c.col) for c in bound if c).most_common(1)
        prefer = common[0][0] if common else None
        for b in kept:
            if isinstance(b, dict) and not b.get("cell") and b.get("value") is not None:
                c = ledger.match(b.get("value"), [b.get("paper_label"), b.get("label")], prefer)
                if c is not None:
                    _bind_bar(b, c, ledger)
        spec["bars"] = kept
        tables |= {ledger.cell(b["cell"]).table for b in kept if isinstance(b, dict) and b.get("cell")}
    rows = spec.get("rows")
    if isinstance(rows, list):
        for i, r in enumerate(rows):
            cells = r.get("cells") if isinstance(r, dict) else r if isinstance(r, list) else None
            if not isinstance(cells, list):
                continue
            ids: list[Optional[str]] = [None] * len(cells)
            for j, v in enumerate(cells):
                if isinstance(v, str) and CELL_RE.match(v):
                    c = ledger.cell(v)
                    if c is None:
                        unbound.append(f"{v.strip()} (table row {i + 1})")
                        cells[j] = ""
                        continue
                    cells[j], ids[j] = c.value, c.id
                    tables.add(c.table)
                elif isinstance(v, str):
                    cells[j] = bind_text(v, ledger)
            if any(ids) and isinstance(r, dict):
                r["cell_ids"] = ids
                source_rows = {ledger.cell(x).row for x in ids if x} if len({x.split(".")[0] for x in ids if x}) == 1 else set()
                if len(source_rows) == 1:
                    first = next(x for x in ids if x)
                    r["quote"] = ledger.row_line(first.split(".")[0], source_rows.pop())
    for key in ("title", "subtitle", "note", "unit"):
        if isinstance(spec.get(key), str):
            spec[key] = bind_text(spec[key], ledger)
    for key in ("steps", "items"):
        if isinstance(spec.get(key), list):
            spec[key] = [bind_text(x, ledger) if isinstance(x, str) else x for x in spec[key]]
    for side in ("left", "right"):
        if isinstance(spec.get(side), dict) and isinstance(spec[side].get("items"), list):
            spec[side]["items"] = [bind_text(x, ledger) if isinstance(x, str) else x for x in spec[side]["items"]]
    if len(tables) == 1:   # the values come from this table, whatever the model wrote ("Figure 2" for Table 1's numbers)
        t = ledger.table(next(iter(tables)))
        if t is not None and t.label.lower() not in str(spec.get("source") or "").lower():
            spec["source"] = t.label
    if unbound:
        spec["unbound"] = unbound
    else:
        spec.pop("unbound", None)
    return spec
