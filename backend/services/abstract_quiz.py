"""
Abstract quiz: the v3 story gate (contract "Story layer", step 7).

The owner's bar: after the manga a reader knows at least what the paper's abstract says, and has seen the key numbers
as evidence. This module measures exactly that on the visible manga. A reader model that never saw the paper answers
short questions about the abstract from transcript(sb) alone, citing panel numbers; a grader scores each answer
strictly against a gold answer taken from the abstract claims; claims whose questions fail go back to the writer
through revise_prompt.

What keeps the score honest:
- only visible text counts (bubbles, narration, and the boards exactly as services.boards draws them, delta badge and
  footnotes included), never panel.detail or the paper card: v2's coverage counted those and reported 10/10 while
  abstract facts were missing from the pictures;
- a correct answer must trace to a cited panel that contains it: a number the cited panels do not show caps the score
  at 0.5, so a famous paper's numbers recalled from memory (or a wrong citation) do not pass;
- no yes/no questions (a guess would pass them); the reader never sees the gold answers; every call runs at
  temperature 0 in JSON mode.
"""

from __future__ import annotations

import asyncio

import dataclasses
import json
import re
import unicodedata
from typing import Optional

from engines.base import BaseEngine, GenerationConfig
from services import storyboarder_v2 as _story
from services.storyboarder_v2 import (LANG_NAMES, Cast, PanelV2, StoryboardV2, _as_list, _ask_json, _claim_ids, _content_tokens,
                                      _optional, _prompt_board, _v3_budget, _v3_rules, split_sentences)

PASS_SCORE = 0.85   # below this the storyboarder revises the script (research target: abstract quiz >= 85%)
MAX_CITED = 4       # panels per answer that count as its citation: citing half the manga proves nothing
NOT_SHOWN = "not shown"


# ----------------------------------------------------------------------------- transcript (pure)

BOARD_NAMES = {"bar": "bar chart", "table": "table", "flow": "flow diagram", "compare": "comparison", "list": "checklist",
               "links": "word links", "diagram": "diagram"}
# what boards.render_board prints besides the spec's own strings: a bar chart's footnote (unit · direction · what the
# badge measures, as in boards._bar_caption), a recap's legend and a links board's tag
DIRECTION_RE = re.compile(r"\b(?:higher|lower|smaller|larger|bigger|less|more|fewer)\s+is\s+better\b|越(?:高|低|小|大|少|多)越好", re.I)
BOARD_WORDS = {"en": {"higher": "higher is better", "lower": "lower is better", "illustrative": "illustration (example weights)",
                      "checked": "tick: a chart in this manga shows it", "unit": "unit: {}", "vs": "{} vs {}"},
               "zh": {"higher": "越高越好", "lower": "越低越好", "illustrative": "示意图（权重为举例）",
                      "checked": "打勾：漫画里有图表展示", "unit": "单位：{}", "vs": "{} 相对 {}"}}
# a links board draws a link whose (normalized) weight reaches LINK_STRONG strong (red), one that reaches LINK_SHOWN thin,
# weaker ones not at all; services.boards owns that scale (boards.LINK_STRONG / LINK_SHOWN, applied to the weights
# normalize_board returns), these stand in only when that module is missing
LINK_STRONG, LINK_SHOWN = 0.5, 0.03


def _line(value) -> str:
    """Visible text on one line ("" for None): a newline inside a bubble must not break the transcript's panel blocks."""
    return " ".join(str(value).split()) if value is not None else ""


def _true(value) -> bool:
    return value is True or str(value).strip().lower() in ("true", "yes", "1")


def _indexes(value) -> set[int]:
    out = set()
    for x in _as_list(value):
        try:
            out.add(int(x))
        except (TypeError, ValueError):
            continue
    return out


def _weight(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _drawn_board(board: dict, boards) -> tuple[Optional[dict], Optional[dict]]:
    """(the spec as render_board draws it, the delta badge it adds) from services.boards: aliases resolved, unusable bars
    and rows dropped, defaults filled in, link weights on its 0-1 scale; spec None when nothing can be drawn. Without
    that module the raw spec is read as it is."""
    if boards is None:
        return board, None
    try:
        spec = boards.normalize_board(board)
        return spec, (boards.delta_badge(spec) if spec else None)
    except Exception as e:  # a bug over there must not take the gate down; the raw spec is close to what is drawn
        print(f"[abstract_quiz] WARNING: board read as written ({type(e).__name__}: {e})")
        return board, None


def _board_lines(board: dict, language: str) -> list[str]:
    """A board as the text a reader sees on it: title, subtitle, every bar with its value (and the gap badge next to the
    paper's bar), table columns and rows, flow steps, both sides of a comparison, list items, link tokens with the links
    drawn, then the grey footnotes; [highlighted] marks what the board highlights. Values stay the paper's strings
    ("2.3 · 10^19"); quotes and paper labels are never drawn, so never listed."""
    boards = _optional("boards")
    spec, badge = _drawn_board(board, boards)
    if not isinstance(spec, dict) or not spec:
        return []
    words = BOARD_WORDS["zh" if str(language).lower().startswith("zh") else "en"]
    kind = _line(spec.get("kind")).lower()
    title, subtitle = _line(spec.get("title")), _line(spec.get("subtitle"))
    name = BOARD_NAMES.get(kind, "board")
    lines = [f"Board ({name}): {title}" if title else f"Board ({name})"] + ([f"Subtitle: {subtitle}"] if subtitle else [])
    notes, source = [_line(spec.get("note"))], _line(spec.get("source"))   # footnotes: bottom left, bottom right
    mark = " [highlighted]"
    if kind == "bar":
        bars = [b for b in _as_list(spec.get("bars")) if isinstance(b, dict)]
        for i, bar in enumerate(bars):
            label, value = _line(bar.get("label")), _line(bar.get("value"))
            if not value:
                continue
            line = (f"- {label}: {value}" if label else f"- {value}") + (mark if _true(bar.get("highlight")) else "")
            rival = badge.get("rival") if badge and i == badge.get("best") else None
            if isinstance(rival, int) and 0 <= rival < len(bars):
                line += f" [gap to {_line(bars[rival].get('label')) or 'the other bar'}: {badge['text']}]"
            lines.append(line)
        unit = _line(spec.get("unit"))
        lower = str(spec.get("higher_is_better", True)).strip().lower() in ("false", "no", "0")
        changes = boards is not None and hasattr(boards, "_changes") and boards._changes(spec)   # as the drawing: no direction on changes
        direction = "" if DIRECTION_RE.search(" ".join([title, subtitle, unit])) or changes else words["lower" if lower else "higher"]
        shown = unit if re.search(r"[^\W\d_]", unit) else words["unit"].format(unit) if unit else ""
        rival = badge.get("rival") if badge else None
        vs = (words["vs"].format(badge["text"], _line(bars[rival].get("label"))) if isinstance(rival, int) and 0 <= rival < len(bars)
              and _line(bars[rival].get("label")) else "")
        notes = [" · ".join(x for x in (shown, direction, vs) if x)]   # the bar chart's only footnote: it draws no note
    elif kind == "table":
        columns = [_line(c) for c in _as_list(spec.get("columns"))]
        if any(columns):
            lines.append("Columns: " + " | ".join(columns))
        highlighted = _indexes(spec.get("highlight_rows"))
        for i, row in enumerate(_as_list(spec.get("rows"))):
            cells = [_line(c) for c in _as_list(row.get("cells") if isinstance(row, dict) else row)]
            if any(cells):
                lines.append("- " + " | ".join(cells) + (mark if i in highlighted else ""))
    elif kind == "flow":
        highlighted = _indexes(spec.get("highlight"))
        steps = [_line(s) + (mark if i in highlighted else "") for i, s in enumerate(_as_list(spec.get("steps"))) if _line(s)]
        if steps:
            lines.append("- " + " → ".join(steps))
    elif kind == "compare":
        for side, sign in (("left", "•"), ("right", "•")):   # neutral bullets, as drawn
            part = spec.get(side) if isinstance(spec.get(side), dict) else {}
            label, items = _line(part.get("label")), [_line(x) for x in _as_list(part.get("items")) if _line(x)]
            if label or items:
                lines.append(f"- {label + ' ' if label else ''}({sign}): " + "; ".join(items))
    elif kind == "list":   # an open claims board draws empty boxes; a recap ticks the lines it showed evidence for
        items = [_line(x) for x in _as_list(spec.get("items")) if _line(x)]
        ticked = _indexes(spec.get("checked")) if spec.get("checked") is not None else set(range(len(items)))
        lines += [f"- {'☐' if _true(spec.get('open')) else '✓' if k in ticked else '•'} {x}" for k, x in enumerate(items)]
        if not _true(spec.get("open")) and spec.get("checked") is not None and len(ticked & set(range(len(items)))) < len(items):
            notes = [" · ".join(x for x in (*notes, words["checked"]) if x)]   # the legend boards draws under such a recap
    elif kind == "diagram":   # what its drawing shows, in words (services.diagram.describe): "[a box · its texts]", "A → B (label)"
        try:
            from services import diagram
            drawn = diagram.describe(spec)
        except Exception:
            drawn = []
        lines += [f"- {_line(x)}" for x in drawn if _line(x)]
    elif kind == "links":
        tokens = [_line(t) for t in _as_list(spec.get("tokens"))]
        weights = [_weight(w) for w in _as_list(spec.get("weights"))]
        if any(tokens):
            lines.append("- Words: " + " ".join(t for t in tokens if t))
        try:
            src = int(spec.get("from"))
        except (TypeError, ValueError):
            src = -1
        if 0 <= src < len(tokens) and tokens[src]:
            strong_at, shown_at = getattr(boards, "LINK_STRONG", LINK_STRONG), getattr(boards, "LINK_SHOWN", LINK_SHOWN)
            pairs = [(t, w) for i, (t, w) in enumerate(zip(tokens, weights)) if i != src and t]
            strong = [f'"{t}"' for t, w in pairs if w >= strong_at]
            weak = [f'"{t}"' for t, w in pairs if shown_at <= w < strong_at]
            drawn = ([f"strong to {', '.join(strong)}"] if strong else []) + ([f"weak to {', '.join(weak)}"] if weak else [])
            lines.append(f'- Links from "{tokens[src]}"' + (": " + "; ".join(drawn) if drawn else ""))
        notes.append(words["illustrative"] if _true(spec.get("illustrative", True)) else "")
        source = ""   # the tag takes the corner other boards print their source in
    return lines + [f"Note: {x}" for x in notes if x] + ([f"Source: {source}"] if source else [])


def _panel_lines(panel: PanelV2, version: int, language: str) -> list[str]:
    """What a reader sees in one panel, in reading order: the narration box, the bubbles, then the board."""
    lines = [f"Narration: {_line(panel.narration)}"] if _line(panel.narration) else []
    lines += [f"{_line(b.speaker) or 'someone'}: {_line(b.text)}" for b in panel.dialogue if _line(b.text)]
    if version < 3:   # v2 drew these labels into the picture; v3 art is wordless
        lines += [f"Label in the picture: {_line(x)}" for x in panel.labels if _line(x)]
    if panel.kind == "board" and isinstance(panel.board, dict) and panel.board:   # lettering draws a board only there
        lines += _board_lines(panel.board, language)
    return lines


def transcript(sb: StoryboardV2) -> str:
    """The VISIBLE manga in reading order, one "Panel n:" block per panel: narration, bubbles as "speaker: text", and the
    board as text. Never panel.detail, never the paper card."""
    return "\n\n".join("\n".join([f"Panel {p.n}:"] + (_panel_lines(p, sb.version, sb.language) or ["(no text)"]))
                       for p in sb.panels)


PANEL_HEAD_RE = re.compile(r"^Panel (\d+):[ \t]*$", re.M)


def _panel_texts(text: str) -> dict[int, str]:
    """Panel number -> that panel's block of a transcript() string."""
    heads = list(PANEL_HEAD_RE.finditer(text or ""))
    out: dict[int, str] = {}
    for k, m in enumerate(heads):
        end = heads[k + 1].start() if k + 1 < len(heads) else len(text)
        n = int(m.group(1))
        out[n] = f"{out.get(n, '')}\n{text[m.end():end].strip()}".strip()
    return out


# ----------------------------------------------------------------------------- numbers and citations (pure)

SUPERSCRIPTS = "⁰¹²³⁴⁵⁶⁷⁸⁹⁻"   # superscript digits and minus (display_value writes 10^19 as 10¹⁹)
SUPERSCRIPT_RE = re.compile(f"[{SUPERSCRIPTS}]+")
SUPERSCRIPT_MAP = str.maketrans(SUPERSCRIPTS, "0123456789-")
NUMBER_RE = re.compile(r"(?<![\d.])\d+(?:\.\d+)?(?!\d)")
TEENS = {"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
         "seventeen": 17, "eighteen": 18, "nineteen": 19}
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
UNITS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}
# ten to ninety-nine in words ("eleven tasks", "twenty-one layers"); smaller ones are ignored like single digits
NUMBER_WORD_RE = re.compile(rf"\b(?:({'|'.join(TEENS)})|({'|'.join(TENS)})(?:[ -]({'|'.join(UNITS)}))?)\b", re.I)
# "panel 12", "panels 3, 4 and 7", "第12格": citations written into an answer are not numbers it states; a reference
# never runs into a value ("In panel 7, 28.4 BLEU" keeps 28.4)
PANEL_REF_RE = re.compile(r"\bpanels?\s*#?\d+(?!\.?\d)(?:\s*(?:,|and|&|or|to|-|–)\s*#?\d+(?!\.?\d))*"
                          r"|第\s*\d+\s*(?:格|幅|个?面板|页)", re.I)


def _numbers(text: str) -> dict[float, str]:
    """Every number in a text (value -> as written). 10¹⁹ reads as 10^19 (NFKC alone would make it 1019), full-width
    digits and thousands separators are normalised, and number words from ten up count ("eleven tasks" = 11)."""
    text = SUPERSCRIPT_RE.sub(lambda m: "^" + m.group().translate(SUPERSCRIPT_MAP), str(text or ""))
    text = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", unicodedata.normalize("NFKC", text))
    out: dict[float, str] = {}
    for m in NUMBER_RE.finditer(text):
        out.setdefault(float(m.group()), m.group())
    for m in NUMBER_WORD_RE.finditer(text):
        teen, tens, unit = (g.lower() if g else "" for g in m.groups())
        out.setdefault(float(TEENS[teen] if teen else TENS[tens] + UNITS.get(unit, 0)), m.group())
    return out


def _answer_numbers(answer: str) -> dict[float, str]:
    """The numbers an answer states, without its panel references and single digits: those are often spelled out on the
    panel ("eight GPUs") and are rarely what a question is about."""
    return {v: s for v, s in _numbers(PANEL_REF_RE.sub(" ", answer)).items() if not (v < 10 and s.isdigit())}


def _citation_problem(question: str, answer: dict, panels: dict[int, str]) -> str:
    """Why an answer cannot get full marks from its citations, or "": it cites no panel of the manga, or it states a
    number its cited panels do not show (recalled from memory, or cited wrongly). Numbers the question gives are free."""
    cited = [n for n in answer["panels"] if n in panels][:MAX_CITED]
    if not cited:
        return "no panel of the manga cited"
    shown = {v for n in cited for v in _numbers(panels[n])}
    asked = _numbers(question)
    missing = [s for v, s in _answer_numbers(answer["answer"]).items() if v not in shown and v not in asked]
    return f"{', '.join(missing)} not in the cited panel(s) {', '.join(map(str, cited))}" if missing else ""


# ----------------------------------------------------------------------------- normalising replies

QID_RE = re.compile(r"(?:Q(?:UESTION)?)?\s*#?\s*0*(\d+)")


def _first(item: dict, *keys):
    """The first of ``keys`` a reply item carries: models rename fields ("panel", "cited_panels", "grade", "reason")."""
    return next((item[k] for k in keys if item.get(k) is not None), None)


def _qid(value) -> str:
    """Question ids as "Q3" whatever the model sends back ("3", "q3", 3, "Q 3", "Question 3", "Q03")."""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    s = _line(value).upper()
    m = QID_RE.fullmatch(s)
    return f"Q{m.group(1)}" if m else s


def _panel_numbers(value) -> list[int]:
    """Cited panels as ints: [7, "8"], 7.0, "Panel 7", "7, 8" and "P7" all work."""
    out: list[int] = []
    for x in _as_list(value):
        if isinstance(x, bool):
            continue
        if isinstance(x, (int, float)):
            found = [int(x)] if float(x).is_integer() else []
        else:
            found = [int(d) for d in re.findall(r"\d+", str(x))]
        out += [n for n in found if n not in out]
    return out


def _score(value) -> float:
    """A grade snapped to 0, 0.5 or 1 (graders also send "1", 0.8 or "partial")."""
    named = {"correct": 1.0, "full": 1.0, "partial": 0.5, "partially correct": 0.5, "half": 0.5, "wrong": 0.0, "incorrect": 0.0}
    if isinstance(value, str) and value.strip().lower() in named:
        return named[value.strip().lower()]
    try:
        x = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 1.0 if x >= 0.75 else 0.5 if x >= 0.25 else 0.0


def _not_shown(answer: str) -> bool:
    a = _line(answer).lower().strip(" .。!！")
    return not a or a.startswith(NOT_SHOWN) or a in ("n/a", "na", "none", "unknown", "未显示", "没有显示")


def _items(data: dict, key: str, field: str) -> list[dict]:
    """The reply's items under ``key`` as dicts. Models also send {"Q1": {...}} or {"Q1": "text"} keyed by question id
    (under the key or as the whole reply), one bare object, plain strings or numbers (read as ``field``), or the list
    under another name ({"results": [...]})."""
    if key not in data:
        if field in data or "id" in data:   # one bare item
            return [data]
        lists = [v for v in data.values() if isinstance(v, list) and any(isinstance(x, dict) for x in v)]
        value = lists[0] if len(lists) == 1 else data
    else:
        value = data[key]
    if isinstance(value, dict):
        if "id" in value:
            return [value]
        return [{**v, "id": v.get("id", k)} if isinstance(v, dict) else {"id": k, field: v}
                for k, v in value.items() if isinstance(v, (dict, str, int, float))]
    return [x if isinstance(x, dict) else {field: x} for x in _as_list(value) if isinstance(x, (dict, str, int, float))]


def _match(items: list[dict], ids: list[str]) -> dict[str, dict]:
    """Reply items by question id. Items that carry no id at all fill the ids still open in order, but only when there
    is exactly one per open id (a model that drops the ids keeps the order); an item with an unknown id is dropped,
    never guessed."""
    wanted, out, anonymous = set(ids), {}, []
    for item in items:
        raw = _first(item, "id", "question_id", "qid")
        if not _line(raw):
            anonymous.append(item)
        elif _qid(raw) in wanted and _qid(raw) not in out:
            out[_qid(raw)] = item
    rest = [i for i in ids if i not in out]
    if anonymous and len(anonymous) == len(rest):
        out.update(zip(rest, anonymous))
    return out


def _numbered(items: list[dict]) -> tuple[list[dict], dict[str, str]]:
    """The items of one call renumbered Q1..Qn, and the way back to their own ids. A model renumbers ids with a gap:
    shown G6 and then G8 (G7 was "not shown", so never sent), Seed graded G8 as "G7", G9 as "G8" and G10 as "G9", and
    every verdict after the gap landed on the wrong question."""
    sent = [dict(item, id=f"Q{i}") for i, item in enumerate(items, start=1)]
    return sent, {s["id"]: item["id"] for s, item in zip(sent, items)}


def _claim_list(claims) -> list[dict]:
    """Abstract claims as [{"id", "text", "quote"}]: dicts from the analysis, or bare strings (id = position)."""
    out: list[dict] = []
    for i, c in enumerate(_as_list(claims)):
        if isinstance(c, dict):
            ids = _claim_ids(c.get("id", i))
            item = {"id": ids[0] if ids else i, "text": _line(c.get("text") or c.get("claim")), "quote": _line(c.get("quote"))}
        else:
            item = {"id": i, "text": _line(c), "quote": ""}
        if item["text"] and item["id"] not in {x["id"] for x in out}:
            out.append(item)
    return out


def _question_list(questions) -> list[dict]:
    """Questions as [{"id", "claim_ids", "question", "answer"}] (and "sentences" when they have them); ids renumbered
    Q1..Qn when they collide."""
    out = [{"id": _qid(q.get("id") or i), "claim_ids": _claim_ids(q.get("claim_ids")), "question": _line(q.get("question")),
            "answer": _line(q.get("answer")), **({"sentences": list(q["sentences"])} if isinstance(q.get("sentences"), list) else {})}
           for i, q in enumerate(_as_list(questions), start=1) if isinstance(q, dict) and _line(q.get("question")) and _line(q.get("answer"))]
    if len({q["id"] for q in out}) < len(out):
        for i, q in enumerate(out, start=1):
            q["id"] = f"Q{i}"
    return out


# ----------------------------------------------------------------------------- model calls

class _JsonMode:
    """The engine with JSON mode switched on for every call, so _ask_json's parsing and repair round apply unchanged."""

    def __init__(self, engine: BaseEngine):
        self.engine = engine

    async def generate_text(self, messages, config: Optional[GenerationConfig] = None):
        return await self.engine.generate_text(messages, dataclasses.replace(config or GenerationConfig(), response_format={"type": "json_object"}))


async def _ask(engine: BaseEngine, prompt: str, *, key: str, usage: Optional[dict], max_tokens: int = 6000) -> dict:
    """One quiz call: JSON mode, thinking off, temperature 0 (a gate must give the same verdict on the same script)."""
    for attempt in range(2):   # an unparsable reply is asked once more (twice broken: the caller records the error)
        try:
            data, used = await _ask_json(_JsonMode(engine), "", prompt, thinking=False, temperature=0.0, max_tokens=max_tokens, key=key)
            break
        except ValueError as e:   # json.JSONDecodeError is a ValueError too ("Invalid control character at ...")
            if attempt or not (isinstance(e, json.JSONDecodeError) or "JSON" in str(e)):
                raise
            print(f"[abstract_quiz] unparsable reply ({type(e).__name__}: {e}); asking once more")
    if usage is not None:
        for k in ("prompt_tokens", "completion_tokens"):
            usage[k] = int(usage.get(k, 0) or 0) + int(used.get(k, 0) or 0)
    return data


# ----------------------------------------------------------------------------- questions

MAX_QUESTIONS = 20


def abstract_sentences(abstract_text: str) -> list[str]:
    """The abstract's sentences the quiz must cover, as storyboarder_v2.verify_claims counts them (a fragment of under
    four words is extraction debris)."""
    return [s for s in split_sentences(abstract_text or "") if len(s.split()) >= 4]


def _quiz_size(n_claims: int, n_sentences: int = 0) -> tuple[int, int]:
    """8-12 questions, fewer for an abstract with very few claims (no padding with trivia), more when it has more claims
    or sentences: every sentence gets a question, and one with several facts gets one per fact."""
    lo = max(min(8, 2 * n_claims), min(n_sentences, MAX_QUESTIONS))
    return lo, max(lo, min(MAX_QUESTIONS, max(12, n_claims, n_sentences + n_sentences // 2)))


def _sentence_claims(sentence: str, claims: list[dict]) -> list[int]:
    """The claims taken from a sentence: at least half of a claim's quote words are in it; else the one claim sharing the
    most words with it (verify_claims gives every sentence a claim, so this is a fallback)."""
    words = _content_tokens(sentence)
    shared = [(len(quote & words), len(quote), c["id"]) for c in claims for quote in [_content_tokens(c.get("quote") or c.get("text") or "")]]
    out = [cid for common, size, cid in shared if size and common >= 0.5 * size]
    best = max(shared, default=(0, 0, None))
    return out or ([best[2]] if best[0] else [])


def quiz_prompt(claims, abstract_text: str = "", problems=()) -> str:
    claims = _claim_list(claims)
    sentences = abstract_sentences(abstract_text)
    lo, hi = _quiz_size(len(claims), len(sentences))
    retry = f"\nYOUR PREVIOUS QUIZ WAS REJECTED: {'; '.join(problems)}. Write the whole quiz again.\n" if problems else ""
    if sentences:   # the fixed eval quizzes cover every sentence, one fact per question: claims can leave a fact out
        abstract = "\nABSTRACT, sentence by sentence:\n" + "\n".join(f"S{i}: {s}" for i, s in enumerate(sentences, start=1)) + "\n"
        cover = (f"- {lo}-{hi} questions. Every sentence S1-S{len(sentences)} gets at least one question, and every fact a sentence states gets its "
                 "own question: what was hard and why it matters, what earlier methods need or assume, what is new, each step of how it "
                 "works, the training method, every result with its number and setting, how general it is, the conclusion. A sentence "
                 "that lists several things (four properties, two kinds of tasks) gets one question that asks for the whole list. Every "
                 "claim below gets at least one question too.\n")
        ids = '- "sentences": the S numbers of the sentences the question checks; "claim_ids": the ids of the claims it checks.'
        shape = '{"questions": [{"id": "Q1", "sentences": [1], "claim_ids": [0], "question": "...", "answer": "..."}]}'
    else:
        abstract = f"\nABSTRACT:\n{abstract_text.strip()}\n" if (abstract_text or "").strip() else ""
        cover = (f"- {lo}-{hi} questions. Every claim below gets at least one question; a claim with several facts (a score and the "
                 "baseline it beats, a training time and its hardware, a result and its dataset) gets one question per fact.\n")
        ids = '- "claim_ids": the ids of the claims the question checks.'
        shape = '{"questions": [{"id": "Q1", "claim_ids": [0], "question": "...", "answer": "..."}]}'
    return f"""Write a short-answer quiz on a research paper's ABSTRACT. It checks whether someone who read a manga about the paper (and never the paper itself) now knows what the abstract says.

Rules
{cover}- A question names the setting it asks about (task, dataset, language pair, model) so it has exactly one correct answer, and it never contains its own answer.
- No yes/no or multiple-choice questions: ask what, which, how or how much, so that a guess cannot pass.
- The gold "answer" is short (at most 25 words) and comes only from the claims and the abstract: numbers copied exactly with their unit and setting (e.g. "28.4 BLEU on WMT 2014 English-to-German"), keeping the abstract's qualifiers (single model, including ensembles, over, about, a small fraction).
- Ask what a reader should take away: the problem, what is new, how it works at the abstract's level, every result with its number, the cost, how general it is, the conclusion. Never ask about authors, affiliations, the venue, dates or wording.
{ids}
Output JSON only: {shape}
{retry}{abstract}
CLAIMS:
{json.dumps(claims, ensure_ascii=False)}"""


def _yes_no(answer: str) -> bool:
    """A gold answer that is only "yes" or "no" ("Yes, it does."): a reader who guesses gets it right."""
    words = re.findall(r"[^\W_]+", answer.lower())
    return bool(words) and words[0] in ("yes", "no") and len(words) <= 4


def _sentence_ids(value, count: int) -> list[int]:
    """Sentence numbers 1..count from "S3", 3 or ["S1", "S2"]."""
    out = []
    for x in _as_list(value):
        m = re.fullmatch(r"\s*S?\s*(\d+)\s*", str(x), re.I)
        if m and 1 <= int(m.group(1)) <= count and int(m.group(1)) not in out:
            out.append(int(m.group(1)))
    return out


def _valid_questions(data: dict, claim_ids: list[int], source: str, sentences: Optional[list[str]] = None,
                     claims: Optional[list[dict]] = None) -> tuple[list[dict], list[str]]:
    """Usable questions in reply order (a question, a gold answer, at least one known claim id; repeats dropped) and
    the rejected ones that would break the gate: a gold number that is not in the claims or the abstract (no manga
    could ever show it), a question that already states its gold numbers, or a yes/no question. With the abstract's
    ``sentences`` each question keeps the sentence numbers it checks, and one without a usable claim id gets the claims
    taken from its sentences."""
    known = _numbers(source)
    out, rejected, seen = [], [], set()
    for item in _items(data, "questions", "question"):
        question = _line(item.get("question") or item.get("q"))
        answer = _line(item.get("answer") or item.get("gold") or item.get("a"))
        ids = [i for i in _claim_ids(item.get("claim_ids", item.get("claims"))) if i in claim_ids]
        where = _sentence_ids(item.get("sentences", item.get("sentence")), len(sentences)) if sentences else []
        if not ids and where and claims:
            ids = list(dict.fromkeys(i for k in where for i in _sentence_claims(sentences[k - 1], claims) if i in claim_ids))
        if not (question and answer and ids) or question.lower() in seen:
            continue
        gold = _answer_numbers(answer)
        invented = [s for v, s in gold.items() if v not in known]
        if invented:
            rejected.append(f'"{question}" has numbers in its answer that are not in the claims or the abstract ({", ".join(invented)})')
        elif gold and all(v in _numbers(question) for v in gold):
            rejected.append(f'"{question}" contains its own answer')
        elif _yes_no(answer):
            rejected.append(f'"{question}" is a yes/no question (a guess passes it): ask what, which, how or how much')
        else:
            seen.add(question.lower())
            out.append({"claim_ids": ids, "question": question, "answer": answer, **({"sentences": where} if sentences else {})})
    return out, rejected


def _pick(items: list[dict], limit: int) -> list[dict]:
    """At most ``limit`` questions in reply order: the first question of every claim and of every abstract sentence is
    kept first, so trimming a long reply never leaves a claim or a sentence untested."""
    keep, covered = set(), set()
    for k, item in enumerate(items):
        tags = {("c", i) for i in item["claim_ids"]} | {("s", i) for i in item.get("sentences") or []}
        if tags - covered:
            keep.add(k)
            covered |= tags
    for k in range(len(items)):
        if len(keep) >= limit:
            break
        keep.add(k)
    return [item for k, item in enumerate(items) if k in keep]


async def make_questions(engine: BaseEngine, claims, abstract_text: str = "", *, usage: Optional[dict] = None) -> list[dict]:
    """8-12 short-answer questions with gold answers from the abstract claims (numbers exact, with their setting), one or
    more per claim: [{"id": "Q1", "claim_ids": [0], "question", "answer"}]. One retry when the reply is too short or
    leaves a claim without a question (a failed retry keeps the first quiz); raises ValueError when nothing usable comes
    back."""
    claims = _claim_list(claims)
    if not claims:
        raise ValueError("no abstract claims to quiz on")
    ids = [c["id"] for c in claims]
    sentences = abstract_sentences(abstract_text)
    lo, hi = _quiz_size(len(claims), len(sentences))
    source = " ".join([abstract_text or ""] + [f"{c['text']} {c['quote']}" for c in claims])
    best: list[dict] = []
    problems: list[str] = []
    for _ in range(2):
        try:
            data = await _ask(engine, quiz_prompt(claims, abstract_text, problems), key="questions", usage=usage)
        except Exception as e:
            if not best or getattr(e, "account", False):
                raise
            print(f"[abstract_quiz] WARNING: the quiz retry failed ({type(e).__name__}: {e}); keeping the first quiz")
            break
        valid, rejected = _valid_questions(data, ids, source, sentences, claims)
        items = _pick(valid, hi)
        missing = [i for i in ids if not any(i in q["claim_ids"] for q in items)]
        untested = [k for k in range(1, len(sentences) + 1) if not any(k in q.get("sentences", ()) for q in items)]
        problems = []
        if len(items) < lo:
            problems.append(f"only {len(items)} usable questions (write {lo}-{hi}, each with a question, an answer and claim_ids)")
        if missing:
            problems.append(f"claims without a question: {', '.join(map(str, missing))}")
        if untested:
            problems.append(f"sentences without a question: {', '.join(f'S{k}' for k in untested)}")

        def reach(qs):   # sentences and claims tested, then size: the retry is kept only when it tests more
            return (len({k for q in qs for k in q.get("sentences", ())}) + len({i for q in qs for i in q["claim_ids"]}), len(qs))

        if reach(items) > reach(best):
            best = items
        if not problems:   # a complete quiz is kept even when a stray question was dropped
            break
        problems += rejected   # the retry learns why its dropped questions were unusable
    if not best:
        raise ValueError("the quiz model returned no usable questions")
    if problems:
        print(f"[abstract_quiz] WARNING: quiz kept with problems: {problems}")
    return [{"id": f"Q{i}", **q} for i, q in enumerate(best, start=1)]


# ----------------------------------------------------------------------------- the naive reader

def reader_prompt(transcript_text: str, questions: list[dict]) -> str:
    asked = [{"id": q["id"], "question": q["question"]} for q in questions]   # never the gold answers
    return f"""You have never read the research paper behind this manga, and you must not use anything you know about it, its authors or its field from anywhere else. Even if you recognise the paper, answer ONLY from the manga text below.

The manga text lists every panel in reading order. "Panel n:" starts a panel, "Narration:" is a caption box, "name: text" is a speech bubble, and "Board (...)" is a chart, table or diagram drawn in the scene: its lines that start with "- " are its bars with their values, its rows, steps or items; [highlighted] marks what the board highlights, "[gap to X: +2.04]" is the difference the chart marks between that bar and bar X, and "Note:" and "Source:" lines are the board's footnotes.

For each question:
- Answer in one short sentence with only what the panels say. Copy numbers exactly as written, together with the setting the panels give them (task, dataset, model size, single model or ensemble). Never compute, round, convert or estimate a number, and never fill a gap with outside knowledge.
- "panels": the numbers of the panels that show your answer (1-3).
- When the manga does not show the answer, answer exactly "not shown" with "panels": []. When it shows only part, give that part and nothing more.
Output JSON only: {{"answers": [{{"id": "Q1", "answer": "...", "panels": [3]}}]}}

MANGA TEXT:
{transcript_text}

QUESTIONS:
{json.dumps(asked, ensure_ascii=False)}"""


async def answer_questions(engine: BaseEngine, transcript_text: str, questions: list[dict], *, usage: Optional[dict] = None) -> list[dict]:
    """A reader that has NOT seen the paper answers from the transcript only, citing panels: [{"id", "answer", "panels"}]
    in question order; "not shown" (no panels) when absent. One follow-up call for questions the reply skipped (when that
    call fails, they stay "not shown"). A reader that answers nothing at all raises ValueError: a broken reply must not
    pass for a manga that shows nothing."""
    questions = _question_list(questions)
    got: dict[str, dict] = {}
    for round_ in range(2):
        pending = [q for q in questions if q["id"] not in got]
        if not pending:
            break
        sent, back = _numbered(pending)
        try:
            data = await _ask(engine, reader_prompt(transcript_text, sent), key="answers", usage=usage)
        except Exception as e:
            if round_ == 0 or getattr(e, "account", False):
                raise
            print(f"[abstract_quiz] WARNING: the reader's follow-up failed ({type(e).__name__}: {e}); those questions count as not shown")
            break
        for local, item in _match(_items(data, "answers", "answer"), list(back)).items():
            qid = back[local]
            answer = _line(_first(item, "answer", "text"))
            got[qid] = {"id": qid, "answer": answer or NOT_SHOWN,
                        "panels": [] if _not_shown(answer) else _panel_numbers(_first(item, "panels", "panel", "cited_panels", "citations"))}
    if questions and len(got) * 2 < len(questions):   # fewer than half answered: the reply broke (a revision would chase noise)
        raise ValueError(f"the reader answered only {len(got)} of {len(questions)} questions")
    return [got.get(q["id"]) or {"id": q["id"], "answer": NOT_SHOWN, "panels": []} for q in questions]


# ----------------------------------------------------------------------------- strict grading

def grade_prompt(items: list[dict]) -> str:
    cited = any("cited_text" in item for item in items)   # grade() without a transcript cannot show the panels
    return f"""Grade each answer of a reading quiz strictly against its gold answer.
The reader never saw the paper: they answered only from the text of a manga about it and cited the panels that show each answer.{' "cited_text" is everything those panels contain.' if cited else ""}

Score every item 1, 0.5 or 0:
- 1: the answer says what the gold answer says. Paraphrase and extra details are fine as long as nothing contradicts the gold answer. Every number matches exactly (28.4 is not 28 or 28.5) and keeps its setting and the gold answer's qualifiers: task or dataset, language pair, model size, single model or ensemble, "over", "about", "including ensembles", "a small fraction", "both large and limited data". Values from which the gold quantity follows directly count (28.4 vs 26.36 for "over 2 BLEU better").
- 0.5: the main point is right, but part of the gold answer is missing or vague (a second fact, the baseline, a key qualifier){"; or the cited text does not show the answer" if cited else ""}.
- 0: wrong, contradicting the gold answer, a different number or setting (an ensemble result given for a single model, another dataset), too vague to check, or "not shown".
Judge only by the gold answer{" and the cited text" if cited else ""}, never by what you know about the paper.
Output JSON only: {{"grades": [{{"id": "Q1", "score": 1, "why": "one short sentence"}}]}}

ITEMS:
{json.dumps(items, ensure_ascii=False)}"""


async def grade(engine: BaseEngine, questions: list[dict], answers: list[dict], transcript_text: str = "", *,
                usage: Optional[dict] = None, cite: bool = True) -> list[dict]:
    """Strict grades in question order: [{"id", "score": 0 | 0.5 | 1, "why"}]. "not shown" scores 0 without a call.
    With the transcript the grader sees what the cited panels contain, and an answer whose numbers the cited panels do
    not show (or that cites no panel of the manga) is capped at 0.5 by code, whatever the grader said. ``cite=False``
    grades answers given without any manga (a closed-book control), which cannot cite a panel: the cap had held every
    closed-book answer at 0.5. One follow-up call for items the grader skipped (when that call fails, they score 0 as
    not graded); a grader that grades nothing at all raises ValueError."""
    questions = _question_list(questions)
    panels = _panel_texts(transcript_text) if transcript_text else {}
    given = {_qid(a.get("id")): {"answer": _line(a.get("answer")), "panels": _panel_numbers(a.get("panels"))}
             for a in _as_list(answers) if isinstance(a, dict)}
    out: dict[str, dict] = {}
    items: list[tuple[dict, dict]] = []   # (what the grader sees, the normalised answer)
    for q in questions:
        a = given.get(q["id"]) or {"answer": NOT_SHOWN, "panels": []}
        if _not_shown(a["answer"]):
            out[q["id"]] = {"id": q["id"], "score": 0.0, "why": "the reader found no answer in the manga"}
            continue
        item = {"id": q["id"], "question": q["question"], "gold": q["answer"], "answer": a["answer"]}
        if panels:
            cited = [n for n in a["panels"] if n in panels][:MAX_CITED]
            item["panels"] = cited
            item["cited_text"] = "\n".join(f"Panel {n}: " + panels[n].replace("\n", " / ") for n in cited) or "(no panel of the manga cited)"
        else:
            item["panels"] = a["panels"][:MAX_CITED]
        items.append((item, a))
    verdicts: dict[str, dict] = {}
    for round_ in range(2):
        pending = [item for item, _a in items if item["id"] not in verdicts]
        if not pending:
            break
        sent, back = _numbered(pending)
        try:
            data = await _ask(engine, grade_prompt(sent), key="grades", usage=usage)
        except Exception as e:
            if round_ == 0 or getattr(e, "account", False):
                raise
            print(f"[abstract_quiz] WARNING: the grader's follow-up failed ({type(e).__name__}: {e}); those items count as not graded")
            break
        verdicts.update({back[local]: g for local, g in _match(_items(data, "grades", "score"), list(back)).items()})
    if items and not verdicts:
        raise ValueError("the grader returned no usable grades")
    for item, a in items:
        g = verdicts.get(item["id"])
        score, why = (_score(_first(g, "score", "grade")), _line(_first(g, "why", "reason"))) if g else (0.0, "not graded")
        problem = "" if not cite else _citation_problem(item["question"], a, panels) if panels else ("" if a["panels"] else "no panel cited")
        if problem and score > 0.5:
            score, why = 0.5, f"{why} (capped at 0.5: {problem})".strip()
        out[item["id"]] = {"id": item["id"], "score": score, "why": why}
    return [out[q["id"]] for q in questions]


# ----------------------------------------------------------------------------- the gate

GRADERS = 3   # independent grades per answer, the median counts: Seed 2.0 Pro samples at temperature 1 whatever is asked


async def _median_grades(engine: BaseEngine, questions: list[dict], answers: list[dict], text: str, usage: dict, n: int = GRADERS) -> list[dict]:
    """The median of ``n`` independent grades per question (a grader that fails is left out; all failing raises): one
    harsh or lenient sample no longer decides whether a claim "fails" and the script is revised."""
    results = await asyncio.gather(*(grade(engine, questions, answers, text, usage=usage) for _ in range(n)), return_exceptions=True)
    runs = [r for r in results if not isinstance(r, BaseException)]
    fatal = next((r for r in results if getattr(r, "account", False)), None)
    if fatal is not None or not runs:
        raise fatal or results[0]
    out = []
    for i, q in enumerate(questions):
        given = sorted((run[i] for run in runs), key=lambda g: g["score"])
        out.append(given[len(given) // 2])
    return out


async def run_quiz(text_engine: BaseEngine, fast_engine: Optional[BaseEngine], sb: StoryboardV2, claims, abstract_text: str = "",
                   *, questions: Optional[list[dict]] = None) -> dict:
    """The whole gate on the visible manga: questions (checker model; pass ``questions`` to reuse a previous set, so the
    scores of revision rounds compare), the naive reader (fast model), the strict grader (checker model, the median of
    GRADERS samples). Returns {"score": 0..1, "questions", "answers", "grades", "failed_claim_ids",
    "untested_claim_ids", "usage"}. Never raises: {"score": None, "error": "...", "usage": {...}} on failure
    (cancellation still propagates)."""
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    try:
        questions = _question_list(questions) or await make_questions(text_engine, claims, abstract_text, usage=usage)
        text = transcript(sb)
        answers = await answer_questions(fast_engine or text_engine, text, questions, usage=usage)
        grades = await _median_grades(text_engine, questions, answers, text, usage)
        scores = {g["id"]: g["score"] for g in grades}
        tested = {i for q in questions for i in q["claim_ids"]}
        result = {
            "score": round(sum(scores.values()) / len(questions), 3),
            "questions": questions,
            "answers": answers,
            "grades": grades,
            # a claim fails when any of its questions falls short: the owner wants every abstract claim visible
            "failed_claim_ids": sorted({i for q in questions if scores.get(q["id"], 0.0) < 1 for i in q["claim_ids"]}),
            "untested_claim_ids": [c["id"] for c in _claim_list(claims) if c["id"] not in tested],
            "usage": usage,
        }
        sentences = abstract_sentences(abstract_text)
        if sentences and any("sentences" in q for q in questions):
            result["untested_sentences"] = [k for k in range(1, len(sentences) + 1) if not any(k in q.get("sentences", ()) for q in questions)]
    except Exception as e:  # a gate, never a crash: the caller keeps the script and records the error
        if getattr(e, "account", False):   # an account problem stops the job instead (engines.byteplus.AccountError)
            raise
        print(f"[abstract_quiz] quiz failed: {type(e).__name__}: {e}")
        return {"score": None, "error": f"{type(e).__name__}: {e}", "usage": usage}
    print(f"[abstract_quiz] score {result['score']:.2f} on {len(questions)} questions; failed claims {result['failed_claim_ids']}")
    return result


# ----------------------------------------------------------------------------- revision

def _feedback(quiz: dict, claim_id: int) -> list[dict]:
    """The quiz questions about a claim that fell short: what was asked, the expected answer, what the reader answered
    from the manga and why it was marked down."""
    quiz = quiz if isinstance(quiz, dict) else {}
    answers = {_qid(a.get("id")): a for a in _as_list(quiz.get("answers")) if isinstance(a, dict)}
    grades = {_qid(g.get("id")): g for g in _as_list(quiz.get("grades")) if isinstance(g, dict)}
    out = []
    for q in _as_list(quiz.get("questions")):
        if not isinstance(q, dict) or claim_id not in _claim_ids(q.get("claim_ids")):
            continue
        g, a = grades.get(_qid(q.get("id")), {}), answers.get(_qid(q.get("id")), {})
        if _score(g.get("score")) < 1:
            out.append({"question": _line(q.get("question")), "expected": _line(q.get("answer")),
                        "reader_answered": _line(a.get("answer")) or NOT_SHOWN, "cited_panels": _panel_numbers(a.get("panels")),
                        "why_marked_down": _line(g.get("why"))})
    return out


def _script_panel(p: PanelV2) -> dict:
    """A panel as the writer sees it (a diagram as its brief and what it shows, as in the fix rounds: its elements and
    coordinates are the designer's, and a revision that edited them shipped without a layout check)."""
    return {"n": p.n, "beat": p.beat, "kind": p.kind, "title": p.title, "characters": p.characters, "scene": p.scene,
            "visual": p.visual, "shows": p.shows, "dialogue": [{"speaker": b.speaker, "text": b.text} for b in p.dialogue], "narration": p.narration,
            "emotions": p.emotions, "board": _prompt_board(p.board) if p.kind == "board" else None, "claims": p.claims}


def revise_prompt(sb: StoryboardV2, failed_claims, cast: Cast, *, paper_text: str = "", quiz: Optional[dict] = None) -> str:
    """Ask the writer for revised or new panels (v3 panel JSON with board specs) that make the failed abstract claims
    visible, under the same panel, board, budget and data rules as the story and fix prompts (storyboarder_v2._v3_rules:
    the revised panels go straight back through lint_v3). ``failed_claims``: claim ids or claim dicts. ``quiz`` (default
    sb.quiz) tells the writer what the reader answered; ``paper_text`` lets new board values carry verbatim quotes.
    Reply: {"panels": [revised panels with "n" | new panels with "after"]}."""
    known = {c["id"]: c for c in _claim_list(sb.abstract_claims)}
    quiz = quiz if quiz is not None else sb.quiz
    failed: dict[int, dict] = {}
    for item in _as_list(failed_claims):
        ids = _claim_ids(item.get("id") if isinstance(item, dict) else item)
        claim = known.get(ids[0]) if ids else None
        if claim is None and isinstance(item, dict):
            claim = (_claim_list([item]) or [None])[0]
        if claim is not None and claim["id"] not in failed:
            failed[claim["id"]] = {**claim, "quiz": _feedback(quiz, claim["id"])}
    language = sb.language if sb.language in LANG_NAMES else "en-US"
    language_rule = ""
    if language != "en-US":   # the rules count English words; a translated script counts characters (about 1.8 per word)
        b = _v3_budget(language)
        scene, board = b["scene"], b["board"]
        language_rule = (f"\n\nLANGUAGE: the script is in {LANG_NAMES[language]}: write every text in it (numbers, model names and "
                         f"dataset names stay exact). Budgets count characters instead of words: a scene panel has at most "
                         f"{scene['bubbles']} bubbles of at most {scene['bubble']} characters each and narration of at most "
                         f"{scene['narration']} characters; a board panel at most {board['bubbles']} bubbles of at most "
                         f"{board['bubble']} characters each and narration of at most {board['narration']} characters.")
    critic = next((m.name for m in cast.members if m.role == "skeptic"), "")
    proof = f"the critic {critic} asks for proof and the board answers" if critic else "someone asks for proof and the board answers"
    # a new panel that teaches a mechanism is a diagram board (round 23: scenes tell the story, boards explain)
    new_how = ('{"after": 9, "kind": "scene", "beat": "how", "title": "...", "characters": ["..."], "scene": "...", "visual": "...", '
               '"shows": "what the picture alone makes clear", "dialogue": [], "narration": "...", "emotions": {}, "board": null, '
               '"claims": [4]}' if _story.SCENE_SHOWS else
               '{"after": 9, "kind": "board", "beat": "how", "title": "...", "characters": ["..."], "scene": "...", "visual": "...", '
               '"shows": "", "dialogue": [{"speaker": "...", "text": "..."}], "narration": "...", "emotions": {}, "board": {"kind": '
               '"diagram", "title": "...", "brief": "the running example\'s real things and the arrangement that shows the step"}, '
               '"claims": [4]}')
    paper = (f"\n\nPAPER TEXT:\n{paper_text.strip()}" if (paper_text or "").strip() else
             "\n\n(The paper text is not attached: take board quotes only from the claims' quotes or from quotes already on the script's boards.)")
    return f"""A reader who never saw the paper read only the visible text of this manga (speech bubbles, narration boxes and the boards drawn in the scenes) and then took a quiz on the paper's ABSTRACT. These abstract claims did not come across:
{json.dumps(list(failed.values()), ensure_ascii=False)}

Revise the script so that every claim above becomes VISIBLE in the panels, with its exact numbers and setting, as part of the story:
- First choice: make the fact visible in an existing panel about the same thing (rewrite its narration or a bubble, or add the missing bar, row, step or item to its board). Return the whole revised panel with its "n".
- Otherwise add a new panel with "after": the n of the panel it follows (at most one new panel per claim). A result with numbers goes on a BOARD panel (a bar chart or table of the paper's values) with 1-2 cast members reacting: {proof}.
- A reader must be able to answer the quiz questions from the text alone: state each number with its setting (task, dataset, model size, single model or ensemble) and the baseline it is compared with, in words or on the board.
- Keep the running example ({_line(sb.running_example) or "the script's own"}) and every character's role; explain any term the first time it appears. A running example that is not from the paper stays an illustration, never a result.
- List in "claims" the ids of the abstract claims each returned panel conveys. Leave every other panel out of the reply, and never remove a fact that another panel relies on.

{_v3_rules(cast)}{language_rule}

OUTPUT: JSON only:
{{"panels": [{{"n": 7, "kind": "board", "beat": "proof", "title": "...", "characters": ["..."], "scene": "...", "visual": "...", "shows": "", "dialogue": [{{"speaker": "...", "text": "..."}}], "narration": "...", "emotions": {{}}, "board": <one board spec as above>, "claims": [3]}}, {new_how}]}}
Revised panels carry "n" and no "after"; new panels carry "after" and no "n". "beat" is one of hook, problem, idea, how, proof, limits, payoff, recap. "board" is null on scene panels.

CAST (use these lowercase names exactly)
{cast.describe()}

ALL ABSTRACT CLAIMS (ids for "claims")
{json.dumps(list(known.values()), ensure_ascii=False)}

SCRIPT
{json.dumps([_script_panel(p) for p in sb.panels], ensure_ascii=False)}{paper}"""
