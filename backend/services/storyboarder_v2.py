"""
Storyboarder v2 — paper → brief → chaptered manga script (JSON), with budget fitting and translation.

Design rules (from the owner's objective + Seedream experiments):
- one idea per panel, ≤2 bubbles × ≤14 words, narration ≤20 words (zh/ja: ≤22 / ≤32 characters)
- all facts come from the paper; exact numbers; no invented results
- no underscores / LaTeX in on-image text (the image model drops them) — plain words or √ ² ×
- text is never truncated: over-budget lines are rewritten by the model in a "fit" pass

v3 (build_v3): the reader must come away knowing the abstract and having seen its key numbers. The brief splits the
abstract into claims with verified quotes; the story runs hook → problem → idea → how → proof → limits → payoff → recap
on one running example, with scene panels and board panels (exhibits drawn by code from the paper's exact values); a
deterministic lint (services.boards / services.evidence, budgets, claim coverage) drives up to two fix rounds; the
fact-check covers board strings; services.abstract_quiz gates the visible text before translation.
"""

from __future__ import annotations

import asyncio
import hashlib
import copy
import importlib
import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field, asdict
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Optional

from engines.base import BaseEngine, GenerationConfig, Message
from services.paper_extractor import PaperText
from services.reader_review import CONTRACT, is_novice, review_story, transcript as reader_transcript


# ----------------------------------------------------------------------------- data

def _as_list(value) -> list:
    """Model replies sometimes send one item where a list belongs ("labels": "Attention"); never iterate a str or dict."""
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _claim_ids(value) -> list[int]:
    """Claim ids of a reply: 3, "3", "C3", "#3", "claim 3", 3.0 and {"id": 3} all mean claim 3."""
    out = []
    for x in _as_list(value):
        if isinstance(x, dict):
            x = x.get("id", x.get("claim"))
        if isinstance(x, bool):
            continue
        if isinstance(x, float) and x.is_integer():
            x = int(x)
        m = re.fullmatch(r"\s*(?:claims?\s*|[C#]\s*)?(\d+)\s*", str(x), re.I)
        if m and int(m.group(1)) not in out:
            out.append(int(m.group(1)))
    return out


def _text(value) -> str:
    """A text field of a reply: a string (or number) stripped; a list or object is no text (str() would draw "['...']")."""
    return str(value).strip() if isinstance(value, (str, int, float)) and not isinstance(value, bool) else ""


def _labels(value) -> list[str]:
    """Label texts of a reply: a string is one label (never split into characters); non-text items are dropped."""
    return [str(x).strip() for x in _as_list(value) if isinstance(x, (str, int, float)) and str(x).strip()]


@dataclass
class Bubble:
    speaker: str
    text: str


@dataclass
class PanelV2:
    n: int
    chapter: str
    title: str
    characters: list[str]
    scene: str
    visual: str
    dialogue: list[Bubble] = field(default_factory=list)
    narration: str = ""
    labels: list[str] = field(default_factory=list)
    emotions: dict[str, str] = field(default_factory=dict)
    covers: list[str] = field(default_factory=list)   # ids of the brief's key points this panel explains
    detail: str = ""     # the precise facts behind the panel, shown as text below the picture (never drawn)
    # v3: "scene" (the cast acting) or "board" (an exhibit the program draws on a blank whiteboard in the scene)
    kind: str = "scene"
    board: Optional[dict] = None     # exhibit spec (services/boards.py), only for kind "board"
    beat: str = ""                   # story beat: hook|problem|idea|how|proof|limits|payoff|recap
    claims: list[int] = field(default_factory=list)   # indexes of StoryboardV2.abstract_claims this panel conveys
    shows: str = ""                  # what a reader understands from the picture alone (a scene's idea; the art check looks for it)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["dialogue"] = [asdict(b) for b in self.dialogue]
        return d

    @classmethod
    def from_dict(cls, d: dict, default_n: int) -> "PanelV2":
        def text(key: str) -> str:
            value = d.get(key)
            return str(value).strip() if value is not None else ""

        dialogue = []
        for b in _as_list(d.get("dialogue")):
            if isinstance(b, dict) and str(b.get("text") or "").strip():   # "  " is no line
                dialogue.append(Bubble(speaker=str(b.get("speaker") or "").strip().lower(), text=str(b["text"]).strip()))
        characters = d.get("characters")
        if isinstance(characters, str):
            characters = characters.split(",")
        elif isinstance(characters, dict):  # {"papi": "curious", ...}
            characters = list(characters)
        emotions = d.get("emotions") if isinstance(d.get("emotions"), dict) else {}
        try:
            n = int(d.get("n", default_n) or default_n)
        except (TypeError, ValueError):
            n = default_n
        return cls(
            n=n,
            chapter=text("chapter").lower() or "main",
            title=text("title"),
            characters=[str(c).strip().lower() for c in _as_list(characters) if c is not None and str(c).strip()],
            scene=text("scene"),
            visual=text("visual"),
            dialogue=dialogue,
            narration=text("narration"),
            labels=_labels(d.get("on_image_labels") or d.get("labels"))[:2],
            emotions={str(k).lower(): str(v).strip() for k, v in emotions.items() if v is not None and str(v).strip()},
            covers=list(dict.fromkeys(c.strip() for x in _as_list(d.get("covers")) if x is not None for c in str(x).split(",") if c.strip())),
            detail=text("detail"),
            kind="board" if text("kind").lower() == "board" or text("type").lower() == "board" else "scene",
            board=d.get("board") if isinstance(d.get("board"), dict) else None,
            beat=text("beat").lower(),
            claims=_claim_ids(d.get("claims", d.get("abstract_claims"))),
            shows=text("shows"),
        )


@dataclass
class Chapter:
    id: str
    title: str
    panel_numbers: list[int] = field(default_factory=list)


@dataclass
class StoryboardV2:
    title: str
    summary: str
    theme: str
    language: str
    chapters: list[Chapter]
    panels: list[PanelV2]
    brief: dict = field(default_factory=dict)
    card: dict = field(default_factory=dict)   # the whole paper in a few lines, shown as text in the reader
    # v3
    version: int = 2
    abstract_claims: list[dict] = field(default_factory=list)   # [{"id": 0, "text": "...", "quote": "..."}] from the paper's abstract
    running_example: str = ""
    quiz: dict = field(default_factory=dict)                    # last abstract-quiz result on the visible text

    def to_dict(self) -> dict:
        d = {
            "title": self.title,
            "summary": self.summary,
            "theme": self.theme,
            "language": self.language,
            "chapters": [asdict(c) for c in self.chapters],
            "panels": [p.to_dict() for p in self.panels],
            "brief": self.brief,
            "card": self.card,
        }
        if self.version >= 3:
            d.update({"version": self.version, "abstract_claims": self.abstract_claims, "running_example": self.running_example, "quiz": self.quiz})
        return d


@dataclass
class CastMember:
    name: str
    role: str          # mentor | student | skeptic
    look: str          # short physical description used in prompts
    voice: str         # how they talk


@dataclass
class Cast:
    theme: str
    style: str                 # art-style sentence for the image prompt
    members: list[CastMember]

    def names(self) -> list[str]:
        return [m.name for m in self.members]

    def describe(self) -> str:
        role_desc = {"mentor": "the professor/mentor", "student": "the student", "skeptic": "the critic"}
        return "\n".join(f"- {m.name}: {role_desc.get(m.role, m.role)}. Looks: {m.look}. Voice: {m.voice}" for m in self.members)


LANG_NAMES = {"en-US": "English", "zh-CN": "Simplified Chinese", "ja-JP": "Japanese"}
BUDGETS = {
    # bubble words (en) / characters (cjk), narration, max bubbles per panel
    "en-US": {"bubble": 14, "narration": 20, "label": 3, "unit": "words"},
    "zh-CN": {"bubble": 22, "narration": 32, "label": 6, "unit": "characters"},
    "ja-JP": {"bubble": 24, "narration": 34, "label": 8, "unit": "characters"},
}
PANEL_RANGES = {"short": (12, 18), "auto": None, "long": (28, 40)}
CHAPTER_PLAN = [
    ("hook", "Why should anyone care?"),
    ("problem", "The old problem"),
    ("idea", "The key idea"),
    ("how", "How it works"),
    ("evidence", "Does it work? The evidence"),
    ("payoff", "Back to the example"),
    ("limits", "Limits and open questions"),
    ("takeaway", "Takeaway"),
]


# ----------------------------------------------------------------------------- helpers

PEOPLE_RE = re.compile(r"\b(?:people|persons?|man|men|woman|women|boys?|girls?|child(?:ren)?|kids?|students?|pupils?|librarians?|teachers?"
                       r"|workers?|crowds?|audiences?|spectators?|guides?|hikers?|scientists?|researchers?|engineers?|doctors?|players?"
                       r"|judges?|referees?|customers?|visitors?|staff|humans?|figures?|clerks?|passengers?|tourists?|chefs?|bakers?|athletes?"
                       r"|drivers?|couriers?|employees?|agents?)\b", re.I)
NO_PEOPLE_RE = re.compile(r"\b(?:no|without|not any|no other|nobody|no one)\b(?:\s+\w+){0,3}?\s+(?:people|persons?|humans?|figures?|crowds?"
                          r"|audiences?|spectators?|staff|workers?|drivers?|couriers?|employees?|agents?)\b", re.I)


# a model or network named after a role is a concept: Right On Q's "teacher model" and "student model" were flagged as
# people to draw in every fix round, and the rewrite named them again; so is an object named after one ("a brass teacher
# weight", "a red driver cap") and a software agent
ROLE_CONCEPT_RE = re.compile(r"\b(?:teachers?|students?|workers?|judges?|players?|experts?|critics?|guides?|drivers?|couriers?|agents?|customers?)"
                             r"(?:['’]s\s+(?:[\w-]+\s+){0,4}|[\s-]+)(?:models?|networks?|nets?|polic(?:y|ies)|llms?|agents?|heads?|branch(?:es)?|weights?|bricks?"
                             r"|blocks?|cards?|cups?|box(?:es)?|stacks?|piles?|desks?|labs?|rooms?|caps?|hats?|bags?|vests?|badges?|scooters?"
                             r"|vans?|bikes?|books?|headsets?|phones?|handsets?)\b"
                             r"|\b(?:ai|llm|rl|software|routing|autonomous|multi)(?:[\s-]+\w+)?[\s-]+agents?\b", re.I)
# the cast member who plays a role is the cast member: the story prompt's cast list says "kumo: the student", and "Kumo
# the cloud student" or "Nezu, playing the driver" were flagged as people in every fit round (11 of 53 English scripts)
CAST_ROLE_RE = re.compile(r"\b(papi|kumo|nezu)\b(?:,?\s+(?:in|wearing)\s+(?:the|a|an|her|his|their)\s+(?:[\w-]+\s+){0,4}?[\w-]+,?(?=\s+(?:as|playing|plays|dressed as|in the role of)\b))?"
                          r",?\s+(?:(?:as|playing|plays|dressed as|in the role of)\s+)?(?:the|a|an)\s+"
                          r"(?:[\w-]+\s+){0,2}?(?:students?|teachers?|mentors?|professors?|critics?|skeptics?|drivers?|couriers?"
                          r"|workers?|agents?|customers?|employees?|clerks?)\b", re.I)
COPIES_RE = re.compile(r"\b(?:\d+|two|three|four|five|six|seven|eight|nine|ten|twelve|many|several|dozens? of)\s+(?:\w+\s+){0,3}?"
                       r"(?:papis?|kumos?|nezus?|dogs?|hedgehogs?|clouds?|copies|clones|versions|twins)\b"
                       r"|\b(?:clones?|copies|mini(?:ature)?|tiny|little|small)\s+(?:versions?\s+of\s+)?(?:papis?|kumos?|nezus?)\b", re.I)


def _mentions_people(text: str) -> bool:
    """A scene that names people to draw ("eight small librarians") or copies of a cast member ("eight identical small
    papis"), not one that says there are none, nor a concept named after a role ("teacher model", "student network"), nor
    the cast member who plays one ("Kumo the cloud student")."""
    text = ROLE_CONCEPT_RE.sub(" ", NO_PEOPLE_RE.sub(" ", CAST_ROLE_RE.sub(r"\1", text)))
    return bool(PEOPLE_RE.search(text) or COPIES_RE.search(text))


DRAWN_CLAIM_RE = re.compile(r"\d|\b(?:charts?|bars?|graphs?|plots?|stacks?|compare[sd]?|comparison|taller|shorter|bigger|smaller)\b", re.I)
CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff66-\uff9f]")
LATIN_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-./][A-Za-z0-9]+)*")
# a web address: kept exact by a translation, and never lettered (a reader cannot click a picture; the paper card has the links)
URL_RE = re.compile(r"\b(?:https?://|www\.)\S+|\b(?-i:[a-z0-9][\w-]*(?:\.[\w-]+)*\.(?:com|org|io|net|edu|ai|dev|co|cn|gov|me|app|"
                    r"info|xyz|ly|gg|sh|tv|uk|de|jp|eu|us))\b(?:/\S*)?", re.I)   # a name like Booking.com or Character.AI is no address


def _code_like(token: str) -> bool:
    """Numbers, acronyms and CamelCase names (28.4, BLEU, GPT-4, ResNet-152) that a translation keeps in Latin script."""
    return any(c.isdigit() for c in token) or sum(c.isupper() for c in token) >= 2


def _latin_keep(brief: dict) -> frozenset:
    """Lower-cased words a translation may keep in Latin script: the brief's capitalised words (Transformer) and glossary terms."""
    text = json.dumps({k: v for k, v in brief.items() if k != "factcheck"}, ensure_ascii=False)
    terms = " ".join(str(g.get("term") or "") if isinstance(g, dict) else str(g) for g in _as_list(brief.get("glossary")))
    return frozenset(w.lower() for w in re.findall(r"\b[A-Z][A-Za-z]{3,}", text) + re.findall(r"[A-Za-z]{4,}", terms))


def _has_script(text: str, language: str, keep: frozenset = frozenset()) -> bool:
    """True when a translated line contains the target script, or only what the translation prompt says to keep exact:
    numbers, acronyms and CamelCase names ("28.4 BLEU！", "ResNet-152？") and the words in ``keep`` ("Transformer！")."""
    if language not in ("zh-CN", "ja-JP"):
        return True
    if CJK_RE.search(text):
        return True
    # a line that is only a web address has nothing to translate (1000 Layer RL's Chinese panel with the project's
    # address kept both of its bubbles in English: the whole panel's translation was rejected twice)
    rest = LATIN_TOKEN_RE.sub(lambda m: " " if _code_like(m.group()) else m.group(), URL_RE.sub(" ", text))
    return all(w.lower() in keep for w in re.findall(r"[A-Za-z]{4,}", rest))


def text_len(text: str, language: str) -> int:
    """English: words. Chinese/Japanese: display width (full-width = 1, ASCII letters/digits = ½), so
    "Transformer纯用注意力" counts 12, not 17."""
    if BUDGETS.get(language, BUDGETS["en-US"])["unit"] == "words":
        return len(text.split())
    compact = re.sub(r"\s+", "", text)
    return int(math.ceil(sum(0.5 if ord(ch) < 128 else 1.0 for ch in compact)))


def _span_end(text: str, start: int, open_ch: str, close_ch: str) -> Optional[int]:
    """End index of the open_ch...close_ch span starting at ``start`` (string-aware); None when it never closes."""
    depth, in_str, esc = 0, False, False
    for j in range(start, len(text)):
        ch = text[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return j + 1
    return None


def _balanced_objects(text: str):
    """Yield every top-level {...} substring (string-aware), in order."""
    i = text.find("{")
    while i != -1:
        end = _span_end(text, i, "{", "}")
        if end is None:
            return
        yield text[i:end]
        i = text.find("{", end)


def _top_level_lists(text: str) -> list[str]:
    """Every [...] that is not inside an object (e.g. a bare list between prose), in order. None at all when an object
    never closes, so a truncated or malformed reply still goes to the repair round."""
    found, k = [], 0
    while True:
        i, j = text.find("[", k), text.find("{", k)
        if j != -1 and (i == -1 or j < i):
            k = _span_end(text, j, "{", "}")
            if k is None:
                return []
            continue
        if i == -1:
            return found
        end = _span_end(text, i, "[", "]")
        if end is None:  # a stray "[" in prose
            k = i + 1
            continue
        found.append(text[i:end])
        k = end


LONE_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")
# Seed cannot write "[CLS]": it emits this placeholder instead (same string in every job), which then reached the reader
LEAKED_TOKEN_RE = re.compile(r"\s?<\[(BOS|[A-Z]+)_never_used_[0-9a-f]+\]>")


def scrub_surrogates(value):
    """Copy of a JSON-like value with every lone UTF-16 surrogate replaced by U+FFFD (json.loads turns a model's half
    emoji escape "\\ud83d" into one, and UTF-8 cannot encode it), the model's leaked "[CLS]" placeholder restored, and
    NaN / Infinity (which json.loads accepts) made null, so storyboard.json stays valid JSON for the browser."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, str):
        value = LEAKED_TOKEN_RE.sub(lambda m: " [CLS]" if m.group(1) == "BOS" else "", value)
        return LONE_SURROGATE_RE.sub("\ufffd", value)
    if isinstance(value, dict):
        return {scrub_surrogates(k): scrub_surrogates(v) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub_surrogates(v) for v in value]
    return value


def _loads(candidate: str):
    """json.loads (retried once without trailing commas), without lone surrogates; raw control characters inside strings
    (a newline in a bubble) are accepted: two quiz gates were lost to "Invalid control character" alone."""
    try:
        return scrub_surrogates(json.loads(candidate, strict=False))
    except json.JSONDecodeError:
        return scrub_surrogates(json.loads(re.sub(r",\s*([}\]])", r"\1", candidate), strict=False))


def extract_json(text: str, key: Optional[str] = None) -> dict:
    """Pull a JSON object out of a model reply; tolerate code fences, prose, trailing commas and bare lists.

    With ``key`` the first object containing that key wins, then one holding it one level down (e.g. {"result": {...}}),
    then a JSON list is wrapped as {key: [...]}: the whole reply, a fenced list, or a list of objects between prose.
    A malformed or cut-off payload raises (never a fallback to a list item or a "[]" from the prose), so the caller's
    repair round runs.
    """
    fenced = re.findall(r"```(?:json)?\s*([\[{].*?[\]}])\s*```", text, re.S | re.I)
    stripped = text.strip()
    candidates = fenced + ([stripped] if stripped.startswith("[") else []) + list(_balanced_objects(text))
    parsed: list = []
    broken = False  # a fenced block or {...} that does not parse: the payload is there but malformed
    last_error: Exception = ValueError("no JSON object in model reply")
    for candidate in candidates:
        try:
            parsed.append(_loads(candidate))
        except json.JSONDecodeError as e:
            last_error, broken = e, broken or candidate != stripped
    if key:
        for data in parsed:
            if isinstance(data, dict) and key in data:
                return data
        for data in parsed:
            if isinstance(data, dict):
                for v in data.values():
                    if isinstance(v, dict) and key in v:
                        return v
        for data in parsed:
            if isinstance(data, list):
                return {key: data}
        lists = []
        for candidate in [] if broken else _top_level_lists(text):  # a broken payload goes to the repair round
            try:
                lists.append(_loads(candidate))
            except json.JSONDecodeError:
                continue
        no_object = not any(isinstance(d, dict) for d in parsed)  # an empty list never replaces a parsed object
        for data in [x for x in lists if any(isinstance(i, dict) for i in x)] + [x for x in lists if not x and no_object]:
            return {key: data}  # "[1]" in prose is a citation, not the payload
        first = text.find("{")
        if first != -1 and text[:first].rstrip().endswith("["):  # a list of objects that does not parse (cut off?)
            raise ValueError("model reply is a JSON list that does not parse")  # never return one of its items
    if not parsed:
        raise last_error
    for data in parsed:
        if isinstance(data, dict):
            return data
    raise ValueError("model reply contained no JSON object")


PAPER_PREFIX = "You work on one research paper for an educational manga. The paper comes first; your task follows it.\n\n"


def _messages(system: str, user: str, paper_text: str = "") -> list[Message]:
    """Paper first: a prompt that ends with the paper text ("PAPER TEXT:\n<body>") sends the paper as the first message,
    identical in every call of a job, so the provider serves it from its prefix cache (cached input costs a fifth on
    Seed and a thirtieth on DeepSeek); the role and the task follow it. Other prompts go as they are."""
    if paper_text:
        for body in (paper_text, paper_text.strip()):
            marker = "\n\nPAPER TEXT:\n" + body
            if body and user.endswith(marker):
                head = [Message.system(PAPER_PREFIX + "PAPER TEXT:\n" + paper_text.strip())]
                return head + ([Message.system(system)] if system else []) + [Message.user(user[:-len(marker)])]
    return [Message.system(system), Message.user(user)] if system else [Message.user(user)]


async def _ask_json(engine: BaseEngine, system: str, user: str, *, thinking: bool, temperature: float, max_tokens: int = 16000,
                    schema: Optional[dict] = None, key: Optional[str] = None, paper: str = "", cache_key: str = "") -> tuple[dict, dict]:
    """One JSON call (with one repair round); ``paper`` (the paper text the prompt ends with) goes first for the prefix
    cache, ``cache_key`` keeps a job's calls on one cache."""
    config = GenerationConfig(temperature=temperature, max_tokens=max_tokens)
    config.thinking = thinking
    config.prompt_cache_key = cache_key or None
    if schema is not None:
        config.response_format = {"type": "json_schema", "json_schema": {"name": "out", "schema": schema, "strict": True}}
    messages = _messages(system, user, paper)
    config.cache_prefix = bool(paper) and messages[0].content.startswith(PAPER_PREFIX)
    response = await engine.generate_text(messages, config)
    usage = dict(response.usage or {})
    if not (response.content or "").strip():   # nothing to repair: a "repaired" empty reply would read as "no problems found"
        raise ValueError(f"the model sent an empty reply (finish_reason={response.finish_reason or 'unknown'})")
    try:
        data = extract_json(response.content, key=key)
    except (ValueError, json.JSONDecodeError):
        # one repair round without schema constraints
        repair = GenerationConfig(temperature=0.0, max_tokens=max_tokens)
        repair.thinking = False
        fixed = await engine.generate_text(
            [Message.user("Return the following as strictly valid JSON (same content, no commentary):\n\n" + scrub_surrogates(response.content[:60000]))], repair)
        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
            usage[k] = int(usage.get(k, 0) or 0) + int((fixed.usage or {}).get(k, 0) or 0)
        try:
            data = extract_json(fixed.content, key=key)
        except (ValueError, json.JSONDecodeError) as e:   # say what came back: the log alone could not tell (Right On Q's quiz)
            raise ValueError(f"no usable JSON after a repair round: {e} (finish_reason {response.finish_reason or 'unknown'}; "
                             f"the reply began {scrub_surrogates(response.content[:60])!r})") from e
    return data, usage


async def _ask_check(engine: BaseEngine, user: str, **kwargs) -> tuple[dict, dict]:
    """A check part (``_ask_json``), asked once more when its reply was empty or no repair could parse it: checks run in
    parallel parts, and one lost part leaves its panels unchecked (a DeepSeek part of Right On Q's fact-check once came
    back empty with finish_reason "stop"). Account and content-filter errors are not ValueErrors and pass through."""
    try:
        return await _ask_json(engine, "", user, **kwargs)
    except ValueError as e:   # json.JSONDecodeError is a ValueError
        print(f"[StoryboarderV2] a check reply could not be used ({e}); asking once more")
        return await _ask_json(engine, "", user, **kwargs)


# ----------------------------------------------------------------------------- prompts

SYSTEM_WRITER = (
    "You are a science communicator and manga scriptwriter. You write accurate, warm, lightly funny scripts that teach "
    "real research papers to smart readers who have ADHD: short lines, one idea per panel, concrete visual metaphors, "
    "frequent payoffs, zero filler, zero hallucination."
)


def brief_prompt(paper: PaperText) -> str:
    caps = "\n".join(paper.captions[:20])
    return f"""Read the paper and write a compact, faithful BRIEF for a manga adaptation.

Rules: every statement must come from the paper; copy exact numbers, dataset names, baselines and units; if something is unclear say "unclear" instead of guessing.

Output JSON only:
{{
 "title": "paper title",
 "field": "one phrase",
 "problem": "2-3 sentences: what was hard/slow/wrong before, in plain language",
 "key_idea": "ONE sentence a smart teenager could repeat",
 "contributions": ["each contribution the authors claim, one sentence each"],
 "mechanisms": [{{"name": "...", "plain": "2 sentences, no jargon", "on_example": "what this mechanism does to the running example, step by step, with its real things (tokens, documents, layers, calls)", "exact": "the precise technical statement with symbols spelled in words"}}],
 "setup": "1-3 sentences: the data, baselines, metrics and model sizes of the experiments",
 "evidence": [{{"claim": "...", "numbers": "exact figures with units, dataset, split, baseline, single model or ensemble", "why_it_matters": "..."}}],
 "limitations": ["only limitations the authors state"],
 "takeaway": "1-2 sentences: the paper's own conclusion",
 "glossary": [{{"term": "...", "plain": "..."}}],
 "fun_facts": ["optional surprising details worth a joke"],
 "points": [{{"id": "P1", "kind": "problem|contribution|mechanism|setup|result|analysis|limitation", "priority": "core|important|detail", "text": "one precise statement with the exact numbers and setting"}}]
}}
"mechanisms": every component a reader needs to understand how the method works (usually 3-8). "evidence": every main result and key ablation or analysis (usually 4-8): go through every results table and figure, including the appendix, and keep each number with its exact setting. "limitations": only what the authors themselves say; never add your own.
"points": 15-30 points that together cover the whole paper: its problem, every contribution, every mechanism, the experimental setup, every main result and key analysis or ablation, and the limitations the authors state. priority "core" = a reader cannot understand the paper without it (usually 8-14 points); "important" = a curious reader should get it; "detail" = everything else worth knowing.

FIGURE/TABLE CAPTIONS (for orientation):
{caps}

PAPER TEXT:
{paper.body}"""


def _beginner_plan_leaves(brief: dict) -> dict:
    """Exact mutable string leaves, only for the opted-in novice contract."""
    plan = brief.get("beginner_plan")
    if brief.get("reader_contract") != CONTRACT or not isinstance(plan, dict):
        return {}
    leaves = {}

    def add(path, parent, key):
        value = parent[key]
        if isinstance(value, str) and value.strip():
            leaves[path] = (parent, key)

    if "goal" in plan:
        add("beginner_plan.goal", plan, "goal")
    for field in ("foundations", "opening_questions"):
        if isinstance(plan.get(field), list):
            for i in range(len(plan[field])):
                add(f"beginner_plan.{field}.{i}", plan[field], i)
    if isinstance(plan.get("steps"), list):
        for i, step in enumerate(plan["steps"]):
            if isinstance(step, dict):
                for field in ("what", "why"):
                    if field in step:
                        add(f"beginner_plan.steps.{i}.{field}", step, field)
    return leaves


def _brief_items(brief: dict) -> list[dict]:
    """Every checkable statement of the brief with an id that says where it lives."""
    items = [{"id": key, "text": brief[key]} for key in ("problem", "key_idea", "setup", "takeaway")
             if isinstance(brief.get(key), str) and brief[key].strip()]
    for key in ("contributions", "limitations"):
        items += [{"id": f"{key}.{i}", "text": t} for i, t in enumerate(_as_list(brief.get(key))) if isinstance(t, str) and t.strip()]
    for i, m in enumerate(_as_list(brief.get("mechanisms"))):
        if isinstance(m, dict):
            items += [{"id": f"mechanisms.{i}.{f}", "text": m[f]} for f in ("plain", "exact") if isinstance(m.get(f), str) and m[f].strip()]
    for i, e in enumerate(_as_list(brief.get("evidence"))):
        if isinstance(e, dict):
            items += [{"id": f"evidence.{i}.{f}", "text": e[f]} for f in ("claim", "numbers") if isinstance(e.get(f), str) and e[f].strip()]
    items += [{"id": f"points.{pt['id']}", "text": str(pt.get("text") or "")} for pt in _points(brief) if str(pt.get("text") or "").strip()]
    items += [{"id": f"abstract_claims.{i}", "text": c["text"]} for i, c in enumerate(_as_list(brief.get("abstract_claims")))
              if isinstance(c, dict) and isinstance(c.get("text"), str) and c["text"].strip()]
    example = brief.get("running_example")
    if isinstance(example, dict) and example.get("from_paper"):   # a made-up case is checked as the story uses it
        items += [{"id": f"running_example.{f}", "text": example[f]} for f in ("text", "old_way", "new_way")
                  if isinstance(example.get(f), str) and example[f].strip()]
        items += [{"id": f"running_example.parts.{i}", "text": _part_line(t, r)} for i, (t, r) in enumerate(_part_pairs(example))
                  if t.strip()]
        if isinstance(example.get("text"), str) and example["text"].strip():   # an average is no instance of the paper's
            items.append({"id": "running_example.from_paper",
                          "text": f"The paper itself shows this one instance ({example.get('source') or 'no place given'}), "
                                  f"not only averages or a pattern over many inputs: {example['text']}"})
    items += [{"id": pid, "text": parent[key]} for pid, (parent, key) in _beginner_plan_leaves(brief).items()]
    return items


def brief_check_prompt(brief: dict, paper: PaperText, items: Optional[list[dict]] = None) -> str:
    plan_rules = ""
    if brief.get("reader_contract") == CONTRACT:
        plan_rules = ("\nFor beginner_plan, check the factual presuppositions of questions as well as statements. "
                      "Keep goal as a concrete task or research aim, never turn an aim into a measured achievement. "
                      "Correct opening_questions as plain questions, with no unexplained abbreviations or results. "
                      "Do not presuppose perfect reference answers, inevitable failure, complete recovery, or speedups "
                      "unless the paper supports them. The quote field is source evidence, not replacement text: "
                      "corrected must preserve the leaf's role (question, goal, foundation, step, or reason). "
                      "Use DELETE if a supported question or statement cannot be supplied. "
                      "Return only exact item ids supplied in this request; never modify sibling fields.\n")
    return f"""You verify the BRIEF of a research paper before it is turned into a manga; every later step trusts it.
For EVERY item below, find the sentence, table or figure of the PAPER that supports it. An item passes only if the paper supports it with the same numbers, the same setting (model size, single model or ensemble, dataset, split, which variant) and the same strength and scope. Flag items that are unsupported, overstated (all, first, only, best, always, completely), misattributed or wrong, and limitations the authors do not state.
For each flagged item give "corrected": the item rewritten so the paper supports it (same role, roughly the same length), or "DELETE" when nothing in the paper supports it.
{plan_rules}Output JSON only: {{"problems": [{{"id": "...", "issue": "...", "quote": "the paper's own words", "corrected": "..."}}]}} — an empty list when everything is supported.

ITEMS:
{json.dumps(_brief_items(brief) if items is None else items, ensure_ascii=False)}

PAPER TEXT:
{paper.body}"""


def storyboard_prompt(brief: dict, cast: Cast, language: str, panel_range: tuple[int, int]) -> str:
    lo, hi = panel_range
    budget = BUDGETS["en-US"]
    chapters = "\n".join(f"- {cid}: {title}" for cid, title in CHAPTER_PLAN)
    return f"""Write the manga script for this paper brief. Write it in English (it is translated later).

CAST (use these lowercase names exactly; keep everyone in role)
{cast.describe()}

CHAPTERS (use these chapter ids in order; every chapter gets at least one panel; "how" gets one panel PER mechanism)
{chapters}

HARD RULES
- Between {lo} and {hi} panels total; choose the count by how much the paper actually contains. Number panels 1..N in reading order.
- ONE idea per panel. Max 2 speech bubbles per panel, each ≤ {budget['bubble']} words. Optional narration box ≤ {budget['narration']} words that states the precise technical fact (numbers go here).
- Every number, name and claim must come from the brief. Never invent results. Use the exact figures.
- No absolute claims (all, every, always, never, nobody) unless the paper itself makes them. No claims about the paper's later impact, adoption or today's practice ("changed everything", "now standard", "to this day"): the takeaway restates the paper's own conclusion.
- Keep the paper's qualifiers (deep, many, most, on this dataset) and tie every number to its exact setting: model size, single model or ensemble, dataset and split. A trend is never turned into a hard threshold.
- Describe each method exactly as the paper does: a simplification may drop detail but must not change what happens. Limitations are only those the paper states; never invent a weakness. Never claim a model beats humans unless the paper does, and credit a gain to everything the paper credits it to.
- "visual": concrete and drawable — poses, expressions, props and the metaphor. Mention only characters listed in "characters". Only the characters listed in "characters" appear. Metaphors use objects, places, diagrams and the cast only — never extra people, children, crowds or other creatures (no guides, hikers, students, librarians). Never ask for written words, numbers or formulas in the picture ("a sentence on the board", "a line of text with one word blacked out", "sums on a chalkboard"): show words as a row of blank cards or blocks, one of them covered or highlighted, and amounts as bars or piles; numbers belong in the narration or labels. Never draw copies or miniature versions of a cast member: show counts with simple objects (eight colored cards, not eight little papis).
- "on_image_labels": ≤ 2 short labels (≤ 3 words) drawn on props/boards; everything else in the picture is wordless.
- On-image text (bubbles, narration, labels) must not contain underscores, LaTeX, backslashes or markdown. Write math in words or with plain symbols (√, ², ×, ÷, ≈). Avoid quotation marks inside bubbles.
- Every panel carries at least one fact from the brief, the hook and the takeaway included (the hook states the problem, the takeaway the main result); no filler panels and no panel that repeats an earlier one. The narration adds a fact the bubbles do not say; it never restates a bubble. The first time a metric or technical term appears (BLEU, F1, a dataset), explain it in plain words using the glossary, in the narration or the detail.
- COVERAGE: each panel lists in "covers" the ids of the brief points it explains. Every "core" point must be covered by at least one panel, and "important" points as far as the panel budget allows; choose the panel count so that every core point fits (one main point per panel; a closely related second point may go in the narration or the detail).
- "detail": 1-3 sentences (≤ 60 words) for readers who want more: the points this panel covers, restated plainly with their exact numbers and settings, and the mechanism in technical terms. Add nothing the points do not say: no judgments such as first, only, best, optimal, all, order of magnitude, unless a point says it. It is shown as text below the picture and never drawn.
- If a visual compares amounts (bars, stacks, piles), the larger or better one in the paper must look larger; never draw a comparison that contradicts the narration.
- Emotions: give each character in the panel one emotion word.
- Pacing for ADHD readers: open with a hook that answers "why should I care", give a payoff every 2-3 panels, let the student ask what the reader is wondering, let the critic voice the limitation, end with a memorable takeaway.

OUTPUT: JSON only, exactly this shape:
{{"title": "short manga title", "summary": "one sentence", "panels": [{{"n": 1, "chapter": "hook", "title": "...", "characters": ["papi", "kumo"], "scene": "...", "visual": "...", "dialogue": [{{"speaker": "papi", "text": "..."}}], "narration": "...", "on_image_labels": ["..."], "emotions": {{"papi": "curious"}}, "covers": ["P1"], "detail": "..."}}]}}

BRIEF:
{json.dumps(brief, ensure_ascii=False)}"""


def fit_prompt(items: list[dict], language: str) -> str:
    budget = BUDGETS.get(language, BUDGETS["en-US"])
    return f"""Rewrite each text so it fits the limit while keeping the exact meaning, numbers and tone ({LANG_NAMES.get(language, language)}).
Limits: speech bubble ≤ {budget['bubble']} {budget['unit']}, narration ≤ {budget['narration']} {budget['unit']}, label on a prop ≤ {budget['label']} {budget['unit']}. No underscores, LaTeX or markdown.
A "visual" item is a scene for the picture that mentions people or copies of a cast member: rewrite it as ONE English sentence (scene and action together) that shows the same idea with objects, places, diagrams and the cast only (each cast member once; counts as simple objects such as cards), and states that nobody else is there.
Return JSON only: {{"items": [{{"id": "...", "text": "rewritten"}}]}}

ITEMS:
{json.dumps(items, ensure_ascii=False)}"""


FACTCHECK_CHUNK = 30   # lines per fact-check call: a long script is checked in parallel parts (a thinking checker takes minutes)


def _factcheck_items(sb: StoryboardV2, paper: PaperText, language: str, panels: Optional[set] = None, card: bool = True) -> list[dict]:
    items = []
    for p in sb.panels:
        if panels is not None and p.n not in panels:
            continue
        for i, b in enumerate(p.dialogue):
            items.append({"id": f"{p.n}.d{i}", "speaker": b.speaker, "text": b.text})
        if p.narration:
            items.append({"id": f"{p.n}.n", "text": p.narration})
        if p.detail:
            items.append({"id": f"{p.n}.x", "text": p.detail})
        # numbers and comparisons in a scene description end up drawn, and so do those in the idea the picture must show;
        # the art directions stay English in every language (the translation never touches them), so only the English
        # fact-check reads them (a Chinese correction reached the image model and the idea check of a zh-CN book)
        if language == "en-US" and p.visual and DRAWN_CLAIM_RE.search(p.visual):
            items.append({"id": f"{p.n}.v", "text": p.visual})
        if language == "en-US" and p.shows and DRAWN_CLAIM_RE.search(p.shows):
            items.append({"id": f"{p.n}.s", "text": p.shows})
        if p.board:
            items += _board_check_items(p, whole=language == "en-US", paper_text=paper.body)
    for key, value in (sb.card.items() if card else ()):
        for i, text in enumerate([value] if isinstance(value, str) else value):
            items.append({"id": f"card.{key}.{i}", "text": text})
    return items


def factcheck_parts(sb: StoryboardV2, paper: PaperText, language: str = "en-US", size: int = FACTCHECK_CHUNK,
                    only: Optional[set] = None, card: Optional[bool] = None) -> list[tuple[set, bool]]:
    """The script (or the panels in ``only``) cut into whole panels of about ``size`` fact-check lines each: (panel
    numbers, with the card). The first part carries the card unless ``card`` is False (default: only for the whole
    script)."""
    parts, current, count = [], set(), 0
    for p in sb.panels:
        if only is not None and p.n not in only:
            continue
        n = len(_factcheck_items(sb, paper, language, {p.n}, card=False))
        if current and count + n > size:
            parts.append(current)
            current, count = set(), 0
        current.add(p.n)
        count += n
    if current or not parts:
        parts.append(current)
    with_card = only is None if card is None else card
    return [(panels, i == 0 and with_card) for i, panels in enumerate(parts)]


def _card_view(sb: StoryboardV2) -> str:
    """The paper card as the fact-check reads it."""
    return json.dumps(sb.card, sort_keys=True, ensure_ascii=False, default=str)


def _checked_view(p: PanelV2) -> str:
    """What the English fact-check reads of a panel (as _factcheck_items does): a panel whose view was checked once needs
    no second check, whatever its number has become since."""
    return json.dumps([[b.speaker, b.text] for b in p.dialogue] + [p.narration, p.detail, p.visual if p.visual and DRAWN_CLAIM_RE.search(p.visual) else "",
                       p.shows if p.shows and DRAWN_CLAIM_RE.search(p.shows) else "", _board_view(p.board)],
                      sort_keys=True, ensure_ascii=False, default=str)


def _board_view(board):
    """A board as the fact-check reads it: a designed diagram's texts (where they sit is no claim, so a drawing moved
    around a corrected label needs no second read), any other board as it is."""
    if not (isinstance(board, dict) and str(board.get("kind") or "").lower() == "diagram" and board.get("elements")):
        return board
    diagram = _optional("diagram")
    try:
        texts = sorted(t for _, t in diagram.texts(board)) if diagram is not None else None
    except Exception:   # a drawing not normalized yet: read as it is
        texts = None
    if texts is None:
        return board
    return {**{k: board.get(k) for k in ("kind", "title", "subtitle", "note", "source") if board.get(k)}, "texts": texts}


def factcheck_prompt(sb: StoryboardV2, paper: PaperText, language: str = "en-US", panels: Optional[set] = None, card: bool = True) -> str:
    items = _factcheck_items(sb, paper, language, panels, card)
    budget = BUDGETS.get(language, BUDGETS["en-US"]) if sb.version < 3 else _v3_budget(language)["scene"]
    unit = "words" if language == "en-US" else "characters"
    boards = _board_check_rules(language) if any(p.board for p in sb.panels) else ""
    target = ("" if language == "en-US" else
              f"\nThe lines are in {LANG_NAMES[language]}: judge them in that language and write every corrected line in natural "
              f"{LANG_NAMES[language]} with full-width punctuation, keeping technical terms, names and numbers exact.")
    return f"""You are a meticulous fact-checker for an educational manga about a research paper.
Compare EVERY line below against the PAPER TEXT. A line passes only if a sentence, table or figure of the paper supports it with the same strength and scope; simplified wording is fine when a reader of the paper would agree with it. Flag a line when it:
- states a number, name, comparison, mechanism or claim that the paper does not support;
- attributes a figure to the wrong model or setting (single model vs ensemble, validation vs test, wrong dataset or model size; e.g. a Transformer-big cost presented as a baseline cost);
- distorts the mechanism (e.g. "divide by the number of queries" when the paper divides by the square root of the key dimension; "only the last layer is trained" when all parameters are fine-tuned);
- overgeneralises: an absolute claim (all, every, always, completely, the whole field) the paper does not make, a result on some tasks stated for all tasks, a gradual trend stated as a hard threshold, or a qualifier of the paper's claim dropped (deep, many, most, on this dataset);
- claims that the method produced a specific output on the story's example ("it translated the sentence correctly", "it got the answer right", "it found the link perfectly") when the paper does not show that output for that example; what a figure of the paper shows may be described as the figure shows it, with the paper's hedges; the corrected line says what the paper's result means for problems like the example, or what its figure shows, never what the paper does not show ("not this specific example", "the paper gives no answer for it");
- claims impact, adoption or present-day status the paper does not state ("changed everything", "now a standard building block", "to this day", "every phone uses this"; also 至今, 直到现在, 现在已经, 今でも, 現在では): an open question is one "the paper leaves open", not one unsolved today;
- says a model beats humans when the paper does not, or compares against only the weaker of several human baselines;
- credits one ingredient with all or most of a gain that the paper attributes to several (e.g. bidirectionality alone when the paper credits it together with the pre-training tasks);
- states a limitation or weakness the paper does not report, or the opposite of what it reports.
Casual tone, simplification and metaphors are fine; do not flag style. When a number is wrong, the corrected line states the paper's figure (with its setting) instead of dropping the number. Ids ending in .v (the scene) or .s (what the picture must make clear) are descriptions that get drawn: flag any number, comparison or claim in them that the paper does not support (a chart must not show the wrong side as bigger); their corrected text still describes only the drawn objects (the same objects, in the right order or size), never abbreviations, labels or written words.{boards}{target}
For each problem give a corrected line that keeps the same meaning intent and roughly the same length (bubbles ≤ {budget['bubble']} {unit}, narration ≤ {budget['narration']} {unit}; ids ending in .x are a panel's detail and ids starting with card. are the paper card, both reader text of up to 60 words), no underscores or LaTeX.
Output JSON only: {{"problems": [{{"id": "4.n", "issue": "...", "corrected": "..."}}]}} — use an empty list when everything is supported.

LINES:
{json.dumps(items, ensure_ascii=False)}

PAPER TEXT:
{paper.body}"""


def _translation_panel(p: PanelV2) -> dict:
    d = {"n": p.n, "title": p.title, "dialogue": [{"speaker": b.speaker, "text": b.text} for b in p.dialogue], "narration": p.narration,
         "labels": list(p.labels), "detail": p.detail}
    if p.board:   # v3: the strings drawn on the board, keyed by their path
        d["board"] = {str(path): text for path, text in _board_strings(p.board)}
    return d


# the cast's props and running jokes, translated the same way in every manga (9 of 14 Chinese scripts made the critic's
# berry tart an egg tart, 蛋挞, which the art never shows)
FIXED_TERMS = {"zh-CN": {"berry tart": "浆果挞", "tart": "浆果挞", "berries": "浆果", "professor": "教授", "PhD student": "博士生",
                         "critic": "评论家", "skeptic": "怀疑派"}}


def _fixed_terms(language: str) -> str:
    terms = FIXED_TERMS.get(language)
    return ("\n- Fixed terms (use exactly these): " + "; ".join(f"{en} = {zh}" for en, zh in terms.items())) if terms else ""


def translate_prompt(sb: StoryboardV2, language: str) -> str:
    budget = BUDGETS[language]
    glossary = [g for g in (sb.brief.get("glossary") or [])][:30]
    names = ", ".join(sorted({str(n).strip().capitalize() for p in sb.panels for n in [*p.characters, *(b.speaker for b in p.dialogue)]
                              if str(n).strip()})) or "the cast's names"
    payload = {
        "title": sb.title,
        "summary": sb.summary,
        "chapters": [{"id": c.id, "title": c.title} for c in sb.chapters],
        "panels": [_translation_panel(p) for p in sb.panels],
        "card": sb.card,
    }
    if sb.version >= 3:
        scene, board = _v3_budget(language)["scene"], _v3_budget(language)["board"]
        limits = (f"each bubble ≤ {scene['bubble']} characters and narration ≤ {scene['narration']} characters on a scene panel; on a "
                  f"panel with a board, each bubble ≤ {board['bubble']} and narration ≤ {board['narration']}; panel title ≤ 12 characters; "
                  f"board strings about as short as the English")
        fields = "labels, detail, every string value of a panel's \"board\" (keep its keys) and every string in card"
    else:
        limits = (f"each bubble ≤ {budget['bubble']} characters, narration ≤ {budget['narration']} characters, panel title ≤ 12 characters, "
                  f"each label ≤ 6 characters")
        fields = "labels, detail and every string in card"
    return f"""Translate this manga script into natural, conversational {LANG_NAMES[language]} for readers who are new to the topic.
Rules:
- Translate ONLY the string values of title, summary, chapter titles, panel titles, dialogue[].text, narration, {fields}. Keep every key, every "n" and every "speaker" unchanged.
- Keep every qualifier and hedge (deep, many, most, some, about, up to, on this dataset) and the claim's exact strength: never strengthen or generalise a statement.
- Keep model names, datasets, metrics and numbers exact (e.g. BLEU, WMT 2014, 28.4, Transformer, GSM8K). Widely used English acronyms (RNN, CNN, GPU, Q/K/V) stay in English; labels that are pure acronyms or numbers stay unchanged. A general term gets its {LANG_NAMES[language]} name, with the English abbreviation in parentheses the first time ("大语言模型（LLM）", "思维链（CoT）").
- Write the way a {LANG_NAMES[language]} science comic for teenagers talks: short, lively sentences in natural word order. Never copy the English sentence structure, appositives ("Kumo, a PhD student, ...") or idioms word for word.
- The cast keep their names in Latin letters ({names}): never transliterate them (their name tabs in the pictures are Latin); roles may be translated ("Kumo，博士生").{_fixed_terms(language)}
- Length limits: {limits} — rephrase to fit while keeping the meaning.
- No underscores, LaTeX or markdown in any text. Use full-width punctuation appropriate for {LANG_NAMES[language]}.
Glossary (term → plain meaning): {json.dumps(glossary, ensure_ascii=False)}
Output JSON only with the same structure.

{json.dumps(payload, ensure_ascii=False)}"""


# a translation that is stronger than its English line (7 of 38 Chinese fact-check findings were added in translation:
# "extremely hard" became 根本没法, "solved" 完美解决), or drops the English line's hedge
ZH_STRONG_RE = re.compile(r"完美|完全|根本|彻底|所有|全部|(?<!第)一定|绝对|总是|永远|毫无|全都|完胜|碾压|任何")
EN_STRONG_RE = re.compile(r"\b(?:perfect(?:ly)?|complete(?:ly)?|entire(?:ly)?|total(?:ly)?|fully|all|every(?:thing|one)?|always|never|"
                          r"absolute(?:ly)?|definitely|whole|any|nothing|none|no)\b", re.I)
HEDGES = [   # (English hedge, the Chinese words that keep it)
    (re.compile(r"\bmost(?:ly)?\b", re.I), re.compile(r"大多|多数|大部分|大都")),
    (re.compile(r"\b(?:about|around|roughly|approximately)\s+\d", re.I), re.compile(r"约|大约|左右|大概|差不多|近")),
    (re.compile(r"\b(?:nearly|almost)\b", re.I), re.compile(r"几乎|将近|接近|差不多|近")),
    (re.compile(r"\bup to\b", re.I), re.compile(r"最高|最多|多达|高达|上限")),
    (re.compile(r"\b(?:over|more than)\s+\d", re.I), re.compile(r"超过|以上|多于|逾|多")),
    (re.compile(r"\bsingle[- ]model\b", re.I), re.compile(r"单模型|单个模型|单一模型|单独")),
    (re.compile(r"\bensembles?\b", re.I), re.compile(r"集成|组合")),
    (re.compile(r"\b(?:may|might)\b", re.I), re.compile(r"可能|或许|也许")),
    (re.compile(r"\b(?:seems?|apparently)\b", re.I), re.compile(r"似乎|好像|看来|看起来")),
]
DOUBLED_RE = re.compile(r"(先先|的的|了了|是是|在在|和和|也也|都都)")


def translation_problems(english: str, translated: str, language: str, names: frozenset = frozenset()) -> list[str]:
    """What a translated line does to its English line that the reader would be misled by: a stronger claim, a lost
    hedge, a transliterated cast name, a doubled character, a prop under another name. Only zh-CN has rules."""
    if language != "zh-CN" or not english or not translated:
        return []
    out = []
    strong = ZH_STRONG_RE.findall(translated)
    if re.search(r"\bcertain(?:ly)?\b", english, re.I):
        strong = [word for word in strong if word != "一定"]
    if strong and not EN_STRONG_RE.search(english):
        out.append(f"stronger than the English ('{strong[0]}'): keep the English line's strength")
    for en_hedge, zh_hedge in HEDGES:
        m = en_hedge.search(english)
        if m and not zh_hedge.search(translated):
            out.append(f"drops the English hedge '{m.group(0).strip()}'")
    for name in names:
        if re.search(rf"\b{re.escape(name)}\b", english, re.I) and not re.search(rf"{re.escape(name)}", translated, re.I):
            out.append(f"the name {name.capitalize()} must stay in Latin letters")
    if "蛋挞" in translated:
        out.append("the tart is a berry tart: 浆果挞, never 蛋挞")
    if "×" in english and "×" not in translated and re.search(r"(?<![A-Za-z])[xX](?![A-Za-z])", translated):
        out.append("write × as ×, not X")
    doubled = DOUBLED_RE.search(translated)
    if doubled:
        out.append(f"a doubled character ('{doubled.group(0)}')")
    return out


def translation_fix_prompt(items: list[dict], language: str) -> str:
    return f"""Correct these {LANG_NAMES[language]} lines of a manga. Each comes with the English line it translates and what is wrong with the translation. Fix only that: keep the English line's meaning and strength, every number, name and technical term exactly, and about the same length; natural {LANG_NAMES[language]} with full-width punctuation; no underscores, LaTeX or markdown.
Output JSON only: {{"items": [{{"id": "...", "text": "corrected line"}}]}}

ITEMS:
{json.dumps(items, ensure_ascii=False)}"""


def backport_prompt(items: list[dict]) -> str:
    return f"""A fact-check of the Chinese translation of a manga script against the PAPER found the problems below. Each item gives the English line the Chinese line was translated from. The English line may have the same problem (an overgeneralisation, a dropped qualifier, a claim or an output the paper does not support), or the translation may have added it.
For each item: if the English line has the problem, return the corrected English line (same intent, about as short, the paper's numbers and qualifiers, no underscores or LaTeX; a line about the story's example says what the paper's result means for problems like it, never what the paper does not show); if the English line is fine as it is, return "KEEP".
Output JSON only: {{"items": [{{"id": "...", "text": "corrected English line or KEEP"}}]}}

ITEMS:
{json.dumps(items, ensure_ascii=False)}

PAPER TEXT:
"""


def _line_at(sb: StoryboardV2, pid: str) -> Optional[str]:
    """The text a fact-check id names ("5.d0" bubble, "5.n" narration, "5.b.<path>" board string), or None."""
    n_str, _, part = str(pid).partition(".")
    panel = next((p for p in sb.panels if str(p.n) == n_str), None)
    if panel is None:
        return None
    if part == "n":
        return panel.narration or None
    if part.startswith("d") and part[1:].isdigit():
        i = int(part[1:])
        return panel.dialogue[i].text if i < len(panel.dialogue) else None
    if part.startswith("b.") and panel.board:
        return dict(_board_strings(panel.board)).get(part[2:])
    return None


def _set_line_at(sb: StoryboardV2, pid: str, text: str) -> bool:
    n_str, _, part = str(pid).partition(".")
    panel = next((p for p in sb.panels if str(p.n) == n_str), None)
    if panel is None:
        return False
    if part == "n" and panel.narration:
        panel.narration = text
        return True
    if part.startswith("d") and part[1:].isdigit() and int(part[1:]) < len(panel.dialogue):
        panel.dialogue[int(part[1:])].text = text
        return True
    if part.startswith("b.") and panel.board:
        new = _set_board_strings(panel.board, {part[2:]: text})
        if new is not panel.board:
            panel.board = new
            return True
    return False


CHAPTER_ALIASES = {
    "intro": "hook", "introduction": "hook", "opening": "hook", "why": "hook", "motivation": "hook",
    "background": "problem", "challenge": "problem", "old problem": "problem",
    "key idea": "idea", "key_idea": "idea", "insight": "idea", "solution": "idea",
    "method": "how", "methods": "how", "mechanism": "how", "mechanisms": "how", "how it works": "how", "approach": "how",
    "results": "evidence", "result": "evidence", "experiments": "evidence", "proof": "evidence",
    "limitations": "limits", "limitation": "limits", "caveats": "limits", "critique": "limits",
    "conclusion": "takeaway", "conclusions": "takeaway", "summary": "takeaway", "ending": "takeaway", "outro": "takeaway",
}


def _normalize_chapter(chapter: str, previous: str) -> str:
    ids = [cid for cid, _ in CHAPTER_PLAN]
    c = (chapter or "").strip().lower()
    if c in ids:
        return c
    if c in CHAPTER_ALIASES:
        return CHAPTER_ALIASES[c]
    for cid in ids:
        if cid in c:
            return cid
    return previous


NARRATOR_RE = re.compile(r"\b(?:narrat(?:or|ion)|caption|voice[- ]?over|text box|title card)\b")


def _match_cast(name: str, valid) -> Optional[str]:
    """'Professor Papi', 'papi (smiling)' → 'papi'; the earliest whole-word name wins ('kumo (to papi)' → 'kumo'),
    independent of set order. None when no cast member is named."""
    n = (name or "").strip().lower()
    if n in valid:
        return n
    hits = [(m.start(), -len(member), member) for member in valid for m in [re.search(rf"\b{re.escape(member)}\b", n)] if m]
    return min(hits)[2] if hits else None


def _is_narrator(speaker: str, valid) -> bool:
    """'Narrator (V.O.)', 'The narrator', 'Caption box' → True; a speaker naming a cast member is that member's speech."""
    return _match_cast(speaker, valid) is None and bool(NARRATOR_RE.search((speaker or "").lower()))


# ----------------------------------------------------------------------------- pipeline steps

def _plain(text: str) -> str:
    """On-image text without markdown or TeX marks: Seedream letters them literally ("*only*")."""
    return re.sub(r"\*+", "", str(text or "").replace("_", " ").replace("\\", "")).strip()


def _normalize_panels(panels: list[PanelV2], cast: Cast, max_bubbles: Optional[int] = 2) -> None:
    """Number panels 1..N and clean them: narrator lines become narration (never dropped) before speech is capped at
    ``max_bubbles`` (None: no cap), names map onto the cast, emotions keep only characters in the panel, unknown chapters
    inherit the previous."""
    for i, p in enumerate(panels, start=1):
        p.n = i
    valid = set(cast.names())
    default_speaker = cast.names()[0]
    previous_chapter = CHAPTER_PLAN[0][0]
    for p in panels:
        speech = []
        for b in p.dialogue:
            if not _is_narrator(b.speaker, valid):
                speech.append(b)
            elif b.text not in p.narration:
                p.narration = f"{p.narration} {b.text}".strip()
        for b in speech:
            b.text = _plain(b.text)
        speech = [b for b in speech if b.text]   # a line of only markup ("**") would be an empty bubble
        p.dialogue = speech[:max_bubbles] if max_bubbles else speech
        p.narration, p.labels = _plain(p.narration), [_plain(lab) for lab in p.labels]
        p.characters = list(dict.fromkeys(m for c in p.characters for m in [_match_cast(c, valid)] if m))
        for b in p.dialogue:
            matched = _match_cast(b.speaker, valid)
            b.speaker = matched or (p.characters[0] if p.characters else default_speaker)
        if not p.characters:
            p.characters = [b.speaker for b in p.dialogue][:1] or [default_speaker]
        for b in p.dialogue:
            if b.speaker not in p.characters:
                p.characters.append(b.speaker)
        p.emotions = {m: v for k, v in p.emotions.items() for m in [_match_cast(k, p.characters)] if m}
        p.chapter = _normalize_chapter(p.chapter, previous_chapter)
        previous_chapter = p.chapter


def _chapters(panels: list[PanelV2]) -> list[Chapter]:
    return [Chapter(id=cid, title=ctitle, panel_numbers=nums) for cid, ctitle in CHAPTER_PLAN
            for nums in [[p.n for p in panels if p.chapter == cid]] if nums]


def _points(brief: dict) -> list[dict]:
    """The brief's key points with stripped ids and lower-case priorities, each id once."""
    out, seen = [], set()
    for pt in _as_list(brief.get("points")):
        pid = str(pt.get("id") or "").strip() if isinstance(pt, dict) else ""
        if pid and pid not in seen:
            seen.add(pid)
            out.append({**pt, "id": pid, "priority": str(pt.get("priority") or "").strip().lower()})
    return out


def uncovered(sb: "StoryboardV2", priority: str = "core") -> list[dict]:
    """The brief's points of this priority that no panel lists in its covers."""
    covered = {c for p in sb.panels for c in p.covers}
    return [pt for pt in _points(sb.brief) if pt["priority"] == priority and pt["id"] not in covered]


def coverage(sb: "StoryboardV2") -> dict:
    """{"core": [covered, total], "important": [covered, total]} for the reader and the job report. A v3 script reports
    the abstract claims its panels convey as its core points (and under "claims")."""
    if sb.version >= 3 and sb.abstract_claims:
        ids = {c.get("id") for c in sb.abstract_claims}
        done = len(ids & {c for p in sb.panels for c in p.claims})
        return {"core": [done, len(ids)], "important": [0, 0], "claims": [done, len(ids)]}
    out = {}
    for priority in ("core", "important"):
        total = [pt for pt in _points(sb.brief) if pt["priority"] == priority]
        out[priority] = [len(total) - len(uncovered(sb, priority)), len(total)]
    return out


UNCLEAR = {"unclear", "unknown", "n/a", "none", "not stated"}


def _card_text(value) -> str:
    """A brief value as reader text: strings as they are, lists of strings joined, anything else (or "unclear") dropped."""
    if isinstance(value, list):
        value = "; ".join(t for t in map(_card_text, value) if t)
    text = value.strip() if isinstance(value, str) else ""
    return "" if text.lower().strip(".") in UNCLEAR else text


def paper_card(brief: dict) -> dict:
    """The whole paper in a few lines (shown as text in the reader, translated and fact-checked with the script)."""
    def items(key: str, limit: int) -> list[str]:
        return [t for t in map(_card_text, _as_list(brief.get(key))) if t][:limit]

    how, results = [], []
    for m in _as_list(brief.get("mechanisms")):
        if isinstance(m, dict):
            name, plain = _card_text(m.get("name")), _card_text(m.get("plain"))
            how.append(f"{name}: {plain}" if name and plain else name or plain)
    for e in _as_list(brief.get("evidence")):
        if isinstance(e, dict):
            claim, numbers = _card_text(e.get("claim")), _card_text(e.get("numbers"))
            results.append(f"{claim} ({numbers})" if claim and numbers else claim)
    card = {"problem": _card_text(brief.get("problem")), "key_idea": _card_text(brief.get("key_idea")),
            "contributions": items("contributions", 6), "how": [h for h in how if h][:8], "results": [r for r in results if r][:8],
            "limitations": items("limitations", 5), "takeaway": _card_text(brief.get("takeaway"))}
    return {k: v for k, v in card.items() if v}


def _insert_position(panels: list[PanelV2], new: PanelV2, after: int) -> int:
    """Index in ``panels`` to insert ``new`` at: inside the run of its own chapter (clamped to it), or right after the
    panel it follows when its chapter is unknown (then it takes that panel's chapter). Never after the takeaway unless
    the new panel is part of it."""
    ids = [cid for cid, _ in CHAPTER_PLAN]
    if new.chapter not in ids:
        k = min(max(after, 0), len(panels))
        takeaway = [i for i, p in enumerate(panels) if p.chapter == "takeaway"]
        if takeaway and k > takeaway[0]:   # the manga keeps ending on its takeaway
            k = takeaway[0]
        new.chapter = panels[k - 1].chapter if k else panels[0].chapter
        return k
    run = [i for i, p in enumerate(panels) if p.chapter == new.chapter]
    if run:
        return min(max(after, run[0]), run[-1] + 1)
    earlier = [i for i, p in enumerate(panels) if p.chapter in ids and ids.index(p.chapter) < ids.index(new.chapter)]
    return earlier[-1] + 1 if earlier else 0


def cover_prompt(sb: "StoryboardV2", cast: Cast, missing: list[dict], max_panels: int) -> str:
    budget = BUDGETS["en-US"]
    script = [{"n": p.n, "chapter": p.chapter, "title": p.title, "covers": p.covers,
               "dialogue": [asdict(b) for b in p.dialogue], "narration": p.narration, "detail": p.detail} for p in sb.panels]
    return f"""The manga script below does not cover these CORE points of the paper:
{json.dumps(missing, ensure_ascii=False)}

Fix that without touching the existing panels:
- Write new panels that explain them (one point per panel, or two closely related points), each with "after": the number of the existing panel it follows, so the story still reads in order. The script may grow to at most {max_panels} panels in total ({max(0, max_panels - len(sb.panels))} more).
- If there is no room, or a point is a small detail of an existing panel, add a sentence to that panel's detail instead: {{"n": 7, "add_detail": "...", "covers": ["P9"]}}. If an existing panel already explains a point but does not list it, just tag it: {{"n": 7, "covers": ["P9"]}}.
- New panels follow the script's rules: cast {", ".join(cast.names())} only (lowercase names), max 2 speech bubbles of ≤ {budget['bubble']} words, optional narration ≤ {budget['narration']} words, ≤ 2 short on-image labels, a concrete drawable "visual" with no extra people and no written words, "covers" with the point ids, and a "detail" of 1-3 sentences. Every statement comes from the points; keep their exact numbers and qualifiers.
Output JSON only: {{"new_panels": [{{"after": 7, "chapter": "how", "title": "...", "characters": ["papi"], "scene": "...", "visual": "...", "dialogue": [{{"speaker": "papi", "text": "..."}}], "narration": "...", "on_image_labels": [], "emotions": {{"papi": "curious"}}, "covers": ["P9"], "detail": "..."}}], "details": []}}

SCRIPT:
{json.dumps(script, ensure_ascii=False)}"""


def _translated_card(card: dict, reply: dict, language: str, keep: frozenset) -> dict:
    """Take each translated card field that has the same shape and is in the target script; keep the English otherwise."""
    out = dict(card)
    for key, value in card.items():
        new = reply.get(key)
        if isinstance(value, str) and isinstance(new, str) and new.strip() and _has_script(new, language, keep):
            out[key] = new.strip()
        elif isinstance(value, list) and isinstance(new, list) and len(new) == len(value) \
                and all(isinstance(x, str) and x.strip() and _has_script(x, language, keep) for x in new):
            out[key] = [x.strip() for x in new]
    return out


# ----------------------------------------------------------------------------- v3: story layer
# The reader test: someone who never saw the paper reads only what is visible in the panels (bubbles, narration, boards)
# and must come away knowing everything the abstract says, having seen its key numbers as evidence.

# story length; an explicit longer range ("long") is kept. The writer drafts 14-17 panels whatever range it is asked for,
# and the books the blind judges found most focused (Oct 23-27) had 17-19; rounds 3-22 asked for 18-26 and split panels in
# the fix rounds, and the same number came back four or five times
V3_PANELS = (16, 24)
QUIZ_PASS = 0.85               # share of abstract questions a test reader must answer (abstract_quiz.PASS_SCORE wins)
QUIZ_EXTRA_PANELS = 4          # how far past its range the quiz revision may grow the script
# round 23: a scene tells the story literally and the boards explain, so the writer's "shows" is dropped and no picture is
# judged against an idea. True keeps it: the teaching pictures of rounds 3-22 (an older book still has them on a redraw)
SCENE_SHOWS = False
# chart panels one fix round may add, for results that no chart draws (_uncharted_evidence). A fix round no longer splits a
# panel in two: the second halves restated the first (round 23's dry runs: drafts of 17 panels grew to 24 by seven splits)
FIX_NEW_PANELS = 3
QUIZ_REVISIONS = 2             # revision rounds while claims still fail (Driver Misconduct: one round 18 → 55%, background left)
BEATS = ("hook", "claims", "problem", "idea", "how", "proof", "payoff", "limits", "recap")
BEAT_CHAPTER = {"hook": "hook", "claims": "hook", "problem": "problem", "idea": "idea", "how": "how", "proof": "evidence",
                "payoff": "payoff", "limits": "limits", "recap": "takeaway"}
# panels that list the abstract's claims instead of teaching them: the quiz reads around them, and a claim counts as
# conveyed (with its numbers) only on the other panels
LIST_BEATS = ("claims", "recap")
BEAT_ALIASES = {
    "stakes": "hook", "intro": "hook", "opening": "hook",
    "why_old_fails": "problem", "old_way": "problem", "old_way_fails": "problem", "background": "problem",
    "key_idea": "idea", "insight": "idea", "solution": "idea",
    "mechanism": "how", "method": "how", "how_it_works": "how", "demo": "how",
    "evidence": "proof", "result": "proof", "results": "proof", "challenge": "proof", "reaction": "proof", "perspective": "proof",
    "limitation": "limits", "limitations": "limits", "caveat": "limits", "caveats": "limits",
    "resolution": "payoff", "ending": "payoff", "conclusion": "payoff", "takeaway": "payoff",
    "summary": "recap", "review": "recap",
    "claim_board": "claims", "claims_board": "claims", "claim": "claims", "teaser": "claims", "promise": "claims", "bet": "claims",
    "predict": "proof", "prediction": "proof", "guess": "proof", "reveal": "proof",
}
# v3 text is typeset by code, so budgets follow comic practice rather than what the image model can letter;
# Chinese gets about 1.8 characters per English word
# "teach": a teaching scene (one with a "shows") leaves its picture room: the top band holds 30-odd words at reading size,
# and the scenes of 2026-10-28 with 38-46 words had their key object lettered over
BUDGETS_V3 = {
    "en-US": {"unit": "words", "scene": {"bubbles": 3, "bubble": 16, "narration": 20, "total": 44},
              "board": {"bubbles": 2, "bubble": 12, "narration": 14, "total": 28}, "teach": {"bubbles": 2, "total": 32}},
    "zh-CN": {"unit": "characters", "scene": {"bubbles": 3, "bubble": 29, "narration": 36, "total": 80},
              "board": {"bubbles": 2, "bubble": 22, "narration": 25, "total": 50}, "teach": {"bubbles": 2, "total": 58}},
}

SYSTEM_STORY = (
    "You are an award-winning manga writer and science communicator. You turn research papers into short, gripping manga "
    "stories that teach the real content, with the paper's real numbers, to smart readers who have ADHD. You never invent facts."
)


def _v3_budget(language: str) -> dict:
    return BUDGETS_V3.get(language) or BUDGETS_V3["en-US" if language == "en-US" else "zh-CN"]


def story_range(panel_range: tuple[int, int]) -> tuple[int, int]:
    """Panel range of a v3 story: 16-24 panels, inside the caller's range where they overlap; an explicitly longer
    range is kept as it is."""
    lo, hi = panel_range
    if lo > V3_PANELS[1]:
        return lo, max(lo, hi)
    lo = min(max(lo, V3_PANELS[0]), V3_PANELS[1])
    return lo, max(lo, min(hi, V3_PANELS[1]))


_WARNED: set[str] = set()


def _own_text(paper_text: str) -> str:
    """The paper text without the figures section (services.paper_figures): a vision model's reading, never a source
    for a number, a quote or a row."""
    figures = _optional("paper_figures")
    return figures.without_figures(paper_text) if figures is not None else str(paper_text or "")


def _optional(module: str):
    """services.boards / evidence / abstract_quiz, imported on use: this module (and the v2 pipeline) still loads without
    them, and a missing or broken one only disables its checks, with one warning per process."""
    try:
        return importlib.import_module(f"services.{module}")
    except Exception as e:  # ImportError, or a module that fails while loading
        if module not in _WARNED:
            _WARNED.add(module)
            print(f"[StoryboarderV2] WARNING: services.{module} unavailable ({type(e).__name__}: {e}); its v3 checks are skipped")
        return None


def _reraise_account(e: BaseException) -> None:
    """A best-effort step never hides an account problem (a refused key, a model not activated, no balance: engines
    mark those errors ``account``): the job must stop and say so, not quietly skip its checks."""
    if getattr(e, "account", False):
        raise e


def _ledger(paper_text: str):
    """The tables read from the PDF layout (services.ledger), or None when the paper text has none or the module is missing."""
    mod = _optional("ledger")
    found = mod.ledger_of(paper_text) if mod is not None else None
    return found if found else None


def _tables_section(paper_text: str) -> str:
    mod = _optional("ledger")
    return mod.section(paper_text) if mod is not None else ""


# ----------------------------------------------------------------------------- v3: abstract and its claims

ABSTRACT_LINE_RE = re.compile(r"^\s*(?:abstract|a b s t r a c t)\s*[.:—–-]?\s*$", re.I)
ABSTRACT_RUN_IN_RE = re.compile(r"^\s*(?:Abstract|ABSTRACT)\s*[.:—–-]\s*(\S.*)$|^\s*ABSTRACT\s+(\S.*)$")
FIRST_SECTION_RE = re.compile(r"^\s*(?:(?:\d{1,2}|[IVX]{1,5})\.?\s+)?(?:introduction|background|related work|motivation|preliminaries)\s*$", re.I)
NUMBERED_HEADING_RE = re.compile(r"^\s*(?:\d{1,2}|[IVX]{1,5})\.?\s+[A-Z][A-Za-z][A-Za-z ,:&\-]{0,60}$")
ABSTRACT_END_RE = re.compile(r"^\s*(?:[∗*†‡§¶]|(?:Figure|Fig\.|Table)\s*\d+[.:]|Keywords\b|Key words\b|Index Terms\b|CCS Concepts\b"
                             r"|Example sentence shown in a figure\b)", re.I)
VENUE_RE = re.compile(r"Conference on|Proceedings of|Workshop on|Symposium on|Under review|Preprint\b|Published as|Copyright\b|©")


def find_abstract(body: str, max_lines: int = 400) -> str:
    """The paper's abstract: from its "Abstract" heading to the first section heading ("1 Introduction"), cut at the first
    footnote (∗ † ‡), venue line ("31st Conference on ..."), caption, or heading-like line of a figure that interrupts it
    ("Standard Prompting"). "" when the paper has no abstract heading."""
    lines = body.splitlines()[:max_lines]
    start, first = None, ""
    for i, line in enumerate(lines):
        if ABSTRACT_LINE_RE.match(line):
            start = i + 1
            break
    if start is None:
        for i, line in enumerate(lines):
            m = ABSTRACT_RUN_IN_RE.match(line)
            if m:
                start, first = i + 1, (m.group(1) or m.group(2)).strip()
                break
    if start is None:
        return ""
    kept = [first] if first else []
    for line in lines[start:start + 80]:
        text = line.strip()
        if not text or re.fullmatch(r"\d{1,2}\.?|[IVX]{1,4}\.", text):   # a page number, or a section number on its own line
            continue
        if ABSTRACT_END_RE.match(text) or VENUE_RE.search(text) or FIRST_SECTION_RE.match(text):
            break
        if kept and NUMBERED_HEADING_RE.match(text) and re.search(r"[.!?]$", kept[-1]):
            break   # "2 Background" after a finished sentence (a wrapped "8 GPUs" never follows one)
        if kept and len(text.split()) <= 4 and text[:1].isupper() and not re.search(r"[.!?,;:)]$", text) \
                and re.search(r"[.!?]", " ".join(kept)):
            break   # a heading inside a figure placed in the abstract's column
        kept.append(text)
    return re.sub(r"\s+", " ", " ".join(kept)).strip()[:4000]


ABBREVIATION_RE = re.compile(r"\b(?:e\.g|i\.e|et al|etc|vs|cf|Fig|Figs|Eq|Eqs|Sec|Tab|approx|resp|No)\.$", re.I)


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9“\"'(\[])", str(text or "").strip())
    out: list[str] = []
    for part in parts:
        if out and ABBREVIATION_RE.search(out[-1]):
            out[-1] = f"{out[-1]} {part}"
        elif part.strip():
            out.append(part.strip())
    return out


DASHES_RE = re.compile(r"[\u2010-\u2015\u2212]")
STOPWORDS = frozenset("a an the of and or to in on for with by from at as is are was were be been that this these those it its "
                      "we our their which while than into over can also both such".split())


def _norm_text(text: str) -> str:
    s = DASHES_RE.sub("-", unicodedata.normalize("NFKC", str(text or "")))
    s = re.sub(r"-\s*\n\s*", "", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def _squash(text: str) -> str:
    """Letters, digits and CJK only: survives spacing, punctuation and the extractor's lost hyphens ("Englishto-German")."""
    return re.sub(r"[^0-9a-z\u3400-\u9fff]+", "", _norm_text(text))


def quote_in(quote: str, source: str) -> bool:
    """True when the quote occurs in the source, ignoring case, spacing and punctuation; "..." may join fragments that
    occur in that order. Fragments shorter than 10 letters prove nothing."""
    frags = [_squash(f) for f in re.split(r"\s*(?:\.\.\.|…)\s*", str(quote or ""))]
    frags = [f for f in frags if f]
    if not frags or any(len(f) < 10 for f in frags):
        return False
    src, pos = _squash(source), 0
    for f in frags:
        j = src.find(f, pos)
        if j < 0:
            return False
        pos = j + len(f)
    return True


def _content_tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z]+|\d+(?:\.\d+)?", _norm_text(text)) if t[0].isdigit() or (len(t) > 2 and t not in STOPWORDS)}


def _best_sentence(texts: list[str], sentences: list[str], threshold: float = 0.5) -> Optional[str]:
    """The sentence sharing the largest share of words with any of ``texts`` (a near-verbatim quote, the plain claim)."""
    scored = [(len(want & _content_tokens(s)) / len(want), s) for want in map(_content_tokens, texts) if want for s in sentences]
    score, best = max(scored, default=(0.0, None))
    return best if score >= threshold else None


def _sentence_covered(sentence: str, quotes: list[str]) -> bool:
    """At least 60% of the sentence's words, and all of its numbers, are in claim quotes taken from it."""
    words = _content_tokens(sentence)
    if not words:
        return True
    got: set[str] = set()
    for q in quotes:
        qt = _content_tokens(q)
        if qt and len(qt & words) >= 0.5 * len(qt):
            got |= qt & words
    return len(got) >= 0.6 * len(words) and {w for w in words if w[0].isdigit()} <= got


def verify_claims(raw, abstract: str, body: str) -> list[dict]:
    """The brief's abstract claims, checked deterministically: each quote must occur in the abstract (in the paper when
    there is no abstract). A claim whose quote is not found is re-anchored on the abstract sentence its words match best
    (at least half of them) or dropped; an abstract sentence that no claim covers becomes a claim of its own, so together the claims
    cover the whole abstract. Ordered as in the abstract, ids 0..N-1."""
    source = abstract or body
    sentences = split_sentences(abstract) if abstract else []
    claims, seen = [], set()
    for item in _as_list(raw):
        if isinstance(item, dict):
            text, quote = str(item.get("text") or item.get("claim") or "").strip(), str(item.get("quote") or "").strip()
        elif isinstance(item, str):
            text = quote = item.strip()
        else:
            continue
        if not quote_in(quote, source):
            anchor = _best_sentence([quote, text], sentences)
            if anchor is None:
                print(f"[StoryboarderV2] dropped an abstract claim whose quote is not in the abstract: {quote[:80]!r}")
                continue
            quote = anchor
        if _squash(quote) in seen:
            continue
        seen.add(_squash(quote))
        claims.append({"text": _plain(text or quote), "quote": quote})
    for s in sentences:
        # the abstract says it, so the reader must get it (a fragment of under four words is extraction debris)
        if len(s.split()) >= 4 and not _sentence_covered(s, [c["quote"] for c in claims]):
            claims.append({"text": s, "quote": s})
    src = _squash(source)

    def position(claim: dict) -> int:
        first = next((f for f in (_squash(x) for x in re.split(r"\.\.\.|…", claim["quote"])) if f), "")
        j = src.find(first) if first else -1
        return j if j >= 0 else len(src)

    claims.sort(key=position)
    return [{"id": i, **c} for i, c in enumerate(claims)]


def paper_examples(raw, body: str, limit: int = 5) -> list[dict]:
    """Example inputs the paper itself shows (a figure's sentence, a sample question), kept only when found in the paper."""
    out: list[dict] = []
    for item in _as_list(raw):
        text = str((item.get("text") or item.get("example") or "") if isinstance(item, dict) else (item or "")).strip()
        text = re.sub(r"^Example sentence shown in a figure:\s*", "", text, flags=re.I)
        source = str(item.get("source") or "").strip() if isinstance(item, dict) else ""
        if text and len(text) <= 400 and quote_in(text, body) and all(_squash(text) != _squash(e["text"]) for e in out):
            out.append({"text": text, "source": source})
    return out[:limit]


EXAMPLE_PARTS = 8


def brief_example(raw) -> dict:
    """The running example the analysis picks: {"text", "from_paper", "source", "parts", "old_way", "new_way"}, or {}.
    Round 23's judges all asked for one concrete case drawn at every step, but no step chose it: the analysis wrote each
    mechanism "on the running example" without naming one, the writer then picked its own in one sentence ("a long
    competitive math problem", "we follow sales agents"), and the diagram designer, given only that sentence, drew
    placeholders, or other details of the case on each diagram (Flexibility Trap's fork was "Since / Therefore" on one
    and "the missing bracket" on the next). One case with its parts in their real words now runs from the analysis to
    every diagram; the brief check reads it against the paper when it claims to come from there."""
    if not isinstance(raw, dict):
        return {}
    text = _text(raw.get("text") or raw.get("example"))
    if not text:
        return {}
    from_paper = str(raw.get("from_paper", True)).strip().lower() not in ("false", "0", "no", "none", "")
    parts: list[str] = []
    roles: list[str] = []
    for item in _as_list(raw.get("parts")):
        x = _text(item.get("text") or item.get("name")) if isinstance(item, dict) else _text(item)
        if x and all(" ".join(x.split()).casefold() != " ".join(y.split()).casefold() for y in parts):   # "+1" is not "-1"
            parts.append(x)
            roles.append(_text(item.get("what") or item.get("role")) if isinstance(item, dict) else "")
    return {"text": text, "from_paper": from_paper, "source": _text(raw.get("source")) if from_paper else "",
            "parts": parts[:EXAMPLE_PARTS], "roles": roles[:EXAMPLE_PARTS], "old_way": _text(raw.get("old_way")),
            "new_way": _text(raw.get("new_way"))}


ROLE_SEP = " — "   # "a — a candidate the teacher favours": a part and what it is in the case, as checkers and prompts see it


def _part_pairs(example: dict) -> list[tuple[str, str]]:
    """(part, what it is) for each part of a running example (an older brief has no roles: "")."""
    parts = [x for x in _as_list(example.get("parts")) if isinstance(x, str)]
    roles = [x if isinstance(x, str) else "" for x in _as_list(example.get("roles"))]
    return [(t, roles[i] if i < len(roles) else "") for i, t in enumerate(parts)]


def _part_line(text: str, role: str) -> str:
    return f"{text}{ROLE_SEP}{role}" if role else text


# ----------------------------------------------------------------------------- v3: prompts

def brief_prompt_v3(paper: PaperText, abstract: str) -> str:
    caps = "\n".join(paper.captions[:20])
    abstract_block = abstract or ("(no abstract heading was found: take the claims from the paper's own summary of its problem, "
                                  "contributions and results in the introduction and conclusion, quoting those sentences)")
    return f"""Read the paper and its ABSTRACT and write a compact, faithful BRIEF for a manga adaptation. A newcomer must understand the task, problem, method and why its steps help; the story must also preserve the abstract's facts, numbers and conditions.

Plan for a novice: enough foundations for the concrete goal and problem, then mechanism steps and why each helps before terminology and data. State the goal as an everyday task with a concrete input and desired output, in words a newcomer understands; distinguish a research aim from a measured achievement. The opening lists 2-3 plain reader questions: no unexplained abbreviations, result numbers, or asserted results. Questions must not presuppose perfect reference answers, inevitable failures, complete recovery, or speedups unsupported by the paper. Foundations explain only necessary concepts, without inventing benefits or guarantees. Each step's why must give a paper-supported reason, not a stronger causal claim. Detailed claims and exact evidence are taught later across successive panels.

Rules: every statement must come from the paper; copy exact numbers, dataset names, baselines and units; if something is unclear say "unclear" instead of guessing.

Output JSON only:
{{
 "title": "paper title",
 "field": "one phrase",
 "beginner_plan": {{"goal": "concrete task", "foundations": ["necessary concepts in dependency order"], "opening_questions": ["2-3 plain reader questions, no result numbers"], "steps": [{{"what": "one mechanism step", "why": "paper-supported reason it helps"}}]}},
 "abstract_claims": [{{"id": 0, "text": "the claim in plain words, one line", "quote": "the exact words of the abstract it comes from"}}],
 "problem": "2-3 sentences: what was hard/slow/wrong before, in plain language",
 "key_idea": "ONE sentence a smart teenager could repeat",
 "contributions": ["each contribution the authors claim, one sentence each"],
 "running_example": {{"text": "the one instance the manga follows, in one sentence with its real content", "from_paper": true, "source": "Figure 3", "parts": [{{"text": "one thing a drawing of it shows, in its real words", "what": "what it is in this case"}}], "old_way": "what the existing approach does to this instance", "new_way": "what the paper's method does to it instead"}},
 "mechanisms": [{{"name": "...", "plain": "2 sentences, no jargon", "on_example": "what this mechanism does to the running example above, step by step, with its parts", "exact": "the precise technical statement with symbols spelled in words"}}],
 "setup": "1-3 sentences: the data, baselines, metrics and model sizes of the experiments",
 "evidence": [{{"claim": "...", "numbers": "exact figures with units, dataset, split, baseline, single model or ensemble", "source": "where the paper reports it (Table 2, Section 6.1)", "higher_is_better": true, "why_it_matters": "..."}}],
 "limitations": ["only limitations the authors state"],
 "takeaway": "1-2 sentences: the paper's own conclusion",
 "examples": [{{"text": "an example input exactly as the paper shows it", "source": "Figure 3"}}],
 "glossary": [{{"term": "...", "plain": "what it is in plain words; for a metric, what it measures and whether higher or lower is better"}}],
 "fun_facts": ["optional surprising details worth a joke"]
}}
"abstract_claims": split the ABSTRACT into atomic claims (usually 6-15) that together cover EVERY sentence of it, in order: the problem, what is new, how it works, every result with its number and setting, and the conclusion. A claim comes from one sentence at most (a sentence with two facts may give two claims, each quoting its own clause; two sentences never make one claim). "text" states the claim in plain words a smart teenager understands, keeping its numbers and qualifiers exactly (single model or ensemble, which dataset, "up to", "over") and every property, relation and name its quote gives ("dispensing with recurrence and convolutions entirely" stays "no recurrence and no convolutions at all", not "a simpler design"); "quote" copies the words of the abstract it comes from character for character (a sentence or a clause of at least five words). Ids are 0, 1, 2, ... in abstract order.
"running_example": ONE INSTANCE the manga follows from its first panel to its last, never a category, a phenomenon or an average: one input with its real content (the words of one question and its answer, the tokens of one prefix and the next tokens a model weighs there, one line of a solution, one parcel on one route, one agent's calls in one day, one robot in one maze). The problem is felt in it, the key idea and every mechanism are shown on it, and the results speak about cases like it. First look at what the paper's figures show (the section FIGURES READ FROM THE IMAGES near the end of the PAPER TEXT describes each one) and at its example inputs, samples in the appendix and case studies: an overview figure often walks one case through the method (a prompt with the model's own continuation, a maze with its start and goal). The best case is the one at which the paper shows its key step acting (where the problem happens and what the method changes there), not merely an output that went wrong. Copy its content exactly as the figure or the text gives it, set "from_paper": true and give its "source". Only when the paper shows no instance (it reports averages or a pattern over many inputs), make up a small instance that mirrors the paper's task exactly, with "from_paper": false and no numbers of its own beyond small counts (the paper's own average may stand beside it as the paper's average). "parts": the 3-8 things a drawing of the instance shows: "text" is one thing's real content in at most 12 words, the actual words, tokens, values and outputs as the instance has them (a token such as "Wait", a line such as "180 - 0.30 × 180 = 54", an answer such as a person's name), never a description of them ("the math problem prompt", "underpredicted candidate tokens", "the attention distribution", "token 1"); "what" says in at most 12 words what that thing is in the case (the student's own next token after "So", a candidate the teacher favours and the student underweights, the wall between start and goal), so that a drawing never strings alternatives into a sentence. A value the figures section gives only as an order is described in words (highest, near zero), never as a number. "old_way" and "new_way": what the existing approach does to this instance and what the paper's method does to it instead, in the paper's terms (with the paper's number when it gives one for such cases), and only what the paper shows or states: never an outcome for this case that the paper does not report.
"mechanisms": every component a reader needs to understand how the method works (usually 3-8), each walked through the running example.
"evidence": every result the abstract mentions, then the key ablations and analyses: go through every results table and figure, including the appendix, and keep each number with its exact setting and where the paper reports it.
"limitations": only what the authors themselves say; never add your own.
"examples": example inputs the paper itself shows that a story could follow from start to end: a sentence from a figure, a sample question, an input and its output (lines starting with "Example sentence shown in a figure:" are such examples). Copy them exactly; [] when the paper shows none.
"glossary": every technical term, metric and dataset that the abstract or the results use, and every name the abstract gives to a problem, effect or mechanism it describes ("this trajectory divergence grows as responses lengthen" names "trajectory divergence"): a reader should meet the paper's own name for it, explained.

ABSTRACT:
{abstract_block}

FIGURE/TABLE CAPTIONS (for orientation):
{caps}

PAPER TEXT:
{paper.body}"""


def _roles(cast: Cast) -> dict[str, str]:
    """Cast names by role (mentor, student, skeptic); missing roles fall back to the members in order."""
    names = cast.names() or ["papi", "kumo", "nezu"]
    by_role: dict[str, str] = {}
    for m in cast.members:
        by_role.setdefault(m.role, m.name)
    return {role: by_role.get(role) or names[min(i, len(names) - 1)] for i, role in enumerate(("mentor", "student", "skeptic"))}


def _v3_rules(cast: Cast) -> str:
    """Panel kinds, board shapes, budgets and data rules shared by the story, fix and quiz-revision prompts."""
    en = BUDGETS_V3["en-US"]
    scene, board = en["scene"], en["board"]
    return f"""PANEL KINDS
- "scene": the cast acting in the running example or the paper's world, drawn literally (the people of the paper played by the cast, the real objects and places), never stand-in symbols for concepts. An illustrator who cannot write draws the picture, so "visual" describes only what is drawn (setting, poses, expressions, objects), never words, numbers, charts or writing; screens, books and boards in a scene are blank. Props never carry writing: no clipboards, notebooks, pens writing, newspapers, posters, signs, plaques, stone tablets, scrolls, calendars, clocks, word cards or screens with text. No object stands in for a concept (no cards, yarn, blocks, jars, cups or balloons that mean something): draw the real things of the situation, and the cast's gestures and faces. No props shaped like a person or creature (robots, dolls, statues, toy animals): they are drawn as extra characters. All words go in "dialogue" and "narration". Only the characters listed in "characters" appear: cast {", ".join(cast.names())} (lowercase names), never other people, crowds, creatures or copies of a cast member.
- "board": a large blank whiteboard in the scene carries an exhibit that a program draws exactly from "board", so it may hold exact text and numbers. "visual" describes only the setting and the cast's poses and reactions next to the board: the setting is the running example's world (the parcel depot, the call centre, the lab), never an empty classroom, and a cast member holds or uses a real object of that world when it fits (a headset, a scooter helmet), so the board panels do not all look alike. Use a board for every result (with the paper's numbers), for every mechanism, step and structure (a diagram), and for the recap. Board shapes (JSON):
  diagram: {{"kind": "diagram", "title": "...", "subtitle": "...", "brief": "what it draws: the running example's real things and the arrangement that shows this panel's point, in one or two sentences (which objects, where, what differs between the two sides, what each arrow does)"}}  (a program designs and draws it from the brief, the panel's words and the paper, every label exact; to change what an existing diagram draws, rewrite its brief, as its "labels" only show what is drawn now: the tokens of the example in the order they are filled, the question and answer and the documents built around them, a pipeline with the example flowing through it, layers with a bypass, calls at two times of the day, a teacher's and a student's choices, the old way beside the new way on the same input. Its labels name the parts in the words' own terms, so the bubbles never spell out what the diagram labels)
  bar: {{"kind": "bar", "title": "...", "subtitle": "what the metric measures; higher is better", "unit": "BLEU", "higher_is_better": true, "bars": [{{"label": "short display name", "value": "@T2.r7.c1", "baseline": true}}, {{"label": "...", "value": "@T2.r9.c1", "highlight": true}}], "source": "Table 2"}}  (3-6 bars from ONE column of one table, so of the same metric, dataset, split and training setup, or, for a trend across positions, sizes or time, the cells of ONE row of one table, one bar per bin: the paper's headline result is the ONE bar with "highlight": true, the strongest earlier system the paper compares with is marked "baseline": true, and other systems from the same column (other variants of the paper's model included) give context; when the claim is that a gap closes (recovering what quantization or a smaller model lost, approaching a teacher, full precision or humans), the reference the gap is measured to (full precision, the teacher, humans) is a bar too when the same column holds it, so the reader sees how much of it closes; every label names the system as the paper does ("OpenAI GPT", "ConvS2S Ensemble"), never "prior best" or "SOTA"; the program draws the gap between the highlighted bar and the baseline, so mark the baseline only when the paper makes that comparison. A "value" cites a cell of the TABLES section (see DATA RULES); a number that no table holds is written out, with "quote": a verbatim snippet of the paper that contains it, and "paper_label": the row or condition name exactly as in the paper)
  table: {{"kind": "table", "title": "...", "subtitle": "...", "columns": ["...", "..."], "rows": [{{"cells": ["Transformer (big)", "@T2.r9.c1"]}}], "highlight_rows": [0], "source": "Table 3"}}  (at most 4 columns and 6 rows; a cell may cite a table cell; a row with numbers that no table holds needs a "quote": a verbatim table row or sentence of the paper containing every number of the row)
  flow: {{"kind": "flow", "title": "...", "subtitle": "...", "steps": ["...", "..."], "highlight": [1], "note": "..."}}  (2-7 steps: a pipeline or an architecture)
  compare: {{"kind": "compare", "title": "...", "subtitle": "...", "left": {{"label": "old way", "items": ["..."]}}, "right": {{"label": "new way", "items": ["..."]}}}}  (at most 4 items a side; left = old, right = new)
  links: {{"kind": "links", "title": "...", "subtitle": "...", "tokens": ["The", "animal", "..."], "from": 1, "weights": [0.1, 0.6], "note": "..."}}  (only for a paper about attention between words: at most 14 tokens of the running example and one weight per token: how much the token at index "from" attends to each; the board is always tagged as an illustration, so never call its weights exact, measured or reproduced from a figure. A theory paper explains its argument with flow and compare boards)
  list: {{"kind": "list", "title": "...", "subtitle": "...", "items": ["..."], "open": false}}  (2-6 lines; "open": true draws empty check boxes: the claims board before the story has checked it; on the recap a line gets a tick only when a chart or table of the story showed its numbers, the program decides, and the other lines get a plain bullet)
  Board text is short: titles at most 10 words, bar labels 5, table columns 4 and cells 6, flow steps 8, compare items 10, list lines 14, link tokens one word each.

DIAGRAMS EXPLAIN, SCENES TELL (a reader looks at the picture before reading a word)
- Explain the problem, the key idea and every mechanism on diagram boards that draw the paper's real things on the running example (its actual tokens, question, documents, layers, calls, parcels), so the reader SEES the structure: what goes where, in what order, what the old way does to the example and what the new way does instead. A diagram may carry several labelled parts of one step; a long mechanism takes two or three diagram panels, each one step further on the same example.
- Scenes carry the story: the student's goal and the problem felt in the running example, the questions, guesses and surprises, the skeptic's doubts, the payoff and the limits. A scene shows its situation literally (a driver at a door holding a parcel; two calls at the start and the end of a working day); a person of the paper is played by a cast member, said once in the narration ("Nezu plays the driver").
- Never make a character say what a drawn object stands for ("This balloon is uncertainty", "this red cap means the driver"): a diagram labels its parts and a scene shows real things. No props that stand in for a concept.
- "shows" is "" on every panel: no picture is asked to explain without words; the boards do.

FOCUS (each panel earns its place)
- Every panel answers the question the previous panel raised, or raises the one the next panel answers. Say each point once: a number, a definition or a claim appears on one panel of the story (the recap repeats the claims once at the end); the next panel goes one step further instead of restating.

EVIDENCE (a reader believes what a chart shows)
- Every result in the BRIEF's "evidence" is shown on a chart (bar or table) with the paper's numbers on the panel that states it, never only said. The central claim gets the chart that could have proven it wrong (the comparison that tests it, e.g. the solutions only one order finds), not only the method's headline score.
- Before a key chart, the student and the skeptic each guess the outcome in words (a direction, never a number), so the chart settles something the reader cares about.

TEXT BUDGETS (a program checks them)
- scene panel: at most {scene['bubbles']} speech bubbles of at most {scene['bubble']} words each; narration at most {scene['narration']} words.
- scene panel: at most {scene['total']} words of bubbles and narration in total; board panel: at most {board['total']}. Big type keeps the manga readable on a phone: say one thing per panel, in as few words as it needs.
- board panel: 1-2 cast members reacting to the board, 1-{board['bubbles']} speech bubbles of at most {board['bubble']} words each; narration at most {board['narration']} words.
- The narration is read before the bubbles, so it never holds the answer to a question a bubble asks (the reader would meet the answer first): a later bubble answers it, or the next panel opens with the answer.
- No underscores, LaTeX, markdown or emoji in any text; write math in words or plain symbols (√, ², ×, ≈).

DATA RULES (a program checks every number and rejects the panel otherwise)
- When the paper's tables could be read, the PAPER TEXT ends with a section "TABLES READ FROM THE PDF LAYOUT": each table has a "columns" line (c1 = BLEU › EN-DE, ...) and row lines ("[T2.r9]   Transformer (big)   c1: 28.4   c2: 41.8"). A number from that section is CITED, never typed: write @<table>.r<row>.c<column> ("@T2.r9.c1") as a bar's "value", as a table cell, or inside a bubble or the narration ("BERT reaches @T1.r5.c2 on MultiNLI"). The program writes the exact number, its row and its table.
- Every other number on a board, in a bubble or in the narration is copied character for character from the paper ("28.4", "2.3 · 10^19", "3.5 days"). Never compute, round, convert or estimate a number: no differences, ratios, percentages or "x times faster" unless the paper states that exact figure.
- A bar value or table row that does not cite a cell comes with a "quote": a verbatim snippet of the PAPER TEXT that contains it.
- A label says exactly which row or setting a value belongs to: model and size, single model or ensemble, dataset and split. Never call a row "prior best", "previous best" or "state of the art" unless the paper says so for that comparison; name the model instead.
- One board compares one metric on one dataset, split and training setup: the bars of a chart come from ONE column of one table (or, for a trend over bins such as positions, sizes or time, from ONE row of one table); never mix metrics, units, datasets or setups (e.g. a semi-supervised result next to supervised-only ones) in one chart: keep one of them on this board, or make it a table with the setup as a column. Compare single models with single models, unless the paper's own claim includes ensembles (then say "ensembles included" in the subtitle). The program checks that the strongest earlier system of the column is on the chart.

FIDELITY
- Every fact, number, name and setting comes from the paper; keep the paper's qualifiers (single model, on this dataset, up to, over, about).
- Never overgeneralize: keep each claim exactly as broad as the paper's (if the paper says recurrent models cannot be parallelized within a training example, do not say it of every older model).
- No claims about later impact, adoption or today's practice. Limitations only as the authors state them. Never say a model beats humans unless the paper does.
- No verdict words: never "prove", "proven", "perfect(ly)", "case closed" or "always works"; the paper showed results on its tests ("held up on these tests", "on WMT 2014", "in this example"). Keep the paper's own hedges ("apparently", "seems", "about", "over").
- General knowledge the paper does not state (e.g. that recurrent networks "forget" early words) is not presented as the paper's claim; when the story needs it, a character says it is background, in one short line.

CLARITY (readers of earlier manga stumbled on each of these)
- Every technical term, abbreviation, method, metric and dataset (GRPO, softmax, U-Net, IV regression, Pass@k) gets a plain explanation the first time a reader sees it, in the same panel: what it is or does, in a few everyday words.
- The first panel never opens with "this", "these", "that" or "such": the reader has seen nothing yet, so name the thing.
- Finish one point before starting the next: when a chapter numbers its points (first, second), it never goes back to an earlier one.
- Every key number says its unit and what it is measured against: "1.60% fewer successful deliveries the next day", "10.26 points above BitDistiller on the 9-benchmark average".
- A board panel's words talk about its own board: a number in its bubbles or narration is a number drawn on that board, and its narration never introduces a different topic."""


def number_lines(claims: list[dict], paper_text: str, per_number: int = 3, width: int = 170) -> str:
    """For each key number of the abstract claims, the table cells that hold it (services.ledger: their ids, rows and
    columns, to cite on a board) and the paper lines that contain it (verbatim, trimmed around the number): quotes the
    writer can copy character for character, and the table rows a comparison comes from."""
    text = _own_text(paper_text)
    ledger = _ledger(text)
    tables_at = text.find("\n" + _optional("ledger").LEDGER_HEAD) if ledger else -1
    lines = [ln.strip() for ln in (text[:tables_at] if tables_at >= 0 else text).splitlines() if ln.strip()]
    out, seen = [], set()
    for claim in claims:
        for written, key in _key_numbers(claim):
            if key in seen or len(re.sub(r"\D", "", written)) < 2:
                continue
            seen.add(key)
            pattern = re.compile(rf"(?<![\d.]){re.escape(written)}(?![\d])")
            cells = [c.describe() for c in ledger.cells_with(key)[:per_number]] if ledger else []
            hits = []
            for ln in lines:
                m = pattern.search(ln)
                if m:
                    start = max(0, m.start() - width // 2)
                    hits.append(("…" if start else "") + ln[start:start + width] + ("…" if start + width < len(ln) else ""))
                if len(hits) >= per_number - (1 if cells else 0):
                    break
            if cells or hits:
                out.append(f"{written}: " + " | ".join(cells + [f'"{h}"' for h in hits]))
    return "\n".join(out)


def _examples_block(brief: dict) -> str:
    examples = [e for e in _as_list(brief.get("examples")) if isinstance(e, dict) and e.get("text")]
    chosen = brief.get("running_example") if isinstance(brief.get("running_example"), dict) else {}
    if chosen.get("text"):   # picked when the paper was read (brief_example): the story and every diagram follow it
        lines = [f"THE RUNNING EXAMPLE: {chosen['text']}" + (f" ({chosen['source']})" if chosen.get("source") else "")
                 + ("" if chosen.get("from_paper") else " (made up, an illustration of the paper's task: a character says once that it is one)")]
        if chosen.get("parts"):
            lines.append("Its parts, in their real words, each with what it is in the case (the diagrams draw these): "
                         + json.dumps([_part_line(t, r) for t, r in _part_pairs(chosen)], ensure_ascii=False))
        if chosen.get("old_way"):
            lines.append(f"What the old way does to it: {chosen['old_way']}")
        if chosen.get("new_way"):
            lines.append(f"What the paper's method does to it: {chosen['new_way']}")
        words = _content_tokens(" ".join([chosen["text"]] + [x for x in _as_list(chosen.get("parts")) if isinstance(x, str)]))

        def same(e: dict) -> bool:   # the paper's own wording of this case (the analysis describes it in one sentence)
            mine = _content_tokens(e["text"])
            return _squash(e["text"]) in _squash(chosen["text"]) or (
                bool(chosen.get("from_paper")) and len(mine & words) >= 0.6 * max(1, len(mine)))
        own = [e["text"] for e in examples if same(e) and _squash(e["text"]) not in _squash(chosen["text"])]
        if own:   # the verified wording, so the hook can quote it exactly
            lines.insert(1, "In the paper's own words: " + " / ".join(own))
        others = [e for e in examples if not same(e)]
        if others:
            lines.append("Other example inputs the paper shows (only to show that a result holds beyond the running example):\n"
                         + "\n".join(f"- {e['text']}" + (f" ({e['source']})" if e.get("source") else "") for e in others))
        return "\n".join(lines)
    if not examples:
        return "The paper shows no example input that a story could follow, so use a small everyday example (see the rules)."
    return "Example inputs the paper itself shows (verified in the paper):\n" + "\n".join(
        f"- {e['text']}" + (f" ({e['source']})" if e.get("source") else "") for e in examples)


NOVICE_SEQUENCE = """NOVICE READER CONTRACT (novice_v1)
Plan enough beginner foundations: concrete goal -> 2-3 plain reader questions -> necessary
background -> concrete problem -> mechanism steps and why EACH helps -> terms and exact data
when needed -> limits -> answers to the opening questions. Use existing problem/idea/how beats
for foundations; no new beat is required. Introduce at most one new concept per panel, explain
its purpose before naming it. The opening claims list is QUESTIONS, never a dense claims board.
For each mechanism, make the participants and their roles visible before using shorthand names.
Show what input each step receives, where its target or feedback comes from, what changes and
why that change helps. Carry the same supported example across these steps; a method name or
an arrow alone is not an explanation. Put this reasoning in visible words or actual board labels,
never only in visual directions, hidden notes or an unwritten diagram request.
Keep every factual detail, qualifier, comparison and exact number across the whole story.
One claim may span successive teaching panels; keep its id on each. Never restuff a complete
claim into one panel or into the opening questions to satisfy coverage, a quiz or a budget.
The recap answers the questions in plain words, not lexical copies. Preserve question order.
Use brief visible illustrative-example labels and source-boundary statements wherever needed.
An illustration is not a measured outcome. Never invent results, example trajectories or causality;
show only supported steps and distinguish aggregate evidence from a single example.
"""


def novice_prompt(prompt: str) -> str:
    """Replace legacy packing instructions, retaining all shared data and drawing rules.

    Applied only at novice call sites; old boards keep their original prompts verbatim.
    """
    replacements = {
        'THE READER TEST (': 'THE READER TEST: from visible panels alone, a newcomer can explain the concrete task, obstacle, method steps, why they help and the limits. Also retain every abstract fact, number, setting and qualifier across the teaching sequence, with key numbers shown as evidence. Factual recall is a separate constraint, not proof of human comprehension.',
        '2. claims (exactly': '2. claims (exactly 1 board panel after the hook): an open list of 2-3 PLAIN READER QUESTIONS about the goal, problem and how it could be helped. No result numbers, unexplained terms or full claims. A concrete question is enough; do not force an extra bet.',
        '3. problem (': '3. problem (as many panels as the prerequisites need): first teach the necessary beginner foundations and concrete goal, then demonstrate the problem on the example. Explain one new idea per panel before its technical name.',
        '9. recap (': '9. recap (last board): answer each opening question in order, using what the teaching panels established and keeping the supported scope. Do not repeat the questions as answers or list full claims.',
        '- A panel that teaches a claim says all of it:': '- A claim can be taught across successive panels, one new concept per panel. Together they retain every part, qualifier and exact number; each carries its claim id.',
        '- Every panel answers the question': '- Each panel advances one step and explains why it helps. A claim can span successive panels; avoid redundant repetition, but keep necessary foundations and all qualifiers.',
        '- The recap lists the claims board': '- The recap answers the opening reader questions in order in natural language, preserving scope; lexical overlap is not required.',
        '- The claims board and the recap only list': '- The opening board asks questions and the recap answers them. The teaching panels must still teach every abstract claim and carry its id.',
        '- "recap" problems:': '- "recap" problems: answer each opening question using facts already taught, in the same order; never copy the question or restuff full claims.',
        '- Budget problems:': '- Budget problems: shorten repetition while keeping the current step and its why. Retain details across successive teaching panels, not all within one panel.',
        '- A claim no panel conveys,': '- A missing claim or number: teach it at the appropriate point after its prerequisites, preserving every qualifier and exact number across successive panels with its claim id.',
        '- "reader" problems:': '- "reader" problems: a reader who never saw the paper stumbled here; fix the missing prerequisite, overload or causal link at its teaching point, before the dependent term/data. Explain purpose and why, one new idea per panel, with only PAPER TEXT facts. Keep all claim details across successive panels; never solve overload by adding more definitions to this panel.',
        '- "term" problems:': '- "term" problems: establish the necessary plain concept before its name, in this panel or an earlier suitable teaching panel. Keep the term, all supported details and budgets without overloading a panel.',
        '- Use ONE running example from': '- Use one running example for supported steps only. Where it cannot demonstrate a step, state the source boundary briefly and teach that step from the paper without inventing a trajectory.',
        '- When THE RUNNING EXAMPLE is given': '- When THE RUNNING EXAMPLE is given, keep its real parts and source and never switch to another one. Demonstrate only steps the source actually supports; do not turn candidate tokens into a generated sequence.',
        '- First choice: make the fact visible': '- First choice: revise the existing successive panels about this claim, each teaching one step with its why; retain all facts across them and keep prerequisites ahead of terms/data.',
        '- Otherwise add a new panel': '- Otherwise add a new panel with "after" (at most one per claim), at the teaching point after its prerequisites. Keep exact results on an evidence board; do not overload the new panel.',
    }
    lines = []
    in_paper = False
    for line in prompt.splitlines():
        if line == "PAPER TEXT:":
            in_paper = True
        if in_paper:
            lines.append(line)
            continue
        # These instructions otherwise forbid truthful source boundaries in examples/payoffs.
        if line.startswith('- New panels:'):
            line = line.split('Add no other panel:', 1)[0] + ' Preserve details across successive teaching panels after prerequisites; never pack a whole claim into one panel. If no space remains, leave overload unresolved for review.'
        elif line.startswith('- When the running example comes from the paper,'):
            line = '- For the paper example show only supported steps. Briefly state where its evidence ends; then distinguish aggregate chart results. Never invent a per-example outcome.'
        elif line.startswith('7. payoff ('):
            line = '7. payoff (1-2): return to the example and show only what the paper supports. A brief source-boundary statement is allowed; aggregate outcomes do not prove this example was solved.'
        else:
            for prefix, replacement in replacements.items():
                if line.startswith(prefix):
                    line = replacement
                    break
        lines.append(line)
    # Keep the first line stable for engine routers; instructions precede any source payload.
    # Quiz routers parse the JSON immediately following the first line. Keep that boundary.
    at = next((i for i, line in enumerate(lines) if line.startswith("Revise the script so")), 1)
    lines.insert(at, NOVICE_SEQUENCE)
    return "\n".join(lines)


def story_prompt(brief: dict, claims: list[dict], cast: Cast, panel_range: tuple[int, int], paper: PaperText) -> str:
    lo, hi = panel_range
    r = _roles(cast)
    mentor, student, skeptic = r["mentor"], r["student"], r["skeptic"]
    facts = {k: v for k, v in brief.items() if k not in ("abstract_claims", "abstract", "examples", "brief_check", "factcheck",
                                                         "running_example")}
    prompt = f"""Turn this research paper into a SHORT MANGA STORY (in English; it is translated later).

THE READER TEST (this is how the manga is judged): a reader who never saw the paper reads ONLY what is visible in the panels: speech bubbles, narration boxes and the boards drawn in the scenes. Afterwards they must be able to explain everything the paper's ABSTRACT says (the problem, what is new, how it works at the level of the abstract, every result with its number and setting, and the conclusion), and they must have SEEN the key numbers as evidence. Anything that is not visible in a panel does not exist for the reader.

ABSTRACT CLAIMS (every claim is conveyed by the visible text or a board of at least one panel, which lists the claim's id in "claims"):
{json.dumps(claims, ensure_ascii=False)}

CAST
{cast.describe()}
{student} is the reader's stand-in: curious, gets stuck, voices the natural misconception and asks what the reader would ask. {mentor} knows the paper and explains everything on the running example. {skeptic} is a sharp reviewer: doubts every claim, demands proof, names the limitations the authors state, and is won over only by data.

RUNNING EXAMPLE
{_examples_block(brief)}
- Use ONE running example from the first panel to the last: the problem is felt in it, the idea is shown on it, every mechanism is demonstrated on it, and the payoff shows what the method changes for it (solved, only when the paper shows it solved).
- When THE RUNNING EXAMPLE is given above, use exactly that case and never switch to another one. The reader sees it first (the hook quotes or shows its parts), and every problem, idea and how diagram draws its parts in their real words (the question, the tokens, the documents, the outputs), one step further on each panel; a diagram's brief names the parts it draws, and when a figure of the paper shows that step, which figure.
- When the running example comes from the paper, say and draw only what the paper shows happening to it (its figures and its text). Where the paper reports no outcome for this one case, never invent one (no after-picture the paper does not draw, no "now it gets it right"), and never spend a line or a panel saying that the paper does not show it: go from the case to the charts, which show the results over many cases.
- Prefer an example the paper itself shows: copy it exactly and set "from_paper": true. Only when none fits, use a small everyday situation that mirrors the paper's task, set "from_paper": false, and let a character say once that it is an illustration; its made-up details are never presented as results, and it brings no numbers of its own beyond small counts.

STORY BEATS ("beat" of each panel, in this order; the panels of one beat form a chapter)
1. hook (1-2 panels): {student} has a concrete goal in the running example and runs into the problem the paper addresses; end on a question the reader wants answered.
2. claims (exactly 1 board panel, right after the hook): a "list" board with "open": true, titled like "What the paper claims", with the abstract's headline claims in 3-5 short lines, each with its key number and the claim's own specifics (its qualities, names and resources), never a vague stand-in: readers take this board, and the recap that repeats it, as the summary ("RL needs long, diverse, challenging, verifiable data", not "RL needs good data"; "synthesis needs outside corpora or stronger teachers", not "needs external help"). {skeptic} doubts the central claim and bets something small from her own world (for example her berry tart) on what she expects instead, said as a prediction a chart can settle ("Any order explores more paths, so it must solve more problems."); {student} takes the bet: together they will look at the evidence. The bet is on the paper's claims as the board lists them, never on the method, plan or product the paper studies: when the paper finds that something backfires, she bets that the paper is wrong about it. The board's subtitle says what the lines are (e.g. "The paper's own summary"), never how the board is drawn.
3. problem (1-3): why the existing approach fails, shown concretely on the example: a scene of {student} running into it, and a diagram board that draws what the old way does to the example; say exactly what the paper says is wrong with it, and no more. The abstract's background claims (what the field does, what it costs, what goes wrong) are each said in full on these panels or the hook's.
4. idea (1-2): the key idea in one memorable sentence, said by {student} after a nudge from {mentor}, on a diagram board of the old way beside the new way on the same example.
5. how (one or two panels per mechanism the abstract relies on): each a diagram board that walks the running example through the mechanism, one step further on each panel (a flow board when it is a pipeline of named steps; links only for attention between words).
6. proof (for EVERY result in the abstract): {skeptic} challenges ("Compared to what?"). For the one or two most important results, first a scene where {student} and {skeptic} each guess in words (no numbers) how it will turn out. Then a board panel answers with the paper's exact numbers against the strongest earlier system and the field (see the bar shape), and a reaction line says what the gap means in the paper's own terms and its scope: which test, single model or ensemble, which training data. One comparison per board; the most telling ablation may get its own board if there is room.
7. payoff (1-2): back to the running example, and SHOW on a board what the idea changes for it: the worked steps or the words that give the answer away when the paper shows them, or the paper's own figure as a links board, hedged the way the paper hedges it; never claim the method produced an output for the example (a correct translation, a right answer) unless the paper shows that output. When the paper does not show this example solved, end on what its result means for problems like it ("89.1% of these grade-school problems now come out right"), never on a line about what the paper does not show. {student} sees what the method changes.
8. limits (1-2): the limits or open questions the authors themselves state, each with the reason the paper gives, voiced by {skeptic}; if the paper states none, what it leaves open, in its own terms. Then the bet is settled in one line: {skeptic} pays up because the results the story showed held on these tests (name one), and keeps the limits as her fair point; never "every claim was verified".
9. recap (the last panel, a board panel): the claims board again ("open": false) with each line's key number, titled like "What the paper showed". It lists what the paper reports; it never says that every claim was verified or checked.
Each chapter opens with a question in a bubble or the narration (e.g. "Why was the old way so slow?"), and its last panel makes the reader want the next one.
The first panel where each cast member appears names them with their role in a bubble or the narration ("Kumo, a PhD student", "Professor Papi", "Nezu, the critic"), so the reader knows who is who.

TEACHING RULES
- Every panel teaches one concrete fact; jokes and emotions ride on top of facts, never instead of them. No filler panels and no panel that repeats an earlier one.
- The claims board and the recap only list the claims: every claim must ALSO be taught on a story panel (problem, idea, how, proof, limits or payoff), which lists the claim's id in "claims". A reader who skips the two lists must still learn the whole abstract.
- A panel that teaches a claim says all of it: every item of a list ("customer experience, delivery success and productivity"), every qualifier ("only short-context tasks", "on two models") and the consequence the abstract names ("parcels returned to merchants"). Two claims share a panel only when both fit whole; never drop a claim's details to make room.
- Explain every technical term, metric and dataset at its first appearance, in a bubble, the narration or the board's subtitle: what it is in plain words and, for a metric, what it measures and which direction is better (e.g. "BLEU: how closely a machine translation matches human translations; higher is better"). This includes words that appear on boards: "ensemble" (several models combined) the first time an ensemble appears, model variants (base vs big: a smaller and a larger version), tasks (what "constituency parsing" means), scores (F1, accuracy) and benchmark names. Board labels are plain words ("Transformer, 4 layers"), never codes like "4l". Use the glossary.
- Every idea and how panel says WHY, not only what: one plain reason that ties the step to the problem it fixes, said the way people talk ("The answer is fixed before the long text exists, so the reward can always be checked."; never "Because X, so Y").
- Never make a character say what a drawn object stands for: a diagram labels its parts, and a scene shows the real things.
- The recap lists the claims board's lines again, in the same order, now each with its key number, so the reader sees every promise of the claims board come back.
- Every comparative claim of the abstract (better, faster, cheaper, a fraction of the cost, beats humans) is shown with the paper's numbers on a board when the paper reports them (the rival's number included, e.g. the human error rate), and a claim made for several settings (e.g. both large and limited training data) shows each setting.
- The narration is read before the bubbles: it never refers to "the above".
- Narration states facts in plain words; bubbles carry the story and the explanations. The narration never repeats a bubble.
- {lo}-{hi} panels in total; choose the count by how much the abstract contains. Number panels 1..N in reading order.

{_v3_rules(cast)}

OUTPUT: JSON only:
{{"title": "short manga title", "summary": "one sentence", "running_example": {{"text": "one sentence: the situation the story follows", "from_paper": true, "source": "where the paper shows it (e.g. Figure 3), or empty"}},
 "panels": [{{"n": 1, "kind": "scene", "beat": "hook", "title": "short panel title", "characters": ["{student}"], "scene": "the setting", "visual": "what is drawn", "shows": "", "dialogue": [{{"speaker": "{student}", "text": "..."}}], "narration": "...", "emotions": {{"{student}": "frustrated"}}, "board": null, "claims": [0]}}]}}

BRIEF (checked against the paper):
{json.dumps(facts, ensure_ascii=False)}

WHERE THE ABSTRACT'S NUMBERS APPEAR IN THE PAPER (cite the @cells; copy quotes from these lines exactly; table rows name the systems and their rivals):
{number_lines(claims, paper.body) or "(no numbers)"}

PAPER TEXT:
{paper.body}"""

    return novice_prompt(prompt) if brief.get("reader_contract") == CONTRACT else prompt


def _prompt_board(board):
    """A board as the writer sees it: a diagram as its title, brief and what its drawing shows in words (services.diagram
    describe: the boxes with their texts, the arrows between them; its coordinates are the designer's, and a rewrite
    that keeps the brief keeps the drawing); any other board as it is."""
    if isinstance(board, dict) and str(board.get("kind") or "").lower() == "diagram" and board.get("elements") is not None:
        out = {k: board[k] for k in ("kind", "title", "subtitle", "brief", "note", "source") if board.get(k)}
        labels = _diagram_lines(board, described=True)
        if labels:
            out["labels"] = labels
        return out
    return board


def _diagram_lines(board, described: bool = False) -> list[str]:
    """A diagram's labels in reading order, or (``described``) what its drawing shows in words (services.diagram)."""
    diagram = _optional("diagram")
    boards = _optional("boards")
    if diagram is None or boards is None:
        return []
    try:
        norm = boards.normalize_board(board)
        if not (isinstance(norm, dict) and norm.get("kind") == "diagram"):
            return []
        return diagram.describe(norm) if described else diagram.lines(norm)
    except Exception:
        return []


def _panel_json(p: PanelV2) -> dict:
    return {"n": p.n, "kind": p.kind, "beat": p.beat, "title": p.title, "characters": list(p.characters), "scene": p.scene,
            "visual": p.visual, "shows": p.shows, "dialogue": [asdict(b) for b in p.dialogue], "narration": p.narration,
            "emotions": dict(p.emotions), "board": _prompt_board(p.board), "claims": list(p.claims)}


def _kept_drawing(old, new):
    """A rewrite returns a diagram as the writer sees it (title, brief, labels): when its brief is unchanged (or left
    out) it keeps the designed elements, and a new brief leaves it to be designed again (StoryboarderV2.design_diagrams).
    Lines the rewrite adds to its labels are asked of the designer through the brief (a quiz revision added the missing
    step there, the old drawing came back and the claim stayed untaught: round 23c's review)."""
    if not (isinstance(old, dict) and isinstance(new, dict)):
        return new
    if str(old.get("kind") or "").lower() != "diagram" or str(new.get("kind") or "").lower() != "diagram":
        return new
    if new.get("elements") or not old.get("elements"):
        return new
    brief = str(new.get("brief") or "").strip()
    # a line counts as added when it names something the drawing does not (a reformatted line of it is no change)
    shown = set().union(*(_content_tokens(x) for x in _diagram_lines(old, described=True) + _diagram_lines(old)))
    added = [str(x).strip() for x in _as_list(new.get("labels")) if isinstance(x, str) and _content_tokens(x) - shown]
    if added and (not brief or brief == str(old.get("brief") or "").strip()):
        brief = (str(old.get("brief") or "").strip().rstrip(".") + ". It also shows: " + "; ".join(added[:4])).strip(". ")
        return {**{k: v for k, v in new.items() if k not in ("labels", "elements", "designed_for")}, "brief": brief}
    if brief and brief != str(old.get("brief") or "").strip():
        return {k: v for k, v in new.items() if k != "labels"}
    kept = {k: v for k, v in new.items() if k != "labels"}
    kept.update({"elements": old["elements"], "brief": old.get("brief", "")})
    for k in ("designed_for",):
        if k in old:
            kept[k] = old[k]
    return kept


def _panel_outline(p: PanelV2) -> dict:
    """A panel as context for a rewrite: its text, and only the kind and title of its board."""
    board = {"kind": p.board.get("kind"), "title": p.board.get("title")} if isinstance(p.board, dict) else None
    return {"n": p.n, "kind": p.kind, "beat": p.beat, "characters": list(p.characters), "dialogue": [asdict(b) for b in p.dialogue],
            "narration": p.narration, "board": board, "claims": list(p.claims)}


# problems a panel's art directions alone can fix: "picture: ..." and "the picture ..." (props, charts, words, repeats)
PICTURE_PROBLEM_RE = re.compile(r"^(?:picture:|the picture )")


def picture_fix_prompt(sb: StoryboardV2, cast: Cast, problems: dict[int, list[str]]) -> str:
    """Picture problems alone: only the panels' art directions are rewritten and their words stay (a full rewrite for a
    picture problem changed the words too: round 5's books had more pictures that explain and readers who learned less)."""
    flagged = [{"n": p.n, "kind": p.kind, "beat": p.beat, "characters": list(p.characters), "scene": p.scene, "visual": p.visual,
                "shows": p.shows, "words": [f"{b.speaker}: {b.text}" for b in p.dialogue] + ([f"narration: {p.narration}"] if p.narration else []),
                "problems": problems[p.n]} for p in sb.panels if p.n in problems]
    # the story around them: round 6's picture fixes saw only the flagged panels, and teaching pictures that explain
    # nothing went from 37% (full rewrites) back to 42-44%
    outline = [{"n": p.n, "beat": p.beat, "kind": p.kind, "characters": list(p.characters), "visual": p.visual,
                "words": " ".join([f"{b.speaker}: {b.text}" for b in p.dialogue] + ([p.narration] if p.narration else []))[:220]}
               for p in sb.panels]
    casting = sorted({sentence.strip() for q in sb.panels for text in [q.narration] + [b.text for b in q.dialogue]
                      for sentence in SENTENCE_SPLIT_RE.split(text or "") if re.search(r"\bplay(?:s|ing)?\b", sentence, re.I)})[:6]
    who = ("\n\nWHO PLAYS WHOM (as the story says):\n" + "\n".join(f"- {x}" for x in casting)) if casting else ""
    return f"""Fix the pictures of these manga panels. A checking program found the problems listed with each panel. Rewrite only each panel's art directions: "scene" (the setting) and "visual" (what is drawn: the setting, the cast's poses, gestures and faces, and the real things of the situation), and set "shows" to "". The words of the panels stay exactly as they are; the boards of the story explain the paper, and a scene tells the story.
- Only the panel's "characters" appear; a person of the paper is played by one of the panel's characters (the role stays with the cast member who plays it in the other panels).
- Draw the situation literally: the real people, places and things of the story (a driver at a door holding a parcel; two calls at the start and at the end of a working day; a student at a laptop), never an object that stands in for a concept (no cards, yarn, blocks, jars, cups or balloons that mean something).
- Nothing to count beyond five.
- Never words, numbers, charts, boards, screens with text, or props shaped like people or creatures in the picture.{who}

Output JSON only: {{"panels": [{{"n": 3, "scene": "...", "visual": "...", "shows": ""}}]}}

THE WHOLE STORY (context only: keep each place's and thing's look and each role's cast member as the other panels draw them):
{json.dumps(outline, ensure_ascii=False)}

PANELS TO FIX:
{json.dumps(flagged, ensure_ascii=False)}"""


def _stages(name: str, text: str) -> bool:
    """Does a picture text put this cast member in the picture? "Nezu's cap lies alone" and "without Nezu" stage nobody."""
    return bool(re.search(rf"(?<!without )(?<!no )\b{re.escape(name)}\b(?!['’]s\b)", text, re.I))


PICTURE_CHECK_KINDS = ("still", "contrast", "reference", "abstract", "contradicts", "unreadable", "repeat", "drift", "mismatch")
PICTURE_CHECK_MAX = 12   # panels per script: one picture-fix call rewrites them all


def _picture_checked(p: PanelV2) -> bool:
    """A scene whose picture must show an idea: one with a "shows" in a teaching beat or the hook. Since round 23 a scene
    tells the story literally and has no "shows" (the boards explain), so new books have none to check."""
    return p.kind == "scene" and bool(p.shows.strip()) and (p.beat == "hook" or p.beat in PICTURE_BEATS)


def picture_check_prompt(sb: StoryboardV2, only: Optional[set] = None, again: bool = False) -> str:
    """A viewer's read of each teaching scene's planned picture beside the words it must carry. Of round 15's teaching
    pictures that passed the art check (the art showed what "shows" asked for), the Claude picture audits judged 26 of 70
    "no" and 43 "partly": the plan itself was a still life of symbols, a prop whose everyday meaning says the opposite
    (a mountain of paper for "data is scarce"), cards whose content no one can read, or an earlier panel's picture again.
    Of the 61 teaching pictures rounds 20-21's audits judged "no", 46 failed in the plan and none in the drawing alone; the
    most common reasons were not on this list: a difference other than the one the words compare (23), a gap or an
    error with no level drawn to measure it against (7), a step no object visibly performs (9)."""
    symbols = "\n".join(f"- {c}: {sym}" if c else f"- {sym}" for c, sym in _picture_symbols(sb))
    story = [{"n": p.n, "beat": p.beat, "kind": p.kind, "characters": list(p.characters), "visual": p.visual} for p in sb.panels]
    panels = [{"n": p.n, "beat": p.beat, "characters": list(p.characters),
               "words": [f"{b.speaker}: {b.text}" for b in p.dialogue] + ([f"narration: {p.narration}"] if p.narration else []),
               "visual": p.visual, "shows": p.shows}
              for p in sb.panels if _picture_checked(p) and (only is None or p.n in only)]
    focus = (" These pictures were just redrawn on paper from an earlier finding: report one only when it still cannot carry its "
             "point." if again else " These panels are new or rewritten; the others were checked before." if only is not None else "")
    return f"""Check the pictures planned for a manga that teaches a research paper. An image model draws each panel from its "visual" alone, without any words; the panel's words are lettered on top afterwards. A reader looks at the picture first, so a teaching picture works when a viewer who sees only the drawing gets the panel's point, or at least its shape (which thing is bigger, what goes where, what changes, what causes what), and the words only have to name the things.
For each TEACHING PANEL, read its words (what it teaches) and its picture plan ("visual", and "shows": the arrangement the picture must make clear). Report a panel only when its picture cannot carry its point, for one of these reasons:
- "still": nothing happens or differs in the picture: the cast hold, tap, point at or look at the objects, or the symbols stand side by side, with no action, before and after, cause and effect, or contrast between them that matches the point.
- "contrast": the picture shows a difference or a change, but not the one the words are about: its two sides (or before and after) must differ in exactly what the panel compares and in nothing else (when the old and the new recipe bake in the same oven and differ in the flour, a small oven beside a big oven teaches the wrong difference; two jars on different shelves say "different shelves" when the point is two jars on the same shelf filled differently); or a with-versus-without, old-versus-new or before-versus-after point is drawn with only one side.
- "reference": the point is an error, a gap, too much or too little, a gain or growth, but nothing drawn shows the level it is measured against, so a big pile and a small pile only say "few and many" (a line on each jar at the level it should reach, the one starting point both paths leave, a dashed line at the old height).
- "abstract": the point is a step no object visibly performs (a model learns, weighs, judges, misjudges, decides), and the picture shows the symbols sitting there; it needs that step's visible result, before and after (the cup filled to the line, the pile that was sorted).
- "contradicts": what the objects mean in everyday life says something else or the opposite (a mountain of paper says plenty when the point is that data is scarce; a sieve letting things through is the sieve working, not failing).
- "unreadable": the point depends on what is written, printed or shown on or inside something (cards that "show" under, correct and over predictions; "the right answer" on a sheet), on exact counts or ratios beyond a few objects ("8 of 10", "1.6%"), or on a meaning only the words give, with no visible link (an empty crate standing for lost work).
- "repeat": the same objects in the same arrangement as an earlier panel (see THE WHOLE STORY), although this panel teaches something new.
- "drift": a symbol means something else than it did in earlier panels or stands for two things in one picture, or a role is played by another cast member than before.
- "mismatch": "visual" and "shows" describe different pictures (the image model draws "visual").
A hook that only introduces the cast or the question may stay as it is. Report at most {PICTURE_CHECK_MAX} panels and one problem per panel, worst first; an empty list when every picture works.{focus}
For each: "n", "kind", "problem" = one sentence on what the picture shows instead of the point, "fix" = a picture that shows the point without words, in at most 40 words, built from the panel's characters and the book's picture symbols that the words of this panel or an earlier one have named: an action and its result, before and after side by side, a tiny thing beside a huge one, objects sorted, poured, splitting, dropped or broken. Nothing to count beyond five (a share is a tiny part beside a huge whole, or one empty slot in a long row), and never words, numbers, charts, screens or props shaped like people or creatures.
Output JSON only: {{"problems": [{{"n": 3, "kind": "still", "problem": "...", "fix": "..."}}]}}

THE BOOK'S PICTURE SYMBOLS (one object per concept, the same in every panel):
{symbols or "(none)"}

THE WHOLE STORY (every panel's picture, in reading order):
{json.dumps(story, ensure_ascii=False)}

TEACHING PANELS:
{json.dumps(panels, ensure_ascii=False)}"""


def fix_prompt(sb: StoryboardV2, cast: Cast, problems: dict[int, list[str]], paper_text: str = "", tables: str = "",
               room: int = 0) -> str:
    """``paper_text`` when a problem needs the paper (evidence, claims, fact-check findings); otherwise ``tables``, the
    ledger section alone, so a rewrite can still cite table cells. ``room``: how many chart panels the reply may add for
    the problems that ask for one (none otherwise: a panel too full is shortened, never split in two)."""
    flagged = [{"panel": _panel_json(p), "problems": problems[p.n]} for p in sb.panels if p.n in problems]
    lines = number_lines(sb.abstract_claims, paper_text) if paper_text else ""
    paper = ((f"\n\nWHERE THE ABSTRACT'S NUMBERS APPEAR IN THE PAPER (cite the @cells; copy quotes from these lines exactly):\n{lines}" if lines else "")
             + f"\n\nPAPER TEXT:\n{paper_text}") if paper_text else (f"\n\n{tables}" if tables else "")
    shorten = ("Add no other panel: where a panel cannot hold everything its problems ask for within its text budget, shorten "
               "it: cut the words that repeat or decorate, move a definition into its board's subtitle, and keep every number "
               "and every part of each claim.")
    split = (f"""- New panels: where a problem asks for a chart panel right after a panel, return that panel (same "n") and ONE new board panel {{"after": n, ...}} with the chart, right after it (beat "proof" for a result; it lists in "claims" the ids it conveys); at most {room} in all. {shorten}"""
             if room > 0 else f"- New panels: none. {shorten}")
    split_out = ' (a new chart panel: {"after": 3, "kind": "board", ...} with the same fields)' if room > 0 else ""
    prompt = f"""Fix the flagged panels of this manga script. A checking program found the problems listed with each panel.
Return a replacement for EVERY flagged panel, with the same "n", that solves all of its problems while keeping its beat, its role in the story and the claims it conveys. Change nothing that is not needed.
- "evidence" problems: a number is not supported by the paper, or a chart's bars do not come from one table column. Cite the table cell that holds it ("value": "@T2.r9.c1"), or copy the value character for character from the paper and give that bar or table row a "quote" (a verbatim snippet of the PAPER TEXT containing its numbers); if nothing in the paper supports a number, remove it (drop that bar or row, or make the point without the number). Never compute, round or convert.
- "board" problems: make the board valid for its kind (see the shapes below), or turn the panel into a scene panel ("kind": "scene", "board": null).
- Budget problems: shorten the text or move a definition into the board's subtitle; never drop a number, a claim or a part of a claim.
{split}
- A claim no panel conveys, or a claim's missing number: state it, with its key number, in this panel's visible text or board, and add its id to "claims".
- "reader" problems: a reader who never saw the paper stumbled here; the Fix is that reader's hint, and every fact, unit and baseline you write comes from the PAPER TEXT. Explain a term in a few plain words in this panel (a bubble, the narration or the board's subtitle), name what "this" points at, and give a number its unit and what it is measured against. For an order problem, drop the numbering ("first", "second") rather than move points around. For a mismatch, rewrite the off-topic sentence about this panel's board, or drop it and remove its claim id from "claims" (the program then finds that claim another panel). Keep the text budgets.
- "board check" problems: the board must show exactly what this panel's words claim. Follow the Fix with the paper's numbers (mark the right bar as the baseline, add the missing rival or setting with its table cell, state the unit on the board and use the same one in the words, explain a label in the subtitle); when the board cannot show the claim, narrow the words to what it shows, but a part of this panel's abstract claims stays in the words, said as what the paper reports, apart from the chart ("The paper also reports shorter answers."). A bar chart draws one gap badge, against the one bar marked as the baseline: for "beats both baselines" mark the stronger one and say the gap to it; it draws a ratio of two scores only when a bar's quote states the paper's own multiple ("52×"), and never a second badge, arrows or curves. Never convert or compute a number.
- "why" problems: add one plain reason from the PAPER TEXT that ties this step to the problem it fixes, said the way people talk ("X, so Y" or "Y, because X"; never "Because X, so Y"); keep the text budgets.
- "recap" problems: rewrite the recap's lines as the claims board's lines in the same order, each with its key number.
- "picture" problems: rewrite "visual" (and "scene" when the problem names it) so the illustrator can draw it: the real people, places and things of the story, drawn literally; keep the words of the panel.
- "term" problems: the reader meets a term here that no panel has explained. Add a few plain words right where it appears ("IV regression, a way to separate cause from coincidence", "FKL (forward KL)"), in a bubble, the narration or the board's subtitle, taking the meaning from the problem or the PAPER TEXT; keep the term itself and the text budgets (shorten other words if needed).

{_v3_rules(cast)}

Output JSON only: {{"panels": [{{"n": 3, "kind": "scene", "beat": "...", "title": "...", "characters": ["..."], "scene": "...", "visual": "...", "shows": "", "dialogue": [{{"speaker": "...", "text": "..."}}], "narration": "...", "emotions": {{}}, "board": null, "claims": [0]}}]}}{split_out}

ABSTRACT CLAIMS:
{json.dumps(sb.abstract_claims, ensure_ascii=False)}

SCRIPT (context only; return only the flagged panels):
{json.dumps([_panel_outline(p) for p in sb.panels], ensure_ascii=False)}

FLAGGED PANELS:
{json.dumps(flagged, ensure_ascii=False)}{paper}"""

    return novice_prompt(prompt) if is_novice(sb) else prompt


CLARITY_KINDS = ("term", "pointer", "mismatch", "order", "number", "prerequisite", "overload", "causal_gap", "example")
CLARITY_MAX = 12   # problems per script: a fix round rewrites every flagged panel


def clarity_prompt(sb: StoryboardV2, only: Optional[set] = None) -> str:
    """A newcomer's read of the whole English script in reading order: what a reader who never saw the paper stumbles on
    (the Claude readers' recurring notes: unexplained terms, "this" pointing at nothing, words beside a board about
    something else, points visited out of order, numbers without a unit or a comparison)."""
    if is_novice(sb):
        focus = f" Panels {', '.join(map(str, sorted(only)))} are new or rewritten; report findings only on these panels." if only is not None else ""
        return ("You are a curious reader who never saw this research paper and knows little about its field. "
                "Read ONLY the visible script in order; do not infer missing facts from prior knowledge. "
                "Check term, pointer, mismatch, order, number, prerequisite (missing foundations), overload "
                "(too many new concepts), causal_gap (a step without why it helps), example (illustration or "
                "aggregate data mistaken for a measured case). Check that the recap answers the opening questions. "
                "For each finding cite exact visible words in problem, give n, kind and a concrete fix naming "
                "what is missing, without inventing facts. At most 12 problems, at most 2 per panel. "
                'Output JSON only: {"problems": [{"n": 3, "kind": "prerequisite", "problem": "...", "fix": "..."}]}. '
                + focus + "\n\nSCRIPT:\n" + reader_transcript(sb))
    panels = [{"n": p.n, "chapter": p.chapter, "kind": p.kind, "narration": p.narration,
               "bubbles": [f"{b.speaker}: {b.text}" for b in p.dialogue], "board": _drawn_board(p.board)} for p in sb.panels]
    glossary = [{"term": _text(g.get("term")), "plain": _text(g.get("plain"))} for g in _as_list(sb.brief.get("glossary"))
                if isinstance(g, dict) and g.get("term")]
    focus = (f"\nPanels {', '.join(map(str, sorted(only)))} are new or rewritten; the others were read before. Report problems on these "
             "panels only, but read the whole script: a term an earlier panel explains is explained." if only is not None else "")
    return f"""You are a curious reader who never saw this research paper and knows little about its field. Read the manga script below in order, the way a reader does: on each panel the narration first, then the speech bubbles, then the board drawn in the scene ("board" is all the text on it: "- " lines are bars or rows with their values, "[gap to X: +2.04]" is the difference the chart draws between the highlighted bar and bar X, so a gap the words state is on the board when the badge shows it).
List the places where you would stumble. Only these kinds:
- "term": a technical term, abbreviation, method, metric or dataset (e.g. GRPO, softmax, U-Net, IV regression, Pass@k) that the reader meets without a plain explanation in that panel or an earlier one.
- "pointer": a line that says "this", "these", "that", "such" or "it" about something the reader has not been shown yet (most often on the first panel of the story or of a chapter).
- "mismatch": a board panel whose words are about something its board does not show: a number not drawn on the board, or narration on another topic (e.g. "the authors found two challenges" beside a results chart).
- "order": the story goes back and forth between points (first, second, then first again) or returns to a point an earlier panel finished.
- "number": a key number whose unit or point of comparison is missing, so the reader cannot say what it measures ("down 1.60%": of what, compared with what?).
Report only what would really confuse a reader; style, jokes and simplifications are fine. At most {CLARITY_MAX} problems and at most 2 per panel, worst first; an empty list when nothing confuses.{focus}
For each: "n" = the panel to change (where the term first appears, where the pointer or the mismatch is, where the jump back happens), "kind", "problem" = one sentence quoting the exact words, "fix" = one sentence naming what is missing ("explain what GRPO is", "say what the 1.60% is measured against"); for a term you may give its plain meaning from the glossary, but never supply a fact, number or baseline of your own: you have not read the paper.
Output JSON only: {{"problems": [{{"n": 3, "kind": "term", "problem": "...", "fix": "..."}}]}}

GLOSSARY (what the paper's terms mean):
{json.dumps(glossary, ensure_ascii=False)}

SCRIPT:
{json.dumps(panels, ensure_ascii=False)}"""


BOARD_CHECK_KINDS = ("comparison", "unit", "label", "consistency", "form")
BOARD_CHECK_MAX = 20   # problems per script; at most 2 per board


def _drawn_board(board) -> str:
    """A board as the reader sees it: bars with their values, the gap badge, footnotes (services.abstract_quiz)."""
    quiz = _optional("abstract_quiz")
    if not isinstance(board, dict) or not board:
        return ""
    try:
        return " / ".join(quiz._board_lines(board, "en-US")) if quiz is not None else board_text(board)
    except Exception:  # a board the drawer cannot read: its raw text
        return board_text(board)


def _checked_board(p: PanelV2) -> bool:
    """A board the board check judges: every drawn board but the claims board (it only lists what the story will show)."""
    return p.kind == "board" and isinstance(p.board, dict) and bool(p.board) and p.beat != "claims"


def board_check_prompt(sb: StoryboardV2, paper_text: str, only: Optional[set] = None, again: bool = False) -> str:
    """The checker compares every board with the words of its panel, as a reader would (the audit of 2026-10-28: 7 of 34
    boards fully backed their words): the comparison the words make, units and scales, labels nobody explains, and the
    same number told differently elsewhere. The paper comes last, so a fix can name the right rival or meaning."""
    boards = [{"n": p.n, "beat": p.beat, "narration": p.narration, "bubbles": [f"{b.speaker}: {b.text}" for b in p.dialogue],
               "board": _drawn_board(p.board), "claims": [c.get("text") for c in sb.abstract_claims if c.get("id") in p.claims]}
              for p in sb.panels if _checked_board(p)]
    others = [{"n": p.n, "words": " ".join([p.narration] + [b.text for b in p.dialogue] + ([board_text(p.board)] if p.board else []))}
              for p in sb.panels if not _checked_board(p)]
    focus = (f"\nPanels {', '.join(map(str, sorted(only)))} are new or rewritten; report problems on these only (a consistency "
             "problem with an older panel is reported on the new one)." if only is not None else "")
    if again and only is not None:   # round 17: half the problems the boards kept were on boards this check had flagged once
        # round 18 also told this second look to narrow the words to the board: OnQ's "shorter answers" and 1000 Layers'
        # manipulation tasks left the books, and both reader tests fell on those books (0.95 -> 0.68 on OnQ's fixed quiz)
        focus += (" They were just rewritten, most of them to fix a problem this check found: report one only when it is there "
                  "now.")
    return f"""Check every board of this manga script against the words of its own panel, the way a reader sees them: on each board panel the narration, then the speech bubbles, then the board drawn in the scene ("board" is all the text on it: "- " lines are bars or rows with their values, "[gap to X: +2.04]" is the gap the chart draws between the highlighted bar and bar X, and the footnote says the unit).
"claims" is context only, the abstract claim the panel helps teach, which the reader never sees: one claim is often taught over several panels, each board showing one part of it, so judge a board by what its own panel's words say, never by a part of its claim those words do not state. Words that give a part of the panel's claims as what the paper reports, without pointing at the chart ("The paper also reports shorter answers."), are not a problem even when the board does not draw that part: the manga must teach the whole abstract, and its recap marks the claims whose evidence is not drawn.
Report only these problems:
- "comparison": the words claim a comparison the board does not show: a rival or setting the words name is missing, the gap badge measures against something other than what the words compare with (e.g. the base model while the words say "beat both baselines"), "first in every setting" with only some settings or no rivals drawn, "cut by half" or "grows over time" with nothing that shows it. A summary the paper reports (wins out of total per setting, an average over its benchmarks) backs an "every" or "all" claim when the board says what was compared; then ask at most for those names in the subtitle, never for every row.
- "unit": the words and the board give the same quantity in different units or scales (percentage points in the words, a fraction such as 0.234 on the board; "13% shorter" beside a log change of -0.145), or a key number on the board has no unit or scale a reader can place. A board in the unit of the paper's table whose subtitle (a table: or its note) gives the paper's own reading of it ("0.081 = 8.1 percentage points", "-0.145 ≈ 13% shorter") is right: never ask to convert its values.
- "label": a bar label, column, row or abbreviation on the board that a newcomer cannot understand and that no earlier panel explains (a baseline's name, "FKL", numbers in parentheses).
- "consistency": the same metric in the same setting has a different value here than on another board or in another panel's words, with nothing that explains why.
- "form": the board's form hides what the words describe: a trend across positions, sizes or time shown as a one-row table instead of a bar chart over the bins the paper reports (the cells of one table row, one bar per bin, at most 7), or a baseline drawn as a category bar among the metrics instead of the reference the others are measured against.
What the program can draw, so that every fix is possible: a bar chart has one bar per label with its value, one highlighted bar and at most ONE gap badge, between the highlighted bar and the one bar marked "baseline" (a difference such as +2.04; a ratio only for costs, times and powers of ten, or when a bar's quote states the paper's own multiple such as "52×", never between percentages or shares), then a title, a subtitle and the unit footnote (a bar chart draws no note); a table has columns and rows; flow, compare and list boards have short lines; a diagram board is a drawing of the mechanism (its lines: "[...]" is a box with what is written in it, "#2" its step number, "A → B (label)" an arrow), which a designer redraws from your fix, so its fix says what the drawing must show or name. Nothing else on a chart: no second badge, no arrows, lines, curves or axis breaks. When the words claim more than a board can show (a gap to two rivals, a ratio of two scores such as "nearly doubled", a trend the board does not plot), the fix adds the missing bars or rows from the paper, or narrows the words to what the board shows ("beats the stronger baseline, Enhanced HotpotQA, by 0.85"; "from 31.65 to 58.82"); a part of the panel's claims never leaves the words: the fix keeps it as what the paper reports, apart from the chart.
At most 2 problems per board and {BOARD_CHECK_MAX} in all, worst first; an empty list when the boards back their words.{focus}
For each: "n" = the board panel to change, "kind", "problem" = one sentence quoting the exact words and board lines, "fix" = one concrete instruction for this panel from the PAPER TEXT (which bar to mark as the baseline, which rival or setting to add with its table cell, the unit to write and how the board states it, what a label means in a few plain words, which value is right); never invent a number: every number comes from the paper, and a conversion the paper does not state is not allowed (say the board's unit in the words instead).
Output JSON only: {{"problems": [{{"n": 12, "kind": "comparison", "problem": "...", "fix": "..."}}]}}

BOARD PANELS:
{json.dumps(boards, ensure_ascii=False)}

OTHER PANELS (for consistency and earlier explanations):
{json.dumps(others, ensure_ascii=False)}

PAPER TEXT:
{paper_text}"""


def fit_prompt_v3(items: list[dict], language: str, novice: bool = False) -> str:
    unit = _v3_budget(language)["unit"]
    prompt = f"""Rewrite each text so it fits its limit ("max", in {unit}) while keeping the exact meaning, every number and the tone ({LANG_NAMES.get(language, language)}). No underscores, LaTeX or markdown.
A "visual" item is a scene for the picture that mentions people or copies of a cast member: rewrite it as ONE English sentence (scene and action together) that shows the same idea with objects, places, diagrams and the cast only (a person of the paper, such as a driver, a worker or a customer, is played by one member of the item's "cast" wearing the role's prop: "Nezu in a red delivery cap"; only the item's "cast", each once; counts as simple objects such as cards), and states that nobody else is there.
Return JSON only: {{"items": [{{"id": "...", "text": "rewritten"}}]}}

ITEMS:
{json.dumps(items, ensure_ascii=False)}"""

    if novice:
        first, rest = prompt.split("\n", 1)
        return first + "\nPreserve each question as a question, its answer as an answer, and the plain prerequisite or reason this line teaches. Do not add jargon or facts. If the exact meaning cannot fit, return the original text instead of deleting the explanation.\n" + rest
    return prompt


def _cited(board: dict) -> dict:
    """The board as the fact-check sees it: a bar bound to a table cell says where its value comes from (table, column,
    row), so a chart titled for one dataset but drawn from another column stands out."""
    out = copy.deepcopy(board)
    for bar in out.get("bars") or []:
        if isinstance(bar, dict) and bar.get("cell"):
            bar.pop("quote", None)   # the row line repeats every column; "from" names the one that is drawn
            bar["from"] = bar.pop("cell")
    return out


def _board_check_items(p: PanelV2, whole: bool, paper_text: str = "") -> list[dict]:
    """Fact-check items of a board: the whole board (".b", its exact spec) and each string on it (".b.<path>"). A bar
    bound to a table cell names its table, column and row, and what the table reports: 1000 Layers drew Table 4's
    training hours (3.23 to 134.15) as "Extreme Depth Performance, Time at goal", and the bare "Table 4, column 1" gave
    the checker nothing to see it by."""
    board = p.board
    ledger = _ledger(paper_text) if whole and paper_text else None
    if ledger and isinstance(board, dict) and any(isinstance(b, dict) and b.get("cell") for b in board.get("bars") or []):
        board = _cited(board)
        for bar in board["bars"]:
            c = ledger.cell(bar.get("from")) if isinstance(bar, dict) and bar.get("from") else None
            if c is not None:
                table = ledger.table(c.table)
                caption = re.sub(rf"^\s*{re.escape(c.table_label)}\s*[.:]?\s*", "", table.caption or "", flags=re.I) if table else ""
                bar["from"] = (f'{c.table_label}, column "{c.column}", row "{c.label}"'
                               + (f' (the table: "{caption[:160]}")' if caption else ""))
    items = [{"id": f"{p.n}.b", "text": json.dumps(board, ensure_ascii=False)}] if whole else []
    return items + [{"id": f"{p.n}.b.{path}", "text": text} for path, text in _board_strings(p.board)]


def _board_check_rules(language: str) -> str:
    strings = ('Ids "N.b.<path>" are single strings on a board that a program draws exactly (a title, label, step, item or cell): '
               'check them like any line; "corrected" is the corrected string, about as short.')
    if language != "en-US":
        return "\n" + strings
    return ("\n" + 'Ids "N.b" are whole boards (charts, tables, diagrams, recap lists) given as their exact spec. ' + strings + """ Also flag:
- a chart or table that mixes different metrics, units, datasets, splits or training setups in one comparison (e.g. BLEU next to training cost, English-German next to English-French scores, or a semi-supervised result next to supervised-only ones);
- a label that presents an ensemble result as a single model (or a single model as an ensemble), or drops the model size or setting its value belongs to;
- a row or label called "prior best", "previous best", "previous state of the art" or similar when the paper does not say so for that comparison;
- a value attached to the wrong row, model or setting (compare the label with its quote and the paper's table);
- a chart whose title, subtitle or unit names another quantity than the table its bars come from reports (each bar's "from" says what the table reports): training time in hours drawn as performance, a cost drawn as accuracy;
- a "higher is better" or "lower is better" that is wrong for what the bars measure: a change in deliveries, accuracy or sales is better when higher even when the bars are negative drops, an error rate, a cost or a delay is better when lower;
- a bar chart whose drawn comparison is not one the paper makes: the program draws the gap between the highlighted bar and the bar marked "baseline", so flag a pair from different setups (semi-supervised vs supervised-only, dev vs test), a single model against an ensemble when the paper claims only a single-model result, or a baseline that is not the strongest earlier system the paper compares with.
For an "N.b" problem, "corrected" says in one sentence how the board must change.""")


# ----------------------------------------------------------------------------- v3: boards, numbers, lint (deterministic)

NUMBER_RE = re.compile(r"(?<![A-Za-z\d.])\d+(?:[.,]\d+)*")
# a paper shows results on its tests; the manga never closes the case for it
VERDICT_RE = re.compile(r"\b(?:prove[sdn]?|proving|proof that|case closed|perfect(?:ly)?|flawless(?:ly)?|always works|never fails)\b", re.I)
# a diagram's brief that asks for a chart (with numbers in it: a data chart, not a schematic of how something is spread)
DATA_CHART_BRIEF_RE = re.compile(r"\b(?:bar(?:s|\s+charts?|\s+graphs?)|histograms?|line\s+(?:charts?|graphs?)|plot(?:s|ted)?|axis|axes|"
                                 r"pie\s+charts?)\b", re.I)
CHART_VALUE_RE = re.compile(r"(?<![\w.:-])\d+(?:[.,]\d+)*%?(?![\w:)-])")   # a value, not the digits of a name, a time or "1)"
# the bet's ending vouching for every claim ("All the tested claims held up. You won the tart.", 15 of 176 books) beside
# a recap that ticks only the lines a chart showed; "All of these held up" too (round 23: beside a recap ticking one of six)
ALL_CLAIMS_RE = re.compile(r"(?<!\bnot )(?<!\bif )\b(?:(?:all|every)\b(?:\s+[\w'’-]+){0,2}?\s+(?:claims?|results?|promises?|findings?)\b"
                           r"[^.!?,]{0,20}?\b(?:held up|checked out|(?:was |were |is |are |been )?(?:verified|confirmed|proven))\b|"
                           r"all\s+(?:of\s+)?(?:these|those|them)\s+(?:held up|checked out|(?:were |are |have been )?"
                           r"(?:verified|confirmed|proven))\b)"
                           r"(?![^.!?]*\?)", re.I)
# props an image model fills with scribbled glyphs (the prototype's clipboard came back reading "好我")
WRITING_PROP_RE = re.compile(r"\b(?:clipboards?|notebooks?|notepads?|newspapers?|posters?|signs?|signboards?|calendars?|clocks?|"
                             r"wall clocks?|pens? (?:writing|scribbling|hovering)|scribbl\w*|jott\w*|menus?|labels?|chalkboards?|"
                             r"plaques?|tablets?|stone slabs?|scrolls?|banners?|certificates?|tickets?|receipts?|maps?|"
                             r"word cards?|cards? (?:with|showing|bearing) (?:words?|text|letters|the sentence)|"
                             r"(?:sentence|words?|letters) (?:written|laid out|spelled)\w*|"
                             r"question marks?|exclamation (?:marks?|points?))\b", re.I)   # the art check reads "?" as writing
# props shaped like a person or a creature: the image model draws them as one more character, and the art check rightly
# reports an extra figure (BERT's "toy robot" was redrawn three times)
FIGURE_PROP_RE = re.compile(r"\b(?:robots?|androids?|dolls?|puppets?|figurines?|statues?|mannequins?|snowm[ae]n|scarecrows?|"
                            r"teddy bears?|plush\w*|stuffed (?:animal|toy)s?|toy (?:animal|dog|cat|bear|bird|person|people)s?)\b", re.I)
# "No other figures or robots are drawn" asks for none (1000 Layer RL was flagged for it in every fix round)
_FIGURE = FIGURE_PROP_RE.pattern[2:-2]
NO_FIGURE_RE = re.compile(rf"\b(?:no|without|not any|nor)\b(?:\s+\w+){{0,3}}?\s+{_FIGURE}"
                          rf"(?:\s*,\s*{_FIGURE}|\s*,?\s*n?or\s+(?:(?!(?:a|an|the|one)\b)\w+\s+)?{_FIGURE})*\b", re.I)
# a chart in a scene: the illustrator draws a fake one, bars without labels or values that a reader takes for the evidence
# (Driver Misconduct's "Kumo looks at the bar chart" came back as two unreadable charts); charts belong on board panels
SCENE_CHART_RE = re.compile(r"\b(?:(?:bar|line|pie)\s+)?(?:charts?|graphs?|diagrams?|flowcharts?|infographics?|histograms?)\b|"
                            r"\b(?:scatter|line|box)\s+plots?\b|\bbar heights?\b|\b(?:rising|growing|shrinking|falling|tall|short)\s+bars\b|"
                            r"\bbars\s+(?:rise|rising|grow|growing|shrink|shrinking|climb\w*)\b|\baxis\b|"
                            r"\b(?:\w+\s+){0,3}bars?\s+on\s+(?:the|a)\s+(?:white)?board\b", re.I)   # "the fake remark bar on the board"
# chat windows and message boxes in a scene: the illustrator draws them as empty speech bubbles that readers take for
# missing text (Hivemind's first panel: "empty reply windows float above them")
CHAT_BOX_RE = re.compile(r"\b(?:(?:chat|reply|message|dialog(?:ue)?|pop-?up|answer|response)\s+(?:windows?|boxes|box|bubbles?|balloons?|panels?|screens?)|"
                         r"(?:speech|thought|text|word|chat|message)\s+(?:bubbles?|balloons?)|text\s+boxes|text\s+box|"
                         r"chat\s+(?:apps?|interfaces?|logs?|threads?|history)|"
                         r"(?:phone|laptop|computer|tablet)?\s*screens?\s+(?:with|showing|full of|filled with)\s+(?:text|messages|replies|answers|chats?))\b", re.I)
# words the picture is asked to show ("a phone that displays DELIVERY FAILED: Customer unavailable", "a card reading 'Sold
# out'"): the illustrator letters them, and the art check rejects every attempt (the delivery paper's first panel)
SPELLED_RE = re.compile(r"\b(?:display(?:s|ing|ed)?|show(?:s|ing)?|read(?:s|ing)?|say(?:s|ing)?|labell?ed|print(?:s|ed)?|"
                        r"flash(?:es|ing)?|spell(?:s|ing|ed)?)\b[\s:]*[\"“‘'][^\"”’']{2,}[\"”’']"
                        r"|\b[A-Z]{3,}(?:[\s:,!]+[A-Z]{2,})+\b")
QUOTED_SPAN_RE = re.compile(r"[\"“”‘’'][^\"“”‘’']{3,}[\"“”‘’']")   # quoted words (the paper's own example) are not verdicts
# a board's words that describe the drawing instead of the paper ("Empty checkboxes: we will verify each" leaked from
# the board spec onto two Chinese and one English claims board)
DRAWING_NOTE_RE = re.compile(r"\b(?:check\s?box(?:es)?|empty box(?:es)?|tick box(?:es)?|we will (?:verify|check)|to be (?:verified|checked)|"
                             r"drawn by|the program|board spec)\b|复选框|勾选框|将逐一验证", re.I)
# a chart element the program never draws: LongZero's "Dashed line = base model score 54.68" (twice), and the reader
# looked for the line that hid that retrieval-only scored below the untrained model
DRAWN_LINE_RE = re.compile(r"\b(?:dashed|dotted)\s+(?:lines?|bars?|box(?:es)?|outlines?)\b|\breference lines?\b|虚线|点线", re.I)
# words that point at a drawn exhibit: on a scene panel there is none ("Steps to reach the same accuracy" narrating a
# chart that was dropped)
CHART_WORDS_RE = re.compile(r"\b(?:this|these)\s+(?:chart|graph|plot|bars|table|diagram)\b|"
                            r"\b(?:the|this)\s+(?:chart|graph|plot)\s+(?:shows|says|proves)\b", re.I)
COMPARATIVE_RE = re.compile(r"\b(?:better|best|beats?|beating|outperform\w*|surpass\w*|exceed\w*|improv\w*|faster|cheaper|fewer|"
                            r"less|lower|higher|more than|over|fraction|state[- ]of[- ]the[- ]art|record|gain\w*|absolute)\b", re.I)
# the first words of the story pointing at something the reader has not seen ("This sameness risks..." opened Hivemind;
# LongZero began "This homework ...")
OPENING_POINTER_RE = re.compile(r"^\W*(?:this|these|those|such|that(?!'s|\s+(?:is|was|'s)\b))\b", re.I)
# a numbered point ("the first finding", "Second, ..."): a chapter that goes back to an earlier number jumps between
# points (Hivemind's panels 8 to 13 alternated first and second)
ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}
ORDINAL_RE = re.compile(r"(?:^|[.!?]\s+|\b(?:the|our|a|its|their)\s+)(first|second|third|fourth|fifth)\b"
                        r"(?=,|\s+(?:finding|result|point|reason|problem|question|claim|discovery|trick|idea|step|part|ingredient|"
                        r"test|experiment|lesson|issue|challenge|component|contribution|effect|insight|observation|case)s?\b)", re.I)
# words that introduce a gap ("more than 2.0 above", "by 2 points"): the number is a difference, not a value on the board
GAP_LEAD_RE = re.compile(r"(?:more than|over|by|about|nearly|almost|roughly|\+|±|up to)\s*$", re.I)
# a links board draws how strongly words attend to each other: it explains attention and confuses on anything else
# (High-accuracy Sampling's "links" between proof steps; no reader could tell what the arcs meant)
ATTENTION_RE = re.compile(r"\battention\b|\battend(?:s|ed|ing)?\b", re.I)
# a term the reader meets before any panel says what it is: readers' most frequent note (IV regression, FKL/DKL, GRPO on
# the four new papers of 2026-10-26). An abbreviation (QAT, LLM, RDiT; not BitDistiller, not "[MASK]") counts when the paper
# uses it; a glossary term always counts
ACRONYM_RE = re.compile(r"(?<![\w\[-])[A-Z][A-Z\d]*[a-z]?[A-Z][A-Z\d]*(?=s?(?![\w\]-]))")
COMMON_ACRONYMS = frozenset("AI GPU CPU US USA UK EU OK PhD TV ID IQ PDF CEO DNA API FAQ PC IT".split())
# an explanation right after the term ("IV regression: ...", "SQuAD v1.1 (...)", "GRPO, a method that ...", "FTR means ...",
# "MATH-500: 500 math problems", "d1, ESPO, SPG are prior diffusion RL methods"), never a number or a pointer ("BLEU: 28.4",
# "MMLU (Table 3)")
TERM_DEF_AFTER_RE = re.compile(r"(?:\s+(?:v?\d[\w.]*|[A-Z][\w.-]*\d[\w.-]*|regression|divergence|loss|training|models?|method|scores?|"
                               r"benchmark|tests?|dataset|design|estimates?|algorithm|framework|tasks?|metric))?\s*(?:"
                               r"(?:\(|:|=|—|–|\s-\s)(?!\s*(?:(?!\d[\d,]*\s+[a-z]{3,})[\d.%+\-−±~≈<>$)]|"
                               r"(?:Table|Tab|Fig|Figure|Sec|Section|Eq|Appendix|App)\b))|"
                               r",\s*(?:a|an|i\.e\.|which|that is|or|meaning)\b|\s+(?:means?|stands? for|is\s+(?:a|an|the|how|when|what|where|whether)|"
                               r"are\s+(?:a|an|the)|measures?|refers? to|counts?|checks?|tests?|asks?)\b|"
                               r"(?:\s*,\s*[\w*^∗.-]+){0,4}(?:,?\s+and\s+[\w*^∗.-]+)?\s+are\s+(?:[\w-]+\s+){0,3}(?:methods?|models?|systems?|"
                               r"baselines?|benchmarks?|tests?|datasets?|tasks?|algorithms?|approaches|techniques|metrics?)\b)", re.I)
# ... or a name given to what was just described ("That is chain-of-thought prompting", "The best image model: Inception";
# not a one-word label such as "Columns: Condition")
TERM_DEF_BEFORE_RE = re.compile(r"(?:\b(?:called|known as|named|termed|dubbed|we call (?:it|this|that))|\b(?:that|this|it)(?:'s|\s+is)"
                                r"(?:\s+(?:a|an|the))?|\w+\s+\w+\s*:)\s*[\"“'‘]?\s*$", re.I)
SMALL_WORDS = frozenset("of in for the and to a an on with by from".split())
# words of a glossary meaning that say nothing about the term ("a standard benchmark; higher is better")
PLAIN_GENERIC = frozenset("higher lower better worse model models used using use standard common commonly data training method "
                          "methods measure measures score scores".split())
PLAIN_DIRECTION_RE = re.compile(r"\b(?:higher|lower|larger|smaller)\b[^.;,]*?\bbetter\b", re.I)
ROMAN_RE = re.compile(r"(?<![\w-])(?:type|phase|part|table|section|stage|class|level|grade|tier|figure|fig\.|chapter|study|appendix|"
                      r"category|round|step)\s+(X{0,3}(?:IX|IV|VI{0,3}|I{1,3}))(?![\w-])", re.I)
TERMS_PER_PANEL = 3
# the picture of a teaching scene shows the idea itself (the audit of 2026-10-28: 0 of 76 panels did; 60 drew only the cast
# standing, walking or reacting), with the book's fixed picture symbols
PICTURE_BEATS = ("problem", "idea", "how", "payoff", "limits")
# words that do not tell one picture symbol from another (colors do: "a red card" and "a grey card" are two symbols)
SYMBOL_GENERIC = frozenset("small large big tiny little bold plain blank simple single shape shaped object objects drawn wordless "
                           "color colored colour coloured one two three four many each every thing things together".split())


def _comparison_shown(panels: list[PanelV2]) -> bool:
    """A panel among these shows a comparison as a drawing: a bar chart with two or more bars, or a table with numbers (a
    table of words is no drawn comparison: the proof-board rule rejects it, and the two rules must agree)."""
    for p in panels:
        b = p.board if p.kind == "board" and isinstance(p.board, dict) else None
        if b and ((b.get("kind") == "table" and _number_tokens(" ; ".join(_board_cells(b))))
                  or (b.get("kind") == "bar" and len(b.get("bars") or []) >= 2)):
            return True
    return False


# the story checks the evidence the panels show; it never certifies the whole paper ("every claim was verified" was a
# template line that readers caught on 9 of 13 reports while some claims had no evidence on any panel)
# "Red = hot leads", "Tangled string = complex routes": a symbol key written as a legend line (letters on both sides)
LEGEND_LINE_RE = re.compile(r"\b(?!(?:lower|higher|more|less|bigger|smaller|larger|longer|shorter|up|down|above|below|positive|"
                            r"negative|left|right)\s*=)[A-Za-z][A-Za-z'’ -]{1,40}?\s*(?<![<>=!])=(?!=)\s*[A-Za-z]"   # not "lower = shorter calls"
                            r"|\b[A-Za-z]+\s*:\s*[A-Za-z]+(?:\s*,\s*[A-Za-z]+\s*:\s*[A-Za-z]+)+", re.I)   # "Chest: answer, string: reasoning"
ALL_VERIFIED_RE = re.compile(r"\b(?:every|all(?: the)?|each)\s+(?:paper\s+)?(?:claims?|lines?|results?|points?|promises?)\s+"
                             r"(?:(?:was|were|is|are|got|have been|has been)\s+)?(?:verified|checked|confirmed|proven|held|delivered)\b|"
                             r"\bverified\s+(?:paper\s+)?claims\b|\bdelivered exactly what it (?:claimed|promised)\b|"
                             r"\bclaims?\s+(?:all\s+)?(?:verified|confirmed)\b|\b(?:confirmed|verified|proven)\s+(?:findings|results|claims)\b|"
                             r"\b(?:findings|results)\s+(?:were|are|got|have been)\s+(?:all\s+)?(?:confirmed|verified|proven)\b|"
                             r"\bconfirmed by the data\b", re.I)
# small numbers a line may write as words ("eight GPUs", "first place", 八块 GPU): they still show the figure
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                "eleven": 11, "twelve": 12, "first": 1, "second": 2, "third": 3}
NUMBER_WORD_RE = re.compile(r"\b(?:" + "|".join(NUMBER_WORDS) + r")\b", re.I)
CJK_NUMERALS = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
YEAR_RE = re.compile(r"(?:19|20)\d\d")
# list markers and citation brackets in a quote are no figures: "two key factors: (1) introducing non-linearity ... (2) ..."
# asked Gated Attention for the numbers 1 and 2, and a mechanism claim was placed on the results pages; "1) ... 2)" and
# full-width "（1）" are the same markers ("1." is not: "by 2. The" ends on a value)
MARKER_RE = re.compile(r"[(（]\s*\d{1,2}\s*[)）]|(?:^|(?<=[\s:;,]))[1-9]\)(?=\s+[^\W\d_])|\[\s*\d{1,3}(?:\s*[,–-]\s*\d{1,3})*\s*\]")
HIDDEN_BOARD_KEYS = frozenset({"kind", "quote", "paper_label", "highlight", "highlight_rows", "illustrative", "from", "weights", "higher_is_better",
                               "cell", "cell_ids", "unbound"})


def _numbers(text: str) -> list[str]:
    """The number tokens of a line, sorted: a rewrite or translation must keep them all."""
    return sorted(re.findall(r"\d+(?:[.,]\d+)*", str(text or "")))


def _number_tokens(text: str) -> list[tuple[str, str]]:
    """(as written, canonical value) of each number in a text, read the way services.boards reads board values:
    "2.3 · 10^19", "2.3 × 10¹⁹" and "2.3e19" are one number, "2.0" equals "2", and digits inside names (GPT-3, CIFAR-10,
    v1.1) are no numbers. Plain decimals when services.boards is unavailable."""
    text = str(text or "")
    boards = _optional("boards")
    if boards is not None:
        try:
            return [(n.text, n.key) for n in boards.find_numbers(text)]
        except Exception as e:  # fall back to plain decimals
            print(f"[StoryboarderV2] number scan failed: {type(e).__name__}: {e}")
    out = []
    for tok in NUMBER_RE.findall(re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)):
        try:
            out.append((tok, format(Decimal(tok.replace(",", ".")).normalize(), "f")))
        except InvalidOperation:
            continue
    return out


def _number_keys(text: str) -> set[str]:
    """Canonical values of the numbers a reader sees in a text, small numbers written as words or Chinese numerals
    included ("eight GPUs", "first place", "两个" show 8, 1 and 2)."""
    text = str(text or "")
    words = [NUMBER_WORDS[m.group().lower()] for m in NUMBER_WORD_RE.finditer(text)]
    return {key for _, key in _number_tokens(text)} | {str(v) for v in words + [CJK_NUMERALS[c] for c in text if c in CJK_NUMERALS]}


def _key_numbers(claim: dict) -> list[tuple[str, str]]:
    """The numbers a reader must see for a claim, as (written, canonical value): those in its abstract quote, except
    years (WMT 2014), list markers and citations ("(1)", "[12]")."""
    out: dict[str, str] = {}
    for written, key in _number_tokens(MARKER_RE.sub(" ", str(claim.get("quote") or claim.get("text") or ""))):
        if not YEAR_RE.fullmatch(written):
            out.setdefault(key, written)
    return [(written, key) for key, written in out.items()]


def _shown_about(claim: dict, written: str, key: str, shown: set) -> bool:
    """A number the claim gives as an approximation ("about 8 percentage points") is shown by a value that rounds to it:
    Better Leads drew 8.1 on its reallocation chart, and the rule asked for "8" on the next panel, whose chart was about
    call length, so the board check and this rule undid each other's fix (round 15)."""
    quote = MARKER_RE.sub(" ", str(claim.get("quote") or claim.get("text") or ""))
    if not re.search(r"(?:about|around|roughly|approximately|nearly|almost|~|≈)\s*" + re.escape(written) + r"(?![\d.,])", quote, re.I):
        return False
    places = len(written.split(".")[1].rstrip("%")) if "." in written else 0
    try:
        want = abs(Decimal(key))
        step = Decimal(1).scaleb(-places)   # half up, as papers round: 4.5 percentage points is "about 5"
        return any(abs(Decimal(v).quantize(step, rounding=ROUND_HALF_UP)) == want for v in shown if re.fullmatch(r"-?\d+(?:\.\d+)?", v))
    except InvalidOperation:
        return False


def _figures_kept(source: str, new: str) -> bool:
    """A translated line still shows every number of the English line (compared as values: "2.0" may become "2", a lone
    "8" may become 八); numbers it adds are fine."""
    have = _number_keys(new)
    return all(key in have for _, key in _number_tokens(source))


_TRANSLATION_CJK_DIGITS = "零〇一二两兩三四五六七八九"
_TRANSLATION_CJK_SCALES = "十百千万萬亿億兆京"
_TRANSLATION_CJK_CHARS = _TRANSLATION_CJK_DIGITS + _TRANSLATION_CJK_SCALES
_TRANSLATION_CJK_RUN = re.compile(
    r"[+-]?(?:(?:[0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?|[" + _TRANSLATION_CJK_CHARS + r"])+"
    r"(?:[.点點・](?:[0-9]+|[" + _TRANSLATION_CJK_CHARS + r"])+)*")


def _translation_cjk_value(token: str) -> str:
    """Bounded positional digits, descending 十/百/千 and 万/亿 sections (traditional forms too).

    Decimal coefficients may precede a scale; fractional digits follow . / 点 / 點 / ・.
    This is not a natural-language parser: no fractions, colloquial elision or financial
    numerals. Unsupported 兆/京 and malformed runs remain opaque, never a numeric prefix.
    """
    raw = token
    token = token.translate(str.maketrans(dict(zip(_TRANSLATION_CJK_DIGITS, "0012223456789")) |
                                        {"萬": "万", "億": "亿", "点": ".", "點": ".", "・": "."}))
    sign = -1 if token.startswith("-") else 1
    token = token.lstrip("+-")

    def parse(part: str) -> Decimal:
        for scale, unit in (("亿", 100000000), ("万", 10000), ("千", 1000), ("百", 100), ("十", 10)):
            if scale not in part:
                continue
            if part.count(scale) != 1:
                raise ValueError("repeated scale")
            head, tail = part.split(scale)
            # Small units take a digit coefficient, not another small-unit expression.
            if unit < 10000 and any(c in head for c in "十百千"):
                raise ValueError("ascending small units")
            coefficient, remainder = parse(head or "1"), parse(tail or "0")
            if remainder >= unit or (tail and coefficient != coefficient.to_integral_value()):
                raise ValueError("overlapping sections")
            return coefficient * unit + remainder
        if not re.fullmatch(r"(?:(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.[0-9]+)?|\.[0-9]+)", part):
            raise ValueError("unsupported numeric syntax")
        return Decimal(part.replace(",", ""))

    try:
        return format((sign * parse(token)).normalize(), "f")
    except (ValueError, ArithmeticError):
        return "unparsed-cjk:" + raw


def _translation_numbers(text: str) -> Counter:
    """Translation-only multiset: retain canonical values and their occurrence counts.

    Reuse the board scanner for commas, percentages and scientific notation. Small
    written numbers count too; percent's Chinese prefix is a unit, not another 100.
    Do not tighten the shared evidence/fact-check guards with this translation rule.
    """
    # Fold full-width ASCII only: NFKC on the whole line would destroy scientific
    # superscripts before the board scanner gets to read them.
    text = "".join(unicodedata.normalize("NFKC", c) if "\uff01" <= c <= "\uff5e" else c
                   for c in str(text or ""))
    text = re.sub(r"百分[之点比]", " ", text)
    text = re.sub(r"唯一|进一步", " ", text)
    text = re.sub(r"(?<![0-9" + _TRANSLATION_CJK_CHARS + r"])万一", " ", text)
    text = re.sub(r"(?<![0-9" + _TRANSLATION_CJK_CHARS + r"])千万(?=要|别|別|不要|不可)", " ", text)
    # Discourse markers and a rate's time unit are not cardinal quantities.
    text = re.sub(r"(^|[.!?]\s+)(?:first|second|third)(?=\s*[,，:：])", r"\1 ", text, flags=re.I)
    text = re.sub(r"\bper\s+second\b", "per time unit", text, flags=re.I)
    scales = {"hundred": 100, "thousand": 1000, "million": 1000000, "billion": 1000000000}
    text = re.sub(r"\b(" + "|".join(NUMBER_WORDS) + r")\s+(hundred|thousand|million|billion)\b",
                  lambda m: str(NUMBER_WORDS[m[1].lower()] * scales[m[2].lower()]), text, flags=re.I)
    # These ordinary words do not assert the number one.
    text = re.sub(r"(?<![第0-9" + _TRANSLATION_CJK_CHARS + r"])一(?:起|样|些|下|直|般|致|旦)", " ", text)
    text = re.sub(r"(?<![第0-9" + _TRANSLATION_CJK_CHARS + r"])一定", " ", text)
    # Bounded lexical contexts for 'sufficient/very', not a blanket 十分 deletion:
    # 十分間 / 得了十分 / 二十分 still assert quantities. Unlisted ambiguous uses
    # remain numeric; this scanner does not attempt general CJK word segmentation.
    text = re.sub(r"(?<![0-9" + _TRANSLATION_CJK_CHARS + r"])十分"
                  r"(?=です|だ|な|に|で(?:す|ある|は)|有用|重要|必要|清楚|明显|明顯|满意|滿意|困难|困難)", " ", text)
    text = re.sub(r"(?:负|負|マイナス)(?=[0-9" + _TRANSLATION_CJK_CHARS + r"])", "-", text)
    text = re.sub(r"(?:正|プラス)(?=[0-9" + _TRANSLATION_CJK_CHARS + r"])", "+", text)
    text = text.replace("−", "-")
    values = Counter()

    def cjk_number(match):
        token = match.group()
        if not any(c in token for c in _TRANSLATION_CJK_CHARS + "点點・"):
            return token  # commas, scientific notation and model names stay with the board scanner
        values[_translation_cjk_value(token)] += 1
        return " "  # consume the entire value, including any Arabic coefficient

    text = _TRANSLATION_CJK_RUN.sub(cjk_number, text)
    boards = _optional("boards")
    if boards is not None:
        # find_numbers' offsets refer to _prep's normalized text. Retain unary signs
        # here only; board/fact-check callers keep their existing unsigned semantics.
        normalized = boards._prep(text)
        for number in boards.find_numbers(text):
            prefix = normalized[:number.start].rstrip()
            negative = prefix.endswith("-") and not re.search(r"(?:[A-Za-z]|[0-9.]\s*)-$", prefix)
            values[format((-Decimal(number.key)).normalize(), "f") if negative else number.key] += 1
    else:
        values.update(key for _, key in _number_tokens(text))
    values.update(str(NUMBER_WORDS[m.group().lower()]) for m in NUMBER_WORD_RE.finditer(text))
    values["2"] += len(re.findall(r"\bboth\b", text, re.I))
    return values


def _translation_numbers_equal(source: str, translated: str) -> bool:
    """Compare quantities, allowing an English article to become a written Chinese classifier.

    'a model' may be 模型 or 一个模型. This allowance cannot hide an explicit added
    digit 1, a lost source 1/one, or any other added/lost value.
    """
    before, after = _translation_numbers(source), _translation_numbers(translated)
    extra_one = after["1"] - before["1"]
    if extra_one > 0 and "1" not in {key for _, key in _number_tokens(translated)}:
        articles = len(re.findall(r"\b(?:a|an)\b", source, re.I))
        classifiers = len(re.findall(r"一[个個种種名位项項条條台套张張只隻件座本]", translated))
        if extra_one <= min(articles, classifiers):
            after["1"] -= extra_one
    return before == after


def _normalized_board(spec):
    boards = _optional("boards")
    if boards is None or not isinstance(spec, dict):
        return spec
    try:
        norm = boards.normalize_board(spec)
    except Exception as e:  # a malformed spec stays as it is, so lint reports it
        print(f"[StoryboarderV2] board normalization failed: {type(e).__name__}: {e}")
        norm = None
    return norm if isinstance(norm, dict) else spec


def board_issues(spec, paper_text: str) -> list[str]:
    """What is wrong with a board spec: its shape (services.boards) and its numbers (services.evidence)."""
    if not isinstance(spec, dict) or not spec:
        return ["board: missing"]
    boards = _optional("boards")
    if boards is None:
        return []
    try:
        norm = boards.normalize_board(spec)
        if not isinstance(norm, dict):
            return [f"board: {x}" for x in (boards.board_problems(spec) or ["not a valid board (unknown kind or missing fields)"])]
        problems = [f"board: {x}" for x in boards.board_problems(norm)]
    except Exception as e:
        return [f"board: not a valid board ({type(e).__name__}: {e})"]
    evidence = _optional("evidence")
    if evidence is not None:
        try:
            problems += [_with_hints(f"evidence: {x}", paper_text) for x in evidence.verify_board(norm, paper_text)]
        except Exception as e:  # a checker bug is no reason to drop the board; the fact-check still reads it
            print(f"[StoryboarderV2] WARNING: board evidence check failed: {type(e).__name__}: {e}")
    return problems


ROW_LABEL_RE = re.compile(r'^evidence: (?:bar|row \d+|table row \d+ \() ?"([^"]+)"\)?(?: cell \d+)?: the value ([^ ]+)')


def _with_hints(problem: str, paper_text: str, limit: int = 3) -> str:
    """An unsupported bar or row gets the paper lines that name it together with numbers, table rows and values close
    to the written one first: the fix round can copy the exact value and quote from there instead of a figure, whose
    extracted labels and values sit on different lines (CoT's Figure 2 "57" is Table 1's "56.9")."""
    m = ROW_LABEL_RE.match(problem)
    boards = _optional("boards")
    if not m or boards is None:
        return problem
    want = {w for w in _content_tokens(m.group(1)) if not w[0].isdigit()}
    target = boards.parse_value(m.group(2))
    if not want:
        return problem
    scored = []
    for ln in _own_text(paper_text).splitlines():
        ln = ln.strip()
        if len(ln) < 8:
            continue
        overlap = len(want & _content_tokens(ln))
        values = [n for n in boards.find_numbers(ln) if not n.short]
        exact = target is not None and any(abs(float(n.key) - target) < 1e-9 for n in values)
        named = overlap >= max(1, math.ceil(0.5 * len(want)))
        if not values or not (named or exact and len(values) >= 2):   # a table row with the value, whatever its row header
            continue
        close = target is not None and any(abs(float(n.key) - target) <= max(1.0, 0.03 * abs(target)) for n in values)
        scored.append((named and close, exact, len(values) >= 2, overlap, -len(scored), ln[:160]))
    hits = [x[-1] for x in sorted(scored, reverse=True)[:limit]]
    return problem + (" Paper lines that name it: " + " | ".join(f'"{h}"' for h in hits) if hits else "")


def board_text(spec) -> str:
    """Everything a reader sees on a board (titles, labels, values, steps, cells, source), not the quotes behind it."""
    if isinstance(spec, dict) and str(spec.get("kind") or "").lower() == "diagram":   # its labels, not its geometry or brief
        labels = _diagram_lines(spec)
        return " ".join([str(spec.get(k) or "") for k in ("title", "subtitle") if spec.get(k)] + labels
                        + [str(spec.get(k) or "") for k in ("note", "source") if spec.get(k)]).strip()
    out: list[str] = []

    def walk(value, key=""):
        if key in HIDDEN_BOARD_KEYS:
            return
        if isinstance(value, str):
            out.append(value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool) and key == "value":
            out.append(str(value))
        elif isinstance(value, dict):
            for k, v in value.items():
                walk(v, str(k))
        elif isinstance(value, list):
            for v in value:
                walk(v, key if key in ("cells", "items", "steps", "tokens", "columns") else "")

    walk(spec if isinstance(spec, dict) else {})
    return " ".join(out)


def visible_text(p: PanelV2) -> str:
    """The reader's view of a panel: bubbles, narration and board."""
    return " ".join([b.text for b in p.dialogue] + [p.narration, board_text(p.board) if p.board else ""]).strip()


def _board_strings(spec) -> list[tuple]:
    """(path, text) of each translatable board string, from services.boards ([] without it)."""
    boards = _optional("boards")
    if boards is None or not isinstance(spec, dict):
        return []
    try:
        return [(path, str(text)) for path, text in boards.board_strings(spec) if isinstance(text, str) and text.strip()]
    except Exception as e:
        print(f"[StoryboarderV2] board strings unavailable: {type(e).__name__}: {e}")
        return []


def _board_all_texts(spec) -> dict[str, str]:
    """{path: text} of every text drawn on a board, numbers and formulas included ({} without services.boards): a
    correction can leave a label that is a bare number or symbol ("about 4 steps" to "4"), which _board_strings never
    lists. Round 23 crashed a whole book on that missing path."""
    boards = _optional("boards")
    if boards is None or not isinstance(spec, dict):
        return {}
    try:
        return {str(path): str(text) for path, text in boards.board_texts(spec)}
    except Exception as e:
        print(f"[StoryboarderV2] board texts unavailable: {type(e).__name__}: {e}")
        return {}


def _set_board_strings(spec, updates: dict[str, str]):
    """The board with some strings replaced (``updates`` keyed by str(path) as the prompts show it), or ``spec`` itself
    when nothing changed: services.boards refuses replacements that change a number."""
    before = {str(path): (path, text) for path, text in _board_strings(spec)}
    real = {before[k][0]: v for k, v in updates.items() if k in before}
    boards = _optional("boards")
    if boards is None or not real:
        return spec
    try:
        new = boards.set_board_strings(spec, real)
    except Exception as e:
        print(f"[StoryboarderV2] board strings not applied: {type(e).__name__}: {e}")
        return spec
    after = {str(path): text for path, text in _board_strings(new)} if isinstance(new, dict) else {}
    return new if any(after.get(k) != text for k, (_, text) in before.items()) else spec


def _set_diagram_numbers(spec, updates: dict[str, str], paper_text: str, example: str = ""):
    """A diagram with the fact-check's corrections of its labels that change a number (services.boards refuses those, as
    a translation must never change a number): a label is no data cell, and the checker corrects it from the paper, so
    the new text stands when every number in it is the paper's (or the made-up running example's). Round 23's review:
    "ByteNet 26.36 BLEU" corrected to the paper's 23.75 was dropped without a word, and the panel counted as checked."""
    if not isinstance(spec, dict) or not isinstance(spec.get("elements"), list):
        return spec
    evidence = _optional("evidence")
    out = None
    for path, value in updates.items():
        m = re.fullmatch(r"elements\.(\d+)\.(text|label)", str(path))
        value = re.sub(r"\s+", " ", str(value or "")).strip()
        if not m or not value or int(m.group(1)) >= len(spec["elements"]):
            continue
        el = spec["elements"][int(m.group(1))]
        old = el.get(m.group(2)) if isinstance(el, dict) else None
        if not isinstance(old, str) or old == value or _numbers(old) == _numbers(value):
            continue   # no label there, or a change of words only (already applied by _set_board_strings)
        try:
            bad = evidence.verify_text_numbers(value, paper_text, allowed=[example]) if evidence is not None else []
        except Exception:
            bad = ["unchecked"]
        if bad:
            continue
        out = out if out is not None else copy.deepcopy(spec)
        out["elements"][int(m.group(1))][m.group(2)] = value
    return spec if out is None else out


LOWER_BETTER_RE = re.compile(r"\b(?:lower|smaller|less|fewer)\s+is\s+better\b|越(?:低|小|少)越好", re.I)
TEX_MARK_RE = re.compile(r"\$")
# markdown bold ("**28.4**") loses its marks; significance stars ("0.234***", "** p<0.05") are the paper's and stay
MD_BOLD_RE = re.compile(r"(?<![\d*)\](^])\*\*(?![*\s)]|\s*p\s*[<≤=])(.+?)(?<=[^\s(*])\*\*(?![*)])")   # never opens on a starred number,
# a legend's "**p<0.05" or a bracketed "(**)"


def _numbers_supported(text: str, paper_text: str) -> bool:
    """No number in the text that the paper lacks (lone digits are fine); True without services.evidence."""
    evidence = _optional("evidence")
    if evidence is None or not str(text or "").strip():
        return True
    try:
        return not evidence.verify_text_numbers(str(text), paper_text)
    except Exception as e:  # a checker bug is no reason to drop text
        print(f"[StoryboarderV2] number check failed: {type(e).__name__}: {e}")
        return True


def _strip_marks(value, key: str = ""):
    """A board without TeX / markdown marks in what is drawn; quotes and paper labels stay verbatim (they are matched
    against the paper)."""
    if key in ("quote", "paper_label"):
        return value
    if isinstance(value, str):
        return re.sub(r"\s+", " ", TEX_MARK_RE.sub("", MD_BOLD_RE.sub(r"\1", value))).strip()
    if isinstance(value, dict):
        return {k: _strip_marks(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [_strip_marks(v) for v in value]
    return value


def _bars_as_table(board: dict) -> dict:
    """Bars that mix units or notations as a table: the values and their quotes stay, and no bar length suggests that
    they compare."""
    table = {"kind": "table", "title": board.get("title", ""), "columns": ["", board.get("unit", "")],
             "rows": [{"cells": [b.get("label", ""), b["value"]], **({"quote": b["quote"]} if b.get("quote") else {})}
                      for b in board["bars"]],
             "highlight_rows": [i for i, b in enumerate(board["bars"]) if b.get("highlight")]}
    return table | {k: board[k] for k in ("subtitle", "note", "source") if board.get(k)}


LEDGER_ROW_RE = re.compile(r"^\[[A-Z]{1,3}\d{1,3}[a-z]?\.r\d+\]")


def _snap_to_paper(bar: dict, paper_text: str, boards) -> Optional[dict]:
    """A bar whose value the paper does not print as written (read off a figure: CoT's "57" and "55") snapped to the paper
    line that names it with the nearest value within max(1, 2%): its exact number and that line as the quote. None when
    no such line exists or two lines are equally near (no guessing between rows)."""
    target = boards.parse_value(bar.get("value"))
    names = [{w for w in _content_tokens(str(bar.get(k) or "")) if not w[0].isdigit()} for k in ("paper_label", "label")]
    names = [n for n in names if n]   # the paper's row name first, the display label second: either may name the line
    if target is None or not names:
        return None
    best: list[tuple[float, str, str]] = []
    for raw in _own_text(paper_text).splitlines():
        ln = raw.strip()
        if LEDGER_ROW_RE.match(ln):
            continue   # a ledger row holds every column of its table row: services.ledger binds near values within one column
        words = _content_tokens(ln)
        if len(ln) < 6 or not any(len(n & words) >= max(1, math.ceil(0.5 * len(n))) for n in names):
            continue
        values = [n for n in boards.find_numbers(ln) if not n.short and n.exponent is None and not YEAR_RE.fullmatch(n.text)]
        # a near value only from a table row (columns, several numbers): a prose sentence that mentions the system can hold
        # any number ("GPT-3 on SST-2 ... (54.3%)" is no reading of the prior best's 55)
        row = len(values) >= 3 or (len(values) >= 2 and len(re.findall(r"\S {2,}(?=\S)", ln)) >= 1)
        for n in values:
            gap = abs(float(n.key) - target)
            if gap == 0 or (row and gap <= max(1.0, 0.02 * abs(target))):
                best.append((gap, n.text, ln[:200]))
    if not best:
        return None
    best.sort(key=lambda t: t[0])
    if len(best) > 1 and best[1][0] == best[0][0] and best[1][1] != best[0][1]:
        return None
    _gap, value, line = best[0]
    print(f"[StoryboarderV2] bar \"{bar.get('label')}\": {bar.get('value')} snapped to the paper's {value} ({line[:80]})")
    return {**bar, "value": value, "quote": line}


def _salvage_board(spec, paper_text: str) -> Optional[dict]:
    """Last resort for a board the fix rounds left invalid, so that its evidence is not lost with it: the normalized board
    without what the paper does not support (bars, rows, steps and items holding a number the paper lacks; a title,
    subtitle, unit, note, source, column or side label holding one is cleared), "lower is better" taken from its own
    words, bars that mix units or notations shown as a table, and $ / ** marks removed. Counts and lengths outside the
    limits stay: the renderer copes with them. None when nothing usable is left or a number is still unsupported."""
    boards, evidence = _optional("boards"), _optional("evidence")
    if boards is None or not isinstance(spec, dict):
        return None

    def ok(text) -> bool:
        return _numbers_supported(str(text or ""), paper_text)

    def alone(entry: dict) -> bool:   # one bar or table row checked on its own: its quote, its label, its paper line
        return evidence is None or not evidence.verify_board(entry, paper_text)

    try:
        board = boards.normalize_board(spec)
        if not isinstance(board, dict):
            return None
        board = _strip_marks(copy.deepcopy(board))
        kind = board["kind"]
        if kind == "bar":
            said = " ".join([board["title"], board.get("subtitle", ""), board.get("unit", "")])
            if board.get("higher_is_better", True) and LOWER_BETTER_RE.search(said):
                board["higher_is_better"] = False   # the board's own words win: the delta badge must not call worse better
            board["bars"] = [b if alone({"kind": "bar", "title": "", "bars": [b]}) else _snap_to_paper(b, paper_text, boards)
                             for b in board["bars"]]
            board["bars"] = [b for b in board["bars"] if b and alone({"kind": "bar", "title": "", "bars": [b]})]
            if not board["bars"]:
                return None
            if len(board["bars"]) < 2 or any(x.startswith("the bars mix") for x in boards.board_problems(board)):
                # a one-bar chart compares nothing, but its number is the panel's evidence: a one-row table shows it (the
                # delivery book's -1.60% lost its made-up zero bar, became a scene and was read over a blank board)
                board, kind = _bars_as_table(board), "table"
        if kind == "table":
            rows = board["rows"]
            kept = [i for i, r in enumerate(rows) if alone({"kind": "table", "title": "", "columns": [], "rows": [r]})]
            board["rows"] = [rows[i] for i in kept]
            board["highlight_rows"] = [kept.index(i) for i in board.get("highlight_rows", []) if i in kept]
            board["columns"] = [c if ok(c) else "" for c in board["columns"]]
            if not board["rows"]:
                return None
        elif kind == "flow":
            kept = [i for i, step in enumerate(board["steps"]) if ok(step)]
            board["highlight"] = [kept.index(i) for i in board.get("highlight", []) if i in kept]
            board["steps"] = [board["steps"][i] for i in kept]
            if len(board["steps"]) < 2:
                return None
        elif kind == "list":
            board["items"] = [x for x in board["items"] if ok(x)]
            if not board["items"]:
                return None
        elif kind == "compare":
            for side in (board["left"], board["right"]):
                side["items"] = [x for x in side["items"] if ok(x)]
                side["label"] = side["label"] if ok(side["label"]) else ""
            if not (board["left"]["items"] and board["right"]["items"]):
                return None
        for key in ("subtitle", "unit", "note", "source"):
            if key in board and not ok(board[key]):
                del board[key]
        if not ok(board.get("title")):
            board["title"] = ""
        final = boards.normalize_board(board)
        if not isinstance(final, dict) or (evidence is not None and evidence.verify_board(final, paper_text)):
            return None
        return final
    except Exception as e:  # a salvage bug must not fail the job: the panel is then drawn as a scene
        print(f"[StoryboarderV2] board salvage failed: {type(e).__name__}: {e}")
        return None


def _example_parts(sb: StoryboardV2) -> list[str]:
    """The parts of the running example the analysis wrote down (brief_example): what the diagrams draw."""
    example = sb.brief.get("running_example") if isinstance(sb.brief.get("running_example"), dict) else {}
    return [x for x in _as_list(example.get("parts")) if isinstance(x, str) and x.strip()]


def _illustrative_example(sb: StoryboardV2) -> str:
    """The running example when it is made up (its own numbers may appear in the text although the paper lacks them).
    Never its parts: the prompt allows a made-up case small counts only (which the number checks ignore anyway), and a
    part "on-time rate 62.5%" let the same number pass as a result in a bubble and on a diagram (round 25's review)."""
    example = sb.brief.get("running_example") if isinstance(sb.brief.get("running_example"), dict) else {}
    return "" if example.get("from_paper", True) else sb.running_example


def _badge_text(p: PanelV2) -> str:
    """The gap the panel's own chart draws ("+0.85", "≈1/5.2"), which the program computed from two of the paper's values:
    words may say it ("beats the stronger baseline by 0.85", as the board check advises). LongZero lost that sentence to
    the number check, which found 0.85 nowhere in the paper."""
    boards = _optional("boards")
    if boards is None or p.kind != "board" or not isinstance(p.board, dict):
        return ""
    try:
        badge = boards.delta_badge(p.board)
    except Exception:  # a checker bug never blocks the script
        return ""
    return str(badge["text"]) if badge else ""


def _recap_panel(sb: StoryboardV2) -> PanelV2:
    recaps = [p for p in sb.panels if p.beat == "recap"]
    boards = [p for p in recaps if p.kind == "board"]
    return (boards or recaps or sb.panels)[-1]


GENERIC_LABEL_RE = re.compile(r"\b(?:prior|previous|earlier|old|best|top|sota|state[\s-]of[\s-]the[\s-]art|baseline|rival|"
                              r"published|reported|leaderboard|#\s*\d+|no\.\s*\d+)\b", re.I)
GENERIC_LABEL_WORDS = frozenset("prior previous earlier old best top sota state art baseline rival published reported leaderboard "
                                "single ensemble model models system systems result results ours".split())
EXACT_CLAIM_RE = re.compile(r"\b(?:exact|exactly|measured|reproduced|actual|real)\b", re.I)


def _label_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", str(text or "").lower()) if len(w) > 1 and w not in GENERIC_LABEL_WORDS
            and w not in STOPWORDS}


def _higher_is_better(spec: dict) -> bool:
    higher = str(spec.get("higher_is_better", True)).strip().lower() not in ("false", "0", "no")
    if spec.get("higher_is_better") is None and re.search(r"lower is better|越低越好|error|cost|time", " ".join(
            str(spec.get(k) or "") for k in ("title", "subtitle", "unit")), re.I):
        higher = False
    return higher


def _ledger_bar_issues(spec: dict, bars: list[dict], lit: list[dict], ledger) -> Optional[list[str]]:
    """Checks of a bar chart whose bars are bound to ledger cells: one column of one table, and the strongest earlier
    system of that column (services.ledger.strongest_rival) on the chart. None when the ledger cannot judge the chart
    (no bound highlight), so the plain rival check runs instead."""
    cells = [(b, c) for b in bars if b.get("cell") and (c := ledger.cell(b["cell"])) is not None]
    if not cells:
        return None
    shapes = {re.sub(r"[\d.,]+", "#", " ".join(re.sub(r"\(\d+\)|›", " ", c.column).split())) for _, c in cells}
    if len(cells) == len(bars) > 1 and len({(c.table, c.row) for _, c in cells}) == 1 and len(shapes) == 1 and "#" in next(iter(shapes)):
        # one row across its bins (call positions "1–3" ... "81–120", sizes, years): a trend, not a field of systems; numbered
        # regression columns ("(1) › Average conversion", "(2) › Profile slope") and benchmarks (GSM8K, MATH-500) are not bins
        return []
    columns = sorted({f'{c.table_label} column c{c.col} ("{c.column}")' for _, c in cells})
    if len(columns) > 1:
        return [f"evidence: the bars come from different table columns ({'; '.join(columns)}): one chart shows one column "
                "of one table (one metric, dataset and setting); keep the bars of one column on this chart, or make it a table board"]
    # a bar that cites nothing, with a value that is not in the cited column, comes from elsewhere (BERT's chart put the
    # leaderboard GLUE scores 72.8 and 80.5 next to Table 1's average 74.0: two different metrics)
    table, col = cells[0][1].table, cells[0][1].col
    held = {n.key for c in ledger.column(table, col) for n in _optional("boards").find_numbers(c.value)} if _optional("boards") else set()
    strays = [str(b.get("label") or b.get("value")) for b in bars if not b.get("cell")
              and not ({n.key for n in _optional("boards").find_numbers(str(b.get("value") or ""))} <= held)]
    if strays:
        return [f'evidence: bars {", ".join(strays)} are not in the column the other bars cite ({columns[0]}): a chart shows the '
                "values of one table column (cite them), or only values the text reports together for one comparison"]
    me = next((c for b, c in cells if b in lit), None) if len(lit) == 1 else None
    if me is None:
        return None
    context = " ".join(str(spec.get(k) or "") for k in ("title", "subtitle", "note"))
    rival = ledger.strongest_rival(me.id, _higher_is_better(spec), context)
    if rival is None or any(c.id == rival.id for _, c in cells):
        return []
    return [f'evidence: the strongest other system in this column ({rival.table_label}, "{rival.column}") is "{rival.label}" '
            f'with {rival.primary} (@{rival.id}), and it is not on the chart: add it ("value": "@{rival.id}") so the reader sees '
            'the whole field. Mark as "baseline" the system the claim of this chart compares with (the strongest earlier '
            'system when the claim is "better than before"); a system that is not a baseline stays a plain context bar']


def _story_board_issues(spec, ledger=None) -> list[str]:
    """Board rules the shape check does not know: who a bar is, what a highlight means, what a links board may claim,
    and (with the tables read from the PDF) one column per chart and its strongest earlier system on it."""
    if not isinstance(spec, dict):
        return []
    kind = str(spec.get("kind") or "").lower()
    problems: list[str] = []
    if kind in ("bar", "bars", "bar_chart", "leaderboard", "chart"):
        bars = [b for b in _as_list(spec.get("bars")) if isinstance(b, dict)]
        lit = [b for b in bars if str(b.get("highlight")).lower() in ("true", "1", "yes")]
        if len(lit) > 1:
            problems.append(f"{len(lit)} highlighted bars: highlight only the paper's headline result; show its other variants "
                            "as plain context bars (a chart compares one result with one baseline)")
        for b in bars:
            label, paper_label = str(b.get("label") or ""), str(b.get("paper_label") or "")
            if GENERIC_LABEL_RE.search(label) and not (_label_words(label) & _label_words(paper_label) if _label_words(paper_label)
                                                        else _label_words(label)):
                if paper_label and not _label_words(paper_label):   # the paper's row is only "(1) Baseline" (Gated Attention)
                    problems.append(f'bar "{label}" names no system, and the paper\'s row is only "{paper_label}": say in a few '
                                    'words what that system is, as the paper text describes it ("Standard attention (baseline)")')
                else:
                    problems.append(f'bar "{label}" names no system: label it with the system the paper reports'
                                    + (f' ("{paper_label}")' if paper_label else "") + '; "prior best" hides what the result is compared with')
        settings = _mixed_settings(bars)
        if settings:
            problems.append(f"the chart mixes {settings}: one setup per chart (the program draws the gap between the "
                            "highlighted bar and the baseline, so both must come from the same setup); keep one setup on this chart, or make it a "
                            "table with the setup as a column")
        base = [b for b in bars if str(b.get("baseline")).lower() in ("true", "1", "yes") and b not in lit]
        judged = _ledger_bar_issues(spec, bars, lit, ledger) if ledger else None
        if judged is not None:   # the ledger knows the whole column: its strongest system must be on the chart
            problems += judged
        elif len(lit) == 1 and base:
            stronger = _stronger_rivals(bars, lit[0], base[0], spec)
            if stronger:
                problems.append(f'the baseline "{base[0].get("label")}" is not the strongest earlier system on the chart '
                                f'({", ".join(stronger)} scores better): mark the strongest one the paper compares with as the baseline')
        boards_mod = _optional("boards")
        zero_ref = boards_mod is not None and any(str(b.get("baseline")).lower() in ("true", "1", "yes")
                                                  and boards_mod.parse_value(b.get("value")) == 0 for b in bars)
        if len(bars) == 2 and not zero_ref and any(str(b.get("baseline")).lower() in ("true", "1", "yes") for b in bars):
            problems.append("only 2 bars: show the field too: add 1-3 other systems from the same table column (same metric, "
                            "dataset and setting), each with its quote, when the paper lists them")
    if kind in ("bar", "bars", "bar_chart", "leaderboard", "chart", "table") and (boards_mod := _optional("boards")) is not None:
        # a lone number on a chart is no evidence: "Compared to what?" asked the skeptic of Driver Misconduct's one-row
        # table, and 1000 Layer RL drew 762 alone although its quote held the 4-layer 308 beside it
        if kind == "table":
            values = [c for r in _as_list(spec.get("rows")) if isinstance(r, dict) for c in _as_list(r.get("cells"))[1:]]
        else:
            values = [b.get("value") for b in _as_list(spec.get("bars")) if isinstance(b, dict)]
        numbers = [str(v) for v in values if boards_mod.find_numbers(str(v if v is not None else ""))]   # "56.9 (+39.0)" too
        if len(numbers) == 1:
            problems.append(f"the board shows one number ({numbers[0]}) and nothing to compare it with, so a reader cannot judge it: "
                            "add what the paper compares it with (the baseline or earlier system, the setting before the change, "
                            "another size), each with its quote, or the paper's related results in the same table; or say the "
                            "number in a bubble instead")
    if kind in ("links", "attention", "word_links"):
        said = " ".join(str(spec.get(k) or "") for k in ("title", "subtitle", "note"))
        if EXACT_CLAIM_RE.search(said):
            problems.append("links board claims exact or measured weights: its weights are an illustration; say \"illustration\" "
                            "or describe what the paper's figure shows in words")
    return problems


# setup words in bar labels: two different ones of a group on one chart are two setups (a semi-supervised result next to a
# WSJ-only one, a dev score next to a test score)
SETTING_GROUPS = {
    "training data": {"wsj": "wsj only", "semi": "semi-supervised", "semisupervised": "semi-supervised", "supervised": "supervised",
                      "limited": "limited data", "large": "large data"},
    "splits": {"dev": "dev", "val": "validation", "validation": "validation", "test": "test"},
}


def _mixed_settings(bars: list[dict]) -> str:
    out = []
    for group, words in SETTING_GROUPS.items():
        if group == "training data":   # "large" is a model size too: only a "large data" phrase counts
            found = set()
            for b in bars:
                label = re.sub(r"\bsemi[\s-]+supervised", "semisupervised", str(b.get("label") or "").lower())
                found |= {words[w] for w in re.findall(r"[a-z]+", label) if w in words and w != "large" and w != "limited"}
                found |= {"large data" for _ in re.findall(r"\blarge(?:r)?\s+(?:data|training)", label)}
                found |= {"limited data" for _ in re.findall(r"\blimited\s+(?:data|training)", label)}
        else:
            found = {words[w] for b in bars for w in re.findall(r"[a-z]+", str(b.get("label") or "").lower()) if w in words}
        if len(found) > 1:
            out.append(f"{group} ({', '.join(sorted(found))})")
    return " and ".join(out)


def _name_and_setting(text) -> tuple[frozenset, frozenset]:
    """A label's name words and setting words: a word that starts with a digit is a setting ("64k", "7b", "2/16"), so
    "Baseline 64k" and "Baseline 128k" are one system in two settings; "Qwen3" stays a name."""
    words = re.findall(r"[a-z0-9]+", str(text or "").lower())
    return frozenset(w for w in words if not w[0].isdigit()), frozenset(w for w in words if w[0].isdigit())


def _baseline_elsewhere(b: dict, baseline: dict, best: dict) -> bool:
    """b is the marked baseline's own system in another setting, while the highlighted bar shares the baseline's setting.
    Gated Attention's RULER at 128k: the board check asked for "Baseline 128k" (31.65) beside gated 128k, and the rival
    check then asked for "Baseline 64k" (37.51), which scores better only because 64k is the easier setting. b differs
    from the baseline only in that shared setting: a version it drops ("Llama 2 7B" -> "Llama 3 8B") makes it another
    system."""
    for key in ("label", "paper_label"):
        names, settings = _name_and_setting(b.get(key))
        base_names, base_settings = _name_and_setting(baseline.get(key))
        shared = _name_and_setting(best.get(key))[1] & base_settings
        if names and names == base_names and shared - settings and base_settings - shared <= settings:
            return True
    return False


def _stronger_rivals(bars: list[dict], best: dict, baseline: dict, spec: dict) -> list[str]:
    """Other systems (not variants of the highlighted one: no shared name word; not the baseline itself in another
    setting) that beat the marked baseline."""
    boards = _optional("boards")
    if boards is None:
        return []
    higher = _higher_is_better(spec)
    value = lambda b: boards.parse_value(b.get("value"))
    base_v = value(baseline)
    mine = ((_label_words(best.get("label", "")) | _label_words(best.get("paper_label", "")))
            - _label_words(baseline.get("label", "")) - _label_words(baseline.get("paper_label", "")))
    if base_v is None:
        return []
    out = []
    for b in bars:
        if b is best or b is baseline or str(b.get("highlight")).lower() in ("true", "1", "yes"):
            continue
        v = value(b)
        variant = bool(mine & (_label_words(b.get("label", "")) | _label_words(b.get("paper_label", ""))))
        upper = bool(UPPER_BOUND_RE.search(f'{b.get("label", "")} {b.get("paper_label", "")}'))   # a reference, not a rival
        if (v is not None and not variant and not upper and not _baseline_elsewhere(b, baseline, best)
                and (v > base_v if higher else v < base_v)):
            out.append(str(b.get("label") or b.get("value")))
    return out


GAP_RE = re.compile(r"(?:[+±]\s*|\bby\s+(?:over\s+|more than\s+|about\s+)?)?(\d+(?:\.\d+)?)\s*(?:-?\s*points?\b|pts?\b|%|percentage points?)",
                    re.I)


# a bin label holds a numeric range ("Calls 1-3", "第1-3通"); "7B", "2-bit" or "Top-5" are settings, not bins
BIN_RANGE_RE = re.compile(r"(?<![A-Za-z0-9.])\d+(?:\.\d+)?\s*[-–—~]\s*\d+(?:\.\d+)?(?![A-Za-z0-9.])")


def _words_vs_chart(p: PanelV2) -> list[str]:
    """A gap the words state ("7.7 points") must be the gap the chart draws, or one of its values: otherwise the reader
    sees two different comparisons in one panel (a leaderboard score in the bubble, a table average on the board)."""
    boards = _optional("boards")
    spec = p.board if isinstance(p.board, dict) else None
    if boards is None or not spec:
        return []
    try:
        norm = boards.normalize_board(spec)
        if not norm or norm.get("kind") != "bar":
            return []
        shown = {round(abs(v), 4) for v in (boards.parse_value(b["value"]) for b in norm["bars"]) if v is not None}   # "a 1.60% drop" is the -1.60 bar
        badge = boards.delta_badge(norm)
        gap = best = None
        if badge and badge.get("kind") == "diff":
            gap = boards.parse_value(badge["text"].replace("\u2212", "-"))
            best = boards.parse_value(norm["bars"][badge["best"]]["value"])
            rival = str(norm["bars"][badge["rival"]].get("label") or "")
            if gap is not None:
                shown.add(round(abs(gap), 4))
    except Exception:  # a checker bug never blocks the script
        return []
    # a number the board prints in its title or subtitle is shown too (a bar chart never draws its note): the reading of a
    # fraction that the unit rule asks for ("0.234 = +23.4 percentage points") made econ_ai_service's "+23 percentage
    # points" a mismatch forever
    for m in SAID_POINTS_RE.finditer(" ".join(str(spec.get(k) or "") for k in ("title", "subtitle"))):   # not a "2-bit" setting
        shown.add(round(abs(float(m.group(1).replace("−", "-"))), 4))
    words = " ".join([b.text for b in p.dialogue] + [p.narration])
    said = [m.group(1) for m in GAP_RE.finditer(words)]
    odd = [x for x in said if not _rounds_to(x, shown)]
    if not odd:
        # bars that are changes, and the words read the highlighted one's own change but never the gap the badge draws:
        # econ_ai_service's "+23.4 points" (the 0.234 bar) under a "+0.242 vs calls 81-120" badge, a difference of two
        # changes, left readers unsure which number was the result (a score with its gap to a rival reads fine)
        written = [m.group(1).replace("−", "-") for m in SAID_POINTS_RE.finditer(words)] + [w for w, _ in _number_tokens(words) if "." in w]
        reading = lambda v: {round(abs(v), 4), round(abs(v) * 100, 4)}
        values = [v for v in (boards.parse_value(b["value"]) for b in norm["bars"]) if v is not None]
        bins = len(norm["bars"]) >= 3 and all(BIN_RANGE_RE.search(str(b.get("label") or "")) for b in norm["bars"])   # "Calls 1-3", ...
        changes = bool(values) and (min(values) < 0 < max(values) or bins)   # changes around zero or bins of one quantity
        if (changes and gap is not None and best is not None and any(_rounds_to(w, reading(best)) for w in written)
                and not any(_rounds_to(w, reading(gap)) for w in written)):
            # the gap is the program's, never a number of the paper, so the words cannot say it: the evidence check and
            # the scrub would remove it
            return [f'the words read the highlighted bar\'s own value ({best:g}), but the badge draws its gap to "{rival}" '
                    f'({badge["text"]}), a number of no table: unmark the baseline so the chart draws no gap']
        return []
    logged = LOG_UNIT_RE.search(" ".join(str(spec.get(k) or "") for k in ("title", "subtitle", "unit")))   # -0.145 is ~13%, not 14.5 points
    top = next((round(abs(v), 4) for b in norm["bars"] if b.get("highlight") for v in [boards.parse_value(b["value"])] if v is not None), None)
    fractions = [] if logged else sorted((v for v in shown if 0 < v < 1 and any(_rounds_to(x, {round(v * 100, 4)}) for x in odd)),
                                         key=lambda v: v != top)
    return [f"the words say {', '.join(odd)} but the chart shows {', '.join(sorted(map(str, shown)))}: make the bubble and "
            "the chart show the same comparison (same table, same metric, same rival)"
            + (f'; when the chart\'s values are fractions, give the paper\'s own reading of one in the subtitle, as the paper '
               f'words it (e.g. "{fractions[0]:g} = {fractions[0] * 100:g} percentage points")' if fractions else "")]


def _rounds_to(written: str, values: set) -> bool:
    """A number as the words write it is one of ``values`` at the words' precision ("23" for 23.4, "5" for 4.5): the
    paper's abstract rounds ("about 23 percentage points") what its table prints."""
    try:
        v = abs(float(written))
    except ValueError:
        return False
    places = len(written.split(".")[1]) if "." in written else 0
    return any(abs(v - x) <= 0.5 * 10 ** -places + 1e-9 for x in values)


# the words claim a win over the whole field ("beat both baselines", "the best of all"): the badge must then measure the
# gap to the strongest other system on the chart (LongZero's badge said +2.74 vs the base model while the bubble said it
# beat both baselines; the strongest one was 0.85 behind)
BEAT_ALL_RE = re.compile(r"\b(?:beat|beats|beating|outperform\w*|surpass\w*|exceed\w*|better than|ahead of|tops?|wins? over)\b[^.!?]{0,40}?"
                         r"\b(?:baselines|all|every|both|others|other methods|other systems|rivals|competitors|the rest|the field|"
                         r"existing|prior|previous)\b|\bthe\s+(?:best|highest|lowest|strongest|top)\b(?![-‑])|\bfirst place\b", re.I)
COMPARE_TO_RE = r"\b(?:than|over|vs\.?|versus|beats?|beating|outperforms?|surpass(?:es)?|above|ahead of|against)\s+(?:the\s+)?"
# a reference, not a rival: an upper bound the paper's result is not meant to beat
UPPER_BOUND_RE = re.compile(r"\b(?:full[- ]precision|fp16|bf16|fp32|16[- ]?bit|teacher|oracle|upper[- ]bound|human|ceiling|"
                            r"reference)\b|全精度|教师|人类", re.I)
# numbers in parentheses beside a value ("0.234 (0.033)"): the paper's standard errors, which readers cannot place
PAREN_NUMBER_RE = re.compile(r"(?<![\w.-])[-−]?\d+(?:\.\d+)?[*∗†‡^]*\s*\(\s*(?!(?:1[89]|20)\d\d\s*\))[-−]?\d+(?:\.\d+)?\s*\)")
SIG_STARRED_RE = re.compile(r"(?<![\w.])[-−]?\d+(?:\.\d+)?\^?[*∗]{1,3}(?![*\w])")   # "0.234***", "0.005^***": a significance mark
# a legend says what the stars mean; "the chance a call is a hot lead" names the outcome, not the legend
STARS_EXPLAINED_RE = re.compile(r"p\s*[<≤]|[*∗]{1,3}\s*[:=]?\s*[<≤]\s*\d|significan|\b(?:by|to be|due to)\s+chance\b|显著", re.I)
# a legend of its own for a star that is a footnote mark ("* estimated from 2019 data"): nothing about chance to explain
STAR_LEGEND_RE = re.compile(r"(?:^|[\s;,(])[*∗]{1,3}\s*[:=]?\s*[A-Za-z\u4e00-\u9fff]")
PAREN_EXPLAINED_RE = re.compile(r"standard error|standard deviation|std\.?|s\.e\.|\bse\b|\bsds?\b|confidence interval|\bci\b|"
                                r"after each (?:score|value|number|estimate|result)|parenthes|in brackets|括号|标准误|标准差|置信区间", re.I)


def _badge_vs_words(p: PanelV2) -> list[str]:
    """The comparison the words make is the one the badge draws: a rival the words name is the marked baseline (unless the
    words also name the baseline and it is the stronger of the two), and a win over the field needs the gap to the strongest
    other system on the chart (not a variant of the paper's own, not an upper bound such as full precision)."""
    boards = _optional("boards")
    spec = p.board if isinstance(p.board, dict) else None
    if boards is None or not spec:
        return []
    try:
        norm = boards.normalize_board(spec)
        if not norm or norm.get("kind") != "bar":
            return []
        badge = boards.delta_badge(norm)
        bars = norm["bars"]
        values = [boards.parse_value(b.get("value")) for b in bars]
    except Exception:  # a checker bug never blocks the script
        return []
    if not badge:
        return []
    words = " ".join([b.text for b in p.dialogue] + [p.narration])
    best, rival = bars[badge["best"]], bars[badge["rival"]]
    rival_label, r = str(rival.get("label") or "").strip(), values[badge["rival"]]
    upper = lambda b: bool(UPPER_BOUND_RE.search(f'{b.get("label", "")} {b.get("paper_label", "")}'))
    # the words read out the paper's value and one other bar's, not the marked baseline's: Gated Attention said "At 64k
    # tokens: baseline 37.51, gated 66.60" beside a badge of -16.30 against "Baseline 4k" (82.90), in two rounds
    quoted = _number_keys(words)
    said = [i for i, b in enumerate(bars) if _number_keys(str(b.get("value") or "")) and _number_keys(str(b.get("value") or "")) <= quoted]
    read = [i for i in said if i not in (badge["best"], badge["rival"]) and not bars[i].get("highlight") and not upper(bars[i])]
    if badge["best"] in said and badge["rival"] not in said and len(read) == 1:
        label = str(bars[read[0]].get("label") or "").strip()
        return [f'the words compare {best.get("value")} with "{label}" ({bars[read[0]].get("value")}), but the badge measures '
                f'the gap to "{rival_label}" ({rival.get("value")}): mark "{label}" as the baseline so the badge shows the '
                f'comparison the words make, or make the words compare with "{rival_label}"']
    # a word the paper's bar shares with the marked rival is a setting ("2/16") or a backbone ("Qwen3-8B"), not its name
    mine = ((_label_words(best.get("label", "")) | _label_words(best.get("paper_label", "")))
            - _label_words(rival.get("label", "")) - _label_words(rival.get("paper_label", "")))
    higher = _higher_is_better(norm)
    rest, rival_named, named = words, False, None
    for i in sorted(range(len(bars)), key=lambda i: -len(str(bars[i].get("label") or "").strip())):   # "GRPO + KL" before "GRPO"
        label = str(bars[i].get("label") or "").strip()
        if len(label) < 3:
            continue
        hit = bool(re.search(COMPARE_TO_RE + re.escape(label) + r"\b", rest, re.I))
        rest = re.sub(re.escape(label), "#", rest, flags=re.I)   # a shorter label never matches inside this one
        if hit and i == badge["rival"]:
            rival_named = True
        elif hit and i != badge["best"] and named is None and not upper(bars[i]):
            named = i
    if named is not None:
        label = str(bars[named].get("label") or "").strip()
        weaker = None not in (values[named], r) and (values[named] <= r if higher else values[named] >= r)
        if not (weaker and rival_named):   # the badge already measures the stronger of the systems the words name
            return [f'the words compare with "{label}", but the badge measures the gap to "{rival_label}": ' + (
                f'make the words compare with "{rival_label}", the stronger system the badge measures against' if weaker else
                f'mark "{label}" as the baseline so the badge shows the comparison the words make, or make the words compare '
                f'with "{rival_label}"')]
    if rival_named:
        return []   # the words make the badge's own comparison
    claim = BEAT_ALL_RE.search(words)
    if not claim or r is None:
        return []
    others = [(values[i], str(b.get("label") or "")) for i, b in enumerate(bars)
              if i not in (badge["best"], badge["rival"]) and values[i] is not None and not b.get("highlight")
              and not (mine & (_label_words(b.get("label", "")) | _label_words(b.get("paper_label", "")))) and not upper(b)
              and not _baseline_elsewhere(b, rival, best)]   # as _stronger_rivals: the baseline at an easier setting is no rival
    stronger = [(v, label) for v, label in others if (v > r if higher else v < r)]
    if not stronger:
        return []
    v, label = (max if higher else min)(stronger)
    return [f'the words say "{claim.group(0).strip()}", but the badge measures the gap to "{rival_label}", not to the strongest '
            f'other system on the chart, "{label}" ({v}): mark "{label}" as the baseline so the badge shows the gap the words '
            "claim, or make the words say what the badge measures"]


def _parentheses_unexplained(p: PanelV2) -> list[str]:
    """A board with numbers in parentheses ("0.234 (0.033)") or significance stars ("0.234***") says what they are;
    readers asked every time."""
    spec = p.board if isinstance(p.board, dict) else None
    if not spec:
        return []
    shown = board_text(spec)
    said = " ".join([shown, p.narration] + [b.text for b in p.dialogue])
    out = []
    m = PAREN_NUMBER_RE.search(shown)
    if m and not PAREN_EXPLAINED_RE.search(said):
        out.append(f'the board shows numbers in parentheses ("{m.group(0)}") that nothing explains: say in the subtitle what they '
                   "are, in the words of the paper's table note (standard errors, standard deviations, a change from a baseline...), "
                   "or drop them")
    star = SIG_STARRED_RE.search(shown)
    legend = said if str(spec.get("kind") or "") != "bar" else " ".join(   # a bar chart never draws its note
        [str(spec.get(k) or "") for k in ("title", "subtitle")] + [p.narration] + [b.text for b in p.dialogue])
    drawn = " ".join(str(spec.get(k) or "") for k in (("title", "subtitle") if str(spec.get("kind") or "") == "bar" else
                                                      ("title", "subtitle", "note")))
    if star and not STARS_EXPLAINED_RE.search(legend) and not STAR_LEGEND_RE.search(drawn):
        out.append(f'the board shows significance stars ("{star.group(0)}") that nothing explains: add the legend of the paper\'s '
                   'table note to the board (a table\'s note, a bar chart\'s subtitle), in plain words ("*** p<0.01: very unlikely to be '
                   'chance"), or drop them')
    return out


POINTS_UNIT_RE = re.compile(r"percentage[\s-]+points?|\bpp\b|\bpercent\b|%|百分点", re.I)
LEAD_NUMBER_RE = re.compile(r"^\s*([-−+]?\d*\.\d+|[-−+]?\d+)(?![\d.])")
READING_RE = re.compile(r"[-−+]?\d*\.?\d+\s*(?:=|≈|~|is about|is|means)\s*[-−+]?\d+(?:\.\d+)?\s*(?:percentage points?|pp\b|%|percent|百分点)", re.I)
# a legend's own percents ("significant at the 1% level", "95% confidence interval") label no value
LEGEND_PCT_RE = re.compile(r"(?:[*∗]{1,3}\s*[:=]?\s*)?[<≤]\s*\d+(?:\.\d+)?\s*%|"   # "* <10%, *** <1%"
                           r"\d+(?:\.\d+)?\s*(?:%|percent\b)(?=[^;.|·]*(?:\b(?:levels?|confidence|ci)\b|水平|置信))", re.I)
CHANGE_LABEL_RE = re.compile(r"chang|effect|differen|gain|drop|increas|decreas|Δ|shift|rise|fall|impact|变化|差异|提高|下降|增加|减少|效应|影响", re.I)
LOG_UNIT_RE = re.compile(r"\blog\b|\bln\b|logarithm|elasticit|对数|弹性", re.I)
SAID_POINTS_RE = re.compile(r"([-−+]?\d+(?:\.\d+)?)\s*(?:percentage points?|-?\s*points?|pp\b|%|percent)", re.I)


def _fractions_as_points(p: PanelV2) -> list[str]:
    """A board whose values are fractions (a change in probability: 0.234) under a percentage-point or percent label,
    while the words read them x100 ("23 percentage points"): econ_ai_service's table said "percentage points" over 0.234,
    and its readers could not tell which was right."""
    found = _fraction_reading(p)
    if found is None:
        return []
    v, reading, n = found
    return [f'unit: the board labels its values as percentage points or percent, but they are fractions: {v:g} is '
            f'{reading}, as the words say ({n:g}). Label them in the unit of the paper\'s table (a change in probability for a '
            f'yes-or-no outcome, a share for a level, log points for a logged one) and give the paper\'s own reading of one '
            f'value once in the subtitle, as the paper words it (e.g. "{v:g} = {reading}")']


def _fraction_reading(p: PanelV2) -> Optional[tuple[float, str, float]]:
    """(the board's value, its reading x100, the number the words say) for _fractions_as_points, or None."""
    spec = p.board if isinstance(p.board, dict) else None
    if not spec or str(spec.get("kind") or "") not in ("bar", "table"):
        return None
    parts = [str(spec.get(k) or "") for k in ("title", "unit")] + [str(c) for c in _as_list(spec.get("columns"))]
    # a reading of a value ("0.234 = +23.4 percentage points") or a legend's percent labels nothing; a bar chart never draws
    # its note, so a label there is not seen
    for k in ("subtitle", "note") if str(spec.get("kind")) == "table" else ("subtitle",):
        text = LEGEND_PCT_RE.sub(" ", READING_RE.sub(" ", str(spec.get(k) or "")))
        parts += [re.sub(r"\d+(?:\.\d+)?\s*(?:%|percent\b)", " ", x) if re.search(r"[*∗]|significan|显著", x, re.I) else x
                  for x in re.split(r"[;|·]|\.\s", text)]   # "*** significant at 1%": a legend clause
    label = " ".join(parts)
    if not POINTS_UNIT_RE.search(label) or LOG_UNIT_RE.search(" ".join(str(spec.get(k) or "") for k in ("title", "subtitle", "unit"))):
        return None   # log points ("log duration": -0.145 is about -13%) and elasticities are no fractions of 100
    raw = [str(b.get("value") or "") for b in _as_list(spec.get("bars")) if isinstance(b, dict)]
    raw += [str(c) for r in _as_list(spec.get("rows")) if isinstance(r, dict) for c in _as_list(r.get("cells"))[1:]]
    if any("%" in v for v in raw):   # "0.5%" is a percent already
        return None
    values = [float(m.group(1).replace("−", "-")) for v in raw if (m := LEAD_NUMBER_RE.match(v))]
    if len(values) < 2 or any(abs(v) >= 1 for v in values):
        return None
    said = [m.group(1).replace("−", "-") for m in SAID_POINTS_RE.finditer(" ".join([p.narration] + [b.text for b in p.dialogue]))]
    if any(abs(float(n)) < 1 and _rounds_to(n, {round(abs(v), 4) for v in values}) for n in said):
        return None   # the words read a value as written ("loses only 0.3 points"): the values are points already
    hit = next(((v, float(n)) for n in said for v in values if abs(float(n)) >= 1 and _rounds_to(n, {round(abs(v) * 100, 4)})), None)
    if hit is None:
        return None
    v, n = hit
    change = any(x < 0 for x in values) or bool(CHANGE_LABEL_RE.search(label))   # a level or a share reads as a percent
    return v, (f"{v * 100:+g} percentage points" if change else f"{v * 100:g}%"), n


GAP_TAIL_RE = re.compile(r"^\s*(?:-?\s*points?|pts?|percentage points?|x|×|times)?\s*(?:higher|lower|more|less|fewer|better|worse|above|"
                         r"below|ahead|behind|up|down|gap|difference|faster|slower|cheaper|smaller|larger|bigger)\b", re.I)
# a model size ("Qwen3-8B", "a 4B model", "1.7 billion parameters") names a setting, not a value a chart would draw
SIZE_TOKEN_RE = re.compile(r"[\d.,]+\s*[BMKT]", re.I)
SIZE_TAIL_RE = re.compile(r"^\s*(?:billion|million|thousand|[BMKT]\b|-?param)", re.I)


def _numbers_off_board(p: PanelV2, claims: Optional[list[dict]] = None) -> list[str]:
    """A value the words of a chart or table panel state must be drawn on that board: a number from another table or
    metric next to the chart reads as a second exhibit nobody drew (Flexibility Trap's "0.6% on HumanEval" beside a
    chart of solved problems). Gaps ("more than 2.0 above", "+2.4 points", "14 times fewer") are _words_vs_chart's;
    years, model sizes and counts under 10 pass, and so do the key numbers of the claims this panel teaches, which the
    claim check asks for here ("41.8 BLEU after 3.5 days" taught on the BLEU chart)."""
    spec = p.board if isinstance(p.board, dict) else None
    if not spec or str(spec.get("kind") or "") not in ("bar", "table"):
        return []
    drawn = _number_keys(board_text(spec))
    drawn |= {key for c in claims or [] if c.get("id") in p.claims for _, key in _key_numbers(c)}
    boards = _optional("boards")
    if boards is not None:
        try:
            norm = boards.normalize_board(spec)
            badge = boards.delta_badge(norm) if norm and norm.get("kind") == "bar" else None
            if badge:
                drawn |= _number_keys(str(badge.get("text") or "").replace("−", "-"))
        except Exception:  # a checker bug never blocks the script
            pass
    odd: list[str] = []
    for text in [b.text for b in p.dialogue] + [p.narration]:
        for written, key in _number_tokens(text):
            if key in drawn or written in odd or YEAR_RE.fullmatch(written) or SIZE_TOKEN_RE.fullmatch(written):
                continue
            try:
                small = "." not in written and abs(Decimal(key)) < 10
            except InvalidOperation:   # "2.3 · 10^19": a value all the same
                small = False
            if small:
                continue
            at = [m for m in re.finditer(rf"(?<![\d.]){re.escape(written)}(?![\d])", text)]
            if at and all(GAP_LEAD_RE.search(text[:m.start()]) or GAP_TAIL_RE.match(text[m.end():].lstrip("%"))
                          or SIZE_TAIL_RE.match(text[m.end():]) for m in at):
                continue
            odd.append(written)
    if not odd:
        return []
    return [f"the words say {', '.join(odd)}, which this {spec.get('kind')} does not show: the reader looks for it on the board "
            "and cannot find it. Draw that value on the board (a bar or row with the paper's number, or the subtitle for a "
            "setting), or say it on a panel whose board shows it, so each panel's words are about its own board"]


# a part's label ("Answer: Sofia Bianchi", "teacher token: \"Wait\"") names its role, not the case: "Who answers the
# question?" passed the example test on the labels' words alone (round 25's review); "14:00", "2:1" and "Third Month:" stay
EXAMPLE_LABEL_RE = re.compile(r"^[^:\"“]{1,30}:\s+(?=\S)")


def _example_tokens(sb: StoryboardV2) -> set[str]:
    """Content words of the running example (its quoted part when the description quotes it) and of its parts, without
    the parts' labels."""
    text = sb.running_example or ""
    quoted = re.findall(r"[\"“]([^\"”]{8,})[\"”]", text)
    parts = [EXAMPLE_LABEL_RE.sub("", x) for x in _example_parts(sb)]
    words = _content_tokens(" ".join((quoted if quoted else [text]) + parts))
    return {_stem(w) for w in words if not w[0].isdigit()} - {"example", "sentence", "training", "input", "mask"}


def _example_beats(sb: StoryboardV2) -> dict[str, bool]:
    """{beat: a panel of it shows the running example} for the hook, the how and the payoff (lint_v3's example test)."""
    example = _example_tokens(sb)
    need = min(3, max(1, math.ceil(0.34 * len(example))))
    return {beat: any(len(example & {_stem(w) for w in _content_tokens(visible_text(p))}) >= need
                      for p in sb.panels if p.beat == beat) for beat in ("hook", "how", "payoff")}


def _stem(word: str) -> str:
    """Crude singular: "balls" and "ball" (and "roger's" and "roger") count as the same word in the example check."""
    return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word


def _teaching_panels(panels: list[PanelV2]) -> list[PanelV2]:
    """The panels that teach (not the claims board or the recap, which only list the claims)."""
    return [p for p in panels if p.beat not in LIST_BEATS] or list(panels)


def _home_panel(sb: StoryboardV2, claim: dict) -> PanelV2:
    """The teaching panel that should carry a claim no teaching panel conveys yet: the one whose words share the most with
    the claim; ties go to evidence panels when the claim has numbers, else to the idea and how panels; then the later."""
    teaching = _teaching_panels(sb.panels)
    want = _content_tokens(str(claim.get("text", "")) + " " + str(claim.get("quote", "")))
    preferred = ("proof",) if _key_numbers(claim) else ("idea", "how", "problem")
    def score(item):
        i, p = item
        return (len(want & _content_tokens(visible_text(p))), p.beat in preferred, p.kind == "board" and bool(_key_numbers(claim)), i)
    return max(enumerate(teaching), key=score)[1]


def _term_forms(term: str) -> list[str]:
    """A glossary term's names: "Quantization-aware training (QAT)" -> ["Quantization-aware training", "QAT"]; "AIME (2024)"
    and "polylog(1/δ)" stay one name."""
    t = str(term).strip()
    m = re.match(r"^(.*?\S)\s+\(([^)]+)\)\s*$", t)
    forms = ([m.group(1)] if not re.search(r"[^\W\d_]", m.group(2)) else [m.group(1), m.group(2)]) if m else [t]
    return [f.strip() for f in forms if len(f.strip()) >= 2]


def _term_re(form: str) -> re.Pattern:
    """An abbreviation matches as written (plural allowed); any other term in any case."""
    exact = bool(ACRONYM_RE.fullmatch(form))
    return re.compile(r"(?<![\w-])" + re.escape(form) + (r"s?" if exact else r"(?:s|es)?") + r"(?![\w-])", 0 if exact else re.I)


def _initials(words: list[str]) -> str:
    """First letters, lower-cased; a word that is itself an abbreviation gives its capitals ("forward KL" -> "fkl")."""
    return "".join("".join(c for c in w if c.isupper()).lower() if sum(c.isupper() for c in w) >= 2 else w[0].lower() for w in words)


def _named_before(acronym: str, before: str) -> bool:
    """"instrumental variable (IV)", "forward KL (FKL)", "Mixture of Experts (MoE)": the words right before the bracket
    start with the abbreviation's capitals."""
    want = "".join(c for c in acronym if c.isupper()).lower()
    words = re.findall(r"[A-Za-z]+", before)[-(len(want) + 4):]
    return len(want) >= 2 and any(_initials(seq).endswith(want) for seq in (words, [w for w in words if w.lower() not in SMALL_WORDS]) if seq)


def _named_after(acronym: str, after: str) -> bool:
    """"KL, Kullback-Leibler divergence, ...", "LR = learning rate": the words right after the abbreviation and its comma,
    colon, dash or "stands for" start with its capitals."""
    want = "".join(c for c in acronym if c.isupper()).lower()
    m = re.match(r"\s*(?:,|:|=|—|–|\s-\s|\s+(?:stands? for|is short for))\s*", after)
    if len(want) < 2 or not m:
        return False
    words = re.findall(r"[A-Za-z]+", after[m.end():])[:len(want) + 3]
    return any(_initials(seq).startswith(want) for seq in (words, [w for w in words if w.lower() not in SMALL_WORDS]) if seq)


def _spelled_out(acronym: str, text: str) -> bool:
    """The words an abbreviation stands for are in one sentence of the text ("large language model" for LLM); only for 3
    capitals or more, since two initials ("is very" for IV) match by chance, and never from abbreviations."""
    want = "".join(c for c in acronym if c.isupper()).lower()
    if len(want) < 3:
        return False
    for sentence in re.split(r"[.!?;:\n]|\s{2,}", text):
        words = [w for w in re.findall(r"[A-Za-z]+", sentence) if sum(c.isupper() for c in w) < 2]
        if any(want in "".join(w[0].lower() for w in seq) for seq in (words, [w for w in words if w.lower() not in SMALL_WORDS])):
            return True
    return False


def _term_explained(forms: list[str], text: str, plain: str = "") -> bool:
    """The text explains the term: an explanation next to one of its names, the full name before the abbreviation's
    bracket, the full name beside the abbreviation, the abbreviation spelled out, or two telling words of the glossary's
    plain meaning (not the term's own, not "higher is better")."""
    for form in forms:
        acronym = bool(ACRONYM_RE.fullmatch(form))
        for m in _term_re(form).finditer(text):
            before, after = text[:m.start()], text[m.end():]
            if TERM_DEF_AFTER_RE.match(after) or TERM_DEF_BEFORE_RE.search(before):
                return True
            if acronym and re.search(r"\(\s*$", before) and re.match(r"\s*\)", after) and _named_before(form, before.rstrip()[:-1]):
                return True
            if acronym and _named_after(form, after):
                return True
    if len(forms) > 1 and all(_term_re(f).search(text) for f in forms):
        return True
    if any(ACRONYM_RE.fullmatch(f) and _spelled_out(f, text) for f in forms):
        return True
    want = _content_tokens(PLAIN_DIRECTION_RE.sub(" ", plain)) - PLAIN_GENERIC - set().union(*(_content_tokens(f) for f in forms))
    return bool(want) and len(want & _content_tokens(text)) >= min(2, len(want))


def _words_text(p: PanelV2) -> str:
    """What a panel says in words: bubbles, narration and the board's titles, notes, steps and items (a compare board's
    sides too); not the bar labels or table cells, which name the systems compared."""
    parts = [p.narration] + [b.text for b in p.dialogue]
    b = p.board if isinstance(p.board, dict) else {}
    parts += [str(b.get(k) or "") for k in ("title", "subtitle", "note")]
    parts += [str(s) for s in _as_list(b.get("steps")) + _as_list(b.get("items")) if isinstance(s, (str, int, float))]
    for side in _as_list(b.get("left")) + _as_list(b.get("right")):
        if isinstance(side, dict):
            parts += [str(side.get("label") or "")] + [str(x) for x in _as_list(side.get("items"))]
    return "  ".join(x for x in parts if x)


SYMBOL_TEXTY_RE = re.compile(r"\b(?:scribbl\w*|writ\w*|print\w*|texts?|letters?|numbers?|numerals?|digits?|labels?|words?|"
                             r"formulas?|equations?|signs?)\b", re.I)


def drawable_symbols(symbols) -> list[dict]:
    """The brief's picture symbols an illustrator can draw: none that carries writing (a pre-printed scroll, a scribbled
    line), a person or figure, a chart or a chat box. Right On Q's "pre-printed paper scroll" was demanded by the picture
    rule and banned by the writing-prop rule in every fix round."""
    out = []
    for item in _as_list(symbols):
        if not isinstance(item, dict) or not _text(item.get("symbol")):
            continue
        text = f'{_text(item.get("symbol"))}'
        if (SYMBOL_TEXTY_RE.search(text) or _mentions_people(text)
                or any(rx.search(text) for rx in (WRITING_PROP_RE, FIGURE_PROP_RE, CHAT_BOX_RE, SCENE_CHART_RE))):
            continue
        out.append({"concept": _text(item.get("concept")), "symbol": _text(item.get("symbol"))})
    return out[:8]


def _picture_symbols(sb: StoryboardV2) -> list[tuple[str, str]]:
    """The brief's (concept, symbol) pairs: the objects that stand for the paper's key concepts in every picture."""
    return [(item["concept"], item["symbol"]) for item in drawable_symbols(sb.brief.get("picture_symbols"))]


# "This smooth river stone is training reward": a symbol named as a key ("this/these <object> is/are/means ...")
THIS_SYMBOL_RE = re.compile(r"\bth(?:is|ese)\s+((?:[\w'-]+\s+){0,4}?[\w'-]+)\s+(?:is|are|means|stands for|shows)\b", re.I)
# words of a symbol's description that say where, not what: a picture with a left and a right pile draws no dot line
POSITION_WORDS = frozenset("left right start end top bottom middle center centre front back side".split())
# the arrangements the prompts recommend, read before a symbol's own words are blanked ("Small empty cup" is no reason
# to miss "one card among rows of empty shelves")
ARRANGEMENT_RE = re.compile(r"\bempty\s+(?:shelves|shelf|slots?|outlines?)\b|\bdashed\s+(?:empty\s+)?outlines?\b", re.I)
SETTLES_BET_RE = re.compile(r"\b(?:bets?|wager(?:ed)?|pays? up|paid up)\b", re.I)
# words a "shows" shares with any visual that say nothing about the objects: the fit's scope line ("and nobody else is
# there") and order words let a claim pass as a description of the drawing
SHOWS_FILLER = frozenset("nobody else there present alone first then after before next".split())


# a picture that teaches stages a change the reader can see: a tiny thing beside a huge one, an arrow from cause to effect,
# before and after, an object poured, moved, sorted, tied or broken. Of the teaching scenes of 2026-10-28 (round 3b) that
# only had the cast hold, tap or point at the symbols, 1 in 16 explained its idea even partly; of the others, 16 in 31.
ACTION_RE = re.compile(
    r"^(?=[\s\S]*\b(?:tiny|small|little|single|thin|short)\b)(?=[\s\S]*\b(?:huge|tall|giant|towering|massive|enormous|thick)\b)|"
    r"→|->|\barrows?\b|\bbefore\b|\bafter\b|\bthen\b|\binto\b|\bout of\b|\bonto\b|\btoward|\baway from\b|"
    r"\bleft\b.{0,80}\bright\b|\b(?:taller|shorter|bigger|smaller|larger|longer|heavier|lighter|wider|narrower|higher|lower)\b|"
    r"\btowers? over\b|\bdwarf|\b(?:stacks?|piles?|builds?|lines?)\s+(?:up|into|onto)\b|"
    r"\bempty\s+(?:shelves|shelf|slots?|spaces?|outlines?|seats?|chairs?|rows?)\b|\bdashed\s+(?:empty\s+)?outlines?\b|"
    r"\bfl(?:y|ies|ew|ying)\s+(?:to|into|toward|off|away|across|back)\b|\bdifferent\s+heights?\b|"
    r"\bdrift\w*|\bdiverg\w*|\bcurv\w*\s+(?:further\s+)?away\b|"
    r"\b(?:pour|spill|flow|grow|shrink|break|broke|snap|ties|tied|tying|connect|sorts|sorted|sorting|fills|filled|filling|empt|tips|"
    r"tipped|tipping|tilt|push|pull|drops|dropped|dropping|falls|fell|falling|moves|moved|moving|slides|slid|feeds|feeding|rolls|"
    r"rolling|melt|leak|overflow|splits|splitting|merg|swap|replac|cuts|cutting|bounc|knock|crumbl|collaps|burst|fades|fading)\w*", re.I)
# a scene that points at a board or its numbers: the art prompt of a scene forbids boards, so the picture shows none, or a
# blank one (a board the fix rounds could not save became a scene, and "Kumo nods at the board" stayed: 19 panels in 30 books)
_NOT_DRAWN = r"(?:(?:white|black)?boards?|text|words|writing|letters|numbers?|charts?|screens?|signs?)"
NO_BOARD_RE = re.compile(rf"\b(?:no|without)\s+{_NOT_DRAWN}(?:\s*(?:,|or|and|nor)\s*{_NOT_DRAWN})*", re.I)
SCENE_BOARD_RE = re.compile(r"\b(?:white|black)?boards?\b(?!\s+games?)|\bhighlighted bars?\b|\b(?:at|to)\s+the\s+(?:numbers?|charts?|graphs?)\b",
                            re.I)


def _settles_bet(p: PanelV2) -> bool:
    """The limits (or payoff) scene where the skeptic pays up: it carries the story, so the rules let it leave "shows"
    empty (every book of 2026-10-28 gave the bet a scene of its own, and the picture rule forced an idea onto it)."""
    return (p.kind == "scene" and p.beat in ("limits", "payoff")
            and bool(SETTLES_BET_RE.search(" ".join([p.title, p.narration] + [b.text for b in p.dialogue]))))


def _same_word(x: str, y: str) -> bool:
    """Two forms of one word: a common start of all but the last two letters, at least three ("tied" ~ "ties", "leaks" ~
    "leaking", "stacked" ~ "stack")."""
    k = 0
    while k < min(len(x), len(y)) and x[k] == y[k]:
        k += 1
    return k >= max(3, min(len(x), len(y)) - 2)


def _shared_words(a: set, b: set) -> int:
    """How many words of ``a`` appear in ``b`` in some form."""
    return sum(1 for x in a if any(_same_word(x, y) for y in b))


def _picture_problems(sb: StoryboardV2) -> dict[int, list[str]]:
    """{n: ["picture: ..."]}: a scene the illustrator cannot draw as planned: it points at a board (a scene panel has none,
    so it is drawn blank), its words name a picture symbol it does not draw (books made with symbols), or a "shows" asks
    for people, writing or a chart."""
    symbols = _picture_symbols(sb)
    out: dict[int, list[str]] = {}
    # the book's own board (a chess or circuit board symbol) is an object the illustrator draws, not the exhibit board
    own_board = any(re.search(r"\bboards?\b", sym, re.I) for _, sym in symbols)
    for p in sb.panels:
        setting = NO_BOARD_RE.sub(" ", f"{p.scene} {p.visual}")   # "No boards, text or numbers appear anywhere"
        if own_board:
            setting = re.sub(r"\bboards?\b", " ", setting)
        if p.kind == "scene" and SCENE_BOARD_RE.search(re.sub(r"\bon board\b|\bboards?\b(?=\s+(?:the|a|an|her|his|their)\b)", " ", setting)):
            out.setdefault(p.n, []).append(
                "picture: the scene points at a board or its numbers, but a scene panel has no board, so it is drawn blank: "
                "take the board out of \"scene\" and \"visual\" (\"Study board\" becomes \"Study room\") and stage the panel's "
                "moment with the real things of the story or the cast's reaction, or make it a board panel")
        # the words name a symbol the picture does not draw: "This smooth river stone is training reward" over a desk of
        # paper, string and a needle (LongZero), and the reader looked for the stone
        if p.kind == "scene":
            visual = {_stem(w) for w in _content_tokens(f"{p.scene} {p.visual}")}
            owns = [(sym, {_stem(w) for w in _content_tokens(sym)} - SYMBOL_GENERIC - POSITION_WORDS) for _, sym in symbols]

            def drawn_sym(sym: str, own: set, named: set) -> bool:   # the words the sentence used, and what tells it apart
                mark = (own - set().union(*(o for s2, o in owns if s2 != sym))) or own
                return (_key_words(own, visual) >= min(2, len(own)) and bool(_key_words(mark, visual))
                        and (not (named & own) or bool(_key_words(named & own, visual))))
            for m in THIS_SYMBOL_RE.finditer(" ".join([p.narration] + [b.text for b in p.dialogue])):
                named = {_stem(w) for w in _content_tokens(m.group(1))} - SYMBOL_GENERIC - POSITION_WORDS
                hits = [(_key_words(named, own), -len(own), sym, own) for sym, own in owns if own and _key_words(named, own) >= min(2, len(own))]
                if not hits:
                    continue
                top = max(h[:2] for h in hits)
                tied = [h for h in hits if h[:2] == top]   # "this playing card" names the red and the grey card alike
                if not any(drawn_sym(sym, own, named) for _, _, sym, own in tied):
                    sym = tied[0][2]
                    out.setdefault(p.n, []).append(
                        f'picture: the words name "{sym}" ("{m.group(0).strip()}"), but the visual does not draw it: put it '
                        "in the picture, large and plainly visible")
        # round 23: a scene tells the story literally and the boards explain it, so a scene no longer has to teach with its
        # picture. That rule asked for metaphor pictures, which needed a legend line each ("this jar is the solution
        # space"), still failed in half the audits, and lost to code-drawn diagrams of the same words in 22 of 24 panels.
        # A "shows" left over is still checked for what no illustrator can draw.
        if p.kind != "scene" or not p.shows.strip():
            continue
        if (_mentions_people(p.shows) or ACRONYM_RE.search(p.shows)
                or any(rx.search(p.shows) for rx in (FIGURE_PROP_RE, WRITING_PROP_RE, CHAT_BOX_RE, SCENE_CHART_RE))):
            out.setdefault(p.n, []).append(
                "picture: \"shows\" asks the illustrator for something it cannot draw (people, figures, writing, a chart, a chat "
                "box or an abbreviation): set \"shows\" to \"\" and let \"visual\" draw the real situation (a person of the "
                "paper is played by a cast member)")
    return out


def _board_values(board) -> frozenset:
    """The numbers a bar chart or table draws (its values and cells, not its labels or title)."""
    if not isinstance(board, dict):
        return frozenset()
    kind = str(board.get("kind") or "")
    if kind == "bar":
        cells = [str(b.get("value") or "") for b in _as_list(board.get("bars")) if isinstance(b, dict)]
    elif kind == "table":
        cells = [str(c) for r in _as_list(board.get("rows")) if isinstance(r, dict) for c in _as_list(r.get("cells"))[1:]]
    else:
        return frozenset()
    return frozenset(_number_keys(" ".join(cells)))


# a stand-in for what the abstract says, on the board readers take as the summary: "needs good data" for "long, diverse,
# challenging, verifiable data" (every LongZero reader answered "good data"), "two core flaws" for the two the quiz asks
VAGUE_LINE_RE = re.compile(r"\b(?:good|better|bad|right|proper|quality|high-quality)\s+(?:training\s+)?(?:data|results?|performance|outputs?|answers?)\b|"
                           r"\b(?:two|three|four|several|some|many|key|core|main)\s+(?:core\s+|key\s+|main\s+|big\s+)?(?:flaws|issues|problems|"
                           r"challenges|barriers|reasons|factors|limitations|weaknesses)\b|\bexternal help\b|\boutside help\b", re.I)


def _vague_list_lines(sb: StoryboardV2) -> dict[int, list[str]]:
    """{n: [...]}: a line of the claims board or the recap that names a claim by a stand-in ("needs good data", "two core
    flaws cause this gap") instead of saying it."""
    if is_novice(sb):
        out = {}
        for p in sb.panels:
            if p.beat == "claims" and isinstance(p.board, dict) and p.board.get("kind") == "list":
                items = [str(x).strip() for x in _as_list(p.board.get("items")) if str(x).strip()]
                if not 2 <= len(items) <= 3 or any(not x.endswith(("?", "？")) for x in items):
                    out[p.n] = ["opening questions: use 2-3 plain reader questions, not complete claims or result numbers"]
        return out
    out: dict[int, list[str]] = {}
    for p in sb.panels:
        if p.beat not in ("claims", "recap") or not isinstance(p.board, dict) or p.board.get("kind") != "list":
            continue
        for item in _as_list(p.board.get("items")):
            m = next((m for m in VAGUE_LINE_RE.finditer(str(item)) if not re.match(r"\s*[:—–-]\s*\S", str(item)[m.end():])), None)
            if m:
                out.setdefault(p.n, []).append(
                    f'the line "{item}" says "{m.group(0)}" where the abstract says what it is: readers take this board as the '
                    "summary, so name it (the qualities, the flaws, the resources), as the abstract does, within the line's 14 words")
    return out


# a line about what the paper does not show, outside the limits: Flexibility Trap's payoff said "The paper reports
# aggregate benchmark gains, not a specific app-download example" (rounds 13 and 14), and readers asked which example
META_LINE_RE = re.compile(r"\b(?:(?:the|this) (?:paper|study)|the authors)\s+(?:does not|doesn['’]t|don['’]t|did not|didn['’]t|never)\s+"
                          r"(?:show|report|solve)\w*"
                          r"|\bnot (?:a|any) (?:specific|exact|particular) (?:[\w-]+ )?example\b"
                          r"|\bnot (?:this|that|our) (?:specific |exact |particular |same )?(?:[\w-]+ )?example\b|\bnot that this example\b", re.I)
# "only shows", "shows no", "gives no", "does not test" also state results ("The paper shows no accuracy drop at 4 bits"): a
# disclaimer only in the payoff or beside the example
META_SOFT_RE = re.compile(r"\bthe paper only shows?\b|\bthe paper (?:shows|gives|has) no\b"
                          r"|\b(?:the|this) (?:paper|study)\s+(?:does not|doesn['’]t|did not|didn['’]t|never)\s+test\w*", re.I)


def _meta_lines(sb: StoryboardV2) -> dict[int, list[str]]:
    """{n: ["meta: ..."]}: a story line that tells the reader what the paper does not show (the fact-check's disclaimers
    were most of them). The limits chapter may say what the authors did not test, and the hook says once that a made-up
    example is an illustration."""
    if is_novice(sb):
        return {}  # Brief source boundaries are allowed; factual/evidence lint still applies.
    out: dict[int, list[str]] = {}
    for p in sb.panels:
        if p.beat in ("hook", "limits", "claims", "recap"):
            continue
        for text in [p.narration] + [b.text for b in p.dialogue]:
            soft = p.beat == "payoff" or re.search(r"\bexample\b", text or "", re.I)
            if m := META_LINE_RE.search(text or "") or (META_SOFT_RE.search(text or "") if soft else None):
                out.setdefault(p.n, []).append(
                    f'meta: "{text[:70]}" talks about what the paper does not show ("{m.group(0)}"): say what it does show for '
                    "this story instead (its result for problems like the running example, or what its figure shows)")
    return out


def _repeats(sb: StoryboardV2) -> dict[int, list[str]]:
    """{n: [...]}: a panel that repeats an earlier one word for word: its picture (the visual) or one of its lines. A
    split's second half and a quiz revision restated the panel before (the delivery book drew one scooter scene twice and
    said "This hurts efficiency, parcels get sent back" on two panels), and readers took it for a misprint."""
    out: dict[int, list[str]] = {}
    pictures: dict[str, int] = {}
    lines: dict[str, int] = {}
    drawn: dict[frozenset, int] = {}
    recent: list[tuple[int, set, set]] = []   # the words (and numbers) of the last panels: "the bar to keep this lead rises" three times over
    run: list[tuple[int, set]] = []            # the panels in a row that teach only run_claim
    run_claim = None
    for p in sb.panels:
        values = _board_values(p.board) if p.kind == "board" and p.beat not in ("claims", "recap") else frozenset()
        if p.kind == "board" and p.beat not in ("claims", "recap") and isinstance(p.board, dict) and not values:
            title = " ".join(sorted(_stem(w) for w in _content_tokens(str(p.board.get("title") or ""))))
            values = frozenset({f"kind:{p.board.get('kind')}", f"title:{title}"}) if title else frozenset()   # a list, flow or compare twice
        subset = next((n for key, n in drawn.items() if len(values) >= 3 and values < key and not any(str(x).startswith("kind:") for x in key)), None)
        if len(values) >= 2 and (drawn.setdefault(values, p.n) != p.n or subset is not None):
            drawn.setdefault(values, subset)
            # the delivery book drew Table 3 twice ("Now let's look at the data") and one bar chart twice with the highlight
            # moved; two charts of one metric for two models share a title, not their numbers
            out.setdefault(p.n, []).append(
                f"the board repeats panel {drawn[values] if drawn[values] != p.n else subset}'s board: show what this panel adds "
                "(another column, setting or result of the paper), or let this panel react to the board before as a scene")
        key = " ".join(_content_tokens(p.visual))
        if p.kind == "scene" and key:
            if key in pictures:
                out.setdefault(p.n, []).append(
                    f"the picture repeats panel {pictures[key]}'s visual word for word: draw what is new here (the next step, "
                    "its result, or the other side of the comparison)")
            pictures.setdefault(key, p.n)
        repeated = False
        for text in [b.text for b in p.dialogue] + [p.narration]:
            key = " ".join(_content_tokens(text))
            if len(key.split()) >= 3 and lines.setdefault(key, p.n) != p.n:
                repeated = True
                out.setdefault(p.n, []).append(
                    f'"{text[:60]}" repeats panel {lines[key]} word for word: say what this panel adds (the next part of the '
                    "claim, its reason or its number), or drop the line")
        if p.beat in ("claims", "recap"):
            continue
        words = " ".join([p.narration] + [b.text for b in p.dialogue])
        said = {_stem(w) for w in _content_tokens(words) if not w[0].isdigit()}
        figures = _number_keys(words)
        again = next((n for n, earlier, _ in reversed(recent[-3:]) if len(said) >= 6 and len(said & earlier) >= 0.7 * len(said)), None)
        last = recent[-2:]
        # a short line with one new word: "This is the attention sink pathology." after two panels of it; a new number or a
        # board is something new
        if again is None and p.kind == "scene" and p.beat in ("problem", "idea", "how") and 3 <= len(said) < 6 and last \
                and len(said - set().union(*(earlier for _, earlier, _ in last))) <= 1 \
                and not figures - set().union(*(numbers for _, _, numbers in last)):
            again = max(reversed(last), key=lambda r: len(said & r[1]))[0]
        if again is not None and not repeated:
            out.setdefault(p.n, []).append(
                f"this panel's words say again what panel {again} said (most of its words are there): say what this panel adds "
                "(the next part of the claim, its reason, its number or its result)")
        elif not repeated and p.beat in ("problem", "idea", "how") and len(p.claims) == 1 and p.claims[0] == run_claim \
                and len(run) >= 2 and len(said) >= 5:
            # the third panel or later of a run on one claim, saying much of an earlier one in new words: Right On Q's four
            # idea panels all said "Dynamic KL weights each token by the teacher-student ratio" while forward and reverse
            # KL never appeared (round 14); the words-again rule above needs 70% and missed them at 64-69%
            twin = next((n for n, earlier in run if len(earlier) >= 5 and len(said & earlier) >= 0.55 * min(len(said), len(earlier))), None)
            if twin is not None:
                out.setdefault(p.n, []).append(
                    f"this panel says much of what panel {twin} said, and both teach only claim {run_claim}: make it add the next "
                    "part (a step of the method, its reason, or a detail of the claim no panel says yet, from the PAPER TEXT), "
                    "not the same point in new words")
        if len(p.claims) == 1 and p.claims[0] == run_claim:
            run.append((p.n, said))
        else:
            run, run_claim = ([(p.n, said)], p.claims[0]) if len(p.claims) == 1 else ([], None)
        recent.append((p.n, said, figures))
    return out


def _uses_symbol(words: set, wanted: set) -> bool:
    return bool(wanted) and len(wanted & words) >= min(2, len(wanted))


def _noun(text: str) -> str:
    """"A small locked wooden box" -> "small locked wooden box"; acronyms keep their case ("AI agent")."""
    text = re.sub(r"^(?:a|an|the)\s+", "", str(text).strip(), flags=re.I)
    first = text.split(" ", 1)[0]
    return text if not text or first.isupper() else text[0].lower() + text[1:]


SYMBOL_KEYS_PER_PANEL = 2


def _key_words(a: set, b: set) -> int:
    """Words of ``a`` said in ``b`` in a near form: the same word, or a common start of at least four letters that leaves
    at most three of the longer word ("reallocating" ~ "reallocation"; "thin" is not "thick", "paper" is not "papi")."""
    def near(x: str, y: str) -> bool:
        k = 0
        while k < min(len(x), len(y)) and x[k] == y[k]:
            k += 1
        return x == y or (k >= 4 and k >= max(len(x), len(y)) - 3) or (k >= 3 and k >= max(len(x), len(y)) - 1)
    return sum(1 for x in a if any(near(x, y) for y in b))


def drawn_symbols(sb: StoryboardV2, text: str) -> list[str]:
    """The book's picture symbols a picture text draws, read as _symbol_keys reads a visual: most of the symbol's words,
    one of them a word no other symbol has ("glowing red playing card" is not drawn by "dull grey playing card")."""
    words = lambda t: {_stem(w) for w in _content_tokens(t)} - SYMBOL_GENERIC - POSITION_WORDS
    pairs = [(symbol, words(symbol)) for _, symbol in _picture_symbols(sb)]
    drawn = words(text)
    out = []
    for i, (symbol, wanted) in enumerate(pairs):
        own = (wanted - set().union(*(w for j, (_, w) in enumerate(pairs) if j != i))) or wanted
        if _uses_symbol(drawn, wanted) and drawn & own:
            out.append(symbol)
    return out


def _symbol_keys(sb: StoryboardV2) -> dict[int, list[str]]:
    """{n: ["symbol: ..."]}: the first scene that draws a picture symbol says what it stands for (readers of the first
    picture-symbol books could not tell what a red card or a leaking bucket meant). Symbols that share words ("glowing red
    playing card", "dull grey playing card") are told apart by the words only one of them has."""
    out: dict[int, list[str]] = {}

    def words(text: str) -> set:   # "left = start, right = end" in a dot line's description draws no dot line
        return {_stem(w) for w in _content_tokens(text)} - SYMBOL_GENERIC - POSITION_WORDS

    cast = {_stem(c.lower()) for q in sb.panels for c in list(q.characters) + [b.speaker for b in q.dialogue] if c}
    pairs = [(concept, symbol, words(symbol), words(concept)) for concept, symbol in _picture_symbols(sb)]
    for i, (concept, symbol, wanted, meaning) in enumerate(pairs):
        others = [q for j, q in enumerate(pairs) if j != i]
        own = (wanted - set().union(*(q[2] for q in others))) or wanted
        own_meaning = (meaning - set().union(*(q[3] for q in others))) or meaning
        scenes = [p for p in sb.panels if p.kind == "scene" and _uses_symbol(drawn := words(p.visual), wanted) and drawn & own]
        if not scenes or not meaning:
            continue
        # at most SYMBOL_KEYS_PER_PANEL keys are asked of one panel: a hook that draws three new symbols gets the third
        # key on the next panel that draws it (three keys could not fit a teaching scene's words)
        home = next((p for p in scenes if len(out.get(p.n, [])) < SYMBOL_KEYS_PER_PANEL), scenes[0])
        told = lambda p: (_key_words(own_meaning, said := words(" ".join([p.narration] + [b.text for b in p.dialogue])) - cast)
                          and _key_words(own, said))   # "reallocating" names "reallocation"; "thin" never names "thick"
        if not any(told(p) for p in scenes if p.n <= home.n):
            out.setdefault(home.n, []).append(
                f'symbol: this panel draws "{symbol}", the book\'s symbol for "{concept}", and no panel has said what it stands '
                f'for: let a character name it in the sentence that says what happens to it, in a bubble or the narration ("This '
                f'{_noun(symbol)} is {_noun(concept)}, and ..."), not as a line of its own or a legend')
    return out


# a reason: "because", "this is why", "so" as a connective ("..., so the student...", "so it can...", "first so Kumo can"),
# a causal ", as" ("..., as fixed paths do not match"), never "so simple" or ", as shown"; and the ways people give one
# without those words (round 15, told to talk plainly): "better leads make this one less worth keeping", "residuals let
# deep networks train", "without these, ...", "the better the next leads, the less ...", "this avoids ...", a "why ...?",
# and the problem a step fixes ("This fixes the weak low-rank bottleneck", "cuts the attention sink")
_MORE_LESS = r"(?:more|less|fewer|better|worse|higher|lower|bigger|smaller|larger|longer|shorter|deeper|faster|slower|harder|easier)"
WHY_RE = re.compile(r"\b(?:makes?|made|making)\s+(?:[\w'’-]+\s+){0,3}?(?:" + _MORE_LESS + r"|possible|impossible|stable|unstable|"
                    r"useless|safer|cheaper|sure|able|unable|reliable|robust|worth)\b|"
                    r"\blets\b|(?<=[a-z,] )let (?!me\b|us\b)[\w-]+ [\w-]+|"
                    r"\bwithout (?:this|these|that|those|it|them)\b|(?:^|[.!?…]\s+|[:;]\s*)without\b[^.!?;]{2,80},|"
                    r"\bthe " + _MORE_LESS + r"\b[^.;!?]{1,80}?\bthe " + _MORE_LESS + r"\b|"
                    r"\b(?:avoids?|avoided|avoiding|prevents?|prevented|preventing|ensures?|ensured|ensuring|guarantees?|guaranteed|"
                    r"reasons?|to (?:let|prevent|ensure|guarantee|protect|fix))\b|\bwhy\b[^.!?]*\?|"
                    r"\bhelps?\s+(?:to\s+)?(?:guard|avoid|prevent|keep|stop|fix|reduce|cut|catch)\b|"
                    r"\b(?:fix(?:es|ed)?|solv(?:es|ed)|cur(?:es|ed)|remov(?:es|ed)|cuts?|breaks?|blocks?|stops?|reduces?|shrinks?|"
                    r"eliminates?|mitigates?|repairs?|undoes)\s+(?:[\w'’-]+\s+){0,4}?(?:problems?|issues?|bugs?|sink|bottlenecks?|"
                    r"mismatch(?:es)?|gaps?|drift|errors?|bias|leaks?|collapse|noise|waste|failures?|divergence|misalignment|trap|"
                    r"instability|anchor)\b|"
                    r"(?:^|[,.;:!?…—–])\s*so\b(?![-,])|\bso (?:that|we|it|they|you|the|each|every|no|a|an|its|his|her|their|our|this|these|those)\b|"
                    r"\bso (?-i:[A-Z][a-z]+)\b|"
                    r",\s*as (?!(?:shown|seen|in|well|expected|noted|follows|before|usual|such|if|of|for|to|with|much|many|long|far|soon|"
                    r"a|an|on|at|by|part|opposed)\b)(?![^,.;:!?]*\b(?:shows?|showed|shown|said|says|saw|see|seen|describes?|described|"
                    r"reports?|reported|notes?|noted|found|finds)\b)[a-z]+ [a-z]+|"   # ", as the paper shows" reports, it gives no reason
                    r"\b(?:because|(?:this|that|which)(?:'s|’s| is) why|which means|this means|since|therefore|thus|hence|due to|"
                    r"thanks to|as a result|that way|the reason|(?:this|which) lets|in order to|to avoid|to stop|to keep|"
                    r"to make sure|otherwise)\b|"
                    # round 18's reasons in plain words: "bad long text can't change a fixed answer", "a single KL cannot
                    # correct both error types", "this routing frees human time", "worth human effort", "simple
                    # correlations hide the true effect", "this method separates cause from coincidence", "hard to
                    # explain by other changes", "mixed answers mean there is something to learn"
                    r"\b(?:can['’]?t|cannot|can not|won['’]?t|will not|no longer)\s+(?:[\w'’-]+\s+){0,2}?(?:change|hurt|harm|break|"
                    r"affect|spoil|ruin|corrupt|mislead|confuse|derail|undo|wreck|sway|fool|distract)\b|"
                    r"\b(?:can['’]?t|cannot|can not|fails? to|unable to)\s+(?:correct|fix|handle|catch|tell|separate|cover|capture|"
                    r"keep up|scale)\b|\b(?:frees|freed|freeing)\b|\bfree\s+up\b|\bsaves?\s+(?:[\w'’-]+\s+){0,2}?(?:time|money|effort|compute|"
                    r"memory|work)\b|\bworth\s+(?:the\s+|human\s+|their\s+|its\s+|our\s+)?(?:effort|time|money|keeping|trying|"
                    r"doing|calling)\b|\b(?:separates|separating|to separate)\b[^.;!?]{1,40}?\bfrom\b|"
                    r"\bhides?\s+(?:the\s+)?(?:true|real)\s+(?:effect|cause|picture|signal)\b|\bhard to explain\b|"
                    r"\bcan(?:not|['’]t) be explained\b|\bmeans?\s+(?:that\s+)?there(?:\s+is|['’]s|\s+are)\b", re.I)


# a student's "why ...?" is the reason only once it is answered; "why didn't anyone ...?" asks for none
WHY_ASK_RE = re.compile(r"\bwhy\b(?!\s+(?:didn['’]t|did not|has no one|hasn['’]t|no one|nobody|not)\b)[^.!?]*\?", re.I)


DIAGRAM_ASKS = 2   # scenes of the idea and how chapters one lint run asks to become diagram boards
DIAGRAM_MIN_PANELS = 8   # a script long enough to tell its story in chapters (a book has 16-24 panels)


def _missing_diagram(sb: StoryboardV2) -> dict[int, list[str]]:
    """{n: [problem]}: the idea and how chapters explain on boards. A scene there that teaches an abstract claim is asked to
    become a diagram board: in a blind A/B of 24 teaching panels of round 22, the same words beside a code-drawn diagram of
    the mechanism on the running example were clearer than the shipped picture in 22, and the metaphor pictures needed a
    legend line each. A book whose problem, idea and how chapters draw no diagram at all gets one (1000 Layers' round 23
    dry run explained its method with a flow of three part names); a list there is no diagram either (Gated Attention's
    whole method was one scene and a list of two phrases, and readers could not say how the gate works, round 13)."""
    if len(sb.panels) < DIAGRAM_MIN_PANELS:
        return {}
    budget = BUDGETS_V3["en-US"]["board"]
    diagram = ('a diagram board that draws the mechanism on the running example ("kind": "diagram" with a "brief": the '
               'example\'s real things and the arrangement that shows this panel\'s point)')
    fit = (f'cut to a board panel\'s budget (at most {budget["bubbles"]} bubbles of {budget["bubble"]} words, '
           f'narration {budget["narration"]}, {budget["total"]} in all)')
    kind = lambda p: str(p.board.get("kind") or "").lower() if p.kind == "board" and isinstance(p.board, dict) else ""
    out: dict[int, list[str]] = {}
    for p in [p for p in sb.panels if p.beat in ("idea", "how") and p.kind == "scene" and p.claims][:DIAGRAM_ASKS]:
        out[p.n] = [f'diagram: this scene teaches claim {", ".join(map(str, p.claims))} with a picture, and a reader learns a '
                    f'mechanism from a drawing of it: make this panel a board panel ("kind": "board") with {diagram}; keep '
                    f'what its words teach, reworded to talk about the board and {fit}']
    explain = [p for p in sb.panels if p.beat in ("problem", "idea", "how")]
    if out or not explain or any(kind(p) == "diagram" for p in explain):
        return out
    order = {"how": 0, "idea": 1, "problem": 2}
    for wanted in ("scene", "list", "flow"):
        home = min((p for p in explain if (p.kind == "scene" if wanted == "scene" else kind(p) == wanted)),
                   key=lambda p: (order[p.beat], p.n), default=None)
        if home is not None:
            what = "this scene" if wanted == "scene" else f"this {wanted}"
            return {home.n: [f"diagram: the problem, idea and how chapters draw no diagram of the mechanism, so {what} has to "
                             f'show it: make this panel a board panel ("kind": "board") with {diagram}; keep what its words '
                             f"teach, reworded to talk about the board and {fit}"]}
    return {}


PROOF_BOARD_KINDS = ("list", "compare", "table", "flow")


def _board_cells(board: dict) -> list[str]:
    """The values a list, compare, table or flow board draws: a table's cells after the first column, a compare board's
    items, a list's items, a flow's steps ([] for other kinds). The first column, the heads and the side labels name the
    systems and settings compared, whose digits are no result ("Qwen3-8B", "WMT 2014", "Depth 4")."""
    norm = _normalized_board(board)
    if not isinstance(norm, dict):
        return []
    kind = norm.get("kind")
    if kind == "table":
        return [str(c) for r in norm.get("rows") or [] for c in (r.get("cells") or [])[1:]]
    if kind == "compare":
        return [str(x) for side in ("left", "right") if isinstance(norm.get(side), dict) for x in norm[side].get("items") or []]
    if kind in ("list", "flow"):
        return [str(x) for x in norm.get("items") or norm.get("steps") or []]
    return []


def _proof_without_numbers(sb: StoryboardV2) -> dict[int, list[str]]:
    """{n: [problem]}: a proof board whose values (_board_cells) hold no number. Readers found no data on "High, reliable /
    Much lower" (a table of words where the paper reports Spearman correlations), on "Shorter responses / Fewer truncations"
    and on "Outperforms all other baselines" (round 13): the evidence chapter showed claims, not results. A scene is
    offered only for a claim without numbers: the comparison rule asks a claim with numbers for a drawn comparison, and a
    scene in its place went back and forth between the two rules."""
    out: dict[int, list[str]] = {}
    claims = {c.get("id"): c for c in sb.abstract_claims if isinstance(c, dict)}
    for p in sb.panels:
        if p.beat != "proof" or p.kind != "board" or not isinstance(p.board, dict):
            continue
        if str((_normalized_board(p.board) or {}).get("kind") or "") not in PROOF_BOARD_KINDS:
            continue
        cells = _board_cells(p.board)
        if not cells or _number_tokens(" ; ".join(cells)):
            continue
        words = "; ".join(c for c in cells if c)[:120]
        draw = ("draw the paper's numbers for this result (a bar chart, or a table whose cells hold the values, each with its quote "
                "from the PAPER TEXT)")
        if any(_key_numbers(claims[cid]) for cid in p.claims if cid in claims):
            out[p.n] = [f"the proof board draws no numbers (only words: {words}), so the reader sees a claim, not its evidence: {draw}"]
        else:
            out[p.n] = [f"the proof board draws only words ({words}), so the reader sees a claim, not its evidence: {draw}; when the "
                        "paper reports no number for it, make this panel a scene whose picture shows the result happening"]
    return out


UNCHARTED_MAX = 3   # results a lint run asks to chart: a fix round adds at most FIX_NEW_PANELS panels
SHORT_DRAFT = 2     # a first draft this many panels or more under the story's range is written once more
CHART_PANEL_ASK = "add a board panel right after this one"   # the one ask a fix round may add a panel for


def _evidence_numbers(text: str) -> list:
    """The numbers of a brief evidence item that identify it on a chart: not years, not small counts."""
    boards = _optional("boards")
    if boards is None:
        return []
    out = []
    for n in boards.find_numbers(str(text or "")):
        try:
            v = abs(float(n.key))
            small = v == int(v) and v < 11
        except (ValueError, OverflowError):   # float("1e400") is inf, and int(inf) raises: a huge number is no small count
            continue
        if n.short or 1900 <= v <= 2100 or small:
            continue
        out.append(n)
    return out


CHART_HOMES = ("idea", "how", "proof", "payoff", "limits")   # the chapters a result's chart may join (never the hook)
UNCHARTED_LEAD = "evidence: the paper's result \""


def _uncharted_evidence(sb: StoryboardV2, paper_text: str, report: bool = False) -> dict[int, list[str]]:
    """{n: [problem]}: results of the brief's "evidence" that the paper reports with numbers and that no chart or table of
    the story draws. In rounds 21-22, 22 of 71 such results were only said, most of them the paper's figures, and readers'
    commonest complaint was a result stated with no chart (Flexibility Trap's 21.3% vs 0.6%, 1000 Layers' 8 of 10 tasks).
    The problem goes to the teaching panel that talks about the result, a proof panel first (else the last proof panel),
    asking for a chart panel after it; each result is asked for once per book (sb.brief["chart_asked"]): 1000 Layers'
    depth-versus-width result is a figure whose values the paper never prints, and every round asked for its chart again,
    the chart failed the number check and became a scene, and five one-line scenes about it piled up (round 23)."""
    boards = _optional("boards")
    if boards is None:
        return {}
    drawn = set()
    for p in sb.panels:
        if p.kind == "board" and isinstance(p.board, dict) and str(p.board.get("kind") or "").lower() in ("bar", "table"):
            drawn |= {n.key for n in boards.find_numbers(board_text(p.board))}
    paper = _own_text(paper_text)
    asked = set() if report else {str(x) for x in _as_list(sb.brief.get("chart_asked"))}   # the report lists them all
    teach = [p for p in _teaching_panels(sb.panels) if p.beat in CHART_HOMES]
    proof = [p for p in teach if p.beat == "proof"] or teach
    out: dict[int, list[str]] = {}
    count = 0
    for e in _as_list(sb.brief.get("evidence")):
        if not isinstance(e, dict) or count >= UNCHARTED_MAX:
            continue
        claim = _evidence_key(e)
        if claim in asked:
            continue
        nums = [n for n in _evidence_numbers(e.get("numbers")) if n.text in paper]
        if not nums or any(n.key in drawn for n in nums):
            continue
        keys = {n.key for n in nums}
        words = {_stem(w) for w in _content_tokens(claim)}
        def score(p):
            said = visible_text(p)
            return (len(keys & {n.key for n in boards.find_numbers(said)}),
                    len(words & {_stem(w) for w in _content_tokens(said)}), p.beat == "proof", -p.n)
        home = max(teach, key=score) if teach else None
        if home is None or score(home)[:2] == (0, 0) or score(home)[1] < 2 and not score(home)[0]:
            home = proof[-1] if proof else None
        if home is None:
            continue
        count += 1
        where = str(e.get("source") or "the paper")
        out.setdefault(home.n, []).append(
            f'{UNCHARTED_LEAD}{claim}" ({where}: {str(e.get("numbers") or "")[:160]}) is '
            f'never drawn on a chart: {CHART_PANEL_ASK} ({{"after": N}}) with a bar chart or table of these numbers from the '
            'paper (each with its quote) and the comparison they make, and one line saying what it shows (when this panel is '
            'a scene that only says this result, make this panel that board panel instead)')
    return out


def _evidence_key(e: dict) -> str:
    """How a brief evidence item is named in an _uncharted_evidence problem and in sb.brief["chart_asked"]."""
    return str(e.get("claim") or e.get("numbers") or "")[:160]


def _chart_asked(problem: str, evidence: list) -> str:
    """The result (_evidence_key) an _uncharted_evidence problem asks a chart for ("" for any other problem): matched
    against the brief's results, the longest first (a claim with '" (' in it was cut short when split out)."""
    if not problem.startswith(UNCHARTED_LEAD):
        return ""
    keys = sorted({_evidence_key(e) for e in evidence if isinstance(e, dict)} - {""}, key=len, reverse=True)
    return next((k for k in keys if problem.startswith(f'{UNCHARTED_LEAD}{k}" (')), "")


def _charted_asks(asks: dict, splits: list, evidence: list) -> list[str]:
    """The results (_evidence_key) a fix round added a chart panel for: after each panel whose chart ask got its new panel,
    the results whose numbers that panel draws (else the first one asked there). A chart that then fails the number check
    is never asked for again (1000 Layers' depth-versus-width chart, read off a figure, became a scene every round and
    five one-line scenes piled up); a result the round had no slot or room for is asked for again later."""
    boards = _optional("boards")
    done: list[str] = []
    for n, keys in asks.items():
        new = next((panel for m, panel, _ in splits if m == n), None)
        if new is None:
            continue
        drawn = {x.key for x in boards.find_numbers(board_text(new.board))} if boards is not None and isinstance(new.board, dict) else set()
        numbers = {_evidence_key(e): e.get("numbers") for e in evidence if isinstance(e, dict)}
        hit = [k for k in keys if any(x.key in drawn for x in _evidence_numbers(numbers.get(k)))]
        done += hit or keys[:1]
    return done


def _without_new_panel(problem: str) -> str:
    """An _uncharted_evidence problem for a round that may add no panel: the panel itself becomes the chart when it is a
    scene that only says the result. Asked for a new panel that the round then refused, the writer lost the panel's
    whole rewrite (round 23's review)."""
    if CHART_PANEL_ASK not in problem:
        return problem
    head = problem.split(f"{CHART_PANEL_ASK} (", 1)[0]
    return (head + "no panel can be added in this round: when this panel is a scene that only says this result, make it a "
            "board panel with a bar chart or table of these numbers from the paper (each with its quote) and the comparison "
            "they make; otherwise leave the result in the words")


# the skeptic's bet (the claims beat) and its settlement: "I bet my berry tart this doesn't hold up." ... "Fine, you win."
BET_RE = re.compile(r"\b(?:I(?:'ll|’ll| will)?\s+bet\b|bet(?:s|ting)?\s+(?:you|my|her|his|this|a|one)\b|wager)", re.I)
BET_STAKE_RE = re.compile(r"\bbet(?:s|ting)?\s+(?:you\s+)?(?:my|her|his|this|that|a|one|your)\s+([a-z'’-]+(?:\s+[a-z'’-]+)?)", re.I)
BET_SETTLED_RE = re.compile(r"\b(?:you win|you won|you've won|I lose|I lost|(?:pays?|paying|paid) up|fair and square|"
                            r"(?:it'?s|they'?re) yours|you(?:'ve)? earned|I owe you|I concede|"
                            r"(?:won|lost|settles?|settled|lose|win) (?:the|our|my|her|this) bet|you were right|I was wrong|"
                            r"here'?s your|fair enough|(?:the )?(?:numbers|results|claims?|data) (?:hold|held)(?: up)?)\b", re.I)
BET_STOP = {"this", "that", "the", "these", "those", "it", "is", "on", "if", "you", "they", "whole", "none", "it's", "it’s",
            "its", "every", "all", "they're", "they’re", "any", "some", "of", "at", "in", "for", "a", "an", "to", "with", "and",
            "or", "not", "no", "will", "would", "can", "could", "be", "was", "are", "works", "work", "beats", "wins", "is"}


def _unsettled_bet(sb: StoryboardV2) -> dict[int, list[str]]:
    """{n: [problem]}: the skeptic bets early (hook, claims or problem) and no later panel settles it (the proof's reaction,
    the payoff, the limits or the recap: "Fine, you win", the stake named again). The writer is told to settle it in the
    limits, and 15 of 16 books of rounds 22 and 23 did; Flexibility Trap's round-23 judge found its bet left open, the
    tart drawn whole again, and the story's arc with it."""
    def words(p: PanelV2) -> str:
        return " ".join([b.text for b in p.dialogue] + [p.narration or ""])
    early = [p for p in sb.panels if p.beat in ("hook", "claims", "problem") and BET_RE.search(words(p))]
    if not early:
        return {}
    # "I bet it doesn't work that way" in the hook is no wager: the stake is the claims board's ("Bet my tart ...")
    bet = next((p for p in early if BET_STAKE_RE.search(words(p))), early[0])
    claimed = {w.lower() for c in sb.abstract_claims if isinstance(c, dict) for w in re.findall(r"[a-z'’-]+", str(c.get("text") or ""), re.I)}
    stakes = {w.lower().strip("'’") for p in early for m in BET_STAKE_RE.finditer(words(p)) for w in m.group(1).split()} \
        - BET_STOP - claimed or {"tart"}   # no stake named: the skeptic's tart, the prompt's stake; the claim's words are no stake
    for q in sb.panels:
        if q.n <= bet.n or q.beat not in ("proof", "payoff", "limits", "recap"):
            continue
        text = words(q)
        if BET_SETTLED_RE.search(text) or (q.beat != "proof" and any(re.search(rf"\b{re.escape(w)}", text, re.I) for w in stakes)):
            return {}
    target = next((q for q in reversed(sb.panels) if q.beat == "limits"), None) \
        or next((q for q in reversed(sb.panels) if q.beat == "payoff"), None)
    if target is None:
        return {}
    said = BET_RE.search(words(bet))
    return {target.n: [f"the bet of panel {bet.n} (\"{words(bet)[max(0, said.start() - 10):said.end() + 50].strip()}\") is never "
                       "settled: in one line here, the skeptic pays up because the results the story showed held on these "
                       "tests (name one), and keeps the limits as a fair point"]}


REPEAT_NUMBER_PANELS = 2   # a result's number may appear in the words of this many panels of the story


def _repeated_numbers(sb: StoryboardV2) -> dict[int, list[str]]:
    """{n: [problem]}: a number said in the words of more than REPEAT_NUMBER_PANELS panels of the story (the claims board and
    the recap aside; a panel reading its own chart counts once). The judges of three eras found the same number told again
    and again (89.1% five times, +10.26 four times) where the next panel should have gone one step further. The panels
    flagged are those that do not teach a claim with that key number, latest first: the claim rule asks for the number on
    a panel that teaches the claim, and flagging that panel made the two rules undo each other's fix (round 23's review).
    Never a year, a model size, a number inside a name ("pass@64") or one in the paper's title ("1000 layers")."""
    boards = _optional("boards")
    if boards is None:
        return {}
    claims = {c.get("id"): c for c in sb.abstract_claims if isinstance(c, dict)}
    # the example's own numbers (a walked-through "180 - 0.30 × 180 = 54"), never a claim's key number a part repeats
    # ("46.7% average attention on the first token" is the headline result: round 25's review)
    headline = {key for c in claims.values() for _, key in _key_numbers(c)}
    example = {n.key for n in boards.find_numbers(str(sb.running_example or ""))} | \
        ({n.key for n in boards.find_numbers(" ".join(_example_parts(sb)))} - headline)
    # the paper's own title (the manga's title only when the brief has none: "28.4 BLEU Without Recurrence" exempted the
    # headline number the rule exists for)
    named = {n.key for n in boards.find_numbers(str(sb.brief.get("title") or sb.title or ""))}
    seen: dict[str, list[tuple[int, str, bool]]] = {}   # key -> [(panel, as written, the panel teaches it)]
    for p in _teaching_panels(sb.panels):
        own = {key for cid in p.claims if cid in claims for _, key in _key_numbers(claims[cid])}
        words = boards._prep(" ".join([p.narration] + [b.text for b in p.dialogue]))   # the string find_numbers' offsets index
        for n in boards._scan(words):
            if (n.short or n.key in example or n.key in named or YEAR_RE.fullmatch(n.text) or SIZE_TOKEN_RE.fullmatch(n.text)
                    or words[max(0, n.start - 1):n.start] == "@"):
                continue
            try:
                if float(n.key) == int(float(n.key)) and abs(float(n.key)) < 11:
                    continue
            except (ValueError, OverflowError):   # 10^360 is a number all the same, and no small count
                pass
            panels = seen.setdefault(n.key, [])
            if all(p.n != m for m, _, _ in panels):
                panels.append((p.n, n.text, n.key in own))
    out: dict[int, list[str]] = {}
    for key, panels in seen.items():
        over = len(panels) - REPEAT_NUMBER_PANELS
        if over <= 0:
            continue
        free = [x for x in panels if not x[2]]
        flagged = (free[-over:] if len(free) >= over else free + [x for x in panels if x[2]][-(over - len(free)):])
        for m, written, _ in sorted(flagged):
            others = [str(k) for k, _, _ in panels if k != m]
            out.setdefault(m, []).append(
                f"this panel says {written} again (panels {', '.join(others)} also say it): say what this panel adds instead "
                "(the next step, the comparison, what it means), and leave the number to the recap")
    return out


def _missing_why(sb: StoryboardV2) -> dict[int, list[str]]:
    """{n: ["why: ..."]}: an idea panel, or a how panel, whose words never give a reason (auditors of the first picture
    books: method panels name the step but never say why it fixes the problem)."""
    out: dict[int, list[str]] = {}
    for i, p in enumerate(sb.panels):
        if p.beat not in ("idea", "how"):
            continue
        words = " ".join([p.narration] + [b.text for b in p.dialogue]
                         + ([str(p.board.get(k) or "") for k in ("subtitle", "note")] if isinstance(p.board, dict) else []))
        reason = WHY_RE.search(WHY_ASK_RE.sub(" ", words))
        asked = next((k for k, b in enumerate(p.dialogue) if WHY_ASK_RE.search(b.text)), None)
        if not reason and asked is not None:   # a "why ...?" counts when a later bubble or the next panel answers it
            after = sb.panels[i + 1] if i + 1 < len(sb.panels) else None
            answer = " ".join([b.text for b in p.dialogue[asked + 1:]]
                              + ([after.narration] + [b.text for b in after.dialogue[:1]] if after else []))
            reason = WHY_RE.search(WHY_ASK_RE.sub(" ", answer))
        if not reason:
            out.setdefault(p.n, []).append(
                "why: this panel says what the method does but not why: add one plain reason that ties this step to the problem "
                "it fixes, from the PAPER TEXT, said the way people talk (\"The answer is fixed first, so the long text cannot "
                "make it wrong.\"; never \"Because X, so Y\")")
    return out


def _recap_mirrors_claims(sb: StoryboardV2) -> dict[int, list[str]]:
    """{n: ["recap: ..."]}: the recap lists the claims board's lines again, in order (a reader looked for the five promises
    of the claims board and found a different list at the end)."""
    lists = lambda beat: [p for p in sb.panels if p.beat == beat and isinstance(p.board, dict) and str(p.board.get("kind") or "").lower() == "list"]
    claims, recap = lists("claims"), lists("recap")
    if not claims or not recap:
        return {}
    items = lambda p: [str(x) for x in _as_list(p.board.get("items")) if str(x).strip()]
    if is_novice(sb):
        questions = [x for p in claims for x in items(p)]
        answers = [x for p in recap for x in items(p)]
        if len(questions) != len(answers) or any(a == q or a.endswith(("?", "？")) for q, a in zip(questions, answers)):
            return {recap[-1].n: ["recap: answer each opening question in order, in plain words with its supported scope; do not copy the questions"]}
        return {}  # Semantic answers, prerequisites and limits are assessed by review_story.
    words = lambda t: {_stem(w) for w in _content_tokens(t) if not w[0].isdigit()}

    def paired(promised: list, told: list) -> bool:
        return len(promised) == len(told) and all(len(words(a) & words(b)) >= 0.4 * len(words(a)) for a, b in zip(promised, told) if words(a))

    def problem(promised: list) -> list[str]:
        return ["recap: the recap must list the claims board's lines again, in the same order, now each with its key number, so "
                "the reader sees every promise come back" + (" (on two recap boards when they do not fit on one: at most 6 lines "
                "each)" if len(promised) > 6 else "") + f". The claims board said: {json.dumps(promised, ensure_ascii=False)}"]
    if len(claims) == len(recap):   # board by board: a list holds 6 lines, so a long abstract gets two of each
        return {r.n: problem(items(c)) for c, r in zip(claims, recap) if not paired(items(c), items(r))}
    promised = [x for c in claims for x in items(c)]
    return {} if paired(promised, [x for r in recap for x in items(r)]) else {recap[-1].n: problem(promised)}


def _unexplained_terms(sb: StoryboardV2, paper_text: str) -> dict[int, list[str]]:
    """{n: ["term: ..."]}: the teaching panel where a reader first meets a glossary term, or an abbreviation the paper
    uses, without an explanation there or on an earlier panel (the claims board may explain it; its mere listing is not
    a first meeting). Abbreviations count only in the words, not as bar labels or table cells (names of compared systems)."""
    def seen(p: PanelV2) -> str:
        return "  ".join([p.narration] + [b.text for b in p.dialogue] + [board_text(p.board) if p.board else ""])

    terms: list[tuple[list[str], str, bool]] = []   # (names, plain meaning, from the glossary)
    for g in _as_list(sb.brief.get("glossary")):
        if isinstance(g, dict) and g.get("term") and _term_forms(g["term"]):
            terms.append((_term_forms(g["term"]), _text(g.get("plain")), True))
    names = sorted({f for forms, _, _ in terms for f in forms}, key=len, reverse=True)
    known = {f.lower() for f in names}

    def bare(p: PanelV2) -> str:   # the words without the glossary's names: "Dynamic KL" is one term, not KL as well
        text = _words_text(p)
        for f in names:
            text = _term_re(f).sub(" ", text)
        return text

    paper = paper_text or ""
    for p in sb.panels:
        words = bare(p)
        numerals = {m.group(1) for m in ROMAN_RE.finditer(words)}   # "Type II error"
        for a in ACRONYM_RE.findall(words):
            if a in COMMON_ACRONYMS or a in numerals or a.lower() in known or not re.search(r"(?<![\w-])" + re.escape(a) + r"(?![\w-])", paper):
                continue
            if a.isupper() and len(a) >= 3 and sum(m.group(0) != a for m in re.finditer(r"(?<![\w-])" + re.escape(a) + r"(?![\w-])", paper, re.I)) >= 3:
                continue   # ONLY, NOT, MASK: a word the paper also writes in lower case is shouted, not abbreviated
            terms.append(([a], "", False))
            known.add(a.lower())
    out: dict[int, list[str]] = {}
    teaching = _teaching_panels(sb.panels)
    for forms, plain, glossary in terms:
        look = seen if glossary else bare
        first = next((p for p in teaching if any(_term_re(f).search(look(p)) for f in forms)), None)
        if first is None or len(out.get(first.n, [])) >= TERMS_PER_PANEL:
            continue
        earlier = [p for p in sb.panels if p.n < first.n]
        if (earlier and _term_explained(forms, "  ".join(seen(p) for p in earlier))) or _term_explained(forms, seen(first), plain) \
                or (earlier and plain and _term_explained(forms, seen(earlier[-1]), plain)):   # described, then named here
            continue
        name = " / ".join(forms)
        if plain:
            out.setdefault(first.n, []).append(
                f"term: a reader meets \"{name}\" here and no panel has said what it is. Explain it in a few plain words in this "
                f"panel (a bubble, the narration or the board's subtitle). It means: {plain}")
        else:
            out.setdefault(first.n, []).append(
                f"term: a reader meets the abbreviation \"{name}\" here and no panel has said what it stands for. Spell it out "
                "and say in a few plain words what it is, as the PAPER TEXT defines it, in this panel (a bubble, the narration "
                "or the board's subtitle)")
    return out


# words of a bar label that describe a system instead of naming it ("Standard attention (baseline)", "Base model 8B")
RIVAL_GENERIC = frozenset("baseline baselines base standard original vanilla default regular plain basic simple full precision "
                          "fixed prior previous old earlier existing usual common naive model models method methods attention "
                          "transformer network training trained dense sparse no without with only random uniform human humans "
                          "reference control average mean the a an of and or".split())


def _unexplained_rivals(sb: StoryboardV2) -> dict[int, list[str]]:
    """{n: ["term: ..."]}: the system a bar chart measures its gap against (the bar marked "baseline"), on the first chart
    that draws it, when no panel up to there says what it is. Readers of rounds 12-15 met "beats BitDistiller by 10.26"
    and "Enhanced HotpotQA" with nothing about what they are; a label that only describes ("Standard attention") needs
    nothing."""
    out: dict[int, list[str]] = {}
    seen = {f.lower() for g in _as_list(sb.brief.get("glossary")) if isinstance(g, dict) and g.get("term")
            for f in _term_forms(g["term"])}   # a glossary term has its own check, with its meaning
    for p in sb.panels:
        board = p.board if p.kind == "board" and isinstance(p.board, dict) else {}
        if str(board.get("kind") or "").lower() != "bar":
            continue
        bars = [b for b in _as_list(board.get("bars")) if isinstance(b, dict)]
        base = next((b for b in bars if b.get("baseline")), None)
        top = next((b for b in bars if b.get("highlight")), None)
        tokens = lambda b: [w for w in re.findall(r"[\w.+-]+", re.sub(r"\([^)]*\)|[*∗^†]", " ", str((b or {}).get("label") or "")))
                            if not re.fullmatch(r"[\d.,:/%+\-–]+[A-Za-z]{0,2}|[A-Z]?\d[\w.]*", w)]
        words = tokens(base)
        while words and words[0].lower() in RIVAL_GENERIC:
            words.pop(0)
        while words and words[-1].lower() in RIVAL_GENERIC:
            words.pop()
        full = " ".join(words)
        # what the baseline shares with the highlighted bar is the setting both are in ("Fixed Dynamic KL" against
        # "On-policy Dynamic KL", "Calls 81-120" against "Calls 1-3"), not the system compared
        shared = {w.lower() for w in tokens(top)}
        name = " ".join(w for w in words if w.lower() not in shared)   # a name has a capital ("BitDistiller", "SPG")
        if not re.search(r"[A-Z]", name) or name.lower() in seen or full.lower() in seen:
            continue
        seen.add(name.lower())
        told = "  ".join(_words_text(q) for q in sb.panels if q.n <= p.n)
        if _term_explained([name], told) or _term_explained([full], told) \
                or (len(name.split()) > 1 and any(_term_explained([w], told) for w in name.split() if len(w) > 2)):
            continue
        out.setdefault(p.n, []).append(
            f'term: the chart measures against "{name}", and no panel has said what it is: say it in a few plain words in this '
            f'board\'s subtitle or a bubble ("{name}: an earlier method that ..."), as the PAPER TEXT describes it')
    return out


def _abstract_terms_missing(sb: StoryboardV2) -> dict[int, list[str]]:
    """{n: ["term: ..."]}: a glossary term that the abstract itself names and no panel names, on the teaching panel that
    says most of what it means (econ_ai_service's script of 2026-10-28 said only "they compared right before and after AI
    launched", and no reader could name the paper's regression-discontinuity-in-time design; the term rule only judges
    a term the script uses, so dropping one was the easy way out). A name met only on the claims board or the recap is
    not named: those lists explain nothing. Without a found abstract (Flexibility Trap's has no heading) the claims'
    quotes stand in for it."""
    abstract = str(sb.brief.get("abstract") or "") or " ".join(str(c.get("quote") or "") for c in sb.abstract_claims if isinstance(c, dict))
    if not abstract.strip():
        return {}
    flat = lambda text: re.sub(r"[-‐‑–\s]+", " ", text)   # "regression discontinuity in time" names the hyphenated term
    abstract, shown = flat(abstract), flat("  ".join(visible_text(p) for p in _teaching_panels(sb.panels)))
    out: dict[int, list[str]] = {}
    for g in _as_list(sb.brief.get("glossary")):
        if not (isinstance(g, dict) and g.get("term")):
            continue
        forms = [flat(f) for f in _term_forms(g["term"])]
        if not forms or not any(_term_re(f).search(abstract) for f in forms) or any(_term_re(f).search(shown) for f in forms):
            continue
        plain = _text(g.get("plain"))
        home = _home_panel(sb, {"text": f"{forms[0]} {plain}", "quote": ""})
        out.setdefault(home.n, []).append(
            f'term: the abstract names "{" / ".join(forms)}" and no teaching panel does: name it on the panel that teaches it (this one, '
            f"unless another panel shows it better), with what it means in a few plain words. It means: {plain}")
    return out


def _teaching_scene(p: PanelV2) -> bool:
    """A scene whose picture must show an idea (it has a "shows" and a teaching beat or the hook): its words leave room."""
    return p.kind == "scene" and bool(p.shows.strip()) and p.beat in PICTURE_BEATS + ("hook",)


def _kind_name(p: PanelV2) -> str:
    return "teaching scene (its picture needs the room)" if _teaching_scene(p) else f"{p.kind} panel"


def _without_terms(text: str, terms: list[str]) -> str:
    """The text without the glossary's term names: "perfect Bayesian equilibrium" is a term the abstract-term rule asks for,
    not the verdict word "perfect"."""
    for f in terms:
        text = _term_re(f).sub(" ", text)
    return text


def lint_v3(sb: StoryboardV2, paper_text: str, report: bool = False) -> dict[int, list[str]]:
    """Deterministic checks of an English v3 script: {panel n: [problems]} for the panels that have any.
    Per panel: kind vs board, the board's shape and evidence, text budgets, numbers missing from the paper.
    Story-wide: a claim no panel conveys, or a claim's key number shown on none of the panels that convey it; these go
    to the recap panel (the last panel when there is none), which restates every claim anyway."""
    out: dict[int, list[str]] = {}
    budget = BUDGETS_V3["en-US"]
    evidence = _optional("evidence")
    allowed = _illustrative_example(sb)
    ledger = _ledger(paper_text)
    cite = _optional("ledger") if ledger else None
    attention = len(ATTENTION_RE.findall(str(paper_text or ""))) >= 3
    terms = [f for g in _as_list(sb.brief.get("glossary")) if isinstance(g, dict) and g.get("term") for f in _term_forms(g["term"])]
    for p in sb.panels:
        problems: list[str] = []
        b = budget["board" if p.kind == "board" else "scene"]
        if _teaching_scene(p):
            b = {**b, **budget["teach"]}
        if p.kind == "board":
            if not isinstance(p.board, dict) or not p.board:
                problems.append("board panel without a board: give it a board (bar, table, flow, compare, links or list) or make it a scene panel")
            else:
                said = _words_vs_chart(p)
                unmark = any("unmark the baseline" in x for x in said)   # the change-bar order: no rule asks for a baseline
                story = [x for x in _story_board_issues(p.board, ledger) if not (unmark and "is not the strongest earlier system" in x)]
                problems += (board_issues(p.board, paper_text) + story + said
                             + _numbers_off_board(p, sb.abstract_claims) + ([] if unmark else _badge_vs_words(p))
                             + _parentheses_unexplained(p) + _fractions_as_points(p))
                brief = str(p.board.get("brief") or "") if _is_diagram(p.board) else ""
                if DATA_CHART_BRIEF_RE.search(brief) and len(CHART_VALUE_RE.findall(brief)) >= 2:
                    # Hivemind's "A bar chart with three bars: 11 votes for river, 10 for weaver, 4 for tie" was drawn with
                    # made-up counts (small numbers pass any check against the paper), and its next diagram was a slope
                    # with no values at all: readers took both for the paper's data (round 23)
                    problems.append("diagram: its brief asks for a chart of numbers; a result's numbers go on a bar or table "
                                    "board with the paper's values (cited or quoted), so make this a bar or table board, or "
                                    "rewrite the brief to draw the mechanism or the things that were compared, with no values")
                if p.board.get("unbound"):
                    problems.append(f"evidence: the board cites {', '.join(p.board['unbound'])}, which name no number cell of the "
                                    "TABLES section: cite an existing cell id, or copy the paper's number with a verbatim quote")
                if p.board.get("kind") == "links" and not attention:
                    problems.append("a links board draws how strongly words attend to each other, and this paper is not about "
                                    "attention between words: readers cannot tell what its arcs mean. Use a flow board (the "
                                    "steps of the argument) or a compare board instead")
            if len(p.characters) > 2:
                problems.append(f"{len(p.characters)} cast members on a board panel: at most 2")
            if not p.dialogue:
                problems.append("board panel without a speech bubble: a cast member reacts to the board")
        elif p.board:
            problems.append('scene panel with a board: make it a board panel ("kind": "board") or remove the board')
        elif not p.dialogue and not p.narration:
            problems.append("no visible text: every panel teaches one fact in a bubble or the narration")
        props = sorted({m.group(0).lower() for m in WRITING_PROP_RE.finditer(f"{p.scene} {p.visual}")})
        if props:
            problems.append(f"the picture has props that carry writing ({', '.join(props)}): the illustrator draws them with "
                            "scribbles; draw the real things without writing (a closed book, a screen turned away) or the cast's "
                            "gestures instead")
        charts = sorted({m.group(0).lower() for m in SCENE_CHART_RE.finditer(f"{p.scene} {p.visual}")}) if p.kind != "board" else []
        if charts:
            problems.append(f"the picture of a scene shows a chart ({', '.join(charts)}): the illustrator draws a fake one, bars "
                            "without labels or values; put the chart on a board panel (the program draws it) or show only the "
                            "characters' reaction")
        figures = sorted({m.group(0).lower() for m in FIGURE_PROP_RE.finditer(NO_FIGURE_RE.sub(" ", f"{p.scene} {p.visual}"))})
        if figures:
            problems.append(f"the picture has props shaped like a person or creature ({', '.join(figures)}): they are drawn as "
                            "extra characters; draw real objects that are not figures, or leave them out")
        spelled = SPELLED_RE.search(f"{p.scene} {p.visual}")
        if spelled:
            problems.append(f'the picture asks the illustrator to draw words ("{spelled.group(0).strip()}"): it cannot write, and the '
                            "art check rejects every attempt; show the situation with objects and faces instead (a phone with a big "
                            "red cross, a parcel left at the door)")
        chats = sorted({m.group(0).strip().lower() for m in CHAT_BOX_RE.finditer(f"{p.scene} {p.visual}")})
        if chats:
            problems.append(f"the picture has chat windows or text boxes ({', '.join(chats)}): the illustrator draws them as "
                            "empty speech bubbles that readers take for missing text; " + (
                                "show the idea with glowing orbs, blocks or other objects the cast handle" if SCENE_SHOWS else
                                "draw the real situation without them (the person at a screen turned away, the cast reacting)"))
        if len(p.dialogue) > b["bubbles"]:
            problems.append(f"{len(p.dialogue)} speech bubbles: at most {b['bubbles']} on a {_kind_name(p)}")
        texts = [(f"bubble {i} ({x.speaker})", x.text, b["bubble"]) for i, x in enumerate(p.dialogue)]
        if p.narration:
            texts.append(("narration", p.narration, b["narration"]))
        total = sum(text_len(text, "en-US") for _, text, _ in texts)
        if total > b["total"]:
            problems.append(f"{total} words of bubbles and narration in total: at most {b['total']} on a {_kind_name(p)} "
                            "(say less: cut the words that repeat or decorate, keep the numbers and the claim)")
        legend = next((m.group(0) for _, text, _ in texts for m in [LEGEND_LINE_RE.search(text)] if m), "")
        if legend:
            problems.append(f'"{legend.strip()}..." reads as a legend: ' + (
                'name the symbol inside a sentence that says what happens to it ("The red cards are the hot leads, and the AI '
                'hands them to people"), never "X = Y"' if SCENE_SHOWS else
                'say it as a sentence about what happens ("The gate scales each word\'s value down"), never "X = Y"'))
        verdicts = sorted({m.group(0).lower() for _, text, _ in texts for m in VERDICT_RE.finditer(_without_terms(QUOTED_SPAN_RE.sub(" ", text), terms))})
        if verdicts:
            problems.append(f"verdict words {', '.join(verdicts)}: the paper shows results on its tests; write \"showed\", "
                            "\"on these tests\" or \"in this example\" instead")
        vouch = next((m.group(0) for _, text, _ in texts for m in [ALL_CLAIMS_RE.search(text)] if m), "")
        if vouch:
            problems.append(f'"{vouch}" vouches for every claim: the story showed the paper\'s evidence for some of them (the '
                            'recap ticks those); say what the charts showed ("The charts back the main result.")')
        address = next((m.group(0) for _, text, _ in texts for m in [URL_RE.search(text)] if m), "")
        if address:
            problems.append(f'a web address in the words ("{address[:60]}"): a reader cannot click a picture; say in plain words '
                            'what is public ("The code is public") and put the address in the panel\'s detail, the text under the picture')
        if isinstance(p.board, dict):
            line = next((m.group(0) for t in [str(p.board.get(k) or "") for k in ("title", "subtitle", "note")] + [x for _, x, _ in texts]
                         for m in [DRAWN_LINE_RE.search(t)] if m), "")
            if line:
                problems.append(f'the words promise a "{line}" that the program never draws: add that system as a bar of the chart, '
                                "or say its value in a bubble")
            note = next((m.group(0) for k in ("title", "subtitle", "note") for m in [DRAWING_NOTE_RE.search(str(p.board.get(k) or ""))] if m), "")
            if note:
                problems.append(f"the board's words describe the drawing (\"{note}\"): its title and subtitle say what the "
                                "lines are about (\"The paper's own summary\"), never how the board is drawn")
        elif p.kind != "board":
            pointed = next((m.group(0) for _, t, _ in texts for m in [CHART_WORDS_RE.search(t)] if m), "")
            if pointed:
                problems.append(f"\"{pointed}\": this is a scene panel with no chart or table; say the fact in words, or make "
                                "it a board panel that draws the exhibit")
        said = [t for _, t, _ in texts] + ([board_text(p.board)] if isinstance(p.board, dict) else [])
        certified = sorted({m.group(0).lower() for t in said for m in ALL_VERIFIED_RE.finditer(t)})
        if certified:
            problems.append(f"\"{certified[0]}\": the story showed evidence for some claims, it did not verify the paper; say "
                            "which results held on which tests (\"the BLEU results held on WMT 2014\"), never that every claim was verified")
        for where, text, limit in texts:
            words = text_len(text, "en-US")
            if words > limit:
                problems.append(f"{where} has {words} words: at most {limit} on a {p.kind} panel")
            if cite is not None and cite.unresolved(text, ledger):
                problems.append(f"evidence: {where} cites {', '.join(cite.unresolved(text, ledger))}, which name no number cell "
                                "of the TABLES section: cite an existing cell, or write the paper's number")
            if evidence is not None:
                try:
                    problems += [f"evidence: {where}: {x}" for x in evidence.verify_text_numbers(text, paper_text, allowed=[allowed, _badge_text(p)])]
                except Exception as e:
                    print(f"[StoryboarderV2] number check failed: {type(e).__name__}: {e}")
        if problems:
            out[p.n] = problems
    if sb.panels:   # the narration is read first, then the bubbles
        first = next((t.strip() for t in [sb.panels[0].narration] + [b.text for b in sb.panels[0].dialogue] if t.strip()), "")
        pointer = OPENING_POINTER_RE.match(first)
        if pointer:
            out.setdefault(sb.panels[0].n, []).append(
                f"the story opens with \"{first[:40]}\": the reader has seen nothing yet, so \"{pointer.group(0).strip()}\" points "
                "at nothing. Name the thing itself (\"Many AIs give the same answer\", not \"This sameness\")")
    reached: dict[str, tuple[int, int]] = {}   # chapter -> (highest point number so far, the panel that reached it)
    for p in _teaching_panels(sb.panels):
        said = [ORDINALS[m.group(1).lower()] for t in [p.narration] + [b.text for b in p.dialogue] for m in ORDINAL_RE.finditer(t)]
        if not said:
            continue
        top, at = reached.get(p.chapter, (0, 0))
        if min(said) < top:
            back = next(w for w, v in ORDINALS.items() if v == min(said))
            far = next(w for w, v in ORDINALS.items() if v == top)
            out.setdefault(p.n, []).append(
                f"this panel goes back to the {back} point after panel {at} reached the {far}: the reader loses track. Finish one "
                "point before starting the next (say this with the earlier point, or drop the numbering)")
        if max(said) > top:
            reached[p.chapter] = (max(said), p.n)
    for n, found in _unexplained_terms(sb, paper_text).items():
        out.setdefault(n, []).extend(found)
    for check in (_picture_problems, _symbol_keys, _missing_why, _missing_diagram, _proof_without_numbers, _recap_mirrors_claims,
                  _abstract_terms_missing, _unexplained_rivals, _repeats, _vague_list_lines, _meta_lines, _repeated_numbers,
                  _unsettled_bet):
        for n, found in check(sb).items():
            out.setdefault(n, []).extend(found)
    for n, found in _uncharted_evidence(sb, paper_text, report=report).items():   # what is left, for the job's report
        out.setdefault(n, []).extend(found)
    if _example_tokens(sb):   # the story follows one example: the reader sees it at the start and sees it solved at the end
        shown = _example_beats(sb)
        for beat in ("hook", "how", "payoff"):
            panels = [p for p in sb.panels if p.beat == beat]
            if panels and not shown[beat]:
                out.setdefault(panels[0].n, []).append(
                    # the method walks the example through its steps: LongZero's question appeared only at the start and
                    # the end, and "the answer is Sofia Bianchi" meant nothing to readers who never saw it built
                    f"the method panels never use the running example ({sb.running_example[:140]}): walk it through one step "
                    "of the method here (what happens to it at this step), in a bubble or the narration" if beat == "how" else
                    f"the {beat} does not show the running example ({sb.running_example[:140]}): quote it (or its key words) "
                    f"in a bubble, the narration or a links board, so the reader sees what the story is about"
                    + (" and what the method changes for it (solved, when the paper shows it solved; otherwise what the "
                       "paper's result means for problems like it)" if beat == "payoff" else ""))
    teaching = _teaching_panels(sb.panels)   # the claims board and the recap list claims; they never count as teaching them
    covered = {c for p in teaching for c in p.claims}
    for claim in sb.abstract_claims if sb.panels else []:
        cid = claim.get("id")
        if cid not in covered:
            home = _home_panel(sb, claim)
            out.setdefault(home.n, []).append(
                f"claim {cid} is not taught by any panel (the claims board and the recap do not count): state it, with its key "
                f"number, in this panel's visible text or board and add {cid} to \"claims\" ({claim.get('text', '')})")
            continue
        conveying = [p for p in teaching if cid in p.claims]
        shown = set().union(*(_number_keys(visible_text(p)) for p in conveying))
        missing = [written for written, key in _key_numbers(claim)   # "2.0" shows 2, "eight" shows 8, 8.1 shows "about 8"
                   if key not in shown and not _shown_about(claim, written, key, shown)]
        if missing:
            # a board panel's words must match its chart, so the number goes to a scene that teaches the claim when there is
            # one (Better Leads round 15: "8" was asked of a call-length chart, and the board check took it off again)
            home = max(conveying, key=lambda p: (p.kind != "board", p.beat == "proof", p.n))
            where = ("add it here, exactly as the paper writes it" if home.kind != "board" else
                     "add it to this board's data from the paper when the chart shows that quantity, otherwise write it in "
                     "this board's subtitle, exactly as the paper writes it (a setting, or what the paper also reports)")
            out.setdefault(home.n, []).append(
                f"claim {cid} ({claim.get('text', '')}) needs its number {', '.join(missing)} visible on a panel that teaches "
                f"it: {where}")
        elif _key_numbers(claim) and COMPARATIVE_RE.search(str(claim.get("text", ""))) and not _comparison_shown(conveying):
            home = max(conveying, key=lambda p: (p.beat == "proof", p.kind == "board", p.n))
            out.setdefault(home.n, []).append(
                f"claim {cid} ({claim.get('text', '')}) is a comparison, but no panel that teaches it draws one: show it on a bar "
                "chart (the paper's result, the strongest earlier system and the field, from one table column) or a table")
        elif thin := _thin_claim(sb, claim, conveying):
            want = _claim_words(str(claim.get("text") or ""))
            home = max(conveying, key=lambda p: (_shared_words(want, _claim_words(visible_text(p))), p.kind == "scene", -p.n))
            out.setdefault(home.n, []).append(
                f"claim {cid} is named but not taught: the panels that teach it never say {', '.join(thin[:8])} "
                f"({claim.get('text', '')}). Say what the claim says, every part and qualifier, in plain words on this panel; "
                "when it does not fit, cut the words that repeat or decorate, or put a part on this panel's board")
    if is_novice(sb):
        for n, hints in out.items():
            out[n] = [h.replace("in plain words on this panel; when it does not fit, cut the words that repeat or decorate, or put a part on this panel's board",
                                "in plain words across successive teaching panels after their prerequisites; preserve all details without overloading a panel")
                        .replace("in this panel's visible text or board and add", "across suitable successive teaching panels after prerequisites and add") for h in hints]
    return out


# words a claim uses that a panel need not repeat to teach it: reporting verbs, the study itself, function words
CLAIM_FILLER = frozenset(
    "paper papers study studies author authors researcher researchers show shows showed shown find finds found finding "
    "findings propose proposes proposed present presents result results work works worked working approach method methods "
    "new use uses used using also based via provide provides provided demonstrate demonstrates suggest suggests existing "
    "their they this that these such which while when where more most both other well but how will very too then being "
    "make making makes get gets got about around across like all any can could would should may might into onto over under "
    "analyze analyzed analyzes track tracked evaluate evaluated compare compared argue argues exactly overall including "
    "include includes one two three four five six seven eight nine ten eleven twelve".split())   # numbers: the key-number check


def _claim_words(text: str) -> set:
    return {_stem(w) for w in _content_tokens(text) if not w[0].isdigit()} - CLAIM_FILLER


def _thin_claim(sb: StoryboardV2, claim: dict, conveying: list[PanelV2]) -> list[str]:
    """The words of an abstract claim that no panel teaching it says, in any form or by a glossary term's other name ("RL"
    says reinforcement learning), when they are most of the claim; [] otherwise. A claim squeezed into a full scene kept
    its id and lost its content (LongZero's "training data that is long, diverse, challenging and verifiable" became "it
    needs good data", the delivery paper's "parcels returned to merchants" vanished), and readers could not answer."""
    want = _claim_words(str(claim.get("text") or ""))
    if len(want) < 4 or not conveying:
        return []
    forms = [_term_forms(g["term"]) for g in _as_list(sb.brief.get("glossary")) if isinstance(g, dict) and g.get("term")]
    said: set = set()
    for p in conveying:
        text = visible_text(p)
        said |= _claim_words(text)
        for names in forms:
            if any(_term_re(f).search(text) for f in names):
                said |= set().union(*(_claim_words(f) for f in names))
    missing = sorted(w for w in want if not any(_same_word(w, y) for y in said))
    if len(missing) <= 0.5 * len(want):
        return []
    quote = _claim_words(str(claim.get("quote") or ""))   # the abstract's own sentence: the brief's text is one paraphrase
    if len(quote) >= 4 and _shared_words(quote, said) >= 0.5 * len(quote):
        return []   # OnQ's panel said "We identify two challenges in closing this gap" and was flagged for "barrier, core"
    return missing


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+")


def scrub_repeats(sb: StoryboardV2) -> list[str]:
    """Last line of defense after the fix rounds: a scene's line that repeats an earlier panel's line word for word goes, and
    so does a scene whose picture repeats an earlier one once it has nothing new to say: no words left, or each line left
    is mostly said by its twin with every number (a fuller wording of a twin line replaces it there, so a qualifier or a
    number is never lost); their claims join the panel said first. A late quiz revision copied panels ("Most prior
    last-mile performance research..." twice), the lint flagged them with no round left, and readers took the book for a
    misprint. A line that brings the running example back on a new picture stays (the payoff resolves the hook's
    question), and board panels keep their bubbles (a board needs a reaction). Returns what was removed (for the job
    report)."""
    removed: list[str] = []
    lines: dict[str, int] = {}
    pictures: dict[str, int] = {}
    by_n = {p.n: p for p in sb.panels}
    kept: list[PanelV2] = []
    key = lambda text: " ".join(_content_tokens(text or ""))
    stems = lambda text: {_stem(w) for w in _content_tokens(text or "") if not w[0].isdigit()}
    words = lambda text: {_stem(w) for w in _content_tokens(text or "")}   # numbers count
    example = _example_tokens(sb)
    for p in sb.panels:
        said = [b.text for b in p.dialogue] + ([p.narration] if p.narration else [])
        if p.kind != "scene" or p.beat in ("claims", "recap"):
            for text in said:
                if len(key(text).split()) >= 3:
                    lines.setdefault(key(text), p.n)
            kept.append(p)
            continue
        picture = key(p.visual)
        copy_picture = bool(picture) and picture in pictures

        def repeats(text: str) -> bool:
            k = key(text)
            if len(k.split()) < 3 or lines.get(k, p.n) == p.n:
                return False
            return copy_picture or len(example & stems(text)) < 3   # the running example called back on a new picture stays
        first = None
        dialogue = []
        for b in p.dialogue:
            if repeats(b.text):
                first = lines[key(b.text)]
                removed.append(f"{p.n}.{b.speaker}: {b.text}")
                continue
            dialogue.append(b)
        narration = p.narration
        if narration and repeats(narration):
            first = lines[key(narration)]
            removed.append(f"{p.n}.narration: {narration}")
            narration = ""
        target = by_n.get(first if first is not None else pictures.get(picture, -1))
        earlier = by_n.get(pictures.get(picture, -1))
        rest = [b.text for b in dialogue] + ([narration] if narration else [])
        mostly_said, swaps = False, []
        if copy_picture and first is not None and earlier is not None and rest:
            have = words(" ".join([b.text for b in earlier.dialogue] + [earlier.narration]))
            for text in rest:   # each line on its own: a short new sentence or a number is something new to say
                twin = next((b for b in earlier.dialogue if words(b.text) < words(text)), None)
                if twin is None and earlier.narration and words(earlier.narration) < words(text):
                    twin = earlier   # the narration
                if twin is not None:
                    swaps.append((twin, text))
                elif len(words(text) & have) < 0.7 * len(words(text)) or {w for w in words(text) - have if w[0].isdigit()}:
                    break
            else:
                mostly_said = True
        if (not dialogue and not narration and (first is not None or copy_picture)) or mostly_said:
            for twin, text in swaps:   # the fuller wording of a line the twin already says
                if twin is earlier:
                    earlier.narration = text
                else:
                    twin.text = text
                lines.setdefault(key(text), earlier.n)
            target = earlier if mostly_said else target
            if target is not None:
                target.claims = list(dict.fromkeys(list(target.claims) + list(p.claims)))
            removed.append(f"panel {p.n}: a repeat of panel {target.n if target is not None else '?'}")
            continue
        for text in rest:   # registered only once the panel is kept (a dropped copy never owns a line)
            if len(key(text).split()) >= 3:
                lines.setdefault(key(text), p.n)
        if first is not None and target is not None:   # what the removed lines taught is taught where it was said first
            target.claims = list(dict.fromkeys(list(target.claims) + list(p.claims)))
        p.dialogue, p.narration = dialogue, narration
        if picture:
            pictures.setdefault(picture, p.n)
        kept.append(p)
    if len(kept) != len(sb.panels):
        sb.panels = kept
    if removed:
        print(f"[StoryboarderV2] removed {len(removed)} repeated lines or panels: {removed[:4]}")
    return removed


def scrub_unsupported(sb: StoryboardV2, paper_text: str) -> list[str]:
    """Last line of defense after the fix rounds: a bubble or narration sentence whose number the paper still does not
    support is removed (an empty bubble goes). Returns what was removed (for the job report)."""
    evidence = _optional("evidence")
    if evidence is None:
        return []
    allowed = _illustrative_example(sb)
    removed = []
    cite = _optional("ledger")
    for p in sb.panels:
        def clean(text: str) -> str:
            keep = []
            for sentence in SENTENCE_SPLIT_RE.split(text or ""):
                try:
                    bad = evidence.verify_text_numbers(sentence, paper_text, allowed=[allowed, _badge_text(p)])
                except Exception:
                    bad = []
                if cite is not None and cite.REF_RE.search(sentence):   # a citation no fix round resolved: never drawn as "@T2.r9.c1"
                    bad = bad or ["unresolved table citation"]
                if bad:
                    removed.append(f"panel {p.n}: {sentence}")
                else:
                    keep.append(sentence)
            return " ".join(keep).strip()
        for b in p.dialogue:
            b.text = clean(b.text)
        p.dialogue = [b for b in p.dialogue if b.text]
        p.narration = clean(p.narration)
        if p.kind == "board" and _is_diagram(p.board) and p.board.get("elements"):
            p.board = _scrub_diagram(p, paper_text, allowed, removed)
    if removed:
        print(f"[StoryboarderV2] removed {len(removed)} sentence(s) with numbers the paper does not support: {removed[:3]}")
    return removed


def _scrub_diagram(p: PanelV2, paper_text: str, allowed: str, removed: list) -> dict:
    """A diagram without the labels whose number the paper still does not back (the designer's check sent them back,
    and a redesign kept one anyway): a free text or an arrow's label goes, a box keeps its shape without the text, and
    a table citation never drawn as "@T2.r5.c1". The rest of the drawing stays."""
    evidence, cite = _optional("evidence"), _optional("ledger")
    board = copy.deepcopy(p.board)
    keep = []
    for i, e in enumerate(board.get("elements") or []):
        for key in ("text", "label"):
            text = e.get(key) if isinstance(e, dict) else None
            if not isinstance(text, str) or not text:
                continue
            try:
                bad = evidence.verify_text_numbers(text, paper_text, allowed=[allowed]) if evidence is not None else []
            except Exception:
                bad = []
            if cite is not None and cite.REF_RE.search(text):
                bad = bad or ["unresolved table citation"]
            if bad:
                removed.append(f"panel {p.n} (diagram label): {text}")
                e.pop(key, None)
        if not (e.get("type") == "text" and not e.get("text")):
            keep.append(e)
    board["elements"] = keep
    return board


def _panel_items(value) -> list[dict]:
    """The panel objects of a reply: a list, one bare panel, or a dict keyed by panel number ({"1": {...}, "2": {...}})."""
    if isinstance(value, dict):
        if any(k in value for k in ("dialogue", "kind", "visual", "board", "narration")):
            return [value]
        out = []
        for key, item in value.items():
            if isinstance(item, dict):
                digits = re.sub(r"\D", "", str(key))
                out.append(item if "n" in item or not digits else {**item, "n": int(digits)})
        return out
    return [x for x in _as_list(value) if isinstance(x, dict)]


def _bubble_item(value):
    """A speech bubble as {"speaker", "text"}: replies sometimes send "kumo: text" or ["kumo", "text"] (a line without a
    speaker goes to the panel's first character)."""
    if isinstance(value, str):
        m = re.match(r"\s*([^:\n]{1,30}):\s*(\S.*)$", value, re.S)
        return {"speaker": m.group(1), "text": m.group(2)} if m and len(m.group(1).split()) <= 3 else {"speaker": "", "text": value}
    if isinstance(value, (list, tuple)) and len(value) == 2 and all(isinstance(x, str) for x in value):
        return {"speaker": value[0], "text": value[1]}
    return value


def _panel_from_reply(item: dict, default_n: int) -> PanelV2:
    """A v3 panel from a model reply: a board makes it a board panel unless the reply explicitly says "scene". A board sent
    as a JSON string and bubbles sent as strings or pairs are read too."""
    item = dict(item)
    if isinstance(item.get("board"), str) and item["board"].strip().startswith("{"):
        try:
            item["board"] = json.loads(item["board"])
        except json.JSONDecodeError:
            pass
    dialogue = item.get("dialogue")
    if isinstance(dialogue, dict) and "text" not in dialogue:   # {"kumo": "line", "papi": "line"}
        dialogue = [{"speaker": k, "text": v} for k, v in dialogue.items()]
    if dialogue is not None:
        item["dialogue"] = [_bubble_item(b) for b in _as_list(dialogue)]
    for key in ("narration", "title", "scene", "visual", "shows"):   # a list of sentences is one text, never "['...']"
        if isinstance(item.get(key), list):
            item[key] = " ".join(str(x).strip() for x in item[key] if isinstance(x, (str, int, float)) and str(x).strip())
    p = PanelV2.from_dict(item, default_n)
    declared = str(item.get("kind") or item.get("type") or "").strip().lower()
    if isinstance(item.get("board"), dict) and item["board"] and declared != "scene":
        p.kind = "board"
    return p


def _valid_boards(panels: list[PanelV2], paper_text: str, among: Optional[set[int]] = None) -> frozenset:
    """Numbers of the board panels (of ``among``) whose board passes every check."""
    return frozenset(p.n for p in panels if (among is None or p.n in among) and p.kind == "board" and isinstance(p.board, dict)
                     and p.board and not board_issues(p.board, paper_text))


def _normalize_beat(beat: str, previous: str) -> str:
    b = re.sub(r"[\s-]+", "_", (beat or "").strip().lower())
    if b in BEATS:
        return b
    if b in BEAT_ALIASES:
        return BEAT_ALIASES[b]
    return next((x for x in BEATS if x in b), previous)


def _normalize_v3_panels(panels: list[PanelV2], cast: Cast, claims: list[dict], ledger=None) -> None:
    """The v2 clean-up without its two-bubble cap (lint judges the budget; nothing is cut), plus beats, the chapters they
    map to, valid claim ids only, no drawn labels, ledger citations resolved (services.ledger: "@T2.r9.c1" becomes the
    cell's exact number), normalized boards, and at most the speakers on a crowded board panel."""
    _normalize_panels(panels, cast, max_bubbles=None)
    valid = {c.get("id") for c in claims}
    previous = BEATS[0]
    binder = _optional("ledger") if ledger else None
    for p in panels:
        p.beat = previous = _normalize_beat(p.beat, previous)
        p.chapter = BEAT_CHAPTER[p.beat]
        p.claims = [c for c in p.claims if c in valid]
        p.labels = []   # v3 art is wordless; every word is typeset or drawn on a board by code
        if not SCENE_SHOWS:
            p.shows = ""
        if binder is not None:
            for b in p.dialogue:
                b.text = binder.bind_text(b.text, ledger)
            p.narration = binder.bind_text(p.narration, ledger)
            if isinstance(p.board, dict):
                p.board = binder.bind_board(p.board, ledger)
        if p.board is not None:
            p.board = _normalized_board(p.board)
        if p.kind == "board" and len(p.characters) > 2:
            speakers = list(dict.fromkeys(b.speaker for b in p.dialogue))
            if len(speakers) <= 2:
                p.characters = (speakers + [c for c in p.characters if c not in speakers])[:2]
                p.emotions = {k: v for k, v in p.emotions.items() if k in p.characters}


# a recap title or subtitle that certifies every line ("Verified Results", "All supported by the study data", "All claims
# tested in this story", "已验证的结论") over lines that only got a bullet; "tested" alone states a scope ("On tested
# locomotion tasks", "Findings on tested benchmarks") and stays
CERTIFYING_TITLE_RE = re.compile(r"\b(?:verified|validated|confirmed|proven|checked|supported)\b|"
                                 r"\ball\s+(?:the\s+)?(?:claims?|results?|findings?)\s+(?:were\s+|are\s+)?tested\b|"
                                 r"\btested\s+in\s+this\s+(?:story|manga|work|study)\b|"
                                 r"已?(?:验证|证实|核实|核查|证明|确认|支持)", re.I)
HONEST_TITLE = {"en-US": "What the paper showed", "zh-CN": "论文展示了什么"}
EXHIBIT_KINDS = ("bar", "table")   # the boards that draw the paper's numbers: a recap line's tick needs one of them


def _exhibits(sb: StoryboardV2) -> list[PanelV2]:
    """The story's charts and tables: the board panels that draw the paper's numbers (not the claims board or recap)."""
    return [p for p in _teaching_panels(sb.panels) if p.kind == "board" and isinstance(p.board, dict)
            and str(p.board.get("kind") or "").lower() in EXHIBIT_KINDS]


def _recap_lists(sb: StoryboardV2) -> list[PanelV2]:
    """The recap's list boards (a long abstract's recap takes two)."""
    return [p for p in sb.panels if p.beat == "recap" and isinstance(p.board, dict)
            and str(p.board.get("kind") or "").lower() == "list" and not p.board.get("open")]


def _drawn_badge(p: PanelV2) -> str:
    """The gap a bar chart's badge draws ("+4.6"), "" for another board."""
    boards = _optional("boards")
    if boards is None or p.board.get("kind") != "bar":
        return ""
    try:
        badge = boards.delta_badge(p.board)
    except Exception:  # a checker bug never blocks the script
        return ""
    return str(badge["text"]) if badge else ""


DIAGRAM_TRIES = 2        # a design, and one redesign with what the lint and the checkers found in it
DIAGRAM_PARALLEL = 4


def _is_diagram(board) -> bool:
    return isinstance(board, dict) and str(board.get("kind") or "").lower() == "diagram"


def _diagram_key(p: PanelV2) -> str:
    """What a diagram is drawn for: its brief (what the writer asked it to show). A drawing made for another brief is drawn
    again. Neither the panel's words nor the board's title are part of it (the program draws the title above the drawing):
    a fact-check's "may" in a bubble, or its fix of the title, threw away the checked drawing and its corrected labels, and
    the new design after the last check shipped unread (round 23's reviews); a drawing that no longer matches its words is
    the board check's to find, and its finding goes to the designer."""
    b = p.board if isinstance(p.board, dict) else {}
    return hashlib.sha1(json.dumps([b.get("brief", "")], ensure_ascii=False).encode()).hexdigest()[:12]


def _drawing_flags(sb: StoryboardV2, found: dict) -> dict[int, list[str]]:
    """The findings of a check that are about a designed diagram's drawing ({n: [problem]}): they go to the designer at
    once, while the panel numbers still hold (findings kept for later went to the wrong diagram after a fix round added
    a panel, and outlived a quiz revision that was undone)."""
    return {n: list(v) for n, v in (found or {}).items()
            if v and any(p.n == n and p.kind == "board" and _is_diagram(p.board) for p in sb.panels)}


DIAGRAM_LAYOUT = "layout: "
LABEL_LENGTH = "keep each label within"   # services.diagram's finding on a label's length: a move cannot fix it   # a request to move things only: every text stays as it is (a correction made it longer)
DIAGRAM_NUMBER = "the label "  # a number or citation on a drawing that the paper does not back: worse than any layout problem
# a board-check finding the designer has already redrawn the diagram for: the writer fixes words only (a new brief threw
# the fresh drawing away and it was drawn again from scratch without the finding, round 23c's review)
DIAGRAM_REDRAWN = (" (the diagram was just redrawn for this finding: change the panel's words only if they need it, and keep "
                   "the board's brief, or the drawing is thrown away)")


def diagram_prompt(sb: StoryboardV2, p: PanelV2, problems: Optional[list[str]] = None, previous: Optional[dict] = None,
                   paper_text: str = "") -> str:
    """The diagram designer (the check model, with the paper): one board's elements from the panel's words, its brief
    and the running example, on the 1200 x 600 canvas of services.diagram. ``problems`` (the layout lint, numbers the
    paper lacks, a checker's findings) come back with the ``previous`` design for one redesign."""
    panels = {x.n: x for x in sb.panels}
    words = lambda x: [f"{b.speaker}: {b.text}" for b in x.dialogue] + ([f"narration: {x.narration}"] if x.narration else [])
    earlier = [{"n": k, "words": words(panels[k]), **({"board": board_text(panels[k].board)} if panels[k].board else {})}
               for k in sorted(panels) if p.n - 3 <= k < p.n]
    claims = [c.get("text") for c in sb.abstract_claims if c.get("id") in p.claims]
    board = p.board if isinstance(p.board, dict) else {}
    redo = ""
    if problems:
        redo = ("\n\nYOUR PREVIOUS DESIGN (fix it; keep what works):\n" + json.dumps(previous or {}, ensure_ascii=False)
                + "\nPROBLEMS FOUND IN IT (fix every one):\n" + "\n".join(f"- {x}" for x in problems))
    return f"""You design ONE diagram for panel {p.n} of an educational manga that teaches a research paper to newcomers. A program draws it on a big whiteboard next to the characters, in a hand-drawn marker style; the panel's words stay in the speech bubbles beside it. Looking at the diagram, a reader must SEE the mechanism, the structure or the difference the words talk about.

Draw the paper's actual mechanism on the running example, with the real things it is about (the tokens of a sentence, a question and its answer, documents, a model, layers, a teacher's and a student's choices, calls in a day, parcels on two days) as simple shapes with short labels. The arrangement is the point: a row of tokens with the order they are filled in, a branch where the choices split, two sides that differ in exactly what the words compare and are the same in everything else, a stack of layers with a bypass arrow, a before and an after, a queue at two times of the day. Each thing stays at its real place (the tokens of a sentence in sentence order, the layers of a network in stack order): an order of events is shown by a small step number in a circle (radius about 36) beside each thing, never by moving the things or by writing the number into a label ("2 180" reads as 2180). No metaphors and no props (no balloons, erasers, keys, berries): the real thing, simplified. Write the example's real content on the things (the real words of its question, the real tokens, the names the paper uses), never a placeholder (token 1, response A, document, item 2) where the example has a real one. Label things with the words' own terms, so a reader matches the bubbles to the drawing. Never draw made-up data: bars or a plotted line with values appear only with the paper's own numbers on them, and a result without them is drawn as the things that were compared or measured (the results go on the story's chart boards); to show how a model spreads probability over a few next tokens, short bars without values are fine, each with its token written in it. Show only what this panel and the earlier ones say: never a step, result or name the story has not reached. A number only when the paper gives it, written exactly as the paper writes it (never computed, rounded or invented; small counts such as 1, 2, 3 for an order are fine); a made-up example is shown as one, once, by a small grey "e.g." note beside it (never inside a box or a token).

PANEL {p.n} (beat: {p.beat}): {json.dumps(words(p), ensure_ascii=False)}
WHAT THE WRITER ASKED THE DIAGRAM TO SHOW: {json.dumps(board.get("brief") or board.get("title") or "", ensure_ascii=False)}
BOARD TITLE AND SUBTITLE (the program draws them above the diagram, so never repeat them inside it): {json.dumps(board.get("title") or "", ensure_ascii=False)}{(" / " + json.dumps(board.get("subtitle"), ensure_ascii=False)) if board.get("subtitle") else ""}
THE ABSTRACT CLAIM IT HELPS TEACH (context; the reader never sees it): {json.dumps(claims, ensure_ascii=False)}
RUNNING EXAMPLE OF THE BOOK (every diagram of the book draws this same case): {json.dumps(sb.running_example, ensure_ascii=False)}{_example_lines(sb, paper_text, board)}
EARLIER PANELS (what the reader has seen just before): {json.dumps(earlier, ensure_ascii=False)}

ELEMENTS on a 1200 x 600 canvas (x to the right, y down):
- box: {{"type": "box", "x": 80, "y": 200, "w": 220, "h": 100, "text": "short label", "tone": "ink", "fill": "none", "dashed": false, "size": 38, "bold": false}}  (x, y = top-left corner; the text is centred inside and wrapped)
- text: {{"type": "text", "x": 600, "y": 60, "text": "short label", "tone": "ink", "size": 38, "anchor": "middle", "bold": false}}  (one line; y is its vertical middle; anchor start, middle or end)
- arrow or line: {{"type": "arrow", "from": [300, 250], "to": [420, 250], "via": [], "curved": false, "tone": "ink", "dashed": false, "heads": "end", "label": "short"}}  (via = corner points; curved bends it through them; the label sits above the middle of its longest segment, or right of it when the segment is vertical; on a curved arrow, beside the middle of the curve)
- circle: {{"type": "circle", "x": 600, "y": 400, "r": 40, "text": "+", "tone": "red", "fill": "none"}}
- mark: {{"type": "mark", "x": 600, "y": 480, "mark": "check", "size": 50, "tone": "red"}}  (a check or a cross)
Tones: "ink" for text and lines, "grey" for secondary things, ONE accent "red" for what this panel is about, "blue" for the other side of a comparison; when the old way and the paper's method appear together, the method is what the panel is about: red, and the old way blue, as on every diagram of the book. Fills: "none", "tint" (a pale fill), "solid" (white text on the tone).

LAYOUT (a program measures every text and sends overlaps back): the board in the panel is short and wide, and a reader may be on a phone, so every text is big: font sizes 32 to 56 (38 by default); at most 12 texts (step numbers count) of at most 5 words each, no sentences; a Latin character is about 0.55 x the size wide and a Chinese one 1 x the size; keep 24 units between a text and any other text, line or box edge; a box holds its text with room to spare; arrows start and end at box edges and never cross a text; stay 30 units inside the canvas; use the whole canvas: a few large things rather than many small ones.{redo}

Output JSON only: {{"elements": [...]}}

PAPER TEXT:
{paper_text}"""


def _example_lines(sb: StoryboardV2, paper_text: str = "", board: Optional[dict] = None) -> str:
    """The running example's parts and what each way does to it, for the diagram designer (brief_example), and the paper's
    own figures of the case and of the panel's step (services.paper_figures). Diagrams designed one by one from the
    one-sentence example drew placeholders or other details of the case on each board; given a flat list of parts, the
    designer of round 25 wrote all eight candidate tokens of one step on every diagram, strung into a sentence as if they
    were the student's path."""
    example = sb.brief.get("running_example") if isinstance(sb.brief.get("running_example"), dict) else {}
    out = ""
    if example.get("text") and not example.get("from_paper", True):
        out += ("\nTHIS CASE IS MADE UP (the paper shows no instance): mark it once with the small grey \"e.g.\" note, and "
                "never draw one of its values as a result")
    if _example_parts(sb):
        out += ("\nITS PARTS IN THEIR REAL WORDS, each with what it is in the case (draw only the parts that this panel's "
                "step acts on, each as what it is: alternatives at one step stand side by side as options, never strung "
                "into a sentence; write each in its real words, cut to its key words when long, never a placeholder "
                "instead): " + json.dumps([_part_line(t, r) for t, r in _part_pairs(example)], ensure_ascii=False))
    figures = _optional("paper_figures")
    if figures is not None and paper_text:
        board = board or {}
        own = figures.figure_refs(str(example.get("source") or ""))[:1] if example.get("from_paper") else []
        named = [n for n in figures.figure_refs(" ".join(str(board.get(k) or "") for k in ("brief", "title", "subtitle"))) if n not in own]
        for n in own:
            if block := figures.figure_block(paper_text, n):
                out += ("\nTHE PAPER'S OWN FIGURE OF THIS CASE (redraw what this panel needs of it, simplified, in its own "
                        "words and order; a value it gives only as an order may set the heights of short bars without values "
                        "where the panel shows how probability is spread, never a written number): " + block)
        for n in named[:2]:
            if block := figures.figure_block(paper_text, n):
                out += ("\nA FIGURE OF THE PAPER THAT THE WRITER NAMED FOR THIS STEP (it may show another case than the "
                        "running example: take its structure, and draw the running example on it): " + block)
    for key, what in (("old_way", "WHAT THE OLD WAY DOES TO IT"), ("new_way", "WHAT THE PAPER'S METHOD DOES TO IT")):
        if isinstance(example.get(key), str) and example[key].strip():
            out += f"\n{what} (draw only the part this panel reaches): {json.dumps(example[key], ensure_ascii=False)}"
    return out


def _diagram_fallback(board: dict) -> dict:
    """A diagram the designer could not draw: a flow board of its brief's parts (never a blank board)."""
    brief = str(board.get("brief") or board.get("title") or "").strip()
    brief = re.sub(r"\s*\(?\b(?:as (?:in|shown in) |like |see )?(?:Figures?|Figs?\.?)\s*\d{1,3}[a-z]?(?:\.[a-z])?(?:\s*(?:,|and|&|–|-)\s*\d{1,3}[a-z]?)*\)?", "", brief, flags=re.I).strip(" ,;:")
    parts = [x.strip(" .;") for x in re.split(r"(?<=[.;:])\s+|,\s+(?:then|and then)\s+|\s+then\s+", brief) if x.strip(" .;")]
    parts = [" ".join(x.split()[:8]) for x in parts][:5]
    if len(parts) < 2:
        words = brief.split()
        parts = [" ".join(words[:8]) or "?", " ".join(words[8:16]) or str(board.get("title") or "")]
    out = {"kind": "flow", "title": board.get("title") or "", "steps": [x for x in parts if x]}
    if board.get("subtitle"):
        out["subtitle"] = board["subtitle"]
    return out


def recap_evidence_prompt(sb: StoryboardV2) -> str:
    """Which of the story's charts and tables shows each recap line: the legend under the recap says a tick means a chart
    or table of the manga shows the line. A rule on words alone ticked "Gate adds non-linearity" beside a metaphor and
    "Depth up to 1024 layers" beside a chart of 4 and 64 layers (round 19's picture audits found wrong marks in 6 of 8
    recaps), and matching words to charts missed "Found a strong Artificial Hivemind effect" beside its two charts."""
    hidden = {"quote", "paper_label", "cell", "cell_ids", "unbound", "from", "weights", "illustrative"}   # not what is drawn

    def drawn(p: PanelV2) -> dict:
        spec = {k: v for k, v in p.board.items() if k not in hidden}
        for key in ("bars", "rows"):
            if isinstance(spec.get(key), list):
                spec[key] = [{k: v for k, v in x.items() if k not in hidden} if isinstance(x, dict) else x for x in spec[key]]
        out = {"n": p.n, "board": spec, "words": [f"{b.speaker}: {b.text}" for b in p.dialogue] + ([f"narration: {p.narration}"] if p.narration else [])}
        badge = _drawn_badge(p)
        if badge:
            out["badge"] = badge
        return out
    lines = [str(x) for p in _recap_lists(sb) for x in _as_list(p.board.get("items"))]
    return f"""A manga that teaches a research paper ends with a recap board: the paper's claims again, one per line. A line gets a tick when one of the manga's charts or tables shows the evidence for it, and a plain dot otherwise; the board's legend reads "tick: a chart in this manga shows it". Decide which chart or table, if any, shows each line.
A chart or table shows a line when it draws the result the line states: the line's numbers (the same values at the line's precision, or the gap the chart's badge draws), or the comparison or effect the line names (which one scores higher, what rises or falls). It does not show a line when it draws:
- a narrower case than the line claims (one model's average for "every benchmark at every size"; two depths for "up to 1024 layers"; one benchmark for "all benchmarks"),
- another quantity or another comparison than the line's,
- nothing the line measures: the line is the method's design (what it does, how or why it works), the paper's background or motivation, its data set, or a release of code, unless a chart or table draws exactly the line's numbers.
Judge from what each chart or table draws; the words around it only tell what it is about.

THE MANGA'S CHARTS AND TABLES:
{json.dumps([drawn(p) for p in _exhibits(sb)], ensure_ascii=False)}

THE RECAP LINES:
{json.dumps([{"i": i, "line": x} for i, x in enumerate(lines)], ensure_ascii=False)}

Output JSON only: {{"lines": [{{"i": 0, "panels": [8]}}, {{"i": 1, "panels": []}}]}}, every line once; "panels" lists the panel numbers whose chart or table shows the line, [] when none does."""


def _honest_recap(sb: StoryboardV2, retitle: bool = True) -> None:
    """The recap ticks a line only when a chart or table of the story shows it, as the board's legend says ("checked" on
    the list board; the other lines get a plain dot). The check model decides which chart shows which line
    (StoryboarderV2.recap_evidence, kept in sb.brief["recap_evidence"]); a line it has not judged (a relayout of an older
    job, a line rewritten since) gets its tick when the story's charts and tables draw all of its numbers (a value, a
    label, the gap a badge draws), or, without numbers, when a chart or table is bound to the claim it restates; never
    for words alone. Earlier rules ticked a line a bubble had said ("Depth up to 1024 layers" beside a chart of 4 and
    64 layers) or a metaphor had explained ("Gate adds non-linearity"), under a legend that promised evidence.
    ``retitle`` (the English script only): replace a title or subtitle that certifies every line. Works on a script in
    any language (a job's relayout re-applies it)."""
    exhibits = _exhibits(sb)
    # the verdict, not the panel numbers it names: _finish_v3 numbers the panels afresh after the build's judgment
    entries = [e for e in _as_list(sb.brief.get("recap_evidence")) if isinstance(e, dict) and isinstance(e.get("line"), str)]
    judged = {e["line"].strip(): bool(e.get("shown")) for e in entries}
    # a line reworded since the check read it (the translation's fact-check, backported: "Existing QAT trains only on..."
    # became "Existing QAT typically trains only on...") keeps the verdict of the line it replaced, in the same place
    # with the same numbers; a changed number is a new claim, judged by the rule below
    now = [str(x).strip() for p in _recap_lists(sb) for x in _as_list(p.board.get("items"))]
    if len(entries) == len(now):
        for e, line in zip(entries, now):
            old = e["line"].strip()
            if line not in judged and old not in now and _number_keys(old) == _number_keys(line):
                judged[line] = bool(e.get("shown"))
    drawn: set = set()
    bound: set = set()
    for p in exhibits:
        drawn |= _number_keys(board_text(p.board)) | _number_keys(_drawn_badge(p))
        bound |= set(p.claims)
    values = set()
    for key in drawn:
        try:
            values.add(abs(float(key)))
        except ValueError:
            pass

    def seen(written: str, key: str) -> bool:   # "23pp" is the chart's 23.4 at the line's precision
        try:
            v = abs(float(key))
        except ValueError:
            return key in drawn
        decimals = re.search(r"\.(\d+)", written)
        places = len(decimals.group(1)) if decimals else 0
        return any(abs(v - x) <= 0.5 * 10 ** -places + 1e-9 for x in values)

    def claim_words(c: dict) -> set:
        return {_stem(w) for w in _content_tokens(f"{c.get('text') or ''} {c.get('quote') or ''}")}

    def shown(i: int, item: str, before: set) -> bool:
        if item.strip() in judged:
            return judged[item.strip()]
        text = MARKER_RE.sub(" ", item)
        found = _number_tokens(text)   # digits only: "two decades" and "五十万" are words, not results
        # a year (WMT 2014) or a setting ("2-bit", "9-benchmark") says what was measured, not the result: "Wins every
        # 2-bit benchmark at every size" kept its tick beside one model's 2-bit average because a chart drew a 2
        numbers = [(w, k) for w, k in found if not YEAR_RE.fullmatch(w)
                   and not (re.fullmatch(r"\d+", w) and re.search(rf"(?<![\w.]){re.escape(w)}(?:/\d+)*-[^\W\d_]", text))]
        if numbers:
            return all(seen(w, k) for w, k in numbers)
        if any(not YEAR_RE.fullmatch(w) for w, _ in found):   # settings only: no result a chart draws
            return False
        want = {_stem(w) for w in _content_tokens(item)}
        if not want:   # a line the rule cannot read (a translation's words) keeps the English script's mark
            return i in before
        scores = [(_key_words(want, claim_words(c)), c.get("id")) for c in sb.abstract_claims if isinstance(c, dict)]
        top = max((k for k, _ in scores), default=0)
        return top >= max(2, len(want) / 3) and any(cid in bound for k, cid in scores if k == top)

    for p in _recap_lists(sb):
        b = p.board
        items = b.get("items") or []
        before = set(b["checked"]) if isinstance(b.get("checked"), list) else set(range(len(items)))
        ticked = [i for i, item in enumerate(items) if shown(i, str(item), before)]
        if len(ticked) < len(items):
            p.board = {**b, "checked": ticked}
        else:
            p.board = {k: v for k, v in b.items() if k != "checked"}
        # "Confirmed findings", "Claims supported by the presented evidence": over bullets, and over ticks too (the story
        # showed the paper's evidence, it confirmed nothing)
        if retitle and CERTIFYING_TITLE_RE.search(str(b.get("title") or "")):
            p.board["title"] = HONEST_TITLE["en-US"]
        if retitle and CERTIFYING_TITLE_RE.search(str(b.get("subtitle") or "")):
            p.board.pop("subtitle", None)


def copy_recap_marks(english: StoryboardV2, other: StoryboardV2, language: str) -> None:
    """A translation's recap gets the English recap's marks line by line (judging the translated words read "五十万" as the
    number 5 and ticked lines the English recap bulleted), and its title and subtitle follow the English ones: gone or
    made honest where the English one was (a relayout judged "本故事中检验过的所有主张" by its own words and kept it over
    the dots while the English "All claims tested in this story" went)."""
    def recaps(sb: StoryboardV2) -> list:
        return [p for p in sb.panels if p.beat == "recap" and isinstance(p.board, dict) and p.board.get("kind") == "list"]
    for src, dst in zip(recaps(english), recaps(other)):   # a long abstract's recap takes two boards
        if len(src.board.get("items") or []) != len(dst.board.get("items") or []):
            continue
        board = {k: v for k, v in dst.board.items() if k != "checked"}
        if isinstance(src.board.get("checked"), list):
            board["checked"] = list(src.board["checked"])
        if not src.board.get("subtitle") or CERTIFYING_TITLE_RE.search(str(board.get("subtitle") or "")):
            board.pop("subtitle", None)
        if src.board.get("title") == HONEST_TITLE["en-US"] or CERTIFYING_TITLE_RE.search(str(board.get("title") or "")):
            board["title"] = HONEST_TITLE.get(language, HONEST_TITLE["en-US"])
        dst.board = board


def _finish_v3(sb: StoryboardV2) -> None:
    """Panels 1..N and chapters from the beats (keeping chapter titles that were already translated). Only teaching
    scenes keep a "shows": a board (its board is the exhibit) and the bet's settlement drawn without the book's symbols
    get no idea check and no extra try."""
    symbols = [{_stem(w) for w in _content_tokens(sym)} - SYMBOL_GENERIC for _, sym in _picture_symbols(sb)]
    for i, p in enumerate(sb.panels, start=1):
        p.n = i
        p.chapter = BEAT_CHAPTER.get(p.beat, p.chapter)
        if isinstance(p.board, dict):
            p.board.pop("unbound", None)
        if p.shows and (p.kind == "board" or (_settles_bet(p) and not any(
                _uses_symbol({_stem(w) for w in _content_tokens(p.shows)}, w) for w in symbols))):
            p.shows = ""
    titles = {c.id: c.title for c in sb.chapters}
    sb.chapters = _chapters(sb.panels)
    for c in sb.chapters:
        c.title = titles.get(c.id, c.title)
    sb.version = 3


def _running_example(raw, brief: dict, panels: list[PanelV2], body: str) -> dict:
    """{"text", "from_paper", "source"}: from_paper only when one of the paper's verified examples (or the example itself)
    really shows up in the story."""
    if isinstance(raw, dict):
        text = str(raw.get("text") or raw.get("example") or "").strip()
        claimed = str(raw.get("from_paper", True)).strip().lower() not in ("false", "0", "no", "none", "")
        source = str(raw.get("source") or "").strip()
    else:
        text, claimed, source = str(raw or "").strip(), True, ""
    story = " ".join([text] + [visible_text(p) for p in panels])
    words = _content_tokens(story)
    used = [e for e in _as_list(brief.get("examples")) if isinstance(e, dict) and e.get("text")
            and len(_content_tokens(e["text"]) & words) >= 0.6 * max(1, len(_content_tokens(e["text"])))]
    from_paper = claimed and (bool(used) or quote_in(text, body))
    if from_paper and used and not source:
        source = str(used[0].get("source") or "")
    return {"text": text, "from_paper": bool(from_paper), "source": source if from_paper else ""}


def _board_translation(board: dict, translated: dict, language: str, keep: frozenset) -> dict[str, str]:
    """The translated board strings to apply: in the target script (or code-like, e.g. a model name) and with exactly
    the numbers of the English string."""
    out = {}
    for path, src in _board_strings(board):
        new = translated.get(str(path))
        if isinstance(new, str) and new.strip() and _has_script(new.strip(), language, keep) and _numbers(new) == _numbers(src):
            out[str(path)] = new.strip()
    return out


def _reply_n(item: dict) -> Optional[int]:
    try:
        return int(item.get("n"))
    except (TypeError, ValueError):
        return None


def _mark_halves(items: list[dict], flagged: set[int]) -> None:
    """A split's rest sent without "after" ({"n": n + 1} or {"n": n} again, right after the flagged panel's own item) gets
    "after": that panel's n: read as a first half, it was dropped (its number was no flagged panel) or it took the place
    of the next flagged panel's own rewrite (a flagged number that a later item repeats)."""
    numbers = [_reply_n(x) if x.get("after") is None else None for x in items]
    prev = None
    for i, item in enumerate(items):
        if item.get("after") is not None:
            prev = None
            continue
        n = numbers[i]
        if (prev is not None and n is not None and n in (prev, prev + 1)
                and (n == prev or n not in flagged or n in numbers[i + 1:] or n in numbers[:i])):
            if n != prev and n in flagged:   # prev's rest, or n's own rest sent as {"n": n} again: when it is refused,
                item["_ambiguous"] = n       # both panels stay whole (fix_v3)
            item["after"], prev = prev, None
            continue
        prev = n if n in flagged else None


class StoryboarderV2:
    def __init__(self, text_engine: BaseEngine, fast_engine: Optional[BaseEngine] = None, check_engine: Optional[BaseEngine] = None,
                 translate_engine: Optional[BaseEngine] = None, cache_key: str = ""):
        """``text_engine`` writes (brief, story, fix rounds, quiz revisions) and grades the quiz; ``check_engine`` checks the
        work (brief check, fact-checks, the back-port of a translation's findings) and ``translate_engine`` translates;
        both default to the writer, and a different model family checks without the writer's blind spots (a blind test:
        DeepSeek V4 Pro found 56% of the errors two Seed passes had left, another Seed pass 10%). ``fast_engine`` fits
        lines and reads for the quiz."""
        self.text_engine = text_engine
        self.fast_engine = fast_engine or text_engine
        self.check_engine = check_engine or text_engine
        self.translate_engine = translate_engine or text_engine
        self.cache_key = cache_key   # one per job: its paper-first calls share the provider's prefix cache
        self.skipped: list[dict] = []   # checks that could not run (the job's quality record says so to the reader)
        self.checked: set[str] = set()  # _checked_view of every English panel the fact-check has read
        self.card_checked = ""          # _card_view of the paper card the English fact-check last read
        self.max_panels = V3_PANELS[1]  # the job's story range ceiling (build_v3 sets it): fix-round splits stay near it
        self.usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}

    async def _warm(self, engine: BaseEngine, paper_text: str) -> None:
        """A 1-token call with the paper first, before parallel calls: they then read the paper from the provider's prefix
        cache (cached input costs a fifth on Seed, a thirtieth on DeepSeek) instead of each paying for it in full. It
        thinks, as the checks that follow do: a cached head serves only calls with the same thinking setting (a warm-up
        without thinking left all three brief-check parts uncached, 2026-10-27)."""
        config = GenerationConfig(temperature=0.0, max_tokens=1)
        config.thinking = True
        config.prompt_cache_key = self.cache_key or None
        config.cache_prefix = True
        try:
            response = await engine.generate_text(_messages("", "Reply OK.\n\nPAPER TEXT:\n" + paper_text, paper_text), config)
            self._add_usage(response.usage or {})
        except Exception as e:  # noqa: BLE001  (a failed warm-up only costs the cache)
            _reraise_account(e)

    def _skip(self, stage: str, e: BaseException) -> None:
        """A best-effort check that could not run: kept for the job's quality record, so an unchecked script is never
        shown as a checked one."""
        self.skipped.append({"stage": stage, "error": f"{type(e).__name__}: {str(e)[:200]}"})

    def _add_usage(self, usage: dict) -> None:
        self.usage["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
        self.usage["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)

    async def analyze(self, paper: PaperText) -> dict:
        brief, usage = await _ask_json(self.text_engine, SYSTEM_WRITER, brief_prompt(paper), thinking=True, temperature=0.2, max_tokens=16000, key="key_idea")
        self._add_usage(usage)
        if not brief.get("title") and paper.title:
            brief["title"] = paper.title
        return brief

    async def check_brief(self, brief: dict, paper: PaperText) -> int:
        """Correct (or drop) brief statements the paper does not support, before anything is written from them: errors in
        the brief otherwise reach the panels, the details and the card. A checker of its own reads the items in parallel
        parts (as the fact-check does). Returns the number of fixes (best effort)."""
        items = _brief_items(brief)
        plan_leaves = _beginner_plan_leaves(brief)
        size = FACTCHECK_CHUNK // 2 if self.check_engine is not self.text_engine else max(1, len(items))
        parts = [items[i:i + size] for i in range(0, len(items), size)] or [[]]
        if len(parts) > 1:
            await self._warm(self.check_engine, paper.body)
        results = await asyncio.gather(*(
            _ask_check(self.check_engine, brief_check_prompt(brief, paper, part), thinking=True, temperature=0.1,
                       max_tokens=12000, key="problems", paper=paper.body, cache_key=self.cache_key)
            for part in parts), return_exceptions=True)
        failed = [r for r in results if isinstance(r, BaseException)]
        for e in failed:
            _reraise_account(e)
        if len(failed) == len(results):   # the brief is still usable unchecked
            print(f"[StoryboarderV2] brief check skipped: {failed[0]}")
            self._skip("brief check", failed[0])
            return 0
        if failed:
            self._skip(f"brief check, {len(failed)} of {len(results)} parts", failed[0])
        data = {"problems": []}
        for part, result in zip(parts, results):
            if not isinstance(result, BaseException):
                part_data, usage = result
                self._add_usage(usage)
                allowed = {entry["id"] for entry in part}
                data["problems"] += [finding for finding in _as_list(part_data.get("problems"))
                                     if not isinstance(finding, dict)
                                     or not str(finding.get("id") or "").strip().startswith("beginner_plan")
                                     or str(finding.get("id") or "").strip() in allowed]
        fixes, drop = [], {}
        points = {pt["id"]: pt for pt in _as_list(brief.get("points")) if isinstance(pt, dict)
                  for pt in [{**pt, "id": str(pt.get("id") or "").strip()}]}
        for item in _as_list(data.get("problems")):
            if not isinstance(item, dict):
                continue
            pid, corrected = str(item.get("id") or "").strip(), _text(item.get("corrected"))
            if not pid or not corrected:
                continue
            delete = corrected.upper().strip(".") == "DELETE"
            parts = pid.split(".")
            if parts[0] == "beginner_plan":
                if pid not in plan_leaves:
                    continue
                parent, key = plan_leaves[pid]
                if isinstance(parent, dict) and key not in parent:
                    continue
                if parts[1] == "opening_questions" and not delete:
                    corrected = corrected.strip('"“”').strip()
                    # A declarative source quotation cannot replace a reader question. Drop this
                    # flagged leaf rather than send either the bad original or a changed role onward.
                    if not corrected.endswith(("?", "？")):
                        delete = True
                fixes.append({"id": pid, "before": parent[key], "after": "" if delete else corrected})
                if delete:
                    if isinstance(parent, dict):
                        del parent[key]
                    else:
                        drop.setdefault(".".join(parts[:2]), set()).add(key)
                else:
                    parent[key] = corrected
            elif parts[0] in ("problem", "key_idea", "setup", "takeaway") and len(parts) == 1 and isinstance(brief.get(pid), str):
                fixes.append({"id": pid, "before": brief[pid], "after": "" if delete else corrected})
                brief[pid] = "" if delete else corrected
            elif parts[0] in ("contributions", "limitations") and len(parts) == 2 and parts[1].isdigit():
                brief[parts[0]] = items = _as_list(brief.get(parts[0]))   # the brief's own list: a correction must stick
                i = int(parts[1])
                if i < len(items):
                    fixes.append({"id": pid, "before": items[i], "after": "" if delete else corrected})
                    if delete:
                        drop.setdefault(parts[0], set()).add(i)
                    else:
                        items[i] = corrected
            elif parts[0] in ("mechanisms", "evidence") and len(parts) == 3 and parts[1].isdigit():
                items, i, f = _as_list(brief.get(parts[0])), int(parts[1]), parts[2]
                if i < len(items) and isinstance(items[i], dict) and f in items[i]:
                    fixes.append({"id": pid, "before": items[i][f], "after": "" if delete else corrected})
                    if delete and f in ("claim", "plain"):
                        drop.setdefault(parts[0], set()).add(i)
                    else:
                        items[i][f] = "" if delete else corrected
            elif parts[0] == "abstract_claims" and len(parts) == 2 and parts[1].isdigit():
                claims, i = _as_list(brief.get("abstract_claims")), int(parts[1])
                if i < len(claims) and isinstance(claims[i], dict):
                    # the abstract says it, so the claim stays: a "DELETE" falls back to the abstract's own words
                    after = str(claims[i].get("quote") or claims[i].get("text") or "") if delete else corrected
                    fixes.append({"id": pid, "before": claims[i].get("text"), "after": after})
                    claims[i]["text"] = after
            elif parts[0] == "points" and len(parts) >= 2:
                pt_id = ".".join(parts[1:])
                for raw in _as_list(brief.get("points")):
                    if isinstance(raw, dict) and str(raw.get("id") or "").strip() == pt_id:
                        fixes.append({"id": pid, "before": raw.get("text"), "after": "" if delete else corrected})
                        if delete:
                            drop.setdefault("points", set()).add(pt_id)
                        else:
                            raw["text"] = corrected
            elif parts[0] == "running_example" and isinstance(brief.get("running_example"), dict):
                example = brief["running_example"]
                if len(parts) == 2 and parts[1] == "from_paper":   # any flag: the paper does not show it, so an illustration
                    fixes.append({"id": pid, "before": example.get("source") or "from the paper", "after": "an illustration"})
                    example["from_paper"], example["source"] = False, ""
                elif len(parts) == 2 and parts[1] in ("text", "old_way", "new_way") and isinstance(example.get(parts[1]), str):
                    fixes.append({"id": pid, "before": example[parts[1]], "after": "" if delete else corrected})
                    if delete and parts[1] == "text":   # the paper does not show this case: the story keeps it as an illustration
                        example["from_paper"], example["source"] = False, ""
                    else:
                        example[parts[1]] = "" if delete else corrected
                elif len(parts) == 3 and parts[1] == "parts" and parts[2].isdigit():
                    pairs, i = _part_pairs(example), int(parts[2])
                    if i < len(pairs):
                        fixes.append({"id": pid, "before": _part_line(*pairs[i]), "after": "" if delete else corrected})
                        if delete:   # dropped after the loop, so the next finding's number still names its part
                            drop.setdefault("running_example.parts", set()).add(i)
                        else:   # "text — what it is" (a correction without a role leaves the role unknown)
                            text, _, role = str(corrected).partition(ROLE_SEP)
                            pairs[i] = (text.strip(), role.strip())
                            example["parts"], example["roles"] = [t for t, _ in pairs], [r for _, r in pairs]
        for key, gone in drop.items():
            if key.startswith("beginner_plan."):
                field = key.split(".")[1]
                values = brief["beginner_plan"][field]
                brief["beginner_plan"][field] = [value for i, value in enumerate(values) if i not in gone]
            elif key == "points":
                brief["points"] = [pt for pt in _as_list(brief.get("points")) if not (isinstance(pt, dict) and str(pt.get("id") or "").strip() in gone)]
            elif key == "running_example.parts":
                example = brief["running_example"]
                pairs = [x for i, x in enumerate(_part_pairs(example)) if i not in gone]
                example["parts"], example["roles"] = [t for t, _ in pairs], [r for _, r in pairs]
            else:
                brief[key] = [x for i, x in enumerate(_as_list(brief.get(key))) if i not in gone]
        if fixes:
            print(f"[StoryboarderV2] brief check corrected {len(fixes)} statements: {[f['id'] for f in fixes]}")
        brief["brief_check"] = fixes
        return len(fixes)

    async def write(self, brief: dict, cast: Cast, panel_range: tuple[int, int], language: str) -> StoryboardV2:
        data, usage = await _ask_json(self.text_engine, SYSTEM_WRITER, storyboard_prompt(brief, cast, language, panel_range), thinking=True, temperature=0.6, max_tokens=24000, key="panels")
        self._add_usage(usage)
        panels = []
        for i, p in enumerate(data.get("panels") or [], start=1):
            try:
                panels.append(PanelV2.from_dict(p, i))
            except Exception as e:  # keep going on a malformed panel
                print(f"[StoryboarderV2] skipping malformed panel {i}: {e}")
        if not panels:
            raise ValueError("storyboard model returned no panels")
        numbers = [p.n for p in panels]
        if len(set(numbers)) == len(numbers):
            panels.sort(key=lambda p: p.n)          # trust the model's numbering when it is unique; else keep reply order
        _normalize_panels(panels, cast)
        return StoryboardV2(
            title=str(data.get("title") or brief.get("title") or "Paper manga"),
            summary=str(data.get("summary") or brief.get("key_idea") or ""),
            theme=cast.theme,
            language="en-US",
            chapters=_chapters(panels),
            panels=panels,
            brief=brief,
            card=paper_card(brief),
        )

    async def cover(self, sb: StoryboardV2, cast: Cast, panel_range: tuple[int, int]) -> int:
        """Give every core point of the brief a panel: missing ones get new panels (up to the panel budget, placed inside
        their own chapter) or a sentence in the detail of the closest panel. Returns the number of points it placed."""
        missing = uncovered(sb)
        if not missing:
            return 0
        try:
            data, usage = await _ask_json(self.text_engine, SYSTEM_WRITER, cover_prompt(sb, cast, missing, panel_range[1]),
                                          thinking=True, temperature=0.4, max_tokens=12000, key="new_panels")
        except Exception as e:  # coverage repair is best effort
            _reraise_account(e)
            print(f"[StoryboarderV2] coverage repair skipped: {e}")
            return 0
        self._add_usage(usage)
        missing_ids = {str(pt["id"]) for pt in missing}
        by_n = {p.n: p for p in sb.panels}
        new_items = [x for x in _as_list(data.get("new_panels")) if isinstance(x, dict)]
        details = [x for x in _as_list(data.get("details")) if isinstance(x, dict)] + [x for x in new_items if x.get("add_detail")]

        def to_detail(n, sentence: str, covers) -> None:   # an existing panel explains the point (or gets a sentence for it)
            try:
                panel = by_n.get(int(n))
            except (TypeError, ValueError):
                return
            if panel is None:
                return
            if sentence and sentence not in panel.detail:
                panel.detail = f"{panel.detail} {sentence}".strip()
            panel.covers += [c for c in PanelV2.from_dict({"covers": covers}, 0).covers if c not in panel.covers]

        for item in details:
            to_detail(item.get("n"), str(item.get("add_detail") or "").strip(), item.get("covers"))
        room = min(max(0, panel_range[1] - len(sb.panels)), len(missing_ids))
        titles = {p.title.strip().lower() for p in sb.panels if p.title.strip()}
        lines = {tuple(b.text for b in p.dialogue) for p in sb.panels if p.dialogue}
        inserts = []
        for i, item in enumerate(x for x in new_items if not x.get("add_detail")):
            try:
                new = PanelV2.from_dict(item, 0)
            except Exception as e:  # keep going on a malformed panel
                print(f"[StoryboarderV2] skipping malformed coverage panel: {e}")
                continue
            try:
                after = int(item.get("after"))
            except (TypeError, ValueError):
                after = len(sb.panels)
            repeated = new.title.strip().lower() in titles or tuple(b.text for b in new.dialogue) in lines
            useful = set(new.covers) & missing_ids and new.visual and (new.dialogue or new.narration) and not repeated
            if not useful:
                continue
            if len(inserts) >= room:   # no room left: its facts go into the detail of the panel it would follow
                to_detail(min(max(after, 1), len(sb.panels)), new.detail or new.narration, new.covers)
                continue
            inserts.append((_insert_position(sb.panels, new, after), i, new))
        if inserts:
            ordered = list(sb.panels)
            for pos, _i, new in sorted(inserts, key=lambda t: (t[0], t[1]), reverse=True):
                ordered.insert(pos, new)
            _normalize_panels(ordered, cast)
            sb.panels, sb.chapters = ordered, _chapters(ordered)
        placed = len(missing) - len(uncovered(sb))
        print(f"[StoryboarderV2] coverage repair placed {placed} of {len(missing)} missing core points ({len(inserts)} new panels)")
        return placed

    def violations(self, sb: StoryboardV2, language: str) -> list[dict]:
        if sb.version >= 3:
            return self._violations_v3(sb, language)
        budget = BUDGETS.get(language, BUDGETS["en-US"])
        items = []
        for p in sb.panels:
            if _mentions_people(f"{p.scene} {p.visual}"):   # Seedream draws whoever the scene mentions
                items.append({"id": f"{p.n}.v", "kind": "visual", "text": f"Scene: {p.scene}. Visual: {p.visual}"})
            for i, b in enumerate(p.dialogue):
                if text_len(b.text, language) > budget["bubble"] or "_" in b.text or "\\" in b.text:
                    items.append({"id": f"{p.n}.d{i}", "kind": "bubble", "text": b.text})
            if p.narration and (text_len(p.narration, language) > budget["narration"] or "_" in p.narration or "\\" in p.narration):
                items.append({"id": f"{p.n}.n", "kind": "narration", "text": p.narration})
            for i, lab in enumerate(p.labels):
                if text_len(lab, language) > budget["label"] or "_" in lab or "\\" in lab:
                    items.append({"id": f"{p.n}.l{i}", "kind": "label", "text": lab})
        return items

    def _violations_v3(self, sb: StoryboardV2, language: str) -> list[dict]:
        """Over-budget lines of a v3 script (limits by panel kind, given per item as "max") and scenes that mention people."""
        budget, items = _v3_budget(language), []
        for p in sb.panels:
            b = budget["board" if p.kind == "board" else "scene"]
            art = language == "en-US"   # a translation draws the English art directions: they are fitted once, in English
            if art and _mentions_people(f"{p.scene} {p.visual}"):   # Seedream draws whoever the scene mentions
                items.append({"id": f"{p.n}.v", "kind": "visual", "text": f"Scene: {p.scene}. Visual: {p.visual}", "cast": list(p.characters)})
            if art and _mentions_people(p.shows):   # ...and whoever the picture's idea names
                items.append({"id": f"{p.n}.s", "kind": "visual", "text": p.shows, "cast": list(p.characters)})
            for i, x in enumerate(p.dialogue):
                if text_len(x.text, language) > b["bubble"] or "_" in x.text or "\\" in x.text:
                    items.append({"id": f"{p.n}.d{i}", "kind": "bubble", "text": x.text, "max": b["bubble"]})
            if p.narration and (text_len(p.narration, language) > b["narration"] or "_" in p.narration or "\\" in p.narration):
                items.append({"id": f"{p.n}.n", "kind": "narration", "text": p.narration, "max": b["narration"]})
        return items

    async def fit(self, sb: StoryboardV2, language: str, rounds: int = 2, only: Optional[set] = None) -> int:
        """Rewrite over-budget lines with the fast model. Returns the number of lines rewritten. In a v3 script a rewrite
        that changes the line's numbers is refused (the longer exact line beats a short wrong one). ``only``: the panels
        to fit (the last check's corrections elsewhere ship as the checker wrote them)."""
        rewritten = 0
        names = {c for q in sb.panels for c in q.characters}
        localized = sb.version >= 3 and language != "en-US"
        retained = sb.brief.get(f"translation_retained_{language}", {}) if localized else {}

        def kept_original(pid):
            return pid in retained and _line_at(sb, pid) == retained[pid]

        for _ in range(rounds):
            items = [x for x in self.violations(sb, language)
                     if (only is None or int(str(x["id"]).split(".")[0]) in only) and not kept_original(x["id"])]
            if not items:
                break
            prompt = fit_prompt_v3(items, language, is_novice(sb)) if sb.version >= 3 else fit_prompt(items, language)
            data, usage = await _ask_json(self.fast_engine, "", prompt, thinking=False, temperature=0.2, max_tokens=8000, key="items")
            self._add_usage(usage)
            by_id = {str(x.get("id")): _text(x.get("text")) for x in _as_list(data.get("items")) if isinstance(x, dict) and _text(x.get("text"))}
            if sb.version >= 3:
                originals = {f"{p.n}.d{i}": b.text for p in sb.panels for i, b in enumerate(p.dialogue)} | {f"{p.n}.n": p.narration for p in sb.panels}
                if localized:
                    requested = {x["id"] for x in items}
                    by_id = {k: v for k, v in by_id.items() if k in requested and
                             (k not in originals or _translation_numbers_equal(originals[k], v))}
                else:
                    by_id = {k: v for k, v in by_id.items() if k not in originals or _numbers(v) == _numbers(originals[k])}
            for p in sb.panels:
                for i, b in enumerate(p.dialogue):
                    new = by_id.get(f"{p.n}.d{i}")
                    if new:
                        b.text, rewritten = new, rewritten + 1
                new = by_id.get(f"{p.n}.n")
                if new:
                    p.narration, rewritten = new, rewritten + 1
                for key in (f"{p.n}.v", f"{p.n}.s"):   # a cast member the rewrite casts (a driver played by nezu) is in the picture
                    for name in sorted(names):
                        if by_id.get(key) and name not in p.characters and _stages(name, by_id[key]):
                            p.characters.append(name)
                new = by_id.get(f"{p.n}.v")
                if new:
                    p.scene, p.visual, rewritten = "", new, rewritten + 1
                new = by_id.get(f"{p.n}.s")
                if new:
                    p.shows, rewritten = new, rewritten + 1
                for i in range(len(p.labels)):
                    new = by_id.get(f"{p.n}.l{i}")
                    if new:
                        p.labels[i], rewritten = new, rewritten + 1
        # last resort: strip forbidden characters (never truncate)
        for p in sb.panels:
            for i, b in enumerate(p.dialogue):
                if not kept_original(f"{p.n}.d{i}"):
                    b.text = _plain(b.text)
            if not kept_original(f"{p.n}.n"):
                p.narration = _plain(p.narration)
            p.labels = [_plain(lab) for lab in p.labels]
        return rewritten

    async def factcheck(self, sb: StoryboardV2, paper: PaperText, language: str = "en-US", board_flags: Optional[dict] = None,
                        panels: Optional[set] = None) -> int:
        """Verify every bubble/narration (and board string) against the paper and apply corrections: once on the English
        script, and again after translation, which can drop a qualifier the first pass added. Returns #fixes.
        Whole-board findings ("N.b") cannot be fixed by replacing a string: they are collected in ``board_flags``
        ({n: [problem]}) for a fix round when the caller passes it. ``panels``: check only these (no card); their fixes
        are added to the ones already recorded. Every English panel checked is remembered (self.checked), so a quiz
        revision re-checks only what it changed."""
        card = panels is None or (language == "en-US" and bool(sb.card) and _card_view(sb) != self.card_checked)
        if panels is not None and not panels and not card:
            return 0
        # a checker of its own (DeepSeek thinks for minutes over a whole script) gets the script in parallel parts that
        # read the paper from the cache a warm-up filled; the writer's own model checks in one call, as before
        if self.check_engine is not self.text_engine:
            parts = factcheck_parts(sb, paper, language, only=panels, card=card)
        else:
            parts = [(None, True)] if panels is None else [(set(panels), card)]
        if panels is not None:
            print(f"[StoryboarderV2] fact-check ({language}) of the changed panels {sorted(panels)}")
        if len(parts) > 1:
            await self._warm(self.check_engine, paper.body)
        results = await asyncio.gather(*(
            _ask_check(self.check_engine, factcheck_prompt(sb, paper, language, panels, card), thinking=True, temperature=0.1,
                       max_tokens=12000, key="problems", paper=paper.body, cache_key=self.cache_key)
            for panels, card in parts), return_exceptions=True)
        failed = [r for r in results if isinstance(r, BaseException)]
        for e in failed:
            _reraise_account(e)
        stage = "fact-check" if language == "en-US" else f"fact-check ({language})"
        if len(failed) == len(results):   # fact-check is best effort
            print(f"[StoryboarderV2] fact-check skipped: {failed[0]}")
            self._skip(stage, failed[0])
            return 0
        if failed:
            print(f"[StoryboarderV2] {len(failed)} of {len(results)} fact-check parts failed: {failed[0]}")
            self._skip(f"{stage}, {len(failed)} of {len(results)} parts", failed[0])
        data = {"problems": []}
        read: set[int] = set()   # the panels of the parts that came back
        for (part_panels, _card), result in zip(parts, results):
            if not isinstance(result, BaseException):
                part, usage = result
                self._add_usage(usage)
                data["problems"] += _as_list(part.get("problems"))
                read |= {p.n for p in sb.panels} if part_panels is None else set(part_panels)
        fixes = []
        board_updates: dict[int, dict[str, tuple[str, str]]] = {}
        unresolved: set[int] = set()   # panels with a finding no correction settled here: read again next time
        for item in _as_list(data.get("problems")):   # a stray shape ("problems": 0) is no reason to fail the job
            if not isinstance(item, dict):
                continue
            pid, corrected = str(item.get("id", "")), _text(item.get("corrected"))
            if not corrected or "." not in pid or not _has_script(corrected, language, _latin_keep(sb.brief)):
                if pid.split(".", 1)[0].isdigit():
                    unresolved.add(int(pid.split(".", 1)[0]))
                continue
            if pid.startswith("card."):
                _, key, index = (pid.split(".") + ["", ""])[:3]
                value = sb.card.get(key)
                if isinstance(value, str) and index in ("", "0"):
                    fixes.append({"id": pid, "before": value, "after": corrected, "issue": item.get("issue", "")})
                    sb.card[key] = corrected
                elif isinstance(value, list) and index.isdigit() and int(index) < len(value):
                    fixes.append({"id": pid, "before": value[int(index)], "after": corrected, "issue": item.get("issue", "")})
                    value[int(index)] = corrected
                continue
            n_str, part = pid.split(".", 1)
            try:
                n = int(n_str)
            except ValueError:
                continue
            panel = next((p for p in sb.panels if p.n == n), None)
            if panel is None:
                continue
            if part == "n":
                fixes.append({"id": pid, "before": panel.narration, "after": corrected, "issue": item.get("issue", "")})
                panel.narration = corrected
            elif part == "x":
                fixes.append({"id": pid, "before": panel.detail, "after": corrected, "issue": item.get("issue", "")})
                panel.detail = corrected
            elif part in ("v", "s") and language != "en-US":
                continue   # never sent: a translation's checker does not rewrite the English art directions
            elif part == "v":
                fixes.append({"id": pid, "before": panel.visual, "after": corrected, "issue": item.get("issue", "")})
                panel.visual = corrected
            elif part == "s":
                fixes.append({"id": pid, "before": panel.shows, "after": corrected, "issue": item.get("issue", "")})
                panel.shows = corrected
            elif part == "b" and panel.board:
                unresolved.add(n)   # a fix round rewrites the board; its new form gets read again
                if board_flags is not None:
                    board_flags.setdefault(n, []).append(f"fact-check: {str(item.get('issue') or '').strip()} Fix: {corrected}".strip())
                fixes.append({"id": pid, "before": "", "after": corrected, "issue": item.get("issue", "")})
            elif part.startswith("b.") and panel.board:
                board_updates.setdefault(n, {})[part[2:]] = (corrected, str(item.get("issue") or ""))
            elif part.startswith("d"):
                try:
                    i = int(part[1:])
                except ValueError:
                    continue
                if i < len(panel.dialogue):
                    fixes.append({"id": pid, "before": panel.dialogue[i].text, "after": corrected, "issue": item.get("issue", "")})
                    panel.dialogue[i].text = corrected
        for n, updates in board_updates.items():
            panel = next(p for p in sb.panels if p.n == n)
            before = {str(path): text for path, text in _board_strings(panel.board)}
            new = _set_board_strings(panel.board, {k: v for k, (v, _) in updates.items()})
            drawn = _is_diagram(panel.board) and bool(panel.board.get("elements"))
            if drawn:   # a label's number the checker corrects from the paper (the board's own rule refuses number changes)
                new = _set_diagram_numbers(new, {k: v for k, (v, _) in updates.items()}, paper.body, _illustrative_example(sb))
                now = _board_all_texts(new)
                left = {k: (v, issue) for k, (v, issue) in updates.items() if k in before and now.get(k) != re.sub(r"\s+", " ", v).strip()}
                if left:   # a correction that could not be written in (its number is not the paper's): read again, and the
                    unresolved.add(n)   # designer gets the finding; it was dropped without a word and counted as checked
                    if board_flags is not None:
                        board_flags.setdefault(n, []).extend(f"fact-check: {issue.strip()} (the label \"{before[k]}\") Fix: {v}".strip()
                                                             for k, (v, issue) in left.items())
            if new is panel.board or new == panel.board:
                continue
            if len(board_issues(new, paper.body)) > len(board_issues(panel.board, paper.body)):   # a correction must not break the board
                print(f"[StoryboarderV2] fact-check correction of board {n} refused: it breaks the board's checks")
                unresolved.add(n)
                continue
            boards_mod = _optional("boards")
            if drawn and boards_mod is not None and board_flags is not None:
                # the correction stands (the paper first), and a label it made longer gets the drawing moved around it:
                # "no gains" became "limited or negative gains" and ran off the board onto the layer boxes (round 23)
                broken = [x for x in boards_mod.diagram_problems(new) if x not in boards_mod.diagram_problems(panel.board)
                          and LABEL_LENGTH not in x]   # a corrected label stands at its length; a relayout only moves things
                if broken:
                    board_flags.setdefault(n, []).extend(f"{DIAGRAM_LAYOUT}{x}" for x in broken)
            panel.board = new
            after = _board_all_texts(new)
            fixes += [{"id": f"{n}.b.{k}", "before": before[k], "after": after.get(k, ""), "issue": issue}
                      for k, (_, issue) in updates.items() if k in before and after.get(k, "") != before[k]]
        if fixes:
            print(f"[StoryboarderV2] fact-check ({language}) corrected {len(fixes)} lines: {[f['id'] for f in fixes]}")
        key = "factcheck" if language == "en-US" else f"factcheck_{language}"
        sb.brief[key] = (list(sb.brief.get(key) or []) if panels is not None else []) + fixes
        self.last_fixes = fixes
        if language == "en-US":   # its corrections are the checker's own words: checked as they stand
            self.checked |= {_checked_view(p) for p in sb.panels if p.n in read - unresolved}
            if any(with_card for (_, with_card), r in zip(parts, results) if not isinstance(r, BaseException)):
                self.card_checked = _card_view(sb)
        return len(fixes)

    async def translate(self, sb: StoryboardV2, language: str, _retry: bool = False) -> StoryboardV2:
        if language == "en-US" or language not in BUDGETS:
            sb.language = language if language in BUDGETS else "en-US"
            return sb
        english = {f"{p.n}.d{i}": b.text for p in sb.panels for i, b in enumerate(p.dialogue)}
        english.update({f"{p.n}.n": p.narration for p in sb.panels if p.narration})
        retained_key = f"translation_retained_{language}"
        if not _retry:
            sb.brief.pop(retained_key, None)

        def retain_originals(panel_numbers):
            if sb.version >= 3:
                originals = {pid: text for pid, text in english.items()
                             if int(pid.split(".")[0]) in panel_numbers and _line_at(sb, pid) == text}
                sb.brief.setdefault(retained_key, {}).update(originals)

        data, usage = await _ask_json(self.translate_engine, "", translate_prompt(sb, language), thinking=False, temperature=0.3, max_tokens=24000, key="panels")
        self._add_usage(usage)
        if _text(data.get("title")) and not _retry:
            sb.title = _text(data["title"])
        if _text(data.get("summary")) and not _retry:
            sb.summary = _text(data["summary"])
        if isinstance(data.get("card"), dict) and not _retry:
            sb.card = _translated_card(sb.card, data["card"], language, _latin_keep(sb.brief))
        ch_titles = {str(c.get("id")): str(c.get("title") or "").strip() for c in _as_list(data.get("chapters")) if isinstance(c, dict)}
        for c in sb.chapters:
            if ch_titles.get(c.id):
                c.title = ch_titles[c.id]
        by_n: dict[int, dict] = {}
        for t in _panel_items(data.get("panels")):
            try:
                by_n[int(t.get("n"))] = t
            except (TypeError, ValueError):
                continue
        keep = _latin_keep(sb.brief)
        applied = 0
        rejected: list[int] = []
        for p in sb.panels:
            t = by_n.get(p.n)
            if not t:
                rejected.append(p.n)
                continue
            td = [b for b in _as_list(t.get("dialogue")) if isinstance(b, dict)]
            speakers_ok = len(td) == len(p.dialogue) and all(
                _match_cast(str(b.get("speaker") or ""), {orig.speaker}) == orig.speaker or not b.get("speaker") for b, orig in zip(td, p.dialogue))
            texts = [_text(b.get("text")) for b in td]
            # the narration is checked like a bubble; an absent key keeps the English, and none is added where there was none
            narration = _text(t["narration"]) if p.narration and t.get("narration") is not None else None
            lines = texts + ([narration] if narration is not None else [])
            if not speakers_ok or any(not x or not _has_script(x, language, keep) for x in lines):
                rejected.append(p.n)
                continue
            originals = [b.text for b in p.dialogue] + ([p.narration] if narration is not None else [])
            if sb.version >= 3 and any(not _translation_numbers_equal(o, x) for o, x in zip(originals, lines)):
                rejected.append(p.n)
                continue
            if _text(t.get("title")):
                p.title = _text(t["title"])
            if narration:
                p.narration = narration
            for b, text in zip(p.dialogue, texts):
                b.text = text
                applied += 1
            tl = _labels(t.get("labels"))
            if tl and len(tl) == len(p.labels):
                p.labels = tl[:2]
            detail = _text(t.get("detail"))
            if p.detail and detail and _has_script(detail, language, keep):   # reader text: an untranslated detail keeps the English
                p.detail = detail
            if p.board and isinstance(t.get("board"), dict):
                p.board = _set_board_strings(p.board, _board_translation(p.board, t["board"], language, keep))
        if rejected and not _retry:
            print(f"[StoryboarderV2] translation rejected for panels {rejected}; re-requesting once")
            sub = StoryboardV2(title=sb.title, summary=sb.summary, theme=sb.theme, language=sb.language, chapters=sb.chapters,
                               panels=[p for p in sb.panels if p.n in rejected], brief=sb.brief, card={}, version=sb.version)
            try:
                await self.translate(sub, language, _retry=True)
            except Exception as e:  # the re-request is best effort: the first pass is kept
                _reraise_account(e)
                print(f"[StoryboarderV2] WARNING: re-request failed ({type(e).__name__}: {e}); panels {rejected} kept their English text")
                if sb.version >= 3:
                    retain_originals(rejected)
                    for n in rejected:
                        self._skip(f"translation {language} panel {n}", RuntimeError("English text retained: translation retry failed"))
        elif rejected:
            print(f"[StoryboarderV2] WARNING: panels {rejected} kept their English text (translation failed twice)")
            if sb.version >= 3:
                retain_originals(rejected)
                for n in rejected:
                    self._skip(f"translation {language} panel {n}", ValueError("English text retained after two rejected translations (missing panel, speaker, script or numeric mismatch)"))
        if not _retry:
            await self._translate_leftovers(sb, language)
            await self._fix_translation(sb, language, english)
        print(f"[StoryboarderV2] translated {applied} bubbles to {language}")
        sb.language = language
        return sb

    async def _fix_translation(self, sb: StoryboardV2, language: str, english: dict[str, str]) -> int:
        """Bubbles and narration whose translation is stronger than the English, drops its hedge, transliterates a
        name or renames a prop (translation_problems) are corrected in one call; a correction must keep the line's
        numbers. Returns the number of lines corrected (recorded in brief["translation_fixes_<language>"])."""
        names = frozenset(n for p in sb.panels for n in p.characters)
        items = []
        for pid, source in english.items():
            line = _line_at(sb, pid)
            found = translation_problems(source, line or "", language, names) if line and line != source else []
            if found:
                items.append({"id": pid, "english": source, "translation": line, "problems": found})
        if not items:
            return 0
        try:
            data, usage = await _ask_json(self.translate_engine, "", translation_fix_prompt(items, language), thinking=False,
                                          temperature=0.2, max_tokens=8000, key="items")
        except Exception as e:  # the translation stays as it is; the translation's fact-check still reads it
            _reraise_account(e)
            print(f"[StoryboarderV2] translation fixes skipped: {type(e).__name__}: {e}")
            self._skip("translation fixes", e)
            return 0
        self._add_usage(usage)
        keep, fixed = _latin_keep(sb.brief), []
        requested = {item["id"] for item in items}
        for item in _as_list(data.get("items")):
            if not isinstance(item, dict):
                continue
            pid, text = str(item.get("id") or ""), _text(item.get("text"))
            before = _line_at(sb, pid)
            if sb.version >= 3:
                # Ignore unsolicited repairs, especially English lines retained after
                # both panel attempts failed. Repair must obey the same numeric rule.
                if pid not in requested or not _translation_numbers_equal(english[pid], text):
                    continue
            if not text or pid not in english or not _has_script(text, language, keep):
                continue
            if sb.version < 3 and (not _figures_kept(english[pid], text) or _number_keys(text) - _number_keys(english[pid])):
                continue
            if _set_line_at(sb, pid, text):
                fixed.append({"id": pid, "before": before, "after": text})
        if fixed:
            print(f"[StoryboarderV2] translation ({language}): corrected {len(fixed)} lines that misread their English: {[f['id'] for f in fixed]}")
        sb.brief[f"translation_fixes_{language}"] = fixed
        return len(fixed)

    async def backport(self, english: StoryboardV2, localized: StoryboardV2, paper: PaperText) -> int:
        """The translation's fact-check findings correct the English line they came from when it has the same problem:
        in the v3 runs, 30 of the 38 problems the Chinese fact-check found were already in the English script, which two
        English fact-checks had passed. One call to the checker; a correction may not bring a number the paper lacks.
        Returns the number of English lines changed (recorded in english.brief["backport"])."""
        fixes = localized.brief.get(f"factcheck_{localized.language}") or []
        items = []
        for f in fixes:
            pid = str(f.get("id") or "")
            line = _line_at(english, pid)
            if line:
                items.append({"id": pid, "english": line, "issue": str(f.get("issue") or ""), "corrected_translation": str(f.get("after") or "")})
        if not items:
            return 0
        try:
            data, usage = await _ask_json(self.check_engine, "", backport_prompt(items) + paper.body, thinking=False, temperature=0.1,
                                          max_tokens=6000, key="items", paper=paper.body, cache_key=self.cache_key)
        except Exception as e:  # the English stays as it is
            _reraise_account(e)
            print(f"[StoryboarderV2] back-port of the {localized.language} fact-check skipped: {type(e).__name__}: {e}")
            self._skip("back-port of the translation's fact-check", e)
            return 0
        self._add_usage(usage)
        changed = []
        for item in _as_list(data.get("items")):
            if not isinstance(item, dict):
                continue
            pid, text = str(item.get("id") or ""), _text(item.get("text"))
            before = _line_at(english, pid)
            if not text or text.upper().strip(" .") == "KEEP" or before is None or text == before:
                continue
            if CJK_RE.search(text) or not _numbers_supported(text, paper.body):
                continue
            if _set_line_at(english, pid, text):
                changed.append({"id": pid, "before": before, "after": text})
        if changed:
            print(f"[StoryboarderV2] the {localized.language} fact-check corrected {len(changed)} English lines too: {[c['id'] for c in changed]}")
        english.brief["backport"] = english.brief.get("backport", []) + changed
        return len(changed)

    async def _translate_leftovers(self, sb: StoryboardV2, language: str) -> None:
        """Details, card lines and board strings the translation left in English get one small re-request (a board string
        must keep its numbers); what still fails stays English with a warning."""
        keep = _latin_keep(sb.brief)
        items = {f"{p.n}.x": p.detail for p in sb.panels if p.detail and not _has_script(p.detail, language, keep)}
        for p in sb.panels:
            items.update({f"{p.n}.b.{path}": text for path, text in _board_strings(p.board) if not _has_script(text, language, keep)})
        for key, value in sb.card.items():
            for i, text in enumerate([value] if isinstance(value, str) else value):
                if not _has_script(text, language, keep):
                    items[f"card.{key}.{i}"] = text
        if not items:
            return
        prompt = (f"Translate each text into natural {LANG_NAMES[language]} for readers new to the topic. Keep technical terms, "
                  f"names and numbers exact; use full-width punctuation. Return JSON only: {{\"items\": [{{\"id\": \"...\", \"text\": \"...\"}}]}}\n\n"
                  + json.dumps([{"id": k, "text": v} for k, v in items.items()], ensure_ascii=False))
        try:
            data, usage = await _ask_json(self.translate_engine, "", prompt, thinking=False, temperature=0.2, max_tokens=8000, key="items")
        except Exception as e:  # best effort: the English stays readable
            _reraise_account(e)
            print(f"[StoryboarderV2] WARNING: {len(items)} reader texts kept their English ({type(e).__name__}: {e})")
            self._skip("translation of reader texts", e)
            return
        self._add_usage(usage)
        done = 0
        for t in _as_list(data.get("items")):
            if not isinstance(t, dict) or str(t.get("id")) not in items:
                continue
            text, pid = _text(t.get("text")), str(t["id"])
            if not text or not _has_script(text, language, keep):
                continue
            n_str, _, rest = pid.partition(".")
            if rest.startswith("b."):
                panel = next((p for p in sb.panels if str(p.n) == n_str and p.board), None)
                if panel is not None and _numbers(text) == _numbers(items[pid]):
                    new = _set_board_strings(panel.board, {rest[2:]: text})
                    if new is not panel.board:
                        panel.board, done = new, done + 1
                continue
            if pid.endswith(".x"):
                panel = next((p for p in sb.panels if f"{p.n}.x" == pid), None)
                if panel is not None:
                    panel.detail, done = text, done + 1
            else:
                _, key, index = pid.split(".")
                if isinstance(sb.card.get(key), str):
                    sb.card[key], done = text, done + 1
                elif isinstance(sb.card.get(key), list) and index.isdigit() and int(index) < len(sb.card[key]):
                    sb.card[key][int(index)], done = text, done + 1
        if done < len(items):
            print(f"[StoryboarderV2] WARNING: {len(items) - done} reader texts kept their English after a re-request")

    async def build(self, paper: PaperText, cast: Cast, language: str, panel_range: tuple[int, int], on_stage=None) -> StoryboardV2:
        if on_stage:
            on_stage("analyzing", "Reading the paper")
        brief = await self.analyze(paper)
        if on_stage:
            on_stage("analyzing", "Checking the brief against the paper")
        await self.check_brief(brief, paper)
        if on_stage:
            on_stage("storyboarding", "Writing the manga script")
        sb = await self.write(brief, cast, panel_range, language)
        await self.fit(sb, "en-US")
        if uncovered(sb):
            if on_stage:
                on_stage("storyboarding", "Adding panels for key points the script missed")
            if await self.cover(sb, cast, panel_range):
                await self.fit(sb, "en-US")
        if on_stage:
            on_stage("storyboarding", "Fact-checking the script against the paper")
        await self.factcheck(sb, paper)
        await self.fit(sb, "en-US")
        if language != "en-US":
            if on_stage:
                on_stage("translating", f"Translating to {LANG_NAMES.get(language, language)}")
            sb = await self.translate(sb, language)
            await self.fit(sb, language)
            if on_stage:
                on_stage("translating", "Fact-checking the translation against the paper")
            if await self.factcheck(sb, paper, language):
                await self.fit(sb, language)
        return sb

    # ------------------------------------------------------------------ v3 story layer

    async def analyze_v3(self, paper: PaperText, abstract: str) -> dict:
        """The brief plus the abstract's claims (quotes verified against the abstract, every sentence covered) and the
        paper's own example inputs (verified in the paper)."""
        for attempt in range(2):   # everything after depends on the brief: a reply no repair could parse is asked once more
            try:
                brief, usage = await _ask_json(self.text_engine, SYSTEM_WRITER, brief_prompt_v3(paper, abstract), thinking=True,
                                               temperature=0.2, max_tokens=16000, key="abstract_claims", paper=paper.body,
                                               cache_key=self.cache_key)
                break
            except ValueError as e:   # json.JSONDecodeError is a ValueError
                if attempt:
                    raise
                print(f"[StoryboarderV2] the brief reply could not be read ({e}); asking once more")
        self._add_usage(usage)
        if not brief.get("title") and paper.title:
            brief["title"] = paper.title
        brief["abstract_claims"] = verify_claims(brief.get("abstract_claims"), abstract, paper.body)
        brief["examples"] = paper_examples(brief.get("examples"), paper.body)
        brief["running_example"] = brief_example(brief.get("running_example"))
        # no stand-in symbols: diagrams draw the real things, and a prop a character had to decode ("this balloon is
        # uncertainty") taught nothing (round 22's blind panel test; the judges of three eras: "every 'this red cap
        # means...' line is a picture failing, and a character spent as a glossary")
        brief["picture_symbols"] = []
        brief["abstract"] = abstract
        brief["reader_contract"] = CONTRACT
        print(f"[StoryboarderV2] abstract of {len(abstract)} chars → {len(brief['abstract_claims'])} claims; "
              f"{len(brief['examples'])} example inputs from the paper")
        return brief

    async def write_v3(self, brief: dict, claims: list[dict], cast: Cast, panel_range: tuple[int, int], paper: PaperText) -> StoryboardV2:
        panels: list[PanelV2] = []
        short: Optional[tuple] = None   # a draft well under the range, kept in case the second one is no better
        note = ""
        for attempt in range(2):   # a reply without panels, or one no repair could parse, is asked once more: the brief is paid for
            try:
                data, usage = await _ask_json(self.text_engine, SYSTEM_STORY, story_prompt(brief, claims, cast, panel_range, paper) + note,
                                              thinking=True, temperature=0.7, max_tokens=24000, key="panels", paper=paper.body,
                                              cache_key=self.cache_key)
            except ValueError as e:   # json.JSONDecodeError is a ValueError ("Expecting ',' delimiter" in a 20-panel reply)
                if attempt:
                    if short is not None:
                        break
                    raise
                print(f"[StoryboarderV2] the story reply could not be read ({e}); asking once more")
                continue
            self._add_usage(usage)
            panels = []
            for i, item in enumerate(_panel_items(data.get("panels")), start=1):
                try:
                    panels.append(_panel_from_reply(item, i))
                except Exception as e:  # keep going on a malformed panel
                    print(f"[StoryboarderV2] skipping malformed panel {i}: {e}")
            if panels and attempt == 0 and len(panels) <= panel_range[0] - SHORT_DRAFT:
                # a draft well under the range is written once more (round 23 split no panels any more, and drafts of 12-13
                # panels stayed that short: Flexibility Trap's dropped the abstract's background, and its reader test fell)
                print(f"[StoryboarderV2] the story has {len(panels)} panels (asked for {panel_range[0]}-{panel_range[1]}); "
                      "asking for the whole story once more")
                short, panels = (panels, data), []
                note = (f"\n\nA FIRST DRAFT OF THIS STORY HAD ONLY {len(short[0])} PANELS: too few for a reader to see every claim "
                        f"of the abstract, its background and its evidence. Write the whole story again in {panel_range[0]}-"
                        f"{panel_range[1]} panels: give each claim of the abstract the panels it needs (the background said in "
                        "full, each step of the mechanism on its own diagram, each result on its own chart), never by saying "
                        "the same thing twice.")
                continue
            if panels:
                break
            print(f"[StoryboarderV2] the story reply had no panels (attempt {attempt + 1})")
        if short is not None and len(panels) <= len(short[0]):   # the second draft is no longer: the first one stands
            panels, data = short
        if not panels:
            raise ValueError("story model returned no panels")
        numbers = [p.n for p in panels]
        if len(set(numbers)) == len(numbers):
            panels.sort(key=lambda p: p.n)          # trust the model's numbering when it is unique; else keep reply order
        _normalize_v3_panels(panels, cast, claims, _ledger(paper.body))
        chosen = brief.get("running_example") if isinstance(brief.get("running_example"), dict) else {}
        if chosen.get("text"):   # the analysis's case, checked against the paper, with the parts every diagram draws
            example = {**chosen, "from_paper": bool(chosen.get("from_paper")),
                       "source": str(chosen.get("source") or "") if chosen.get("from_paper") else ""}
        else:
            example = _running_example(data.get("running_example"), brief, panels, paper.body)
        brief["running_example"] = example
        if not panel_range[0] <= len(panels) <= panel_range[1]:
            print(f"[StoryboarderV2] the story has {len(panels)} panels (asked for {panel_range[0]}-{panel_range[1]})")
        sb = StoryboardV2(title=str(data.get("title") or brief.get("title") or "Paper manga"),
                          summary=str(data.get("summary") or brief.get("key_idea") or ""), theme=cast.theme, language="en-US",
                          chapters=_chapters(panels), panels=panels, brief=brief, card=paper_card(brief), version=3,
                          abstract_claims=claims, running_example=example["text"])
        if chosen.get("text") and _example_tokens(sb) and sum(_example_beats(sb).values()) < 2:
            # a story that follows another case keeps its own: the lints, the designer and the quiz revision would
            # otherwise ask every panel for a case the story never told (round 25's review)
            own = _running_example(data.get("running_example"), brief, panels, paper.body)
            print(f"[StoryboarderV2] the story does not follow the analysis's running example; keeping its own: {own['text'][:120]}")
            brief["running_example"], sb.running_example = own, own["text"]
        return sb

    def _replace_panels(self, sb: StoryboardV2, cast: Cast, items, allowed: set[int], keep_boards: frozenset = frozenset(),
                        ledger=None) -> int:
        """Put the model's replacement panels in place by "n" (only numbers in ``allowed``); a reply without usable numbers
        is matched in order when it holds exactly one panel per allowed number. A panel in ``keep_boards`` (its board was
        valid) keeps that board when the reply drops it: the problems were elsewhere, and a recap must not vanish because
        a rewrite left out an unchanged field. Returns the number of panels replaced."""
        items = _panel_items(items)
        by_n: dict[int, dict] = {}
        for item in items:
            try:
                n = int(item.get("n"))
            except (TypeError, ValueError):
                continue
            if n in allowed and n not in by_n:
                by_n[n] = item
        if not by_n and items and len(items) == len(allowed):
            by_n = dict(zip(sorted(allowed), items))
        count = 0
        for i, old in enumerate(sb.panels):
            item = by_n.get(old.n)
            if item is None:
                continue
            try:
                new = _panel_from_reply(item, old.n)
            except Exception as e:  # keep the old panel
                print(f"[StoryboarderV2] skipping malformed replacement for panel {old.n}: {e}")
                continue
            if not (new.dialogue or new.narration or new.board):
                continue
            new.n, new.beat, new.title = old.n, new.beat or old.beat, new.title or old.title
            new.scene, new.visual = new.scene or old.scene, new.visual or old.visual
            if not new.shows and new.kind == old.kind and ("shows" not in item or (
                    new.visual == old.visual and new.beat in PICTURE_BEATS and not _settles_bet(new))):
                new.shows = old.shows   # a reply that left the key out, or the same picture: the same idea (a hook or the
                                        # bet may clear it on purpose: the rules let a story-only scene leave it empty)
            new.characters = new.characters or list(old.characters)
            if "claims" not in item and "abstract_claims" not in item:   # a reply that forgot the field keeps the claims
                new.claims = list(old.claims)
            if old.n in keep_boards and not (isinstance(new.board, dict) and new.board):
                new.kind, new.board = "board", old.board
            new.board = _kept_drawing(old.board, new.board)
            sb.panels[i] = new
            count += 1
        if count:
            _normalize_v3_panels(sb.panels, cast, sb.abstract_claims, ledger)
        return count

    def _split_room(self, sb: StoryboardV2, problems: dict[int, list[str]]) -> int:
        """How many panels one fix round may add: one chart panel for each flagged panel whose problems ask for one (a
        result no chart draws), at most FIX_NEW_PANELS, while the script stays within the job's range plus half the quiz
        revision's headroom (the other half is for the claims the test reader shows are missing)."""
        asks = sum(1 for found in problems.values() if any(CHART_PANEL_ASK in x for x in found))
        return max(0, min(asks, FIX_NEW_PANELS, self.max_panels + QUIZ_EXTRA_PANELS // 2 - len(sb.panels)))

    @staticmethod
    def _accepted_splits(items: list[dict], flagged: set[int], room: int, chartable: Optional[set] = None) -> tuple[list, set]:
        """The new panels of a fix reply ({"after": n}) that will be inserted, as (n, panel, item): at most one after each
        flagged panel, only after those in ``chartable`` (whose problems ask for a chart panel; any flagged panel when it
        is None), and ``room`` in all. A new panel whose "after" names no flagged panel belongs to the flagged panel
        returned just before it (a writer that counted in the new numbering). Also the flagged panels whose refused new
        panel was the rest of a split (not a chart board): their first half must not apply either, so the old panel stays
        whole (a first half alone stopped mid-explanation and the claim's details were lost); a refused chart leaves the
        panel's own rewrite standing (it lost the whole rewrite, round 23's review)."""
        prev, halves = None, []
        for i, item in enumerate(items):
            if item.get("after") is None:
                try:
                    prev = int(item.get("n"))
                except (TypeError, ValueError):
                    prev = None
                continue
            try:
                after = int(item.get("after"))
            except (TypeError, ValueError):
                after = None
            if after in flagged:
                halves.append((0, i, after, item))
            elif prev in flagged:   # after the halves that name their panel: a stray never takes a named half's place
                halves.append((1, i, prev, item))
            prev = None
        accepted: list[tuple[int, PanelV2, dict]] = []
        refused: set[int] = set()
        # a chart-ask panel's one slot goes to a board before a scene listed ahead of it
        board_first = lambda h: str(h[3].get("kind") or ("board" if h[3].get("board") else "")).lower() != "board" if chartable and h[2] in chartable else False
        for _, _, n, item in sorted(halves, key=lambda h: (h[0], board_first(h), h[1])):
            try:
                new = _panel_from_reply(item, 0)
            except Exception as e:   # keep going on a malformed panel: a split is never a reason to fail the job
                print(f"[StoryboarderV2] skipping malformed split panel: {type(e).__name__}: {e}")
                new = None
            if (new is not None and (new.dialogue or new.narration or new.board) and (chartable is None or n in chartable)
                    and len(accepted) < room and all(n != m for m, _, _ in accepted)):
                accepted.append((n, new, item))
            elif new is None or new.kind != "board" or chartable is None or n not in chartable:
                refused.add(n)   # a rest of a split (a scene, or a board the panel's problems never asked for)
        return accepted, refused - {n for n, _, _ in accepted}

    def _insert_splits(self, sb: StoryboardV2, cast: Cast, splits: list, paper_text: str) -> int:
        """The accepted second halves (_accepted_splits), each placed right after its panel (by object: the reply's own
        numbers can name the next panel) with its beat, chapter, setting and, unless the reply lists them, its claims. A
        fix round could only shorten a full teaching scene, and the claim's details went (the lint's own advice to give
        the rest a neighbouring panel was thrown away). Returns the number of panels added."""
        parents = {p.n: p for p in sb.panels}   # after _replace_panels: a flagged n holds its first half
        added: list[int] = []
        for n, new, item in splits:
            old = parents.get(n)
            if old is None:
                continue
            new.n, new.beat, new.chapter = 0, old.beat, old.chapter   # _finish_v3 numbers it
            new.scene = new.scene or old.scene
            new.characters = new.characters or list(old.characters)
            if "claims" not in item and "abstract_claims" not in item:   # the rest of the same claim
                new.claims = list(old.claims)
            sb.panels.insert(next(i for i, p in enumerate(sb.panels) if p is old) + 1, new)
            added.append(n)
        if added:
            _normalize_v3_panels(sb.panels, cast, sb.abstract_claims, _ledger(paper_text))
            _finish_v3(sb)
            print(f"[StoryboarderV2] fix round split panels {sorted(added)}")
        return len(added)

    def _demote_boards(self, sb: StoryboardV2, paper_text: str) -> None:
        """After the fix rounds, a board that is still invalid keeps what the paper supports (_salvage_board: problems
        the renderer copes with stay, unsupported entries go). Only when nothing usable is left is the panel drawn as a
        scene, with the board's title as narration unless the title holds a number the paper lacks. A scene panel loses
        a stray board."""
        for p in sb.panels:
            board = p.board if isinstance(p.board, dict) and p.board else None
            title = _plain(str(board.get("title") or "")) if board else ""
            if not _numbers_supported(title, paper_text):
                title = ""   # what made the board fail must not move into the narration
            if p.kind == "board":
                problems = board_issues(board, paper_text) if board else ["board: missing"]
                if not problems:
                    continue
                salvaged = _salvage_board(board, paper_text) if board else None
                if salvaged is not None:
                    print(f"[StoryboarderV2] panel {p.n}: board kept after the fix rounds without what the paper does not "
                          f"support ({len(problems)} problems before)")
                    p.board = salvaged
                    continue
                print(f"[StoryboarderV2] panel {p.n}: the board is still invalid; drawn as a scene ({'; '.join(problems)[:400]})")
                said = _content_tokens(" ".join(str(board.get(k) or "") for k in ("title", "subtitle", "note"))) if board else set()
                own = _content_tokens(p.narration)
                if own and said and len(own & said) >= 0.5 * len(own) and p.dialogue:
                    p.narration = ""   # a caption of the chart ("Attention weights for the verb 'making'") without the chart
                p.kind, p.board = "scene", None   # its title would caption a chart nobody sees ("Lower is better.")
                if not p.dialogue and not p.narration and title:
                    p.narration = title if title[-1] in ".!?" else f"{title}."
            elif p.board:
                p.board = None

    async def fix_v3(self, sb: StoryboardV2, cast: Cast, paper: PaperText, rounds: int = 2, extra: Optional[dict] = None,
                     only: Optional[set] = None, split: bool = True) -> int:
        """Lint, then one model call that rewrites the flagged panels (same n), at most ``rounds`` times; a board that is
        still invalid afterwards becomes a scene panel. ``extra`` ({n: [problem]}, e.g. whole-board findings of the
        fact-check) joins the first round; ``only`` limits the rounds to those panels (the last check after the quiz
        must not rewrite the scored script). Returns the number of panels replaced."""
        replaced = 0
        outside = [p for p in sb.panels if only is not None and p.n not in only]   # by object: a split renumbers them
        for r in range(rounds):
            problems = lint_v3(sb, paper.body)
            if r == 0:
                for n, found in (extra or {}).items():
                    problems.setdefault(n, []).extend(found)
            numbers = {p.n for p in sb.panels}
            problems = {n: v for n, v in problems.items() if n in numbers and v and (only is None or n in only)}
            if not problems:
                break
            print(f"[StoryboarderV2] lint round {r + 1}: problems on panels {sorted(problems)}")
            pictures = {n: v for n, v in problems.items() if all(PICTURE_PROBLEM_RE.match(x) for x in v)}
            if pictures:   # the art directions alone: the words of these panels stay as they are
                replaced += await self._fix_pictures(sb, cast, pictures)
                problems = {n: v for n, v in problems.items() if n not in pictures}
                if not problems:
                    continue
            needs_paper = any(x.startswith(("evidence", "claim", "fact-check", "reader", "the words say", "board check",
                                            "the board shows numbers in parentheses", "the board shows significance stars", "why",
                                            "unit", "the proof board draws", "the board shows one number", "diagram", "meta",
                                            "the payoff does not show", "this panel says much of"))
                              or (x.startswith("term") and "It means:" not in x)   # an abbreviation is spelled out from the paper
                              for found in problems.values() for x in found)
            room = self._split_room(sb, problems) if split else 0
            chartable = {n for n, v in problems.items() if any(CHART_PANEL_ASK in x for x in v)}
            evidence = _as_list(sb.brief.get("evidence"))
            asks = {n: keys for n, v in problems.items() for keys in [[c for x in v for c in [_chart_asked(x, evidence)] if c]] if keys}
            if room == 0 and chartable:   # no panel may be added: the ask becomes one this round can answer
                problems = {n: [_without_new_panel(x) for x in v] for n, v in problems.items()}
            data = None
            for attempt in range(2):   # a reply whose JSON stays broken after its repair is asked for once more (Right On Q
                try:                   # lost a whole round to a missing comma, and nine problems shipped)
                    data, usage = await _ask_json(self.text_engine, SYSTEM_STORY,
                                                  fix_prompt(sb, cast, problems, paper.body if needs_paper else "", _tables_section(paper.body),
                                                             room=room),
                                                  thinking=False, temperature=0.3, max_tokens=12000, key="panels", paper=paper.body,
                                                  cache_key=self.cache_key)
                    break
                except Exception as e:  # the script stays as it is; invalid boards are salvaged or demoted below
                    _reraise_account(e)
                    print(f"[StoryboarderV2] lint fix {'skipped' if attempt else 'failed, asking once more'}: {type(e).__name__}: {e}")
                    if attempt or not isinstance(e, ValueError):
                        self._skip("lint fix round", e)
                        break
            if data is None:
                break
            self._add_usage(usage)
            items = _panel_items(data.get("panels"))
            _mark_halves(items, set(problems))
            stray = sorted({str(x.get("n")) for x in items if x.get("after") is None and x.get("n") is not None and _reply_n(x) not in problems})
            if stray:
                print(f"[StoryboarderV2] fix round: ignored reply panels numbered {stray} (no flagged panel has that number)")
            # a new panel only for a chart a problem asks for (round 23: splits of long panels made 17-panel drafts into
            # 24-panel books that said the same number again)
            splits, refused = self._accepted_splits(items, set(problems), room, chartable)
            refused |= {x["_ambiguous"] for x in items if x.get("_ambiguous") is not None and x.get("after") in refused}
            done = _charted_asks(asks, splits, evidence)
            if done:   # each result gets one added chart per book (_uncharted_evidence), whatever becomes of it
                sb.brief["chart_asked"] = list(dict.fromkeys([*map(str, _as_list(sb.brief.get("chart_asked"))), *done]))
            for x in items:
                x.pop("_ambiguous", None)
            if refused:
                print(f"[StoryboarderV2] fix round: no new panel after panels {sorted(refused)}; they stay whole")

            def kept(x: dict) -> bool:
                try:
                    return int(x.get("n")) not in refused
                except (TypeError, ValueError):
                    return True
            old_boards = {p.n: p.board for p in sb.panels if isinstance(p.board, dict) and p.board}
            dropped = {_reply_n(x) for x in items if x.get("after") is None and ("board" in x and not x.get("board") or x.get("kind") == "scene")}
            for n, new, _ in splits:   # a rest marked "kind": "board" without its spec draws the board its first half dropped
                if new.kind == "board" and not (isinstance(new.board, dict) and new.board) and n in dropped and n in old_boards:
                    new.board = copy.deepcopy(old_boards[n])
            # a drop on purpose stands when the panel's own problems are about that board (a repeat, a fact-check finding)
            about = {n for n in dropped if any(x.startswith(("the board repeats", "fact-check", "a links board", "the board shows one number",
                                                             "the proof board draws only words"))
                                               for x in problems.get(n, []))}
            drawn_key = lambda b: _board_values(b) or frozenset([board_text({k: v for k, v in b.items() if k not in ("title", "subtitle", "note", "source")})])
            binder, ledger_now = _optional("ledger"), _ledger(paper.body)

            def as_drawn(board: dict) -> dict:   # cited cells bound and the shape normalized, as the old board already is
                try:
                    return _normalized_board(binder.bind_board(board, ledger_now) if binder and ledger_now else board) or board
                except Exception:
                    return board

            def moves(n: int, rest: dict) -> bool:
                old, rest = old_boards[n], as_drawn(rest)
                if n in dropped or (drawn_key(rest) == drawn_key(old) and rest.get("kind") == old.get("kind")):
                    return True
                title = str(rest.get("title") or "").strip().lower()
                if not title or title != str(old.get("title") or "").strip().lower():
                    return False
                values = _board_values(old)   # two charts of one title are one board only with (most of) the same numbers
                shared = len(_board_values(rest) & values)
                if not values or shared >= max(2, len(values) - 1):
                    return True
                cells = lambda b: {str(x.get("cell")) for x in _as_list(b.get("bars")) if isinstance(x, dict) and x.get("cell")}
                rows = lambda b: {tuple(c.split(".")[:2]) for c in cells(b)}
                return bool(cells(old) & cells(rest)) and rows(old) <= rows(rest)   # the same cells and rows, some values corrected
            moved = about | {n for n, new, _ in splits if isinstance(new.board, dict) and new.board and n in old_boards and moves(n, new.board)}
            replaced += self._replace_panels(sb, cast, [x for x in items if x.get("after") is None and kept(x)], set(problems),
                                             _valid_boards(sb.panels, paper.body, set(problems)) - moved, ledger=_ledger(paper.body))
            replaced += self._insert_splits(sb, cast, splits, paper.body)
        self._demote_boards(sb, paper.body)
        skip = {p.n for p in outside}   # the last check's corrections elsewhere ship as the checker wrote them
        bare = {n: v for n, v in _picture_problems(sb).items()
                if n not in skip and any("teaches nothing" in x or "drawn blank" in x for x in v)}
        if bare:   # a last rewrite left a teaching scene without its idea, or a board became a scene: one bounded call more
            replaced += await self._fix_pictures(sb, cast, bare)
        return replaced

    async def _fix_pictures(self, sb: StoryboardV2, cast: Cast, problems: dict[int, list[str]]) -> int:
        """One call that rewrites the art directions ("scene", "visual", "shows") of the panels in ``problems``
        (picture_fix_prompt); their words never change. Returns the number of panels changed."""
        try:
            data, usage = await _ask_json(self.text_engine, SYSTEM_STORY, picture_fix_prompt(sb, cast, problems), thinking=False,
                                          temperature=0.3, max_tokens=8000, key="panels", cache_key=self.cache_key)
        except Exception as e:  # the panels keep their pictures; the job record says the fix did not run
            _reraise_account(e)
            print(f"[StoryboarderV2] picture fix skipped: {type(e).__name__}: {e}")
            self._skip("picture fix", e)
            return 0
        self._add_usage(usage)
        flagged = {p.n: p for p in sb.panels if p.n in problems}
        changed = set()
        for item in _panel_items(data.get("panels")):
            p = flagged.get(_reply_n(item))
            if p is None:
                continue
            visual = _text(item.get("visual"))
            if visual:
                p.scene, p.visual = _text(item.get("scene")) or p.scene, visual
                changed.add(p.n)
            if SCENE_SHOWS and "shows" in item and (_text(item.get("shows")) or p.beat not in PICTURE_BEATS or _settles_bet(p)):
                p.shows = _text(item.get("shows"))   # a teaching scene never loses its picture's idea (round 23: none has one)
                changed.add(p.n)
            for name in cast.names():   # a cast member the new picture stages is in the panel (the art prompt says who is not)
                if name not in p.characters and _stages(name, f"{p.scene} {p.visual} {p.shows}"):
                    p.characters.append(name)
        if changed:
            print(f"[StoryboarderV2] picture fix: panels {sorted(changed)}")
            # the same screen every art direction gets (people the scene names are drawn): the picture step runs after the last fit
            await self.fit(sb, "en-US", only=changed)
        else:   # a reply that names no flagged panel changes nothing, and the pictures ship as they were
            print(f"[StoryboarderV2] picture fix changed none of panels {sorted(problems)}: reply {json.dumps(data, ensure_ascii=False)[:300]}")
        return len(changed)

    async def clarity_check(self, sb: StoryboardV2, only: Optional[set] = None) -> dict[int, list[str]]:
        """The checker reads the English script as a newcomer (clarity_prompt): {n: ["reader (kind): ... Fix: ..."]} for a
        fix round, at most CLARITY_MAX problems and 2 per panel, on the panels in ``only`` when given (after a quiz
        revision: the new and rewritten ones); the list is kept in sb.brief["clarity"] for the job report (a revision's
        problems are added to it). Best effort: a check that cannot run is recorded and changes nothing."""
        try:
            data, usage = await _ask_check(self.check_engine, clarity_prompt(sb, only), thinking=True, temperature=0.1, max_tokens=8000,
                                           key="problems", cache_key=self.cache_key)
        except Exception as e:  # the script stays as it is
            _reraise_account(e)
            print(f"[StoryboarderV2] reader check skipped: {type(e).__name__}: {e}")
            self._skip("reader check", e)
            return {}
        self._add_usage(usage)
        numbers = {p.n for p in sb.panels}
        out: dict[int, list[str]] = {}
        kept: list[dict] = []
        for item in _as_list(data.get("problems")):
            if not isinstance(item, dict) or len(kept) >= CLARITY_MAX:
                continue
            try:
                n = int(item.get("n"))
            except (TypeError, ValueError):
                continue
            kind, problem, fix = _text(item.get("kind")).lower(), _text(item.get("problem")), _text(item.get("fix"))
            if n not in numbers or (only is not None and n not in only) or not problem or len(out.get(n, [])) >= 2:
                continue
            kind = kind if kind in CLARITY_KINDS else "clarity"
            out.setdefault(n, []).append(f"reader ({kind}): {problem}" + (f" Fix: {fix}" if fix else ""))
            kept.append({"n": n, "kind": kind, "problem": problem, "fix": fix})
        if kept:
            print(f"[StoryboarderV2] reader check: {len(kept)} problems on panels {sorted(out)}")
        sb.brief["clarity"] = (list(sb.brief.get("clarity") or []) if only is not None else []) + kept
        return out

    async def board_check(self, sb: StoryboardV2, paper: PaperText, only: Optional[set] = None, again: bool = False) -> dict[int, list[str]]:
        """The checker compares each board with its panel's words (board_check_prompt): {n: ["board check (kind): ... Fix:
        ..."]} for a fix round, at most BOARD_CHECK_MAX problems and 2 per board, on the panels in ``only`` when given; kept
        in sb.brief["board_check"] (a second look at rewritten boards, ``again``, in sb.brief["board_recheck"]) for the job
        report. Best effort, like the reader check."""
        if not any(_checked_board(p) and (only is None or p.n in only) for p in sb.panels):
            return {}
        try:
            data, usage = await _ask_check(self.check_engine, board_check_prompt(sb, paper.body, only, again), thinking=True, temperature=0.1,
                                           max_tokens=8000, key="problems", paper=paper.body, cache_key=self.cache_key)
        except Exception as e:  # the script stays as it is
            _reraise_account(e)
            print(f"[StoryboarderV2] board {'re-check' if again else 'check'} skipped: {type(e).__name__}: {e}")
            self._skip("board re-check" if again else "board check", e)
            return {}
        self._add_usage(usage)
        boards = {p.n for p in sb.panels if _checked_board(p)}
        out: dict[int, list[str]] = {}
        kept: list[dict] = []
        for item in _as_list(data.get("problems")):
            if not isinstance(item, dict) or len(kept) >= BOARD_CHECK_MAX:
                continue
            try:
                n = int(item.get("n"))
            except (TypeError, ValueError):
                continue
            kind, problem, fix = _text(item.get("kind")).lower(), _text(item.get("problem")), _text(item.get("fix"))
            if n not in boards or (only is not None and n not in only) or not problem or len(out.get(n, [])) >= 2:
                continue
            kind = kind if kind in BOARD_CHECK_KINDS else "comparison"
            out.setdefault(n, []).append(f"board check ({kind}): {problem}" + (f" Fix: {fix}" if fix else ""))
            kept.append({"n": n, "kind": kind, "problem": problem, "fix": fix})
        if kept:
            print(f"[StoryboarderV2] board check: {len(kept)} problems on panels {sorted(out)}")
        key = "board_recheck" if again else "board_check"
        sb.brief[key] = (list(sb.brief.get(key) or []) if only is not None else []) + kept
        return out

    async def recap_evidence(self, sb: StoryboardV2) -> None:
        """Which chart or table of the story shows each recap line (recap_evidence_prompt), kept in
        sb.brief["recap_evidence"] as [{"line", "panels"}] for _honest_recap, which ticks the lines a chart shows (a
        relayout reads the same answers, so it needs no model call). Best effort: without an answer _honest_recap judges
        by the charts' numbers and the claims they are bound to."""
        lines = [str(x) for p in _recap_lists(sb) for x in _as_list(p.board.get("items"))]
        sb.brief.pop("recap_evidence", None)
        if not lines:
            return
        if not _exhibits(sb):   # nothing drawn: every line gets the dot, no question to ask
            sb.brief["recap_evidence"] = [{"line": x, "panels": [], "shown": False} for x in lines]
            return
        try:
            data, usage = await _ask_check(self.check_engine, recap_evidence_prompt(sb), thinking=True, temperature=0.1,
                                           max_tokens=8000, key="lines", cache_key=self.cache_key)
        except Exception as e:  # the program's rule decides the marks
            _reraise_account(e)
            print(f"[StoryboarderV2] recap evidence skipped: {type(e).__name__}: {e}")
            self._skip("recap evidence", e)
            return
        self._add_usage(usage)
        charts = {p.n for p in _exhibits(sb)}
        found: dict[int, list[int]] = {}
        for item in _as_list(data.get("lines")):
            if not isinstance(item, dict):
                continue
            try:
                i = int(item.get("i"))
            except (TypeError, ValueError):
                continue
            panels = []
            for n in _as_list(item.get("panels")):
                try:
                    if int(n) in charts:
                        panels.append(int(n))
                except (TypeError, ValueError):
                    pass
            if 0 <= i < len(lines) and i not in found:
                found[i] = sorted(set(panels))
        sb.brief["recap_evidence"] = [{"line": lines[i], "panels": found[i], "shown": bool(found[i])} for i in sorted(found)]
        print(f"[StoryboarderV2] recap evidence: {sum(bool(v) for v in found.values())} of {len(lines)} lines shown by a chart")

    def _diagram_problems(self, sb: StoryboardV2, board: dict, paper_text: str, notes: Optional[list] = None) -> list[str]:
        """What is wrong with a designed diagram, numbers first: a number on it that the paper lacks (a made-up running
        example's own numbers are fine, as in the bubbles), a table citation drawn as it is ("@T2.r5.c1"), what the
        normalizer had to drop (an arrow it could not read vanished without a word), then the layout lint."""
        boards, evidence, diagram = _optional("boards"), _optional("evidence"), _optional("diagram")
        out: list[str] = []
        cite = _optional("ledger")
        allowed = [_illustrative_example(sb)]
        for _path, text in (diagram.texts(board) if diagram is not None else []):
            if evidence is not None:
                try:
                    out += [f'{DIAGRAM_NUMBER}"{text}": {x}' for x in evidence.verify_text_numbers(text, paper_text, allowed=allowed)]
                except Exception as e:
                    print(f"[StoryboarderV2] diagram number check failed: {type(e).__name__}: {e}")
            if cite is not None and cite.REF_RE.search(text):
                out.append(f'{DIAGRAM_NUMBER}"{text}": a table citation is drawn as it is written; write the number itself, '
                           "exactly as the paper prints it")
        out += [f"the design: {x}" for x in (notes or [])]
        out += list(boards.diagram_problems(board)) if boards is not None else []
        return out

    async def design_diagrams(self, sb: StoryboardV2, paper: PaperText, requests: Optional[dict] = None,
                              only: Optional[set] = None) -> set[int]:
        """Every diagram board whose drawing is missing or was made for another brief (_diagram_key), and every one with
        ``requests`` (what a checker found in it), gets its elements from the check model (diagram_prompt) and a
        redesign while the lint finds problems, at most DIAGRAM_TRIES calls; a design without wrong numbers beats one
        with them, then the fewest problems wins. Requests that are all DIAGRAM_LAYOUT (a correction made a label longer)
        keep every text: a redesign that changes one is not taken, so nothing unchecked is drawn. A diagram that cannot
        be designed becomes a flow board of its brief, never a blank board; a drawing whose redesign failed keeps its
        drawing, marked as tried for its brief (no later call tries it again unasked). ``only``: these panels alone.
        Recorded in sb.brief["diagrams"] ({n: {"tries", "problems"}}). Returns the panels whose drawing changed. Why diagrams: on 24 teaching panels of round
        22 a blind judge found the same words beside a diagram of the mechanism clearer than the shipped picture 22 times
        out of 24."""
        boards, diagram = _optional("boards"), _optional("diagram")
        if boards is None or diagram is None:
            return set()
        requests = {int(k): list(v) for k, v in (requests or {}).items() if v}
        todo = [p for p in sb.panels if p.kind == "board" and _is_diagram(p.board) and (only is None or p.n in only)
                and (not p.board.get("elements") or p.board.get("designed_for") != _diagram_key(p) or p.n in requests)]
        if not todo:
            return set()
        gate = asyncio.Semaphore(DIAGRAM_PARALLEL)
        words = diagram.read_texts   # every text and mark a reader reads; only the step badges may change
        wrong = lambda found: sum(1 for x in found if x.startswith(DIAGRAM_NUMBER))

        async def one(p: PanelV2):
            async with gate:
                best, best_probs, tries = None, None, 0
                current = boards.normalize_board(p.board) if p.board.get("elements") else None
                asked = list(requests.get(p.n, []))
                keep = bool(asked) and current is not None and all(x.startswith(DIAGRAM_LAYOUT) for x in asked)
                # a relayout keeps every text, so a label's length is never its to fix (the fact-check's text stands)
                movable = (lambda xs: [x for x in xs if LABEL_LENGTH not in x]) if keep else (lambda xs: xs)
                keep_note = ["keep every text exactly as it is in YOUR PREVIOUS DESIGN (they were checked against the paper): "
                             "only move, resize or re-route things so that everything fits"] if keep else []
                if keep:   # the drawing as it is stays the one to beat
                    best, best_probs = current, movable(self._diagram_problems(sb, current, paper.body))
                    asked = asked + keep_note
                problems = asked or None
                previous = {"elements": current["elements"]} if problems and current else None
                failure, designs = None, 0
                for _ in range(DIAGRAM_TRIES + 1):   # a call that fails outright is asked again, once
                    if designs >= DIAGRAM_TRIES or tries - designs >= 2:
                        break
                    tries += 1
                    try:
                        data, usage = await _ask_json(self.check_engine, "", diagram_prompt(sb, p, problems, previous, paper.body),
                                                      thinking=True, temperature=0.3, max_tokens=16000, key="elements",
                                                      paper=paper.body, cache_key=self.cache_key)
                    except Exception as e:   # an empty or broken reply is asked again (one failed call made a flow of the brief)
                        _reraise_account(e)
                        print(f"[StoryboarderV2] diagram {p.n} not designed: {type(e).__name__}: {e}")
                        failure = e
                        continue
                    self._add_usage(usage)
                    designs += 1
                    spec = {k: v for k, v in p.board.items() if k not in ("elements", "designed_for")}
                    spec["elements"] = data.get("elements") if isinstance(data.get("elements"), list) else []
                    norm, notes = boards._normalize(spec)
                    if not isinstance(norm, dict) or not norm.get("elements"):
                        problems = ["the design had no usable elements" + (f" ({'; '.join(notes[:6])})" if notes else "")]
                        previous = None
                        continue
                    found = movable(self._diagram_problems(sb, norm, paper.body, notes))
                    if keep and words(norm) != words(current):   # never taken: a changed text was never checked
                        problems = [f"it changed the texts, which must stay exactly as they are: {', '.join(words(current))}"] + found
                        previous = {"elements": norm["elements"]}
                        continue
                    if best is None or (wrong(found), len(found)) < (wrong(best_probs), len(best_probs)):
                        best, best_probs = norm, found
                    if not found:
                        break
                    problems, previous = found + keep_note, {"elements": norm["elements"]}   # a retry still keeps the texts
                if best is None and failure is not None:
                    self._skip(f"diagram {p.n}", failure)
                return p, best, best_probs, tries

        results = await asyncio.gather(*(one(p) for p in todo))
        record = dict(sb.brief.get("diagrams") or {})
        changed: set[int] = set()
        for p, best, probs, tries in results:
            if best is None:
                if p.board.get("elements"):   # a redesign that failed keeps the drawing it had, tried for this brief
                    p.board = dict(p.board, designed_for=_diagram_key(p))
                    continue
                p.board = _diagram_fallback(p.board)
                record[str(p.n)] = {"tries": tries, "fallback": True}
                changed.add(p.n)
                continue
            best = dict(best, designed_for=_diagram_key(p))
            if best != p.board:
                changed.add(p.n)
            p.board = best
            record[str(p.n)] = {"tries": tries, "problems": [x[:160] for x in probs]}
        sb.brief["diagrams"] = record
        print(f"[StoryboarderV2] diagrams designed: {sorted(p.n for p, b, _, _ in results if b is not None)}"
              + (f", fell back: {sorted(p.n for p, b, _, _ in results if b is None)}" if any(b is None for _, b, _, _ in results) else ""))
        return changed

    async def picture_check(self, sb: StoryboardV2, only: Optional[set] = None, again: bool = False,
                            append: bool = False) -> dict[int, list[str]]:
        """The checker reads each teaching scene's picture plan beside its words (picture_check_prompt): {n: ["picture
        check (kind): ... Fix: ..."]} for the picture fix, at most PICTURE_CHECK_MAX panels and one problem each, on the
        panels in ``only`` when given; kept in sb.brief["picture_check"] for the job report (a look at some panels, or a
        later look with ``append``, adds to the earlier record). Best effort, like the reader check."""
        if not any(_picture_checked(p) and (only is None or p.n in only) for p in sb.panels):
            return {}
        try:
            data, usage = await _ask_check(self.check_engine, picture_check_prompt(sb, only, again), thinking=True, temperature=0.1,
                                           max_tokens=8000, key="problems", cache_key=self.cache_key)
        except Exception as e:  # the pictures stay as they are
            _reraise_account(e)
            print(f"[StoryboarderV2] picture check skipped: {type(e).__name__}: {e}")
            self._skip("picture check", e)
            return {}
        self._add_usage(usage)
        numbers = {p.n for p in sb.panels if _picture_checked(p)}
        out: dict[int, list[str]] = {}
        kept: list[dict] = []
        for item in _as_list(data.get("problems")):
            if not isinstance(item, dict) or len(kept) >= PICTURE_CHECK_MAX:
                continue
            try:
                n = int(item.get("n"))
            except (TypeError, ValueError):
                continue
            kind, problem, fix = _text(item.get("kind")).lower(), _text(item.get("problem")), _text(item.get("fix"))
            if n not in numbers or (only is not None and n not in only) or not problem or n in out:
                continue
            kind = kind if kind in PICTURE_CHECK_KINDS else "still"
            out[n] = [f"picture check ({kind}): {problem}" + (f" Fix: {fix}" if fix else "")]
            kept.append({"n": n, "kind": kind, "problem": problem, "fix": fix})
        if kept:
            print(f"[StoryboarderV2] picture check: {len(kept)} problems on panels {sorted(out)}")
        sb.brief["picture_check"] = (list(sb.brief.get("picture_check") or []) if only is not None or append else []) + kept
        return out

    async def _check_pictures(self, sb: StoryboardV2, cast: Cast, paper: PaperText, only: Optional[set] = None,
                              append: bool = False, linted: Optional[set] = None) -> None:
        """The picture check, and a picture fix for what it finds (the art directions only: the words stay); the rewritten
        pictures get one more look, with what the lint rejects in them, and one more fix, and nothing loops. A rewrite that
        draws a symbol before any panel's words say what it stands for gets one fix round of its words, where a character
        names it (Hivemind's dry run: one box repeating itself beside three boxes giving the same answer, the paper's point,
        drew the box and the blocks before any panel said they were models and answers). ``linted``: the panels whose
        lint picture problems join a look at every panel (those in ``only`` join a look at some)."""
        keys = _symbol_keys(sb)
        rewritten: set[int] = set()
        for again in (False, True):
            found = await self.picture_check(sb, only, again=again, append=append)
            lint_on = only if only is not None else linted
            if lint_on:   # what the lint rejects in these pictures (a writing prop, a repeat, a role word in "shows")
                for n, problems in lint_v3(sb, paper.body).items():
                    picture = [x for x in problems if PICTURE_PROBLEM_RE.match(x)]
                    if n in lint_on and picture:
                        found.setdefault(n, []).extend(picture)
            if not found:
                break
            before = {p.n: (p.scene, p.visual, p.shows) for p in sb.panels}
            await self._fix_pictures(sb, cast, found)
            only = {p.n for p in sb.panels if p.n in found and (p.scene, p.visual, p.shows) != before[p.n]}
            rewritten |= only
            if not only:
                break
        unkeyed = {n: v for n, v in _symbol_keys(sb).items() if n in rewritten and len(v) > len(keys.get(n, []))}
        if unkeyed:   # words only, no new panel; the new words are fact-checked later (their checked view changed)
            views = {_checked_view(p) for p in sb.panels}
            await self.fix_v3(sb, cast, paper, rounds=1, extra=unkeyed, only=set(unkeyed), split=False)
            await self.fit(sb, "en-US", only={p.n for p in sb.panels if _checked_view(p) not in views})

    async def _sync_pictures(self, sb: StoryboardV2, cast: Cast) -> None:
        """A scene whose "visual" or "shows" alone the last fact-check corrected gets the other one rewritten to agree: the
        image model draws "visual" and the art check judges "shows" (in 24 books of rounds 14-15, 22 scenes had only one of
        the two corrected, and LongZero's weaker-model pile was drawn the size the old "visual" said)."""
        parts: dict[int, set] = {}
        for fix in getattr(self, "last_fixes", []) or []:
            n, _, part = str(fix.get("id", "")).partition(".")
            if part in ("v", "s") and n.isdigit():
                parts.setdefault(int(n), set()).add(part)
        problems: dict[int, list[str]] = {}
        for p in sb.panels:
            done = parts.get(p.n, set())
            if p.kind != "scene" or len(done) != 1 or not (p.visual.strip() and p.shows.strip()):
                continue
            fixed, other = ("shows", "visual") if done == {"s"} else ("visual", "shows")
            problems[p.n] = [f'picture: the fact-check corrected "{fixed}" to "{getattr(p, fixed)}", and "{other}" still describes the '
                             f'old picture: rewrite "{other}" so both describe the same picture, keeping "{fixed}" as it is']
        self.last_fixes = []
        if problems:   # the reply's fields stand together: putting one back could split the pair again (the checker reads it again)
            await self._fix_pictures(sb, cast, problems)

    async def _redraw_flagged(self, sb: StoryboardV2, paper: PaperText, flags: dict) -> dict[int, list[str]]:
        """The fact-check's whole-board findings on diagrams go to the designer at once (the writer only sees a diagram's
        brief and labels, and its rewrite of the brief threw the drawing away: Flexibility Trap's "the fix" became eight
        blank boxes); its layout requests (a corrected label grew) only move the drawing. Returns the other findings,
        for the writer's fix round."""
        drawing = _drawing_flags(sb, flags)
        if drawing:
            await self.design_diagrams(sb, paper, requests=drawing)
        return {n: v for n, v in flags.items() if n not in drawing}

    def _unchecked(self, sb: StoryboardV2) -> set[int]:
        """The panels whose English text the fact-check has not read in its present form (new, rewritten or shortened)."""
        return {p.n for p in sb.panels if _checked_view(p) not in self.checked}

    async def _check_script(self, sb: StoryboardV2, cast: Cast, paper: PaperText, changed_only: bool = False) -> None:
        """The reader check first, and its fix (with a lint round after it, so a claim a rewrite dropped finds another
        panel) and fit; then the fact-check reads the final English text, rewrites included, and the boards it flags get
        one more fix round. ``changed_only`` (after a quiz revision): both checks look only at the panels the fact-check
        has not read in their present form; the rest of the script was checked before and has not changed."""
        only = self._unchecked(sb) if changed_only else None
        if only is not None and not only:   # a changed card alone still gets its fact-check
            await self.factcheck(sb, paper, panels=only)
            await self._sync_pictures(sb, cast)
            return
        clarity, board = await asyncio.gather(self.clarity_check(sb, only), self.board_check(sb, paper, only))
        # a finding on a diagram goes to its designer now, while the numbers hold, and to the writer as well (it may be
        # about the words), told that the drawing was redone; the second look below reads the new drawing
        seen = {(_checked_view(p), _drawn_board(p.board)) for p in sb.panels}   # by content: a fix round may renumber
        drawing = _drawing_flags(sb, board)
        await self.design_diagrams(sb, paper, requests=drawing)
        board = {n: [x + DIAGRAM_REDRAWN if n in drawing else x for x in v] for n, v in board.items()}
        found = {n: list(clarity.get(n, [])) + list(board.get(n, [])) for n in sorted(set(clarity) | set(board))}
        if found:
            views = {_checked_view(p) for p in sb.panels}
            await self.fix_v3(sb, cast, paper, rounds=2, extra=found)
            await self.fit(sb, "en-US")
            await self.design_diagrams(sb, paper)   # a diagram the rewrite gave a new brief is drawn before the second look
            redone = {p.n for p in sb.panels if _checked_board(p) and (_checked_view(p), _drawn_board(p.board)) not in seen}
            again = await self.board_check(sb, paper, redone, again=True) if board and redone else {}
            if again:   # the rewritten boards get one more look, and what it still finds one more fix
                await self.design_diagrams(sb, paper, requests=_drawing_flags(sb, again))
                views = {_checked_view(p) for p in sb.panels}
                await self.fix_v3(sb, cast, paper, rounds=1, extra=again, only=set(again), split=False)
                await self.fit(sb, "en-US", only={p.n for p in sb.panels if _checked_view(p) not in views})
                await self.design_diagrams(sb, paper)
        flags: dict[int, list[str]] = {}
        await self.factcheck(sb, paper, board_flags=flags, panels=self._unchecked(sb) if changed_only else None)
        await self._sync_pictures(sb, cast)
        flags = await self._redraw_flagged(sb, paper, flags)
        if flags:
            await self.fix_v3(sb, cast, paper, rounds=1, extra=flags)
        await self.design_diagrams(sb, paper)   # a rewrite that gave a diagram a new brief

    async def _last_check(self, sb: StoryboardV2, cast: Cast, paper: PaperText) -> None:
        """Lines that changed after their fact-check (shortened by fit, rewritten by the last fix round, a flagged board
        rewritten, an undone revision's leftovers) get one more check before translation; a board it flags gets a fix
        round (the lint checks its numbers), and nothing loops."""
        flags: dict[int, list[str]] = {}
        await self.factcheck(sb, paper, board_flags=flags, panels=self._unchecked(sb))
        views = {p.n: _checked_view(p) for p in sb.panels}
        await self._sync_pictures(sb, cast)
        synced = {p.n for p in sb.panels if _checked_view(p) != views.get(p.n)}
        if synced:   # a drawn claim the sync rewrote gets the checker's read; its corrections are final (nothing loops)
            await self.factcheck(sb, paper, panels=synced)
            self.last_fixes = []
        flags = await self._redraw_flagged(sb, paper, flags)
        if flags:   # the flagged boards only (the quiz scored the rest), then the lines that round changed are fitted
            before = {_checked_view(p) for p in sb.panels}
            # no new panel after the last read (a chart inserted here shipped unread)
            await self.fix_v3(sb, cast, paper, rounds=1, extra=flags, only=set(flags), split=False)
            await self.fit(sb, "en-US", only={p.n for p in sb.panels if _checked_view(p) not in before})
        await self.design_diagrams(sb, paper)   # a rewrite that gave a diagram a new brief
        # nothing is drawn after this check that it has not read: a diagram drawn since gets the checker's read, its
        # corrections are final, and a label they lengthen only moves the drawing around it (round 23's review: the
        # design made after the last check shipped labels nobody had read)
        drawn = {p.n for p in sb.panels if p.kind == "board" and _is_diagram(p.board)} & self._unchecked(sb)
        if drawn:
            final: dict[int, list[str]] = {}
            await self.factcheck(sb, paper, board_flags=final, panels=drawn)
            layout = {n: [x for x in v if x.startswith(DIAGRAM_LAYOUT)] for n, v in final.items()}
            layout = {n: v for n, v in layout.items() if v}
            if layout:   # a relayout that keeps every text, and nothing else
                await self.design_diagrams(sb, paper, requests=layout, only=set(layout))
            left = {n: [x for x in v if not x.startswith(DIAGRAM_LAYOUT)] for n, v in final.items()}
            for n, v in left.items():
                if v:
                    self._skip(f"diagram {n} after the last check", ValueError("; ".join(v)[:300]))

    async def _run_quiz(self, quiz, sb: StoryboardV2, questions: Optional[list] = None) -> dict:
        """services.abstract_quiz.run_quiz on the teaching panels (the claims board and the recap only list the claims, so a
        reader must learn them from the story itself), normalized: float score (None on failure), failed claim ids, its
        usage counted."""
        taught = copy.copy(sb)
        taught.panels = _teaching_panels(sb.panels)
        result = await quiz.run_quiz(self.text_engine, self.fast_engine, taught, sb.abstract_claims,
                                     abstract_text=str(sb.brief.get("abstract") or ""), questions=questions)
        if not isinstance(result, dict):
            raise ValueError("the abstract quiz returned no result")
        if isinstance(result.get("usage"), dict):
            self._add_usage(result["usage"])
        out = {k: v for k, v in result.items() if k != "usage"}
        try:
            out["score"] = None if result.get("score") is None else float(result["score"])
        except (TypeError, ValueError):
            out["score"] = None
        failed: list[int] = []
        for x in _as_list(result.get("failed_claim_ids", result.get("failed_claims", result.get("failed")))):
            if isinstance(x, dict):   # failed questions instead of ids
                failed += _claim_ids(x.get("claim_ids", x.get("claims", x.get("claim_id", x.get("claim")))))
            else:
                failed += _claim_ids(x)
        out["failed_claim_ids"] = list(dict.fromkeys(failed))
        out.setdefault("questions", [])
        return out

    async def _revise_for_quiz(self, quiz, sb: StoryboardV2, cast: Cast, paper: PaperText, panel_range: tuple[int, int], result: dict) -> bool:
        """One revision round for the claims the test reader missed (services.abstract_quiz.revise_prompt): revised panels
        come back with their "n", new ones with "after" and are placed inside their chapter by _insert_position (a few at
        most). Then lint/fix, fact-check and fit run again. The script only changes when the reply applies."""
        failed = [c for c in sb.abstract_claims if c.get("id") in set(result["failed_claim_ids"])]
        if not failed:
            return False
        room = max(0, min(len(failed), panel_range[1] + QUIZ_EXTRA_PANELS - len(sb.panels)))
        for attempt in range(2):   # a reply with no usable panel (BERT's left six claims untaught) is asked once more
            data, usage = await _ask_json(self.text_engine, SYSTEM_STORY, quiz.revise_prompt(sb, failed, cast, paper_text=paper.body, quiz=result),
                                          thinking=True, temperature=0.4, max_tokens=12000, key="panels", paper=paper.body,
                                          cache_key=self.cache_key)
            self._add_usage(usage)
            items = [x for key in ("panels", "replace", "new_panels", "revised_panels", "revisions") for x in _panel_items(data.get(key))]
            if any(isinstance(x, dict) and (x.get("dialogue") or x.get("narration") or x.get("board")) for x in items):
                break
            print(f"[StoryboarderV2] the quiz revision reply held no usable panel (keys {list(data)[:5]}); asking once more")
        work = copy.copy(sb)
        work.panels = panels = copy.deepcopy(sb.panels)
        numbers = {p.n for p in panels}
        last = max((p.n for p in panels if p.beat != "recap"), default=0)
        for item in items:   # a new panel numbered past the script ("n": 19) instead of placed ("after": 12) still counts
            try:
                n = int(item.get("n")) if item.get("after") is None and item.get("n") is not None else None
            except (TypeError, ValueError):
                n = None
            if n is not None and n not in numbers:
                item["after"] = max(1, min(n - 1, last))
        replaced = self._replace_panels(work, cast, [x for x in items if x.get("after") is None], numbers,
                                        _valid_boards(panels, paper.body), ledger=_ledger(paper.body))
        inserts = []
        for i, item in enumerate(x for x in items if x.get("after") is not None):
            if len(inserts) >= room:
                break
            try:
                new, after = _panel_from_reply(item, 0), int(item.get("after"))
            except (TypeError, ValueError) as e:  # keep going on a malformed panel
                print(f"[StoryboarderV2] skipping malformed quiz panel: {e}")
                continue
            if not (new.dialogue or new.narration or new.board):
                continue
            new.beat = _normalize_beat(new.beat, "proof")
            new.chapter = BEAT_CHAPTER[new.beat]
            inserts.append((_insert_position(panels, new, after), i, new))
        if not replaced and not inserts:   # the reader still misses these claims: say so in the job's quality record
            print(f"[StoryboarderV2] WARNING: the quiz revision changed nothing ({len(items)} panels in the reply)")
            self._skip("quiz revision", ValueError(f"the reply changed no panel ({len(items)} panels in it); claims "
                                                   f"{[c['id'] for c in failed]} stay untaught"))
            return False
        for pos, _i, new in sorted(inserts, key=lambda t: (t[0], t[1]), reverse=True):
            panels.insert(pos, new)
        _normalize_v3_panels(panels, cast, sb.abstract_claims, _ledger(paper.body))
        sb.panels = panels
        _finish_v3(sb)
        print(f"[StoryboarderV2] quiz revision for claims {[c['id'] for c in failed]}: {replaced} panels revised, {len(inserts)} added")
        await self.fix_v3(sb, cast, paper, rounds=2)
        # the revision's diagrams are drawn before its checks and its quiz: a new diagram was board-checked, fact-checked
        # and scored as its title alone, and the gate undid or kept a round on a script that was not the one shipped
        await self.design_diagrams(sb, paper)
        await self._check_script(sb, cast, paper, changed_only=True)
        await self.fit(sb, "en-US")
        return True

    async def quiz_gate(self, sb: StoryboardV2, cast: Cast, paper: PaperText, panel_range: tuple[int, int]) -> dict:
        """The abstract quiz on the visible text of the teaching panels (services.abstract_quiz). While any claim fails (the
        owner's bar is the whole abstract, not a share of it), a revision round for the failed claims and the same quiz
        again, at most QUIZ_REVISIONS rounds; a round that lowers the score is undone and the next one tries again. sb.quiz is the last kept
        result, with "before" (the first) and "rounds" (each round's score). Never raises: a quiz that fails leaves the
        script as it is and records the error in sb.quiz."""
        quiz = _optional("abstract_quiz")
        if quiz is None or not sb.abstract_claims:
            sb.quiz = {"score": None, "failed_claim_ids": [], "questions": [],
                       "error": "abstract quiz unavailable" if quiz is None else "no abstract claims"}
            return sb.quiz
        try:
            first = await self._run_quiz(quiz, sb)
            if first["score"] is None:   # intermittent (BERT's re-run lost its gate this way; a replay of the same script ran)
                print(f"[StoryboarderV2] the abstract quiz could not run ({first.get('error')}); running it once more")
                first = await self._run_quiz(quiz, sb)
        except Exception as e:  # the quiz is a gate, never a reason to fail the job
            _reraise_account(e)
            print(f"[StoryboarderV2] WARNING: abstract quiz failed: {type(e).__name__}: {e}")
            self._skip("abstract quiz", e)
            sb.quiz = {"score": None, "failed_claim_ids": [], "questions": [], "error": f"{type(e).__name__}: {e}"}
            return sb.quiz
        sb.quiz = first
        print(f"[StoryboarderV2] abstract quiz: score {first['score']}, failed claims {first['failed_claim_ids']}")
        if first["score"] is None:   # run_quiz never raises: a quiz that could not run comes back with its error
            self._skip("abstract quiz", RuntimeError(first.get("error") or "no score"))
            return sb.quiz
        before = {"score": first["score"], "failed_claim_ids": first["failed_claim_ids"]}
        current, rounds = first, []
        for round_ in range(QUIZ_REVISIONS):
            if not current["failed_claim_ids"]:
                break
            kept = copy.deepcopy((sb.panels, sb.card, sb.brief))
            def undo(reason: str) -> None:   # a round that cannot be measured or lost ground goes: the shipped script is the scored one
                sb.panels, sb.card, sb.brief = kept
                _finish_v3(sb)
                sb.quiz = {**sb.quiz, "rounds": rounds, "undone": round_ + 1}
                print(f"[StoryboarderV2] revision {round_ + 1} undone: {reason}")

            try:
                if not await self._revise_for_quiz(quiz, sb, cast, paper, panel_range, current):
                    break
            except Exception as e:  # keep the script and the last result
                _reraise_account(e)
                print(f"[StoryboarderV2] WARNING: quiz revision failed: {type(e).__name__}: {e}")
                self._skip("quiz revision", e)
                undo("it failed")
                sb.quiz = {**sb.quiz, "revision_error": f"{type(e).__name__}: {e}"}
                break
            try:
                after = await self._run_quiz(quiz, sb, questions=first.get("questions") or None)   # same questions: comparable
                if after["score"] is None:
                    raise RuntimeError(after.get("error") or "no score")
            except Exception as e:  # the script was revised; its new score is unknown
                _reraise_account(e)
                self._skip("abstract quiz after the revision", e)
                undo("its quiz could not run")
                break
            rounds.append({"score": after["score"], "failed_claim_ids": after["failed_claim_ids"]})
            print(f"[StoryboarderV2] abstract quiz after revision {round_ + 1}: score {after['score']}")
            if after["score"] < current["score"]:   # a round that lost ground is undone, and the next round tries again from
                undo(f"it lowered the score ({current['score']} → {after['score']})")   # the kept script (Right On Q kept
                continue                                                                # a first revision 0.82 → 0.75)
            current = after
            sb.quiz = {**after, "before": before, "rounds": rounds}
        return sb.quiz

    async def build_v3(self, paper: PaperText, cast: Cast, language: str, panel_range: tuple[int, int], on_stage=None) -> StoryboardV2:
        """The v3 story layer: abstract → brief with verified abstract claims → a story with boards → deterministic lint and
        fix rounds → fact-check (boards included) → abstract quiz gate → translation (boards included), fit and the
        translation's fact-check. Panels end up numbered 1..N and chaptered by their beats."""
        def stage(status: str, message: str) -> None:
            if on_stage:
                on_stage(status, message)

        rng = story_range(panel_range)
        self.max_panels = rng[1]
        abstract = str(getattr(paper, "abstract", "") or "").strip() or find_abstract(paper.body)
        stage("analyzing", "Reading the paper and its abstract")
        brief = await self.analyze_v3(paper, abstract)
        stage("analyzing", "Checking the brief against the paper")
        await self.check_brief(brief, paper)
        claims = [c for c in _as_list(brief.pop("abstract_claims", [])) if isinstance(c, dict)]
        stage("storyboarding", "Writing the story")
        sb = await self.write_v3(brief, claims, cast, rng, paper)
        trail: list[dict] = []

        def lint_after(step: str) -> dict:   # what each step leaves, so a problem that ships can be traced to its step
            found = lint_v3(sb, paper.body, report=step == "scrub")   # the last one lists every uncharted result
            trail.append({"after": step, "panels": len(sb.panels),
                          "found": {str(n): [x[:90] for x in v] for n, v in sorted(found.items())}})
            print(f"[StoryboarderV2] {self.cache_key} lint after {step}: {sum(len(v) for v in found.values())} problems "
                  f"on panels {sorted(found)}")
            return found

        lint_after("write")
        stage("storyboarding", "Checking every board, number and abstract claim")
        await self.fix_v3(sb, cast, paper)
        await self.fit(sb, "en-US")
        lint_after("fix")
        # the diagrams are drawn before the checks, so the board check and the fact-check read their labels
        stage("storyboarding", "Drawing the diagrams")
        await self.design_diagrams(sb, paper)
        # before the checks: the fact-check reads the new pictures (the dry run's LongZero redrew panel 10's order against the
        # paper), and the fix rounds after it clean up what its corrections leave
        stage("storyboarding", "Checking that each picture shows its panel's point")
        await self._check_pictures(sb, cast, paper)
        lint_after("pictures")
        drawn = {(p.scene, p.visual, p.shows) for p in sb.panels}   # the pictures the check has seen
        stage("storyboarding", "Reading the script as a newcomer, then fact-checking it against the paper")
        await self._check_script(sb, cast, paper)
        await self.fit(sb, "en-US")
        lint_after("check")
        stage("storyboarding", "Testing whether a reader learns the abstract from the panels")
        await self.quiz_gate(sb, cast, paper, rng)
        lint_after("quiz")
        # every teaching picture gets one more look, and the last check reads the rewrites: on rounds 20-21's finished
        # scripts a fresh look flagged a quarter of the teaching pictures, and blind judges preferred 43 of its 55 rewrites
        # to the shipped plans (10 the other way); the pictures written since the first look (the script check's fixes,
        # splits and corrections, quiz revisions) also bring what the lint rejects in them (a clipboard, nothing happening)
        redrawn = {p.n for p in sb.panels if (p.scene, p.visual, p.shows) not in drawn}
        stage("storyboarding", "Looking at every teaching picture once more")
        await self._check_pictures(sb, cast, paper, append=True, linted=redrawn)
        lint_after("pictures again")
        await self.design_diagrams(sb, paper)   # a diagram whose brief changed since (the last check reads it)
        await self._last_check(sb, cast, paper)
        lint_after("last check")
        scrubbed = scrub_unsupported(sb, paper.body) + scrub_repeats(sb)
        stage("storyboarding", "Checking which chart shows each line of the recap")
        await self.recap_evidence(sb)
        _honest_recap(sb)
        _finish_v3(sb)
        sb.brief["lint"] = {str(n): found for n, found in lint_after("scrub").items()}   # what is left, for the job report
        sb.brief["lint_trail"] = trail
        if scrubbed:
            sb.brief["scrubbed"] = scrubbed
        if language != "en-US":
            stage("translating", f"Translating to {LANG_NAMES.get(language, language)}")
            sb = await self.translate(sb, language)
            await self.fit(sb, language)
            stage("translating", "Fact-checking the translation against the paper")
            if await self.factcheck(sb, paper, language):
                await self.fit(sb, language)
        _finish_v3(sb)
        if is_novice(sb):
            await review_story(sb, self.check_engine, on_usage=self._add_usage)
        return sb


def panel_range_for(choice: str, config_min: int, config_max: int) -> tuple[int, int]:
    rng = PANEL_RANGES.get(choice or "auto")
    return rng if rng else (config_min, config_max)
