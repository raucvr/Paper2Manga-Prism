"""Test doubles for the Prism v2 backend: fake engines, scripted model replies and small builders.

Nothing in here touches the network. The fakes mimic the contracts the pipeline relies on:
``BaseEngine.generate_text`` returning a ``TextResponse`` with ``usage``, and Seedream-style
``generate_image`` returning an ``ImageResponse`` with ``usage``/``size`` attributes attached.
"""

from __future__ import annotations

import asyncio
import copy
import base64
import inspect
import io
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Optional

from PIL import Image

from engines.base import (
    BaseEngine,
    GenerationConfig,
    ImageContent,
    ImageGenerationConfig,
    ImageResponse,
    Message,
    TextResponse,
)
from services.storyboarder_v2 import CHAPTER_PLAN, Bubble, Cast, CastMember, PanelV2

FIXTURES = Path(__file__).resolve().parent / "fixtures"
ATTENTION_PDF = FIXTURES / "attention_is_all_you_need.pdf"
COT_PDF = FIXTURES / "chain_of_thought.pdf"

# Distinct colours so assembled images can be checked panel by panel.
PALETTE = [(220, 40, 40), (40, 170, 60), (40, 70, 220), (230, 200, 30), (160, 40, 200), (30, 190, 200)]

# Every scripted bubble contains this marker, so image/vision fakes can tell which panel a prompt is about.
PANEL_MARK = re.compile(r"Line for panel (\d+)")
EXPECTED_RE = re.compile(r"EXPECTED \(JSON\): (.*)\n")


# ----------------------------------------------------------------------------- images

def image_bytes(width: int = 1, height: int = 1, color=(255, 0, 0), fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format=fmt)
    return buf.getvalue()


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


TINY_PNG_B64 = b64(image_bytes())


# ----------------------------------------------------------------------------- engines

def prompt_text(messages: list[Message]) -> str:
    return "\n\n".join(m.content for m in messages)


async def _resolve(value):
    return await value if inspect.isawaitable(value) else value


class FakeEngine(BaseEngine):
    """Text / vision engine fake.

    Replies come from ``router(prompt, messages, config)`` when given, otherwise from a FIFO queue.
    A reply may be a str, a dict (sent as JSON), a ``TextResponse``, an exception (raised) or an
    awaitable resolving to one of those. Every call is recorded as ``(messages, config)``.
    """

    def __init__(self, replies=(), *, router: Optional[Callable] = None, usage: Optional[dict] = None, model: str = "fake-seed"):
        super().__init__(api_key="test-key-not-real", base_url="http://fake.invalid", model=model)
        self.replies = list(replies)
        self.router = router
        self.usage = dict(usage or {"prompt_tokens": 1000, "completion_tokens": 200})
        self.calls: list[tuple[list[Message], Optional[GenerationConfig]]] = []
        self.closed = 0

    @property
    def name(self) -> str:
        return "Fake text engine"

    @property
    def supports_image_generation(self) -> bool:
        return False

    @property
    def supports_vision(self) -> bool:
        return True

    @property
    def prompts(self) -> list[str]:
        return [prompt_text(messages) for messages, _ in self.calls]

    async def generate_text(self, messages: list[Message], config: Optional[GenerationConfig] = None) -> TextResponse:
        self.calls.append((messages, config))
        await asyncio.sleep(0)  # behave like a real awaitable call: let other tasks run
        if self.router is not None:
            reply = self.router(prompt_text(messages), messages, config)
        elif self.replies:
            reply = self.replies.pop(0)
        else:
            raise AssertionError("FakeEngine got an unexpected call (no reply queued)")
        reply = await _resolve(reply)
        if isinstance(reply, BaseException):
            raise reply
        if isinstance(reply, TextResponse):
            return reply
        content = reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)
        return TextResponse(content=content, model=self.model, usage=dict(self.usage), finish_reason="stop")

    async def generate_text_stream(self, messages, config=None):
        raise NotImplementedError("streaming is not used by the v2 pipeline")

    async def close(self) -> None:
        self.closed += 1


class FakeImageEngine(BaseEngine):
    """Seedream fake: records ``{"prompt", "config", "refs"}`` per call and returns one small image.

    By default the image is 1/16 of the requested size (1024x1536 -> 64x96) and its colour encodes the
    panel number found in the prompt plus how often that panel was drawn before, so a re-render
    produces different bytes. ``fixed_size`` overrides the size. Calls whose prompt mentions a panel
    in ``fail_panels`` raise. ``hook(prompt)`` (sync or async) runs before the reply.
    """

    def __init__(self, *, fail_panels=(), hook: Optional[Callable] = None, usage: Optional[dict] = None,
                 fmt: str = "JPEG", fixed_size: Optional[tuple[int, int]] = None):
        super().__init__(api_key="test-key-not-real", base_url="http://fake.invalid", model="fake-seedream")
        self.fail_panels = set(fail_panels)
        self.hook = hook
        self.usage = dict(usage or {"output_tokens": 6144})
        self.fmt = fmt
        self.fixed_size = fixed_size
        self.calls: list[dict[str, Any]] = []
        self.closed = 0
        self._drawn: Counter = Counter()

    @property
    def name(self) -> str:
        return "Fake Seedream"

    @property
    def supports_image_generation(self) -> bool:
        return True

    @property
    def supports_vision(self) -> bool:
        return False

    async def generate_text(self, messages, config=None):
        raise NotImplementedError

    async def generate_text_stream(self, messages, config=None):
        raise NotImplementedError

    async def generate_image(self, prompt: str, config: Optional[ImageGenerationConfig] = None,
                             reference_images: Optional[list[ImageContent]] = None) -> ImageResponse:
        config = config or ImageGenerationConfig()
        self.calls.append({"prompt": prompt, "config": config, "refs": list(reference_images or [])})
        await asyncio.sleep(0)
        if self.hook is not None:
            await _resolve(self.hook(prompt))
        panels = [int(x) for x in PANEL_MARK.findall(prompt)]
        if self.fail_panels & set(panels):
            raise RuntimeError(f"Seedream exploded for panel(s) {panels}")
        first = panels[0] if panels else 0
        color = PALETTE[(first + self._drawn[first]) % len(PALETTE)]
        self._drawn[first] += 1
        width, height = self.fixed_size or (max(1, config.width // 16), max(1, config.height // 16))
        data = image_bytes(width, height, color, self.fmt)
        mime = "image/png" if self.fmt == "PNG" else "image/jpeg"
        response = ImageResponse(images=[ImageContent.from_base64(b64(data), mime)], model=self.model, prompt=prompt)
        response.usage = dict(self.usage)
        response.size = f"{config.width}x{config.height}"
        return response

    async def close(self) -> None:
        self.closed += 1


class FakeConfig:
    """Stand-in for ``config_loader.ConfigManager`` (only what engine_select uses)."""

    def __init__(self, core_engine: dict, providers: dict):
        self.core_engine = core_engine
        self.providers = providers

    def get_raw_config(self) -> dict:
        return {"core_engine": self.core_engine}

    def get_provider(self, name: str):
        return self.providers.get(name)


# ----------------------------------------------------------------------------- cast / panels

LOOKS = {
    "papi": "a plump brown dog professor with round glasses",
    "kumo": "a fluffy light-blue cloud creature with two green leaves",
    "nezu": "a brown hedgehog holding a clipboard",
}


def make_cast(theme: str = "kumomo") -> Cast:
    return Cast(theme=theme, style="Watercolor picture-book manga, soft pastel palette.", members=[
        CastMember("papi", "mentor", LOOKS["papi"], "patient"),
        CastMember("kumo", "student", LOOKS["kumo"], "curious"),
        CastMember("nezu", "skeptic", LOOKS["nezu"], "dry"),
    ])


def sheetless_cast() -> Cast:
    """A cast described only in words, with no reference sheets (as the kumomo trio is when its sheets are missing)."""
    return Cast(theme="custom", style="Ink-wash manga.", members=[
        CastMember("mochi", "mentor", "a round white rice-cake creature with a tiny bow", "patient"),
        CastMember("pip", "student", "a small green sprout with two leaves", "curious"),
        CastMember("rook", "skeptic", "a grey crow with a sharp beak", "dry"),
    ])


def make_sheets(names=("papi", "kumo", "nezu")) -> dict[str, ImageContent]:
    return {n: ImageContent.from_base64(b64(image_bytes(2, 2, PALETTE[i])), "image/png") for i, n in enumerate(names)}


def make_panel(n: int = 1, *, chapter: str = "how", title: str = "", characters=("papi", "kumo"),
               dialogue=(("papi", "Attention lets every word look at every other word."),), narration: str = "",
               labels=(), emotions: Optional[dict] = None, scene: str = "A cozy classroom.",
               visual: str = "Papi points at a board full of arrows.") -> PanelV2:
    return PanelV2(n=n, chapter=chapter, title=title or f"Panel {n}", characters=list(characters), scene=scene, visual=visual,
                   dialogue=[Bubble(s, t) for s, t in dialogue], narration=narration, labels=list(labels),
                   emotions=dict(emotions or {}))


# ----------------------------------------------------------------------------- scripted models

_PAPER_LINES = [
    "Attention Is All You Need",
    "Abstract",
    "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder.",
    "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
    "1 Introduction",
    "Recurrent models typically factor computation along the symbol positions of the input and output sequences, which precludes parallelization within training examples.",
    "Attention mechanisms have become an integral part of compelling sequence modeling and transduction models in various tasks.",
    "3 Model Architecture",
    "The encoder maps an input sequence of symbol representations to a sequence of continuous representations, and the decoder generates an output sequence one element at a time.",
    "Scaled dot-product attention computes the dot products of the query with all keys, divides each by the square root of the key dimension, and applies a softmax function.",
    "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.",
    "6 Results",
    "On the WMT 2014 English-to-German translation task, the big transformer model outperforms the best previously reported models by more than 2.0 BLEU, establishing a new state-of-the-art BLEU score of 28.4.",
    "On the WMT 2014 English-to-French translation task, our big model achieves a BLEU score of 41.0 after training for 3.5 days on eight GPUs.",
    "7 Conclusion",
    "In this work, we presented the Transformer, the first sequence transduction model based entirely on attention, replacing the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention.",
    "References",
    "[1] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.",
]
LONG_PAPER_TEXT = "\n".join(_PAPER_LINES)

BRIEF = {
    "title": "Attention Is All You Need",
    "field": "machine translation",
    "problem": "Recurrent networks read one word at a time, so training is slow and long-range links are hard.",
    "key_idea": "Let every word attend to every other word at once.",
    "mechanisms": [{"name": "self-attention", "plain": "Words vote on which other words matter.",
                    "metaphor": "a classroom where everyone can talk to everyone", "exact": "softmax of QK over root dk times V"}],
    "evidence": [{"claim": "Better translation", "numbers": "28.4 BLEU on WMT 2014 EN-DE", "why_it_matters": "new state of the art"}],
    "limitations": ["Cost grows with the square of the sequence length."],
    "takeaway": "Attention alone is enough.",
    "glossary": [{"term": "BLEU", "plain": "a translation quality score"}],
}


def storyboard_payload(num_panels: int) -> dict:
    """A budget-compliant English storyboard reply (no fit pass needed)."""
    casts = [["papi", "kumo"], ["kumo", "nezu"], ["papi", "nezu"]]
    panels = []
    for i in range(1, num_panels + 1):
        chars = casts[(i - 1) % len(casts)]
        panels.append({
            "n": i,
            "chapter": CHAPTER_PLAN[(i - 1) % len(CHAPTER_PLAN)][0],
            "title": f"Panel {i} title",
            "characters": chars,
            "scene": f"Scene {i} in a sunny lab.",
            "visual": f"{chars[0]} explains idea {i} next to a big arrow diagram.",
            "dialogue": [{"speaker": chars[0], "text": f"Line for panel {i} please."}],
            "narration": f"Fact number {i} from the paper.",
            "on_image_labels": ["Attention"],
            "emotions": {c: "curious" for c in chars},
        })
    return {"title": "Attention, Please!", "summary": "Attention replaces recurrence.", "panels": panels}


def translation_of(payload: dict) -> dict:
    """A zh-CN reply for ``translate_prompt``'s JSON payload (within the CJK budgets)."""
    return {
        "title": "注意力就是一切",
        "summary": "注意力机制取代了循环结构。",
        "chapters": [{"id": c["id"], "title": f"第{i}章"} for i, c in enumerate(payload["chapters"], start=1)],
        "panels": [{
            "n": p["n"],
            "title": f"第{p['n']}格",
            "dialogue": [{"speaker": b["speaker"], "text": _zh_line(f"第{p['n']}格的台词", b["text"])} for b in p["dialogue"]],
            "narration": _zh_line(f"论文中的第{p['n']}个事实。", p.get("narration") or ""),
            "labels": ["注意力"] * len(p.get("labels") or []),
            **({"board": {k: _board_zh(v) for k, v in p["board"].items()}} if p.get("board") else {}),
        } for p in payload["panels"]],
    }


def _zh_line(line: str, english: str) -> str:
    """Keep the fixture's figures without inventing a panel number in its translated prose.

    v2's source already contains the panel number, so those fixture strings stay the same.
    This is a scripted translation stub, not a general-purpose number parser.
    """
    words = dict(zip("one two three four five six seven eight nine ten eleven twelve".split(), range(1, 13)))
    source = re.findall(r"\d+(?:[.,]\d+)*", english)
    source += [str(words[w.lower()]) for w in re.findall(r"\b(?:" + "|".join(words) + r")\b", english, re.I)]
    remaining = list(source)
    for n in re.findall(r"\d+(?:[.,]\d+)*", line):
        if n not in remaining:
            line, remaining = "译文", list(source)
            break
        remaining.remove(n)
    return line + "".join(f" {n}" for n in remaining)


def _board_zh(text: str) -> str:
    """A zh-CN board string: Chinese with the English string's numbers; a string without letters stays as it is."""
    if not re.search(r"[A-Za-z]", text):
        return text
    return "板书" + "".join(f" {x}" for x in re.findall(r"\d+(?:[.,]\d+)*", text))


# ----------------------------------------------------------------------------- v3 story layer (scripted)

V3_RESULT = ("Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing "
             "best results by over 2 BLEU.")
V3_EXAMPLE = "The animal didn't cross the street because it was too tired."
# LONG_PAPER_TEXT plus a results sentence in the abstract, a small results table and an example sentence from a figure
V3_PAPER_TEXT = "\n".join(_PAPER_LINES[:4] + [V3_RESULT] + _PAPER_LINES[4:13] + [
    "Table 2: BLEU scores on the English-to-German newstest2014 test.",
    "ByteNet 23.75",
    "ConvS2S Ensemble 26.36",
    "Transformer (big) 28.4",
] + _PAPER_LINES[13:16] + ["Example sentence shown in a figure: The animal didn't cross the street because it was too tired ."]
    + _PAPER_LINES[16:])

# the brief's claims quote the abstract both papers share; V3_PAPER_TEXT's results sentence becomes claim 3 on its own
V3_CLAIMS = [
    {"id": 0, "text": "The best translation models were complex recurrent or convolutional networks with an encoder and a decoder.",
     "quote": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder."},
    {"id": 1, "text": "The Transformer is a new, simple architecture built only on attention.",
     "quote": "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms"},
    {"id": 2, "text": "It drops recurrence and convolutions entirely.", "quote": "dispensing with recurrence and convolutions entirely"},
]
BRIEF_V3 = {**BRIEF, "abstract_claims": V3_CLAIMS, "examples": [{"text": V3_EXAMPLE, "source": "Figure 3"}],
            "glossary": BRIEF["glossary"] + [{"term": "attention", "plain": "every word looks at every other word and weighs how much each matters"}]}


def _v3_panel(n: int, beat: str, chars, dialogue, narration: str = "", claims=(), board: Optional[dict] = None) -> dict:
    # the visual carries the panel marker: v3 art prompts hold the visual, not the bubbles
    return {"n": n, "kind": "board" if board else "scene", "beat": beat, "title": f"Panel {n} title", "characters": list(chars),
            "scene": "A cozy study with a big window.",
            "visual": f"{chars[0]} {'points at the big whiteboard' if board else 'ties blank cards together with red yarn'} (Line for panel {n}).",
            "dialogue": [{"speaker": s, "text": x} for s, x in dialogue], "narration": narration,
            "emotions": {c: "curious" for c in chars}, "board": board, "claims": list(claims),
            "shows": "" if board else f"the idea of panel {n}: blank cards tied together by red yarn"}


def story_payload_v3() -> dict:
    """A v3 story reply that passes lint on LONG_PAPER_TEXT and V3_PAPER_TEXT: every beat, a flow, a table and a recap
    board, every number from the paper, claims 0-3 covered (3 exists only for V3_PAPER_TEXT)."""
    return {"title": "Attention, Please!", "summary": "Attention alone replaces recurrence and convolutions.",
            "running_example": {"text": f"Kumo wants to understand one sentence at once: {V3_EXAMPLE}", "from_paper": True, "source": "Figure 3"},
            "panels": [
        _v3_panel(1, "hook", ["kumo"], [("kumo", "'The animal didn't cross the street because it was too tired.' Who is 'it'?")],
                  "Translation models turn a sentence in one language into a sentence in another.", [0]),
        _v3_panel(2, "problem", ["papi", "kumo"], [("papi", "The best models so far are recurrent or convolutional networks with an encoder and a decoder."),
                                                   ("kumo", "So they go word by word?")],
                  "Recurrent models go position by position, which blocks parallel work.", [0]),
        _v3_panel(3, "idea", ["papi", "kumo"], [("papi", "What if every word looks at every word at once?"),
                                                ("kumo", "So it looks straight at animal!")],
                  "The Transformer is built only on attention: no recurrence and no convolutions.", [1, 2],
                  board={"kind": "diagram", "title": "Every word looks at every word",
                         "brief": "the sentence's words in a row, with arrows from 'it' to 'animal', 'street' and 'tired'"}),
        _v3_panel(4, "how", ["papi"], [("papi", "So it looks at animal, street and tired all at once.")], "", [1, 2],
                  board={"kind": "flow", "title": "The Transformer, end to end", "steps": ["Words in", "Encoder", "Decoder", "Words out"],
                         "highlight": [1, 2], "note": "attention only, no recurrence"}),
        _v3_panel(5, "proof", ["nezu", "papi"], [("nezu", "Nice story. Where are the numbers?"), ("papi", "Right here on the board.")]),
        _v3_panel(6, "proof", ["nezu", "papi"], [("nezu", "28.4 on English-German. Fine, that is real."),
                                                 ("papi", "More than 2.0 BLEU above the best earlier models.")], "", [3],
                  board={"kind": "table", "title": "WMT 2014 translation (BLEU)", "subtitle": "WMT: a translation contest; BLEU: overlap with human translations",
                         "columns": ["Task", "BLEU"],
                         "rows": [{"cells": ["English-German", "28.4"], "quote": "establishing a new state-of-the-art BLEU score of 28.4"},
                                  {"cells": ["English-French", "41.0"], "quote": "our big model achieves a BLEU score of 41.0"}],
                         "highlight_rows": [0], "source": "Section 6"}),
        _v3_panel(7, "limits", ["nezu", "papi"], [("nezu", "So it still writes the output one element at a time?"),
                                                  ("papi", "Yes. The decoder generates the output one element at a time.")]),
        _v3_panel(8, "payoff", ["kumo", "papi"], [("kumo", "It looks straight at the animal: the animal was too tired!"),
                                                  ("papi", "That is attention doing its job.")], "", [1]),
        _v3_panel(9, "recap", ["kumo", "nezu"], [("nezu", "Fine. The numbers convinced me.")], "", [0, 1, 2, 3],
                  board={"kind": "list", "title": "What the paper showed",
                         "items": ["Old: recurrent or convolutional encoder-decoder models", "New: the Transformer, built only on attention",
                                   "No recurrence and no convolutions at all", "28.4 BLEU on WMT 2014 English-German"]}),
    ]}


# what the fake diagram designer draws: two boxes and a labelled arrow, a layout the diagram lint passes
DIAGRAM_DESIGN = {"elements": [
    {"type": "box", "x": 80, "y": 300, "w": 300, "h": 140, "text": "animal", "tone": "blue"},
    {"type": "arrow", "from": [380, 370], "to": [700, 370], "tone": "red", "label": "looks at"},
    {"type": "box", "x": 700, "y": 220, "w": 360, "h": 300, "text": "it", "tone": "red"}]}
V3_BUDGET = {"scene": (3, 16, 20), "board": (2, 12, 14)}   # bubbles, words per bubble, narration words
PAPER_NUMBERS = set(re.findall(r"\d+(?:\.\d+)?", V3_PAPER_TEXT + "\n" + LONG_PAPER_TEXT))


def fixed_panel(panel: dict, problems: list[str]) -> dict:
    """What the fake writer returns for a flagged panel: lines with a number the paper lacks replaced, bars without
    paper values dropped, the claims named in the problems added, bubbles and words cut to the budget."""
    out = json.loads(json.dumps(panel))
    max_bubbles, max_words, max_narration = V3_BUDGET["board" if out.get("kind") == "board" else "scene"]

    def clean(text: str, limit: int) -> str:
        if any(len(x) > 1 and x not in PAPER_NUMBERS for x in re.findall(r"\d+(?:\.\d+)?", text)):
            text = "Recurrent models read one position at a time."
        return " ".join(text.split()[:limit])

    out["dialogue"] = [{**b, "text": clean(b["text"], max_words)} for b in out.get("dialogue") or []][:max_bubbles]
    out["narration"] = clean(out.get("narration") or "", max_narration)
    board = out.get("board")
    if isinstance(board, dict) and board.get("kind") == "bar":
        board["bars"] = [b for b in board.get("bars") or [] if str(b.get("value")) in PAPER_NUMBERS]
    out["claims"] = sorted(set(out.get("claims") or []) | {int(x) for p in problems for x in re.findall(r"claim (\d+)", p)})
    return out


class PaperModel:
    """Router for the text FakeEngine: answers the brief / storyboard / fact-check / translation / fit prompts
    like Seed would, and the v3 story-layer prompts (brief with abstract claims, story, lint fix, abstract quiz:
    questions, reader, grader, revision). Prompts are recognised by how the user message starts, never by paper content.

    v3 knobs: ``story(payload) -> payload`` edits the story reply; ``fix(panel, problems) -> panel`` replaces
    ``fixed_panel``; ``quiz_misses``: per quiz round, the claim ids whose questions the reader cannot answer;
    ``clarity_problems``: what the reader check reports; ``board_problems``: what the board check reports;
    ``picture_problems``: what the picture check reports; ``two_frames(prompt) -> reply`` replaces the plan of two frames
    for a picture that missed its idea; ``recap_evidence(charts, lines) -> reply`` replaces the recap evidence answer
    (by default a line is shown by each chart that draws all of its numbers)."""

    def __init__(self, num_panels: int = 3, on_storyboard: Optional[Callable] = None, factcheck_problems=(), *,
                 story: Optional[Callable] = None, fix: Optional[Callable] = None, quiz_misses=(), clarity_problems=(),
                 board_problems=(), picture_problems=(), two_frames: Optional[Callable] = None,
                 recap_evidence: Optional[Callable] = None):
        self.num_panels = num_panels
        self.on_storyboard = on_storyboard
        self.factcheck_problems = list(factcheck_problems)
        self.clarity_problems = list(clarity_problems)
        self.board_problems = list(board_problems)
        self.picture_problems = list(picture_problems)
        self.two_frames = two_frames
        self.recap_evidence = recap_evidence
        self.story = story
        self.fix = fix or fixed_panel
        self.quiz_misses = [set(x) for x in quiz_misses]
        self.quiz_claims: dict[str, int] = {}   # question text -> claim id, so the reader knows what it cannot answer
        self.reader_rounds = 0
        self.seen: list[str] = []

    def __call__(self, prompt: str, messages, config):
        user = messages[-1].content
        if user.startswith("Read the paper and write a compact, faithful BRIEF"):
            self.seen.append("brief")
            return BRIEF
        if user.startswith("You verify the BRIEF of a research paper"):
            self.seen.append("briefcheck")
            return {"problems": []}
        if user.startswith("Write the manga script"):
            self.seen.append("storyboard")
            return self._storyboard()
        if user.startswith("You are a meticulous fact-checker"):
            self.seen.append("factcheck")
            return {"problems": list(self.factcheck_problems)}
        if user.startswith("You are a curious reader who never saw this research paper"):
            self.seen.append("clarity")
            return {"problems": list(self.clarity_problems)}
        if user.startswith("Check every board of this manga script against the words of its own panel"):
            self.seen.append("boardcheck")
            return {"problems": list(self.board_problems)}
        if user.startswith("Check the pictures planned for a manga"):
            self.seen.append("picturecheck")
            return {"problems": list(self.picture_problems)}
        if user.startswith("A manga that teaches a research paper ends with a recap board"):
            self.seen.append("recap_evidence")
            charts = json.loads(user.split("THE MANGA'S CHARTS AND TABLES:\n", 1)[1].split("\n\nTHE RECAP LINES:", 1)[0])
            lines = json.loads(user.split("THE RECAP LINES:\n", 1)[1].split("\n\nOutput JSON only", 1)[0])
            if self.recap_evidence is not None:
                return self.recap_evidence(charts, lines)

            def shows(chart: dict, line: str) -> bool:
                numbers = re.findall(r"\d+(?:\.\d+)?", line)
                return bool(numbers) and all(x in json.dumps(chart["board"], ensure_ascii=False) for x in numbers)
            return {"lines": [{"i": x["i"], "panels": [c["n"] for c in charts if shows(c, x["line"])]} for x in lines]}
        if user.startswith("Translate this manga script"):
            self.seen.append("translate")
            return translation_of(json.loads(user.rsplit("\n\n", 1)[1]))
        if user.startswith("Rewrite each text so it fits"):
            self.seen.append("fit")
            items = json.loads(user.split("ITEMS:\n", 1)[1])
            return {"items": [{"id": item["id"], "text": "A shorter line."} for item in items]}
        if user.startswith("Read the paper and its ABSTRACT and write a compact, faithful BRIEF"):
            self.seen.append("brief_v3")
            return BRIEF_V3
        if user.startswith("Turn this research paper into a SHORT MANGA STORY"):
            self.seen.append("story")
            payload = story_payload_v3()
            return self.story(payload) if self.story is not None else payload
        if user.startswith("Fix the flagged panels of this manga script"):
            self.seen.append("fix")
            flagged = json.loads(user.split("FLAGGED PANELS:\n", 1)[1].split("\n\n", 1)[0])
            return {"panels": [self.fix(item["panel"], item["problems"]) for item in flagged]}
        if user.startswith("This manga panel's picture did not come out"):
            self.seen.append("two_frames")
            if self.two_frames is not None:
                return self.two_frames(user)
            who = re.search(r"Only these characters appear: ([^;]+);", user)
            name = who.group(1).split(",")[0].strip() if who and who.group(1).strip() != "none" else "papi"
            return {"frames": [{"visual": f"{name} sets blank cards in two separate piles.", "shows": "two separate piles of blank cards"},
                               {"visual": f"{name} pushes the two piles into one mixed pile.", "shows": "one mixed pile of blank cards"}]}
        if "You design ONE diagram for panel" in user:
            self.seen.append("diagram")
            return copy.deepcopy(DIAGRAM_DESIGN)
        if user.startswith("Fix the pictures of these manga panels"):
            self.seen.append("pictures")
            flagged = json.loads(user.split("PANELS TO FIX:\n", 1)[1])
            return {"panels": [{"n": item["n"], "scene": item["scene"], "visual": item["visual"], "shows": item["shows"]} for item in flagged]}
        if user.startswith("Write a short-answer quiz on a research paper's ABSTRACT"):
            self.seen.append("quiz")
            return self._quiz(json.loads(user.rsplit("CLAIMS:\n", 1)[1]), len(re.findall(r"^S\d+: ", user, re.M)))
        if user.startswith("You have never read the research paper behind this manga"):
            self.seen.append("reader")
            return self._answers(json.loads(user.rsplit("QUESTIONS:\n", 1)[1]))
        if user.startswith("Grade each answer of a reading quiz strictly"):
            self.seen.append("grade")
            return {"grades": [{"id": item["id"], "score": 1, "why": "matches the gold answer"} for item in json.loads(user.rsplit("ITEMS:\n", 1)[1])]}
        if user.startswith("A reader who never saw the paper read only the visible text of this manga"):
            self.seen.append("revise")
            failed = json.loads(user.split("did not come across:\n", 1)[1].split("\n\n", 1)[0])
            script = json.loads(user.split("\nSCRIPT\n", 1)[1].split("\n\nPAPER TEXT:", 1)[0])
            after = max([p["n"] for p in script if p.get("beat") == "proof"] or [len(script)])
            return {"panels": [{"after": after, "kind": "scene", "beat": "proof", "title": f"Claim {c['id']}", "characters": ["papi", "kumo"],
                                "scene": "A cozy study.", "visual": "papi holds up one finger.", "board": None, "claims": [c["id"]],
                                "dialogue": [{"speaker": "papi", "text": " ".join(c["text"].split()[:22])}], "narration": ""} for c in failed]}
        raise AssertionError(f"unexpected text prompt: {user[:120]!r}")

    def _quiz(self, claims: list[dict], sentences: int = 0) -> dict:
        per = 2 if 2 * len(claims) <= 12 else 1
        questions = []
        for c in claims:
            for k in range(per):
                question = f"What does the abstract say in part {chr(65 + len(questions) % 26)}{'' if len(questions) < 26 else 'x'}?"
                self.quiz_claims[question] = c["id"]
                questions.append({"id": f"Q{len(questions) + 1}", "claim_ids": [c["id"]], "question": question, "answer": c["text"],
                                  **({"sentences": [len(questions) % sentences + 1]} if sentences else {})})
        return {"questions": questions}

    def _answers(self, questions: list[dict]) -> dict:
        misses = self.quiz_misses[self.reader_rounds] if self.reader_rounds < len(self.quiz_misses) else set()
        self.reader_rounds += 1
        return {"answers": [{"id": q["id"], "answer": "not shown", "panels": []} if self.quiz_claims.get(q["question"]) in misses
                            else {"id": q["id"], "answer": "As the manga shows.", "panels": [1]} for q in questions]}

    async def _storyboard(self) -> dict:
        if self.on_storyboard is not None:
            await _resolve(self.on_storyboard())
        return storyboard_payload(self.num_panels)


def exact_verdict(expected: list[dict], overall: float = 9) -> dict:
    """A judge verdict in which every expected text is exact and every character matches its sheet."""
    panels = []
    for p in expected:
        texts = [e["text"] for e in p["expected_text"]]
        panels.append({
            "panel_index": p["panel_index"],
            "transcription": texts,
            "text_items": [{"text": t, "found": True, "exact": True, "note": ""} for t in texts],
            "characters": [{"name": c, "present": True, "matches_reference": True, "note": ""} for c in p["characters"]],
            "extra_characters": [],
            "issues": [],
        })
    return {"panels": panels, "overall": overall, "worst_problem": "none"}


class JudgeModel:
    """Router for the vision FakeEngine: reads ``EXPECTED (JSON)`` from the QA prompt and returns a verdict.

    Everything passes by default. For panels in ``fail_first_for`` the first judgement marks the first
    bubble as not exact and scores the page ``fail_overall``.
    """

    def __init__(self, *, fail_first_for=(), fail_overall: float = 5, overall: float = 9):
        self.fail_first_for = set(fail_first_for)
        self.fail_overall = fail_overall
        self.overall = overall
        self.judged: Counter = Counter()

    def __call__(self, prompt: str, messages, config):
        match = EXPECTED_RE.search(prompt)
        assert match, "QA prompt without an EXPECTED (JSON) line"
        expected = json.loads(match.group(1))
        verdict = exact_verdict(expected, self.overall)
        for p_exp, p_res in zip(expected, verdict["panels"]):
            numbers = PANEL_MARK.findall(" ".join(e["text"] for e in p_exp["expected_text"]))
            key = int(numbers[0]) if numbers else 0
            self.judged[key] += 1
            if key in self.fail_first_for and self.judged[key] == 1:
                p_res["text_items"][0].update(exact=False, note="reads 'Lne for panel'")
                verdict["overall"] = self.fail_overall
        return verdict
