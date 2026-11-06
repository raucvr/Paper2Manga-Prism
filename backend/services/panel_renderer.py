"""
Panel renderer v2 — one Seedream image per panel (or per two-panel strip).

Prompt template (≈180–260 words, under Seedream's 600-word guidance), derived from the experiments:
  style + reference mapping → layout line → explicit character scope → scene / action / emotions
  → quoted bubbles with speakers (+ tail rule) → narration → ≤2 labels → text rules → "no other text" rule

v3 (``text_free``): build_art_prompt draws the picture with NO text at all — bubbles, narration and board exhibits are
typeset by code afterwards — and reserves room for them: a calm top band (scene) or a large blank whiteboard (board).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Sequence

from engines.base import BaseEngine, ImageContent, ImageGenerationConfig, ImageResponse
from services.storyboarder_v2 import CHAT_BOX_RE, PICTURE_BEATS, SCENE_CHART_RE, Cast, PanelV2

ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]


def _ordinal(i: int) -> str:
    return ORDINALS[i] if i < len(ORDINALS) else f"#{i + 1}"

TEXT_RULES = {
    "en-US": "Render every quoted text exactly as written, in English, in a large, clean, legible hand-lettered comic font.",
    "zh-CN": "Render every quoted text exactly as written, in Simplified Chinese characters, large and legible, in a clean rounded comic font.",
    "ja-JP": "Render every quoted text exactly as written, in Japanese, large and legible, in a clean rounded comic font.",
}
BUBBLE_RULES = "Each speech bubble has a tail pointing at its speaker's mouth; characters without a bubble are silent. Nothing is cut off; no extra words."
NO_STRAY_TEXT = ("Apart from the quoted texts above, the image contains NO other letters, words, numbers or symbols: any diagram, poster, board, "
                 "screen, book or sign is wordless (plain boxes, arrows, shapes, dots or scribble lines instead of writing).")
SHEET_CAPTIONS = " Never copy the captions on the reference sheets (e.g. \"Soft Spikes\")."


@dataclass
class RenderResult:
    image: ImageContent
    prompt: str
    usage: dict
    size: str
    seconds: float


def _reference_block(cast: Cast, ref_names: list[str], force_ordinals: bool = False) -> str:
    looks = {m.name: m.look for m in cast.members}
    if not ref_names:
        return ""
    if len(ref_names) == 1 and not force_ordinals:
        n = ref_names[0]
        return f" The reference image is {n}: {looks.get(n, n)}. Draw {n} exactly as in the reference sheet: same species, colors, face and proportions."
    parts = [f"The {_ordinal(i)} reference image is {n}: {looks.get(n, n)}." for i, n in enumerate(ref_names)]
    return " " + " ".join(parts) + " Draw every character exactly as in its reference sheet: same species, colors, face and proportions."


def _looks_block(cast: Cast, present: list[str], ref_names: list[str]) -> str:
    """Looks of the characters in the picture that have no reference sheet (Seedream would otherwise get a bare name)."""
    looks = [f"{m.name}: {m.look}" for m in cast.members if m.name in present and m.name not in ref_names]
    return f" Characters without a reference sheet look like this: {'; '.join(looks)}." if looks else ""


def _scope_line(cast: Cast, present: list[str], noun: str = "panel") -> str:
    others = [m.name for m in cast.members if m.name not in present]
    line = f"Characters in this {noun}: {', '.join(present)} only." if present else f"No characters in this {noun}."
    if others:
        line += f" {', '.join(others)} do NOT appear."
    return line + (" No other people, children, animals or creatures appear anywhere in the picture; apart from the listed characters, "
                   "every room, office, court, street, vehicle and building in it is empty of people.")


# a scene that mentions written text makes Seedream letter gibberish ("a sentence with one word blacked out")
TEXTY_RE = re.compile(r"\b(?:sentences?|text|words?|letters?|writing|written|captions?|headlines?|paragraphs?|notes?|numbers?|digits?"
                      r"|sums?|equations?|formulas?|arithmetic|calculations?)\b|\d", re.I)
TEXTY_HINT = ("Apart from the quoted texts, draw any sentence, word, number or formula in this scene as blank cards, blocks or bars "
              "with no letters or digits.")


def _panel_block(panel: PanelV2, label: Optional[str] = None, per_panel_scope: bool = False) -> str:
    lines = []
    if label:
        lines.append(label)
    if per_panel_scope:
        lines.append(f"Characters in this panel: {', '.join(panel.characters)} only; nobody else.")
    if panel.scene:
        lines.append(f"Scene: {panel.scene}")
    action = panel.visual or ""
    if panel.emotions:
        action += " Emotions: " + ", ".join(f"{k} is {v}" for k, v in panel.emotions.items()) + "."
    lines.append(f"Action: {action}")
    if TEXTY_RE.search(f"{panel.scene} {panel.visual}"):
        lines.append(TEXTY_HINT)
    if panel.dialogue:
        lines.append("Speech bubbles:")
        for b in panel.dialogue:
            lines.append(f'- {b.speaker} says: "{b.text}"')
        speakers = sorted({b.speaker for b in panel.dialogue})
        silent = [c for c in panel.characters if c not in speakers]
        if silent:
            lines.append(f"Speakers: {', '.join(speakers)}. {', '.join(silent)} {'is' if len(silent) == 1 else 'are'} silent.")
    else:
        lines.append("No speech bubbles in this panel.")
    if panel.narration:
        lines.append(f'Narration box along the top edge: "{panel.narration}"')
    if panel.labels:
        lines.append("Labels drawn on props: " + ", ".join(f'"{x}"' for x in panel.labels[:2]))
    return "\n".join(lines)


def _feedback_block(feedback: Optional[str]) -> str:
    return f"\n\nThe previous attempt had these problems — fix them: {feedback}" if feedback else ""


def build_single_prompt(panel: PanelV2, cast: Cast, language: str, ref_names: list[str], feedback: Optional[str] = None, anchor: bool = False) -> str:
    style = cast.style + _reference_block(cast, ref_names, force_ordinals=anchor) + _looks_block(cast, panel.characters, ref_names)
    if anchor:
        style += (f" The {_ordinal(len(ref_names))} reference image is the previous panel of this same manga: match its palette, line weight and "
                  "lighting, but do not copy its composition, poses or any of its text.")
    return (
        f"{style}\nA single manga panel, portrait, with a thin black border and a small white margin. {_scope_line(cast, panel.characters)}\n\n"
        f"{_panel_block(panel)}\n\n{TEXT_RULES.get(language, TEXT_RULES['en-US'])} {BUBBLE_RULES} {NO_STRAY_TEXT}"
        f"{SHEET_CAPTIONS if ref_names else ''}{_feedback_block(feedback)}"
    )


def build_strip_prompt(panels: list[PanelV2], cast: Cast, language: str, ref_names: list[str], feedback: Optional[str] = None) -> str:
    present = []
    for p in panels:
        for c in p.characters:
            if c not in present:
                present.append(c)
    blocks = "\n\n".join(_panel_block(p, label=f"PANEL {i + 1} ({pos}):", per_panel_scope=True) for i, (p, pos) in enumerate(zip(panels, ["left", "right"])))
    return (
        f"{cast.style}{_reference_block(cast, ref_names)}{_looks_block(cast, present, ref_names)}\nA manga strip with exactly two equal panels side by side, separated by a white gutter, each with a thin black "
        f"border. Each panel is its own scene as described. {_scope_line(cast, present, noun='strip')}\n\n{blocks}\n\n"
        f"{TEXT_RULES.get(language, TEXT_RULES['en-US'])} {BUBBLE_RULES} {NO_STRAY_TEXT}{SHEET_CAPTIONS if ref_names else ''}"
        f"{_feedback_block(feedback)}"
    )


# ----------------------------------------------------------------------------- v3: text-free art (the text is typeset later)

WORDLESS = ("WORDLESS IMAGE: no letters, words, numbers, digits, written symbols, speech bubbles, chat windows, text boxes, captions or "
            "signatures anywhere (arrows and simple shapes that the scene asks for are fine); "
            "screens, books, papers, clipboards, cards, boards, signs and clock faces are blank or show only simple shapes.")
ART_SHEET_CAPTIONS = " Never copy the captions on the reference sheets."
ART_TEXTY_HINT = "Draw anything in this scene that would carry writing (a sentence, word, number or formula) as blank cards, blocks or bars."
SCENE_COMPOSITION = ("Composition: the top 28% of the panel is calm, plain background (wall or sky) with nothing in it, reserved for text "
                     "added later; no whiteboard, blackboard or screen anywhere in the picture; all heads are below that band")
BOARD_COMPOSITION = ("Composition: the top 22% of the panel is plain empty wall reserved for text. Below it, a LARGE, COMPLETELY BLANK white "
                     "whiteboard spans almost the full width of the panel and about 48% of its height, seen straight on, evenly lit, with a "
                     "thin dark frame. Nothing stands in front of it or overlaps it, and nothing is written or drawn on it.")
ART_MAX_WORDS = 450
# a teaching scene names what its picture alone must make clear: the objects that carry the idea are drawn big and uncovered
# (the audit of 2026-10-28: key props were tiny or hidden, e.g. eight answer cards in a strip at the bottom edge)
SHOWS_LINE = ("The picture itself must make this clear, without any words: {shows}. Draw the objects that carry it LARGE and in "
              "front, in the lower two thirds of the panel, fully visible and not covered by anyone.")
SHOWS_MAX_WORDS = 60
# a scene that mentions writing gets the blank-cards hint ("nezu scribbles on his clipboard" came back with glyphs on it)
ART_TEXTY_RE = re.compile(TEXTY_RE.pattern + r"|\b(?:writes?|scribbl\w*|jot(?:s|ted|ting)?)\b", re.I)
# quoted words invite the image model to letter them: dialogue never enters the art prompt, quoted bits of a scene are dropped
# (a straight single quote only opens after a non-word character, so "kumo's" and "papis'" stay)
CAPS_RUN_RE = re.compile(r"\b[A-Z]{3,}(?:[\s:,!]+[A-Z]{2,})+\b[:!]?")   # words in capitals: a sign's text, never a name
QUOTED_RE = re.compile(r"\"[^\"\n]*\"|“[^”\n]*”|‘[^’\n]*’|「[^」\n]*」|『[^』\n]*』|«[^»\n]*»|(?<![\w'])'[^'\n]+'(?!\w)")
# the exhibit is drawn by code, so the picture needs a blank board, never a chart the image model would make up
CHART_RE = re.compile(r"\b(?:(?:bar|line|pie|flow|results?|comparison|data|scores?|summary|recap)\s+)?"
                      r"(?:charts?|graphs?|plots?|diagrams?|flowcharts?|infographics?|histograms?|pipelines?|slides?)\b", re.I)
QUALIFIED_RE = re.compile(r"\b(?:comparison|results?|data|scores?|summary|recap|numbers?|values?)\s+(?:tables?|lists?|bars?)\b", re.I)
KIND_WORDS = {"bar": r"bars?", "table": r"tables?", "list": r"lists?|checklists?", "flow": r"arrows?", "links": r"arrows?",
              "diagram": r"arrows?|boxes|diagrams?"}   # what the program draws on the board, never the illustrator
# "the kitchen table" is furniture even when the exhibit is a table
FURNITURE_CUES = frozenset("sit sits sat sitting seated around on under beneath kitchen dining coffee wooden picnic work side "
                           "round lab breakfast folding cluttered".split())
# a clause describing what is on the board ("the board where the Transformer bar towers over the rest") invites a chart
BOARD_CONTENT_RE = re.compile(r"\b((?:white)?board)\s+(?:where|that shows|which shows|that lists|which lists|showing|shows|displaying"
                              r"|listing|comparing|full of|covered (?:with|in)|filled with)\b.*?(?=[,;.!?]|\s+(?:while|as|but|and then)\b|$)",
                              re.I)
_SUPERSCRIPTS = "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻"
# a whole number token: 28.4, 1,000, 4.9%, 3:00, 1/2, 10x, 2.3 · 10^19, 2.3×10¹⁹, 1e19, 10¹⁹ (both ends of "4-5");
# never a piece of a name (P100, GPT-2, ConvS2S, 3D, 50th)
NUMBER_RE = re.compile(rf"(?<![\w.])(?<![A-Za-z][\-−])[+\-−±~≈]?\d+(?:[.,:/]\d+)*(?:\s*[·×x*]\s*10(?:\^?[+\-−]?\d+|[{_SUPERSCRIPTS}]+))?"
                       rf"(?:[eE][+\-−]?\d+)?(?:\s*%|[{_SUPERSCRIPTS}]+|[x×])?(?!\w)")
OFF_PANEL = "something just out of view"
BLANK_CARDS = "small glowing orbs"   # not cards: the art check counts an empty box or card floating in the air as a text box
COUNT_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve")


def _tidy(text: str) -> str:
    return re.sub(r"\s+([,.;:!?])", r"\1", re.sub(r"\s+", " ", text)).strip()


def _unquoted(text: Optional[str]) -> str:
    text = CAPS_RUN_RE.sub(" ", QUOTED_RE.sub(" ", text or ""))   # "displays DELIVERY FAILED" would be lettered
    return _tidy(re.sub(r"[\"“”「」『』«»]", "", text))   # stray marks of an unbalanced quote go too


def _no_numbers(text: str) -> str:
    """Digits invite the image model to draw them: a small count becomes a word ("4 ovens" -> "four ovens"), any other
    number (a result, a percentage, a year) is dropped; names with digits in them (P100, GPT-2) stay."""
    def word(m: re.Match) -> str:
        s = m.group(0)
        return COUNT_WORDS[int(s)] if re.fullmatch(r"[0-9]+", s) and int(s) < len(COUNT_WORDS) else " "   # not isdigit: "10¹⁹" is
    return _tidy(NUMBER_RE.sub(word, text))


def _no_chart(text: str) -> str:
    """A chart the writer left in a scene ("Kumo looks at the bar chart") would come back as a fake one, and a scene has
    no board to put it on: the characters look at something out of view instead."""
    return re.sub(rf"\b(?:(?:the|a|an|this|that|these|those|their|its)\s+)?(?:(?:rising|growing|shrinking|falling|tall|short|big|small)\s+)?"
                  rf"(?:{SCENE_CHART_RE.pattern})", OFF_PANEL, text, flags=re.I)


def _no_chat(text: str) -> str:
    """Chat windows and text boxes come back as empty speech bubbles ("empty reply windows float above them" on
    Hivemind's first panel), which readers take for missing text: they become glowing orbs."""
    return re.sub(rf"\b(?:(?:the|a|an|this|that|these|those|their|its|some|several|many)\s+)?(?:(?:empty|blank|floating|glowing|open|"
                  rf"small|big|many|several|identical)\s+)*(?:{CHAT_BOX_RE.pattern})", BLANK_CARDS, text, flags=re.I)


# "a white slip marked with a red cross" came back as a first-aid plus: the delivery book's fake-remark slips read as medical
# kits to the audit, while the words meant a mark of failure (the Chinese text says 红叉, an X)
RED_CROSS_RE = re.compile(r"\bred[- ]cross(es)?\b", re.I)


def _x_marks(text: str) -> str:
    return RED_CROSS_RE.sub(lambda m: "red X mark" + ("s" if m.group(1) else ""), text)


def _symbol_words(symbol: str) -> str:
    """A book symbol as the image model is told it (its reference sheet and the line that points at the sheet): cleaned
    like the panel's own words, since the sheet is copied into every panel that draws the symbol."""
    return " ".join(_no_numbers(_x_marks(_unquoted(symbol))).split())


def _clip_words(text: str, limit: Optional[int]) -> str:
    words = text.split()
    if limit is None or len(words) <= limit:
        return text
    return " ".join(words[:limit]).rstrip(",;:-") + "." if limit > 0 else ""


def _board_text(text: str, kind: str) -> str:
    """A board panel's setting or action with every chart word turned into "board", what is on the board left out and
    its numbers dropped."""
    text = QUALIFIED_RE.sub("board", CHART_RE.sub("board", text))
    if kind in KIND_WORDS:
        def board(m: re.Match) -> str:
            before = re.findall(r"[a-z]+", text[max(0, m.start() - 40):m.start()].lower())[-3:]
            return m.group(0) if kind == "table" and FURNITURE_CUES & set(before) else "board"
        text = re.sub(rf"\b(?:{KIND_WORDS[kind]})\b", board, text, flags=re.I)
    text = re.sub(r"\b((?:white)?board)(?:['’]s)?(?:\s+board)+\b", r"\1", _no_numbers(text), flags=re.I)
    return _tidy(BOARD_CONTENT_RE.sub(r"\1", text))


def stage_order(panel: PanelV2) -> list[str]:
    """Left to right: the speakers in the order they speak (bubbles are read left to right), then the silent characters."""
    speakers = [b.speaker for b in panel.dialogue if b.speaker in panel.characters]
    return list(dict.fromkeys([*speakers, *panel.characters]))


def _placement(names: list[str]) -> str:
    if len(names) == 1:
        return f"{names[0]} in the center"
    if len(names) == 2:
        return f"{names[0]} on the left and {names[1]} on the right"
    if len(names) == 3:
        return f"{names[0]} on the left, {names[1]} in the middle and {names[2]} on the right"
    return f"from left to right {', '.join(names[:-1])} and {names[-1]}" if names else ""


def art_shows(panel: PanelV2) -> str:
    """A teaching scene's idea exactly as SHOWS_LINE gives it to the image model (no quotes, charts, chat boxes or digits,
    at most SHOWS_MAX_WORDS words); the idea check judges this same text, never words the image model was told to leave out.
    A story-only scene (an introduction, a challenge, the bet) has none: only the teaching beats and the hook are drawn to it."""
    shows = _x_marks(_no_chat(_unquoted(panel.shows)))
    if panel.beat not in PICTURE_BEATS + ("hook",) or SCENE_CHART_RE.search(shows):
        return ""   # a chart cannot be the drawn idea: "something just out of view towers over..." would be asked for, and judged
    return _clip_words(_no_numbers(shows).strip().rstrip("."), SHOWS_MAX_WORDS)


def build_art_prompt(panel: PanelV2, cast: Cast, ref_names: list[str], feedback: Optional[str] = None, anchor: bool = False,
                     shape: str = "portrait", props: Sequence[str] = ()) -> str:
    """v3: the picture of one panel with NO text in it — dialogue, narration, labels and the board's exhibit are typeset
    by code afterwards, so none of them is ever sent. The top stays calm for that text, the first speaker stands on the
    left so the bubbles read in order, and a board panel gets a large blank whiteboard. At most ART_MAX_WORDS words, plus
    one line per symbol sheet. ``shape``: "landscape" for one of two frames stacked in a panel (two_frames_prompt).
    ``props``: the book symbols whose reference pictures follow the character sheets (and the anchor), in that order."""
    style = cast.style + _reference_block(cast, ref_names, force_ordinals=anchor or bool(props)) + _looks_block(cast, panel.characters, ref_names)
    if anchor:
        style += (f" The {_ordinal(len(ref_names))} reference image is the previous panel of this same manga: match its palette, "
                  "line weight and lighting, but do not copy its composition or poses.")
    first = len(ref_names) + (1 if anchor else 0)
    # outside the word budget: a symbol's sheet must never cost the panel its own action or a retry its feedback
    sheet_lines = "".join(f" The {_ordinal(first + i)} reference image shows the {_symbol_words(symbol)}: draw it exactly as that "
                          "image shows it (same shape, colors and proportions)." for i, symbol in enumerate(props))
    style += sheet_lines
    head = f"{style}\nA single manga panel, {shape}, with a thin black border and a small white margin. {_scope_line(cast, panel.characters)}"
    # as lettering: a "board" panel without an exhibit is lettered as a scene, so it is drawn as one
    board = panel.kind == "board" and isinstance(panel.board, dict) and bool(panel.board)
    setting, action = _x_marks(_no_chat(_unquoted(panel.scene))), _x_marks(_no_chat(_unquoted(panel.visual)))
    if board:
        kind = str(panel.board.get("kind") or "").lower()
        setting, action = _board_text(setting, kind), _board_text(action, kind)
    else:
        setting, action = _no_numbers(_no_chart(setting)), _no_numbers(_no_chart(action))
    emotions = (_no_numbers(_unquoted(" Emotions: " + ", ".join(f"{k} is {v}" for k, v in panel.emotions.items()) + "."))
                if panel.emotions else "")
    order = stage_order(panel)
    where = _placement(order)
    if board:
        composition = BOARD_COMPOSITION + (f" At the bottom, below the board, {where} {'is' if len(order) == 1 else 'are'} shown from "
                                           "the chest up." if where else "")
    else:
        composition = SCENE_COMPOSITION + (f"; {where}." if where else ".")
    texty = not board and ART_TEXTY_RE.search(f"{panel.scene} {panel.visual} {panel.shows}")
    shows = "" if board else art_shows(panel)
    rules = WORDLESS + (ART_SHEET_CAPTIONS if ref_names else "")
    fix = _unquoted(feedback)

    def assemble(setting_words: Optional[int], action_words: Optional[int], fix_words: Optional[int]) -> str:
        lines = [f"Setting: {_clip_words(setting, setting_words)}"] if setting else []
        act = _tidy(f"{_clip_words(action, action_words)} {emotions}")
        if act:
            lines.append(f"Action (draw nothing on the board): {act}" if board else f"Action: {act}")
        if shows:
            lines.append(SHOWS_LINE.format(shows=shows))
        if texty:
            lines.append(ART_TEXTY_HINT)
        lines.append(composition)
        return f"{head}\n\n" + "\n".join(lines) + f"\n\n{rules}{_feedback_block(_clip_words(fix, fix_words))}"

    # only an over-long prompt is clipped: the scene and the feedback give way, the rules never do
    for caps in ((None, None, None), (45, 90, 80), (30, 60, 60), (20, 40, 40), (15, 25, 25), (15, 25, 0)):
        prompt = assemble(*caps)
        if len(prompt.split()) - len(sheet_lines.split()) <= ART_MAX_WORDS:
            break
    return prompt


# A teaching scene whose picture still misses its idea after every try is drawn as two frames stacked in its panel. Of
# round 17's 13 such scenes, blind judges preferred the two frames 9 times and the single picture 3 times (1 tie); where
# the single picture had come out, it won 7 to 2 (a contrast inside one picture reads at a glance), so only a missed idea
# gets frames. Most of the misses asked one picture for several moments ("taps each stack, then pushes them together").
TWO_FRAMES_PROMPT = """This manga panel's picture did not come out: the image model draws one frozen instant per picture, and this plan asks for more than it shows at once{why}. Draw the panel as two frames stacked in it, top then bottom, the way a manga shows a change.
- The top frame shows the first moment (the before, the cause, or the 'without'); the bottom frame the second (the after, the effect, or the 'with'). Each frame is ONE frozen instant: a state, or an action caught midway with its result already visible. Never "first ... then ..." inside one frame, never a frame split into halves.
- Draw the same objects the same way in both frames (same colors, sizes and places), so that the one visible difference between the frames is the panel's point.
- Only these characters appear: {characters}; a frame may leave any of them out. Nothing to count beyond five (a share is a tiny part beside a huge whole); never words, numbers, charts, boards, screens with text, or props shaped like people or creatures; use the objects the words name, so the reader knows what each stands for.
- "visual": what is drawn in that frame (setting, poses, objects and what happens to them). "shows": one plain sentence, the visible arrangement that carries this frame's part of the idea, with a cast member named by name, never by a role word or an abbreviation.

THE PANEL (its words are lettered over both frames):
scene: {scene}
visual (the plan that did not come out): {visual}
shows: {shows}
words: {words}

Output JSON only: {{"frames": [{{"visual": "...", "shows": "..."}}, {{"visual": "...", "shows": "..."}}]}}"""
FRAME_MIN_PIXELS = 921_600   # the image model's smallest picture (1280x720)
# One reference picture per book symbol, drawn before the panels and passed beside the character sheets wherever a
# panel draws the symbol. Round 18's Gated Attention named its gate "small round grey dimmer switch" in every visual
# and got knobs, pucks, keys, clamps and a sieve; with the sheets, blind judges found every symbol steadier (8 to 0, 2
# even) and preferred 13 pictures to 4 (15 even) over two books.
PROP_PROMPT = ("{style}\nA reference sheet of one prop for an illustrated book: {symbol}. One single {symbol}, drawn alone and "
               "centered on a plain white background, seen from slightly above, large and fully visible, in the same "
               "picture-book style. No people, no characters, no animals, no text, no labels, no numbers, no other objects.")
PROP_SIZE = (1024, 1024)
MAX_REFS = 10   # reference images the image model takes in one call
MAX_PROPS = 4   # symbol sheets per picture (the validated probe passed at most 4)


def two_frames_prompt(panel: PanelV2, problems: list[str]) -> str:
    """The plan of two frames for a teaching scene whose best picture missed its idea (``problems``: that picture's)."""
    idea = next((str(p) for p in problems if str(p).startswith(("idea not shown:", "idea unclear:"))), "")
    words = [f"{b.speaker}: {b.text}" for b in panel.dialogue] + ([f"narration: {panel.narration}"] if panel.narration else [])
    return TWO_FRAMES_PROMPT.format(why=f" (the check of the best attempt: {idea[:300]})" if idea else "",
                                    characters=", ".join(panel.characters) or "none", scene=panel.scene, visual=panel.visual,
                                    shows=panel.shows, words=" | ".join(words) or "(none)")


def parse_two_frames(data) -> Optional[list[dict]]:
    """[{"visual", "shows"}, {"visual", "shows"}] from a two_frames_prompt reply; None unless both frames have both (a frame
    without "shows" could not be idea-checked)."""
    frames = data.get("frames") if isinstance(data, dict) else None
    if not isinstance(frames, list) or len(frames) != 2 or not all(isinstance(f, dict) for f in frames):
        return None
    out = [{"visual": " ".join(str(f.get("visual") or "").split()), "shows": " ".join(str(f.get("shows") or "").split())} for f in frames]
    return out if all(f["visual"] and f["shows"] for f in out) else None


def frame_size(panel_size: tuple[int, int]) -> tuple[int, int]:
    """The size asked for one of two frames stacked in a panel: the panel's width by half its height (1024x768 of a
    1024x1536 panel), scaled up to the image model's smallest picture, in multiples of 16 (1120x832)."""
    w, h = int(panel_size[0]), int(panel_size[1]) // 2
    k = max(1.0, (FRAME_MIN_PIXELS / max(1, w * h)) ** 0.5)
    return -(-int(w * k) // 16) * 16, -(-int(h * k) // 16) * 16


def select_reference_names(panels: list[PanelV2], sheets: dict[str, ImageContent], cast: Cast) -> list[str]:
    """Sheets of the characters that appear (in cast order); all sheets when none of them has one; none when nobody
    appears (a frame of objects only: given the sheets, the image model drew the cast into it anyway)."""
    present = {c for p in panels for c in p.characters}
    if not present:
        return []
    names = [m.name for m in cast.members if m.name in present and m.name in sheets]
    if not names:
        names = [m.name for m in cast.members if m.name in sheets]
    return names


class PanelRenderer:
    def __init__(self, image_engine: BaseEngine, cast: Cast, sheets: dict[str, ImageContent], language: str,
                 panel_size: tuple[int, int] = (1024, 1536), strip_size: tuple[int, int] = (2240, 1120),
                 fast: bool = False, output_format: str = "jpeg", text_free: bool = False):
        self.engine = image_engine
        self.cast = cast
        self.sheets = sheets
        self.language = language
        self.panel_size = panel_size
        self.strip_size = strip_size
        self.fast = fast
        self.output_format = output_format
        self.text_free = text_free   # v3: wordless art (build_art_prompt); the text is typeset afterwards

    def _config(self, size: tuple[int, int]) -> ImageGenerationConfig:
        cfg = ImageGenerationConfig(width=size[0], height=size[1], style=self.cast.theme)
        cfg.extra = {"fast": self.fast, "output_format": self.output_format, "watermark": False}
        return cfg

    async def _generate(self, prompt: str, refs: list[ImageContent], size: tuple[int, int]) -> RenderResult:
        import time
        t0 = time.time()
        response: ImageResponse = await self.engine.generate_image(prompt, self._config(size), reference_images=refs)
        return RenderResult(image=response.images[0], prompt=prompt, usage=getattr(response, "usage", {}) or {},
                            size=getattr(response, "size", None) or f"{size[0]}x{size[1]}", seconds=round(time.time() - t0, 1))

    async def render_panel(self, panel: PanelV2, feedback: Optional[str] = None, anchor: Optional[ImageContent] = None,
                           size: Optional[tuple[int, int]] = None, props: Sequence[tuple[str, ImageContent]] = ()) -> RenderResult:
        """``size``: another picture size than the panel's (frame_size for one of two stacked frames). ``props``: (symbol,
        reference picture) of the book symbols this picture draws (v3), passed after the character sheets."""
        names = select_reference_names([panel], self.sheets, self.cast)
        refs = [self.sheets[n] for n in names]
        use_anchor = anchor is not None and len(refs) < MAX_REFS
        if use_anchor:
            refs = refs + [anchor]
        size = tuple(size) if size else self.panel_size
        if self.text_free:
            props = list(props)[:max(0, min(MAX_PROPS, MAX_REFS - len(refs)))]
            refs = refs + [image for _, image in props]
            prompt = build_art_prompt(panel, self.cast, names, feedback=feedback, anchor=use_anchor,
                                      shape="landscape" if size[0] > size[1] else "portrait", props=[s for s, _ in props])
        else:
            prompt = build_single_prompt(panel, self.cast, self.language, names, feedback=feedback, anchor=use_anchor)
        return await self._generate(prompt, refs, size)

    async def render_prop(self, symbol: str) -> RenderResult:
        """The reference picture of one book symbol (PROP_PROMPT), with no reference image of its own."""
        return await self._generate(PROP_PROMPT.format(style=self.cast.style, symbol=_symbol_words(symbol)), [], PROP_SIZE)

    async def render_strip(self, panels: list[PanelV2], feedback: Optional[str] = None) -> RenderResult:
        if self.text_free:   # v3 letters every panel on its own art; a lettered strip prompt here would put text in the art
            raise ValueError("text-free (v3) art is drawn one panel at a time; use render_panel")
        names = select_reference_names(panels, self.sheets, self.cast)
        refs = [self.sheets[n] for n in names]
        prompt = build_strip_prompt(panels, self.cast, self.language, names, feedback=feedback)
        return await self._generate(prompt, refs, self.strip_size)
