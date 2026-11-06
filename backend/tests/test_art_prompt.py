"""services.panel_renderer v3: the text-free art prompt (build_art_prompt) and PanelRenderer(text_free=True)."""

import base64
import io
import re

import pytest
from PIL import Image

from engines.base import BaseEngine, ImageContent, ImageGenerationConfig, ImageResponse
from services.cast import KUMOMO_LOOKS, THEME_STYLE, VOICES
from services.panel_renderer import (ART_MAX_WORDS, ART_SHEET_CAPTIONS, ART_TEXTY_HINT, BOARD_COMPOSITION, SCENE_COMPOSITION, TEXT_RULES,
                                     WORDLESS, PanelRenderer, _board_text, _no_numbers, _unquoted, build_art_prompt, stage_order)
from services.storyboarder_v2 import Bubble, Cast, CastMember, PanelV2

LOOKS = {"papi": "a plump brown dog professor", "kumo": "a fluffy light-blue cloud creature", "nezu": "a brown hedgehog with a clipboard"}
QUOTES = re.compile(r"[\"“”「」『』«»]")


# ----------------------------------------------------------------------------- local fakes and builders

def small_cast() -> Cast:
    return Cast("kumomo", "Watercolor picture-book manga, soft pastel palette.",
                [CastMember(n, r, LOOKS[n], "calm") for n, r in (("papi", "mentor"), ("kumo", "student"), ("nezu", "skeptic"))])


def kumomo_cast() -> Cast:   # the real theme sentence and looks: the longest prompts
    return Cast("kumomo", THEME_STYLE["kumomo"], [CastMember(n, r, KUMOMO_LOOKS[n], VOICES[r])
                                                  for n, r in (("papi", "mentor"), ("kumo", "student"), ("nezu", "skeptic"))])


def make(characters=("papi", "kumo"), dialogue=(), narration="", kind="scene", board=None, labels=(), emotions=None,
         scene="A cozy kitchen at night.", visual="papi leans on the counter while kumo blinks.") -> PanelV2:
    return PanelV2(n=4, chapter="how", title="Panel title", characters=list(characters), scene=scene, visual=visual,
                   dialogue=[Bubble(s, t) for s, t in dialogue], narration=narration, labels=list(labels), emotions=dict(emotions or {}),
                   kind=kind, board=board)


class ImageFake(BaseEngine):
    """Seedream stand-in: records {"prompt", "config", "refs"} and returns a 1x1 PNG."""

    def __init__(self):
        super().__init__(api_key="test-key-not-real", base_url="http://fake.invalid", model="fake-seedream")
        self.calls = []

    @property
    def name(self):
        return "Fake Seedream"

    @property
    def supports_image_generation(self):
        return True

    @property
    def supports_vision(self):
        return False

    async def generate_text(self, messages, config=None):
        raise NotImplementedError

    async def generate_text_stream(self, messages, config=None):
        raise NotImplementedError

    async def generate_image(self, prompt, config=None, reference_images=None):
        self.calls.append({"prompt": prompt, "config": config or ImageGenerationConfig(), "refs": list(reference_images or [])})
        buf = io.BytesIO()
        Image.new("RGB", (1, 1), (255, 0, 0)).save(buf, format="PNG")
        response = ImageResponse(images=[ImageContent.from_base64(base64.b64encode(buf.getvalue()).decode(), "image/png")],
                                 model=self.model, prompt=prompt)
        response.usage, response.size = {"output_tokens": 6144}, "1024x1536"
        return response


def sheets():
    return {n: ImageContent.from_base64(f"c2hlZXQt{n}", "image/png") for n in ("papi", "kumo", "nezu")}


BUBBLES = (("kumo", "Wait, every word looks at every other word at once?"), ("papi", "Yes: 28.4 BLEU, more than 2 points above the rest."))
NARRATION = "BLEU scores a translation by its overlap with human translations."
BAR_BOARD = {"kind": "bar", "title": "WMT 2014 English to German (BLEU)", "source": "Table 2",
             "bars": [{"label": "ConvS2S ensemble", "value": "26.36"}, {"label": "Transformer (big)", "value": "28.4", "highlight": True}]}


# ----------------------------------------------------------------------------- the scene prompt

def test_scene_prompt_is_wordless_and_keeps_the_top_calm():
    prompt = build_art_prompt(make(dialogue=BUBBLES, narration=NARRATION, labels=["Query", "Key"]), small_cast(), ["papi", "kumo"])
    assert "A single manga panel, portrait, with a thin black border and a small white margin." in prompt
    assert "Characters in this panel: papi, kumo only. nezu do NOT appear." in prompt
    assert "Setting: A cozy kitchen at night.\nAction: papi leans on the counter while kumo blinks." in prompt
    assert ("Composition: the top 28% of the panel is calm, plain background (wall or sky) with nothing in it, reserved for text "
            "added later; no whiteboard, blackboard or screen anywhere in the picture; all heads are below that band; "
            "kumo on the left and papi on the right.") in prompt
    assert WORDLESS in prompt and ART_SHEET_CAPTIONS.strip() in prompt
    for text in [t for _, t in BUBBLES] + [NARRATION, "Query", "Key", "Panel title"]:
        assert text not in prompt
    assert not QUOTES.search(prompt) and "Speech bubbles" not in prompt and "Narration box" not in prompt
    assert all(rule not in prompt for rule in TEXT_RULES.values())


@pytest.mark.parametrize("characters, dialogue, order, placement", [
    (["papi"], [], ["papi"], "papi in the center"),
    (["papi", "kumo"], [], ["papi", "kumo"], "papi on the left and kumo on the right"),
    (["papi", "kumo"], [("kumo", "Hi."), ("papi", "Hello.")], ["kumo", "papi"], "kumo on the left and papi on the right"),
    (["papi", "kumo", "nezu"], [("nezu", "Prove it."), ("papi", "Look.")], ["nezu", "papi", "kumo"],
     "nezu on the left, papi in the middle and kumo on the right"),
    (["papi", "kumo", "nezu"], [("kumo", "Oh."), ("kumo", "Wow.")], ["kumo", "papi", "nezu"],
     "kumo on the left, papi in the middle and nezu on the right"),
])
def test_the_first_speaker_stands_on_the_left(characters, dialogue, order, placement):
    """Bubbles are read left to right, so the first speaker is drawn on the left and the silent characters come last."""
    p = make(characters=characters, dialogue=dialogue)
    assert stage_order(p) == order
    assert f"all heads are below that band; {placement}." in build_art_prompt(p, small_cast(), characters)


def test_quoted_text_in_the_scene_and_feedback_never_reaches_the_image_model():
    p = make(dialogue=[("kumo", "IT WORKED!")], visual='kumo shouts "IT WORKED!" and waves a sign saying “Attention!” at papi.',
             scene='A lab with a poster reading 「注意力」 and a note saying "hi.')
    prompt = build_art_prompt(p, small_cast(), ["papi", "kumo"], feedback='drawn text: remove "好我" from the clipboard')
    assert "IT WORKED" not in prompt and "Attention!" not in prompt and "注意力" not in prompt and "好我" not in prompt
    assert "Action: kumo shouts and waves a sign saying at papi." in prompt and not QUOTES.search(prompt)
    assert "Setting: A lab with a poster reading and a note saying hi." in prompt   # an unbalanced quote loses its mark
    assert prompt.endswith("The previous attempt had these problems — fix them: drawn text: remove from the clipboard")


def test_no_dialogue_or_narration_text_is_ever_in_the_prompt():
    lines = ["Zebra quartz seventeen.", "转换器只用注意力。", "Self-attention links any two words in one step!", "O(1) vs O(n)"]
    for kind in ("scene", "board"):
        for n in range(1, 4):
            p = make(characters=["papi", "kumo", "nezu"][:n], kind=kind, board=BAR_BOARD if kind == "board" else None,
                     dialogue=[(["papi", "kumo", "nezu"][i % n], lines[i]) for i in range(3)], narration=lines[3])
            prompt = build_art_prompt(p, small_cast(), p.characters, feedback="missing character: papi must be in the picture")
            assert not any(line in prompt for line in lines) and not QUOTES.search(prompt)


def test_emotions_and_the_writing_hint():
    p = make(visual="nezu scribbles furiously on his clipboard.", characters=["nezu"], emotions={"nezu": "suspicious"})
    prompt = build_art_prompt(p, small_cast(), ["nezu"])
    assert "Action: nezu scribbles furiously on his clipboard. Emotions: nezu is suspicious." in prompt
    assert ART_TEXTY_HINT in prompt and "clipboards" in WORDLESS and "clock faces" in WORDLESS
    assert ART_TEXTY_HINT not in build_art_prompt(make(), small_cast(), ["papi", "kumo"])


# ----------------------------------------------------------------------------- the board prompt

def test_board_prompt_asks_for_a_large_blank_board_and_no_chart():
    p = make(kind="board", board=BAR_BOARD, dialogue=BUBBLES, scene="Lab whiteboard.",
             visual="papi points at the bar chart where the 28.4 bar towers over the others; kumo gasps at the 2.3 · 10^19 FLOPs.")
    prompt = build_art_prompt(p, small_cast(), ["papi", "kumo"])
    # what is on the board ("where the 28.4 bar towers over the others") is left out: the picture gets a blank board
    assert "Action (draw nothing on the board): papi points at the board; kumo gasps at the FLOPs." in prompt
    assert ("Composition: the top 22% of the panel is plain empty wall reserved for text. Below it, a LARGE, COMPLETELY BLANK white "
            "whiteboard spans almost the full width of the panel and about 48% of its height, seen straight on, evenly lit, with a thin "
            "dark frame. Nothing stands in front of it or overlaps it, and nothing is written or drawn on it. At the bottom, below the "
            "board, kumo on the left and papi on the right are shown from the chest up.") in prompt
    for text in ("28.4", "26.36", "10^19", "WMT", "ConvS2S", "Table 2", "chart", "top 28%", ART_TEXTY_HINT) + tuple(t for _, t in BUBBLES):
        assert text not in prompt
    assert WORDLESS in prompt


def test_board_settings_lose_their_chart_words_too():
    p = make(kind="board", board=BAR_BOARD, scene="The lab, a bar chart of 28.4 BLEU on the whiteboard.")
    assert "Setting: The lab, a board of BLEU on the whiteboard." in build_art_prompt(p, small_cast(), ["papi", "kumo"])


def test_feedback_is_kept_whole_when_the_prompt_has_room():
    feedback = "; ".join(f"extra figure: remove the grey cat number {w} sitting on the windowsill" for w in "abcdefghij")   # ~110 words
    prompt = build_art_prompt(make(), small_cast(), ["papi", "kumo"], feedback=feedback)
    assert prompt.endswith(feedback) and len(prompt.split()) <= ART_MAX_WORDS


def test_a_board_panel_with_one_character():
    prompt = build_art_prompt(make(characters=["nezu"], kind="board", board={"kind": "list", "items": ["a", "b"]}), small_cast(), ["nezu"])
    assert "At the bottom, below the board, nezu in the center is shown from the chest up." in prompt


@pytest.mark.parametrize("text, kind, expected", [
    ("Papi points at the bar chart.", "bar", "Papi points at the board."),
    ("Papi taps the training cost bar.", "bar", "Papi taps the training cost board."),
    ("Kumo studies the comparison table on the whiteboard.", "flow", "Kumo studies the board on the whiteboard."),
    ("Papi points at the table.", "table", "Papi points at the board."),
    ("Papi sits at the kitchen table next to the board.", "bar", "Papi sits at the kitchen table next to the board."),
    ("Nezu reads the recap list and the whiteboard's graph.", "list", "Nezu reads the board and the whiteboard."),
    ("Papi traces the flowchart arrows; 4.9% fewer steps.", "flow", "Papi traces the board; fewer steps."),
    ("Kumo steps back, amazed.", "flow", "Kumo steps back, amazed."),
    # reviewer fixes: furniture stays furniture, the board's content goes, numbers go whole and names with digits stay
    ("Papi sits at the kitchen table, pointing at the table on the whiteboard.", "table",
     "Papi sits at the kitchen table, pointing at the board on the whiteboard."),
    ("Papi taps the board showing the Transformer bar, grinning.", "bar", "Papi taps the board, grinning."),
    ("Papi points at the bar chart where the 28.4 bar towers over the rest while kumo cheers.", "bar",
     "Papi points at the board while kumo cheers."),
    ("Kumo gasps at 2.3×10¹⁹ FLOPs and the 1e19 bar.", "bar", "Kumo gasps at FLOPs and the board."),
    ("Papi holds 2 markers next to 8 P100 GPUs and a GPT-2 bar.", "bar", "Papi holds two markers next to eight P100 GPUs and a GPT-2 board."),
])
def test_chart_words_and_numbers_leave_the_board_text(text, kind, expected):
    assert _board_text(text, kind) == expected


# ----------------------------------------------------------------------------- references, anchor, feedback, length

def test_reference_mapping_anchor_and_sheetless_looks():
    cast = small_cast()
    p = make(characters=["papi", "nezu"], dialogue=[("nezu", "Prove it.")])
    two = build_art_prompt(p, cast, ["papi", "nezu"], anchor=True)
    assert f"The first reference image is papi: {LOOKS['papi']}." in two and f"The second reference image is nezu: {LOOKS['nezu']}." in two
    assert "The third reference image is the previous panel of this same manga" in two and "do not copy its composition or poses" in two
    partial = build_art_prompt(p, cast, ["papi"])
    assert f"Characters without a reference sheet look like this: nezu: {LOOKS['nezu']}." in partial
    bare = build_art_prompt(p, cast, [])
    assert "reference image" not in bare and ART_SHEET_CAPTIONS.strip() not in bare and WORDLESS in bare


def test_feedback_block_comes_last():
    prompt = build_art_prompt(make(), small_cast(), ["papi", "kumo"], feedback="busy top: keep the top 28% plain; clone: draw kumo only once")
    assert prompt.endswith("The previous attempt had these problems — fix them: busy top: keep the top 28% plain; clone: draw kumo only once")
    assert "previous attempt" not in build_art_prompt(make(), small_cast(), ["papi", "kumo"])


@pytest.mark.parametrize("kind", ["scene", "board"])
def test_long_scenes_and_feedback_stay_under_the_word_limit(kind):
    long_visual = " ".join(["papi waves a long wooden pointer at the far corner while kumo hops from foot to foot."] * 14)
    long_scene = " ".join(["A sunlit lab full of glass jars, potted plants, humming machines and stacked crates."] * 8)
    feedback = "; ".join(["missing character: kumo must be in the picture, with the whole head visible",
                          "extra figure: remove a grey cat sitting on the windowsill; nobody but papi, kumo and nezu appears",
                          "drawn text: remove every letter, digit and symbol (on the clock face, the poster); every surface stays blank"] * 5)
    p = make(characters=["papi", "kumo", "nezu"], kind=kind, board=BAR_BOARD if kind == "board" else None, scene=long_scene,
             visual=long_visual, emotions={"papi": "proud", "kumo": "amazed", "nezu": "skeptical"},
             dialogue=[("nezu", "Show me the numbers."), ("papi", "Here they are.")])
    prompt = build_art_prompt(p, kumomo_cast(), ["papi", "kumo", "nezu"], feedback=feedback, anchor=True)
    assert len(prompt.split()) <= ART_MAX_WORDS == 450
    assert WORDLESS in prompt and "Composition:" in prompt and "fix them: missing character: kumo" in prompt
    assert "Emotions: papi is proud, kumo is amazed, nezu is skeptical." in prompt
    plain = build_art_prompt(make(characters=["papi", "kumo", "nezu"], kind=kind, board=BAR_BOARD), kumomo_cast(), ["papi", "kumo", "nezu"])
    assert len(plain.split()) < 330   # typical prompts stay well under the limit


# ----------------------------------------------------------------------------- PanelRenderer

async def test_text_free_renderer_draws_with_the_art_prompt():
    image, sh = ImageFake(), sheets()
    renderer = PanelRenderer(image, small_cast(), sh, "zh-CN", text_free=True)
    p = make(characters=["kumo", "papi"], dialogue=[("kumo", "为什么这么慢？")], narration="循环网络一次只读一个词。")
    result = await renderer.render_panel(p, feedback="busy top: keep the top plain")
    (call,) = image.calls
    assert call["prompt"] == result.prompt == build_art_prompt(p, small_cast(), ["papi", "kumo"], feedback="busy top: keep the top plain")
    assert call["refs"] == [sh["papi"], sh["kumo"]] and (call["config"].width, call["config"].height) == (1024, 1536)
    assert "为什么这么慢" not in result.prompt and "循环网络" not in result.prompt and WORDLESS in result.prompt

    anchor = ImageContent.from_base64("YW5jaG9y", "image/png")
    await renderer.render_panel(make(characters=["nezu"]), anchor=anchor)
    assert image.calls[1]["refs"] == [sh["nezu"], anchor] and "The second reference image is the previous panel" in image.calls[1]["prompt"]


async def test_the_default_renderer_keeps_the_lettering_prompt_and_strips_still_work():
    image = ImageFake()
    renderer = PanelRenderer(image, small_cast(), sheets(), "en-US")
    assert renderer.text_free is False
    result = await renderer.render_panel(make(dialogue=[("papi", "Hello there.")]))
    assert '- papi says: "Hello there."' in result.prompt and TEXT_RULES["en-US"] in result.prompt and WORDLESS not in result.prompt
    await renderer.render_strip([make(dialogue=[("papi", "Left.")]), make(dialogue=[("kumo", "Right.")])])
    assert "exactly two equal panels side by side" in image.calls[1]["prompt"]


async def test_a_frame_of_two_is_drawn_landscape_at_the_size_asked_for():
    from services.panel_renderer import frame_size
    image = ImageFake()
    renderer = PanelRenderer(image, small_cast(), sheets(), "en-US", text_free=True)
    size = frame_size(renderer.panel_size)
    assert size == (1120, 832) and size[0] * size[1] >= 921_600   # half the 1024x1536 panel, at least the model's smallest picture
    await renderer.render_panel(make(), size=size)
    await renderer.render_panel(make())
    (frame, panel) = image.calls
    assert (frame["config"].width, frame["config"].height) == size and "A single manga panel, landscape," in frame["prompt"]
    assert (panel["config"].width, panel["config"].height) == (1024, 1536) and "A single manga panel, portrait," in panel["prompt"]
    w, h = frame_size((512, 768))
    assert w > h and w * h >= 921_600 and w % 16 == h % 16 == 0
    nobody = build_art_prompt(make(characters=()), small_cast(), [])   # a frame may leave every character out
    assert "No characters in this panel. papi, kumo, nezu do NOT appear." in nobody


def test_a_picture_that_missed_its_idea_is_planned_as_two_frames():
    from services.panel_renderer import parse_two_frames, two_frames_prompt
    p = make(dialogue=[("papi", "We train on two types of tasks.")], narration="Mixing them avoids gaps.",
             visual="papi taps each stack once, then pushes both stacks together into one pile.")
    p.shows = "two stacks pushed into one mixed pile"
    prompt = two_frames_prompt(p, ["busy top: keep the top plain",
                                   "idea unclear: the picture must show two stacks pushed into one pile; no pushing of the stacks"])
    assert "(the check of the best attempt: idea unclear: the picture must show two stacks pushed into one pile; no pushing" in prompt
    assert "Only these characters appear: papi, kumo;" in prompt and "papi: We train on two types of tasks." in prompt
    assert "narration: Mixing them avoids gaps." in prompt and p.visual in prompt and "busy top" not in prompt
    assert parse_two_frames({"frames": [{"visual": " papi  holds two stacks ", "shows": "two stacks"},
                                        {"visual": "one pile", "shows": "one mixed pile"}]}) == [
        {"visual": "papi holds two stacks", "shows": "two stacks"}, {"visual": "one pile", "shows": "one mixed pile"}]
    for bad in ({"frames": [{"visual": "one", "shows": "a"}]}, {"frames": [{"visual": "", "shows": "a"}, {"visual": "two", "shows": "b"}]},
                {"frames": [{"visual": "one", "shows": "a"}, {"visual": "two"}]},   # a frame without shows is never idea-checked
                {"frames": "x"}, [], None):
        assert parse_two_frames(bad) is None, bad


async def test_a_book_symbol_is_drawn_from_its_reference_picture():
    # round 18's Gated Attention drew "small round grey dimmer switch" as knobs, pucks, keys, clamps and a sieve
    from services.panel_renderer import MAX_PROPS, MAX_REFS, PROP_SIZE
    image, sh = ImageFake(), sheets()
    renderer = PanelRenderer(image, small_cast(), sh, "en-US", text_free=True)
    prop = ImageContent.from_base64("cHJvcA==", "image/png")
    await renderer.render_panel(make(characters=["papi"]), props=[("small round grey dimmer switch", prop)])
    (call,) = image.calls
    assert call["refs"] == [sh["papi"], prop]
    assert "The first reference image is papi:" in call["prompt"]   # numbered: two pictures follow, not one
    assert ("The second reference image shows the small round grey dimmer switch: draw it exactly as that image shows it"
            in call["prompt"])
    many = [(f"prop {w}", prop) for w in "abcdefghijkl"]
    await renderer.render_panel(make(characters=["papi", "kumo"]), props=many)
    assert len(image.calls[1]["refs"]) == 2 + MAX_PROPS <= MAX_REFS and "shows the prop d:" in image.calls[1]["prompt"]
    assert "prop e" not in image.calls[1]["prompt"]
    result = await renderer.render_prop("black circular drain hole")
    assert image.calls[2]["refs"] == [] and (image.calls[2]["config"].width, image.calls[2]["config"].height) == PROP_SIZE
    assert "One single black circular drain hole, drawn alone" in result.prompt and "no text" in result.prompt
    lettered = PanelRenderer(image, small_cast(), sh, "en-US")   # the v2 lettered prompt draws no symbol sheets
    await lettered.render_panel(make(characters=["papi"]), props=[("dimmer", prop)])
    assert image.calls[3]["refs"] == [sh["papi"]]


def test_symbol_sheets_never_cost_a_panel_its_action_or_a_retry_its_feedback():
    # round 20's review: the sheet lines sat in the prompt's unclipped head, and the word cap cut the action and dropped
    # the retry's feedback instead (a paid redraw that never asked for the fix)
    p = make(characters=("papi", "kumo", "nezu"), scene="A cluttered workshop with tall shelves of tools, jars and boxes. " * 3,
             visual="papi pours a stream of white tiles through a wide copper sieve into a tall glass jar of glowing green "
                    "marbles, while kumo turns a small round grey dimmer switch and nezu watches the black drain hole. " * 3)
    p.beat, p.shows = "how", "the tiles pass through the sieve into the jar"
    feedback = "extra figure: remove a small bird; nobody but papi, kumo and nezu appears"
    props = ["small round grey dimmer switch", "wide copper sieve", "tall glass jar of glowing green marbles", "black drain hole"]
    plain = build_art_prompt(p, kumomo_cast(), ["papi", "kumo", "nezu"], feedback=feedback)
    with_sheets = build_art_prompt(p, kumomo_cast(), ["papi", "kumo", "nezu"], feedback=feedback, props=props)
    assert "The previous attempt had these problems" in plain   # the case only means something if the plain prompt keeps it
    body = lambda text: text.split("Action:", 1)[1]
    assert body(with_sheets) == body(plain)   # the same action, shows line, rules and feedback
    assert all(f"shows the {s}: draw it exactly" in with_sheets for s in props)
    # the symbol is told like the panel's own words: no quotes, digits or first-aid crosses copied into every panel
    cleaned = build_art_prompt(make(characters=["papi"]), small_cast(), ["papi"], props=['white slip with a red cross and a "3"'])
    assert "shows the white slip with a red X mark and a: draw it" in cleaned and '"3"' not in cleaned


async def test_a_symbol_sheet_is_drawn_from_the_cleaned_symbol():
    image = ImageFake()
    renderer = PanelRenderer(image, small_cast(), sheets(), "en-US", text_free=True)
    await renderer.render_prop("white slip marked with a red cross, numbered 3")
    (call,) = image.calls
    assert "white slip marked with a red X mark, numbered" in call["prompt"] and "red cross" not in call["prompt"]
    assert "3" not in call["prompt"].split("for an illustrated book:", 1)[1]


async def test_text_free_renderer_refuses_strips():
    image = ImageFake()
    renderer = PanelRenderer(image, small_cast(), sheets(), "en-US", text_free=True)
    with pytest.raises(ValueError, match="one panel at a time"):
        await renderer.render_strip([make(), make()])
    assert image.calls == []


# ----------------------------------------------------------------------------- reviewer fixes

@pytest.mark.parametrize("text, expected", [
    ("Kumo stares at 12 stacked recipe cards.", "Kumo stares at twelve stacked recipe cards."),
    ("Kumo pulls 4 cake layers out of 4 ovens.", "Kumo pulls four cake layers out of four ovens."),
    ("A 28.4 trophy, a 4.9% sticker and 1,000 cards.", "A trophy, a sticker and cards."),
    ("About 2.3 · 10^19 operations, 10¹⁹ more, or 1e19.", "About operations, more, or."),
    ("Eight P100 GPUs, GPT-2, 3D glasses and the 50th try.", "Eight P100 GPUs, GPT-2, 3D glasses and the 50th try."),
    ("A poster from 2017 and the 13 cards.", "A poster from and the cards."),
    ("At 3:00 kumo eats 1/2 a cake, 10x faster, 4-5 bites.", "At kumo eats a cake, faster, four-five bites."),
])
def test_digits_never_reach_the_image_model(text, expected):
    """A small count becomes a word, any other number goes: digits in the prompt get drawn as digits."""
    assert _no_numbers(text) == expected


def test_scene_numbers_and_emotions_are_spelled_or_dropped():
    p = make(characters=["kumo"], scene="A kitchen with 3 ovens.", visual="Kumo stares at 12 stacked recipe cards and a 28.4 trophy.",
             emotions={"kumo": "100% overwhelmed"})
    prompt = build_art_prompt(p, small_cast(), ["kumo"])
    assert "Setting: A kitchen with three ovens.\nAction: Kumo stares at twelve stacked recipe cards and a trophy. Emotions: kumo is overwhelmed." in prompt
    assert ART_TEXTY_HINT in prompt   # the raw scene had digits: the blank-cards hint still applies
    scene_lines = [line for line in prompt.splitlines() if line.startswith(("Setting:", "Action:"))]
    assert len(scene_lines) == 2 and not re.search(r"\d", " ".join(scene_lines))   # only the layout keeps its "top 28%"


def test_single_quoted_text_goes_and_apostrophes_stay():
    assert _unquoted("kumo holds a card reading 'Attention' next to papi's hat in the papis' den") == \
        "kumo holds a card reading next to papi's hat in the papis' den"
    prompt = build_art_prompt(make(visual="nezu waves a sign saying 'Prove it!' at papi's desk."), small_cast(), ["papi", "kumo"])
    assert "Prove it" not in prompt and "at papi's desk." in prompt


@pytest.mark.parametrize("board", [None, {}])
def test_a_board_panel_without_an_exhibit_is_drawn_as_a_scene(board):
    """lettering letters a "board" panel without a board spec as a scene: the art keeps the calm top band, no whiteboard."""
    prompt = build_art_prompt(make(kind="board", board=board, visual="papi points at the bar chart."), small_cast(), ["papi", "kumo"])
    assert SCENE_COMPOSITION in prompt and BOARD_COMPOSITION not in prompt and "Action: papi points at something just out of view." in prompt


def test_a_chart_left_in_a_scene_is_drawn_as_a_blank_board():
    # Driver Misconduct's "Kumo looks at the bar chart" came back as two fake charts without labels or values
    prompt = build_art_prompt(make(scene="Kumo looks at the bar chart", visual="Nezu traces the rising bar heights, then a slide."),
                              small_cast(), [])
    assert "Setting: Kumo looks at something just out of view" in prompt and "chart" not in prompt.lower()
    assert "Nezu traces something just out of view, then a slide." in prompt   # a playground slide is not a chart
    prompt = build_art_prompt(make(visual="Papi taps the fake remark bar on the board"), small_cast(), [])
    assert "Action: Papi taps something just out of view" in prompt


def test_chat_windows_in_a_scene_are_drawn_as_glowing_orbs():
    # Hivemind's first panel: "empty reply windows float above them" came back as three empty speech bubbles
    prompt = build_art_prompt(make(scene="Desk with floating chat windows", visual="Kumo slumps, empty reply windows float above them."),
                              small_cast(), [])
    assert "Setting: Desk with small glowing orbs" in prompt and "small glowing orbs float above them" in prompt
    assert "window" not in prompt.lower().replace("chat windows, text boxes", "")   # only the wordless rule names them
    assert "chat windows, text boxes" in WORDLESS
    assert "opens a window" in build_art_prompt(make(visual="Kumo opens a window."), small_cast(), [])   # a real window stays


def test_a_teaching_scene_asks_for_its_idea_in_big_uncovered_objects():
    p = make(visual="papi holds a small card next to a tall stack of blank pages")
    p.beat, p.shows = "how", 'the "locked" card comes first, then 8 pages grow around it'
    prompt = build_art_prompt(p, small_cast(), [])
    line = next(x for x in prompt.splitlines() if x.startswith("The picture itself must make this clear"))
    assert "the card comes first, then eight pages grow around it" in line and "LARGE and in front" in line   # quoted words go
    assert not QUOTES.search(prompt) and "8" not in line
    board = make(kind="board", board={"kind": "list", "title": "T", "items": ["a", "b"]})
    board.shows = "should never be used"
    assert "The picture itself must make this clear" not in build_art_prompt(board, small_cast(), [])
    assert "The picture itself must make this clear" not in build_art_prompt(make(), small_cast(), [])   # no shows line
    p.beat = "proof"                                    # a story-only scene (the skeptic's challenge) is not drawn to an idea
    assert "The picture itself must make this clear" not in build_art_prompt(p, small_cast(), [])


def test_a_chart_is_never_the_idea_a_picture_is_asked_to_show():
    # "the tall bar chart towers over a tiny one" became "something just out of view towers over a tiny one": drawn and
    # judged against nothing
    from services.panel_renderer import art_shows
    p = make(visual="papi holds a tall stack of cards next to a tiny one")
    p.beat, p.shows = "how", "the tall bar chart towers over a tiny one"
    assert art_shows(p) == "" and "The picture itself must make this clear" not in build_art_prompt(p, small_cast(), [])
    p.shows = "the tall stack of cards towers over a tiny one"
    assert art_shows(p) == "the tall stack of cards towers over a tiny one"


def test_a_red_cross_is_drawn_as_an_x_mark():
    # the delivery book's "white slip marked with a red cross" came back as a first-aid plus, and the audit read the
    # fake-remark slips as medical kits; the words meant a mark of failure (红叉)
    from services.panel_renderer import art_shows
    p = make(visual="nezu slips a white red-cross slip onto a parcel; two red crosses lie on the floor")
    p.beat, p.shows = "how", "the parcel with the red cross slip falls"
    prompt = build_art_prompt(p, small_cast(), [])
    assert "red X mark slip" in prompt and "two red X marks" in prompt and "red cross" not in prompt.lower()
    assert art_shows(p) == "the parcel with the red X mark slip falls"
    p.visual = "kumo walks across the street"   # no mark: nothing changes
    assert "walks across the street" in build_art_prompt(p, small_cast(), [])
