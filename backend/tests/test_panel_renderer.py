"""services.panel_renderer: Seedream prompt construction, reference selection and engine calls."""

import base64

import pytest

from engines.base import ImageContent
from services.cast import KUMOMO_LOOKS, THEME_STYLE, VOICES
from services.panel_renderer import (
    BUBBLE_RULES,
    NO_STRAY_TEXT,
    TEXT_RULES,
    PanelRenderer,
    build_single_prompt,
    build_strip_prompt,
    select_reference_names,
)
from services.storyboarder_v2 import Cast, CastMember

from .fakes import LOOKS, TINY_PNG_B64, FakeImageEngine, make_cast, make_panel, make_sheets


def test_single_prompt_has_scope_bubbles_speakers_narration_and_text_rules():
    panel = make_panel(3, characters=["papi", "kumo"], dialogue=[("papi", "Attention lets every word look at every other word.")],
                       narration="Self-attention links all 512 positions in one step.", labels=["Query", "Key"],
                       emotions={"papi": "proud", "kumo": "amazed"}, scene="A sunny classroom.",
                       visual="Papi points at a web of arrows between word cards.")
    prompt = build_single_prompt(panel, make_cast(), "en-US", ["papi", "kumo"])

    assert "Characters in this panel: papi, kumo only. nezu do NOT appear." in prompt
    assert '- papi says: "Attention lets every word look at every other word."' in prompt
    assert "Speakers: papi. kumo is silent." in prompt
    assert 'Narration box along the top edge: "Self-attention links all 512 positions in one step."' in prompt
    assert 'Labels drawn on props: "Query", "Key"' in prompt
    assert "Scene: A sunny classroom." in prompt and "Emotions: papi is proud, kumo is amazed." in prompt
    assert TEXT_RULES["en-US"] in prompt and BUBBLE_RULES in prompt and NO_STRAY_TEXT in prompt
    assert "previous attempt" not in prompt and "previous panel" not in prompt


@pytest.mark.parametrize("language, rule", [
    ("en-US", "in English"), ("zh-CN", "in Simplified Chinese characters"), ("ja-JP", "in Japanese"), ("ko-KR", "in English"),
])
def test_text_rule_follows_the_language(language, rule):
    assert rule in build_single_prompt(make_panel(1), make_cast(), language, ["papi"])


def test_chinese_prompt_quotes_every_bubble():
    panel = make_panel(1, characters=["papi", "kumo"], dialogue=[("papi", "注意力就是一切。"), ("kumo", "真的吗？")], narration="BLEU 达到 28.4。")
    prompt = build_single_prompt(panel, make_cast(), "zh-CN", ["papi", "kumo"])
    assert '- papi says: "注意力就是一切。"' in prompt and '- kumo says: "真的吗？"' in prompt
    assert 'Narration box along the top edge: "BLEU 达到 28.4。"' in prompt
    assert "Speakers:" not in prompt  # everyone in the panel speaks
    assert "Simplified Chinese" in prompt


def test_reference_block_ordinals_match_ref_names():
    cast = make_cast()
    panel = make_panel(1, characters=["papi", "nezu"], dialogue=[("nezu", "Prove it.")])
    two = build_single_prompt(panel, cast, "en-US", ["papi", "nezu"])
    assert f"The first reference image is papi: {LOOKS['papi']}." in two
    assert f"The second reference image is nezu: {LOOKS['nezu']}." in two
    assert "third reference image" not in two
    one = build_single_prompt(panel, cast, "en-US", ["kumo"])
    assert f"The reference image is kumo: {LOOKS['kumo']}." in one and "first reference image" not in one
    assert "reference image" not in build_single_prompt(panel, cast, "en-US", [])


def test_feedback_block_and_style_anchor_sentence():
    prompt = build_single_prompt(make_panel(1), make_cast(), "en-US", ["papi", "kumo"],
                                 feedback='papi\'s bubble must read exactly "Hi"', anchor=True)
    assert prompt.endswith('The previous attempt had these problems — fix them: papi\'s bubble must read exactly "Hi"')
    assert "The third reference image is the previous panel of this same manga" in prompt
    assert "do not copy its composition" in prompt


def test_panel_without_dialogue_says_so():
    prompt = build_single_prompt(make_panel(1, characters=["kumo"], dialogue=[]), make_cast(), "en-US", ["kumo"])
    assert "No speech bubbles in this panel." in prompt
    assert "Speech bubbles:" not in prompt and "Speakers:" not in prompt
    assert "Characters in this panel: kumo only. papi, nezu do NOT appear." in prompt


def test_strip_prompt_labels_left_and_right_panels():
    left = make_panel(1, characters=["papi"], dialogue=[("papi", "Left side line.")])
    right = make_panel(2, characters=["kumo"], dialogue=[("kumo", "Right side line.")])
    prompt = build_strip_prompt([left, right], make_cast(), "en-US", ["papi", "kumo"])
    assert prompt.index("PANEL 1 (left):") < prompt.index('"Left side line."') < prompt.index("PANEL 2 (right):") < prompt.index('"Right side line."')
    assert "exactly two equal panels side by side" in prompt
    assert "Characters in this strip: papi, kumo only. nezu do NOT appear." in prompt
    assert NO_STRAY_TEXT in prompt


def test_select_reference_names_uses_present_characters_in_cast_order():
    cast, sheets = make_cast(), make_sheets()
    assert select_reference_names([make_panel(1, characters=["nezu", "papi"])], sheets, cast) == ["papi", "nezu"]
    assert select_reference_names([make_panel(1, characters=["kumo"]), make_panel(2, characters=["nezu"])], sheets, cast) == ["kumo", "nezu"]
    partial = {"papi": sheets["papi"], "nezu": sheets["nezu"]}
    assert select_reference_names([make_panel(1, characters=["kumo"])], partial, cast) == ["papi", "nezu"]  # nobody present has one
    assert select_reference_names([make_panel(1)], {}, cast) == []
    # a frame of objects only gets no sheet: given all three, the image model drew kumo into round 19's smoke test frame
    assert select_reference_names([make_panel(1, characters=[])], sheets, cast) == []


def test_typical_panel_prompt_stays_under_450_words():
    cast = Cast("kumomo", THEME_STYLE["kumomo"], [CastMember(n, r, KUMOMO_LOOKS[n], VOICES[r])
                                                  for n, r in (("papi", "mentor"), ("kumo", "student"), ("nezu", "skeptic"))])
    panel = make_panel(
        5, characters=["papi", "kumo", "nezu"],
        dialogue=[("papi", "Each word asks every other word how much it matters, all at once."),
                  ("kumo", "So the whole sentence talks at the same time instead of waiting in line?")],
        narration="Self-attention connects any two positions in one step, while a recurrent network needs up to n sequential steps.",
        labels=["Query", "Key"], emotions={"papi": "proud", "kumo": "amazed", "nezu": "skeptical"},
        scene="A bright classroom with a big wooden board, afternoon light through tall windows and paper planes on the desks.",
        visual=("Papi stands at the board drawing glowing strings that connect a row of word cards, every card linked to every other "
                "card. Kumo floats above the desk with wide sparkling eyes, paws on cheeks. Nezu leans on the doorframe with a "
                "clipboard, one eyebrow raised, tapping a pencil."))
    prompt = build_single_prompt(panel, cast, "en-US", ["papi", "kumo", "nezu"])
    assert len(prompt.split()) < 450   # Seedream's prompt guide recommends staying under 600 words
    with_feedback = build_single_prompt(panel, cast, "en-US", ["papi", "kumo", "nezu"],
                                        feedback="kumo's bubble must read exactly \"So the whole sentence...\" (missing word)")
    assert len(with_feedback.split()) < 450


async def test_render_panel_sends_seedream_knobs_and_present_character_refs():
    image = FakeImageEngine(fixed_size=(1, 1), fmt="PNG")
    sheets = make_sheets()
    renderer = PanelRenderer(image, make_cast(), sheets, "en-US", fast=True)
    result = await renderer.render_panel(make_panel(2, characters=["kumo", "papi"], dialogue=[("kumo", "Why so slow?")]))

    (call,) = image.calls
    config = call["config"]
    assert (config.width, config.height) == (1024, 1536)
    assert config.extra == {"fast": True, "output_format": "jpeg", "watermark": False}
    assert call["refs"] == [sheets["papi"], sheets["kumo"]]  # present characters only, in cast order
    assert call["prompt"] == result.prompt and "The first reference image is papi" in result.prompt
    assert result.image.mime_type == "image/png" and base64.b64decode(result.image.data).startswith(b"\x89PNG")
    assert result.usage == {"output_tokens": 6144} and result.size == "1024x1536"


async def test_render_panel_with_anchor_and_render_strip():
    image = FakeImageEngine(fixed_size=(1, 1), fmt="PNG")
    sheets = make_sheets()
    renderer = PanelRenderer(image, make_cast(), sheets, "zh-CN", output_format="png")
    anchor = ImageContent.from_base64(TINY_PNG_B64)

    await renderer.render_panel(make_panel(1, characters=["nezu"], dialogue=[("nezu", "证明一下。")]), anchor=anchor, feedback="fix the tail")
    single = image.calls[0]
    assert single["refs"] == [sheets["nezu"], anchor]
    assert single["config"].extra == {"fast": False, "output_format": "png", "watermark": False}
    assert "The second reference image is the previous panel" in single["prompt"] and "fix them: fix the tail" in single["prompt"]

    await renderer.render_strip([make_panel(1, characters=["papi"], dialogue=[("papi", "左边。")]),
                                 make_panel(2, characters=["kumo"], dialogue=[("kumo", "右边。")])])
    strip = image.calls[1]
    assert (strip["config"].width, strip["config"].height) == (2240, 1120)   # 2.5 MP: same price tier as 2048x1024
    assert strip["refs"] == [sheets["papi"], sheets["kumo"]]
    assert "PANEL 2 (right):" in strip["prompt"]


def test_characters_without_a_sheet_get_their_looks_in_the_prompt():
    """Sheet-less themes used to send Seedream bare names; characters with a sheet keep the sheet sentence only."""
    panel = make_panel(1, characters=["papi", "kumo"], dialogue=[("papi", "Hello.")])
    partial = build_single_prompt(panel, make_cast(), "en-US", ["papi"])
    assert f"Characters without a reference sheet look like this: kumo: {LOOKS['kumo']}." in partial
    assert f"papi: {LOOKS['papi']}" not in partial.split("look like this:")[1]

    none = build_strip_prompt([panel, make_panel(2, characters=["nezu"])], make_cast(), "en-US", [])
    assert f"papi: {LOOKS['papi']}; kumo: {LOOKS['kumo']}; nezu: {LOOKS['nezu']}." in none

    full = build_single_prompt(panel, make_cast(), "en-US", ["papi", "kumo"])
    assert "without a reference sheet" not in full


def test_the_caption_rule_is_only_added_when_sheets_are_attached():
    from services.panel_renderer import SHEET_CAPTIONS
    panel = make_panel(1, characters=["papi"], dialogue=[("papi", "Hello.")])
    assert SHEET_CAPTIONS in build_single_prompt(panel, make_cast(), "en-US", ["papi"])
    assert SHEET_CAPTIONS not in build_single_prompt(panel, make_cast(), "en-US", [])
    assert SHEET_CAPTIONS in build_strip_prompt([panel, panel], make_cast(), "en-US", ["papi"])


def test_the_scope_line_keeps_rooms_and_streets_empty():
    """A scene 'past office floors' filled the offices with five workers in all three attempts (ResNet zh run)."""
    prompt = build_single_prompt(make_panel(1, characters=["papi"]), make_cast(), "en-US", ["papi"])
    assert "apart from the listed characters, every room, office, court, street, vehicle and building in it is empty of people" in prompt


def test_a_scene_that_mentions_written_text_asks_for_blank_cards():
    """BERT ja strip: 'papi points at a sentence with one word completely blacked out' came back as gibberish lettering."""
    from services.panel_renderer import TEXTY_HINT
    texty = make_panel(3, characters=["papi"], visual="papi points at a sentence with one word completely blacked out.")
    plain = make_panel(4, characters=["papi"], visual="papi holds up a plain power drill.")
    assert TEXTY_HINT in build_single_prompt(texty, make_cast(), "ja-JP", ["papi"])
    assert TEXTY_HINT not in build_single_prompt(plain, make_cast(), "ja-JP", ["papi"])
