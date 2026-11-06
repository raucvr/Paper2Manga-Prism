"""services.art_check: grounding JSON variants → pixels, the vision call (local fakes only), art problems, ranking."""

import base64
import dataclasses
import io
import json
import re

import numpy as np
import pytest
from PIL import Image, ImageDraw

from engines.base import BaseEngine, ImageContent, TextResponse
from services import lettering
from services.art_check import (CLONE, MAX_TRIES, Grounding, GroundingError, _is_nothing, art_problems, calm_band, clean_rect,
                                grounding_from_json, is_board_panel, locate, rank_key)
from services.storyboarder_v2 import Bubble, Cast, CastMember, PanelV2

NAMES = ["papi", "kumo", "nezu"]
SIZE = (1024, 1536)


# ----------------------------------------------------------------------------- local fakes and builders

class VisionFake(BaseEngine):
    """Replies from a queue (a str, a dict sent as JSON, or an exception that is raised); records (messages, config)."""

    def __init__(self, replies, usage=None):
        super().__init__(api_key="test-key-not-real", base_url="http://fake.invalid", model="fake-vision")
        self.replies = list(replies)
        self.usage = dict(usage or {"prompt_tokens": 1500, "completion_tokens": 120})
        self.calls = []

    @property
    def name(self):
        return "Fake vision"

    @property
    def supports_image_generation(self):
        return False

    @property
    def supports_vision(self):
        return True

    async def generate_text(self, messages, config=None):
        self.calls.append((messages, config))
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        content = reply if isinstance(reply, str) else json.dumps(reply)
        return TextResponse(content=content, model=self.model, usage=dict(self.usage))

    async def generate_text_stream(self, messages, config=None):
        raise NotImplementedError


def png(width, height, color=(255, 255, 255)) -> ImageContent:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return ImageContent.from_base64(base64.b64encode(buf.getvalue()).decode("ascii"), "image/png")


def cast() -> Cast:
    return Cast("kumomo", "Watercolor picture-book manga.", [
        CastMember("papi", "mentor", "a plump brown dog professor", "patient"),
        CastMember("kumo", "student", "a fluffy light-blue cloud creature", "curious"),
        CastMember("nezu", "skeptic", "a brown hedgehog with a clipboard", "dry"),
    ])


BAR = {"kind": "bar", "title": "BLEU", "bars": [{"label": "ConvS2S", "value": "26.36"}, {"label": "Transformer", "value": "28.4", "highlight": True}]}


def panel(characters=("papi", "kumo"), kind="scene", board=None) -> PanelV2:
    """A board panel gets an exhibit unless one is given: lettering (and so art_check) only treats it as a board then."""
    return PanelV2(n=1, chapter="how", title="", characters=list(characters), scene="", visual="", kind=kind,
                   board=board if board is not None else (dict(BAR) if kind == "board" else None))


def noise(h, w, seed=0) -> np.ndarray:
    """High-contrast 4-px blocks: every 8-px cell is full of edges, nothing is blank."""
    blocks = np.random.default_rng(seed).integers(0, 2, (h // 4 + 1, w // 4 + 1), dtype=np.uint8) * 200 + 20
    return np.repeat(np.kron(blocks, np.ones((4, 4), dtype=np.uint8))[:h, :w, None], 3, axis=2)


def scene_art(w=400, h=600, calm=0.30) -> Image.Image:
    a = noise(h, w)
    a[: int(h * calm)] = (246, 241, 230)     # plain wall
    return Image.fromarray(a, "RGB")


BOARD = (30, 150, 370, 450)                    # frame; the blank surface is 6 px inside it
SURFACE = [(36, 156), (364, 156), (364, 444), (36, 444)]


def board_art(w=400, h=600, clutter=False, board=BOARD) -> Image.Image:
    a = noise(h, w, seed=1)
    x1, y1, x2, y2 = board
    a[y1:y2, x1:x2] = (40, 40, 40)
    a[y1 + 6:y2 - 6, x1 + 6:x2 - 6] = (250, 250, 248)
    img = Image.fromarray(a, "RGB")
    if clutter:   # marker scribbles every 40 px: no large blank rectangle is left
        d = ImageDraw.Draw(img)
        for x in range(60, 360, 40):
            d.line([(x, 160), (x, 440)], fill=(30, 30, 90), width=3)
        for y in range(180, 440, 40):
            d.line([(40, y), (360, y)], fill=(30, 30, 90), width=3)
    return img


# ----------------------------------------------------------------------------- grounding JSON → Grounding

def test_grounding_has_exactly_the_fields_lettering_reads():
    assert [f.name for f in dataclasses.fields(Grounding)] == ["size", "heads", "board", "bubbles", "text_found", "extras", "matches", "objects"]
    g = Grounding(size=(10, 20))   # the empty grounding is a valid fallback
    assert (g.heads, g.board, g.bubbles, g.text_found, g.extras, g.matches) == ({}, None, [], [], [], {})


PX_QUAD = [(41, 455), (980, 455), (980, 1184), (41, 1184)]   # [[40, 296], [957, 296], [957, 771], [40, 771]] at 1024x1536


@pytest.mark.parametrize("boards", [
    [{"corners": [[40, 296], [957, 296], [957, 771], [40, 771]]}],          # the documented shape
    [[[40, 296], [957, 296], [957, 771], [40, 771]]],                       # a bare list of 4 points (seen in the prototype)
    [{"corners": [[40, 296, 957, 771]]}],                                   # 4 numbers inside "corners" (seen)
    [[[40, 296, 957, 771]]],                                                # 4 numbers, nested (seen)
    [40, 296, 957, 771],                                                    # one bare box instead of a list of boards
    [[40, 296], [957, 296], [957, 771], [40, 771]],                         # one bare list of points
    [{"corners": [[40, 296, 957, 296], [957, 296], [957, 771], [40, 771]]}],  # a repeated corner (seen)
    [{"corners": [[957, 771], [40, 771], [40, 296], [957, 296]]}],          # any order: returned clockwise from top-left
    [{"corners": [{"x": 40, "y": 296}, {"x": 957, "y": 296}, {"x": 957, "y": 771}, {"x": 40, "y": 771}]}],
    {"corners": [[40, 296], [957, 296], [957, 771], [40, 771]]},            # one object instead of a list
])
def test_board_shapes_the_model_sends_become_one_pixel_quad(boards):
    g = grounding_from_json({"boards": boards}, SIZE, NAMES)
    assert g.board == PX_QUAD


@pytest.mark.parametrize("boards", [
    [{"corners": [[40, 296], [957, 296], [957, 771]]}],                     # 3 corners
    [{"corners": [[40, 296], [957, 296], [957, 771], [40, 771], [500, 900]]}],  # 5 distinct corners
    [[40, 296, 957, 1771]],                                                 # outside the 0-1000 frame (pixels?)
    [[40, 296, 42, 771]],                                                   # a sliver
    [{"corners": "the whiteboard"}], [None], [], None, "none",
])
def test_invalid_boards_are_dropped(boards):
    assert grounding_from_json({"boards": boards}, SIZE, NAMES).board is None


def test_the_largest_board_wins_and_small_overshoot_is_clamped():
    data = {"boards": [{"corners": [[390, 438, 447, 489]]}, [[-4, 200], [1006, 200], [1006, 800], [-4, 800]]]}
    assert grounding_from_json(data, (1000, 1000), NAMES).board == [(0, 200), (1000, 200), (1000, 800), (0, 800)]


def test_names_map_onto_the_cast_clones_and_strangers_become_extras():
    data = {"characters": [
        {"name": "papi (brown dog)", "head": [470, 425, 835, 635], "matches_reference": True},
        {"name": "Kumo", "head": [135, 490, 390, 655], "matches_reference": "false"},
        {"name": "kumo (blue cloud creature)", "head": [600, 700, 700, 800]},   # drawn twice
        {"name": "a grey cat", "head": [10, 10, 100, 100]},                    # not a cast member
        {"name": "nezu", "head": None},                                        # not located
        {"name": "nezu", "head": [1, 2, 3, 4], "present": False},
    ]}
    g = grounding_from_json(data, SIZE, NAMES)
    assert g.heads == {"papi": (481, 653, 855, 975), "kumo": (138, 753, 399, 1006)}
    assert g.matches == {"papi": True, "kumo": False}
    assert g.extras == [CLONE.format(name="kumo"), "a grey cat"]


def test_head_boxes_are_sorted_clamped_or_dropped():
    data = {"characters": [{"name": "papi", "head": [400, 600, 100, 300]},          # corners swapped
                           {"name": "kumo", "head": [990, 100, 1004, 200]},         # overshoots by 0.4%
                           {"name": "nezu", "head": [100, 100, 400, 1300]}]}        # not normalized
    g = grounding_from_json(data, (1000, 2000), NAMES)
    assert g.heads == {"papi": (100, 600, 400, 1200), "kumo": (990, 200, 1000, 400)}


def test_characters_given_as_a_mapping():
    g = grounding_from_json({"characters": {"papi": [100, 100, 300, 300], "kumo": {"head": [500, 500, 700, 700], "matches_reference": False}}},
                            (1000, 1000), NAMES)
    assert g.heads == {"papi": (100, 100, 300, 300), "kumo": (500, 500, 700, 700)} and g.matches == {"kumo": False}


def test_text_bubbles_extras_and_placeholders():
    data = {
        "characters": [{"name": "papi", "head": [100, 500, 300, 700]}, {"name": "kumo", "head": [600, 500, 800, 700]}],
        "text_or_symbols": [{"text": "12", "where": "on the clock face"}, "好我", "none", {"text": "None", "where": ""},
                            {"text": "", "where": "nezu's clipboard"}, "12 (on the clock face)"],
        "speech_bubbles": [[60, 120, 530, 423], {"box": [505, 45, 953, 365]}, [1, 1, 2, 2]],
        "extra_creatures": ["No other creatures.", "papi", "another kumo", "a small bird next to kumo", {"description": "a goldfish"}, "nezu"],
    }
    g = grounding_from_json(data, (1000, 1000), NAMES)
    assert g.text_found == ["12 (on the clock face)", "好我", "writing (on nezu's clipboard)"]
    assert g.bubbles == [(60, 120, 530, 423), (505, 45, 953, 365)]
    # "papi" only repeats a located character; "nezu" is a cast member drawn without a box, so it counts
    assert g.extras == [CLONE.format(name="kumo"), "a small bird next to kumo", "a goldfish", "nezu"]


@pytest.mark.parametrize("data", [{}, {"characters": "none", "boards": "n/a", "text_or_symbols": None, "extra_creatures": "none"},
                                  ["not", "an", "object"], None])
def test_empty_or_odd_replies_give_an_empty_grounding(data):
    assert grounding_from_json(data, SIZE, NAMES) == Grounding(size=SIZE)


# ----------------------------------------------------------------------------- locate (one vision call)

GOOD_REPLY = {"characters": [{"name": "papi (brown dog)", "head": [100, 500, 400, 700], "matches_reference": True},
                             {"name": "kumo", "head": [600, 550, 900, 750], "matches_reference": True}],
              "boards": [], "speech_bubbles": [], "text_or_symbols": [], "extra_creatures": []}


def sheets():
    return {n: png(2, 2, c) for n, c in zip(NAMES, [(200, 120, 60), (120, 180, 230), (90, 60, 40)])}


async def test_locate_sends_the_present_sheets_first_and_asks_for_strict_json():
    vision, sh, image = VisionFake([GOOD_REPLY]), sheets(), png(200, 300)
    usage = {"prompt_tokens": 10, "completion_tokens": 1}
    g = await locate(vision, image, cast(), ["kumo", "papi"], sh, usage=usage)

    ((messages, config),) = vision.calls
    assert messages[0].images == [sh["papi"], sh["kumo"], image]       # cast order, the panel last
    assert config.response_format == {"type": "json_object"} and config.temperature == 0.0 and config.thinking is False
    prompt = messages[0].content
    assert "Image 1 is the reference sheet of papi. Image 2 is the reference sheet of kumo." in prompt
    assert "The LAST image is one panel" in prompt and "normalized to 0-1000" in prompt
    assert "should NOT be in it" in prompt and "nezu (a brown hedgehog with a clipboard)" in prompt
    assert g.size == (200, 300) and g.heads == {"papi": (20, 150, 80, 210), "kumo": (120, 165, 180, 225)}
    assert usage == {"prompt_tokens": 1510, "completion_tokens": 121}


async def test_locate_without_sheets_describes_the_characters_instead():
    vision = VisionFake([GOOD_REPLY])
    await locate(vision, png(100, 150), cast(), ["papi"], {})
    (messages, _), = vision.calls
    assert len(messages[0].images) == 1 and "reference sheet of" not in messages[0].content
    assert "papi (a plump brown dog professor)" in messages[0].content and "reference sheet or description" in messages[0].content


async def test_locate_asks_again_after_bad_json_and_counts_every_try():
    # prose, a cut-off object, then fenced JSON on the third and last try
    vision = VisionFake(["I see a dog and a cloud.", '{"characters": [{"name": "papi", "head": [1, 2', "```json\n" + json.dumps(GOOD_REPLY) + "\n```"])
    usage = {}
    g = await locate(vision, png(100, 100), cast(), ["papi", "kumo"], sheets(), usage=usage)
    assert len(vision.calls) == 3 and set(g.heads) == {"papi", "kumo"}
    assert usage == {"prompt_tokens": 4500, "completion_tokens": 360}


async def test_a_reply_without_grounding_keys_is_asked_again():
    vision = VisionFake([{"verdict": "fine"}, GOOD_REPLY])
    g = await locate(vision, png(100, 100), cast(), ["papi", "kumo"], sheets())
    assert len(vision.calls) == 2 and "papi" in g.heads


async def test_locate_gives_up_after_three_unusable_replies():
    vision = VisionFake(["nope", "{broken", "[]", "never asked"])
    with pytest.raises(GroundingError):
        await locate(vision, png(100, 100), cast(), ["papi"], sheets())
    assert len(vision.calls) == MAX_TRIES == 3


async def test_engine_errors_propagate_without_another_call():
    vision = VisionFake([RuntimeError("ModelArk unreachable"), GOOD_REPLY])
    with pytest.raises(RuntimeError, match="unreachable"):
        await locate(vision, png(100, 100), cast(), ["papi"], sheets())
    assert len(vision.calls) == 1


# ----------------------------------------------------------------------------- pixel checks

def heads_below(w=400, h=600):
    return {"papi": (40, 360, 160, 470), "kumo": (240, 380, 360, 480)}


def test_a_scene_with_a_calm_top_and_every_character_has_no_problems():
    art = scene_art()
    g = Grounding(size=art.size, heads=heads_below(), matches={"papi": True, "kumo": True})
    assert art_problems(g, panel(), art) == []
    band = calm_band(art, list(g.heads.values()))
    assert band[1] < 20 and (band[3] - band[1]) / 600 > 0.25 and (band[2] - band[0]) / 400 > 0.9


@pytest.mark.parametrize("calm, heads", [(0.08, heads_below()),                                        # top is busy
                                         (0.30, {"papi": (40, 20, 360, 160), "kumo": (240, 380, 360, 480)})])  # a head in the band
def test_a_busy_top_is_reported_with_its_size(calm, heads):
    art = scene_art(calm=calm)
    problems = art_problems(Grounding(size=art.size, heads=heads), panel(), art)
    assert len(problems) == 1 and problems[0].startswith("busy top: the calm area at the top is only ")
    assert "keep the top 28% plain" in problems[0]


def test_a_narrow_calm_column_is_not_a_band():
    a = noise(600, 400)
    a[:300, :150] = (246, 241, 230)          # tall, but only 37% of the width
    art = Image.fromarray(a, "RGB")
    assert calm_band(art) is None
    assert any(p.startswith("busy top:") for p in art_problems(Grounding(size=art.size, heads=heads_below()), panel(), art))


def test_character_problems():
    art = scene_art()
    g = Grounding(size=art.size, heads={"kumo": (240, 380, 360, 480), "nezu": (150, 400, 230, 480)}, matches={"kumo": False},
                  extras=[CLONE.format(name="kumo"), "a grey cat", "papi"])
    problems = art_problems(g, panel(["papi", "kumo"]), art)
    assert problems == [
        "missing character: papi must be in the picture, with the whole head visible",
        "wrong look: kumo must look exactly like its reference (same species, colors and face)",
        "not in this panel: remove nezu; only papi and kumo appear",
        "clone: draw kumo only once",
        "extra figure: remove a grey cat; nobody but papi and kumo appears",   # "papi" (missing) is not reported twice
    ]


def test_drawn_text_names_the_place_never_the_glyphs():
    art = scene_art()
    g = Grounding(size=art.size, heads=heads_below(), text_found=["12 (on the clock face)", "好我", "O(n)", "writing (on nezu's clipboard)"],
                  bubbles=[(10, 10, 100, 60)])
    problems = art_problems(g, panel(), art)
    assert problems == ["drawn text: remove every letter, digit and symbol (on the clock face, nezu's clipboard); every surface stays blank",
                        "drawn bubbles: remove the speech bubbles, chat windows and text boxes; the picture has no bubbles or captions"]
    assert "好我" not in " ".join(problems) and "12" not in " ".join(problems)


def test_a_board_panel_with_a_big_clean_board_passes():
    art = board_art()
    g = Grounding(size=art.size, heads={"papi": (40, 480, 160, 580)}, board=SURFACE)
    assert art_problems(g, panel(["papi"], kind="board"), art) == []
    x1, y1, x2, y2 = clean_rect(art, SURFACE)
    assert (x2 - x1) / 400 > 0.7 and (y2 - y1) / 600 > 0.4 and 36 <= x1 and x2 <= 364 and 156 <= y1 and y2 <= 444


def test_a_cluttered_board_is_too_small():
    art = board_art(clutter=True)
    problems = art_problems(Grounding(size=art.size, heads={"papi": (40, 480, 160, 580)}, board=SURFACE), panel(["papi"], kind="board"), art)
    assert len(problems) == 1 and problems[0].startswith("small board: the blank part of the whiteboard is only ")
    assert "(needs 55% x 25%)" in problems[0]


def test_a_head_in_front_of_the_board_splits_its_clean_part():
    art = board_art()
    g = Grounding(size=art.size, heads={"papi": (150, 200, 250, 600)}, board=SURFACE)
    assert [p[:12] for p in art_problems(g, panel(["papi"], kind="board"), art)] == ["small board:"]


def test_a_board_panel_without_a_board():
    art = scene_art()
    problems = art_problems(Grounding(size=art.size, heads=heads_below()), panel(kind="board"), art)
    assert problems == ["no board: draw a LARGE, COMPLETELY BLANK whiteboard across almost the full width, with nothing in front of it"]


def test_scene_panels_ignore_boards_and_board_panels_ignore_the_band():
    art = board_art(board=(30, 316, 370, 576))      # the board in the lower half, a busy top half
    g = Grounding(size=art.size, heads={"papi": (40, 584, 160, 600)}, board=[(36, 322), (364, 322), (364, 570), (36, 570)])
    assert art_problems(g, panel(["papi"], kind="board"), art) == []
    assert [p[:9] for p in art_problems(g, panel(["papi"], kind="scene"), art)] == ["busy top:"]


def test_a_grounding_made_on_a_larger_copy_is_scaled_to_the_art():
    art = board_art()
    g = Grounding(size=(800, 1200), heads={"papi": (80, 960, 320, 1160)}, board=[(x * 2, y * 2) for x, y in SURFACE])
    assert art_problems(g, panel(["papi"], kind="board"), art) == []


# ----------------------------------------------------------------------------- ranking attempts

MISSING = ["missing character: kumo must be in the picture, with the whole head visible"]
EXTRA = ["extra figure: remove a grey cat; nobody but papi appears"]
TEXT = ["drawn text: remove every letter, digit and symbol; every surface stays blank"]
SMALL = ["small board: the blank part of the whiteboard is only 50% x 20% of the panel (needs 55% x 25%); make it bigger"]
BUSY = ["busy top: the calm area at the top is only 10% of the panel height (needs 14%); keep the top 28% plain"]


def test_rank_key_higher_is_better_characters_extras_text_board_band():
    """The contract's (no missing chars, no extras, no text, board ok, band ok): the best attempt has the highest key."""
    attempts = [MISSING + BUSY, EXTRA, TEXT + BUSY, SMALL, BUSY, [], MISSING]
    assert sorted(attempts, key=rank_key, reverse=True) == [[], BUSY, SMALL, TEXT + BUSY, EXTRA, MISSING, MISSING + BUSY]
    assert max(attempts, key=rank_key) == []


def test_rank_key_picks_what_the_job_loop_should_publish():
    """jobs._render_panel_v3 keeps an attempt when ``rank > best[0]`` (higher is better, like _attempt_rank): a clean
    retry must replace a flawed first attempt, and a worse retry must not."""
    def publish(attempts):
        best = None
        for problems in attempts:
            rank = rank_key(problems)
            if best is None or rank > best[0]:
                best = (rank, problems)
            if not problems:
                break
        return best[1]

    assert publish([BUSY, []]) == []
    assert publish([BUSY, TEXT, MISSING]) == BUSY
    assert publish([MISSING, EXTRA, SMALL]) == SMALL
    # jobs notes an attempt whose art could not be located as "unchecked: ...": it never replaces a checked attempt
    unchecked = ["unchecked: the art could not be checked (GroundingError)"]
    assert publish([MISSING + EXTRA + TEXT, unchecked]) == MISSING + EXTRA + TEXT
    assert publish([unchecked, BUSY, unchecked]) == BUSY and publish([unchecked, []]) == []


def test_rank_key_breaks_ties_by_how_much_room_is_missing():
    near = ["small board: the blank part of the whiteboard is only 50% x 24% of the panel (needs 55% x 25%); x"]
    far = ["small board: the blank part of the whiteboard is only 30% x 10% of the panel (needs 55% x 25%); x"]
    none = ["no board: draw a LARGE, COMPLETELY BLANK whiteboard"]
    assert sorted([none, far, near], key=rank_key, reverse=True) == [near, far, none]
    tight = ["busy top: the calm area at the top is only 12% of the panel height (needs 14%); x"]
    busy = ["busy top: the calm area at the top is only 0% of the panel height (needs 14%); x"]
    assert rank_key(tight) > rank_key(busy)


def test_rank_key_on_real_problems_and_after_a_json_round_trip():
    art = scene_art(calm=0.05)
    problems = art_problems(Grounding(size=art.size, heads={"papi": (40, 360, 160, 470)}, text_found=["A (on a sign)"]), panel(), art)
    key = rank_key(problems)
    assert key[:8] == (0, -1, 0, -1, 0, 0, 0, -1) and key[-1] == -3
    assert rank_key(json.loads(json.dumps(problems))) == key
    assert rank_key([]) == (0,) * 10 and rank_key(None) == rank_key([])
    assert rank_key(["something else entirely"])[:9] == (0,) * 9   # unknown notes only count in the total


# ----------------------------------------------------------------------------- reviewer fixes: odd reply shapes

def test_a_head_given_as_corner_points_is_located():
    """A missing head costs a re-render: four corners (or two) are read as their bounding box."""
    data = {"characters": [{"name": "papi", "head": [[100, 500], [300, 500], [300, 700], [100, 700]]},
                           {"name": "kumo", "head": [{"x": 600, "y": 500}, {"x": 800, "y": 700}]}]}
    g = grounding_from_json(data, (1000, 1000), NAMES)
    assert g.heads == {"papi": (100, 500, 300, 700), "kumo": (600, 500, 800, 700)}


@pytest.mark.parametrize("board", [
    {"x1": 40, "y1": 296, "x2": 957, "y2": 771},
    {"top_left": [40, 296], "top_right": [957, 296], "bottom_right": [957, 771], "bottom_left": [40, 771]},
    {"corners": {"tl": {"x": 40, "y": 296}, "tr": {"x": 957, "y": 296}, "br": {"x": 957, "y": 771}, "bl": {"x": 40, "y": 771}}},
])
def test_more_board_shapes(board):
    assert grounding_from_json({"boards": [board]}, SIZE, NAMES).board == PX_QUAD


def test_a_box_with_width_and_height_is_never_misread_as_corners():
    # {"x", "y", "width", "height"}: reading the four numbers as x1, y1, x2, y2 would invent a wrong board
    assert grounding_from_json({"boards": [{"x": 40, "y": 296, "width": 917, "height": 475}]}, SIZE, NAMES).board is None


@pytest.mark.parametrize("entry", ["no", "No.", "nobody", "No one", "none.", "[]", "{}", "empty", "Nothing else", "not applicable",
                                   "None visible", "No creatures other than the named characters", "There are no other figures here."])
def test_entries_that_report_nothing_cost_no_rerender(entry):
    assert _is_nothing(entry)
    g = grounding_from_json({"characters": [{"name": "papi", "head": [100, 500, 300, 700]}], "extra_creatures": [entry],
                             "text_or_symbols": [entry]}, (1000, 1000), NAMES)
    assert (g.extras, g.text_found) == ([], [])


@pytest.mark.parametrize("entry", ["a small bird", "no smoking sign", "NO (on the door sign)", "nobody's hat", "a grey cat, no tail"])
def test_real_findings_are_not_mistaken_for_nothing(entry):
    assert not _is_nothing(entry)


def test_drawn_text_keyed_by_place():
    g = grounding_from_json({"text_or_symbols": {"the clock face": ["12", "3", "6"], "nezu's clipboard": "好我"}}, SIZE, NAMES)
    assert g.text_found == ["12, 3, 6 (on the clock face)", "好我 (on nezu's clipboard)"]


def test_the_glyphs_never_reach_the_feedback_through_the_place_or_an_extra():
    """The model sometimes names the place by its writing ("the sign reading BLEU"): the feedback goes into the next art
    prompt, so the words, quotes, digits and CJK glyphs it reported are left out of it."""
    data = {"characters": [{"name": "papi", "head": [100, 600, 300, 750]}, {"name": "kumo", "head": [600, 600, 800, 750]}],
            "text_or_symbols": [{"text": "BLEU 28.4", "where": "on the poster reading BLEU 28.4"},
                                {"text": "好我", "where": "nezu's clipboard with 好我 scribbled"},
                                {"text": "A", "where": "a sign"}, {"text": "x", "where": 'the mug saying "hi"'}],
            "extra_creatures": ["a cat in a shirt saying “MEOW 2”", "3 small birds"]}
    art = scene_art()
    problems = art_problems(grounding_from_json(data, art.size, NAMES), panel(), art)
    joined = " ".join(problems)
    for glyphs in ("BLEU", "28", "好", "我", "MEOW", "hi", "3", "2"):
        assert not re.search(rf"(?<!\w){glyphs}(?!\w)", joined), glyphs
    assert '"' not in joined and "“" not in joined
    assert ("drawn text: remove every letter, digit and symbol (on the poster reading, nezu's clipboard with scribbled, "
            "a sign, the mug saying); every surface stays blank") in problems
    assert "extra figure: remove a cat in a shirt saying; small birds; nobody but papi and kumo appears" in problems


async def test_locate_takes_the_size_from_the_caller():
    """jobs decoded the art already: with ``size`` nothing is decoded, so even a URL image works."""
    vision = VisionFake([GOOD_REPLY])
    image = ImageContent.from_url("https://example.invalid/art.jpg", "image/jpeg")
    g = await locate(vision, image, cast(), ["papi", "kumo"], sheets(), size=(200, 300))
    assert g.size == (200, 300) and g.heads["papi"] == (20, 150, 80, 210)
    with pytest.raises(ValueError, match="inline image"):
        await locate(VisionFake([GOOD_REPLY]), image, cast(), ["papi"], sheets())


async def test_the_prompt_asks_for_whole_image_coordinates_and_one_text_entry_per_surface():
    vision = VisionFake([GOOD_REPLY])
    await locate(vision, png(100, 150), cast(), ["papi"], sheets())
    (messages, config), = vision.calls
    assert "normalized to 0-1000 over that whole image" in messages[0].content
    assert "one entry per surface that carries it" in messages[0].content and config.max_tokens >= 4000


# ----------------------------------------------------------------------------- reviewer fixes: problems

def test_a_board_panel_without_an_exhibit_is_judged_as_a_scene():
    """lettering letters a "board" panel without a board spec as a scene, so its art needs the calm top band, not a board."""
    art = scene_art()
    g = Grounding(size=art.size, heads=heads_below())
    assert not is_board_panel(panel(kind="board", board={})) and is_board_panel(panel(kind="board"))
    assert art_problems(g, panel(kind="board", board={}), art) == []
    assert art_problems(g, panel(kind="board"), art)[0].startswith("no board:")


def test_a_grounding_without_a_size_is_used_as_is():
    """lettering.Grounding() defaults to size (0, 0): the boxes are taken in the art's pixels instead of being blown up."""
    art = scene_art()
    assert art_problems(Grounding(size=(0, 0), heads=heads_below()), panel(), art) == []


def test_a_clone_of_someone_who_should_not_be_there_is_only_told_to_go():
    art = scene_art()
    g = Grounding(size=art.size, heads={**heads_below(), "nezu": (180, 400, 230, 470)}, extras=[CLONE.format(name="nezu")])
    assert art_problems(g, panel(), art) == ["not in this panel: remove nezu; only papi and kumo appear"]


# ----------------------------------------------------------------------------- the interfaces lettering and jobs rely on

def test_the_board_check_measures_what_lettering_draws_into():
    """clean_rect is lettering's own measurement: a board passes here exactly when lettering gets that rectangle."""
    for art, quad, exclude in [(board_art(), SURFACE, []), (board_art(clutter=True), SURFACE, []),
                               (board_art(), SURFACE, [(150, 200, 250, 600)])]:
        ours, theirs = clean_rect(art, quad, exclude), lettering.clean_rect(art, quad, exclude=exclude)
        assert ours == (tuple(int(round(v)) for v in theirs) if theirs else None)


def test_art_check_grounding_is_what_lettering_reads():
    """Same fields as lettering.Grounding, and letter_panel takes an art_check Grounding: tails aim at the located heads."""
    assert [f.name for f in dataclasses.fields(Grounding)] == [f.name for f in dataclasses.fields(lettering.Grounding)]
    art = scene_art(400, 600)
    g = grounding_from_json({"characters": [{"name": "papi (brown dog)", "head": [100, 600, 400, 780]},
                                            {"name": "kumo", "head": [600, 620, 900, 800]}]}, art.size, NAMES)
    p = panel()
    p.dialogue = [Bubble("papi", "Look at this."), Bubble("kumo", "Wow.")]
    img, layout = lettering.letter_panel(art, p, g, "en-US")
    assert img.size[0] == art.size[0] and [b["speaker"] for b in layout["bubbles"]] == ["papi", "kumo"]
    for b in layout["bubbles"]:
        if b["tail"]:   # the tip points down toward its speaker's head (the bubbles sit in the calm top band)
            x1, y1, x2, y2 = g.heads[b["speaker"]]
            assert b["tail"][1][1] > b["rect"][1] and abs(b["tail"][1][0] - (x1 + x2) / 2) < abs(b["tail"][1][0] - art.size[0] * 2)


def test_plain_marks_are_not_drawn_text():
    g = grounding_from_json({"text_or_symbols": [{"text": "measurement markings", "where": "glass beaker"},
                                                 {"text": "tick marks", "where": "the ruler"}, "scale lines",
                                                 {"text": "12", "where": "the ruler"}, {"text": 'markings "ABC"', "where": "a box"}]},
                           SIZE, NAMES)
    assert g.text_found == ["12 (on the ruler)", 'markings "ABC" (on a box)']   # digits and quoted letters still count


def test_a_narrower_but_tall_blank_board_is_big_enough():
    from services import art_check as ac
    panel = PanelV2(n=1, chapter="evidence", title="t", characters=["papi"], scene="s", visual="v", kind="board",
                    board={"kind": "list", "title": "T", "items": ["a", "b"]})
    art = Image.new("RGB", (1000, 1500), (250, 250, 250))
    def problems_for(x2, y2):   # a framed blank board from (40, 300) to (x2, y2)
        img = art.copy()
        ImageDraw.Draw(img).rectangle((40, 300, x2, y2), outline=(40, 40, 40), width=8)
        g = Grounding(size=img.size, heads={"papi": (700, 1200, 900, 1400)}, board=[(40, 300), (x2, 300), (x2, y2), (40, y2)])
        return [p for p in ac.art_problems(g, panel, img) if p.startswith(("small board", "no board"))]
    assert problems_for(600, 1000) == []          # ~52% x 44%: narrower than 55% but tall: room for the chart
    assert problems_for(480, 1000)                # ~40% wide: too narrow whatever its height
    assert problems_for(600, 560)                 # ~52% x 15%: too short


# ----------------------------------------------------------------------------- does the picture show the idea?

SHOWS = "the small padlocked card comes first and a bold arrow leads from it to a tall stack of pages"


async def test_the_idea_check_turns_a_verdict_into_feedback_and_object_boxes():
    from services.art_check import check_idea
    image = png(400, 600)
    yes = VisionFake([{"shown": "yes", "missing": "", "objects": [[100, 500, 600, 900]]}])
    assert await check_idea(yes, image, SHOWS, (400, 600)) == (None, [(40, 300, 240, 540)])
    assert SHOWS in yes.calls[0][0][0].content and yes.calls[0][1].response_format == {"type": "json_object"}
    partly = VisionFake([{"shown": "partly", "missing": "the arrow is tiny", "objects": []}])
    problem, boxes = await check_idea(partly, image, SHOWS, (400, 600))
    assert problem.startswith("idea unclear: the picture must show the small padlocked card") and "the arrow is tiny" in problem
    no = VisionFake(["not json", {"shown": "no", "missing": "no card at all"}])   # a reply that is not JSON is asked again
    problem, boxes = await check_idea(no, image, SHOWS, (400, 600))
    assert problem.startswith("idea not shown:") and "missing: no card at all" in problem and boxes == [] and len(no.calls) == 2
    with pytest.raises(GroundingError):
        await check_idea(VisionFake(["?", "{}"]), image, SHOWS, (400, 600))


def test_an_attempt_that_misses_its_idea_ranks_below_one_that_shows_it():
    missed = ["idea not shown: the picture must show the card"]
    unclear = ["idea unclear: the picture must show the card"]
    busy = ["busy top: the calm area at the top is only 10% of the panel height (needs 14%); x"]
    assert sorted([missed, busy, unclear, []], key=rank_key, reverse=True) == [[], busy, unclear, missed]
    assert rank_key(["drawn text: remove every letter"]) < rank_key(missed)   # drawn text is still worse
    unjudged = ["idea unclear: (not checked)"]   # how jobs ranks an attempt whose idea check failed
    assert rank_key(missed) < rank_key(unjudged) < rank_key([])


def test_a_teaching_arrow_is_not_drawn_text():
    # the pictures now ask for "a bold arrow from cause to effect": the grounding must not report it as writing
    from services.art_check import _texts
    assert _texts([{"text": "bold arrow", "where": "between the card and the stack"}, {"text": "→ →", "where": "the desk"},
                   "a big red arrow"]) == []
    assert _texts([{"text": "EXIT arrow", "where": "the wall"}, {"text": "arrow 3", "where": "the floor"}]) == [
        "EXIT arrow (on the wall)", "arrow 3 (on the floor)"]
    # a symbol's simple shape is an object too ("a slip marked with a red cross", "a green check mark sticker")
    assert _texts([{"text": "red cross", "where": "the paper slip"}, {"text": "green check mark", "where": "each sticker"},
                   {"text": "✓", "where": "the sticker"}, "stars and dots"]) == []
    assert _texts([{"text": "OK", "where": "the sticker"}]) == ["OK (on the sticker)"]


async def test_a_diagnosis_that_asks_for_writing_is_never_repeated_to_the_image_model():
    from services.art_check import check_idea
    vision = VisionFake([{"shown": "partly", "missing": "no visible scores or labels showing which answer is best"}])
    problem, _ = await check_idea(vision, png(400, 600), SHOWS, (400, 600))
    assert problem.startswith("idea unclear:") and "scores" not in problem and "labels" not in problem
    assert "never text, labels, numbers or scores" in vision.calls[0][0][0].content


@pytest.mark.parametrize("words", ["red X cross", "red cross + lines", "five green check marks", "red cross symbol", "right arrow",
                                   "arrow pointing to the stack", "upward arrow", "check and cross marks", "→ arrow", "❌", "✅", "➡️", "✔️"])
def test_the_wordings_the_locate_model_uses_for_teaching_marks_are_not_text(words):
    # all seen in econ_delivery's reports (2026-10-28): each pushed a panel to its attempt maximum and erased the mark
    from services.art_check import _texts
    assert _texts([{"text": words, "where": "the paper slip"}]) == []


@pytest.mark.parametrize("words", ["EXIT arrow", "arrow 3", "big red X", "plus sign", "arrow labeled A", "5 check marks",
                                   "arrow pointing to EXIT", "√", "×"])
def test_marks_that_carry_writing_still_count(words):
    from services.art_check import _texts
    assert _texts([{"text": words, "where": "the desk"}]) == [f"{words} (on the desk)"]


def test_any_mark_on_a_board_is_drawn_text():
    # a board must stay blank for the exhibit the code draws: an arrow or a tick there is redrawn, as before the shapes rule
    from services.art_check import LOCATE_PROMPT, _texts
    assert _texts([{"text": "arrows", "where": "the whiteboard"}, {"text": "green check marks", "where": "the board's edge"}]) == [
        "arrows (on the whiteboard)", "green check marks (on the board's edge)"]
    assert _texts([{"text": "arrows", "where": "the cardboard box"}]) == []
    assert "any mark drawn on a whiteboard or board is" in LOCATE_PROMPT


async def test_the_idea_check_reads_a_json_boolean_and_partial_wordings():
    from services.art_check import check_idea
    image = png(400, 600)
    assert (await check_idea(VisionFake([{"shown": True, "objects": []}]), image, SHOWS, (400, 600)))[0] is None
    problem, _ = await check_idea(VisionFake([{"shown": "partially", "objects": []}]), image, SHOWS, (400, 600))
    assert problem.startswith("idea unclear:")
    problem, _ = await check_idea(VisionFake([{"shown": False, "objects": []}]), image, SHOWS, (400, 600))
    assert problem.startswith("idea not shown:")


def test_a_dotted_arrow_is_a_shape():
    from services.art_check import _just_marks
    assert _just_marks("dotted arrow") and _just_marks("two dots")


def test_two_frames_are_stacked_into_the_panel_with_one_grounding():
    # a teaching picture that missed its idea is redrawn as two frames: lettering reads one grounding in the panel's pixels
    from services.art_check import compose_frames
    top, bottom = Image.new("RGB", (1120, 832), (200, 0, 0)), Image.new("RGB", (1120, 832), (0, 0, 200))
    first = Grounding(size=(1120, 832), heads={"papi": (112, 416, 336, 624)}, objects=[(560, 416, 1008, 832)], matches={"papi": True})
    second = Grounding(size=(1120, 832), heads={"papi": (784, 416, 1008, 624), "kumo": (112, 416, 336, 624)},
                       objects=[(0, 0, 1120, 416)], matches={"papi": False}, text_found=["A (on the card)"])
    art, g = compose_frames([top, bottom], [first, second], SIZE)
    assert art.size == SIZE and art.getpixel((10, 10)) == (200, 0, 0) and art.getpixel((10, 1530)) == (0, 0, 200)
    # the top frame keeps papi's bubble tail; papi's head in the bottom frame is a head no speaker owns, kept clear like
    # every head (as an object, the lettering dropped it before moving a line to the strip)
    assert g.size == SIZE and g.heads == {"papi": (102, 384, 307, 576), "kumo": (102, 1152, 307, 1344), "#2 papi": (717, 1152, 922, 1344)}
    assert g.objects == [(512, 384, 922, 768), (0, 768, 1024, 1152)]
    assert g.matches == {"papi": False} and g.text_found == ["A (on the card)"] and g.board is None
    art, g = compose_frames([top, bottom], [first, None], SIZE)   # a frame whose art could not be located adds nothing
    assert set(g.heads) == {"papi"} and g.objects == [(512, 384, 922, 768)]
    # most important first, as the lettering drops the last first: each frame's objects in turn, so the bottom frame's
    # key object is not the first to go under a bubble; a box over half its own frame is the frame, not an object
    first.objects = [(0, 0, 1120, 832), (560, 416, 1008, 832), (0, 0, 112, 104)]
    art, g = compose_frames([top, bottom], [first, second], SIZE)
    assert g.objects == [(512, 384, 922, 768), (0, 768, 1024, 1152), (0, 0, 102, 96)]
