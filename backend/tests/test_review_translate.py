"""Review findings S3 (translation acceptance) and S8 (unexpected model-reply shapes) in services.storyboarder_v2.

Plain tests lock in the fixes applied on 2026-10-22. Tests marked ``xfail(strict=True)`` describe the correct
behaviour for scenarios the current code still gets wrong (see each marker's reason); when the source is fixed
they XPASS, strict mode turns that into a failure, and the marker should then be deleted.
"""

import json

import pytest

from services.paper_extractor import PaperText
from services.storyboarder_v2 import (
    Chapter,
    PanelV2,
    StoryboarderV2,
    StoryboardV2,
    _has_script,
    extract_json,
)

from .fakes import FakeEngine, PaperModel, make_cast, make_panel


def storyboard(panels, chapters=None) -> StoryboardV2:
    return StoryboardV2(title="T", summary="S", theme="kumomo", language="en-US", chapters=list(chapters or []), panels=panels)


def payload_of(prompt: str) -> dict:
    """The JSON payload at the end of ``translate_prompt``."""
    return json.loads(prompt.rsplit("\n\n", 1)[1])


def two_speaker_panel(n=1, **kwargs):
    return make_panel(n, dialogue=[("papi", "Hello there, my friend."), ("kumo", "Why is it so slow?")], **kwargs)


GOOD_TWO = [{"speaker": "papi", "text": "你好，朋友。"}, {"speaker": "kumo", "text": "为什么这么慢？"}]


# ============================================================================= S3 — translation acceptance (fixed parts)

@pytest.mark.parametrize("bad_dialogue", [
    pytest.param([{"speaker": "kumo", "text": "为什么这么慢？"}, {"speaker": "papi", "text": "你好，朋友。"}], id="swapped-speakers"),
    pytest.param([{"speaker": "papi", "text": "你好，朋友。为什么这么慢？"}], id="merged-into-one"),
    pytest.param(GOOD_TWO + [{"speaker": "kumo", "text": "再说一遍？"}], id="extra-bubble"),
    pytest.param([GOOD_TWO[0], {"speaker": "kumo", "text": ""}], id="empty-text"),
    pytest.param([GOOD_TWO[0], {"speaker": "kumo"}], id="missing-text"),
    pytest.param([GOOD_TWO[0], {"speaker": "kumo", "text": "Why is it so slow?"}], id="left-in-english"),
    pytest.param([GOOD_TWO[0], {"speaker": "nezu", "text": "为什么这么慢？"}], id="wrong-speaker"),
    pytest.param([GOOD_TWO[0], "为什么这么慢？"], id="bubble-as-string"),
])
async def test_s3_bad_panel_translation_is_rejected_then_re_requested(bad_dialogue):
    sb = storyboard([two_speaker_panel(1, title="Hi", narration="Fact one."), make_panel(2, dialogue=[("nezu", "Prove it.")])])
    engine = FakeEngine([
        {"title": "标题", "panels": [
            {"n": 1, "title": "第一格", "dialogue": bad_dialogue, "narration": "事实一。"},
            {"n": 2, "dialogue": [{"speaker": "nezu", "text": "证明一下。"}]},
        ]},
        {"title": "另一个标题", "panels": [{"n": 1, "title": "第一格", "dialogue": GOOD_TWO, "narration": "事实一。"}]},
    ])

    await StoryboarderV2(engine).translate(sb, "zh-CN")

    p1, p2 = sb.panels
    assert [(b.speaker, b.text) for b in p1.dialogue] == [("papi", "你好，朋友。"), ("kumo", "为什么这么慢？")]
    assert (p1.title, p1.narration) == ("第一格", "事实一。")
    assert p2.dialogue[0].text == "证明一下。"
    assert len(engine.calls) == 2
    assert [p["n"] for p in payload_of(engine.prompts[1])["panels"]] == [1]  # only the rejected panel is re-requested
    assert sb.title == "标题"  # the retry never overwrites title/summary
    assert sb.language == "zh-CN"


async def test_s3_panel_rejected_twice_stays_entirely_english_with_a_warning(capsys):
    panel = two_speaker_panel(1, title="Hi", narration="Fact one.", labels=["Query"])
    merged = {"panels": [{"n": 1, "title": "第一格", "dialogue": [{"speaker": "papi", "text": "你好。为什么？"}], "narration": "事实一。",
                          "labels": ["查询"]}]}
    engine = FakeEngine([merged, merged])

    await StoryboarderV2(engine).translate(storyboard([panel]), "zh-CN")

    # no half-translated panel: the bubbles must never be re-assigned, so title/narration/labels stay English too
    assert [(b.speaker, b.text) for b in panel.dialogue] == [("papi", "Hello there, my friend."), ("kumo", "Why is it so slow?")]
    assert (panel.title, panel.narration, panel.labels) == ("Hi", "Fact one.", ["Query"])
    assert len(engine.calls) == 2  # exactly one re-request, no loop
    assert "kept their English text" in capsys.readouterr().out


@pytest.mark.parametrize("speaker", ["Papi", " papi ", "papi (smiling)", "Professor Papi", None, ""])
async def test_s3_speaker_spelling_variants_and_missing_speakers_are_accepted(speaker):
    bubble = {"text": "你好，朋友。"} if speaker is None else {"speaker": speaker, "text": "你好，朋友。"}
    engine = FakeEngine([{"panels": [{"n": 1, "dialogue": [bubble, GOOD_TWO[1]]}]}])
    sb = storyboard([two_speaker_panel(1)])
    await StoryboarderV2(engine).translate(sb, "zh-CN")
    assert [(b.speaker, b.text) for b in sb.panels[0].dialogue] == [("papi", "你好，朋友。"), ("kumo", "为什么这么慢？")]
    assert len(engine.calls) == 1


async def test_s3_wrapped_reply_and_string_label_are_applied():
    reply = ('Sure:\n{"translation": {"title": "注意力", "summary": "摘要", "panels": '
             '[{"n": 1, "title": "第一格", "dialogue": [{"speaker": "papi", "text": "你好。"}], "labels": "注意力"}]}}')
    engine = FakeEngine([reply])
    sb = storyboard([make_panel(1, dialogue=[("papi", "Hello there.")], labels=["Attention"])])

    await StoryboarderV2(engine).translate(sb, "zh-CN")

    p = sb.panels[0]
    assert (sb.title, sb.summary, p.title, p.dialogue[0].text) == ("注意力", "摘要", "第一格", "你好。")
    assert p.labels == ["注意力"]  # a string label is one label, never split into characters
    assert len(engine.calls) == 1


async def test_s3_string_label_is_not_split_into_characters_to_match_the_label_count():
    # iterating "查询" would give two one-character labels and match the two English labels
    engine = FakeEngine([{"panels": [{"n": 1, "dialogue": [{"speaker": "papi", "text": "你好。"}], "labels": "查询"}]}])
    sb = storyboard([make_panel(1, dialogue=[("papi", "Hello there.")], labels=["Query", "Key"])])
    await StoryboarderV2(engine).translate(sb, "zh-CN")
    assert sb.panels[0].labels == ["Query", "Key"]


@pytest.mark.parametrize("text, language, expected", [
    ("你好，朋友。", "zh-CN", True),
    ("こんにちは！", "ja-JP", True),
    ("Transformer很快。", "zh-CN", True),   # mixed is fine as long as the target script is there
    ("OK！", "zh-CN", True),                # short tokens are not treated as untranslated English
    ("GPU！", "zh-CN", True),
    ("2017", "ja-JP", True),
    ("Hello there.", "zh-CN", False),
    ("Hello there。", "zh-CN", False),       # full-width punctuation alone is not the target script
    ("Why is it so slow?", "ja-JP", False),
    ("Anything at all", "en-US", True),
    ("https://wang-kevin3290.github.io/scaling-crl/", "zh-CN", True),   # nothing to translate in a web address
    ("代码在 github.com/foo/scaling-crl", "zh-CN", True),
    ("Visit www.example.org today", "zh-CN", False),                      # the words around it still are
    ("huggingface.co/datasets/openai/mrcr", "zh-CN", True),
])
def test_s3_has_script(text, language, expected):
    assert _has_script(text, language) is expected


async def test_s3_build_re_requests_a_panel_whose_first_translation_was_left_in_english():
    class Tampered(PaperModel):
        """First translation reply leaves panel 2's bubble untranslated; the retry is a normal translation."""

        def __init__(self):
            super().__init__(num_panels=3)
            self.translations = []

        def __call__(self, prompt, messages, config):
            reply = super().__call__(prompt, messages, config)
            user = messages[-1].content
            if user.startswith("Translate this manga script"):
                self.translations.append([p["n"] for p in payload_of(user)["panels"]])
                if len(self.translations) == 1:
                    for p in reply["panels"]:
                        if p["n"] == 2:
                            p["dialogue"][0]["text"] = "Line for panel 2 please."
            return reply

    model = Tampered()
    sb = await StoryboarderV2(FakeEngine(router=model)).build(PaperText("T", "body", []), make_cast(), "zh-CN", (3, 6))

    assert model.translations == [[1, 2, 3], [2]]
    assert [p.dialogue[0].text for p in sb.panels] == ["第1格的台词", "第2格的台词", "第3格的台词"]
    assert sb.language == "zh-CN"


# ============================================================================= S3 — fixed in round 3

@pytest.mark.parametrize("text", ["BLEU 28.4", "28.4 BLEU！", "LSTM？", "WMT 2014：BLEU 28.4"])
def test_s3_acronym_and_number_lines_count_as_translated(text):
    assert _has_script(text, "zh-CN") is True


async def test_s3_acronym_only_bubble_does_not_keep_the_panel_in_english():
    good = {"panels": [{"n": 1, "title": "新纪录", "dialogue": [{"speaker": "papi", "text": "28.4 BLEU！"}],
                        "narration": "在WMT 2014英德翻译上创下新纪录。"}]}
    engine = FakeEngine([good, good])
    sb = storyboard([make_panel(1, title="New record", dialogue=[("papi", "28.4 BLEU!")], narration="A new record on WMT 2014 English-German.")])

    await StoryboarderV2(engine).translate(sb, "zh-CN")

    p = sb.panels[0]
    assert (p.title, p.dialogue[0].text, p.narration) == ("新纪录", "28.4 BLEU！", "在WMT 2014英德翻译上创下新纪录。")
    assert len(engine.calls) == 1


async def test_s3_untranslated_narration_is_re_requested():
    engine = FakeEngine([
        {"panels": [{"n": 1, "dialogue": [{"speaker": "papi", "text": "你好。"}], "narration": "The base model cost far less to train."}]},
        {"panels": [{"n": 1, "dialogue": [{"speaker": "papi", "text": "你好。"}], "narration": "基础模型的训练成本低得多。"}]},
    ])
    sb = storyboard([make_panel(1, dialogue=[("papi", "Hello there.")], narration="The base model cost far less to train.")])

    await StoryboarderV2(engine).translate(sb, "zh-CN")

    assert sb.panels[0].narration == "基础模型的训练成本低得多。"
    assert len(engine.calls) == 2


async def test_s3_empty_translated_narration_does_not_erase_the_fact():
    engine = FakeEngine([
        {"panels": [{"n": 1, "dialogue": [{"speaker": "papi", "text": "你好。"}], "narration": ""}]},
        {"panels": [{"n": 1, "dialogue": [{"speaker": "papi", "text": "你好。"}], "narration": "WMT 2014英德：28.4 BLEU。"}]},
    ])
    sb = storyboard([make_panel(1, dialogue=[("papi", "Hello there.")], narration="28.4 BLEU on WMT 2014 English-German.")])

    await StoryboarderV2(engine).translate(sb, "zh-CN")

    assert sb.panels[0].narration.strip() != ""


@pytest.mark.parametrize("retry_replies", [
    pytest.param(["sorry, I cannot do that", "still not JSON"], id="unparsable"),
    pytest.param([RuntimeError("ModelArk 400: request rejected")], id="engine-error"),
])
async def test_s3_failed_re_request_keeps_the_first_pass(retry_replies):
    engine = FakeEngine([{"title": "标题", "panels": [{"n": 1, "dialogue": [{"speaker": "papi", "text": "你好。"}]}]}, *retry_replies])
    sb = storyboard([make_panel(1, dialogue=[("papi", "Hello there.")]), make_panel(2, dialogue=[("kumo", "Why so slow?")])])

    await StoryboarderV2(engine).translate(sb, "zh-CN")

    assert [p.dialogue[0].text for p in sb.panels] == ["你好。", "Why so slow?"]
    assert (sb.title, sb.language) == ("标题", "zh-CN")


# ============================================================================= S8 — unexpected reply shapes (fixed parts)

ITEMS = [{"id": "1.d0", "text": "A shorter line."}, {"id": "1.n", "text": "A shorter fact."}]


@pytest.mark.parametrize("reply", [
    pytest.param(json.dumps(ITEMS), id="bare-list"),
    pytest.param("  \n" + json.dumps(ITEMS, indent=1) + "\n", id="bare-list-whitespace"),
    pytest.param("```json\n" + json.dumps(ITEMS) + "\n```", id="fenced-list"),
    pytest.param("Here:\n```json\n" + json.dumps(ITEMS) + "\n```\nHope it helps!", id="fenced-list-with-prose"),
    pytest.param(json.dumps({"result": {"items": ITEMS}}), id="wrapped-one-level"),
    pytest.param('Format example: {"id": "x"}\nAnswer: ' + json.dumps({"items": ITEMS}), id="first-object-with-key-wins"),
    pytest.param(json.dumps(ITEMS)[:-1] + ",]", id="bare-list-trailing-comma"),
])
def test_s8_extract_json_finds_the_keyed_payload(reply):
    assert extract_json(reply, key="items") == {"items": ITEMS}


def test_s8_extract_json_without_a_key_still_returns_the_first_object():
    assert extract_json('{"a": 1} and {"b": 2}') == {"a": 1}
    assert extract_json('{"problems": []}', key="problems") == {"problems": []}
    assert extract_json("[]", key="problems") == {"problems": []}


async def test_s8_repair_round_usage_is_counted():
    engine = FakeEngine(["Here is the brief: attention (sorry, no JSON)", '{"key_idea": "Attention replaces recurrence."}'],
                        usage={"prompt_tokens": 1000, "completion_tokens": 200})
    writer = StoryboarderV2(engine)
    await writer.analyze(PaperText("Attention Is All You Need", "body", []))
    assert len(engine.calls) == 2
    assert writer.usage == {"prompt_tokens": 2000, "completion_tokens": 400}


@pytest.mark.parametrize("reply", [
    pytest.param([{"id": "1.d0", "text": "A shorter line."}], id="bare-list"),
    pytest.param({"items": ["junk", None, 7, {"id": "1.d0"}, {"id": "1.d0", "text": "A shorter line."}]}, id="non-dict-items"),
])
async def test_s8_fit_tolerates_bare_lists_and_junk_items(reply):
    fast = FakeEngine([reply])
    sb = storyboard([make_panel(1, dialogue=[("papi", " ".join(["long"] * 16))])])
    assert await StoryboarderV2(FakeEngine(), fast_engine=fast).fit(sb, "en-US") == 1
    assert sb.panels[0].dialogue[0].text == "A shorter line."


@pytest.mark.parametrize("reply", [
    pytest.param([{"id": "1.n", "issue": "wrong model", "corrected": "The big model reached 28.4 BLEU."}], id="bare-list"),
    pytest.param({"problems": ["junk", None, {"id": None, "corrected": "x"}, {"id": 1, "corrected": "x"}, {"id": "x.n", "corrected": "x"},
                               {"id": "1.dq", "corrected": "x"}, {"id": "1.d7", "corrected": "x"},
                               {"id": "1.n", "issue": "wrong model", "corrected": "The big model reached 28.4 BLEU."}]}, id="malformed-items"),
])
async def test_s8_factcheck_tolerates_bare_lists_and_malformed_items(reply):
    sb = storyboard([make_panel(1, dialogue=[("papi", "Hi.")], narration="The base model reached 28.4 BLEU.")])
    assert await StoryboarderV2(FakeEngine([reply])).factcheck(sb, PaperText("T", "body", [])) == 1
    assert sb.panels[0].narration == "The big model reached 28.4 BLEU."
    assert sb.panels[0].dialogue[0].text == "Hi."


@pytest.mark.parametrize("reply", [
    pytest.param([{"n": 1, "dialogue": [{"speaker": "papi", "text": "你好。"}]}], id="bare-list"),
    pytest.param({"panels": ["junk", None, {"n": None, "dialogue": [{"speaker": "papi", "text": "错的。"}]},
                             {"n": "first", "dialogue": [{"speaker": "papi", "text": "错的。"}]},
                             {"n": "1", "dialogue": [{"speaker": "papi", "text": "你好。"}]}]}, id="junk-and-null-n"),
])
async def test_s8_translate_tolerates_bare_lists_null_n_and_junk_panels(reply):
    engine = FakeEngine([reply])
    sb = storyboard([make_panel(1, dialogue=[("papi", "Hello there.")])])
    await StoryboarderV2(engine).translate(sb, "zh-CN")
    assert sb.panels[0].dialogue[0].text == "你好。"
    assert len(engine.calls) == 1


async def test_s8_translate_with_null_n_everywhere_keeps_english_without_crashing():
    reply = {"panels": [{"n": None, "dialogue": [{"speaker": "papi", "text": "你好。"}]}]}
    engine = FakeEngine([reply, reply])
    sb = storyboard([make_panel(1, dialogue=[("papi", "Hello there.")])])
    await StoryboarderV2(engine).translate(sb, "zh-CN")
    assert sb.panels[0].dialogue[0].text == "Hello there."
    assert (len(engine.calls), sb.language) == (2, "zh-CN")


async def test_s8_write_tolerates_bare_lists_null_n_and_junk_panels():
    def panel(title, **extra):
        return {"chapter": "hook", "title": title, "characters": ["papi"], "dialogue": [{"speaker": "papi", "text": f"{title}."}], **extra}

    sb = await StoryboarderV2(FakeEngine([[panel("One", n=None), "junk", None, panel("Two", n=None), panel("Three", n="x")]])).write(
        {"title": "Brief"}, make_cast(), (1, 6), "en-US")
    assert [(p.n, p.title) for p in sb.panels] == [(1, "One"), (2, "Two"), (3, "Three")]
    assert sb.title == "Brief"


# ============================================================================= S8 — fixed in round 3

BARE_LIST_WITH_PROSE = [
    pytest.param(json.dumps(ITEMS) + "\n\nNote: I kept every number exact.", id="trailing-note"),
    pytest.param("Here are the rewritten lines:\n" + json.dumps(ITEMS), id="lead-in"),
    pytest.param("```JSON\n" + json.dumps(ITEMS) + "\n```", id="uppercase-fence"),
]


@pytest.mark.parametrize("reply", BARE_LIST_WITH_PROSE)
def test_s8_bare_list_with_surrounding_prose_is_unwrapped(reply):
    assert extract_json(reply, key="items") == {"items": ITEMS}


async def test_s8_factcheck_applies_a_bare_problem_list_followed_by_a_note():
    problems = [{"id": "1.n", "issue": "that figure belongs to the big model", "corrected": "The big model reached 28.4 BLEU."}]
    engine = FakeEngine([json.dumps(problems) + "\n\nOnly one line needed a fix.", {"problems": problems}])
    sb = storyboard([make_panel(1, dialogue=[("papi", "Hi.")], narration="The base model reached 28.4 BLEU.")])

    assert await StoryboarderV2(engine).factcheck(sb, PaperText("T", "body", [])) == 1
    assert sb.panels[0].narration == "The big model reached 28.4 BLEU."


async def test_s8_write_accepts_a_bare_panel_list_followed_by_a_note():
    panels = [{"n": 1, "chapter": "hook", "title": "Why", "characters": ["papi"], "dialogue": [{"speaker": "papi", "text": "Hello."}]}]
    engine = FakeEngine([json.dumps(panels) + "\n\nLet me know if you want changes.", {"panels": panels}])
    sb = await StoryboarderV2(engine).write({"title": "Brief"}, make_cast(), (1, 3), "en-US")
    assert [p.title for p in sb.panels] == ["Why"]


def test_s8_storyboard_string_label_is_one_label():
    assert PanelV2.from_dict({"on_image_labels": "Attention"}, 1).labels == ["Attention"]


@pytest.mark.parametrize("emotions", ["happy", ["happy"]])
async def test_s8_malformed_emotions_do_not_drop_the_panel(emotions):
    reply = {"panels": [
        {"n": 1, "chapter": "evidence", "title": "Results", "characters": ["papi"], "dialogue": [{"speaker": "papi", "text": "28.4 BLEU."}],
         "emotions": emotions},
        {"n": 2, "chapter": "takeaway", "title": "End", "characters": ["kumo"], "dialogue": [{"speaker": "kumo", "text": "Wow."}]},
    ]}
    sb = await StoryboarderV2(FakeEngine([reply])).write({}, make_cast(), (1, 3), "en-US")
    assert [p.title for p in sb.panels] == ["Results", "End"]


async def test_s8_null_chapter_title_keeps_the_original():
    engine = FakeEngine([{"chapters": [{"id": "hook", "title": None}], "panels": [{"n": 1, "dialogue": [{"speaker": "papi", "text": "你好。"}]}]}])
    sb = storyboard([make_panel(1, dialogue=[("papi", "Hello there.")])], chapters=[Chapter("hook", "Why should anyone care?", [1])])
    await StoryboarderV2(engine).translate(sb, "zh-CN")
    assert sb.chapters[0].title == "Why should anyone care?"
