"""Offline regression coverage for v3 translation's numeric acceptance boundary."""

import json

import pytest

from services.storyboarder_v2 import StoryboarderV2, _translation_numbers, translation_problems
from .fakes import FakeEngine
from .test_story_v3 import v3_panel, v3_storyboard


def reply(text, narration="", n=7):
    return {"panels": [{"n": n, "dialogue": [{"speaker": "papi", "text": text}], "narration": narration}]}


@pytest.mark.parametrize("field", ["dialogue", "narration"])
async def test_number_drop_then_corrected_retry(field):
    source, good, bad = "It takes 3.5 days.", "需要3.5天。", "需要几天。"
    panel = v3_panel(7, dialogue=[("papi", source if field == "dialogue" else "Look here.")],
                     narration=source if field == "narration" else "")
    responses = [reply(bad), reply(good)] if field == "dialogue" else [reply("看这里。", bad), reply("看这里。", good)]
    engine = FakeEngine(responses)
    writer = StoryboarderV2(engine)
    await writer.translate(v3_storyboard([panel]), "zh-CN")
    assert (panel.dialogue[0].text if field == "dialogue" else panel.narration) == good
    assert len(engine.calls) == 2
    assert not writer.skipped


@pytest.mark.parametrize("source,bad", [
    ("It takes 3.5 days.", "需要几天。"),
    ("It takes 3.5 days.", "需要3.5天和8块GPU。"),
    ("Both runs take 8 and 8 days.", "两次运行需要8天。"),
    ("It takes 8 days.", "需要8天，再加8天。"),
    ("100 samples", "一百亿个样本"),
    ("100 samples", "一百億個のサンプル"),
    ("Scores are 3 and 5.", "分数为三点五。"),
    ("It takes 3.5 days.", "需要负三点五天。"),
])
@pytest.mark.parametrize("field", ["dialogue", "narration"])
async def test_twice_invalid_keeps_english_and_reports_panel(source, bad, field):
    panel = v3_panel(7, dialogue=[("papi", source if field == "dialogue" else "Look here.")],
                     narration=source if field == "narration" else "")
    payload = reply(bad) if field == "dialogue" else reply("看这里。", bad)
    engine = FakeEngine([payload, payload])
    writer = StoryboarderV2(engine)
    await writer.translate(v3_storyboard([panel]), "zh-CN")
    assert (panel.dialogue[0].text if field == "dialogue" else panel.narration) == source
    assert len(engine.calls) == 2
    assert writer.skipped
    assert "zh-CN" in json.dumps(writer.skipped) and "7" in json.dumps(writer.skipped)


@pytest.mark.parametrize("source,translated", [
    ("It takes 2.0 days and 1,000 samples.", "需要2天和1000个样本。"),
    ("It uses 8 GPUs.", "使用八块GPU。"),
    ("It uses eight GPUs.", "使用8块GPU。"),
    ("It uses 12 GPUs.", "使用十二块GPU。"),
    ("It uses 1,000 GPUs.", "使用一千块GPU。"),
    ("Both scores are 8 and 8.", "两个分数为8和8。"),
    ("The scores are 8 and 8.", "分数为八和八。"),
    ("GPT-3 uses 8 GPUs.", "GPT-3使用8块GPU。"),
    ("It reaches 50%.", "达到百分之50。"),
    ("It reaches 50%.", "达到百分之五十。"),
    ("It reaches 50%.", "达到５０％。"),
    ("Look here.", "看这里。"),
    ("We work together.", "我们一起工作。"),
])
async def test_valid_canonical_values_need_no_retry(source, translated):
    panel = v3_panel(7, dialogue=[("papi", source)])
    engine = FakeEngine([reply(translated)])
    writer = StoryboarderV2(engine)
    await writer.translate(v3_storyboard([panel]), "zh-CN")
    assert panel.dialogue[0].text == translated
    assert len(engine.calls) == 1 and not writer.skipped


@pytest.mark.parametrize("bad", [{"panels": []}, reply("Hello there."),
                                  {"panels": [{"n": 7, "dialogue": [{"speaker": "nezu", "text": "看这里。"}]}]},
                                  RuntimeError("offline failure")])
async def test_retry_failure_is_observable(bad):
    panel = v3_panel(7, dialogue=[("papi", "Look here.")])
    engine = FakeEngine([{"panels": []}, bad])
    writer = StoryboarderV2(engine)
    await writer.translate(v3_storyboard([panel]), "ja-JP")
    assert panel.dialogue[0].text == "Look here."
    assert len(engine.calls) == 2
    assert writer.skipped and "ja-JP" in json.dumps(writer.skipped) and "7" in json.dumps(writer.skipped)


async def test_downstream_repairs_cannot_overwrite_rejected_dialogue():
    panel = v3_panel(7, dialogue=[("papi", "It takes 3.5 days.")])
    panel.detail = "More detail."
    other = v3_panel(8, dialogue=[("papi", "Most runs take 8 and 8 days.")])
    first = reply("需要几天。")
    first["panels"] += reply("运行需要8和8天。", n=8)["panels"]
    engine = FakeEngine([first, reply("需要几天。"),
                         {"items": [{"id": "7.x", "text": "更多细节。"}, {"id": "7.d0", "text": "需要几天。"}]},
                         {"items": [{"id": "7.d0", "text": "需要3.5天。"}, {"id": "8.d0", "text": "大多数运行需要8天。"}]}])
    writer = StoryboarderV2(engine)
    await writer.translate(v3_storyboard([panel, other]), "zh-CN")
    assert panel.dialogue[0].text == "It takes 3.5 days."
    assert panel.detail == "更多细节。"
    assert other.dialogue[0].text == "运行需要8和8天。"
    assert len(engine.calls) == 4 and writer.skipped
    assert [p["n"] for p in json.loads(engine.prompts[1].rsplit("\n\n", 1)[1])["panels"]] == [7]


async def test_v2_numeric_acceptance_is_unchanged():
    panel = v3_panel(7, dialogue=[("papi", "It takes 3.5 days.")])
    sb = v3_storyboard([panel])
    sb.version = 2
    engine = FakeEngine([reply("需要几天。")])
    await StoryboarderV2(engine).translate(sb, "zh-CN")
    assert panel.dialogue[0].text == "需要几天。" and len(engine.calls) == 1


@pytest.mark.parametrize("field", ["dialogue", "narration"])
@pytest.mark.parametrize("replacement,accepted", [("大多数运行达到百分之五十。", True),
                                                    ("大多数运行达到目标。", False),
                                                    ("大多数运行达到50%，需要八天。", False)])
async def test_wording_repair_keeps_canonical_numbers(field, replacement, accepted):
    source, initial = "Most runs reach 50%.", "运行达到50%。"
    panel = v3_panel(7, dialogue=[("papi", source if field == "dialogue" else "Look here.")],
                     narration=source if field == "narration" else "")
    payload = reply(initial) if field == "dialogue" else reply("看这里。", initial)
    pid = "7.d0" if field == "dialogue" else "7.n"
    engine = FakeEngine([payload, {"items": [{"id": pid, "text": replacement}]}])
    writer = StoryboarderV2(engine)
    await writer.translate(v3_storyboard([panel]), "zh-CN")
    assert (panel.dialogue[0].text if field == "dialogue" else panel.narration) == (replacement if accepted else initial)
    assert len(engine.calls) == 2


@pytest.mark.parametrize("source,translated,language", [
    ("30,000 samples", "3万个样本", "zh-CN"),
    ("30,000 samples", "３万個のサンプル", "ja-JP"),
    ("10,000,000,000 samples", "一百亿个样本", "zh-CN"),
    ("10,000,000,000 samples", "百億個のサンプル", "ja-JP"),
    ("This is certain.", "这是一定的。", "zh-CN"),
    ("This is sufficient.", "これは十分です。", "ja-JP"),
    ("This is sufficiently clear.", "これは十分に明確です。", "ja-JP"),
    ("This is very useful.", "这十分有用。", "zh-CN"),
    ("It takes 10 minutes.", "十分間かかります。", "ja-JP"),
    ("It takes 20 minutes.", "二十分間かかります。", "ja-JP"),
    ("It scored 10 points.", "得了十分。", "zh-CN"),
    ("The first law.", "第一定律。", "zh-CN"),
    ("It takes 3.5 days.", "需要三点五天。", "zh-CN"),
    ("The change is -3.5.", "变化为负三点五。", "zh-CN"),
    ("The change is -3.5.", "変化はマイナス三・五です。", "ja-JP"),
])
async def test_cjk_translation_regressions_need_no_retry(source, translated, language):
    panel = v3_panel(7, dialogue=[("papi", source)])
    engine = FakeEngine([reply(translated)])
    writer = StoryboarderV2(engine)
    await writer.translate(v3_storyboard([panel]), language)
    assert panel.dialogue[0].text == translated
    assert len(engine.calls) == 1 and not writer.skipped


@pytest.mark.parametrize("source,translated", [
    ("123456789", "一億二千三百四十五万六千七百八十九"),
    ("230000000", "2億3千万"),
    ("35000", "三点五万"),
    ("35000", "3.5萬"),
    ("5000", ".5万"),
    ("35000", "3万5千"),
    ("2026", "二〇二六"),
    ("-0.5", "負零點五"),
    ("-3.5", "マイナス３．５"),
    ("+3.5", "正三点五"),
    ("50%", "百分之五十"),
    ("3.5%", "百分之三点五"),
    ("1,000", "１，０００"),
    ("2.3e4", "２．３×１０⁴"),
    ("2.3e4", "二万三千"),
    ("-2.3e4", "負二万三千"),
    ("-2.3e4", "−２．３×１０⁴"),
    ("-3.5 and 5", "负三点五和五"),
])
def test_cjk_and_existing_numeric_normalization(source, translated):
    assert _translation_numbers(source) == _translation_numbers(translated)


@pytest.mark.parametrize("source,translated", [
    ("100", "一百兆"),  # unsupported scales must not silently leave a valid prefix
    ("3 and 5", "三点五"),
    ("3 and 5", "三點五"),
    ("3 and 5", "三・五"),
    ("3.5", "負三點五"),
    ("30000 and 30000", "三万"),
    ("30000", "三万和3万"),
    ("", "十分間"),
    ("", "二十分"),
    ("", "得了十分。"),
    ("", "第一定律。"),
    ("3.5 and 40000", "3.5e4万"),  # unsupported compound stays opaque
    ("-3.5 and 5", "三点五和负五"),  # sign belongs to its value
    ("1000", "十百"),
])
def test_cjk_numbers_cannot_be_truncated_split_or_dropped(source, translated):
    assert _translation_numbers(source) != _translation_numbers(translated)


@pytest.mark.parametrize("replacement,accepted", [("大多数运行使用一百亿个样本。", False),
                                                    ("大多数运行使用一百个样本。", True)])
async def test_magnitude_guard_also_applies_to_wording_repair(replacement, accepted):
    source, initial = "Most runs use 100 samples.", "运行使用100个样本。"
    panel = v3_panel(7, dialogue=[("papi", source)])
    engine = FakeEngine([reply(initial), {"items": [{"id": "7.d0", "text": replacement}]}])
    await StoryboarderV2(engine).translate(v3_storyboard([panel]), "zh-CN")
    assert panel.dialogue[0].text == (replacement if accepted else initial)
    assert len(engine.calls) == 2


def test_certain_only_exempts_equivalent_wording():
    assert not translation_problems("This is certain.", "这是一定的。", "zh-CN")
    assert not translation_problems("The first law.", "第一定律。", "zh-CN")
    assert translation_problems("This may work.", "这一定有效。", "zh-CN")
    assert translation_problems("This is certain.", "这完美无缺。", "zh-CN")
