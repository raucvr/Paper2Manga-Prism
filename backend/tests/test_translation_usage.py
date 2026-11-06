"""Common nonnumeric words and quantities identified by cross-review."""
import pytest

from services.storyboarder_v2 import StoryboarderV2
from .fakes import FakeEngine
from .test_story_v3 import v3_panel, v3_storyboard


@pytest.mark.parametrize("source,translated", [
    ("This is a model.", "这是一个模型。"),
    ("First, we train.", "首先，我们训练。"),
    ("1,000 samples per second.", "每秒1000个样本。"),
    ("8 percentage points.", "8个百分点。"),
    ("This is unique.", "这是唯一的。"),
    ("Improve it further.", "进一步改进。"),
    ("Just in case.", "以防万一。"),
    ("Be sure to keep it.", "千万要保留它。"),
    ("One million samples.", "一百万个样本。"),
    ("1.2 million samples.", "120万个样本。"),
    ("3 billion samples.", "30亿个样本。"),
])
async def test_common_language_and_magnitudes_do_not_trigger_retries(source, translated):
    panel = v3_panel(7, dialogue=[("papi", source)])
    response = {"panels": [{"n": 7, "dialogue": [{"speaker": "papi", "text": translated}]}]}
    engine = FakeEngine([response])
    writer = StoryboarderV2(engine)
    await writer.translate(v3_storyboard([panel]), "zh-CN")
    assert panel.dialogue[0].text == translated
    assert len(engine.calls) == 1 and not writer.skipped


@pytest.mark.parametrize("source,translated,accepted", [
    ("This is a model.", "这是模型。", True),
    ("It uses 1 GPU.", "使用一块GPU。", True),
    ("It uses one GPU.", "使用一块GPU。", True),
    ("It uses 1 GPU.", "使用GPU。", False),
    ("It uses one GPU.", "使用GPU。", False),
    ("This is a model.", "这是1个模型。", False),
    ("This is a model.", "这是两个模型。", False),
])
async def test_article_allowance_does_not_hide_numeric_addition_or_loss(source, translated, accepted):
    panel = v3_panel(7, dialogue=[("papi", source)])
    response = {"panels": [{"n": 7, "dialogue": [{"speaker": "papi", "text": translated}]}]}
    engine = FakeEngine([response] if accepted else [response, response])
    writer = StoryboarderV2(engine)
    await writer.translate(v3_storyboard([panel]), "zh-CN")
    assert panel.dialogue[0].text == (translated if accepted else source)
    assert bool(writer.skipped) is not accepted


@pytest.mark.parametrize("source,accepted", [("Tap 11 times.", True), ("Tap 10 times.", False)])
async def test_lexical_exclusions_do_not_truncate_a_number(source, accepted):
    translated = "敲十一下。"
    panel = v3_panel(7, dialogue=[("papi", source)])
    response = {"panels": [{"n": 7, "dialogue": [{"speaker": "papi", "text": translated}]}]}
    engine = FakeEngine([response] if accepted else [response, response])
    writer = StoryboarderV2(engine)
    await writer.translate(v3_storyboard([panel]), "zh-CN")
    assert panel.dialogue[0].text == (translated if accepted else source)
    assert bool(writer.skipped) is not accepted
