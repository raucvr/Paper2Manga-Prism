"""The localized length pass cannot undo translation's numeric acceptance boundary."""
import pytest

from services.storyboarder_v2 import StoryboarderV2
from .fakes import FakeEngine
from .test_story_v3 import v3_panel, v3_storyboard


@pytest.mark.parametrize("field", ["dialogue", "narration"])
@pytest.mark.parametrize("replacement", ["使用几块GPU。", "使用八块GPU，训练十轮。"])
async def test_localized_fit_refuses_lost_or_added_written_numbers(field, replacement):
    original = "为了让读者理解这项实验的背景和研究过程，我们使用八块GPU。" * 3
    panel = v3_panel(7, dialogue=[("papi", original if field == "dialogue" else "看这里。")],
                     narration=original if field == "narration" else "")
    pid = "7.d0" if field == "dialogue" else "7.n"
    fast = FakeEngine([{"items": [{"id": pid, "text": replacement}]}])
    writer = StoryboarderV2(FakeEngine(), fast_engine=fast)
    assert await writer.fit(v3_storyboard([panel]), "zh-CN", rounds=1) == 0
    assert (panel.dialogue[0].text if field == "dialogue" else panel.narration) == original


async def test_localized_fit_accepts_canonical_number_rephrasing():
    panel = v3_panel(7, dialogue=[("papi", "研究团队在本次实验中使用八块GPU，并认真记录训练过程中的实验结果。")])
    fast = FakeEngine([{"items": [{"id": "7.d0", "text": "使用8块GPU。"}]}])
    writer = StoryboarderV2(FakeEngine(), fast_engine=fast)
    assert await writer.fit(v3_storyboard([panel]), "zh-CN", rounds=1) == 1
    assert panel.dialogue[0].text == "使用8块GPU。"


async def test_fit_leaves_rejected_translation_original_untouched():
    original = "We trained eight models on long benchmark runs and carefully recorded all results."
    panel = v3_panel(7, dialogue=[("papi", original)])
    bad = {"panels": [{"n": 7, "dialogue": [{"speaker": "papi", "text": "我们训练了模型。"}]}]}
    fast = FakeEngine([{"items": [{"id": "7.d0", "text": "我们训练了八个模型。"}]}])
    writer = StoryboarderV2(FakeEngine([bad, bad]), fast_engine=fast)
    sb = await writer.translate(v3_storyboard([panel]), "zh-CN")
    assert writer.skipped and panel.dialogue[0].text == original
    assert await writer.fit(sb, "zh-CN", rounds=1) == 0
    assert not fast.calls
    assert panel.dialogue[0].text == original


async def test_unsolicited_fit_reply_cannot_replace_retained_line():
    original = "We trained eight models on long benchmark runs and carefully recorded all results."
    long_chinese = "这是用来向读者解释实验背景和研究过程的完整说明文字，需要进行长度压缩。" * 2
    first = v3_panel(7, dialogue=[("papi", original)])
    second = v3_panel(8, dialogue=[("papi", "Look here.")])
    bad = {"panels": [{"n": 7, "dialogue": [{"speaker": "papi", "text": "我们训练了模型。"}]},
                      {"n": 8, "dialogue": [{"speaker": "papi", "text": long_chinese}]}]}
    fast = FakeEngine([{"items": [{"id": "7.d0", "text": "我们训练了八个模型。"},
                                  {"id": "8.d0", "text": "看这里。"}]}])
    writer = StoryboarderV2(FakeEngine([bad, bad]), fast_engine=fast)
    sb = await writer.translate(v3_storyboard([first, second]), "zh-CN")
    assert await writer.fit(sb, "zh-CN", rounds=1) == 1
    assert '"7.d0"' not in fast.prompts[0]
    assert first.dialogue[0].text == original and second.dialogue[0].text == "看这里。"
