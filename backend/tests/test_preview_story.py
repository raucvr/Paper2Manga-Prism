"""The offline preview must show real text, disclose missing boards, and never leak hidden context."""
import copy

from tools.preview_story import build_preview
from services.jobs import _storyboard_from_dict
from services.reader_review import visible_fingerprint


def sample():
    return {"title": "HIDDEN_TITLE", "language": "zh-CN", "version": 3,
            "brief": {"reader_contract": "novice_v1", "glossary": ["HIDDEN_GLOSSARY"]},
            "card": {"summary": "HIDDEN_CARD"}, "panels": [{"n": 1, "chapter": "how", "kind": "board",
            "narration": "学生先写，再由老师指导。", "detail": "HIDDEN_DETAIL", "visual": "HIDDEN_VISUAL",
            "dialogue": [{"speaker": "papi", "text": "老师看学生已经写出的内容。"}],
            "board": {"kind": "list", "title": "两步", "items": ["学生自己生成", "老师逐步指导"]}}]}


def test_visible_words_only_and_escaped_model_text():
    data = sample()
    data["panels"][0]["dialogue"][0]["text"] = '<script>alert("model output")</script>'
    document, report = build_preview([(data, "中文")], render=lambda *_: b'png')
    assert 'HIDDEN_' not in document
    assert '&lt;script&gt;alert(&quot;model output&quot;)&lt;/script&gt;' in document
    assert '<script>alert' not in document
    assert "学生自己生成" in document and "老师逐步指导" in document
    assert report["board_errors"] == [] and report["character_art_generated"] is False


def test_failed_board_is_visible_and_reported_without_error_payload():
    def fail(*_):
        raise ValueError("PRIVATE_PATH_OR_PROVIDER_DETAIL")
    document, report = build_preview([(sample(), "中文")], render=fail)
    assert "这张图未能绘制" in document and "老师逐步指导" in document
    assert "PRIVATE_PATH" not in document
    assert report["board_errors"] == [{"panel": 1, "language": "zh-CN", "error_type": "ValueError"}]


def test_stale_review_cannot_vouch_for_changed_words():
    data = sample()
    sb = _storyboard_from_dict(data)
    data["brief"]["reader_review"] = {"status": "reviewed", "language": "zh-CN", "fingerprint": visible_fingerprint(sb)}
    current, _ = build_preview([(data, "中文")], render=lambda *_: b'png')
    assert "自动阅读检查未列出问题" in current
    changed = copy.deepcopy(data)
    changed["panels"][0]["narration"] = "新的说法，需要重新检查。"
    stale, _ = build_preview([(changed, "中文")], render=lambda *_: b'png')
    assert "还没有可用的自动阅读检查" in stale
    assert "自动阅读检查未列出问题" not in stale


def test_variants_have_separate_targets_and_real_board_zoom():
    document, _ = build_preview([(sample(), "新版"), (sample(), "旧版")], render=lambda *_: b'png')
    assert 'aria-controls="story-0"' in document and 'id="story-0-p1"' in document
    assert 'aria-controls="story-1"' in document and 'id="story-1-p1"' in document
    assert 'class="zoom"' in document and '<dialog id="board-dialog"' in document
