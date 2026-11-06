"""Standalone introductory captions stay before dialogue without changing the art plan."""
from copy import deepcopy

import pytest
from PIL import ImageChops

from services import lettering as L
from .test_lettering import BAR, HEADS_SCENE, StubBoard, make_art

NARRATION = '这里的“块”是模型的一段计算，不是新添的模块。'


@pytest.fixture(scope='module')
def fonts():
    return L.FontSet(config={})


def compose_fixture(fonts, chart_mode=None):
    art = make_art(heads=HEADS_SCENE.values())
    ctx = L._Ctx(art, 'zh-CN', fonts, HEADS_SCENE)
    result = L._plan(ctx, [('papi', '只调整已有的一块。'), ('kumo', '然后再检查这次变化。')],
                     (40, 40, 984, 620), [])
    assert result is not None
    size, bubbles = result
    plan = L._Plan(narration_mode='strip', bubbles=bubbles, size=size, fallback=True)
    if chart_mode:
        plan.kind, plan.spec, plan.chart_mode = 'board', BAR, chart_mode
        if chart_mode != 'strip':
            plan.chart = (450, 660, 960, 770)
    return art, ctx, plan


@pytest.mark.parametrize('chart_mode', [None, 'board', 'card'])
def test_only_narration_moves_above_with_exact_pixels_and_geometry(fonts, monkeypatch, chart_mode):
    art, ctx, plan = compose_fixture(fonts, chart_mode)
    original_rects = [L._ints(b.rect) for b in plan.bubbles]
    captured = {}
    strip = L._strip

    def capture(*args, **kwargs):
        out, layout = strip(*args, **kwargs)
        captured.update(image=out.copy(), layout=deepcopy(layout))
        return out, layout

    monkeypatch.setattr(L, '_strip', capture)
    out, layout = L._compose(ctx, plan, art.size, NARRATION, StubBoard())
    old, old_strip = captured['image'], captured['layout']
    shift = out.height - art.height
    assert shift > 0 and layout['art_offset'] == [0, shift]
    assert layout['art_size'] == list(art.size) and layout['size'] == list(out.size)
    assert layout['strip']['placement'] == 'above'
    assert layout['narration']['rect'] == layout['strip']['rect']
    assert 0 <= layout['strip']['rect'][1] < layout['strip']['rect'][3] <= shift
    assert layout['strip']['font_size'] == old_strip['font_size']
    assert layout['strip']['items'] == old_strip['items']
    assert layout['strip']['items'][0]['text'] == NARRATION
    # Every rendered art pixel (including names, tails and chart) is moved intact.
    assert ImageChops.difference(out.crop((0, shift, out.width, out.height)),
                                old.crop((0, 0, art.width, art.height))).getbbox() is None
    assert ImageChops.difference(out.crop((0, 0, out.width, shift)),
                                old.crop((0, art.height, old.width, old.height))).getbbox() is None
    for bubble, rect, placed in zip(layout['bubbles'], original_rects, plan.bubbles):
        assert bubble['rect'] == [rect[0], rect[1]+shift, rect[2], rect[3]+shift]
        assert bubble['font_size'] == placed.block.face.size
        assert bubble['text'] == placed.text
    # Recompose without the strip to obtain the original tail coordinates.
    plain_plan = deepcopy(plan)
    plain_plan.narration_mode = None
    _, plain = L._compose(ctx, plain_plan, art.size, '', StubBoard())
    for moved, original in zip(layout['bubbles'], plain['bubbles']):
        assert moved['tail'] == ([[x, y+shift] for x, y in original['tail']] if original['tail'] else None)
    if chart_mode:
        rect = L._ints(plan.chart)
        assert layout['chart'] == [rect[0], rect[1]+shift, rect[2], rect[3]+shift]
    else:
        assert layout['chart'] is None
    for rect in HEADS_SCENE.values():
        x1, y1, x2, y2 = map(int, rect)
        assert ImageChops.difference(out.crop((x1, y1+shift, x2, y2+shift)), art.crop((x1,y1,x2,y2))).getbbox() is None


@pytest.mark.parametrize('mixed', ['dialogue', 'chart', 'no_narration'])
def test_other_fallback_strips_remain_below(fonts, mixed):
    art, ctx, plan = compose_fixture(fonts, 'strip' if mixed == 'chart' else None)
    narration = NARRATION
    if mixed in ('dialogue', 'no_narration'):
        plan.leftover = [('nezu', '还需要说明适用范围。')]
    if mixed == 'no_narration':
        narration, plan.narration_mode = '', None
    out, layout = L._compose(ctx, plan, art.size, narration, StubBoard())
    assert out.height > art.height
    assert 'art_offset' not in layout
    assert 'placement' not in layout['strip']
    assert layout['strip']['rect'][1] >= art.height
    assert layout['bubbles'][0]['rect'] == L._ints(plan.bubbles[0].rect)
    if mixed == 'chart':
        assert layout['chart'][1] >= art.height
    else:
        assert layout['strip']['items'][-1]['speaker'] == 'nezu'
