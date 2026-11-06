"""Panel 24 geometry: a lower dialogue tail must stop before the teaching clamp."""
from copy import deepcopy

from PIL import Image, ImageChops, ImageDraw

from services import lettering as L

HEADS = {'kumo': (43, 994, 273, 1226), 'papi': (249, 779, 627, 1095),
         'nezu': (724, 986, 959, 1218)}
CLAMP = (291, 1137, 652, 1299)
RECTS = [('kumo', (114, 265, 506, 379)), ('papi', (26, 401, 986, 731)),
         ('nezu', (104, 1328, 996, 1496))]


def test_compose_protects_clamp_from_tail_without_covering_faces():
    # Real measured heads, clamp and bubble rectangles; synthetic pixels keep the
    # regression offline and independent of generated, untracked output files.
    art = Image.new('RGB', (1024, 1536), (255, 247, 235))
    draw = ImageDraw.Draw(art)
    for head in HEADS.values():
        draw.rectangle(head, fill=(150, 100, 60))
    draw.rectangle(CLAMP, fill=(80, 100, 110))
    fonts = L.FontSet(config={})
    ctx = L._Ctx(art, 'zh-CN', fonts, HEADS, objects=[CLAMP])
    face = fonts.face('zh-CN', 40, emphasis=True)
    block = L.TextBlock(face, L._line_height(40), ['说明'], face.length('说明'), L._line_height(40))
    bubbles = [L.PlacedBubble(speaker, '说明', block, (x1+x2)/2, (y1+y2)/2,
                             (x2-x1)/2-ctx.stroke, (y2-y1)/2-ctx.stroke, ctx.stroke)
               for speaker, (x1,y1,x2,y2) in RECTS]
    plan = L._Plan(bubbles=bubbles)
    # Control confirms these coordinates exercise the old collision, rather than
    # passing only because the tail happened to point elsewhere.
    unprotected = L._Ctx(art, 'zh-CN', fonts, HEADS)
    old, old_layout = L._compose(unprotected, deepcopy(plan), art.size, '', None)
    assert ImageChops.difference(old.crop(CLAMP), art.crop(CLAMP)).getbbox() is not None

    out, layout = L._compose(ctx, plan, art.size, '', None)
    assert ImageChops.difference(out.crop(CLAMP), art.crop(CLAMP)).getbbox() is None
    for head in HEADS.values():
        assert ImageChops.difference(out.crop(head), art.crop(head)).getbbox() is None
    assert [b['rect'] for b in layout['bubbles']] == [list(rect) for _,rect in RECTS]
    assert [b['font_size'] for b in layout['bubbles']] == [40,40,40]
    assert layout['fallback'] is False and layout['strip'] is None
    tail = layout['bubbles'][-1]['tail']
    assert tail is not None and tail != old_layout['bubbles'][-1]['tail']
    # Same attachment, shorter tail toward Nezu: no path rerouting around props.
    assert tail[0] == old_layout['bubbles'][-1]['tail'][0]
    assert tail[1][1] > CLAMP[3]
    assert layout['bubbles'][-1]['chip'] is True
