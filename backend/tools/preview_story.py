"""Export real storyboard words and rendered boards as an offline reading preview.

No model calls or character art. Example:
  .venv/bin/python backend/tools/preview_story.py --script output/run/storyboard.zh-CN.json \
      --script output/run/storyboard.en-US.json --before output/run/before-storyboard.json \
      --out output/run/preview.html
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
from pathlib import Path
import sys

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from services.jobs import _storyboard_from_dict
from services.reader_review import _board_lines, is_novice, visible_fingerprint


def escape(value) -> str:
    return html.escape(str(value or ""), quote=True)


def board_image(spec, language: str) -> bytes:
    from services.boards import render_board
    image = render_board(spec, language, (1000, 720)).convert("RGB")
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def reading_notes(sb) -> str:
    if not is_novice(sb):
        return ""
    review = sb.brief.get("reader_review")
    if (not isinstance(review, dict) or review.get("fingerprint") != visible_fingerprint(sb)
            or review.get("language") != sb.language or review.get("status") == "unassessed"):
        message = "这一版最终文字还没有可用的自动阅读检查。"
    elif review.get("status") == "needs_revision":
        findings = review.get("problems") or []
        lines = [f'<li>第 {escape(x.get("n"))} 格：{escape(x.get("problem"))}</li>'
                 for x in findings if isinstance(x, dict) and x.get("problem")]
        message = "自动阅读检查还发现了需要讲清楚的地方。" + ("<ul>" + "".join(lines) + "</ul>" if lines else "")
    elif review.get("status") == "reviewed":
        message = "自动阅读检查未列出问题；这不代表真实读者已经理解。"
    else:
        message = "这一版最终文字还没有可用的自动阅读检查。"
    return ('<details class="reading-note"><summary>讲解还有哪些地方待确认</summary><div>'
            + message + '<p>这是 AI 提供的修改线索。内容是否易懂，仍由实际阅读来判断。</p></div></details>')


def render_story(sb, key: str, label: str, render=board_image) -> tuple[str, list[dict]]:
    panels, failures = [], []
    for p in sb.panels:
        content = f'<article class="panel" id="{key}-p{p.n}"><span class="number">第 {p.n} 格</span>'
        if p.narration:
            content += f'<p class="narration">{escape(p.narration)}</p>'
        # Reading order follows the actual manga: narration, dialogue, then board.
        for bubble in p.dialogue:
            content += (f'<div class="speech"><span class="speaker">{escape(bubble.speaker)}</span>'
                        f'<p>{escape(bubble.text)}</p></div>')
        if p.kind == "board" and isinstance(p.board, dict) and p.board:
            try:
                image = base64.b64encode(render(p.board, sb.language)).decode("ascii")
                content += (f'<img class="board" src="data:image/png;base64,{image}" '
                            f'alt="第 {p.n} 格的图解，文字内容可在下方展开" width="1000" height="720" loading="lazy">'
                            '<button type="button" class="zoom">放大图解</button>')
            except Exception as exc:
                failures.append({"panel": p.n, "language": sb.language, "error_type": type(exc).__name__})
                content += '<p class="board-error" role="status">这张图未能绘制，先查看下方的图中文字。</p>'
            lines = _board_lines(p.board, sb.language)
            content += ('<details class="board-text"><summary>图中文字</summary><div>'
                        + "".join(f'<p>{escape(line)}</p>' for line in lines) + '</div></details>')
        content += '</article>'
        panels.append(content)
    section = (f'<section class="story" id="{key}" lang="{escape(sb.language)}" aria-labelledby="{key}-title">'
               f'<h2 id="{key}-title">{escape(label)}</h2><p class="count">共 {len(sb.panels)} 格</p>'
               + reading_notes(sb) + "".join(panels) + '</section>')
    return section, failures


STYLE = """
:root{color-scheme:light;--ink:#17212b;--muted:#52606d;--blue:#184d87;--line:#d9dedf}
*{box-sizing:border-box}body{margin:0;background:#f5f4ef;color:var(--ink);font-family:system-ui,"PingFang SC","Microsoft YaHei",sans-serif;font-size:18px;line-height:1.8}
main{max-width:720px;margin:auto;padding:28px 18px 64px;overflow-wrap:anywhere}h1{font-size:1.65rem;line-height:1.4;margin:0 0 12px}h2{font-size:1.25rem;margin-bottom:0}
.intro{color:var(--muted);font-size:.95rem}.intro strong{color:var(--ink)}nav{display:flex;gap:10px;flex-wrap:wrap;margin:24px 0}
button{font:inherit;font-size:1rem;border:1px solid var(--ink);border-radius:100px;padding:9px 18px;background:white;color:var(--ink);cursor:pointer;min-height:44px}
button[aria-pressed="true"]{background:var(--ink);color:white}button:focus-visible,summary:focus-visible{outline:3px solid #8063d8;outline-offset:4px}
.count{margin-top:0;color:var(--muted);font-size:.9rem}.panel{background:white;border:1px solid var(--line);border-radius:18px;padding:20px;margin:20px 0 28px}
.number{display:inline-block;color:var(--blue);font-size:.82rem;font-weight:700;border-bottom:2px solid #b5cbe1;margin-bottom:8px}
.narration{margin:10px 0 16px;color:#354653}.speech{display:flex;gap:12px;margin:12px 0;align-items:flex-start}.speaker{font-size:.76rem;color:var(--blue);background:#edf3fa;padding:2px 7px;border-radius:6px;margin-top:5px;flex:none;max-width:5em}.speech p{margin:0;min-width:0}
.board{display:block;width:100%;height:auto;margin:20px 0 6px;border:1px solid #eee;border-radius:8px}.board-error{color:#8e391e;border-left:3px solid #bd6a2a;padding-left:12px}
details{font-size:.88rem;color:var(--muted)}summary{cursor:pointer;min-height:36px;padding:4px 0}details p{margin:8px 0}.board-text{border-top:1px dashed var(--line);margin-top:12px}.reading-note{padding:12px 16px;background:#fff7e7;border-radius:12px;margin:14px 0}
.reading-note li{margin:8px 0}.story[hidden]{display:none}footer{font-size:.85rem;color:var(--muted);margin-top:36px}
.zoom{font-size:.85rem;padding:5px 14px;border-color:var(--line)}dialog{width:min(96vw,1100px);max-width:96vw;max-height:92dvh;border:1px solid var(--line);border-radius:12px;padding:14px}dialog::backdrop{background:#17212baa}.zoom-scroll{overflow:auto;max-height:75dvh;margin-top:12px}.zoom-scroll img{display:block;width:1000px;max-width:none;height:auto}
@media(max-width:400px){main{padding:20px 12px 48px}.panel{padding:16px 13px}.speech{gap:8px}body{font-size:17px}}
"""

SCRIPT = """(()=>{const buttons=[...document.querySelectorAll('button[data-view]')];
function show(id){document.querySelectorAll('section.story').forEach(s=>s.hidden=s.id!==id);buttons.forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.view===id)));}
buttons.forEach(b=>b.addEventListener('click',()=>show(b.dataset.view)));if(buttons.length)show(buttons[0].dataset.view);
const dialog=document.getElementById('board-dialog');document.querySelectorAll('button.zoom').forEach(b=>b.addEventListener('click',()=>{const source=b.previousElementSibling;const img=dialog.querySelector('img');img.src=source.src;img.alt=source.alt;dialog.showModal();}));dialog.querySelector('button').addEventListener('click',()=>dialog.close());})();"""


def build_preview(stories: list[tuple[dict, str]], render=board_image) -> tuple[str, dict]:
    if not stories:
        raise ValueError("At least one storyboard is required")
    sections, buttons, failures = [], [], []
    for index, (data, label) in enumerate(stories):
        sb = _storyboard_from_dict(data)
        key = f'story-{index}'
        section, errors = render_story(sb, key, label, render)
        sections.append(section)
        failures.extend(errors)
        buttons.append(f'<button type="button" data-view="{key}" aria-controls="{key}" aria-pressed="false">{escape(label)}</button>')
    document = ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
                '<title>论文讲解 · 实际生成分镜</title><style>' + STYLE + '</style></head><body><main>'
                '<header><h1>这篇论文，按顺序讲清楚</h1><p class="intro"><strong>实际生成的分镜文字与图解。</strong>'
                '角色画面尚未绘制，先看每一步是否讲得明白。不同版本可以切换比较。</p></header>'
                '<nav aria-label="选择阅读版本">' + ''.join(buttons) + '</nav>' + ''.join(sections)
                + '<footer>试着合上页面，用自己的话说：问题是什么、改了哪一步、为什么可能有帮助？卡住的位置就是需要重写的位置。</footer>'
                '</main><dialog id="board-dialog" aria-label="放大图解"><button type="button">关闭图解</button>'
                '<div class="zoom-scroll"><img alt=""></div></dialog><script>' + SCRIPT + '</script></body></html>')
    return document, {"versions": len(stories), "board_errors": failures, "character_art_generated": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--script', type=Path, action='append', required=True)
    parser.add_argument('--before', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    names = {"zh-CN": "新版 · 中文", "en-US": "新版 · English", "ja-JP": "新版 · 日本語"}
    stories = []
    for path in args.script:
        data = json.loads(path.read_text(encoding='utf-8'))
        stories.append((data, names.get(data.get('language'), '新版')))
    if args.before:
        stories.append((json.loads(args.before.read_text(encoding='utf-8')), '旧版'))
    document, report = build_preview(stories)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(document, encoding='utf-8')
    args.out.with_suffix('.report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))
    if report['board_errors']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
