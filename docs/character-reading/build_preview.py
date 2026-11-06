"""Build an offline reader from completed, lettered character panels. No model calls.

Run from the isolated worktree:
  .venv/bin/python docs/character-reading/build_preview.py --run output/character-reading-20261106
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from services.reader_review import _board_lines
from services.boards import render_board

NAMES = {"papi": "Papi", "kumo": "Kumo", "nezu": "Nezu"}


def esc(value):
    return html.escape(str(value or ""), quote=True)


STYLE = """
:root{color-scheme:light;--ink:#302a23;--muted:#685f55;--paper:#faf5e9;--accent:#8c4a2d}
*{box-sizing:border-box}html{scroll-padding-top:76px}body{margin:0;background:var(--paper);color:var(--ink);font:18px/1.7 system-ui,"PingFang SC",sans-serif}
a{color:var(--accent)}header,main,footer{max-width:1000px;margin:auto;padding:24px}header{padding-top:48px}h1{font-size:clamp(28px,5vw,44px);line-height:1.3;margin:12px 0}h2{font-size:22px;line-height:1.5}p{margin:12px 0}.eyebrow,.hint{color:var(--muted);font-size:15px}.intro{max-width:660px}.cast{display:flex;flex-wrap:wrap;gap:8px}.cast span{border:1px solid #d0c1a9;border-radius:20px;padding:4px 12px;font-size:14px}
nav{position:sticky;top:0;z-index:2;background:#faf5e9f5;border-block:1px solid #d0c1a9;padding:9px 20px;display:flex;align-items:center;justify-content:center;gap:12px;flex-wrap:wrap;font-size:15px}select,button{font:inherit;color:var(--ink);background:#fffdfa;border:1px solid #bcae96;border-radius:8px;min-height:44px;padding:8px 14px;cursor:pointer}select{max-width:55vw}button:hover{background:#f4e6d1}button:focus-visible,a:focus-visible,summary:focus-visible,select:focus-visible{outline:3px solid #966626;outline-offset:3px}
article{margin:0 0 48px;scroll-margin-top:12px}article h2{margin:10px 0 16px}.number{font-size:13px;color:var(--accent);letter-spacing:.06em}.art-link{display:block;padding:0;background:transparent;border:0;border-radius:0;width:100%;line-height:0}.art-link img{display:block;width:100%;height:auto;box-shadow:0 4px 18px #302a2314}.controls{display:flex;gap:8px;flex-wrap:wrap;padding:12px 0}.controls button{font-size:15px}details{border-top:1px solid #d0c1a9;padding-top:8px}summary{cursor:pointer;min-height:44px;padding:8px 0}.transcript{background:#fffdf7;border-radius:10px;padding:14px 18px;font-size:18px}.transcript p{margin:12px 0;overflow-wrap:anywhere}.transcript strong{color:var(--accent)}.board-copy{border-top:1px dashed #d0c1a9;margin-top:16px;padding-top:12px}
dialog{width:min(96vw,1120px);max-width:96vw;max-height:94dvh;padding:12px;border:1px solid #bcae96;border-radius:12px;background:var(--paper)}dialog::backdrop{background:#17140ecc}.dialog-bar{display:flex;justify-content:space-between;align-items:center;gap:12px}.dialog-bar p{font-size:15px;margin:0}.zoom-scroll{max-height:78dvh;overflow:auto;margin-top:10px;overscroll-behavior:contain}.zoom-scroll img{display:block;width:auto;max-width:none;height:auto}footer{font-size:15px;color:var(--muted);padding-bottom:60px}
@media(min-width:1100px){main{max-width:920px}}@media(max-width:600px){header,footer{padding:24px 16px}main{padding:24px 8px}article h2,article>.number,article>details,article>.controls{margin-left:8px;margin-right:8px}nav{padding:8px;gap:8px}nav label{font-size:14px}h2{font-size:19px}article{margin-bottom:36px}}
"""

SCRIPT = """(()=>{
const dialog=document.querySelector('dialog'),target=dialog.querySelector('img'),caption=dialog.querySelector('p');let opener=null;
document.querySelectorAll('[data-zoom]').forEach(button=>button.addEventListener('click',()=>{opener=button;target.src=button.dataset.zoom;target.alt=button.dataset.title;caption.textContent=button.dataset.title+' · 可上下、左右滚动';dialog.showModal();dialog.querySelector('.zoom-scroll').scrollTo(0,0);}));
dialog.querySelector('button').addEventListener('click',()=>dialog.close());dialog.addEventListener('close',()=>opener?.focus({preventScroll:true}));
document.getElementById('chapter').addEventListener('change',event=>{const panel=document.getElementById(event.target.value);panel.scrollIntoView();panel.querySelector('h2').focus({preventScroll:true});});
})();"""


def build(run: Path):
    data = json.loads((run / "storyboard.zh-CN.json").read_text())
    out = run / "public"
    missing = [p["n"] for p in data["panels"] if not (out / f'panel_{p["n"]:02d}.jpg').is_file()]
    if missing:
        raise ValueError(f"Missing lettered panels: {missing}")
    (out / "boards").mkdir(exist_ok=True)
    sections, chapters, report = [], {}, []
    count = len(data["panels"])
    for order, panel in enumerate(data["panels"], 1):
        n, title = panel["n"], panel["title"]
        chapter = panel.get("chapter") or "正文"
        chapters.setdefault(chapter, f"panel-{n}")
        src = f"panel_{n:02d}.jpg"
        with Image.open(out / src) as image:
            width, height = image.size
        content = (f'<article id="panel-{n}"><span class="number">{order:02d} / {count} · {esc(chapter)}</span>'
                   f'<h2 tabindex="-1">{esc(title)}</h2><button class="art-link" type="button" data-zoom="{src}" '
                   f'data-title="{esc(title)}" aria-label="放大第{order}格：{esc(title)}">'
                   f'<img src="{src}" width="{width}" height="{height}" loading="{"eager" if order == 1 else "lazy"}" '
                   f'alt="第{order}格：{esc(title)}。完整对白可在下方展开。"></button>'
                   f'<div class="controls"><button type="button" data-zoom="{src}" data-title="{esc(title)}">放大画面</button>')
        board_lines = []
        if panel.get("kind") == "board" and panel.get("board"):
            board_name = f"boards/board_{n:02d}.png"
            render_board(panel["board"], data["language"], (1000, 720)).save(out / board_name)
            board_lines = _board_lines(panel["board"], data["language"])
            content += f'<button type="button" data-zoom="{board_name}" data-title="第{order}格图解">单独看图解</button>'
        content += '</div><details><summary>阅读本格文字</summary><div class="transcript">'
        if panel.get("narration"):
            content += f'<p>{esc(panel["narration"])}</p>'
        for bubble in panel.get("dialogue", []):
            content += f'<p><strong>{esc(NAMES.get(bubble["speaker"], bubble["speaker"]))}：</strong>{esc(bubble["text"])}</p>'
        if board_lines:
            content += '<div class="board-copy">' + ''.join(f'<p>{esc(line)}</p>' for line in board_lines) + '</div>'
        content += '</div></details></article>'
        sections.append(content)
        report.append({"order": order, "panel_id": n, "image": src, "size": [width, height], "board": bool(board_lines)})
    nav = ''.join(f'<option value="{target}">{esc(chapter)}</option>' for chapter, target in chapters.items())
    document = ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
                f'<title>{esc(data["title"])} · Prism漫画</title><style>{STYLE}</style></head><body>'
                '<header><div class="eyebrow">PRISM · 论文漫画</div>'
                f'<h1>{esc(data["title"])}</h1><p class="intro">一个会解题的模型，压小以后还会解题吗？和三个伙伴一起，看懂 OnQ 做了什么、实验说明了什么。</p>'
                '<div class="cast"><span>Kumo 小云 · 提问</span><span>Papi 小狗 · 讲解</span><span>Nezu 小刺猬 · 追问依据</span></div>'
                f'<p class="hint">共 {count} 格。点击画面放大；图表还可以单独打开。每格下方都有可展开的文字。</p></header>'
                '<nav aria-label="章节导航"><label for="chapter">跳到章节</label><select id="chapter">'+nav+'</select><a href="#panel-1">回到开头</a></nav><main>'
                + ''.join(sections) + '</main><footer>读完试着回答：为什么变小会损失能力？学生自己生成的文字有什么用？老师提供了什么学习目标？'
                '<p>阅读试作。角色画面已绘制；论文结论仍以图中标明的实验范围为准。</p></footer>'
                '<dialog aria-label="放大阅读"><div class="dialog-bar"><p></p><button type="button">关闭</button></div><div class="zoom-scroll"><img alt=""></div></dialog>'
                f'<script>{SCRIPT}</script></body></html>')
    (out / "index.html").write_text(document, encoding="utf-8")
    (run / "preview-report.json").write_text(json.dumps({"character_art_generated": True, "panels": report}, ensure_ascii=False, indent=2) + "\n")
    print(f"{count} completed panels: {out / 'index.html'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    build(parser.parse_args().run.resolve())
