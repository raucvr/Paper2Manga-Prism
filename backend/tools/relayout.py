"""
Re-letter a finished v3 job from its saved art and grounding: free (no image or vision call), so lettering, board and
script changes can be checked on paid art. Optionally translates the script first (text model only).

usage: .venv/bin/python -I backend/tools/relayout.py output/<job_id> --out DIR [--language zh-CN] [--translate] [--storyboard FILE]

Reads art_NN.jpg, grounding_NN.json and storyboard.json (or --storyboard) from the job folder, writes
DIR/panel_NN.<lang>.jpg, DIR/layout_NN.<lang>.json and DIR/contact.<lang>.jpg. Never writes into the job folder.
--translate: translate the job's script into --language (with fit and the translation fact-check against paper.txt) and
save it as DIR/storyboard.<lang>.json before lettering.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from PIL import Image  # noqa: E402


def load_grounding(path: Path):
    """grounding_NN.json as the job runner reads it (the idea's object boxes included)."""
    from services.jobs import _grounding_from_file
    return _grounding_from_file(path)


async def translated(job: Path, sb, language: str):
    from services.engine_select import build_engines
    from services.paper_extractor import extract_text_only
    from services.storyboarder_v2 import StoryboarderV2
    engines = build_engines()
    writer = StoryboarderV2(engines.text, engines.fast_text, engines.check, engines.translate)
    paper = extract_text_only((job / "paper.txt").read_text(encoding="utf-8"), title=sb.title)
    out = await writer.translate(copy.deepcopy(sb), language)
    await writer.fit(out, language)
    if await writer.factcheck(out, paper, language):
        await writer.fit(out, language)
    for engine in {id(e): e for e in (engines.text, engines.fast_text, engines.vision, engines.image)}.values():
        try:
            await engine.close()
        except Exception:  # noqa: BLE001
            pass
    print(f"translation usage: {writer.usage}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--language")
    ap.add_argument("--translate", action="store_true")
    ap.add_argument("--storyboard", type=Path)
    args = ap.parse_args()

    from services.jobs import _storyboard_from_dict
    from services.lettering import FontSet, letter_panel
    sys.path.insert(0, str(BACKEND_DIR / "tools"))
    from dry_run import contact

    sb = _storyboard_from_dict(json.loads((args.storyboard or args.job / "storyboard.json").read_text(encoding="utf-8")))
    language = args.language or sb.language
    args.out.mkdir(parents=True, exist_ok=True)
    if args.translate and language != sb.language:
        sb = asyncio.run(translated(args.job, sb, language))
        (args.out / f"storyboard.{language}.json").write_text(json.dumps(sb.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
    fonts, made, report = FontSet(), [], {}
    for p in sb.panels:
        art_path = args.job / f"art_{p.n:02d}.jpg"
        if not art_path.exists():
            report[p.n] = "no art"
            continue
        with Image.open(art_path) as im:
            art = im.convert("RGB")
        img, layout = letter_panel(art, p, load_grounding(args.job / f"grounding_{p.n:02d}.json"), language, fonts)
        path = args.out / f"panel_{p.n:02d}.{language}.jpg"
        img.convert("RGB").save(path, quality=92)
        (args.out / f"layout_{p.n:02d}.{language}.json").write_text(json.dumps(layout, ensure_ascii=False, indent=1), encoding="utf-8")
        made.append(path)
        report[p.n] = "fallback" if layout.get("fallback") else ("board_error: " + layout["board_error"]) if layout.get("board_error") else "ok"
    contact(made, args.out / f"contact.{language}.jpg")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
