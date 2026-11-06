"""
Script-only dry run of the v3 story layer: real text models, no images. It writes what a reader would see, so the
story, the code-drawn boards and the abstract quiz can be judged (and iterated on) for the text cost of a job: tens of
cents, up to about $0.80 per paper with DeepSeek V4 Pro as the checker (pass --checker and --translator to change it).

usage: .venv/bin/python -I backend/tools/dry_run.py <paper.pdf> --out DIR [--languages en-US,zh-CN] [--panels auto] [--mock]
                                          [--quiz QUIZ.json] [--writer MODEL] [--checker MODEL] [--translator MODEL]

One English script is written, then translated into every other language (the art of a v3 job serves all languages).
Per language it writes storyboard.<lang>.json, transcript.<lang>.txt and boards/board_NN.<lang>.png; with --mock also
mock/<lang>/panel_NN.jpg (the panel lettered on a blank stand-in: grey heads at the bottom, a white board on board
panels) and mock/contact.<lang>.jpg, and tables.txt (the tables read from the PDF layout, as the script cites them).
--quiz FILE ([{"id", "q", "a"}], written independently of the script) is answered from the English transcript with the
full manga, without the recap panels, and closed book, then graded, always by the same Seed reader and grader
(QUIZ_READER, QUIZ_GRADER), so runs with different writers compare.
--writer / --checker / --translator override core_engine's text_model / check_model / translate_model for this run
("byteplus/deepseek-v4-pro-ga-260813"). Costs are priced per model (services.pricing).
-I keeps the PDF's folder off sys.path; the backend folder is added explicitly.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import sys
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from PIL import Image, ImageDraw  # noqa: E402

STAND_IN = (1024, 1536)
QUIZ_READER = "byteplus/dola-seed-2-1-turbo-260628"
QUIZ_GRADER = "byteplus/seed-2-0-pro-260328"


def stand_in(panel) -> tuple[Image.Image, object]:
    """A blank stand-in for the art: plain paper, grey heads at the bottom (left to right in speaking order), a white
    board in the middle of board panels. Returns (image, lettering.Grounding)."""
    from services.lettering import Grounding
    from services.panel_renderer import stage_order
    W, H = STAND_IN
    img = Image.new("RGB", (W, H), (236, 231, 222))
    d = ImageDraw.Draw(img)
    board = panel.kind == "board" and isinstance(panel.board, dict) and bool(panel.board)
    quad = None
    if board:
        x1, y1, x2, y2 = 40, int(H * 0.22), W - 40, int(H * 0.70)
        d.rectangle((x1, y1, x2, y2), fill=(252, 252, 250), outline=(60, 60, 60), width=8)
        quad = [(x1 + 6, y1 + 6), (x2 - 6, y1 + 6), (x2 - 6, y2 - 6), (x1 + 6, y2 - 6)]
    names = stage_order(panel) or list(panel.characters)
    heads = {}
    top = int(H * (0.74 if board else 0.62))
    for i, name in enumerate(names):
        cx = int(W * (i + 1) / (len(names) + 1))
        box = (cx - 110, top, cx + 110, top + 200)
        d.ellipse(box, fill=(150, 146, 140))
        d.rectangle((cx - 150, top + 190, cx + 150, H), fill=(170, 166, 160))
        heads[name] = box
    return img, Grounding(size=(W, H), heads=heads, board=quad)


def contact(paths: list[Path], out: Path, cols: int = 4, width: int = 360) -> None:
    thumbs = []
    for p in paths:
        with Image.open(p) as im:
            im = im.convert("RGB")
            thumbs.append(im.resize((width, round(im.height * width / im.width))))
    if not thumbs:
        return
    rows = (len(thumbs) + cols - 1) // cols
    cell_h = max(t.height for t in thumbs)
    sheet = Image.new("RGB", (cols * (width + 8) + 8, rows * (cell_h + 8) + 8), (255, 255, 255))
    for i, t in enumerate(thumbs):
        sheet.paste(t, (8 + (i % cols) * (width + 8), 8 + (i // cols) * (cell_h + 8)))
    sheet.save(out, quality=88)


def teaching_only(sb):
    """The storyboard without its recap panels: what the manga teaches before it sums up."""
    taught = copy.deepcopy(sb)
    taught.panels = [p for p in taught.panels if (p.beat or "") != "recap"]
    return taught


async def fixed_quiz(engines, sb, quiz_path: Path) -> dict:
    """``engines``: the quiz's own reader (fast_text) and grader (text), never the writer under test."""
    from services import abstract_quiz as aq
    raw = json.loads(quiz_path.read_text(encoding="utf-8"))
    raw = raw.get("questions", raw) if isinstance(raw, dict) else raw
    questions = [{"id": str(q.get("id") or f"Q{i + 1}"), "claim_ids": [], "question": q.get("q") or q.get("question"),
                  "answer": q.get("a") or q.get("answer")} for i, q in enumerate(raw)]
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    out = {}
    for name, text in (("full", aq.transcript(sb)), ("teaching_only", aq.transcript(teaching_only(sb))),
                       ("closed_book", "(This manga has no panels.)")):
        try:
            answers = await aq.answer_questions(engines.fast_text, text, questions, usage=usage)
            grades = await aq.grade(engines.text, questions, answers, text, usage=usage, cite=name != "closed_book")
        except Exception as e:  # noqa: BLE001  (closed book may legitimately answer nothing)
            out[name] = {"score": 0.0, "error": f"{type(e).__name__}: {str(e)[:200]}"}
            continue
        total = sum(float(g.get("score") or 0) for g in grades)
        out[name] = {"score": round(total / max(1, len(questions)), 3), "points": f"{total:g}/{len(questions)}",
                     "items": [{"id": q["id"], "q": q["question"], "gold": q["answer"], "answer": a.get("answer"),
                                "panels": a.get("panels"), "score": g.get("score"), "why": g.get("why")}
                               for q, a, g in zip(questions, answers, grades)]}
    out["usage"] = usage
    return out


def write_language(sb, lang: str, out: Path, fonts, mock: bool) -> dict:
    from services import abstract_quiz as aq
    from services import boards, lettering
    (out / f"storyboard.{lang}.json").write_text(json.dumps(sb.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
    (out / f"transcript.{lang}.txt").write_text(aq.transcript(sb), encoding="utf-8")
    (out / "boards").mkdir(exist_ok=True)
    board_errors = {}
    for p in sb.panels:
        if p.kind == "board" and isinstance(p.board, dict) and p.board:
            try:
                boards.render_board(p.board, lang, (900, 640), fonts).convert("RGB").save(out / "boards" / f"board_{p.n:02d}.{lang}.png")
            except Exception as e:  # noqa: BLE001
                board_errors[p.n] = f"{type(e).__name__}: {e}"
    report = {"panels": len(sb.panels), "boards": sum(p.kind == "board" for p in sb.panels), "board_errors": board_errors}
    if mock:
        folder = out / "mock" / lang
        folder.mkdir(parents=True, exist_ok=True)
        made, fallbacks = [], []
        for p in sb.panels:
            art, grounding = stand_in(p)
            img, layout = lettering.letter_panel(art, p, grounding, lang, fonts)
            path = folder / f"panel_{p.n:02d}.jpg"
            img.save(path, quality=90)
            (folder / f"layout_{p.n:02d}.json").write_text(json.dumps(layout, ensure_ascii=False, indent=1), encoding="utf-8")
            made.append(path)
            if layout.get("fallback") or layout.get("board_error"):
                fallbacks.append(p.n)
        contact(made, out / "mock" / f"contact.{lang}.jpg")
        report["mock_fallbacks"] = fallbacks
    return report


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--languages", default="en-US")
    ap.add_argument("--panels", default="auto")
    ap.add_argument("--theme", default="kumomo")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--quiz", type=Path)
    ap.add_argument("--writer", help="text_model for this run (provider/model)")
    ap.add_argument("--checker", help="check_model for this run (provider/model)")
    ap.add_argument("--translator", help="translate_model for this run (provider/model)")
    ap.add_argument("--no-figures", action="store_true", help="do not read the paper's figures with the vision model")
    args = ap.parse_args()

    from config_loader import get_config
    from services.cast import build_cast
    from services.engine_select import build_engines
    from services.lettering import FontSet
    from services.paper_extractor import extract_paper
    from services.pricing import text_usd, usage_usd
    from services.storyboarder_v2 import StoryboarderV2, panel_range_for

    languages = [x.strip() for x in args.languages.split(",") if x.strip()]
    primary, others = "en-US", [x for x in languages if x != "en-US"]
    args.out.mkdir(parents=True, exist_ok=True)
    ms = get_config().manga_settings
    ms = ms() if callable(ms) else ms
    lo, hi = ms.min_panels, ms.max_panels

    t0 = time.time()
    pdf = args.pdf.read_bytes()
    paper = extract_paper(pdf, 120_000, args.pdf.name)
    from services.ledger import section
    (args.out / "tables.txt").write_text(section(paper.body) or "(no tables read from the PDF layout)\n", encoding="utf-8")
    engines = build_engines({"text_model": args.writer, "check_model": args.checker, "translate_model": args.translator})
    raw_ms = get_config().get_raw_config().get("manga_settings", {}) or {}
    if not args.no_figures and str(raw_ms.get("read_figures", True)).strip().lower() not in ("false", "0", "no", "off"):
        from services.paper_figures import add_figures, section as figures_section
        figs = await add_figures(paper, pdf, engines.vision, cache_dir=Path(__file__).resolve().parents[2] / "output" / "_figures")
        (args.out / "figures.txt").write_text(figures_section(figs) or "(no figures read)\n", encoding="utf-8")
    cast, _sheets = build_cast(args.theme)
    writer = StoryboarderV2(engines.text, engines.fast_text, engines.check, engines.translate,
                            cache_key=f"dry-{args.out.name}-{int(t0)}")
    stages = []
    sb = await writer.build_v3(paper, cast, primary, panel_range_for(args.panels, lo, hi),
                               on_stage=lambda s, m: (stages.append((round(time.time() - t0, 1), s, m)), print(f"[{time.time() - t0:6.1f}s] {m}")))
    fonts = FontSet()
    report = {"paper": paper.title, "seconds": {}, "stages": stages, "languages": {}, "tables": paper.stats.get("tables"),
              "models": {"writer": getattr(engines.text, "model", ""), "checker": getattr(engines.check, "model", ""),
                         "translator": getattr(engines.translate, "model", ""), "fast": getattr(engines.fast_text, "model", "")}}
    report["seconds"]["script_en"] = round(time.time() - t0, 1)
    report["languages"][primary] = write_language(sb, primary, args.out, fonts, args.mock)
    report["lint_left"] = sb.brief.get("lint")
    report["writer_quiz"] = {k: v for k, v in (sb.quiz or {}).items() if k in ("score", "failed_claim_ids", "untested_claim_ids", "rounds", "first_score")}
    report["abstract_claims"] = sb.abstract_claims
    report["running_example"] = sb.running_example
    for lang in others:
        t1 = time.time()
        translated = await writer.translate(copy.deepcopy(sb), lang)
        await writer.fit(translated, lang)
        if await writer.factcheck(translated, paper, lang):
            await writer.fit(translated, lang)
            if await writer.backport(sb, translated, paper):
                await writer.fit(sb, "en-US")
                write_language(sb, primary, args.out, fonts, args.mock)   # the English files with the corrected lines
        report["seconds"][f"translate_{lang}"] = round(time.time() - t1, 1)
        report["languages"][lang] = write_language(translated, lang, args.out, fonts, args.mock)
    report["usage"] = dict(writer.usage)
    report["usage_by_model"] = engines.token_usage()
    report["usd_by_model"] = {m: round(text_usd(m, u["prompt_tokens"], u["completion_tokens"], u["cached_tokens"]), 4)
                              for m, u in report["usage_by_model"].items()}
    report["usd_script"] = round(usage_usd(report["usage_by_model"]), 4)
    report["cached_share"] = {m: round(u["cached_tokens"] / max(1, u["prompt_tokens"]), 3) for m, u in report["usage_by_model"].items()}
    if args.quiz:
        quiz_engines = build_engines({"text_model": QUIZ_GRADER, "fast_text_model": QUIZ_READER})
        quiz = await fixed_quiz(quiz_engines, sb, args.quiz)
        report["fixed_quiz"] = {k: (v["score"] if isinstance(v, dict) and "score" in v else v) for k, v in quiz.items() if k != "usage"}
        report["usd_quiz"] = round(usage_usd(quiz_engines.token_usage()), 4)
        (args.out / "fixed_quiz.json").write_text(json.dumps(quiz, ensure_ascii=False, indent=1), encoding="utf-8")
        await quiz_engines.close()
    report["seconds"]["total"] = round(time.time() - t0, 1)
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    try:
        await engines.close()
    except Exception:  # noqa: BLE001
        pass
    print(json.dumps({k: report[k] for k in ("paper", "models", "tables", "seconds", "usd_script", "writer_quiz", "languages") if k in report}
                     | ({"fixed_quiz": report["fixed_quiz"]} if "fixed_quiz" in report else {}), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
