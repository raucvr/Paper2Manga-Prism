"""
Images-only abstract quiz on a finished manga: the evaluation behind the Seed numbers in README.md and docs/v4-report.md,
run by the pipeline's own models.

A vision model sees only the lettered panels, in reading order (what a reader sees), and answers fixed questions written
from the paper's abstract before any manga existed (eval/quizzes/<paper>.json: [{"id", "q", "a"}]). A text model grades
the answers against the gold answers (services.abstract_quiz.grade). The same reader then answers without the panels
(closed book): what it already knew about the paper, so a famous paper's score can be read against it.

usage: .venv/bin/python -I backend/tools/eval_quiz.py <output/JOB_ID | dry-run dir> --quiz eval/quizzes/bert.json
                                           [--language zh-CN] [--reader provider/model] [--grader provider/model]
                                           [--out result.json]

Panels: a job's panel_NN.<language>.jpg (bilingual jobs) or panel_NN.jpg; a dry run's mock/<language>/panel_NN.jpg.
Defaults: reader = core_engine.vision_model, grader = core_engine.text_model (Seed 2.0 Pro, as in the published numbers:
the quiz grades with thinking off, and DeepSeek V4 Pro grades unreliably that way). About two cents per run.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import re
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from PIL import Image  # noqa: E402

READ_WIDTH = 768   # Seed reads every image at a fixed token cost; a smaller upload keeps the request light


def find_panels(folder: Path, language: str) -> list[Path]:
    """The lettered panels of a job or dry-run folder in reading order."""
    def numbered(paths):
        found = []
        for p in paths:
            m = re.match(r"panel_(\d+)(?:\.([A-Za-z-]+))?\.jpg$", p.name)
            if m:
                found.append((int(m.group(1)), m.group(2) or "", p))
        return found
    mock = folder / "mock" / language
    if mock.is_dir():
        return [p for _, _, p in sorted(numbered(mock.glob("panel_*.jpg")))]
    items = numbered(folder.glob("panel_*.jpg"))
    localized = sorted((n, p) for n, lang, p in items if lang == language)
    if localized:
        return [p for _, p in localized]
    return [p for _, p in sorted((n, p) for n, lang, p in items if not lang)]


def as_image(path: Path):
    from engines.base import ImageContent
    with Image.open(path) as im:
        im = im.convert("RGB")
        if im.width > READ_WIDTH:
            im = im.resize((READ_WIDTH, round(im.height * READ_WIDTH / im.width)), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=88)
    return ImageContent.from_base64(base64.b64encode(buf.getvalue()).decode(), "image/jpeg")


def reader_prompt(n: int, questions: list[dict]) -> str:
    asked = [{"id": q["id"], "question": q["question"]} for q in questions]
    return f"""These are the {n} panels of a manga about a research paper, in reading order (the first image is panel 1). Read every speech bubble, narration box, chart and table in them.
Answer each question using ONLY what the panels show, never what you may know about the paper. When the panels do not show the answer, answer "not shown".
For each answer list the panels that show it and say where it came from: "chart" (a chart or table drawn in a panel), "text" (a bubble or a narration box) or "none".
Output JSON only: {{"answers": [{{"id": "Q1", "answer": "...", "panels": [3], "source": "chart"}}]}}

QUESTIONS:
{json.dumps(asked, ensure_ascii=False)}"""


def closed_book_prompt(title: str, questions: list[dict]) -> str:
    asked = [{"id": q["id"], "question": q["question"]} for q in questions]
    return f"""Answer each question about the research paper "{title}" from what you already know about it. When you do not know, answer "not shown".
Output JSON only: {{"answers": [{{"id": "Q1", "answer": "..."}}]}}

QUESTIONS:
{json.dumps(asked, ensure_ascii=False)}"""


async def ask(engine, prompt: str, images=None) -> dict:
    from engines.base import GenerationConfig, Message
    from services.storyboarder_v2 import extract_json
    config = GenerationConfig(temperature=0.0, max_tokens=6000)
    config.thinking = False
    for attempt in range(2):
        response = await engine.generate_text([Message.user(prompt, images=images or [])], config)
        try:
            return extract_json(response.content, key="answers")
        except (ValueError, json.JSONDecodeError):
            if attempt:
                raise
    return {}


def answers_of(data: dict, questions: list[dict]) -> list[dict]:
    got = {}
    for item in data.get("answers") or []:
        if isinstance(item, dict) and item.get("id") is not None:
            got[str(item["id"]).strip()] = item
    out = []
    for q in questions:
        item = got.get(q["id"], {})
        panels = [int(x) for x in re.findall(r"\d+", json.dumps(item.get("panels", []))) if int(x) > 0]
        out.append({"id": q["id"], "answer": str(item.get("answer") or "not shown").strip(), "panels": panels,
                    "source": str(item.get("source") or "none")})
    return out


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--quiz", type=Path, required=True)
    ap.add_argument("--language", default="")
    ap.add_argument("--reader", help="vision model (provider/model); default core_engine.vision_model")
    ap.add_argument("--grader", help="text model (provider/model); default core_engine.text_model")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    from services import abstract_quiz as aq
    from services.engine_select import build_engines
    from services.pricing import usage_usd

    folder = args.folder
    job = json.loads((folder / "job.json").read_text(encoding="utf-8")) if (folder / "job.json").exists() else {}
    report = json.loads((folder / "report.json").read_text(encoding="utf-8")) if (folder / "report.json").exists() else {}
    language = args.language or job.get("language") or "en-US"
    title = job.get("title") or report.get("paper") or folder.name
    panels = find_panels(folder, language)
    if not panels:
        sys.exit(f"no lettered panels for {language} in {folder}")
    raw = json.loads(args.quiz.read_text(encoding="utf-8"))
    raw = raw.get("questions", raw) if isinstance(raw, dict) else raw
    questions = [{"id": str(q.get("id") or f"Q{i + 1}"), "claim_ids": [], "question": q.get("q") or q.get("question"),
                  "answer": q.get("a") or q.get("answer")} for i, q in enumerate(raw)]

    engines = build_engines({"vision_model": args.reader, "text_model": args.grader})
    reader, grader = engines.vision, engines.text
    images = [as_image(p) for p in panels]
    seen = answers_of(await ask(reader, reader_prompt(len(images), questions), images), questions)
    blind = answers_of(await ask(reader, closed_book_prompt(title, questions)), questions)
    graded = await aq.grade(grader, questions, seen)
    graded_blind = await aq.grade(grader, questions, blind, cite=False)   # no manga, so nothing to cite

    def score(grades) -> float:
        return round(sum(float(g.get("score") or 0) for g in grades) / max(1, len(questions)), 3)

    result = {
        "folder": str(folder), "language": language, "panels": len(panels), "title": title,
        "reader": getattr(reader, "model", ""), "grader": getattr(grader, "model", ""),
        "score": score(graded), "closed_book": score(graded_blind),
        "from_charts": sum(1 for a, g in zip(seen, graded) if "chart" in a["source"].lower() and float(g.get("score") or 0) > 0),
        "items": [{"id": q["id"], "q": q["question"], "gold": q["answer"], "answer": a["answer"], "panels": a["panels"],
                   "source": a["source"], "score": g.get("score"), "why": g.get("why"),
                   "closed_book": b["answer"], "closed_book_score": gb.get("score")}
                  for q, a, g, b, gb in zip(questions, seen, graded, blind, graded_blind)],
        "usd": round(usage_usd(engines.token_usage()), 4),
    }
    await engines.close()
    text = json.dumps(result, ensure_ascii=False, indent=1)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("title", "language", "panels", "reader", "grader", "score", "closed_book",
                                             "from_charts", "usd")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
