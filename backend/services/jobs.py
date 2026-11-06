"""
Job manager for the v2 pipeline.

A job runs as a background asyncio task inside the API process. Its state is a plain dict that is
persisted to ``output/<job_id>/job.json`` after every change, so polling, server restarts and later
"regenerate panel" calls work from disk. Finished panels are written immediately as ``panel_NN.jpg``
(or ``strip_NN-MM.jpg`` in strip layout) and served under ``/output/<job_id>/``.

Pipeline: parse → analyze → storyboard (+fit, fact-check) → translate (+fit) → render panels in parallel
(bounded by ``manga_settings.concurrency``) with vision QA and targeted regeneration → assemble.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import copy
import difflib
import io
import json
import os
import re
import secrets
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from PIL import Image

from config_loader import as_int, get_config, PROJECT_ROOT
from engines.base import GenerationConfig, ImageContent, Message
from services.cast import build_cast
from services.engine_select import Engines, build_engines, preflight as engine_preflight
from services.panel_renderer import PanelRenderer, RenderResult, art_shows
from services.paper_extractor import PaperText, extract_paper, extract_text_only
from services.paper_figures import add_figures
from services.qa_judge import (MEANING_CRITICAL, PanelQA, QAResult, _canon, adopt_wording, agreed_alternatives, same_facts)
from services.storyboarder_v2 import (LANG_NAMES, Chapter, PanelV2, StoryboardV2, StoryboarderV2, _has_script, coverage,
                                      extract_json, panel_range_for)

OUTPUT_DIR = PROJECT_ROOT / "output"
TERMINAL = {"completed", "failed", "cancelled"}
VALID_THEMES = {"kumomo"}
VALID_LANGUAGES = {"en-US", "zh-CN", "ja-JP"}
VALID_LAYOUTS = {"single", "strip"}
VALID_RANGES = {"auto", "short", "long"}
MIN_TEXT_CHARS = 500
FRAME_TRIES = 2   # drawings per frame of a teaching scene drawn as two frames (JobManager._two_frames)
# what a failed redraw puts back, so the panel keeps its previous picture
RESTORE_KEYS = ("status", "image_url", "width", "height", "qa", "strip_index", "strip_width", "strip_height", "art_url", "layout",
                "image_urls", "frames")
LONE_SURROGATE = re.compile(r"[\ud800-\udfff]")


async def _review_final_scripts(writer: StoryboarderV2, scripts: dict[str, StoryboardV2]) -> None:
    """Review the words that ship, after every translation and English backport has finished."""
    from services.reader_review import is_novice, review_story, visible_fingerprint

    for language, script in scripts.items():
        if not is_novice(script):
            continue
        previous = script.brief.get("reader_review")
        if not isinstance(previous, dict):
            previous = {}
        if (previous.get("fingerprint") == visible_fingerprint(script)
                and previous.get("language") == language
                and previous.get("status") in {"reviewed", "needs_revision", "unassessed"}):
            continue
        await review_story(script, writer.check_engine, on_usage=writer._add_usage)


def _public_reader_reviews(scripts: dict[str, StoryboardV2]) -> dict:
    """An old diagnostic cannot vouch for edited words, including after a relayout."""
    from services.reader_review import is_novice, visible_fingerprint

    reviews = {}
    for language, script in scripts.items():
        if not is_novice(script):
            continue
        fingerprint = visible_fingerprint(script)
        previous = script.brief.get("reader_review")
        if (isinstance(previous, dict) and previous.get("fingerprint") == fingerprint
                and previous.get("language") == language
                and previous.get("status") in {"reviewed", "needs_revision", "unassessed"}):
            reviews[language] = copy.deepcopy(previous)
        else:
            reviews[language] = {"status": "unassessed", "language": language, "fingerprint": fingerprint,
                                 "answers": [], "problems": [],
                                 "reason": "The final wording has no matching reader review."}
    return reviews


JPEG_MAX_SIDE = 65_000   # libjpeg's hard limit is 65,535 px per side
CHECK_PROMPT = """You check the lettering of an educational manga about a research paper, written in {language}.
{situation}
Script line: "{expected}"
New line: "{drawn}"
Other text in this panel: {context}
Answer accept=true only if ALL of these hold:
1. The new line is correct, natural {language}: no typo; no missing, extra, malformed or invented character.
2. It means the same in this context: same meaning and strength of claim, same negation, direction, quantity, unit,
   technical term and hedge, and a question stays a question.
3. A reader who knows the paper would find both lines equally accurate.
If unsure, answer false. Reply with JSON only: {{"accept": true or false, "reason": "..."}}"""
ADOPTED = "The image AI drew one line with different characters than the script (the new line)."
REWRITTEN = "The image AI could not letter one line, so it was rewritten in more common words (the new line)."
REPHRASE_PROMPT = """An image AI letters the lines of an educational manga about a research paper, written in {language}.
It drew this line wrong twice in a row, the same way both times.
Script line: "{expected}"
What it drew instead: {drawn}
Characters it keeps getting wrong: {chars}
Replace only the word it got wrong with a common synonym of the same strength and meaning (for example a more frequent
word written with different characters), and keep the rest of the line exactly as it is: every number, name, technical
term, negation and the sentence ending stay unchanged, and the line must not get longer. Reply with JSON only:
{{"text": "..."}}"""


def _line_texts(panels: list[PanelV2]) -> dict:
    """{(panel n, "bubble", index) or (panel n, "narration", None): text} for every lettered line of the panels."""
    out = {}
    for p in panels:
        for i, b in enumerate(p.dialogue):
            out[(p.n, "bubble", i)] = b.text
        if p.narration:
            out[(p.n, "narration", None)] = p.narration
    return out


def _failing_chars(expected: str, drawn: str) -> set[str]:
    """The characters of ``expected`` that the drawing got wrong or left out (all of them when nothing was readable)."""
    a, b = _canon(expected), _canon(drawn)
    if not b:
        return set(a)
    return {c for tag, i1, i2, _j1, _j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
            if tag in ("replace", "delete") for c in a[i1:i2] if not c.isspace()}


def _set_line(panels: list[PanelV2], key: tuple, text: str) -> None:
    n, kind, index = key
    panel = next(p for p in panels if p.n == n)
    if kind == "bubble":
        panel.dialogue[index].text = text
    else:
        panel.narration = text


class JobCancelled(Exception):
    """Raised inside the pipeline when the job's cancel flag is set."""


def _now() -> str:
    # milliseconds: the reader tells "saved since my request" by updated_at, and a redraw can fail within a second
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


def _clean(value):
    """Copy of a JSON-like value without lone UTF-16 surrogates: a model reply may carry half an emoji escape
    ("\\ud83d"), which json.loads accepts but UTF-8 cannot encode, so job.json and the API would break."""
    if isinstance(value, str):
        return LONE_SURROGATE.sub("\ufffd", value)
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def _settings() -> dict:
    config = get_config()
    ms = config.manga_settings
    raw = config.get_raw_config().get("manga_settings", {}) or {}
    return {
        "layout": ms.layout if ms.layout in VALID_LAYOUTS else "single",
        "panel_width": ms.panel_width,
        "panel_height": ms.panel_height,
        "strip_width": as_int(raw.get("strip_width"), 2240),     # 2.5 MP: still the cheapest Seedream price tier (<= 2.61 MP)
        "strip_height": as_int(raw.get("strip_height"), 1120),
        "concurrency": ms.concurrency,
        "qa_enabled": ms.qa_enabled,
        "qa_blind_check": ms.qa_blind_check,
        "max_attempts": ms.max_attempts,
        "min_panels": ms.min_panels,
        "max_panels": ms.max_panels,
        "style_anchor": ms.style_anchor,
        "fast": ms.fast,
        "output_format": config.output_settings.image_format,
        "pipeline": getattr(ms, "pipeline", "v3"),
        "bilingual": bool(getattr(ms, "bilingual", True)),
        "read_figures": str(raw.get("read_figures", True)).strip().lower() not in ("false", "0", "no", "off"),
    }


FIGURE_CACHE = "_figures"         # output/_figures: the vision model's readings of each PDF's figures, by file hash

BILINGUAL = ("en-US", "zh-CN")   # a v3 job letters its (wordless) art in both; the job's own language comes first


def _languages(language: str, settings: dict) -> list[str]:
    if settings.get("bilingual") and language in BILINGUAL:
        return [language] + [x for x in BILINGUAL if x != language]
    return [language]


def _panel_texts(p: PanelV2) -> dict:
    """The words of a panel in one language (the reader's text under the picture and the screen-reader description)."""
    return {"title": p.title, "dialogue": [{"speaker": b.speaker, "text": b.text} for b in p.dialogue], "narration": p.narration,
            "board": p.board}


@dataclass
class JobRuntime:
    """Everything that is not JSON-serialisable."""
    cancel: asyncio.Event = field(default_factory=asyncio.Event)
    task: Optional[asyncio.Task] = None
    relayout_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    engines: Optional[Engines] = None
    cast = None
    sheets: dict = field(default_factory=dict)
    storyboard: Optional[StoryboardV2] = None
    storyboards: dict = field(default_factory=dict)   # v3: language -> StoryboardV2 (the job's own language is rt.storyboard)
    renderer: Optional[PanelRenderer] = None
    qa: Optional[PanelQA] = None
    anchor: Optional[ImageContent] = None
    props: dict = field(default_factory=dict)   # v3: book symbol -> its reference picture (props/prop_NN.jpg)
    fonts: object = None             # v3: lettering.FontSet
    pdf_bytes: Optional[bytes] = None
    text: Optional[str] = None
    filename: str = ""


class Job:
    def __init__(self, state: dict, output_dir: Path = OUTPUT_DIR, runtime: Optional[JobRuntime] = None):
        self.state = state
        self.output_dir = output_dir
        self.rt = runtime or JobRuntime()

    @property
    def id(self) -> str:
        return self.state["job_id"]

    @property
    def dir(self) -> Path:
        return self.output_dir / self.id

    @property
    def running(self) -> bool:
        return self.rt.task is not None and not self.rt.task.done()


def _new_state(job_id: str, *, title: str, theme: str, language: str, layout: str, qa: bool, panel_range: str, fast: bool, source: str) -> dict:
    return {
        "job_id": job_id,
        "status": "queued",
        "stage_message": "Queued",
        "created_at": _now(),
        "updated_at": _now(),
        "title": title or "",
        "summary": "",
        "language": language,
        "theme": theme,
        "layout": layout,
        "progress": {"done": 0, "total": 0, "percent": 0},
        "chapters": [],
        "panels": [],
        "combined_url": None,
        "storyboard_url": None,
        "cost": {"images_generated": 0, "image_tokens": 0, "text_prompt_tokens": 0, "text_completion_tokens": 0, "elapsed_seconds": 0.0},
        "error": None,
        "params": {"qa": qa, "panel_range": panel_range, "fast": fast, "source": source, "title_given": bool(title)},
        "_started": time.time(),
    }


def _panel_state(p: PanelV2) -> dict:
    return {
        "n": p.n, "chapter": p.chapter, "title": p.title, "characters": list(p.characters),
        "dialogue": [{"speaker": b.speaker, "text": b.text} for b in p.dialogue], "narration": p.narration,
        "detail": p.detail, "covers": list(p.covers),   # reader text below the picture; the key points it explains
        "kind": p.kind, "board": p.board, "beat": p.beat, "claims": list(p.claims),
        # attempts: cumulative, bumped only when a new picture is published (the reader's image cache key);
        # render_attempts: image attempts of the latest render (1 = the first attempt passed)
        "status": "pending", "image_url": None, "width": None, "height": None, "attempts": 0, "render_attempts": 0, "qa": None,
        "strip_index": None, "strip_width": None, "strip_height": None, "error": None,
    }


def _grounding_from_file(path: Path):
    """grounding_NN.json (saved with the art) as a lettering.Grounding; None when missing or saved as unchecked."""
    from services.lettering import Grounding
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    return Grounding(size=tuple(data.get("size") or (0, 0)), heads={k: tuple(v) for k, v in (data.get("heads") or {}).items()},
                     board=[tuple(p) for p in data["board"]] if data.get("board") else None, bubbles=data.get("bubbles") or [],
                     text_found=data.get("text_found") or [], extras=data.get("extras") or [], matches=data.get("matches") or {},
                     objects=[tuple(b) for b in data.get("objects") or [] if isinstance(b, (list, tuple)) and len(b) == 4])


def _public_quiz(quiz: Optional[dict]) -> Optional[dict]:
    """The abstract check on the visible manga text: score and which claims failed, the score before the revisions and
    after each one (the questions stay in storyboard.json)."""
    if not quiz:
        return None
    before = quiz.get("before") if isinstance(quiz.get("before"), dict) else {}
    return {"score": quiz.get("score"), "failed_claim_ids": list(quiz.get("failed_claim_ids") or []),
            "questions": len(quiz.get("questions") or []), "error": quiz.get("error"), "before": before.get("score"),
            "rounds": [r.get("score") for r in quiz.get("rounds") or [] if isinstance(r, dict)], "undone": quiz.get("undone")}


def _storyboard_from_dict(d: dict) -> StoryboardV2:
    panels = [PanelV2.from_dict(p, i) for i, p in enumerate(d.get("panels", []), start=1)]
    chapters = [Chapter(id=c["id"], title=c.get("title", c["id"]), panel_numbers=list(c.get("panel_numbers", []))) for c in d.get("chapters", [])]
    return StoryboardV2(title=d.get("title", ""), summary=d.get("summary", ""), theme=d.get("theme", "kumomo"), language=d.get("language", "en-US"),
                        chapters=chapters, panels=panels, brief=d.get("brief", {}), card=d.get("card") or {},
                        version=int(d.get("version") or 2), abstract_claims=list(d.get("abstract_claims") or []),
                        running_example=str(d.get("running_example") or ""), quiz=dict(d.get("quiz") or {}))


class JobManager:
    def __init__(self, output_dir: Path = OUTPUT_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, Job] = {}

    # ------------------------------------------------------------------ public API
    def create(self, *, pdf_bytes: Optional[bytes], text: Optional[str], filename: str = "", title: str = "", theme: str = "kumomo",
               language: str = "en-US", layout: Optional[str] = None, qa: Optional[bool] = None, panel_range: str = "auto",
               fast: Optional[bool] = None) -> dict:
        if not pdf_bytes and not (text and len(text.strip()) >= MIN_TEXT_CHARS):
            raise ValueError(f"Provide a PDF file or at least {MIN_TEXT_CHARS} characters of text")
        settings = _settings()
        theme = theme if theme in VALID_THEMES else "kumomo"
        language = language if language in VALID_LANGUAGES else "en-US"
        layout = layout if layout in VALID_LAYOUTS else settings["layout"]
        panel_range = panel_range if panel_range in VALID_RANGES else "auto"
        qa = settings["qa_enabled"] if qa is None else bool(qa)
        fast = settings["fast"] if fast is None else bool(fast)
        job_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(2)
        state = _new_state(job_id, title=title, theme=theme, language=language, layout=layout, qa=qa, panel_range=panel_range, fast=fast,
                           source=filename or ("text" if text else "pdf"))
        job = Job(state, self.output_dir)
        job.rt.pdf_bytes, job.rt.text, job.rt.filename = pdf_bytes, text, filename
        self.jobs[job_id] = job
        job.dir.mkdir(parents=True, exist_ok=True)
        if pdf_bytes:
            (job.dir / "source.pdf").write_bytes(pdf_bytes)
        self._persist(job)
        job.rt.task = asyncio.create_task(self._run(job))
        return self.public(job)

    def get(self, job_id: str) -> Optional[dict]:
        job = self._load(job_id)
        return self.public(job) if job else None

    def list(self, limit: int = 20) -> list[dict]:
        ids = set(self.jobs)
        if self.output_dir.exists():
            ids |= {p.name for p in self.output_dir.iterdir() if (p / "job.json").exists()}
        out = []
        for jid in sorted(ids, reverse=True)[:limit]:
            job = self._load(jid)
            if job:
                s = job.state
                summary = {k: s.get(k) for k in ("job_id", "status", "title", "created_at", "progress", "language", "theme")}
                if s.get("redraw_panels"):   # a user's redraw: the list says "Redrawing", not "Drawing x/y"
                    summary["redraw_panels"] = s["redraw_panels"]
                out.append(_clean(summary))
        return out

    async def cancel(self, job_id: str) -> Optional[dict]:
        job = self._load(job_id)
        if not job:
            return None
        if job.state["status"] not in TERMINAL or job.running:
            job.rt.cancel.set()
            if job.state["status"] not in TERMINAL:   # a task that already settled (or has not started yet) keeps its status
                job.state["status"] = "cancelled"
                job.state["stage_message"] = "Cancelled"
                self._settle_active_panels(job)
                self._persist(job)
            if job.running and asyncio.current_task() is not job.rt.task:
                job.rt.task.cancel()
        return self.public(job)

    async def regenerate_panel(self, job_id: str, n: int) -> Optional[dict]:
        job = self._load(job_id)
        if not job:
            return None
        if job.state["status"] not in TERMINAL or job.running or job.rt.relayout_lock.locked():
            raise ValueError("Job is still running")
        if not any(p["n"] == n for p in job.state["panels"]):
            raise ValueError(f"No panel {n}")
        job.rt.cancel = asyncio.Event()
        job.rt.task = asyncio.create_task(self._regenerate(job, n))
        return {"status": "queued"}

    def public(self, job: Job) -> dict:
        s = job.state
        out = {k: v for k, v in s.items() if not k.startswith("_")}
        out["panels"] = [{k: v for k, v in p.items() if k != "error" or v} for p in s["panels"]]
        out["cost"] = dict(s["cost"])
        if s["status"] not in TERMINAL or job.running:
            out["cost"]["elapsed_seconds"] = round(time.time() - s.get("_started", time.time()), 1)
        return _clean(out)

    # ------------------------------------------------------------------ persistence
    def _persist(self, job: Job) -> None:
        job.state["updated_at"] = _now()
        job.dir.mkdir(parents=True, exist_ok=True)
        tmp = job.dir / "job.json.tmp"
        state = _clean({k: v for k, v in job.state.items() if k != "_started"})
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, job.dir / "job.json")

    def _load(self, job_id: str) -> Optional[Job]:
        if job_id in self.jobs:
            return self.jobs[job_id]
        if not re.fullmatch(r"[0-9]{8}_[0-9]{6}_[0-9a-f]{4}", job_id or ""):
            return None
        path = self.output_dir / job_id / "job.json"
        if not path.exists():
            return None
        state = json.loads(path.read_text(encoding="utf-8"))
        state["_started"] = time.time()
        job = Job(state, self.output_dir)
        # no task can be running in a fresh process: settle whatever was in flight when the last one stopped
        before_redraw = state.pop("_before_redraw", None) or {}
        redraw_cut = state.pop("redraw_panels", None) is not None or bool(before_redraw)
        changed = redraw_cut
        if state.get("status") not in TERMINAL and redraw_cut:   # a user's redraw was cut off: the job goes back as it was
            if before_redraw.get("status") in TERMINAL:
                state["status"], state["error"] = before_redraw["status"], before_redraw.get("error")
            else:   # job.json written without the pre-redraw status: a redraw only runs on a stopped job, so judge by its panels
                pending = any(p.get("status") == "pending" for p in state.get("panels", []))
                state["status"], state["error"] = ("cancelled" if pending else "completed"), None
            state["stage_message"] = "Redraw interrupted by a server restart"
            changed = True
        elif state.get("status") not in TERMINAL:   # the process died mid-run
            state["status"] = "failed"
            state["error"] = state.get("error") or "Server restarted while the job was running"
            state["stage_message"] = "Interrupted by a server restart"
            changed = True
        for p in state.get("panels", []):
            if p.get("status") in ("rendering", "qa"):
                if p.get("image_url") and (job.dir / Path(p["image_url"]).name).exists():
                    p["status"] = "done"
                else:
                    p["status"], p["error"] = "failed", "interrupted by a server restart"
                changed = True
        if changed:
            self._persist(job)
        self.jobs[job_id] = job
        return job

    @staticmethod
    def _settle_active_panels(job: Job) -> None:
        """After a Stop or a failure: panels that were being drawn go back to 'done' (old image kept) or 'pending'."""
        for p in job.state["panels"]:
            if p["status"] in ("rendering", "qa"):
                p["status"] = "done" if p.get("image_url") and p.get("width") else "pending"

    def _set_stage(self, job: Job, status: str, message: str) -> None:
        if job.rt.cancel.is_set():
            raise JobCancelled()
        job.state["status"] = status
        job.state["stage_message"] = message
        self._persist(job)

    def _add_text_usage(self, job: Job, usage: dict) -> None:
        job.state["cost"]["text_prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
        job.state["cost"]["text_completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)

    def _merge_tokens(self, job: Job, engines: Engines) -> None:
        """Add what the runtime's engines spent, per model, to the job's cost ("text_by_model"): the job report prices each
        model on its own (the writer, the checker and the vision model may differ)."""
        try:
            spent = engines.token_usage()
        except Exception:  # noqa: BLE001 - cost accounting never fails a job
            return
        by_model = job.state["cost"].setdefault("text_by_model", {})
        for model, used in spent.items():
            mine = by_model.setdefault(model, {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0})
            for key in mine:
                mine[key] += int(used.get(key, 0) or 0)
        if spent:
            self._persist(job)

    async def _close_runtime(self, job: Job) -> None:
        """Close the engines and drop what a finished task holds (a Job stays in memory for the life of the process);
        a later redraw rebuilds it in _ensure_runtime."""
        rt = job.rt
        engines = rt.engines
        if engines is not None:
            self._merge_tokens(job, engines)
        rt.engines, rt.renderer, rt.qa = None, None, None
        rt.cast, rt.sheets, rt.anchor = None, {}, None
        rt.pdf_bytes, rt.text = None, None
        if engines:
            try:
                await asyncio.shield(engines.close())
            except Exception:
                pass

    async def preflight(self) -> None:
        """Before a job is created: the API key is set (else ValueError) and every text and vision model answers
        (services.engine_select.preflight raises the AccountError of one that cannot: not activated, no balance)."""
        engines = build_engines()
        try:
            await engine_preflight(engines)
        finally:
            await engines.close()

    # ------------------------------------------------------------------ pipeline
    async def _run(self, job: Job) -> None:
        s = job.state
        try:
            await self._prepare(job)
            await self._render_all(job)
            if job.rt.cancel.is_set():
                raise JobCancelled()
            self._set_stage(job, "assembling", "Stitching the long image")
            stitch_error = None
            try:   # every panel is drawn and paid for: a stitching failure must not fail the manga
                await self._stitch(job)
            except Exception as e:  # noqa: BLE001
                stitch_error = f"{type(e).__name__}: {str(e)[:200]}"
                print(f"[Job {job.id}] long image failed: {stitch_error}")
            failed = [p["n"] for p in s["panels"] if p["status"] != "done"]
            s["status"] = "completed"
            s["stage_message"] = "Done" if not failed else f"Done with {len(failed)} failed panel(s): {failed}"
            if stitch_error:
                s["stage_message"] += f" (the long image could not be built: {stitch_error})"
        except (asyncio.CancelledError, JobCancelled):
            s["status"], s["stage_message"] = "cancelled", "Cancelled"
            self._settle_active_panels(job)
        except Exception as e:  # noqa: BLE001
            if getattr(e, "account", False):   # the user must act in the ModelArk console: the message says how
                s["status"], s["error"], s["stage_message"] = "failed", str(e), "Stopped: the ModelArk account needs attention"
                self._settle_active_panels(job)
                return
            import traceback
            traceback.print_exc()
            s["status"], s["error"], s["stage_message"] = "failed", f"{type(e).__name__}: {e}", "Failed"
            self._settle_active_panels(job)
        finally:
            s["cost"]["elapsed_seconds"] = round(time.time() - s.get("_started", time.time()), 1)
            self._persist(job)
            await self._close_runtime(job)

    async def _prepare(self, job: Job) -> None:
        """parse → storyboard; fills job.rt and job.state['panels']."""
        s, rt = job.state, job.rt
        settings = _settings()
        self._set_stage(job, "parsing", "Reading the PDF")
        pdf = rt.pdf_bytes
        try:
            if pdf:
                paper = await asyncio.to_thread(extract_paper, pdf, 120_000, rt.filename)
            else:
                paper = extract_text_only(rt.text or "", title=s["title"])
        finally:
            rt.pdf_bytes, rt.text = None, None   # source.pdf is on disk; free the memory (also after a failure or Stop)
        if len(paper.body) < MIN_TEXT_CHARS:
            raise ValueError("Could not extract enough text from this PDF (is it scanned? try a text PDF)")
        s["title"] = s["title"] or paper.title or Path(rt.filename).stem or "Paper"
        rt.engines = build_engines()
        if pdf and settings.get("read_figures", True):   # what the figures show joins the paper text every step reads
            self._set_stage(job, "parsing", "Reading the paper's figures")
            figs = await add_figures(paper, pdf, rt.engines.vision, cache_dir=self.output_dir / FIGURE_CACHE)
            if figs:
                (job.dir / "figures.json").write_text(json.dumps(figs, ensure_ascii=False, indent=1), encoding="utf-8")
        del pdf
        (job.dir / "paper.txt").write_text(paper.body, encoding="utf-8")   # with the figures and ledger sections: relayouts cite the same cells
        if paper.tables:
            (job.dir / "tables.json").write_text(json.dumps(paper.tables, ensure_ascii=False, indent=1), encoding="utf-8")
        rt.cast, rt.sheets = build_cast(s["theme"])
        writer = StoryboarderV2(rt.engines.text, rt.engines.fast_text, rt.engines.check, rt.engines.translate, cache_key=f"prism-{job.id}")

        def on_stage(status: str, message: str) -> None:
            self._set_stage(job, status, message)

        panel_range = panel_range_for(s["params"]["panel_range"], settings["min_panels"], settings["max_panels"])
        if settings.get("pipeline") == "v3":   # one English script, translated into every language the art is lettered in
            languages = _languages(s["language"], settings)
            english = await writer.build_v3(paper, rt.cast, "en-US", panel_range, on_stage=on_stage)
            scripts = {"en-US": english}
            for lang in languages:
                if lang != "en-US":
                    scripts[lang] = await self._localize(job, writer, english, paper, lang)
            await _review_final_scripts(writer, {lang: scripts[lang] for lang in languages})
            rt.storyboards = {lang: scripts[lang] for lang in languages}
            storyboard = scripts[s["language"]]
            s["languages"] = languages
        else:
            storyboard = await writer.build(paper, rt.cast, s["language"], panel_range, on_stage=on_stage)
        self._add_text_usage(job, writer.usage)
        rt.storyboard = storyboard
        if not s["params"].get("title_given"):
            s["title"] = storyboard.title or s["title"]
        s["summary"] = storyboard.summary
        s["chapters"] = [{"id": c.id, "title": c.title, "panel_numbers": c.panel_numbers} for c in storyboard.chapters]
        s["panels"] = [_panel_state(p) for p in storyboard.panels]
        s["paper_card"], s["coverage"] = storyboard.card or None, coverage(storyboard)
        s["version"] = storyboard.version
        if storyboard.version >= 3:
            s["layout"] = "single"   # v3 letters each panel itself; two-panel strips are a v2 lettering trick
            s["quiz"] = _public_quiz(storyboard.quiz)
        english_sb = rt.storyboards.get("en-US", storyboard) if rt.storyboards else storyboard
        s["quality"] = {   # what was checked: a reader must be able to tell a checked manga from one whose checks could not run
            "skipped": list(writer.skipped),
            "lint_left": sum(len(v) for v in (english_sb.brief.get("lint") or {}).values()),
            "scrubbed": len(english_sb.brief.get("scrubbed") or []),
            "reader_check": len(english_sb.brief.get("clarity") or []),   # stumbles the newcomer's read sent to a fix round
            "board_check": len(english_sb.brief.get("board_check") or []),   # boards that did not back their words
            "board_recheck": len(english_sb.brief.get("board_recheck") or []),   # ...and still did not once rewritten
            "picture_check": len(english_sb.brief.get("picture_check") or []),   # picture plans that could not carry their point
            "unchecked_art": [],
            "unchecked_idea": [],   # teaching scenes whose picture was published without a verdict on its idea
            # a role whose own model refused a text (content filter) or is not activated, done by the writer's model instead
            "fallbacks": sorted({note for e in (rt.engines.check, rt.engines.translate) for note in getattr(e, "fallbacks", [])}),
            "reader_reviews": _public_reader_reviews(rt.storyboards or {storyboard.language: storyboard}),
        }
        s["progress"] = {"done": 0, "total": len(storyboard.panels), "percent": 0}
        (job.dir / "storyboard.json").write_text(json.dumps(_clean(storyboard.to_dict()), ensure_ascii=False, indent=1), encoding="utf-8")
        s["storyboard_url"] = f"/output/{job.id}/storyboard.json"
        if len(rt.storyboards) > 1:
            for lang, other in rt.storyboards.items():
                if lang != s["language"]:
                    (job.dir / f"storyboard.{lang}.json").write_text(json.dumps(_clean(other.to_dict()), ensure_ascii=False, indent=1),
                                                                    encoding="utf-8")
            for entry in s["panels"]:
                entry["texts"] = {lang: _panel_texts(self._panel_in(job, lang, entry["n"])) for lang in rt.storyboards}
            s["localized"] = {   # what else the reader shows in words: title, summary, chapter titles, the paper card
                lang: {"title": s["title"] if s["params"].get("title_given") else other.title or s["title"],
                       "summary": other.summary, "paper_card": other.card or None,
                       "chapters": [{"id": c.id, "title": c.title, "panel_numbers": c.panel_numbers} for c in other.chapters]}
                for lang, other in rt.storyboards.items() if lang != s["language"]}
        self._build_renderers(job, settings)
        self._set_stage(job, "rendering", f"Drawing {len(storyboard.panels)} panels")

    async def _localize(self, job: Job, writer: StoryboarderV2, english: StoryboardV2, paper, language: str) -> StoryboardV2:
        """The English v3 script in another language: translation (board strings included, numbers locked), fit to the
        budgets, and the translation's fact-check against the paper. The panels and their art stay the same."""
        from services.storyboarder_v2 import _finish_v3, lint_v3
        self._set_stage(job, "translating", f"Translating to {LANG_NAMES.get(language, language)}")
        sb = await writer.translate(copy.deepcopy(english), language)
        await writer.fit(sb, language)
        self._set_stage(job, "translating", f"Fact-checking the {LANG_NAMES.get(language, language)} text against the paper")
        if await writer.factcheck(sb, paper, language):
            await writer.fit(sb, language)
            if await writer.backport(english, sb, paper):   # what the check found in the translation, fixed in the English too
                await writer.fit(english, "en-US")
                found = lint_v3(english, paper.body, report=True)   # the English that ships changed after build_v3's last lint
                english.brief["lint"] = {str(n): v for n, v in found.items()}
                english.brief.setdefault("lint_trail", []).append(
                    {"after": f"backport {language}", "panels": len(english.panels),
                     "found": {str(n): [x[:90] for x in v] for n, v in sorted(found.items())}})
        _finish_v3(sb)
        return sb

    def _panel_in(self, job: Job, language: str, n: int) -> PanelV2:
        """Panel n of the script in ``language`` (the job's own script when that language has none)."""
        rt = job.rt
        for sb in (rt.storyboards.get(language), rt.storyboard):
            for p in (sb.panels if sb else []):
                if p.n == n:
                    return p
        raise KeyError(f"panel {n} is not in the script")

    def _build_renderers(self, job: Job, settings: dict) -> None:
        s, rt = job.state, job.rt
        v3 = int(s.get("version") or 2) >= 3
        rt.renderer = PanelRenderer(rt.engines.image, rt.cast, rt.sheets, s["language"],
                                    panel_size=(settings["panel_width"], settings["panel_height"]),
                                    strip_size=(settings["strip_width"], settings["strip_height"]),
                                    fast=bool(s["params"].get("fast")), output_format=settings["output_format"],
                                    **({"text_free": True} if v3 else {}))
        if v3:   # the program letters the panels; the vision model only checks the art
            from services.lettering import FontSet
            rt.qa, rt.fonts = None, FontSet()
            return
        rt.qa = PanelQA(rt.engines.vision, rt.sheets, s["language"], cast_names=rt.cast.names(),
                        cast_looks={m.name: m.look for m in rt.cast.members},
                        blind_check=bool(settings.get("qa_blind_check", False))) if s["params"].get("qa") else None

    def _groups(self, job: Job, only: Optional[list[int]] = None) -> list[list[PanelV2]]:
        allp = job.rt.storyboard.panels
        wanted = {p.n for p in allp} if only is None else set(only)
        if job.state["layout"] == "strip" and int(job.state.get("version") or 2) < 3:
            # strips are fixed pairs (1,2),(3,4)… so regenerating one panel re-renders its pair
            pairs = [allp[k:k + 2] for k in range(0, len(allp), 2)]
            return [pair for pair in pairs if any(p.n in wanted for p in pair)]
        return [[p] for p in allp if p.n in wanted]

    def _redraw_numbers(self, job: Job, n: int) -> list[int]:
        """The panels a redraw of panel n re-renders, from the job state alone (the storyboard may be unavailable):
        n itself, or both halves of its strip (the same fixed pairs as _groups)."""
        numbers = [p["n"] for p in job.state["panels"]]
        if job.state.get("layout") != "strip" or n not in numbers:
            return [n]
        k = numbers.index(n) // 2 * 2
        return numbers[k:k + 2]

    async def _render_all(self, job: Job, only: Optional[list[int]] = None) -> None:
        settings = _settings()
        sem = asyncio.Semaphore(settings["concurrency"])
        groups = self._groups(job, only)
        await self._ensure_props(job)

        async def worker(group: list[PanelV2]) -> None:
            async with sem:
                if job.rt.cancel.is_set():
                    return
                snapshot = [dict(self._panel_entry(job, p.n)) for p in group]
                try:
                    await self._render_group(job, group, settings)
                except asyncio.CancelledError:
                    raise
                except Exception as e:  # noqa: BLE001 — one broken panel must not kill the job
                    if getattr(e, "account", False):   # ... but an account problem does: stop every panel, say what to do
                        raise
                    import traceback
                    traceback.print_exc()
                    self._settle_failed_group(job, group, snapshot, f"{type(e).__name__}: {str(e)[:300]}")
                # unguarded on purpose: if job.json cannot be written the job fails as a whole (siblings are stopped below)
                self._update_progress(job)

        tasks = [asyncio.create_task(worker(g)) for g in groups]
        try:
            await asyncio.gather(*tasks)
        except BaseException:   # Stop, or an exception that escaped a worker: nothing may keep drawing afterwards
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    async def _ensure_props(self, job: Job) -> None:
        """v3: one reference picture per picture symbol of the book (panel_renderer.PROP_PROMPT), drawn once before the
        panels and saved in props/ (a redraw reuses it); every picture that draws a symbol gets its sheet beside the
        character sheets, so the symbol looks the same in every panel. A sheet that cannot be drawn is left out."""
        from services.storyboarder_v2 import _picture_symbols
        rt, s = job.rt, job.state
        if rt.storyboard is None or rt.renderer is None or not rt.renderer.text_free:
            return
        saved = {p.get("symbol"): p.get("file") for p in s.get("props") or [] if isinstance(p, dict)}
        record = []
        for i, (_concept, symbol) in enumerate(_picture_symbols(rt.storyboard)):
            if rt.cancel.is_set():
                return
            name = saved.get(symbol) or f"props/prop_{i:02d}.jpg"
            path = job.dir / name
            if symbol not in rt.props and path.exists():
                rt.props[symbol] = ImageContent.from_base64(base64.b64encode(path.read_bytes()).decode("ascii"), "image/jpeg")
            if symbol not in rt.props:
                s["stage_message"] = f"Drawing the book's symbols ({i + 1})"
                try:
                    result = await rt.renderer.render_prop(symbol)
                    raw, _w, _h = self._decode_image(result.image)
                except asyncio.CancelledError:
                    raise
                except Exception as e:  # noqa: BLE001  (the panels are drawn without this sheet)
                    if getattr(e, "account", False):
                        raise
                    print(f"[Job {job.id}] the sheet of symbol {symbol!r} could not be drawn: {type(e).__name__}: {str(e)[:200]}")
                    continue
                s["cost"]["images_generated"] += 1
                s["cost"]["image_tokens"] += int((result.usage or {}).get("output_tokens", 0) or 0)
                path.parent.mkdir(parents=True, exist_ok=True)
                self._write_atomic(path, raw)
                rt.props[symbol] = ImageContent.from_base64(base64.b64encode(raw).decode("ascii"), "image/jpeg")
            record.append({"symbol": symbol, "file": name})
        s["props"] = record
        self._persist(job)

    def _props_of(self, job: Job, panel: PanelV2) -> list[tuple[str, ImageContent]]:
        """(symbol, reference picture) of the book symbols this picture's plan draws."""
        from services.storyboarder_v2 import drawn_symbols
        rt = job.rt
        if not rt.props or rt.storyboard is None:
            return []
        return [(sym, rt.props[sym]) for sym in drawn_symbols(rt.storyboard, f"{panel.visual} {panel.shows}") if sym in rt.props]

    def _panel_entry(self, job: Job, n: int) -> dict:
        return next(p for p in job.state["panels"] if p["n"] == n)

    def _settle_failed_group(self, job: Job, group: list[PanelV2], snapshot: list[dict], error: str) -> None:
        """A render that published no new picture: panels that had one keep it and report the failed redraw,
        the others are marked failed."""
        had_image = any(sn.get("status") == "done" and sn.get("image_url") for sn in snapshot)
        for p, sn in zip(group, snapshot):
            entry = self._panel_entry(job, p.n)
            if had_image:
                entry.update({k: sn.get(k) for k in RESTORE_KEYS})
                entry["error"] = f"redraw failed: {error}"
            else:
                entry["status"], entry["error"] = "failed", error

    def _update_progress(self, job: Job) -> None:
        s = job.state
        done = sum(1 for p in s["panels"] if p["status"] in ("done", "failed"))
        total = max(1, len(s["panels"]))
        s["progress"] = {"done": done, "total": len(s["panels"]), "percent": int(100 * done / total)}
        self._persist(job)

    @staticmethod
    def _decode_image(image: ImageContent) -> tuple[bytes, int, int]:
        if image.is_url or not image.is_base64:
            raise ValueError("engine returned an image URL instead of inline data")
        raw = base64.b64decode(image.data)
        with Image.open(io.BytesIO(raw)) as im:
            im.load()   # a full decode: verify() skips JPEG scan data, so a truncated JPEG would pass and break assembly
            w, h = im.size
        return raw, w, h

    @staticmethod
    def _write_atomic(path: Path, data: bytes) -> None:
        """Write through a (uniquely named) temp file, so a failed write never destroys the previous good file."""
        tmp = path.with_name(f"{path.name}.{secrets.token_hex(4)}.tmp")
        try:
            tmp.write_bytes(data)
            os.replace(tmp, path)
        except BaseException:
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)
            raise

    @staticmethod
    def _attempt_rank(qa: Optional[QAResult], score: float) -> tuple:
        """Higher is better. A real verdict beats "qa unavailable" (an unchecked retry must not replace a checked
        attempt), then: all text exact, characters, layout, number of exact lines, score."""
        if qa is None:   # QA off: the first image is kept
            return (True, True, True, True, 0, score)
        return (getattr(qa, "available", True), qa.text_exact == qa.text_total, qa.characters_ok, qa.layout_ok, qa.text_exact, score)

    def _drawing_message(self, job: Job, group: list[PanelV2], attempt: int) -> str:
        retry = f" (retry {attempt})" if attempt > 1 else ""
        if job.state.get("redraw_panels"):   # a user's redraw must not read like a resumed run
            return f"Redrawing {self._panels_label([p.n for p in group])}{retry}"
        span = f"{group[0].n}" + (f"-{group[-1].n}" if len(group) > 1 else "")
        return f"Drawing panel {span} of {len(job.state['panels'])}{retry}"

    @staticmethod
    def _panels_label(numbers: list[int]) -> str:
        return f"panel {numbers[0]}" if len(numbers) == 1 else f"panels {numbers[0]}-{numbers[-1]}"

    async def _render_group(self, job: Job, group: list[PanelV2], settings: dict) -> None:
        if int(job.state.get("version") or 2) >= 3:
            await self._render_panel_v3(job, group[0], settings)
            return
        s, rt = job.state, job.rt
        entries = [self._panel_entry(job, p.n) for p in group]
        snapshot = [dict(e) for e in entries]           # restored if every attempt fails on a regenerate
        base_attempts = max(int(e.get("attempts") or 0) for e in entries)
        max_attempts = settings["max_attempts"]
        best: Optional[tuple] = None  # (rank, score, RenderResult, QAResult | None, raw, w, h, line texts)
        # what the attempts draw: the script, plus lines rewritten after repeated lettering failures; the script itself
        # only takes the texts of the attempt that is published
        work = copy.deepcopy(group)
        script = _line_texts(group)
        fails: dict[tuple, list[str]] = {}
        rewrites: list[tuple[str, str, list[str]]] = []   # (old line, new line, how the old line was misdrawn)
        feedback = None
        last_error = None
        made = 0
        for attempt in range(1, max_attempts + 1):
            if rt.cancel.is_set():
                break
            made = attempt
            for e in entries:   # `attempts` (the reader's image cache key) changes only when a new picture is published
                e["status"], e["error"] = "rendering", None
            s["stage_message"] = self._drawing_message(job, group, attempt)
            self._persist(job)
            try:
                anchor = rt.anchor if settings["style_anchor"] and len(group) == 1 else None
                result = await (rt.renderer.render_panel(work[0], feedback=feedback, anchor=anchor) if len(work) == 1
                                else rt.renderer.render_strip(work, feedback=feedback))
                raw, w, h = self._decode_image(result.image)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                if getattr(e, "account", False):   # an inactive model or no balance: every other panel would fail the same way
                    raise
                last_error = f"{type(e).__name__}: {str(e)[:300]}"
                print(f"[Job {job.id}] panel {[p.n for p in group]} attempt {attempt} failed: {last_error}")
                if "PolicyViolation" in last_error or "SensitiveContent" in last_error:
                    feedback = ("the previous attempt was blocked by the content filter for possible similarity to copyrighted material: "
                                + ("draw only the original characters from the reference sheets" if rt.sheets else
                                   "draw simple original mascots that follow the descriptions above")
                                + ", with no logos, brands, celebrities or existing cartoon characters")
                continue
            s["cost"]["images_generated"] += 1
            s["cost"]["image_tokens"] += int((result.usage or {}).get("output_tokens", 0) or 0)
            qa: Optional[QAResult] = None
            texts = _line_texts(work)
            score = 10.0
            if rt.qa is not None:
                for e in entries:
                    e["status"] = "qa"
                self._persist(job)
                qa = await rt.qa.judge(result.image, work)
                self._add_text_usage(job, rt.qa.usage)
                rt.qa.usage = {"prompt_tokens": 0, "completion_tokens": 0}
                if qa.needs_retry and qa.available:
                    adoptions, qa = await self._check_drawn_wording(job, work, qa)
                    for a in adoptions:   # later attempts letter the adopted wording, so prompt, judge and feedback agree
                        texts[(a["n"], a["kind"], a["index"])] = a["drawn"]
                        _set_line(work, (a["n"], a["kind"], a["index"]), a["drawn"])
                score = qa.score
                print(f"[Job {job.id}] panel {[p.n for p in group]} attempt {attempt}: score={qa.score} text={qa.text_exact}/{qa.text_total} retry={qa.needs_retry} issues={qa.issues[:2]}")
            rank = self._attempt_rank(qa, score) + (texts == script,)   # on a tie the script's own wording wins
            if best is None or rank >= best[0]:
                best = (rank, score, result, qa, raw, w, h, texts)
            if qa is None or not qa.needs_retry:
                break
            if attempt < max_attempts:
                rewrites += await self._rephrase_repeated_failures(job, work, qa, fails)
            feedback = self._retry_feedback(qa, rewrites)

        if best is None:
            self._settle_failed_group(job, group, snapshot, last_error or ("cancelled" if rt.cancel.is_set() else "generation failed"))
            if made:
                for e in entries:
                    e["attempts"], e["render_attempts"] = base_attempts + made, made
            return
        _rank, score, result, qa, raw, w, h, texts = best
        if qa is not None:   # a rewritten or adopted line enters the script only if the published picture letters it exactly
            missed = {(group[m["panel"] - 1].n, m["kind"], m["index"]) for m in qa.mismatches if 0 < m["panel"] <= len(group)}
            texts = {key: script[key] if key in missed and key in script else text for key, text in texts.items()}
        ext = "png" if result.image.mime_type == "image/png" else "jpg"
        name = f"panel_{group[0].n:02d}.{ext}" if len(group) == 1 else f"strip_{group[0].n:02d}-{group[-1].n:02d}.{ext}"
        self._write_atomic(job.dir / name, raw)   # the picture first: a failed write must leave the script untouched
        if texts != script and not self._apply_texts(job, group, texts):
            qa = self._unsynced(qa, script, texts)
        self._sync_entries(job, group)
        for i, e in enumerate(entries):
            e.update({"status": "done", "image_url": f"/output/{job.id}/{name}", "width": w if len(group) == 1 else w // 2, "height": h,
                      "qa": qa.to_public() if qa else None, "error": None, "attempts": base_attempts + made, "render_attempts": made})
            if len(group) > 1:
                e.update({"strip_index": i, "strip_width": w, "strip_height": h})
        if settings["style_anchor"] and rt.anchor is None and len(group) == 1:
            rt.anchor = result.image
        try:   # prompts.jsonl is a log: failing to append to it must not undo a published picture
            with (job.dir / "prompts.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(_clean({"panels": [p.n for p in group], "attempt_score": score, "seconds": result.seconds,
                                            "prompt": result.prompt}), ensure_ascii=False) + "\n")
        except OSError as e:
            print(f"[Job {job.id}] could not log the prompt of panel {[p.n for p in group]}: {e}")

    async def _render_panel_v3(self, job: Job, panel: PanelV2, settings: dict) -> None:
        """Text-free art → the vision model locates the cast and the blank board and lists art problems → retry with those
        problems as feedback → the program letters the best attempt (bubbles, narration, charts) in the job language."""
        from services import art_check, lettering
        s, rt = job.state, job.rt
        entry = self._panel_entry(job, panel.n)
        snapshot = dict(entry)
        base_attempts = int(entry.get("attempts") or 0)
        # a teaching scene gets one try more: its picture must also show the idea (the first picture books left 5-11
        # scenes per book with the idea unclear after 3 tries)
        teaching = bool(art_shows(panel)) and not art_check.is_board_panel(panel)
        best, made, last_error = await self._draw_attempts(job, panel, settings["max_attempts"] + (1 if teaching else 0))
        frames = None
        if best is not None and teaching and not rt.cancel.is_set() and any(str(p).startswith(art_check.IDEA_LABELS) for p in best[5]):
            framed, extra = await self._two_frames(job, panel, best)
            made += extra
            if framed is not None:
                best, frames = framed
        if best is None:
            self._settle_failed_group(job, [panel], [snapshot], last_error or ("cancelled" if rt.cancel.is_set() else "generation failed"))
            if made:
                entry["attempts"], entry["render_attempts"] = base_attempts + made, made
            return
        _rank, result, raw, art, grounding, problems, idea_unchecked = best
        art_name = f"art_{panel.n:02d}.jpg"
        self._write_atomic(job.dir / art_name, raw)   # the drawing as Seedream made it: re-lettering never pays for art again
        self._write_atomic(job.dir / f"grounding_{panel.n:02d}.json",   # ...nor for the vision call that located the cast
                           json.dumps(asdict(grounding) if grounding is not None else None, ensure_ascii=False).encode("utf-8"))
        languages = s.get("languages") or [s["language"]]
        lettered: dict[str, tuple] = {}
        try:
            for lang in languages:   # the same art in every language: only the words change
                lettered[lang] = await asyncio.to_thread(lettering.letter_panel, art, panel if lang == s["language"] else
                                                         self._panel_in(job, lang, panel.n), grounding, lang, rt.fonts)
        except Exception as e:  # noqa: BLE001  (the art is saved: a later re-lettering can still use it)
            self._settle_failed_group(job, [panel], [snapshot], f"lettering failed: {type(e).__name__}: {str(e)[:300]}")
            entry["attempts"], entry["render_attempts"] = base_attempts + made, made
            return
        urls = self._write_lettered(job, panel.n, lettered)
        main, layout = lettered[s["language"]]
        if len(urls) > 1:
            entry["image_urls"] = urls
        entry.update({"status": "done", "image_url": urls[s["language"]], "art_url": f"/output/{job.id}/{art_name}",
                      "width": main.width, "height": main.height, "layout": layout, "error": None,
                      "qa": {"available": True, "score": 10.0 if not problems else max(1.0, 9.0 - 2 * len(problems)),
                             "text_exact": None, "text_total": None, "issues": problems[:6]},
                      "attempts": base_attempts + made, "render_attempts": made})
        # a picture drawn as two frames says so (their plans and problems); a redraw as one picture drops the record
        two = s.setdefault("quality", {}).setdefault("two_frames", [])
        if frames:
            entry["frames"] = [{k: v for k, v in f.items() if k != "prompt"} for f in frames]
            if panel.n not in two:
                two.append(panel.n)
        else:
            entry.pop("frames", None)
            if panel.n in two:
                two.remove(panel.n)
        # the job record lists the published teaching pictures whose idea was never judged (the check failed, or the art
        # could not be located at all); a redraw whose idea was judged leaves the list
        unjudged = s.setdefault("quality", {}).setdefault("unchecked_idea", [])
        if idea_unchecked or (grounding is None and art_shows(panel) and not art_check.is_board_panel(panel)):
            if panel.n not in unjudged:
                unjudged.append(panel.n)
        elif panel.n in unjudged:
            unjudged.remove(panel.n)
        try:   # prompts.jsonl is a log: failing to append to it must not undo a published picture
            with (job.dir / "prompts.jsonl").open("a", encoding="utf-8") as fh:
                for prompt in [f["prompt"] for f in frames] if frames else [result.prompt]:
                    fh.write(json.dumps(_clean({"panels": [panel.n], "problems": problems, "seconds": result.seconds, "prompt": prompt}),
                                        ensure_ascii=False) + "\n")
        except OSError as e:
            print(f"[Job {job.id}] could not log the prompt of panel {panel.n}: {e}")

    async def _draw_attempts(self, job: Job, panel: PanelV2, tries: int, size: Optional[tuple[int, int]] = None,
                             what: str = "") -> tuple[Optional[tuple], int, Optional[str]]:
        """Up to ``tries`` drawings of one picture (``panel``'s art, or one frame of it at ``size``): each is located and
        checked, each retry is told the last one's problems. Returns (the best attempt or None, drawings asked for, the
        last error); the best is (rank, result, raw, art, grounding, problems, idea_unchecked)."""
        from services import art_check
        s, rt = job.state, job.rt
        entry = self._panel_entry(job, panel.n)
        best: Optional[tuple] = None
        feedback, last_error, made = None, None, 0
        props = self._props_of(job, panel)
        for attempt in range(1, tries + 1):
            if rt.cancel.is_set():
                break
            made = attempt
            entry["status"], entry["error"] = "rendering", None
            s["stage_message"] = self._drawing_message(job, [panel], attempt)
            self._persist(job)
            try:
                result = await rt.renderer.render_panel(panel, feedback=feedback, size=size, props=props)
                raw, _w, _h = self._decode_image(result.image)
                art = Image.open(io.BytesIO(raw)).convert("RGB")
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                if getattr(e, "account", False):   # an inactive model or no balance: every other panel would fail the same way
                    raise
                last_error = f"{type(e).__name__}: {str(e)[:300]}"
                print(f"[Job {job.id}] panel {panel.n}{what} attempt {attempt} failed: {last_error}")
                if "PolicyViolation" in last_error or "SensitiveContent" in last_error:
                    feedback = ("the previous attempt was blocked by the content filter for possible similarity to copyrighted material: "
                                + ("draw only the original characters from the reference sheets" if rt.sheets else
                                   "draw simple original mascots that follow the descriptions above")
                                + ", with no logos, brands, celebrities or existing cartoon characters")
                continue
            s["cost"]["images_generated"] += 1
            s["cost"]["image_tokens"] += int((result.usage or {}).get("output_tokens", 0) or 0)
            entry["status"] = "qa"
            self._persist(job)
            usage: dict = {}
            unchecked = idea_unchecked = False
            try:
                grounding = await art_check.locate(rt.engines.vision, result.image, rt.cast, panel.characters, rt.sheets,
                                                   usage=usage, size=art.size)
                problems = art_check.art_problems(grounding, panel, art)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001  (a new drawing cannot fix the vision call: keep this paid art, lettered blind)
                if getattr(e, "account", False):   # the vision model is not usable at all: stop instead of lettering every panel blind
                    raise
                print(f"[Job {job.id}] panel {panel.n}{what} attempt {attempt}: the art could not be checked: {type(e).__name__}: {str(e)[:200]}")
                grounding, problems, unchecked = None, [f"unchecked: the art could not be checked ({type(e).__name__})"], True
                quality = s.setdefault("quality", {}).setdefault("unchecked_art", [])
                if panel.n not in quality:
                    quality.append(panel.n)
            shows = art_shows(panel)   # a teaching scene's idea exactly as the art prompt asked for it
            if grounding is not None and shows and not art_check.is_board_panel(panel):
                try:   # a teaching scene: does the picture alone show its idea? (a failed verdict never blocks the art)
                    idea, grounding.objects = await art_check.check_idea(rt.engines.vision, result.image, shows, art.size, usage=usage)
                    if idea:
                        problems.append(idea)
                except asyncio.CancelledError:
                    raise
                except Exception as e:  # noqa: BLE001
                    if getattr(e, "account", False):
                        raise
                    print(f"[Job {job.id}] panel {panel.n}{what} attempt {attempt}: the idea check failed: {type(e).__name__}: {str(e)[:200]}")
                    idea_unchecked = True
            self._add_text_usage(job, usage)
            # an idea never judged ranks like an unclear one: a real "yes" beats it, a real "no" does not (no redraw is paid for it)
            rank = art_check.rank_key(problems + ([f"{art_check.IDEA_LABELS[1]} (not checked)"] if idea_unchecked else []))
            print(f"[Job {job.id}] panel {panel.n}{what} attempt {attempt}: problems={problems[:3]}")
            if best is None or rank > best[0]:
                best = (rank, result, raw, art, grounding, problems, idea_unchecked)
            if not problems or unchecked:
                break
            feedback = "; ".join(problems[:4])
        return best, made, last_error

    async def _two_frames(self, job: Job, panel: PanelV2, single: tuple) -> tuple[Optional[tuple], int]:
        """A teaching scene whose best picture still misses its idea, drawn as two frames stacked in its panel (see
        panel_renderer.TWO_FRAMES_PROMPT): the writer plans them, each frame gets up to FRAME_TRIES drawings with the same
        checks, and the two best are stacked at the single picture's size. Returns ((best, frames) or None, drawings asked
        for); None keeps the single picture: no plan, a frame without a drawing, or frames with more character, figure or
        drawn-text problems than it."""
        from services import art_check
        from services.panel_renderer import frame_size, parse_two_frames, two_frames_prompt
        from services.storyboarder_v2 import _stages
        rt = job.rt
        art, problems = single[3], single[5]
        config = GenerationConfig(temperature=0.4, max_tokens=4000, response_format={"type": "json_object"})
        try:
            response = await rt.engines.text.generate_text([Message.user(two_frames_prompt(panel, problems))], config)
            self._add_text_usage(job, response.usage or {})
            parts = parse_two_frames(extract_json(response.content or ""))
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001  (the single picture stays)
            if getattr(e, "account", False):
                raise
            print(f"[Job {job.id}] panel {panel.n}: no two-frame plan: {type(e).__name__}: {str(e)[:200]}")
            return None, 0
        if not parts:
            print(f"[Job {job.id}] panel {panel.n}: no two-frame plan in the reply")
            return None, 0
        frames = []
        for part in parts:
            frame = copy.deepcopy(panel)
            frame.visual, frame.shows = part["visual"], part["shows"]
            frame.characters = [c for c in panel.characters if _stages(c, f"{frame.visual} {frame.shows}")]
            frame.emotions = {k: v for k, v in panel.emotions.items() if k in frame.characters}
            frames.append(frame)
        if not all(art_shows(f) for f in frames):   # a frame whose idea cannot be judged never replaces a judged picture
            print(f"[Job {job.id}] panel {panel.n}: a frame of the plan has no idea to show; the single picture stays")
            return None, 0
        drawn, made = [], 0
        for where, frame in zip(("top", "bottom"), frames):
            best, n, _ = await self._draw_attempts(job, frame, FRAME_TRIES, size=frame_size(art.size), what=f" ({where} frame)")
            made += n
            if best is None or rt.cancel.is_set():
                return None, made
            drawn.append((where, frame, best))
        frame_problems = [f"{p} ({where} frame)" for where, _, b in drawn for p in b[5]]
        if art_check.rank_key(frame_problems)[:4] < art_check.rank_key(problems)[:4]:
            print(f"[Job {job.id}] panel {panel.n}: the two frames have worse problems than the single picture; it stays")
            return None, made
        composite, grounding = art_check.compose_frames([b[3] for _, _, b in drawn], [b[4] for _, _, b in drawn], art.size)
        buf = io.BytesIO()
        composite.save(buf, format="JPEG", quality=92)
        best = (art_check.rank_key(frame_problems), drawn[-1][2][1], buf.getvalue(), composite, grounding, frame_problems,
                any(b[6] for _, _, b in drawn))
        frames = [{"where": where, "visual": f.visual, "shows": f.shows, "characters": list(f.characters), "issues": b[5][:4],
                   "prompt": b[1].prompt} for where, f, b in drawn]
        print(f"[Job {job.id}] panel {panel.n}: drawn as two frames, problems={frame_problems[:3]}")
        return (best, frames), made

    def _write_lettered(self, job: Job, n: int, lettered: dict) -> dict:
        """Write panel n's lettered pictures ({language: (image, layout)}); returns {language: url}. The job's own language
        is panel_NN.jpg, another one panel_NN.<lang>.jpg (with its layout_NN.<lang>.json)."""
        urls = {}
        for lang, (img, lay) in lettered.items():
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=92)
            name = f"panel_{n:02d}.jpg" if lang == job.state["language"] else f"panel_{n:02d}.{lang}.jpg"
            self._write_atomic(job.dir / name, buf.getvalue())
            if len(lettered) > 1:
                self._write_atomic(job.dir / f"layout_{n:02d}.{lang}.json", json.dumps(lay, ensure_ascii=False).encode("utf-8"))
            urls[lang] = f"/output/{job.id}/{name}"
        return urls

    async def _recap_evidence(self, job: Job, english) -> None:
        """Which chart or table shows each recap line, for a job built before the build asked (StoryboarderV2.recap_evidence:
        one text call, its cost counted, the verdict kept in the script's brief); the program's rule decides when the call
        cannot run. Without it a relayout re-marked every line by the rule alone: 342 ticks went in 144 saved jobs, "Found a
        strong Artificial Hivemind effect" among them, beside its two charts."""
        from services.storyboarder_v2 import StoryboarderV2
        own = job.rt.engines is None
        try:
            engines = job.rt.engines or build_engines()
        except Exception as e:  # noqa: BLE001  (no key, no config: the rule decides)
            print(f"[Job {job.id}] recap evidence skipped: {type(e).__name__}: {e}")
            return
        try:
            writer = StoryboarderV2(engines.text, engines.fast_text, engines.check, engines.translate, cache_key=f"prism-{job.id}")
            await writer.recap_evidence(english)
            self._add_text_usage(job, writer.usage)
        except Exception as e:  # noqa: BLE001  (an account error, a broken reply: the rule decides)
            print(f"[Job {job.id}] recap evidence skipped: {type(e).__name__}: {e}")
        finally:
            if own:
                try:
                    await engines.close()
                except Exception:  # noqa: BLE001
                    pass

    async def relayout(self, job_id: str) -> Optional[dict]:
        """Re-letter every drawn panel of a finished v3 job from its saved art and grounding with the current lettering and
        board code, in every language: no image or vision call. A re-lettered panel publishes a new picture, so its
        ``attempts`` (the reader's image cache key) goes up; ``render_attempts`` and the art stay."""
        job = self._load(job_id)
        if not job:
            return None
        if job.state["status"] not in TERMINAL or job.running or job.rt.relayout_lock.locked():
            raise ValueError("Job is still running")
        # No await separates the busy check from acquiring this uncontended lock. Regenerate likewise
        # registers its task before yielding, so neither operation can pass the other's guard. Reject
        # competing requests instead of queueing stale relayouts; exceptions/cancellation release it.
        async with job.rt.relayout_lock:
            return await self._relayout(job)

    async def _relayout(self, job: Job) -> dict:
        """Re-letter while the caller holds this job's relayout lock (single API process only)."""
        s, rt = job.state, job.rt
        if int(s.get("version") or 2) < 3:
            raise ValueError("Only v3 jobs keep their art apart from the lettering")
        from services import lettering
        if rt.storyboard is None:
            rt.storyboard = _storyboard_from_dict(json.loads((job.dir / "storyboard.json").read_text(encoding="utf-8")))
        self._load_scripts(job)
        from services.storyboarder_v2 import _honest_recap, copy_recap_marks
        # the recap's ticks follow the current rule too, judged on the English script and copied line by line to the
        # translations, as a build does; a job built before the chart-evidence check gets its verdict once (one text
        # call, kept in the script), after that the rule needs no model call
        scripts = {lang: x for lang, x in [(s["language"], rt.storyboard), *rt.storyboards.items()] if x is not None}
        english = scripts.get("en-US")
        if english is not None and "recap_evidence" not in english.brief:
            await self._recap_evidence(job, english)
        if english is not None:
            _honest_recap(english)
        for lang, script in scripts.items():
            if script is english:
                continue
            if english is not None:
                copy_recap_marks(english, script, lang)
            else:
                _honest_recap(script, retitle=False)
        if any(script.brief.get("reader_contract") == "novice_v1" for script in scripts.values()):
            s.setdefault("quality", {})["reader_reviews"] = _public_reader_reviews(scripts)
        self._write_atomic(job.dir / "storyboard.json", json.dumps(_clean(rt.storyboard.to_dict()), ensure_ascii=False,
                                                                     indent=1).encode("utf-8"))
        for lang, other in rt.storyboards.items():
            if lang != s["language"] and other is not None:
                self._write_atomic(job.dir / f"storyboard.{lang}.json", json.dumps(_clean(other.to_dict()), ensure_ascii=False,
                                                                                  indent=1).encode("utf-8"))
        for entry in s["panels"]:   # the reader's text under each picture follows the scripts (a retitled recap, its marks)
            try:
                entry["board"] = self._panel_in(job, s["language"], entry["n"]).board
                if entry.get("texts"):
                    entry["texts"] = {lang: _panel_texts(self._panel_in(job, lang, entry["n"])) for lang in entry["texts"]}
            except KeyError:
                continue
        fonts, languages, count = lettering.FontSet(), s.get("languages") or [s["language"]], 0
        for entry in s["panels"]:
            art_path = job.dir / f"art_{entry['n']:02d}.jpg"
            if entry.get("status") != "done" or not art_path.exists():
                continue
            with Image.open(art_path) as im:
                art = im.convert("RGB")
            grounding = _grounding_from_file(job.dir / f"grounding_{entry['n']:02d}.json")
            lettered = {lang: await asyncio.to_thread(lettering.letter_panel, art, self._panel_in(job, lang, entry["n"]), grounding,
                                                      lang, fonts) for lang in languages}
            urls = self._write_lettered(job, entry["n"], lettered)
            main, layout = lettered[s["language"]]
            entry.update({"image_url": urls[s["language"]], "width": main.width, "height": main.height, "layout": layout,
                          "attempts": int(entry.get("attempts") or 0) + 1})
            if len(urls) > 1:
                entry["image_urls"] = urls
            count += 1
        if count:
            await self._stitch(job)
        self._persist(job)
        return {"relettered": count}

    async def _check_drawn_wording(self, job: Job, group: list[PanelV2], qa: QAResult) -> tuple[list[dict], QAResult]:
        """Seedream redraws some rare words as a common synonym every time (莫大な -> 巨大な), so retrying cannot fix them.
        A drawn wording that both readers agree on and that passes wording_gate is put to the text model; the accepted
        ones are returned with a copy of the verdict that counts them as exact. Nothing is changed here: the wording only
        enters the script if this attempt is the one published (_apply_texts)."""
        rt = job.rt
        accepted = []
        for m in (agreed_alternatives(qa) if rt.engines is not None else []):
            panel = group[m["panel"] - 1] if 0 < m["panel"] <= len(group) else None
            if panel is None or (m["kind"] == "bubble" and not (m["index"] is not None and m["index"] < len(panel.dialogue))):
                continue
            drawn = adopt_wording(m["expected"], m["judge"])
            if drawn != m["expected"] and await self._meaning_check(job, panel, m["expected"], drawn, ADOPTED):
                accepted.append({**m, "n": panel.n, "drawn": drawn})
        if not accepted:
            return [], qa
        adjusted = copy.deepcopy(qa)
        for a in accepted:
            old, new = a["expected"], a["drawn"]
            changes = [(_canon(old)[i1:i2], _canon(new)[j1:j2])
                       for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, _canon(old), _canon(new), autojunk=False).get_opcodes()
                       if tag != "equal"]

            def about_the_change(issue: str) -> bool:   # "uses 巨大 instead of the expected 莫大", "must read exactly <old>"
                c = _canon(issue)
                return f'must read exactly "{old}"' in issue or any(x and y and x in c and y in c for x, y in changes)

            def renamed(issue: str) -> str:   # a wrong tail still applies, now to the adopted line
                return issue.replace(f'"{old[:40]}"', f'"{new[:40]}"').replace(f'"{old}"', f'"{new}"')

            adjusted.issues = [renamed(i) for i in adjusted.issues if not about_the_change(i)]
            adjusted.layout_issues = [renamed(i) for i in adjusted.layout_issues if not about_the_change(i)]
            adjusted.mismatches = [m for m in adjusted.mismatches if (m["panel"], m["kind"], m["index"]) != (a["panel"], a["kind"], a["index"])]
            adjusted.text_exact = min(adjusted.text_total, adjusted.text_exact + 1)
        if not adjusted.layout_issues:   # the notes that failed the layout were all about the adopted wording
            adjusted.layout_ok = True
        adjusted.needs_retry = (adjusted.text_exact < adjusted.text_total or not adjusted.characters_ok or not adjusted.layout_ok
                                or adjusted.low_score)
        return accepted, adjusted

    async def _meaning_check(self, job: Job, panel: PanelV2, old: str, new: str, situation: str) -> bool:
        """The main text model must confirm that ``new`` is correct and means exactly what ``old`` means."""
        rt, s = job.rt, job.state
        context = [b.text for b in panel.dialogue] + ([panel.narration] if panel.narration else [])
        prompt = CHECK_PROMPT.format(language=LANG_NAMES.get(s["language"], s["language"]), situation=situation, expected=old, drawn=new,
                                     context=json.dumps([t for t in context if t != old], ensure_ascii=False))
        config = GenerationConfig(temperature=0.0, max_tokens=200)
        config.thinking = False
        try:
            response = await rt.engines.text.generate_text([Message.user(prompt)], config)
            self._add_text_usage(job, response.usage)
            verdict = extract_json(response.content)
        except Exception as e:  # noqa: BLE001
            print(f"[Job {job.id}] wording check failed for panel {panel.n}: {type(e).__name__}: {e}")
            return False
        return isinstance(verdict, dict) and verdict.get("accept") is True

    @staticmethod
    def _retry_feedback(qa: QAResult, rewrites: list[tuple]) -> Optional[str]:
        """The problems for the next attempt, minus any that quote a rewritten line or how it was misdrawn (in any quote
        style), which would put the old wording back into the prompt; a wrong tail still applies, now to the new line."""
        out = []
        for issue in qa.issues:
            c = _canon(issue)
            hits = [(old, new) for old, new, misdrawn in rewrites
                    if any(len(_canon(x)) >= 4 and _canon(x)[:30] in c for x in [old, *misdrawn])]
            if not hits:
                out.append(issue)
            elif issue.startswith("the tail of"):
                for old, new in hits:
                    issue = issue.replace(f'"{old[:40]}"', f'"{new[:40]}"')
                out.append(issue)
        return "; ".join(out[:4]) if out else None

    async def _rephrase_repeated_failures(self, job: Job, work: list[PanelV2], qa: QAResult, fails: dict) -> list[tuple]:
        """A line misdrawn the same way in two attempts in a row is a word Seedream cannot letter: it is rewritten in more
        common words for the next attempt (same facts, confirmed by the text model). Random glitches are left to the
        retry, and English is never rewritten (its lettering errors are random typos). Returns (old, new, misdrawn)."""
        if job.state["language"] == "en-US":
            return []
        current = {}
        for m in qa.mismatches:
            if 0 < m["panel"] <= len(work):
                current[(work[m["panel"] - 1].n, m["kind"], m["index"])] = (m["expected"], m.get("blind") or m.get("judge") or "")
        for key in [k for k in fails if k not in current]:
            del fails[key]
        out = []
        for key, (expected, drawn) in current.items():
            fails.setdefault(key, []).append((drawn, _failing_chars(expected, drawn)))
            seen, old = fails[key], _line_texts(work).get(key)
            # systematic: the same characters failed both times (掩 drawn as 密, then as 掎), not glitches in different places
            repeated = seen[-1][1] & seen[-2][1] if len(seen) >= 2 else set()
            if not repeated or not old:
                continue
            panel = next(p for p in work if p.n == key[0])
            new = await self._rephrase(job, panel, old, [d for d, _chars in seen[-2:] if d][:1], "".join(sorted(repeated)))
            if new:
                _set_line(work, key, new)
                out.append((old, new, [d for d, _chars in seen if d]))
                fails[key] = []
                print(f"[Job {job.id}] panel {key[0]}: rewrote {old!r} as {new!r} after {''.join(sorted(repeated))!r} failed twice")
        return out

    async def _rephrase(self, job: Job, panel: PanelV2, old: str, drawn: list[str], chars: str = "") -> Optional[str]:
        rt, s = job.rt, job.state
        if rt.engines is None:
            return None
        language = s["language"]
        prompt = REPHRASE_PROMPT.format(language=LANG_NAMES.get(language, language), expected=old,
                                        drawn=json.dumps(drawn, ensure_ascii=False) if drawn else "(nothing readable)",
                                        chars=chars or "(see above)")
        config = GenerationConfig(temperature=0.2, max_tokens=300)
        config.thinking = False
        try:
            response = await rt.engines.text.generate_text([Message.user(prompt)], config)
            self._add_text_usage(job, response.usage)
            reply = extract_json(response.content)
        except Exception as e:  # noqa: BLE001
            print(f"[Job {job.id}] rewrite failed: {type(e).__name__}: {e}")
            return None
        new = str(reply.get("text") or "").strip() if isinstance(reply, dict) else ""
        critical = lambda t: sorted(c for c in t if c in MEANING_CRITICAL)
        if not new or new == old or "_" in new or "\\" in new or not same_facts(old, new) or not _has_script(new, language) \
                or critical(old) != critical(new) or len(new) > len(old) + max(2, len(old) // 8):
            print(f"[Job {job.id}] panel {panel.n}: rewrite {new!r} of {old!r} refused by the guards")
            return None
        if not await self._meaning_check(job, panel, old, new, REWRITTEN):
            print(f"[Job {job.id}] panel {panel.n}: rewrite {new!r} of {old!r} refused by the meaning check")
            return None
        return new

    def _apply_texts(self, job: Job, group: list[PanelV2], texts: dict) -> bool:
        """Give the script the lines the published picture was checked against (adopted or rewritten wording), in
        storyboard.json and the job state in one step with no await, so the reader's text matches the picture; on a
        failed write nothing changes."""
        rt, s = job.rt, job.state
        before = _line_texts(group)
        changes = [(key, before[key], text) for key, text in texts.items() if key in before and before[key] != text]
        if not changes:
            return True
        for key, _old, new in changes:
            _set_line(group, key, new)
        try:
            data = json.dumps(_clean(rt.storyboard.to_dict()), ensure_ascii=False, indent=1).encode("utf-8")
            self._write_atomic(job.dir / "storyboard.json", data)
        except Exception as e:  # noqa: BLE001
            for key, old, _new in changes:
                _set_line(group, key, old)
            print(f"[Job {job.id}] could not save the published wording to storyboard.json: {type(e).__name__}: {e}")
            return False
        s.setdefault("text_adjustments", []).extend({"panel": key[0], "from": old, "to": new} for key, old, new in changes)
        print(f"[Job {job.id}] published wording: {[(old, new) for _key, old, new in changes]}")
        return True

    @staticmethod
    def _unsynced(qa: Optional[QAResult], script: dict, texts: dict) -> Optional[QAResult]:
        """The script could not take the published wording: those lines do not match the reader's text, so they are not exact."""
        if qa is None:
            return None
        out = copy.deepcopy(qa)
        for key, new in texts.items():
            if key in script and script[key] != new:
                out.text_exact = max(0, out.text_exact - 1)
                out.issues.insert(0, f'the picture letters "{new}" but the script still reads "{script[key]}"')
        out.needs_retry = True
        return out

    def _sync_entries(self, job: Job, group: list[PanelV2]) -> None:
        """The reader's selectable text is whatever the published picture was checked against (idempotent)."""
        for p in group:
            entry = self._panel_entry(job, p.n)
            entry["dialogue"] = [{"speaker": b.speaker, "text": b.text} for b in p.dialogue]
            entry["narration"] = p.narration

    # ------------------------------------------------------------------ assembly
    def _panel_images(self, job: Job, language: Optional[str] = None) -> list[Image.Image]:
        images = []
        for p in sorted(job.state["panels"], key=lambda x: x["n"]):
            url = (p.get("image_urls") or {}).get(language) or p["image_url"] if language else p["image_url"]
            if p["status"] != "done" or not url:
                continue
            path = job.dir / Path(url).name
            if not path.exists():
                continue
            try:
                with Image.open(path) as src:
                    im = src.convert("RGB")
            except (OSError, ValueError) as e:   # one unreadable file must not block the long image (or every later redraw)
                print(f"[Job {job.id}] panel {p['n']} left out of the long image: {type(e).__name__}: {e}")
                continue
            if p.get("strip_index") is not None:
                half = im.width // 2
                im = im.crop((half * p["strip_index"], 0, half * (p["strip_index"] + 1), im.height))
            images.append(im)
        return images

    def _assemble(self, job: Job, width: int = 1024, gap: int = 24, language: Optional[str] = None) -> Optional[bytes]:
        """The long image as JPEG bytes (None when nothing is drawn). This runs in a worker thread that a Stop cannot
        interrupt, so it touches neither job.state nor the disk: the caller publishes the result on success only."""
        images = self._panel_images(job, language)
        if not images:
            return None
        natural_h = sum(im.height * width / im.width for im in images) + gap * (len(images) - 1)
        if natural_h > JPEG_MAX_SIDE:   # JPEG cannot exceed 65,535 px: narrow the long image instead of failing
            width = max(64, int(width * (JPEG_MAX_SIDE - gap * (len(images) - 1)) / (natural_h - gap * (len(images) - 1))))
        resized = []
        for im in images:
            h = max(1, int(im.height * width / im.width))
            resized.append(im.resize((width, h), Image.LANCZOS))
        total_h = sum(im.height for im in resized) + gap * (len(resized) - 1)
        canvas = Image.new("RGB", (width, total_h), "white")
        y = 0
        for im in resized:
            canvas.paste(im, (0, y))
            y += im.height + gap
        buf = io.BytesIO()
        canvas.save(buf, format="JPEG", quality=90, optimize=True)
        return buf.getvalue()

    def _publish_combined(self, job: Job, data: Optional[bytes], language: Optional[str] = None) -> None:
        """Write combined.jpg and give combined_url a new key in one step on the event loop, so a key never serves two files.
        Another language's long image is combined.<lang>.jpg under combined_urls[lang]."""
        if not data:
            return
        name = f"combined.{language}.jpg" if language else "combined.jpg"
        self._write_atomic(job.dir / name, data)
        url = f"/output/{job.id}/{name}?v={int(time.time() * 1000)}"
        if language:
            job.state.setdefault("combined_urls", {})[language] = url
        else:
            job.state["combined_url"] = url

    async def _stitch(self, job: Job) -> None:
        """The long image of the job's language, then of each other language its panels are lettered in."""
        self._publish_combined(job, await asyncio.to_thread(self._assemble, job))   # a Stop discards the orphaned thread's result
        for lang in (job.state.get("languages") or [])[1:]:
            self._publish_combined(job, await asyncio.to_thread(self._assemble, job, language=lang), lang)

    # ------------------------------------------------------------------ regenerate
    async def _ensure_runtime(self, job: Job) -> None:
        rt, s = job.rt, job.state
        if rt.storyboard is None:
            rt.storyboard = _storyboard_from_dict(json.loads((job.dir / "storyboard.json").read_text(encoding="utf-8")))
        self._load_scripts(job)
        if rt.engines is None:
            rt.engines = build_engines()
            rt.renderer, rt.qa = None, None
        if rt.cast is None:
            rt.cast, rt.sheets = build_cast(s["theme"])
        if rt.renderer is None:
            self._build_renderers(job, _settings())

    def _load_scripts(self, job: Job) -> None:
        """The other languages' scripts of a bilingual v3 job (storyboard.<lang>.json), when they are not in memory."""
        s, rt = job.state, job.rt
        languages = s.get("languages") or []
        if len(languages) > 1 and set(rt.storyboards) != set(languages):
            rt.storyboards = {s["language"]: rt.storyboard}
            for lang in languages:
                path = job.dir / f"storyboard.{lang}.json"
                if lang != s["language"] and path.exists():
                    rt.storyboards[lang] = _storyboard_from_dict(json.loads(path.read_text(encoding="utf-8")))

    async def _regenerate(self, job: Job, n: int) -> None:
        s = job.state
        previous_status, previous_error = s["status"], s.get("error")
        redrawn = self._redraw_numbers(job, n)
        label = self._panels_label(redrawn)
        before = {m: dict(self._panel_entry(job, m)) for m in redrawn}
        phase = "setup"
        try:
            await self._ensure_runtime(job)
            s["status"], s["stage_message"], s["error"], s["redraw_panels"] = "rendering", f"Redrawing {label}", None, redrawn
            s["_before_redraw"] = {"status": previous_status, "error": previous_error}   # in job.json, not public: _load puts it back
            self._persist(job)
            phase = "draw"
            await self._render_all(job, only=[n])
            phase = "stitch"
            s["stage_message"] = "Stitching the long image"
            await self._stitch(job)
            pending = [p["n"] for p in s["panels"] if p["status"] not in ("done", "failed")]
            failed = [p["n"] for p in s["panels"] if p["status"] == "failed"]
            if pending:
                s["status"] = previous_status if previous_status in TERMINAL else "cancelled"
                failed_redraw = any(self._panel_entry(job, m).get("error") for m in redrawn)
                head = f"Redraw of {label} failed" if failed_redraw else f"{label.capitalize()} redrawn"
                s["stage_message"] = f"{head}; {len(pending)} panel(s) still not drawn"
            else:
                s["status"] = "completed"
                s["stage_message"] = "Done" if not failed else f"Done with {len(failed)} failed panel(s): {failed}"
        except (asyncio.CancelledError, JobCancelled):
            # Stop puts the job back as it was (a completed manga stays completed), and so every panel without a new picture
            for m, old in before.items():
                entry = self._panel_entry(job, m)
                if entry.get("attempts") == old.get("attempts"):
                    entry.update({k: old.get(k) for k in (*RESTORE_KEYS, "error")})
            self._settle_active_panels(job)
            s["status"] = previous_status if previous_status in TERMINAL else "cancelled"
            s["stage_message"] = f"Redraw of {label} stopped"
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            reason = f"{type(e).__name__}: {str(e)[:200]}"
            if phase == "stitch":   # the panel was drawn; only the long image is stale
                s["stage_message"] = f"{label.capitalize()} redrawn, but the long image could not be updated: {reason}"
            else:
                s["stage_message"] = f"Redraw of {label} failed: {reason}"
                for m, old in before.items():   # panels no render settled (e.g. the engines could not be built): say why on them
                    entry = self._panel_entry(job, m)
                    if entry["status"] in ("rendering", "qa") or (entry.get("attempts"), entry.get("error")) == (old.get("attempts"), old.get("error")):
                        entry.update({k: old.get(k) for k in RESTORE_KEYS})
                        entry["error"] = f"redraw failed: {reason}"
            self._settle_active_panels(job)
            s["status"] = previous_status
        finally:
            s.pop("redraw_panels", None)
            s.pop("_before_redraw", None)
            s["error"] = previous_error if s["status"] == "failed" else None
            self._persist(job)
            await self._close_runtime(job)


_manager: Optional[JobManager] = None


def get_job_manager() -> JobManager:
    global _manager
    if _manager is None:
        _manager = JobManager()
    return _manager
