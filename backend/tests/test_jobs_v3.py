"""The v3 job path end to end with fakes: text-free art → grounding → art problems as retry feedback → lettering."""

from __future__ import annotations

import asyncio
import json
import re

import pytest
from PIL import Image

from . import fakes
from .fakes import V3_PAPER_TEXT

LOCATE_MARK = "Find things in that panel only"
IDEA_MARK = "It is meant to show, so that a reader who reads no words understands it"
LIGHT = [(246, 244, 238), (240, 238, 232), (250, 248, 243)]   # board-white art, so blank-board checks pass


class GroundingModel:
    """Vision router: LOCATE prompts get every present character's head low in the panel and one blank board in the
    middle; ``hook(call_number, reply)`` may change the reply. Idea checks (a teaching scene's picture) say "yes" with one
    object box unless ``idea(call_number, prompt)`` returns another reply. Anything else goes to the v2 judge."""

    def __init__(self, judge, hook=None, idea=None):
        self.judge, self.hook, self.idea, self.located, self.ideas = judge, hook, idea, 0, []

    def __call__(self, prompt, messages, config):
        if IDEA_MARK in prompt:
            self.ideas.append(prompt)
            reply = {"shown": "yes", "missing": "", "objects": [[100, 450, 600, 750]]}
            return (self.idea(len(self.ideas), prompt) if self.idea else None) or reply
        if LOCATE_MARK not in prompt:
            return self.judge(prompt, messages, config)
        self.located += 1
        line = prompt.split("Characters who should be in the panel:", 1)[1].split("\n", 1)[0]
        line = line.split(" These cast members should NOT", 1)[0]
        names = [n for n in ("papi", "kumo", "nezu") if re.search(rf"\b{n}\b", line)]
        reply = {"characters": [{"name": n, "head": [60 + 300 * i, 780, 240 + 300 * i, 920], "matches_reference": True}
                                for i, n in enumerate(names)],
                 "boards": [{"corners": [[60, 300], [940, 300], [940, 720], [60, 720]]}],
                 "speech_bubbles": [], "text_or_symbols": [], "extra_creatures": []}
        return (self.hook(self.located, reply) if self.hook else None) or reply


@pytest.fixture
def v3_env(job_env, monkeypatch):
    job_env.settings.update(pipeline="v3", concurrency=1)
    job_env.vision.router = GroundingModel(job_env.judge)
    job_env.image.fixed_size = (512, 768)
    monkeypatch.setattr(fakes, "PALETTE", LIGHT)
    return job_env


@pytest.fixture
def v3_teaching_env(v3_env, monkeypatch):
    """A book whose scenes keep the writer's "shows" (the teaching pictures of rounds 3-22): the art stage judges each such
    picture against its idea, as it still does for an older book on a redraw. Round 23's books have none."""
    from services import storyboarder_v2
    monkeypatch.setattr(storyboarder_v2, "SCENE_SHOWS", True)
    return v3_env


def _file(env, state, url: str):
    assert url.startswith(f"/output/{state['job_id']}/")
    return env.output_dir / state["job_id"] / url.rsplit("/", 1)[1]


async def test_v3_job_letters_every_panel_and_keeps_the_raw_art(v3_env):
    state = await v3_env.run(text=V3_PAPER_TEXT)

    assert state["status"] == "completed", state.get("error")
    assert (state["version"], state["layout"]) == (3, "single")
    assert state["quiz"]["score"] == 1.0 and state["quiz"]["questions"] > 0
    panels = state["panels"]
    assert len(panels) >= 9 and all(p["status"] == "done" for p in panels)
    kinds = {p["kind"] for p in panels}
    assert kinds == {"scene", "board"}
    for p in panels:
        art, lettered = _file(v3_env, state, p["art_url"]), _file(v3_env, state, p["image_url"])
        assert art.name == f"art_{p['n']:02d}.jpg" and lettered.name == f"panel_{p['n']:02d}.jpg"
        with Image.open(art) as a, Image.open(lettered) as b:
            assert a.size == (512, 768) and b.width == 512 and b.height >= 768
            assert (p["width"], p["height"]) == b.size
        assert isinstance(p["layout"], dict) and p["render_attempts"] == 1 and p["qa"]["issues"] == []
        saved = json.loads((art.parent / f"grounding_{p['n']:02d}.json").read_text(encoding="utf-8"))
        assert saved["size"] == [512, 768] and set(saved["heads"]) == set(p["characters"])   # re-lettering needs no vision call
        if p["kind"] == "board":
            assert p["layout"].get("chart") or p["layout"].get("board_error") is None
    # one picture and one grounding call per panel; every art prompt is text-free and carries no dialogue
    assert state["cost"]["images_generated"] == len(panels) == v3_env.vision.router.located
    prompts = [c["prompt"] for c in v3_env.image.calls]
    assert all("WORDLESS IMAGE" in pr for pr in prompts)
    lines = [b["text"] for p in panels for b in p["dialogue"]]
    assert lines and not any(line[:40] in pr for line in lines for pr in prompts)


async def test_v3_job_redraws_art_with_its_problems_as_feedback(v3_env):
    def hook(n, reply):
        if n == 1:   # the first panel's first drawing has numerals on a clock and a second kumo
            reply["text_or_symbols"] = [{"text": "12 3", "where": "the wall clock"}]
            reply["extra_creatures"] = ["a small bird"]
        return reply

    v3_env.vision.router.hook = hook
    state = await v3_env.run(text=V3_PAPER_TEXT)

    assert state["status"] == "completed", state.get("error")
    first = state["panels"][0]
    assert first["render_attempts"] == 2 and first["qa"]["issues"] == [] and first["qa"]["score"] == 10.0
    retry = v3_env.image.calls[1]["prompt"]
    assert "drawn text: remove every letter, digit and symbol (on the wall clock)" in retry
    assert "extra figure: remove a small bird" in retry and "12 3" not in retry
    assert all(p["render_attempts"] == 1 for p in state["panels"][1:])


async def test_v3_job_publishes_the_best_attempt_when_every_attempt_has_problems(v3_teaching_env):
    v3_env = v3_teaching_env
    v3_env.vision.router.hook = lambda n, reply: {**reply, "extra_creatures": ["a cat"]} if n <= 3 else reply
    state = await v3_env.run(text=V3_PAPER_TEXT)

    first = state["panels"][0]   # the hook is a teaching scene: one try more than max_attempts (2 here), all with the cat
    assert first["status"] == "done" and first["render_attempts"] == 3
    assert first["qa"]["issues"] and first["qa"]["issues"][0].startswith("extra figure: remove a cat")
    assert first["qa"]["score"] < 10
    assert _file(v3_env, state, first["image_url"]).exists()


async def test_v3_job_keeps_the_paid_art_when_the_vision_model_gives_no_grounding(v3_env):
    v3_env.vision.router.hook = lambda n, reply: "I cannot help with that."
    state = await v3_env.run(text=V3_PAPER_TEXT)

    assert state["status"] == "completed", state.get("error")
    for p in state["panels"]:
        assert p["status"] == "done" and p["render_attempts"] == 1   # a new drawing cannot fix the vision call
        assert p["qa"]["issues"][0].startswith("unchecked: the art could not be checked")
        assert _file(v3_env, state, p["image_url"]).exists()
    assert state["cost"]["images_generated"] == len(state["panels"])


async def test_v3_job_counts_grounding_tokens_as_text_cost(v3_env):
    state = await v3_env.run(text=V3_PAPER_TEXT)
    # the vision fake reports 2000 prompt tokens per call; the story calls report their own usage on top
    assert state["cost"]["text_prompt_tokens"] >= 2000 * len(state["panels"])


async def test_v2_setting_still_runs_the_lettered_pipeline(job_env):
    job_env.settings["pipeline"] = "v2"
    state = await job_env.run()
    assert state["status"] == "completed" and state.get("version", 2) == 2
    assert all("art_url" not in p or not p["art_url"] for p in state["panels"])


async def test_bilingual_v3_job_letters_the_same_art_in_both_languages(v3_env):
    v3_env.settings["bilingual"] = True
    state = await v3_env.run(text=V3_PAPER_TEXT, language="zh-CN")

    assert state["status"] == "completed", state.get("error")
    assert state["languages"] == ["zh-CN", "en-US"] and state["language"] == "zh-CN"
    job_dir = v3_env.output_dir / state["job_id"]
    assert (job_dir / "storyboard.json").exists() and (job_dir / "storyboard.en-US.json").exists()
    zh = json.loads((job_dir / "storyboard.json").read_text(encoding="utf-8"))
    en = json.loads((job_dir / "storyboard.en-US.json").read_text(encoding="utf-8"))
    assert (zh["language"], en["language"]) == ("zh-CN", "en-US") and len(zh["panels"]) == len(en["panels"])
    for p in state["panels"]:
        urls = p["image_urls"]
        assert set(urls) == {"zh-CN", "en-US"} and urls["zh-CN"] == p["image_url"]
        assert urls["en-US"].endswith(f"panel_{p['n']:02d}.en-US.jpg")
        assert _file(v3_env, state, urls["en-US"]).exists() and _file(v3_env, state, urls["zh-CN"]).exists()
        assert (job_dir / f"layout_{p['n']:02d}.en-US.json").exists()
        assert set(p["texts"]) == {"zh-CN", "en-US"} and p["texts"]["zh-CN"]["dialogue"] == p["dialogue"]
    # one drawing per panel serves both languages
    assert state["cost"]["images_generated"] == len(state["panels"]) == v3_env.vision.router.located
    assert state["combined_url"] and state["combined_urls"]["en-US"].split("?")[0].endswith("combined.en-US.jpg")
    assert (job_dir / "combined.en-US.jpg").exists()
    english = state["localized"]["en-US"]
    assert english["title"] == en["title"] and [c["id"] for c in english["chapters"]] == [c["id"] for c in state["chapters"]]


async def test_bilingual_v3_redraw_reletters_both_languages(v3_env):
    v3_env.settings["bilingual"] = True
    state = await v3_env.run(text=V3_PAPER_TEXT, language="en-US")
    first = state["panels"][0]
    before = {lang: _file(v3_env, state, url).stat().st_mtime_ns for lang, url in first["image_urls"].items()}
    located = v3_env.vision.router.located
    v3_env.manager.jobs[state["job_id"]].rt.storyboards = {}   # as after a restart: the scripts come back from disk

    await v3_env.manager.regenerate_panel(state["job_id"], 1)
    await v3_env.manager.jobs[state["job_id"]].rt.task
    after = v3_env.manager.get(state["job_id"])
    redrawn = after["panels"][0]
    assert set(redrawn["image_urls"]) == {"en-US", "zh-CN"} and redrawn["attempts"] > first["attempts"]
    assert v3_env.vision.router.located == located + 1
    for lang, url in redrawn["image_urls"].items():
        assert _file(v3_env, after, url).stat().st_mtime_ns > before[lang]


async def test_monolingual_v3_job_has_no_second_language(v3_env):
    v3_env.settings["bilingual"] = False
    state = await v3_env.run(text=V3_PAPER_TEXT, language="zh-CN")
    assert state["languages"] == ["zh-CN"]
    assert all("image_urls" not in p and "texts" not in p for p in state["panels"])
    assert not (v3_env.output_dir / state["job_id"] / "storyboard.en-US.json").exists()


async def test_v3_job_keeps_the_paid_art_when_lettering_fails(v3_env, monkeypatch):
    from services import lettering

    def broken(*_args, **_kwargs):
        raise lettering.LayoutError("art is missing or too small to letter")

    monkeypatch.setattr(lettering, "letter_panel", broken)
    state = await v3_env.run(text=V3_PAPER_TEXT)
    job_dir = v3_env.output_dir / state["job_id"]
    for p in state["panels"]:
        assert p["status"] == "failed" and p["error"].startswith("lettering failed: LayoutError")
        assert (job_dir / f"art_{p['n']:02d}.jpg").exists() and (job_dir / f"grounding_{p['n']:02d}.json").exists()
    assert state["cost"]["images_generated"] == len(state["panels"])


async def test_relayout_reletters_from_the_saved_art_without_paying_again(v3_env):
    v3_env.settings["bilingual"] = True
    state = await v3_env.run(text=V3_PAPER_TEXT, language="en-US")
    images, located = len(v3_env.image.calls), v3_env.vision.router.located
    before = {p["n"]: (p["attempts"], p["render_attempts"], _file(v3_env, state, p["image_urls"]["zh-CN"]).stat().st_mtime_ns)
              for p in state["panels"]}
    v3_env.manager.jobs[state["job_id"]].rt.storyboard = None      # as after a restart: everything comes back from disk
    v3_env.manager.jobs[state["job_id"]].rt.storyboards = {}

    assert await v3_env.manager.relayout(state["job_id"]) == {"relettered": len(state["panels"])}
    after = v3_env.manager.get(state["job_id"])
    assert len(v3_env.image.calls) == images and v3_env.vision.router.located == located   # no image or vision call
    for p in after["panels"]:
        attempts, render_attempts, mtime = before[p["n"]]
        assert (p["attempts"], p["render_attempts"]) == (attempts + 1, render_attempts)
        assert _file(v3_env, after, p["image_urls"]["zh-CN"]).stat().st_mtime_ns > mtime
    assert await v3_env.manager.relayout("20990101_000000_beef") is None


async def test_relayout_reapplies_the_recap_rule_to_every_language(v3_env):
    import json as _json
    v3_env.settings["bilingual"] = True
    state = await v3_env.run(text=V3_PAPER_TEXT, language="en-US")
    job_dir = v3_env.output_dir / state["job_id"]
    for name in ("storyboard.json", "storyboard.zh-CN.json"):   # as saved by an older rule: a line without its tick
        data = _json.loads((job_dir / name).read_text(encoding="utf-8"))
        recap = next(p for p in data["panels"] if p.get("beat") == "recap" and isinstance(p.get("board"), dict))
        recap["board"]["checked"] = []
        (job_dir / name).write_text(_json.dumps(data, ensure_ascii=False), encoding="utf-8")
    v3_env.manager.jobs[state["job_id"]].rt.storyboard = None
    v3_env.manager.jobs[state["job_id"]].rt.storyboards = {}

    await v3_env.manager.relayout(state["job_id"])
    marks = []
    for name in ("storyboard.json", "storyboard.zh-CN.json"):
        data = _json.loads((job_dir / name).read_text(encoding="utf-8"))
        recap = next(p for p in data["panels"] if p.get("beat") == "recap" and isinstance(p.get("board"), dict))
        assert recap["board"].get("checked") != []   # recomputed from what the story shows (and saved)
        marks.append(recap["board"].get("checked"))
    assert marks[0] == marks[1]   # the translation copies the English marks: its own words are never judged


async def test_relayout_asks_the_recap_evidence_once_for_a_job_built_before_it(v3_env):
    # without the check model's verdict a relayout re-marked every recap line by the rule alone (342 ticks went in 144
    # saved jobs); a job built before the check gets the verdict once, kept in its script
    import json as _json
    state = await v3_env.run(text=V3_PAPER_TEXT, language="en-US")
    job_dir = v3_env.output_dir / state["job_id"]
    data = _json.loads((job_dir / "storyboard.json").read_text(encoding="utf-8"))
    assert "recap_evidence" in data["brief"]          # a new build keeps the verdict
    data["brief"].pop("recap_evidence")               # as saved before the check existed
    (job_dir / "storyboard.json").write_text(_json.dumps(data, ensure_ascii=False), encoding="utf-8")
    job = v3_env.manager.jobs[state["job_id"]]
    job.rt.storyboard, job.rt.storyboards = None, {}
    calls = v3_env.paper.seen.count("recap_evidence")
    await v3_env.manager.relayout(state["job_id"])
    assert v3_env.paper.seen.count("recap_evidence") == calls + 1
    saved = _json.loads((job_dir / "storyboard.json").read_text(encoding="utf-8"))
    assert saved["brief"]["recap_evidence"]           # kept: the next relayout needs no call
    job.rt.storyboard, job.rt.storyboards = None, {}
    await v3_env.manager.relayout(state["job_id"])
    assert v3_env.paper.seen.count("recap_evidence") == calls + 1


async def test_relayout_refreshes_the_readers_text_of_the_recap(v3_env):
    # the scripts and the pictures got the new marks and title; the text under the picture kept "Verified Results"
    v3_env.settings["bilingual"] = True
    state = await v3_env.run(text=V3_PAPER_TEXT, language="en-US")
    job = v3_env.manager.jobs[state["job_id"]]
    entry = next(e for e in job.state["panels"] if e.get("beat") == "recap")
    entry["board"] = {**entry["board"], "title": "Verified Results", "checked": []}   # as saved under an older rule
    entry["texts"]["zh-CN"]["board"] = {**entry["texts"]["zh-CN"]["board"], "checked": []}
    await v3_env.manager.relayout(state["job_id"])
    recap = next(p for p in job.rt.storyboard.panels if p.beat == "recap")
    zh = next(p for p in job.rt.storyboards["zh-CN"].panels if p.beat == "recap")
    assert entry["board"] == recap.board and entry["board"].get("title") != "Verified Results"
    assert entry["texts"]["zh-CN"]["board"] == zh.board and entry["texts"]["en-US"]["board"] == recap.board


async def test_relayout_refuses_v2_and_running_jobs(job_env, v3_env):
    v3_env.settings["pipeline"] = "v2"
    state = await v3_env.run()
    with pytest.raises(ValueError, match="Only v3 jobs"):
        await v3_env.manager.relayout(state["job_id"])


async def test_a_scene_whose_picture_misses_its_idea_is_redrawn_and_its_objects_are_kept(v3_teaching_env):
    v3_env = v3_teaching_env
    # the picture audit of 2026-10-28: no panel's art showed its idea on its own; the vision model now checks the shows line
    def idea(call, prompt):
        if call == 1:
            return {"shown": "no", "missing": "the cards are not tied together", "objects": []}

    v3_env.vision.router.idea = idea
    state = await v3_env.run(text=V3_PAPER_TEXT)

    assert state["status"] == "completed", state.get("error")
    first = next(p for p in state["panels"] if p["kind"] == "scene" and p.get("render_attempts") == 2)
    redraw = [c["prompt"] for c in v3_env.image.calls if "idea not shown" in c["prompt"]]
    assert len(redraw) == 1 and "the cards are not tied together" in redraw[0]
    teaching = [p for p in state["panels"] if p["kind"] == "scene" and p["beat"] in ("hook", "problem", "idea", "how", "payoff", "limits")]
    assert sum("The picture itself must make this clear" in c["prompt"] for c in v3_env.image.calls) == len(teaching) + 1
    saved = json.loads((v3_env.output_dir / state["job_id"] / f"grounding_{first['n']:02d}.json").read_text(encoding="utf-8"))
    assert saved["objects"] and first["qa"]["issues"] == []
    assert len(v3_env.vision.router.ideas) == len(teaching) + 1      # boards and story-only scenes are never asked


TEACHING = ("hook", "problem", "idea", "how", "payoff", "limits")


def _misses(verdict: str):
    """An idea check that judges every single picture of the fake story (its shows line ties cards with red yarn) as
    ``verdict`` and every frame of two as shown."""
    return lambda call, prompt: {"shown": verdict, "missing": "the cards are not tied together", "objects": []} if "red yarn" in prompt else None


async def test_a_teaching_picture_that_never_shows_its_idea_is_drawn_as_two_frames(v3_teaching_env):
    v3_env = v3_teaching_env
    # round 17: of 13 teaching scenes whose picture missed its idea after every try, blind judges preferred the panel
    # drawn as two frames (each one moment) 9 times and the single picture 3 times
    v3_env.vision.router.idea = _misses("partly")
    state = await v3_env.run(text=V3_PAPER_TEXT)

    assert state["status"] == "completed", state.get("error")
    teaching = [p for p in state["panels"] if p["kind"] == "scene" and p["beat"] in TEACHING]
    assert teaching and sorted(state["quality"]["two_frames"]) == [p["n"] for p in teaching]
    tries = v3_env.settings["max_attempts"] + 1
    for p in teaching:
        assert [f["where"] for f in p["frames"]] == ["top", "bottom"] and p["frames"][1]["shows"] == "one mixed pile of blank cards"
        assert p["render_attempts"] == tries + 2 and p["qa"]["issues"] == []   # each frame came out at the first try
        with Image.open(_file(v3_env, state, p["art_url"])) as art:
            assert art.size == (512, 768)   # the single picture's size: page layout and lettering are unchanged
        saved = json.loads((v3_env.output_dir / state["job_id"] / f"grounding_{p['n']:02d}.json").read_text(encoding="utf-8"))
        assert saved["size"] == [512, 768] and any(box[1] >= 384 for box in saved["objects"])   # the bottom frame's objects
    frames = [c for c in v3_env.image.calls if "A single manga panel, landscape," in c["prompt"]]
    assert len(frames) == 2 * len(teaching) and all(c["config"].width > c["config"].height for c in frames)
    assert v3_env.paper.seen.count("two_frames") == len(teaching)
    assert state["cost"]["images_generated"] == len(v3_env.image.calls)
    logged = (v3_env.output_dir / state["job_id"] / "prompts.jsonl").read_text(encoding="utf-8").splitlines()
    assert sum("landscape" in line for line in logged) == 2 * len(teaching)   # both frames' prompts are logged


async def test_two_frames_with_drawn_text_leave_the_single_picture(v3_teaching_env):
    v3_env = v3_teaching_env
    from services.jobs import FRAME_TRIES
    v3_env.vision.router.idea = _misses("no")
    last = {"frame": False}
    v3_env.image.hook = lambda prompt: last.update(frame="A single manga panel, landscape," in prompt)
    v3_env.vision.router.hook = lambda n, reply: (
        {**reply, "text_or_symbols": [{"text": "A B", "where": "the cards"}]} if last["frame"] else reply)
    state = await v3_env.run(text=V3_PAPER_TEXT)

    assert state["status"] == "completed", state.get("error")
    teaching = [p for p in state["panels"] if p["kind"] == "scene" and p["beat"] in TEACHING]
    assert teaching and state["quality"]["two_frames"] == []
    for p in teaching:   # drawn text is worse than a missed idea: the single picture stays, and the frames were paid for
        assert "frames" not in p and p["qa"]["issues"][0].startswith("idea not shown:")
        assert p["render_attempts"] == v3_env.settings["max_attempts"] + 1 + 2 * FRAME_TRIES
    assert state["cost"]["images_generated"] == len(v3_env.image.calls)


async def test_a_frame_of_objects_only_is_drawn_without_the_cast_sheets(v3_teaching_env):
    v3_env = v3_teaching_env
    # round 19's smoke test: an objects-only frame was given all three sheets, the image model drew kumo into it, and the
    # extra figure threw the frames away
    v3_env.vision.router.idea = _misses("partly")
    v3_env.paper.two_frames = lambda prompt: {"frames": [
        {"visual": "Two separate piles of blank cards sit on a table.", "shows": "two separate piles of blank cards"},
        {"visual": "papi pushes the two piles into one mixed pile.", "shows": "one mixed pile of blank cards"}]}
    state = await v3_env.run(text=V3_PAPER_TEXT)

    assert state["status"] == "completed", state.get("error")
    tops = [c for c in v3_env.image.calls if "landscape" in c["prompt"] and "Two separate piles" in c["prompt"]]
    assert tops and all(c["refs"] == [] and "No characters in this panel." in c["prompt"] for c in tops)
    teaching = [p for p in state["panels"] if p["kind"] == "scene" and p["beat"] in TEACHING]
    assert all(p["frames"][0]["characters"] == [] for p in teaching)


async def test_a_plan_with_a_frame_whose_idea_cannot_be_judged_draws_no_frames(v3_teaching_env):
    v3_env = v3_teaching_env
    # a frame whose "shows" names a chart has no idea the check can judge: it never replaces a judged picture
    v3_env.vision.router.idea = _misses("partly")
    v3_env.paper.two_frames = lambda prompt: {"frames": [
        {"visual": "papi sets blank cards in two separate piles.", "shows": "two separate piles of blank cards"},
        {"visual": "papi stands beside a bar chart.", "shows": "a bar chart where one bar towers"}]}
    state = await v3_env.run(text=V3_PAPER_TEXT)

    assert state["status"] == "completed", state.get("error")
    teaching = [p for p in state["panels"] if p["kind"] == "scene" and p["beat"] in TEACHING]
    assert teaching and all("frames" not in p and p["render_attempts"] == v3_env.settings["max_attempts"] + 1 for p in teaching)
    assert not any("landscape" in c["prompt"] for c in v3_env.image.calls)


async def test_a_teaching_picture_without_a_plan_of_two_frames_keeps_its_single_drawing(v3_teaching_env):
    v3_env = v3_teaching_env
    v3_env.vision.router.idea = _misses("partly")
    v3_env.paper.two_frames = lambda prompt: {"frames": [{"visual": "papi holds one stack."}]}
    state = await v3_env.run(text=V3_PAPER_TEXT)

    assert state["status"] == "completed", state.get("error")
    teaching = [p for p in state["panels"] if p["kind"] == "scene" and p["beat"] in TEACHING]
    assert teaching and all("frames" not in p and p["render_attempts"] == v3_env.settings["max_attempts"] + 1 for p in teaching)
    assert not any("landscape" in c["prompt"] for c in v3_env.image.calls)
    assert all(p["qa"]["issues"][0].startswith("idea unclear:") for p in teaching)


async def test_new_books_draw_no_symbol_sheets_and_an_older_book_keeps_its_own(v3_env, monkeypatch):
    # a brief model that still offers symbols: the story never gets them, so no prop sheet is drawn
    monkeypatch.setattr(fakes, "BRIEF_V3", {**fakes.BRIEF_V3, "picture_symbols": [
        {"concept": "attention", "symbol": "red yarn thread"}, {"concept": "a gate", "symbol": "small round grey dimmer switch"}]})
    state = await v3_env.run(text=V3_PAPER_TEXT)

    assert state["status"] == "completed", state.get("error")
    assert not any("A reference sheet of one prop" in c["prompt"] for c in v3_env.image.calls)
    assert not state.get("props")
    assert state["cost"]["images_generated"] == len(v3_env.image.calls)
    # a book made with symbols (saved before they were dropped) draws its sheets once on a redraw and gives them to the
    # pictures that draw them (every fake visual "ties blank cards together with red yarn")
    job_id = state["job_id"]
    folder = v3_env.output_dir / job_id
    for name in [f.name for f in folder.glob("storyboard*.json")]:
        saved = json.loads((folder / name).read_text(encoding="utf-8"))
        saved["brief"]["picture_symbols"] = [{"concept": "attention", "symbol": "red yarn thread"}]
        (folder / name).write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")
    scenes = [p for p in state["panels"] if p["kind"] == "scene"]
    before = len(v3_env.image.calls)
    v3_env.manager.jobs.pop(job_id)   # a fresh runtime, as after a restart
    assert await v3_env.manager.regenerate_panel(job_id, scenes[0]["n"]) == {"status": "queued"}
    await asyncio.wait_for(v3_env.manager.jobs[job_id].rt.task, timeout=10)
    again = v3_env.image.calls[before:]
    sheets = [c for c in again if "A reference sheet of one prop" in c["prompt"]]
    assert [c["prompt"].split("for an illustrated book: ", 1)[1].split(".", 1)[0] for c in sheets] == ["red yarn thread"]
    yarn = [c for c in again if "shows the red yarn thread: draw it exactly" in c["prompt"]]
    assert yarn and all(len(c["refs"]) >= 2 for c in yarn)   # a cast sheet and the yarn's sheet


async def test_a_failed_idea_check_never_blocks_the_art(v3_teaching_env):
    v3_env = v3_teaching_env
    def idea(call, prompt):
        raise RuntimeError("vision model down")

    v3_env.vision.router.idea = idea
    state = await v3_env.run(text=V3_PAPER_TEXT)
    assert state["status"] == "completed" and all(p["render_attempts"] == 1 for p in state["panels"])
    teaching = sorted(p["n"] for p in state["panels"] if p["kind"] == "scene" and p["beat"] != "proof")
    assert sorted(state["quality"]["unchecked_idea"]) == teaching   # the record says which ideas no one judged


async def test_the_unjudged_idea_record_follows_the_published_picture(v3_teaching_env):
    v3_env = v3_teaching_env
    # a redraw whose idea was judged leaves the record, and a teaching picture whose art could not be located joins it
    down = {"on": True}

    def idea(call, prompt):
        if down["on"]:
            raise RuntimeError("vision model down")

    v3_env.vision.router.idea = idea
    state = await v3_env.run(text=V3_PAPER_TEXT)
    job_id, unjudged = state["job_id"], sorted(state["quality"]["unchecked_idea"])
    assert unjudged
    down["on"] = False
    assert await v3_env.manager.regenerate_panel(job_id, unjudged[0]) == {"status": "queued"}
    await asyncio.wait_for(v3_env.manager.jobs[job_id].rt.task, timeout=10)
    assert sorted(v3_env.manager.get(job_id)["quality"]["unchecked_idea"]) == unjudged[1:]


async def test_a_teaching_picture_that_could_not_be_located_counts_as_unjudged(v3_teaching_env):
    v3_env = v3_teaching_env
    v3_env.vision.router.hook = lambda n, reply: "I cannot help with that."
    state = await v3_env.run(text=V3_PAPER_TEXT)
    teaching = sorted(p["n"] for p in state["panels"] if p["kind"] == "scene" and p["beat"] in
                      ("hook", "problem", "idea", "how", "payoff", "limits"))
    assert teaching and sorted(state["quality"]["unchecked_idea"]) == teaching
