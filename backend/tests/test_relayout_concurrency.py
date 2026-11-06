"""User-visible mutation exclusion, exercised with the real fake-engine job pipeline."""
import asyncio

import pytest

from .fakes import V3_PAPER_TEXT
from .test_jobs_v3 import v3_env  # noqa: F401 - shared pipeline fixture


@pytest.mark.parametrize("competing", ["relayout", "regenerate"])
async def test_relayout_rejects_a_competing_mutation(v3_env, monkeypatch, competing):
    state = await v3_env.run(text=V3_PAPER_TEXT)
    manager, job_id = v3_env.manager, state["job_id"]
    entered, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def gated_stitch(job):
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            await release.wait()

    monkeypatch.setattr(manager, "_stitch", gated_stitch)
    first = asyncio.create_task(manager.relayout(job_id))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        with pytest.raises(ValueError, match="Job is still running"):
            if competing == "relayout":
                await manager.relayout(job_id)
            else:
                await manager.regenerate_panel(job_id, state["panels"][0]["n"])
    finally:
        release.set()
        await asyncio.wait_for(first, 5)
        job = manager.jobs[job_id]
        if job.running:
            await asyncio.wait_for(job.rt.task, 5)
    # The completed operation must not leave the task permanently busy.
    assert await manager.relayout(job_id) == {"relettered": len(state["panels"])}


async def test_failed_relayout_releases_mutation_guard(v3_env, monkeypatch):
    state = await v3_env.run(text=V3_PAPER_TEXT)
    manager, job_id = v3_env.manager, state["job_id"]
    stitch = manager._stitch

    async def failed_stitch(job):
        raise OSError("synthetic stitch failure")

    monkeypatch.setattr(manager, "_stitch", failed_stitch)
    with pytest.raises(OSError, match="synthetic stitch failure"):
        await manager.relayout(job_id)
    monkeypatch.setattr(manager, "_stitch", stitch)
    assert await manager.relayout(job_id) == {"relettered": len(state["panels"])}


async def test_cancelled_relayout_request_releases_mutation_guard(v3_env, monkeypatch):
    state = await v3_env.run(text=V3_PAPER_TEXT)
    manager, job_id = v3_env.manager, state["job_id"]
    stitch = manager._stitch
    entered = asyncio.Event()

    async def waiting_stitch(job):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(manager, "_stitch", waiting_stitch)
    request = asyncio.create_task(manager.relayout(job_id))
    try:
        await asyncio.wait_for(entered.wait(), 5)
    finally:
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
    monkeypatch.setattr(manager, "_stitch", stitch)
    assert await manager.relayout(job_id) == {"relettered": len(state["panels"])}
