"""Re-letter saved art using reviewed grounding; offline by default.

The run folder must contain storyboard.zh-CN.json, art/, grounding_NN.json,
scene-objects.json and art-manifest.json. This command does not generate art.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from services import art_check, lettering
from services.jobs import _storyboard_from_dict


def render(run: Path):
    storyboard = _storyboard_from_dict(json.loads((run / "storyboard.zh-CN.json").read_text()))
    manifest = json.loads((run / "art-manifest.json").read_text())
    objects = json.loads((run / "scene-objects.json").read_text())["panels"]
    fonts = lettering.FontSet()
    public = run / "public"
    public.mkdir(exist_ok=True)
    reports = []
    for panel in storyboard.panels:
        src = run / "art" / f"art_{panel.n:02d}.png"
        digest = hashlib.sha256(src.read_bytes()).hexdigest()
        if digest != manifest[str(panel.n)]["sha256"]:
            raise ValueError(f"Panel {panel.n}: art changed; recheck grounding before lettering")
        reviewed = objects.get(str(panel.n), {})
        if not reviewed.get("reviewed") or reviewed.get("image_sha256") != digest:
            raise ValueError(f"Panel {panel.n}: object review is missing or belongs to different art")
        grounding = art_check.Grounding(**json.loads((run / f"grounding_{panel.n:02d}.json").read_text()))
        grounding.objects = reviewed["objects"]
        with Image.open(src) as image:
            art = image.convert("RGB")
        out, layout = lettering.letter_panel(art, panel, grounding, storyboard.language, fonts)
        out.save(public / f"panel_{panel.n:02d}.jpg", quality=94)
        (run / f"layout_{panel.n:02d}.json").write_text(json.dumps(layout, ensure_ascii=False, indent=2) + "\n")
        reports.append({"panel": panel.n, "size": list(out.size), "fallback": layout["fallback"],
                        "strip_placement": (layout.get("strip") or {}).get("placement", "below" if layout.get("strip") else None),
                        "board_error": layout.get("board_error"), "art_sha256": digest})
    (run / "render-report.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n")
    print(f"Rendered {len(reports)} panels without model calls")
    if any(p["board_error"] for p in reports):
        raise SystemExit("One or more boards failed; inspect render-report.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    render(parser.parse_args().run.resolve())
