"""Build the reviewed Chinese case without any model calls."""
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'backend'))
from tools.preview_story import build_preview

HERE = Path(__file__).resolve().parent
out = REPO / 'output/readability-case/index.html'
stories = [(json.loads((HERE / 'storyboard.zh-CN.json').read_text()), '中文教学重写'),
           (json.loads((HERE / 'before-storyboard.json').read_text()), '原有脚本')]
page, report = build_preview(stories)
page = page.replace('实际生成的分镜文字与图解。', '生成后经论文对照修订的文字与图解。')
page = page.replace('这篇论文，按顺序讲清楚', '压小以后还会解题吗？')
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(page, encoding='utf-8')
out.with_suffix('.report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(out)
if report['board_errors']:
    raise SystemExit(2)
