// Offline regression tests of the job quality disclosure: what the helper says, and what the real QualityNote renders.
// The TypeScript sources are transpiled in memory with the project's own compiler, so nothing is written to disk.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { test } from "node:test";

const require = createRequire(import.meta.url);
const ts = require("typescript");
const React = require("react");
const { renderToStaticMarkup } = require("react-dom/server");

const source = (relative) => {
  const path = new URL(relative, import.meta.url);
  const { outputText } = ts.transpileModule(readFileSync(path, "utf8"), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.React, target: ts.ScriptTarget.ES2020, esModuleInterop: true },
  });
  return outputText;
};

const evaluate = (code, requireModule) => {
  const module = { exports: {} };
  new Function("require", "module", "exports", code)(requireModule, module, module.exports);
  return module.exports;
};

const helper = () => evaluate(source("../src/lib/quality-notes.ts"), require);

// JobView's other imports (the reader, the store, ...) play no part in the note: they are stubbed.
const stub = new Proxy({}, { get: (_, key) => (key === "__esModule" ? true : () => null) });
const QualityNote = (react = React) =>
  evaluate(source("../src/components/JobView.tsx"), (name) => {
    if (name === "@/lib/quality-notes") return helper();
    if (name === "react") return react;
    if (name === "lucide-react") return require(name);
    return stub;
  }).QualityNote;

const render = (quality) => renderToStaticMarkup(React.createElement(QualityNote(), { quality }));

const FALLBACK = "deepseek-v4-pro: its content filter refused the text; seed-2-0 took over";
const SKIPPED = { stage: "fact-check", error: "TimeoutError: no reply" };

const kinds = (notes) => notes.sections.map((s) => s.kind);

test("legacy jobs: absent, null, empty or counters-only quality says nothing", () => {
  const { qualityNotes } = helper();
  const legacy = [
    undefined,
    null,
    {},
    { skipped: [], unchecked_art: [], unchecked_idea: [], fallbacks: [] },
    { skipped: null, unchecked_art: null, unchecked_idea: null, fallbacks: null },
    { lint_left: 2, scrubbed: 1 },
  ];
  for (const quality of legacy) {
    assert.equal(qualityNotes(quality), null, JSON.stringify(quality));
    assert.equal(render(quality), "", JSON.stringify(quality));
  }
});

test("fallback only: discloses the backup model and never says a check did not run", () => {
  const notes = helper().qualityNotes({ skipped: [], unchecked_art: [], unchecked_idea: [], fallbacks: [FALLBACK] });
  assert.deepEqual(kinds(notes), ["fallback"]);
  assert.equal(notes.label, "Backup model used");
  assert.deepEqual(notes.sections[0].items, [FALLBACK]);
  assert.match(notes.sections[0].explanation, /writer's model/);
  assert.match(notes.sections[0].explanation, /less independent/);

  const html = render({ fallbacks: [FALLBACK] });
  assert.ok(html.includes("Backup model used"));
  assert.doesNotMatch(html, /did not run|never ran|not checked|could not be checked|incomplete/i);
});

test("unchecked idea only: says the picture's idea went unjudged, not that the art or the script did", () => {
  const notes = helper().qualityNotes({ unchecked_idea: [7, 2] });
  assert.deepEqual(kinds(notes), ["unchecked_idea"]);
  assert.equal(notes.label, "Some pictures unchecked");
  assert.deepEqual(notes.sections[0].items, ["Panels 2, 7"]);
  assert.match(notes.sections[0].explanation, /idea/);

  const html = render({ unchecked_idea: [7, 2] });
  assert.ok(html.includes("Panels 2, 7"));
  assert.doesNotMatch(html, /model|did not run|did not finish/i);
});

test("art and idea are separate reasons, even for the same panel", () => {
  const notes = helper().qualityNotes({ unchecked_art: [3], unchecked_idea: [3] });
  assert.deepEqual(kinds(notes), ["unchecked_art", "unchecked_idea"]);
  assert.deepEqual(notes.sections.map((s) => s.items), [["Panel 3"], ["Panel 3"]]);
  assert.notEqual(notes.sections[0].explanation, notes.sections[1].explanation);
});

test("combined: every reason stays visible, in the closed disclosure's markup too", () => {
  const quality = { skipped: [SKIPPED], unchecked_art: [4], unchecked_idea: [5, 6], fallbacks: [FALLBACK, "second note"] };
  const notes = helper().qualityNotes(quality);
  assert.deepEqual(kinds(notes), ["fallback", "skipped", "unchecked_art", "unchecked_idea"]);
  assert.equal(notes.label, "Check details");

  const html = render(quality);
  // React escapes the apostrophes of the text it renders, so the raw notes are compared as rendered
  for (const text of [FALLBACK, "second note", "fact-check", SKIPPED.error, "Panel 4", "Panels 5, 6"]) {
    assert.ok(html.includes(text), text);
  }
  for (const { heading, explanation } of notes.sections) {
    assert.ok(html.includes(heading.replaceAll("'", "&#x27;")), heading);
    assert.ok(html.includes(explanation.replaceAll("'", "&#x27;")), explanation);
  }
});

test("skipped: generic wording fits a check or a translation step, without claiming it never ran", () => {
  const notes = helper().qualityNotes({ skipped: [{ stage: "translation of reader texts", error: "ValueError: bad reply" }] });
  assert.deepEqual(kinds(notes), ["skipped"]);
  assert.equal(notes.label, "Some steps incomplete");
  assert.deepEqual(notes.sections[0].items, ["translation of reader texts: ValueError: bad reply"]);
  const said = JSON.stringify(notes);
  assert.doesNotMatch(said, /did not run|never ran|checks? (was|were) skipped|no check/i);
  assert.match(notes.sections[0].explanation, /may contain mistakes/);
});

test("malformed entries do not produce a note or an empty reason", () => {
  const { qualityNotes } = helper();
  assert.equal(qualityNotes({ fallbacks: ["", "   "], unchecked_idea: [], skipped: [] }), null);
  const notes = qualityNotes({ skipped: [{ stage: "", error: "" }] });
  assert.deepEqual(notes.sections[0].items, ["A step"]);
});

test("the details work without hover: a native disclosure with its text in the page", () => {
  const html = render({ fallbacks: [FALLBACK], unchecked_idea: [1] });
  assert.match(html, /^<details[\s>]/);
  assert.match(html, /<summary[\s>]/);
  assert.ok(html.includes("Check details"), "the summary names the disclosure for screen readers on any screen size");
  assert.ok(!/<details[^>]*\stitle=/.test(html) && !/<summary[^>]*\stitle=/.test(html), "nothing is left to a hover-only title");
});

// The real QualityNote is called as a plain function, with its hooks stubbed, to reach the handler it attaches to <details>.
// The handler then runs against hand-made stand-ins for the event and the element. This pins the handler's decisions only:
// no DOM or browser is involved, so native toggling, real focus and the event's path to the window are not exercised here.
const READER_KEYS = ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "PageUp", "PageDown"];
const SPACE_KEYS = [" ", "Spacebar"];

const keyboardNote = (open) => {
  const element = QualityNote({ ...React, useRef: (current) => ({ current }), useEffect: () => {} })({ quality: { fallbacks: [FALLBACK] } });
  assert.equal(element.type, "details");

  const seen = { focused: 0 };
  const summary = { focus: () => (seen.focused += 1) };
  const details = { open, querySelector: (selector) => (selector === "summary" ? summary : null) };
  const press = (key) => {
    const pressed = { stopped: false, prevented: false };
    element.props.onKeyDown({
      key,
      currentTarget: details,
      stopPropagation: () => (pressed.stopped = true),
      preventDefault: () => (pressed.prevented = true),
    });
    return pressed;
  };
  return { details, seen, press };
};

test("keyboard, closed note: the reader's page keys pass through untouched", () => {
  for (const key of READER_KEYS) {
    const { details, seen, press } = keyboardNote(false);
    assert.deepEqual(press(key), { stopped: false, prevented: false }, key);
    assert.equal(details.open, false, key);
    assert.equal(seen.focused, 0, key);
  }
});

test("keyboard, open note: the page keys stay in the note and keep their native scrolling", () => {
  for (const key of READER_KEYS) {
    const { details, seen, press } = keyboardNote(true);
    assert.deepEqual(press(key), { stopped: true, prevented: false }, key);
    assert.equal(details.open, true, key);
    assert.equal(seen.focused, 0, key);
  }
});

test("keyboard: Space never reaches the reader, open or closed, and is left to toggle the native disclosure", () => {
  for (const open of [false, true]) {
    for (const key of SPACE_KEYS) {
      const { details, seen, press } = keyboardNote(open);
      assert.deepEqual(press(key), { stopped: true, prevented: false }, `${JSON.stringify(key)}, open=${open}`);
      assert.equal(details.open, open, "the handler leaves the toggling to the browser");
      assert.equal(seen.focused, 0);
    }
  }
});

test("keyboard: Escape closes an open note and refocuses its summary, then the page keys pass through again", () => {
  const { details, seen, press } = keyboardNote(true);
  assert.deepEqual(press("Escape"), { stopped: true, prevented: false });
  assert.equal(details.open, false);
  assert.equal(seen.focused, 1);

  assert.deepEqual(press("ArrowRight"), { stopped: false, prevented: false }, "the review's case: focus is on the summary of a closed note");
  assert.equal(seen.focused, 1);
});

test("keyboard: Escape on a closed note does nothing and passes through", () => {
  const { details, seen, press } = keyboardNote(false);
  assert.deepEqual(press("Escape"), { stopped: false, prevented: false });
  assert.equal(details.open, false);
  assert.equal(seen.focused, 0);
});

test("reader review: unresolved explanations identify language and panel", () => {
  const quality = { reader_reviews: { "zh-CN": { status: "needs_revision", problems: [
    { n: 4, kind: "causal_gap", problem: "没有说明老师为什么要看学生写出的内容。", fix: "Explain the next-step teaching." },
  ] } } };
  const notes = helper().qualityNotes(quality);
  assert.equal(notes.label, "Reading notes");
  assert.deepEqual(kinds(notes), ["reader_review"]);
  assert.match(notes.sections[0].items[0], /^zh-CN: Panel 4/);
  assert.match(render(quality), /没有说明老师/);
  assert.match(notes.sections[0].explanation, /not a test of what a person understood/);
});

test("reader review: unavailable or malformed diagnostic never implies comprehension passed", () => {
  for (const status of ["unassessed", "unexpected"]) {
    const notes = helper().qualityNotes({ reader_reviews: { "en-US": { status, reason: "private engine detail", problems: null } } });
    assert.match(notes.sections[0].items[0], /not received a usable reading review/);
    assert.doesNotMatch(JSON.stringify(notes), /private engine detail|passed/);
  }
  assert.equal(helper().qualityNotes({ reader_reviews: [] }), null);
});

test("reader review: clean AI diagnostic does not publish a comprehension success badge", () => {
  assert.equal(render({ reader_reviews: { "en-US": { status: "reviewed", problems: [] } } }), "");
  assert.equal(helper().qualityNotes({ reader_reviews: { "en-US": null } }), null);
  const combined = helper().qualityNotes({ reader_reviews: { "zh-CN": { status: "unassessed" } }, unchecked_art: [2] });
  assert.equal(combined.label, "Check details");
  assert.deepEqual(kinds(combined), ["reader_review", "unchecked_art"]);
});
