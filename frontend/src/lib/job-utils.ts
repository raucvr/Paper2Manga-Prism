/** Pure helpers for working with `JobState` in the UI. */

import type {
  JobChapter,
  JobLanguage,
  JobPanel,
  JobPaperCard,
  JobQuiz,
  JobState,
  JobStatus,
  PanelStatus,
} from "@/lib/api";

// ---------------------------------------------------------------------------
// Statuses
// ---------------------------------------------------------------------------

export const TERMINAL_STATUSES: ReadonlySet<JobStatus> = new Set<JobStatus>([
  "completed",
  "failed",
  "cancelled",
]);

export function isTerminal(status: JobStatus): boolean {
  return TERMINAL_STATUSES.has(status);
}

/** Panel statuses that will still change without user action. */
export const ACTIVE_PANEL_STATUSES: ReadonlySet<PanelStatus> = new Set<PanelStatus>([
  "pending",
  "rendering",
  "qa",
]);

/** A user-triggered redraw is running: only `redraw_panels` get drawn, not every pending panel. */
export function isRedrawOnly(job: Pick<JobState, "status" | "redraw_panels">): boolean {
  return !isTerminal(job.status) && (job.redraw_panels?.length ?? 0) > 0;
}

/** "Redrawing panel 3…" / "Redrawing panels 3–4…" */
export function redrawLabel(panelNumbers: readonly number[]): string {
  const ns = [...panelNumbers].sort((a, b) => a - b);
  if (ns.length <= 1) return `Redrawing panel ${ns[0] ?? ""}…`;
  const contiguous = ns.every((n, i) => i === 0 || n === ns[i - 1] + 1);
  return `Redrawing panels ${contiguous ? `${ns[0]}–${ns[ns.length - 1]}` : ns.join(", ")}…`;
}

export type PipelineStage =
  | "parsing"
  | "analyzing"
  | "storyboarding"
  | "translating"
  | "rendering"
  | "assembling";

/** The pipeline in order, as shown by the progress stepper. */
export const PIPELINE: ReadonlyArray<{ id: PipelineStage; label: string }> = [
  { id: "parsing", label: "Read" },
  { id: "analyzing", label: "Analyze" },
  { id: "storyboarding", label: "Storyboard" },
  { id: "translating", label: "Translate" },
  { id: "rendering", label: "Draw" },
  { id: "assembling", label: "Assemble" },
];

/** Friendly, low-pressure copy for each job status. */
export const STATUS_COPY: Record<JobStatus, { title: string; message: string }> = {
  queued: { title: "Queued", message: "Getting in line… your paper is next." },
  parsing: { title: "Reading", message: "Reading your paper…" },
  analyzing: { title: "Analyzing", message: "Finding the big ideas…" },
  storyboarding: { title: "Storyboarding", message: "Sketching the story, panel by panel…" },
  translating: { title: "Translating", message: "Translating the dialogue…" },
  rendering: { title: "Drawing", message: "Drawing the panels…" },
  assembling: { title: "Assembling", message: "Putting the pages together…" },
  completed: { title: "Done", message: "Your manga is ready!" },
  failed: { title: "Failed", message: "Something went wrong while making this manga." },
  cancelled: { title: "Stopped", message: "This manga was stopped before it was finished." },
};

/** Index of the job's stage in `PIPELINE` (-1 = queued, PIPELINE.length = completed). */
export function pipelineIndex(status: JobStatus): number {
  if (status === "queued") return -1;
  if (status === "completed") return PIPELINE.length;
  return PIPELINE.findIndex((s) => s.id === status);
}

/** Fraction (0–1) of panels that are drawn. */
export function panelFraction(job: JobState): number {
  const { done, total } = job.progress ?? { done: 0, total: 0, percent: 0 };
  if (total > 0) return clamp01(done / total);
  if (job.panels.length === 0) return 0;
  return clamp01(job.panels.filter((p) => p.status === "done").length / job.panels.length);
}

const STAGE_START: Record<JobStatus, number> = {
  queued: 0,
  parsing: 2,
  analyzing: 8,
  storyboarding: 15,
  translating: 27,
  rendering: 32,
  assembling: 96,
  completed: 100,
  failed: 0,
  cancelled: 0,
};

/**
 * Overall progress (0–100) across the whole pipeline. Drawing dominates the
 * wall-clock time, so it gets most of the bar; the early stages still move it
 * so the bar never looks stuck at 0 %.
 */
export function overallPercent(job: JobState): number {
  const fraction = panelFraction(job);
  switch (job.status) {
    case "completed":
      return 100;
    case "rendering":
      return Math.round(STAGE_START.rendering + (STAGE_START.assembling - STAGE_START.rendering) * fraction);
    case "failed":
    case "cancelled":
      return Math.round(fraction * 100);
    default:
      return STAGE_START[job.status];
  }
}

// ---------------------------------------------------------------------------
// Chapters
// ---------------------------------------------------------------------------

/** A stretch of consecutive panels in the same chapter: one chapter section of the reader. */
export interface ChapterRun {
  /** Index into `job.chapters`. */
  chapterIndex: number;
  title: string;
  /** Index (in reading order) of the run's first panel. */
  start: number;
  /** The run's panel numbers, in reading order. */
  panelNumbers: number[];
}

export interface ChapterRuns {
  runs: ChapterRun[];
  /** Panel number -> index into `runs` (panels before the first known chapter have none). */
  runOf: Map<number, number>;
}

/**
 * Chapter sections in reading order (`panels` sorted by `n`): runs of consecutive panels with the same chapter, taken
 * from the panel's chapter id, else from the chapter whose `panel_numbers` list it; a panel with neither stays in the
 * current run. A chapter whose panels are not contiguous gives one run per stretch, so headings don't repeat out of
 * order and the current chapter doesn't flip back and forth.
 */
export function buildChapterRuns(chapters: JobChapter[], panels: JobPanel[]): ChapterRuns {
  const byId = new Map<string, number>();
  const byPanel = new Map<number, number>();
  chapters.forEach((chapter, index) => {
    if (!byId.has(chapter.id)) byId.set(chapter.id, index);
    for (const n of chapter.panel_numbers ?? []) {
      if (!byPanel.has(n)) byPanel.set(n, index);
    }
  });
  const runs: ChapterRun[] = [];
  const runOf = new Map<number, number>();
  panels.forEach((panel, i) => {
    const chapterIndex = byId.get(panel.chapter) ?? byPanel.get(panel.n);
    if (chapterIndex !== undefined && chapterIndex !== runs[runs.length - 1]?.chapterIndex) {
      runs.push({ chapterIndex, title: chapters[chapterIndex].title, start: i, panelNumbers: [] });
    }
    const run = runs[runs.length - 1];
    if (!run) return;
    run.panelNumbers.push(panel.n);
    runOf.set(panel.n, runs.length - 1);
  });
  return { runs, runOf };
}

// ---------------------------------------------------------------------------
// Panels & images
// ---------------------------------------------------------------------------

export interface ImageSize {
  w: number;
  h: number;
}

/** Most generated panels are 1024×1536 portrait. */
export const DEFAULT_PANEL_ASPECT = 2 / 3;

export function isStripPanel(panel: JobPanel): boolean {
  return panel.strip_index === 0 || panel.strip_index === 1;
}

/**
 * Width / height of what the reader shows for this panel. For strip panels
 * that is one half of the shared image. Measured image size (after load) wins
 * over the size reported by the API.
 */
export function panelAspect(panel: JobPanel, natural: ImageSize | undefined, fallback: number): number {
  const strip = isStripPanel(panel);
  if (natural && natural.w > 0 && natural.h > 0) {
    return strip ? natural.w / 2 / natural.h : natural.w / natural.h;
  }
  if (strip) {
    if (panel.strip_width && panel.strip_height) return panel.strip_width / 2 / panel.strip_height;
    return fallback;
  }
  if (panel.width && panel.height) return panel.width / panel.height;
  return fallback;
}

/** Aspect ratio to use for panels whose image doesn't exist yet. */
export function defaultAspect(panels: JobPanel[]): number {
  for (const panel of panels) {
    if (isStripPanel(panel)) {
      if (panel.strip_width && panel.strip_height) return panel.strip_width / 2 / panel.strip_height;
    } else if (panel.width && panel.height) {
      return panel.width / panel.height;
    }
  }
  return DEFAULT_PANEL_ASPECT;
}

/**
 * Cache-busting version per image URL. A regenerated panel may be written to
 * the same path, so the URL gets `?v=<attempts>`; strip halves share one image,
 * so the version is the max over every panel using that URL.
 */
export function buildImageVersions(panels: JobPanel[]): Map<string, number> {
  const versions = new Map<string, number>();
  for (const panel of panels) {
    if (!panel.image_url) continue;
    versions.set(panel.image_url, Math.max(versions.get(panel.image_url) ?? 0, panel.attempts || 0));
  }
  return versions;
}

export function versionedUrl(url: string, version: number | undefined): string {
  if (!version || url.startsWith("data:") || url.startsWith("blob:")) return url;
  return `${url}${url.includes("?") ? "&" : "?"}v=${version}`;
}

// ---------------------------------------------------------------------------
// Panel errors
// ---------------------------------------------------------------------------

const REDRAW_FAILED = /^redraw failed:?\s*/i;
// engines.byteplus.AccountError ends its message with "(ModelArk <code>)"
const ACCOUNT_ERROR =
  /\s*\(ModelArk (?:AuthenticationError|ModelNotOpen|InvalidEndpointOrModel|AccountOverdueError|QuotaExceeded|SetLimitExceeded|AccessDenied|OperationDenied|40[134])[^)]*\)\s*$/;

/** The latest redraw of this panel failed and the previous picture was kept. */
export function redrawFailed(panel: Pick<JobPanel, "error">): boolean {
  return !!panel.error && REDRAW_FAILED.test(panel.error);
}

/** Short, plain-language reason for a panel's `error` (null when there is none). */
export function panelErrorReason(error: string | null | undefined): string | null {
  if (!error) return null;
  const raw = error.replace(REDRAW_FAILED, "").trim();
  // An account problem the server already explained (a model not activated, no balance, a spending limit): its own
  // English sentence says what to do; waiting would not help, so it must never read as "busy".
  if (ACCOUNT_ERROR.test(raw)) return raw.replace(ACCOUNT_ERROR, "").split(" / ")[0].trim();
  if (/SensitiveContent|PolicyViolation/i.test(raw)) return "The image service's content filter blocked the picture.";
  if (/API key|HTTP 401\b|Unauthorized|Authentication/i.test(raw)) return "The API key is missing or was not accepted.";
  if (/not enabled/i.test(raw)) return "The image provider isn't enabled in the server config.";
  if (/HTTP 429\b|rate.?limit/i.test(raw)) return "The image service is busy right now.";
  if (/time[sd]? ?out/i.test(raw)) return "The image service took too long to answer.";
  if (/unreachable|connect|network/i.test(raw)) return "Couldn't reach the image service.";
  if (/HTTP 5\d\d\b/.test(raw)) return "The image service had a problem.";
  if (/interrupted by a server restart/i.test(raw)) return "The server restarted while this panel was being drawn.";
  if (/^cancelled$/i.test(raw)) return "It was stopped before it finished.";
  // Anything else: the server's message, without the exception name, JSON wrapping or request id.
  const message = /"message"\s*:\s*"([^"]+)"/.exec(raw)?.[1] ?? raw.replace(/^[\w.]*(?:Error|Exception)\s*:\s*/, "");
  let text = message.replace(/\s*Request id:.*$/i, "").trim() || "Something went wrong";
  if (text.length > 140) text = `${text.slice(0, 137).trimEnd()}…`;
  text = text.charAt(0).toUpperCase() + text.slice(1);
  return /[.!?…]$/.test(text) ? text : `${text}.`;
}

// ---------------------------------------------------------------------------
// Text
// ---------------------------------------------------------------------------

const NARRATOR_SPEAKERS = new Set(["", "narrator", "narration", "caption"]);

/** "papi" -> "Papi"; returns null for narrator-like speakers. */
export function speakerName(speaker: string): string | null {
  const s = (speaker ?? "").trim();
  if (NARRATOR_SPEAKERS.has(s.toLowerCase())) return null;
  return s.charAt(0).toUpperCase() + s.slice(1);
}

// Literal class names so Tailwind keeps them. All pass WCAG AA on white/cream.
const SPEAKER_COLORS = [
  "text-primary-700",
  "text-secondary-700",
  "text-emerald-700",
  "text-violet-700",
  "text-amber-700",
  "text-rose-700",
];

const speakerKey = (speaker: string) => (speaker ?? "").trim().toLowerCase();

/**
 * One colour per speaker for the whole manga, in order of first appearance, so
 * the main characters always get clearly different colours.
 */
export function buildSpeakerColors(panels: JobPanel[]): Map<string, string> {
  const colors = new Map<string, string>();
  for (const panel of panels) {
    for (const line of panel.dialogue ?? []) {
      const key = speakerKey(line.speaker);
      if (!key || NARRATOR_SPEAKERS.has(key) || colors.has(key)) continue;
      colors.set(key, SPEAKER_COLORS[colors.size % SPEAKER_COLORS.length]);
    }
  }
  return colors;
}

/** Colour for a speaker (from `buildSpeakerColors`, else a stable hash). */
export function speakerColor(speaker: string, colors?: Map<string, string>): string {
  const key = speakerKey(speaker);
  const assigned = colors?.get(key);
  if (assigned) return assigned;
  let hash = 0;
  for (let i = 0; i < key.length; i++) hash = (hash * 31 + key.charCodeAt(i)) >>> 0;
  return SPEAKER_COLORS[hash % SPEAKER_COLORS.length];
}

export function jobTitle(job: Pick<JobState, "title">): string {
  return job.title?.trim() || "Your manga";
}

// ---------------------------------------------------------------------------
// Information layer (shown as text, never drawn; absent in older jobs)
// ---------------------------------------------------------------------------

/** The panel's "More detail" text, trimmed ("" when there is none). */
export function panelDetail(panel: Pick<JobPanel, "detail">): string {
  return typeof panel.detail === "string" ? panel.detail.trim() : "";
}

/** Kana, CJK ideographs, Hangul, and CJK / full-width punctuation. */
const CJK_CHAR = /[\u3000-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff\uff00-\uffef]/;

/**
 * `lang` for one string of reader text. In a Chinese or Japanese job the backend keeps the English original when a
 * detail or card field has no translation, so a string without any CJK character is marked as English.
 */
export function textLang(text: string, jobLang: string | undefined): string | undefined {
  return jobLang && /^(zh|ja|ko)\b/i.test(jobLang) && !CJK_CHAR.test(text) ? "en" : jobLang;
}

export type PaperCardSectionId = keyof JobPaperCard;

/** The paper card's sections in reading order (`list`: shown as bullet points). */
export const PAPER_CARD_SECTIONS: ReadonlyArray<{ id: PaperCardSectionId; label: string; list: boolean }> = [
  { id: "problem", label: "Problem", list: false },
  { id: "key_idea", label: "Key idea", list: false },
  { id: "contributions", label: "What's new", list: true },
  { id: "how", label: "How it works", list: true },
  { id: "results", label: "Results", list: true },
  { id: "limitations", label: "Limitations", list: true },
  { id: "takeaway", label: "Takeaway", list: false },
];

export interface PaperCardSection {
  id: PaperCardSectionId;
  label: string;
  list: boolean;
  /** Never empty: the paragraph(s), or the bullet points. */
  items: string[];
}

function textItems(value: unknown): string[] {
  return (Array.isArray(value) ? value : [value])
    .map((v) => (typeof v === "string" || typeof v === "number" ? String(v).trim() : ""))
    .filter(Boolean);
}

/**
 * The card's sections that have content, in reading order; empty when there is no card (older jobs).
 * Tolerates a string where a list is expected and the other way round.
 */
export function paperCardSections(card: JobPaperCard | null | undefined): PaperCardSection[] {
  if (!card || typeof card !== "object") return [];
  return PAPER_CARD_SECTIONS.flatMap(({ id, label, list }) => {
    const items = textItems(card[id]);
    return items.length > 0 ? [{ id, label, list, items }] : [];
  });
}

export interface PointCoverage {
  covered: number;
  total: number;
}

/** A `[covered, total]` pair from `JobState.coverage`; null when missing, malformed or there are no such points. */
export function pointCoverage(pair: unknown): PointCoverage | null {
  if (!Array.isArray(pair) || pair.length < 2) return null;
  const covered = Math.round(Number(pair[0]));
  const total = Math.round(Number(pair[1]));
  if (!Number.isFinite(covered) || !Number.isFinite(total) || total <= 0) return null;
  return { covered: Math.min(Math.max(covered, 0), total), total };
}

// ---------------------------------------------------------------------------
// v3: words typeset into the picture
// ---------------------------------------------------------------------------

/** Pipeline v3 and later letter every picture in code: its words are exact, so the text under it is only a copy. */
export function isTypesetJob(job: Pick<JobState, "version">): boolean {
  return Number(job.version) >= 3;
}

/** Languages the panels are lettered in (the job's own first); one entry for single-language and older jobs. */
export function jobLanguages(job: Pick<JobState, "language" | "languages">): string[] {
  const langs = (job.languages ?? []).filter((l) => typeof l === "string" && l);
  return langs.length > 0 ? langs : [job.language];
}

/** Short toggle label of a reading language. */
export function languageLabel(lang: string): string {
  return lang.startsWith("zh") ? "中" : lang.startsWith("ja") ? "日" : "EN";
}

/** The reading language to show: the saved choice when this job has it, else the job's own language. */
export function readingLanguage(job: Pick<JobState, "language" | "languages">, preferred: string | null): string {
  return preferred && jobLanguages(job).includes(preferred) ? preferred : job.language;
}

/**
 * A panel as read in `lang`: the picture lettered in that language and its words, when the job has them (v3 bilingual
 * jobs); the panel itself otherwise. Nothing else changes, so every view works on the result as on the original.
 */
export function localizePanel(panel: JobPanel, lang: string, primary: string): JobPanel {
  if (lang === primary) return panel;
  const url = panel.image_urls?.[lang];
  const words = panel.texts?.[lang];
  if (!url && !words) return panel;
  return {
    ...panel,
    image_url: url ?? panel.image_url,
    ...(words
      ? {
          title: words.title ?? panel.title,
          dialogue: words.dialogue ?? panel.dialogue,
          narration: words.narration ?? panel.narration,
          board: words.board === undefined ? panel.board : words.board,
        }
      : {}),
  };
}

/** The job as read in `lang`: its long image and language follow the reading language. */
export function localizeJob(job: JobState, lang: string): JobState {
  if (lang === job.language) return job;
  const words = job.localized?.[lang];
  return {
    ...job,
    language: lang as JobState["language"],
    combined_url: job.combined_urls?.[lang] ?? job.combined_url,
    ...(words?.title ? { title: words.title } : {}),
    ...(words?.summary ? { summary: words.summary } : {}),
    ...(words?.paper_card ? { paper_card: words.paper_card } : {}),
    ...(words?.chapters?.length ? { chapters: words.chapters } : {}),
  };
}

/**
 * The abstract check's score as a whole percentage, or null when there is none (older jobs) or it gave no verdict.
 * The score is a fraction (0–1); a value above 1 is taken to be a percentage already.
 */
export function quizPercent(quiz: JobQuiz | null | undefined): number | null {
  const score = quiz?.score;
  if (typeof score !== "number" || !Number.isFinite(score) || score < 0 || score > 100) return null;
  return Math.round(score <= 1 ? score * 100 : score);
}

export interface BoardLine {
  text: string;
  /** The paper's own result, or a row / step the board highlights. */
  strong?: boolean;
  /** A heading inside the exhibit: table columns, the sides of a comparison. */
  muted?: boolean;
}

/** A board panel's exhibit as text, as the board prints it. */
export interface BoardSummary {
  title: string;
  subtitle: string;
  /** One line per bar, row, step or item, in the order they are drawn. */
  lines: BoardLine[];
  /** Grey footnote at the bottom left: unit and "higher is better" on bar charts, else the note. */
  caption: string;
  /** Grey footnote at the bottom right: the source (on word-links boards, the "illustration" tag). */
  source: string;
}

/** Words the board itself adds (backend `services/boards.py`, `_TXT`), by board language. */
const BOARD_WORDS = {
  en: { higher: "higher is better", lower: "lower is better", illustrative: "illustration (example weights)" },
  zh: { higher: "越高越好", lower: "越低越好", illustrative: "示意图（权重为举例）" },
};
const DIRECTION_RE = /\b(?:higher|lower|smaller|larger|bigger|less|more|fewer)\s+is\s+better\b|越(?:高|低|小|大|少|多)越好/i;
const LOWER_BETTER_RE = /\b(?:lower|smaller|less|fewer)\s+is\s+better\b|越(?:低|小|少)越好/i;

const SUPERSCRIPT: Record<string, string> = {
  "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹", "-": "⁻", "−": "⁻",
};

/** A board value as the board prints it: "2.3 · 10^19" -> "2.3 · 10¹⁹". */
export function displayValue(value: string): string {
  return value.replace(/\^([-−]?\d+)/g, (_, exp: string) => Array.from(exp, (c) => SUPERSCRIPT[c] ?? c).join(""));
}

function recordOf(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : null;
}

function listOf(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function indexSet(value: unknown): Set<number> {
  return new Set(listOf(value).map(Number).filter(Number.isInteger));
}

/**
 * A board's exhibit (`JobPanel.board`, normalized by the backend's `services/boards.py`) as text: what the picture's
 * chart, table or list says, for screen readers and the panel's "Text" section. Values stay the paper's exact strings;
 * the footnotes follow what the board prints in `language`. Null when there is no exhibit; malformed parts are skipped.
 */
export function boardSummary(board: unknown, language?: string): BoardSummary | null {
  const b = recordOf(board);
  if (!b) return null;
  const words = BOARD_WORDS[language?.toLowerCase().startsWith("zh") ? "zh" : "en"];
  const title = textItems(b.title)[0] ?? "";
  const subtitle = textItems(b.subtitle)[0] ?? "";
  let caption = textItems(b.note).join(" ");
  let source = textItems(b.source).join(" ");
  const lines: BoardLine[] = [];
  const add = (text: string, style: Omit<BoardLine, "text"> = {}) => {
    if (text) lines.push({ text, ...style });
  };

  switch (textItems(b.kind)[0]?.toLowerCase()) {
    case "bar": {
      for (const bar of listOf(b.bars).map(recordOf)) {
        const value = textItems(bar?.value)[0];
        if (!bar || !value) continue;
        add([...textItems(bar.label).slice(0, 1), displayValue(value)].join(": "), { strong: bar.highlight === true });
      }
      // The board says which way is better unless its title, subtitle or unit already does.
      const unit = textItems(b.unit)[0] ?? "";
      const said = `${title} ${subtitle} ${unit}`;
      const higher =
        typeof b.higher_is_better === "boolean"
          ? b.higher_is_better
          : typeof b.lower_is_better === "boolean"
            ? !b.lower_is_better
            : !LOWER_BETTER_RE.test(said);
      caption = [unit, DIRECTION_RE.test(said) ? "" : words[higher ? "higher" : "lower"]].filter(Boolean).join(" · ");
      break;
    }
    case "table": {
      // Empty cells stay as "–" so every row lines up with the columns.
      const cells = (row: unknown) => listOf(row).map((cell) => textItems(cell)[0] ?? "–");
      const columns = cells(b.columns);
      if (columns.some((c) => c !== "–")) add(columns.join(" · "), { muted: true });
      const highlighted = indexSet(b.highlight_rows);
      listOf(b.rows).forEach((row, i) => {
        const values = cells(Array.isArray(row) ? row : recordOf(row)?.cells);
        if (values.some((c) => c !== "–")) add(values.join(" · "), { strong: highlighted.has(i) });
      });
      break;
    }
    case "flow": {
      const highlighted = indexSet(b.highlight);
      let step = 0;
      listOf(b.steps).forEach((raw, i) => {
        const text = textItems(raw)[0];
        if (text) add(`${++step}. ${text}`, { strong: highlighted.has(i) });
      });
      break;
    }
    case "compare":
      // Left: the old way (crossed out on the board); right: the new way (ticked).
      for (const [side, mark] of [[b.left, "✗"], [b.right, "✓"]] as const) {
        const s = recordOf(side);
        if (!s) continue;
        add(textItems(s.label).join(" "), { muted: true });
        for (const item of textItems(s.items)) add(`${mark} ${item}`);
      }
      break;
    case "list":
      for (const item of textItems(b.items)) add(`✓ ${item}`);
      break;
    case "links": {
      // The example sentence, then the links drawn from its focus word: strong ones in red (at least half the largest
      // weight), weaker ones in blue (a fifth); the faint grey ones are left out, like the weights themselves.
      const tokens = listOf(b.tokens).map((t) => textItems(t)[0] ?? "");
      const weights = listOf(b.weights).map((w) => Number(w) || 0);
      const from = Number(b.from);
      add(tokens.filter(Boolean).join(" "));
      if (Number.isInteger(from) && tokens[from]) {
        const top = Math.max(0, ...weights.filter((_, i) => i !== from)) || 1;
        const linked = (keep: (rel: number) => boolean) =>
          tokens.filter((t, i) => i !== from && t && keep((weights[i] ?? 0) / top)).map((t) => `“${t}”`).join(", ");
        const strong = linked((rel) => rel >= 0.5);
        const weaker = linked((rel) => rel >= 0.2 && rel < 0.5);
        const parts = [strong && `strong to ${strong}`, weaker && `weaker to ${weaker}`].filter(Boolean);
        add(`Links from “${tokens[from]}”${parts.length > 0 ? `: ${parts.join("; ")}` : ""}`);
      }
      source = b.illustrative === false ? "" : words.illustrative;
      break;
    }
    case "diagram":
      // The drawing's words in the designer's order: what is written in its boxes and beside its things, and each
      // arrow's label after an arrow (the shapes themselves are the picture).
      for (const raw of listOf(b.elements)) {
        const e = recordOf(raw);
        if (!e) continue;
        const kind = textItems(e.type)[0]?.toLowerCase();
        const text = textItems(e.text)[0] ?? "";
        const label = textItems(e.label)[0] ?? "";
        if (kind === "arrow" || kind === "line") add(label && `${kind === "arrow" ? "→" : "—"} ${label}`);
        // a step badge (a small circle with a number) gives an order, not a word; a number in a box or on its own is a value
        else if (!(kind === "circle" && Number(e.r) <= 45 && /^\d{1,2}$/.test(text))) add(text);
      }
      break;
  }

  if (!title && !subtitle && lines.length === 0) return null;
  return { title, subtitle, lines, caption, source };
}

/** The exhibit drawn on a board panel (the backend draws `board` on board panels only), in the job's language. */
export function panelBoard(panel: Pick<JobPanel, "kind" | "board">, language?: string): BoardSummary | null {
  return panel.kind === "board" ? boardSummary(panel.board, language) : null;
}

/** The panel has words to show as text: narration, its board's exhibit, or (with `withDialogue`) its bubbles. */
export function panelHasWords(panel: JobPanel, withDialogue: boolean): boolean {
  return (withDialogue && (panel.dialogue?.length ?? 0) > 0) || !!panel.narration?.trim() || panelBoard(panel) !== null;
}

// ---------------------------------------------------------------------------
// Languages
// ---------------------------------------------------------------------------

/** Languages offered for new jobs; older jobs may also be "ja-JP". */
export const NEW_JOB_LANGUAGES: ReadonlyArray<JobLanguage> = ["en-US", "zh-CN"];

export function isNewJobLanguage(language: string): boolean {
  return (NEW_JOB_LANGUAGES as ReadonlyArray<string>).includes(language);
}

// ---------------------------------------------------------------------------
// Downloads
// ---------------------------------------------------------------------------

function slugify(text: string): string {
  return (
    text
      .toLowerCase()
      .replace(/[^a-z0-9぀-ヿ㐀-鿿]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .slice(0, 60) || "manga"
  );
}

function extensionOf(url: string): string {
  const dataMatch = /^data:image\/([a-z0-9.+-]+)/i.exec(url);
  if (dataMatch) return dataMatch[1].toLowerCase().startsWith("svg") ? "svg" : dataMatch[1].toLowerCase();
  const pathMatch = /\.([a-z0-9]{2,5})(?:[?#]|$)/i.exec(url);
  return pathMatch ? pathMatch[1].toLowerCase() : "jpg";
}

export function panelFileName(job: Pick<JobState, "title">, panel: JobPanel): string {
  const url = panel.image_url ?? "";
  return `${slugify(jobTitle(job))}-panel-${String(panel.n).padStart(2, "0")}.${extensionOf(url)}`;
}

export function combinedFileName(job: Pick<JobState, "title">, url: string): string {
  return `${slugify(jobTitle(job))}.${extensionOf(url)}`;
}

// ---------------------------------------------------------------------------
// Misc
// ---------------------------------------------------------------------------

export function clamp01(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.min(1, Math.max(0, value));
}

/** "3 minutes ago" style label for an ISO timestamp. */
export function formatRelativeTime(iso: string, now: number = Date.now()): string {
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return "";
  const seconds = (t - now) / 1000;
  const abs = Math.abs(seconds);
  if (abs < 45) return "just now";
  const rtf = new Intl.RelativeTimeFormat("en", { numeric: "auto" });
  if (abs < 3600) return rtf.format(Math.round(seconds / 60), "minute");
  if (abs < 86400) return rtf.format(Math.round(seconds / 3600), "hour");
  return rtf.format(Math.round(seconds / 86400), "day");
}
