/**
 * Typed client for the Prism backend.
 *
 * Requests go to `/api/...`, which next.config.js rewrites to the FastAPI
 * backend (NEXT_PUBLIC_API_URL, default http://localhost:8000), so the browser
 * never needs CORS and the backend port can change without a rebuild.
 * Generated images are served by the backend under `/output/...`, which is
 * proxied the same way.
 */

// ---------------------------------------------------------------------------
// Legacy types (`/api/manga/*`, kept for compatibility; the new UI uses jobs)
// ---------------------------------------------------------------------------

export interface UploadPDFResponse {
  filename: string;
  total_pages: number;
  full_text: string;
  text_preview: string;
  image_count: number;
  text_chunks: number;
}

export interface GenerateRequest {
  text: string;
  title?: string;
  character: string;
  language: string;
  /** Accepted by the backend for compatibility; the storyboarder decides the count. */
  num_panels?: number;
}

export interface StoryboardPanel {
  panel_number: number;
  panel_type: string;
  characters: string[];
  character_emotions: Record<string, string>;
  visual_description: string;
  dialogue: Record<string, string>;
  visual_metaphor: string;
  props: string[];
  background: string;
  layout_hint: string;
}

export interface StoryboardResponse {
  title: string;
  summary: string;
  character_theme: string;
  panel_count: number;
  panels: StoryboardPanel[];
}

export interface MangaPanel {
  panel_number: number;
  characters: string[];
  dialogue: Record<string, string>;
  image_base64: string;
}

export interface MangaResponse {
  title: string;
  character_theme: string;
  panel_count: number;
  panels: MangaPanel[];
  combined_image_base64?: string | null;
}

export interface ProgressResponse {
  stage: string;
  current_panel: number;
  total_panels: number;
  message: string;
  progress_percent: number;
}

// ---------------------------------------------------------------------------
// Config (`GET /api/config`, unchanged)
// ---------------------------------------------------------------------------

export interface ProviderStatus {
  name: string;
  enabled: boolean;
  has_api_key: boolean;
  models: string[];
}

export interface ConfigResponse {
  default_model: string;
  fallback_text_model: string;
  providers: ProviderStatus[];
  manga_settings: Record<string, unknown>;
  output_settings: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// Job API (`/api/jobs`, Prism v2)
// ---------------------------------------------------------------------------

export type JobStatus =
  | "queued"
  | "parsing"
  | "analyzing"
  | "storyboarding"
  | "translating"
  | "rendering"
  | "assembling"
  | "completed"
  | "failed"
  | "cancelled";

export type PanelStatus = "pending" | "rendering" | "qa" | "done" | "failed";

export type JobTheme = "kumomo";
/** New jobs are "en-US" or "zh-CN"; "ja-JP" is only found in older jobs. */
export type JobLanguage = "en-US" | "zh-CN" | "ja-JP";
export type JobLayout = "single" | "strip";
export type JobPanelRange = "auto" | "short" | "long";

export interface JobDialogueLine {
  speaker: string;
  text: string;
}

export interface JobPanelQA {
  score: number;
  text_exact: number;
  text_total: number;
  issues: string[];
  /** False when the quality check gave no usable verdict (the score means nothing then). Absent in older jobs. */
  available?: boolean;
}

/** v3: "scene" (the cast acting) or "board" (an exhibit drawn by code on a blank whiteboard in the scene). */
export type JobPanelKind = "scene" | "board";

/**
 * v3 board panels: the exhibit drawn on the board (bar chart, table, flow, compare, list or word links), with the
 * paper's exact values, in the job's language. Shapes follow the backend's `services/boards.py`; any field may be missing.
 */
export interface JobBoard {
  kind?: string;
  title?: string;
  subtitle?: string;
  source?: string;
  [key: string]: unknown;
}

/** v3: where the words were typeset on the panel picture (pixel rects in that image). Not used by the reader yet. */
export interface JobPanelLayout {
  version?: number;
  size?: [number, number];
  narration?: { rect: number[]; font_size: number } | null;
  bubbles?: Array<{ speaker: string; rect: number[]; tail?: unknown; font_size: number; lines: string[] }>;
  chart?: number[] | null;
  /** The words didn't fit on the art and went into a caption strip under it. */
  fallback?: boolean;
  [key: string]: unknown;
}

export interface JobPanel {
  n: number;
  /** Chapter id (matches `JobChapter.id`). */
  chapter: string;
  title: string;
  characters: string[];
  dialogue: JobDialogueLine[];
  narration: string;
  status: PanelStatus;
  image_url: string | null;
  width: number | null;
  height: number | null;
  /**
   * Image version (cumulative): goes up when a new picture is published, never before the file is written, so
   * `image_url?v=<attempts>` changes whenever the file does. A render whose attempts all failed also bumps it once.
   */
  attempts: number;
  /** Image attempts made by the latest render of this panel (1 = the first attempt passed). Absent in older jobs. */
  render_attempts?: number;
  qa: JobPanelQA | null;
  /**
   * Why the latest render of this panel failed. `"redraw failed: <reason>"` means a redraw failed and the
   * previous picture was kept. Cleared (absent or null) after the next successful render.
   */
  error?: string | null;
  /** Strip layout only: which half (0 = left, 1 = right) of the shared `image_url` this panel is. */
  strip_index?: 0 | 1 | null;
  /** Strip layout only: size of the shared two-panel image. */
  strip_width?: number | null;
  strip_height?: number | null;
  /**
   * The precise facts behind the panel (1–3 sentences: exact numbers, settings, the mechanism in technical terms),
   * in the job's language. Shown as text under the picture, never drawn. Absent in older jobs; may be empty.
   */
  detail?: string;
  /** Ids of the paper's key points this panel explains (not shown to readers). Absent in older jobs. */
  covers?: string[];
  /** Scene or board. Absent in older jobs (all scenes). */
  kind?: JobPanelKind;
  /** The exhibit of a board panel; null (or absent) for scenes. */
  board?: JobBoard | null;
  /** v3: the panel's story beat (hook, problem, idea, how, proof, limits, payoff, recap); `chapter` follows from it. */
  beat?: string;
  /** v3: indexes of the abstract's claims this panel conveys (not shown to readers). */
  claims?: number[];
  /** v3: the picture before lettering (no words); `image_url` is the lettered panel. Null until drawn. */
  art_url?: string | null;
  /** v3: where the words were typeset on `image_url`; null until drawn. */
  layout?: JobPanelLayout | null;
  /**
   * v3 bilingual jobs: the same picture lettered in each language of `JobState.languages` (the job's own language is
   * `image_url` too). Absent in single-language and older jobs.
   */
  image_urls?: Record<string, string> | null;
  /** v3 bilingual jobs: the panel's words in each language (the reader's text copy and screen-reader description). */
  texts?: Record<string, JobPanelTexts> | null;
}

/** A bilingual job's other-language words outside the panels. */
export interface JobLocalized {
  title?: string;
  summary?: string;
  paper_card?: JobPaperCard | null;
  chapters?: JobChapter[];
}

/** One panel's words in one language (v3 bilingual jobs). */
export interface JobPanelTexts {
  title?: string;
  dialogue?: JobDialogueLine[];
  narration?: string;
  board?: JobBoard | null;
}

export interface JobChapter {
  id: string;
  title: string;
  panel_numbers: number[];
}

export interface JobProgress {
  done: number;
  total: number;
  percent: number;
}

export interface JobCost {
  images_generated: number;
  image_tokens: number;
  text_prompt_tokens: number;
  text_completion_tokens: number;
  elapsed_seconds: number;
}

/** The whole paper in a few lines, in the job's language. Shown as text in the reader; any field may be missing. */
export interface JobPaperCard {
  problem?: string;
  key_idea?: string;
  contributions?: string[];
  how?: string[];
  results?: string[];
  limitations?: string[];
  takeaway?: string;
}

/**
 * v3 abstract check: questions about the paper's abstract, answered by an AI reader who saw only the panels' visible
 * text (narration, dialogue, board text and values), then graded strictly (full, half or no credit per question).
 */
export interface JobQuiz {
  /** Average grade, 0–1; null when the check gave no verdict (see `error`). */
  score: number | null;
  /** How many questions were asked. */
  questions?: number;
  /** Ids of the abstract's claims whose questions did not get full credit. */
  failed_claim_ids?: number[];
  error?: string | null;
}

/** How many of the paper's key points the panels explain, as `[covered, total]` per priority. */
export interface JobCoverage {
  core: [number, number];
  important: [number, number];
}

export interface JobState {
  job_id: string;
  status: JobStatus;
  stage_message: string;
  created_at: string;
  updated_at: string;
  title: string;
  summary: string;
  language: JobLanguage;
  theme: JobTheme;
  layout: JobLayout;
  progress: JobProgress;
  chapters: JobChapter[];
  /** Empty until the storyboard exists; afterwards one entry per panel. */
  panels: JobPanel[];
  /**
   * Set only while a user-triggered redraw runs: the panels being redrawn (both panels of a strip pair).
   * Then no other panel is drawn, and `status` is "rendering" until the redraw ends.
   */
  redraw_panels?: number[] | null;
  /** Set when `status === "completed"`. */
  combined_url: string | null;
  /** v3 bilingual jobs: the long image in each other language (the job's own language is `combined_url`). */
  combined_urls?: Record<string, string> | null;
  /** v3: every language the panels are lettered in, the job's own language first. Absent in older jobs. */
  languages?: string[] | null;
  /** v3 bilingual jobs: title, summary, chapter titles and paper card in each other language. */
  localized?: Record<string, JobLocalized> | null;
  storyboard_url: string | null;
  cost: JobCost;
  error: string | null;
  /** Summary of the whole paper. Absent (or null) in older jobs and before the storyboard exists. */
  paper_card?: JobPaperCard | null;
  /** How much of the paper the panels cover. Absent (or null) in older jobs. */
  coverage?: JobCoverage | null;
  /** Pipeline version: 3 = text-free art with the words typeset into the picture by code. Absent in older jobs. */
  version?: number;
  /** v3: the abstract check's result. Absent (or null) in older jobs and before the storyboard exists. */
  quiz?: JobQuiz | null;
  /** What was checked: steps that did not finish, backup-model use and pictures left unchecked. Absent in older jobs. */
  quality?: JobQuality | null;
}

export interface JobQuality {
  /** Final visible text, by language. An AI diagnostic, never a human comprehension score. */
  reader_reviews?: Record<string, {
    status: "reviewed" | "needs_revision" | "unassessed";
    language: string;
    fingerprint: string;
    answers?: { question: string; answer: string; evidence: { n: number; quote: string }[] }[];
    problems?: { n: number; kind: string; problem: string; fix: string }[];
    reason?: string;
  }>;
  /** Checks that could not run, e.g. {stage: "fact-check", error: "TimeoutError: ..."}. */
  skipped?: { stage: string; error: string }[];
  /** Problems the deterministic script checks still saw after the fix rounds. */
  lint_left?: number;
  /** Sentences removed because the paper does not support their numbers. */
  scrubbed?: number;
  /** Panels lettered without a check of the art (the vision call failed). */
  unchecked_art?: number[];
  /** Teaching scenes whose picture was published without a verdict on its idea. May overlap `unchecked_art`. */
  unchecked_idea?: number[];
  /** Work done by the writer's model because a role's own model could not take it: the work ran, on a backup model. */
  fallbacks?: string[];
}

export interface JobSummary {
  job_id: string;
  status: JobStatus;
  title: string;
  created_at: string;
  progress: JobProgress;
  language: JobLanguage;
  theme: JobTheme;
  /** Present only while a user-triggered redraw runs (as in `JobState`). */
  redraw_panels?: number[] | null;
}

export interface JobListResponse {
  jobs: JobSummary[];
}

export interface CreateJobResponse {
  job_id: string;
  status: JobStatus;
}

export interface RegeneratePanelResponse {
  status: "queued";
}

export interface CancelJobResponse {
  status: "cancelled";
}

// ---------------------------------------------------------------------------
// Transport
// ---------------------------------------------------------------------------

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "/api";

/** Non-2xx response; `detail` is the backend's human-readable message. */
export class ApiError extends Error {
  readonly status: number;
  readonly detail: string;

  constructor(status: number, detail: string) {
    super(`Request failed (${status}): ${detail}`);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

/** FastAPI errors: `{"detail": "..."}`, or a list of validation errors for 422s. */
function formatDetail(detail: unknown): string {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((item) =>
        item && typeof item === "object" && "msg" in item ? String((item as { msg: unknown }).msg) : JSON.stringify(item),
      )
      .join("; ");
  }
  return JSON.stringify(detail);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init);
  if (!res.ok) {
    let detail: string | null = null;
    try {
      const body: unknown = await res.json();
      detail =
        body && typeof body === "object" && "detail" in body
          ? formatDetail((body as { detail: unknown }).detail)
          : JSON.stringify(body);
    } catch {
      /* non-JSON error body, e.g. the dev proxy's page when the backend is down */
    }
    if (!detail) {
      detail =
        res.status >= 500
          ? `The Prism server isn't responding (HTTP ${res.status}). Check that the backend is running, then try again.`
          : res.statusText || `HTTP ${res.status}`;
    }
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as T;
}

function postJSON<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

const jobPath = (jobId: string) => `/jobs/${encodeURIComponent(jobId)}`;

/**
 * `POST /api/jobs` (multipart). Fields: `file` (PDF) or `text`, plus optional
 * `title`, `theme`, `language`, `layout`, `qa`, `panel_range`, `fast`.
 * Do not set a Content-Type header: the browser adds the multipart boundary.
 */
export function createJob(form: FormData): Promise<CreateJobResponse> {
  return request<CreateJobResponse>("/jobs", { method: "POST", body: form });
}

/** `GET /api/jobs/{id}`; never cached, so polling always sees fresh state. */
export function getJob(jobId: string, init?: { signal?: AbortSignal }): Promise<JobState> {
  return request<JobState>(jobPath(jobId), { cache: "no-store", signal: init?.signal });
}

/** `GET /api/jobs` — the latest 20 jobs. */
export function listJobs(init?: { signal?: AbortSignal }): Promise<JobListResponse> {
  return request<JobListResponse>("/jobs", { cache: "no-store", signal: init?.signal });
}

/** `POST /api/jobs/{id}/panels/{n}/regenerate` — the panel goes back to `rendering`, then `done`. */
export function regeneratePanel(jobId: string, n: number): Promise<RegeneratePanelResponse> {
  return request<RegeneratePanelResponse>(`${jobPath(jobId)}/panels/${n}/regenerate`, {
    method: "POST",
  });
}

/** `POST /api/jobs/{id}/cancel`. */
export function cancelJob(jobId: string): Promise<CancelJobResponse> {
  return request<CancelJobResponse>(`${jobPath(jobId)}/cancel`, { method: "POST" });
}

export const api = {
  // --- Job API (Prism v2) ---
  createJob,
  getJob,
  listJobs,
  regeneratePanel,
  cancelJob,

  // --- Legacy endpoints (kept for compatibility) ---
  uploadPDF(file: File): Promise<UploadPDFResponse> {
    const form = new FormData();
    form.append("file", file);
    return request<UploadPDFResponse>("/manga/upload-pdf", { method: "POST", body: form });
  },

  generateStoryboard(req: GenerateRequest): Promise<StoryboardResponse> {
    return postJSON<StoryboardResponse>("/manga/storyboard", req);
  },

  generateManga(req: GenerateRequest): Promise<MangaResponse> {
    return postJSON<MangaResponse>("/manga/generate", req);
  },

  getProgress(): Promise<ProgressResponse> {
    return request<ProgressResponse>("/manga/progress", { cache: "no-store" });
  },

  getConfig(): Promise<ConfigResponse> {
    return request<ConfigResponse>("/config");
  },
};

/** Turn any thrown value into a short message suitable for the UI. */
export function describeError(err: unknown): string {
  if (err instanceof ApiError) return err.detail || err.message;
  if (err instanceof TypeError) {
    // fetch() rejects with a TypeError when the server can't be reached.
    return "Can't reach the Prism server. Check that the backend is running, then try again.";
  }
  if (err instanceof Error) return err.message;
  return "Something went wrong. Please try again.";
}
