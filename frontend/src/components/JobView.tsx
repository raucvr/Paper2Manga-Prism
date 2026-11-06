"use client";

import React, { useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  Check,
  ChevronDown,
  CircleSlash,
  ClipboardCheck,
  Link2,
  Loader2,
  Plus,
  RotateCcw,
  Settings,
  WifiOff,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import { GenerationProgress, JobStatusBar } from "@/components/GenerationProgress";
import { Reader, type ReaderHandle } from "@/components/Reader";
import { api, describeError, type JobQuality, type JobQuiz, type JobState } from "@/lib/api";
import { isNewJobLanguage, isTerminal, jobTitle, quizPercent } from "@/lib/job-utils";
import { qualityNotes } from "@/lib/quality-notes";
import { useJob, type DemoSource } from "@/hooks/useJob";
import { buildJobForm, useAppStore } from "@/store/app-store";

interface JobViewProps {
  /** Job to poll (ignored when `demo` is given). */
  jobId: string | null;
  /** Local demo job: rendered as-is, no polling. */
  demo: DemoSource | null;
  /** Back to the upload screen. */
  onHome: () => void;
  /** Open another job (used after "try again" creates a new one). */
  onOpenJob: (jobId: string) => void;
}

function isPlainLeftClick(e: React.MouseEvent) {
  return e.button === 0 && !e.metaKey && !e.ctrlKey && !e.shiftKey && !e.altKey;
}

// ---------------------------------------------------------------------------
// Top bar
// ---------------------------------------------------------------------------

/** "Abstract check: 92%": how much of the paper's abstract a reader gets from the panels alone. Nothing without a score. */
function QuizBadge({ quiz }: { quiz: JobQuiz | null | undefined }) {
  const percent = quizPercent(quiz);
  if (percent === null) return null;
  const questions = quiz?.questions;
  const count = typeof questions === "number" && Number.isInteger(questions) && questions > 0 ? `${questions} ` : "";
  return (
    <span
      className="flex shrink-0 items-center gap-1 rounded-full bg-white/70 px-2 py-0.5 text-xs font-medium text-muted-foreground ring-1 ring-black/5"
      title={`Automatic check: an AI reader given only the panels' words scored ${percent}% on ${count}questions about the paper's abstract.`}
    >
      <ClipboardCheck className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
      <span className="sr-only sm:not-sr-only">Abstract check: </span>
      <span className="tabular-nums">{percent}%</span>
    </span>
  );
}

/**
 * What the reader should know about the job's checks; nothing for a job with nothing to disclose. A native disclosure, so
 * the details open by tap and keyboard and reach screen readers. The panel is placed against the top bar, to fit a phone.
 */
export function QualityNote({ quality }: { quality: JobQuality | null | undefined }) {
  const detailsRef = useRef<HTMLDetailsElement>(null);

  useEffect(() => {
    const closeOutside = (e: PointerEvent) => {
      const details = detailsRef.current;
      if (details?.open && !details.contains(e.target as Node)) details.open = false;
    };
    document.addEventListener("pointerdown", closeOutside);
    return () => document.removeEventListener("pointerdown", closeOutside);
  }, []);

  const notes = qualityNotes(quality);
  if (!notes) return null;

  const onKeyDown = (e: React.KeyboardEvent<HTMLDetailsElement>) => {
    // The reader's page keys listen on the window. Space must toggle the note, not turn the page; while the note is open
    // every key stays here, so the arrows scroll its details. Closed, the other keys still reach the reader.
    if (e.currentTarget.open || e.key === " " || e.key === "Spacebar") e.stopPropagation();
    if (e.key === "Escape" && e.currentTarget.open) {
      e.currentTarget.open = false;
      e.currentTarget.querySelector("summary")?.focus();
    }
  };

  return (
    <details ref={detailsRef} className="group shrink-0" onKeyDown={onKeyDown}>
      <summary className="flex h-10 min-w-10 cursor-pointer list-none items-center justify-center rounded-full focus-visible:outline-none [&::-webkit-details-marker]:hidden [&:focus-visible>span]:ring-2 [&:focus-visible>span]:ring-primary-400">
        <span className="flex items-center gap-1 rounded-full bg-amber-50 px-2 py-0.5 text-xs font-medium text-amber-800 ring-1 ring-amber-200">
          <AlertTriangle className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
          <span className="sr-only sm:not-sr-only">{notes.label}</span>
          <ChevronDown className="hidden h-3 w-3 shrink-0 transition-transform group-open:rotate-180 motion-reduce:transition-none sm:block" aria-hidden="true" />
        </span>
      </summary>
      <div
        tabIndex={0}
        className="absolute inset-x-2 top-full z-30 max-h-[70dvh] space-y-3 overflow-y-auto rounded-2xl bg-white p-3 text-xs text-foreground shadow-xl ring-1 ring-black/5 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-400 sm:left-auto sm:right-4 sm:w-96"
      >
        {notes.sections.map((section) => (
          <section key={section.kind}>
            <h2 className="font-semibold text-amber-800">{section.heading}</h2>
            <p className="mt-0.5 text-muted-foreground">{section.explanation}</p>
            <ul className="mt-1 list-disc space-y-0.5 pl-4">
              {section.items.map((item, i) => (
                <li key={i} className="break-words">
                  {item}
                </li>
              ))}
            </ul>
          </section>
        ))}
      </div>
    </details>
  );
}

function JobAppBar({
  title,
  lang,
  isDemo,
  quiz,
  quality,
  onHome,
}: {
  title: string;
  lang?: string;
  isDemo: boolean;
  /** v3 abstract check (absent in older jobs). */
  quiz?: JobQuiz | null;
  /** Which checks ran (absent in older jobs). */
  quality?: JobQuality | null;
  onHome: () => void;
}) {
  const setSettingsOpen = useAppStore((s) => s.setSettingsOpen);
  const [copyState, setCopyState] = useState<"idle" | "copied" | "failed">("idle");
  const resetTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const copyLink = async () => {
    try {
      await navigator.clipboard.writeText(window.location.href);
      setCopyState("copied");
    } catch {
      setCopyState("failed");
    }
    if (resetTimer.current) clearTimeout(resetTimer.current);
    resetTimer.current = setTimeout(() => setCopyState("idle"), 2000);
  };

  const iconButton =
    "flex h-10 w-10 shrink-0 items-center justify-center rounded-full transition-colors hover:bg-primary/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-400";

  return (
    <header className="relative flex h-[var(--appbar-h)] shrink-0 items-center gap-1 px-2 sm:gap-2 sm:px-4">
      <a
        href="/"
        onClick={(e) => {
          if (!isPlainLeftClick(e)) return;
          e.preventDefault();
          onHome();
        }}
        className="flex shrink-0 items-center gap-1.5 rounded-full px-2 py-1 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-400"
        aria-label="Prism home"
      >
        <span className="text-2xl" aria-hidden="true">
          🔮
        </span>
        <span className="hidden bg-gradient-to-r from-primary-500 to-secondary-500 bg-clip-text text-lg font-bold text-transparent sm:inline">
          Prism
        </span>
      </a>

      <h1 lang={lang} className="min-w-0 flex-1 truncate text-sm font-semibold sm:text-base" title={title}>
        {title}
      </h1>
      {isDemo && (
        <span className="shrink-0 rounded-full bg-secondary-100 px-2 py-0.5 text-xs font-medium text-secondary-800">
          Demo
        </span>
      )}
      <QuizBadge quiz={quiz} />
      <QualityNote quality={quality} />

      <button type="button" onClick={copyLink} className={iconButton} aria-label="Copy a link to this manga" title="Copy link">
        {copyState === "copied" ? (
          <Check className="h-5 w-5 text-emerald-600" aria-hidden="true" />
        ) : (
          <Link2 className="h-5 w-5" aria-hidden="true" />
        )}
      </button>
      <span className="sr-only" aria-live="polite">
        {copyState === "copied" ? "Link copied" : copyState === "failed" ? "Couldn't copy the link" : ""}
      </span>

      <Button size="sm" variant="outline" onClick={onHome} className="shrink-0 gap-1.5 px-3" aria-label="Start a new manga">
        <Plus className="h-4 w-4" aria-hidden="true" />
        <span className="hidden sm:inline">New manga</span>
      </Button>

      <button type="button" onClick={() => setSettingsOpen(true)} className={iconButton} aria-label="Open settings" title="Settings">
        <Settings className="h-5 w-5" aria-hidden="true" />
      </button>
    </header>
  );
}

// ---------------------------------------------------------------------------
// Building blocks
// ---------------------------------------------------------------------------

function Notice({
  tone,
  title,
  children,
  actions,
}: {
  tone: "error" | "neutral";
  title: string;
  children?: React.ReactNode;
  actions?: React.ReactNode;
}) {
  return (
    <div className="mx-auto w-full max-w-md px-4 py-10">
      <Card
        className={tone === "error" ? "border-red-200 bg-red-50 p-6 text-center" : "p-6 text-center"}
        role={tone === "error" ? "alert" : undefined}
      >
        <div className="mb-3 text-4xl" aria-hidden="true">
          {tone === "error" ? "😿" : "🌙"}
        </div>
        <h2 className={tone === "error" ? "text-lg font-semibold text-red-700" : "text-lg font-semibold"}>{title}</h2>
        {children && <div className="mt-2 text-sm text-muted-foreground">{children}</div>}
        {actions && <div className="mt-5 flex flex-wrap justify-center gap-2">{actions}</div>}
      </Card>
    </div>
  );
}

function BannerLine({
  tone,
  icon,
  children,
  action,
  onDismiss,
}: {
  tone: "error" | "warning" | "neutral" | "success";
  icon: React.ReactNode;
  children: React.ReactNode;
  action?: React.ReactNode;
  onDismiss?: () => void;
}) {
  const tones = {
    error: "bg-red-50 text-red-800 ring-red-200",
    warning: "bg-amber-50 text-amber-900 ring-amber-200",
    neutral: "bg-white/85 text-foreground ring-black/5",
    success: "bg-emerald-50 text-emerald-900 ring-emerald-200",
  } as const;
  return (
    <div
      role={tone === "error" ? "alert" : "status"}
      className={`flex items-center gap-2 rounded-2xl px-3 py-2 text-sm shadow-sm ring-1 ${tones[tone]}`}
    >
      <span className="shrink-0" aria-hidden="true">
        {icon}
      </span>
      <p className="min-w-0 flex-1">{children}</p>
      {action}
      {onDismiss && (
        <button
          type="button"
          onClick={onDismiss}
          className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full hover:bg-black/5 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-400"
          aria-label="Dismiss"
        >
          <X className="h-4 w-4" aria-hidden="true" />
        </button>
      )}
    </div>
  );
}

const SMALL_ACTION =
  "shrink-0 rounded-full bg-white px-3 py-1 text-xs font-semibold shadow-sm ring-1 ring-black/10 hover:bg-gray-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-400 disabled:opacity-60";

// ---------------------------------------------------------------------------
// Job view
// ---------------------------------------------------------------------------

export function JobView({ jobId, demo, onHome, onOpenJob }: JobViewProps) {
  const { job, loading, pollError, notFound, etaMs, actionError, clearActionError, regenerate, cancel, refresh } =
    useJob(jobId, demo);
  const readerRef = useRef<ReaderHandle>(null);
  const [retrying, setRetrying] = useState(false);
  const [retryError, setRetryError] = useState<string | null>(null);

  // An older job in a language that is no longer offered (ja-JP) goes back to the form to pick another.
  const canResubmit = (j: JobState) =>
    !demo && !!useAppStore.getState().jobFiles[j.job_id] && isNewJobLanguage(j.language);

  /** Failed job: send the same PDF again with the same options, or go back to the form pre-filled. */
  const retry = async () => {
    if (!job) return;
    const store = useAppStore.getState();
    store.applyJobOptions(job);
    const file = store.jobFiles[job.job_id];
    if (!file || !canResubmit(job)) {
      onHome();
      return;
    }
    setRetrying(true);
    setRetryError(null);
    try {
      const { job_id } = await api.createJob(buildJobForm(file, useAppStore.getState()));
      useAppStore.getState().rememberJobFile(job_id, file);
      onOpenJob(job_id);
    } catch (err) {
      setRetryError(describeError(err));
    } finally {
      setRetrying(false);
    }
  };

  const retryButton = (j: JobState) => (
    <Button onClick={retry} disabled={retrying} className="gap-2">
      {retrying ? (
        <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
      ) : (
        <RotateCcw className="h-4 w-4" aria-hidden="true" />
      )}
      {canResubmit(j) ? "Try again" : "Upload the PDF again"}
    </Button>
  );

  const title = job ? jobTitle(job) : loading ? "Loading…" : "Prism";
  let content: React.ReactNode;

  if (notFound) {
    content = (
      <Notice
        tone="error"
        title="We couldn't find this manga"
        actions={<Button onClick={onHome}>Start a new manga</Button>}
      >
        The link may be mistyped, or the server no longer has this job.
      </Notice>
    );
  } else if (!job) {
    content = loading ? (
      <div className="flex flex-col items-center gap-3 py-24 text-muted-foreground" role="status">
        <Loader2 className="h-8 w-8 animate-spin text-primary-500" aria-hidden="true" />
        <p>Loading your manga…</p>
      </div>
    ) : (
      <Notice
        tone="error"
        title="Couldn't load this manga"
        actions={
          <Button onClick={refresh} className="gap-2">
            <RotateCcw className="h-4 w-4" aria-hidden="true" />
            Try again
          </Button>
        }
      >
        {pollError} We&apos;ll keep trying in the background.
      </Notice>
    );
  } else if (job.panels.length === 0) {
    if (job.status === "failed") {
      content = (
        <Notice
          tone="error"
          title="This manga couldn't be made"
          actions={
            <>
              {retryButton(job)}
              <Button variant="outline" onClick={onHome}>
                Start over
              </Button>
            </>
          }
        >
          <p>{job.error || "Something went wrong on the server."}</p>
          {retryError && <p className="mt-2 font-medium text-red-700">{retryError}</p>}
        </Notice>
      );
    } else if (job.status === "cancelled") {
      content = (
        <Notice tone="neutral" title="This manga was stopped" actions={<Button onClick={onHome}>Start a new manga</Button>}>
          It was stopped before any panels were drawn.
        </Notice>
      );
    } else {
      content = (
        <>
          {pollError && (
            <div className="mx-auto max-w-2xl px-4 pt-4">
              <BannerLine tone="warning" icon={<WifiOff className="h-4 w-4" />}>
                Lost contact with the server. Retrying…
              </BannerLine>
            </div>
          )}
          <GenerationProgress job={job} etaMs={etaMs} onCancel={cancel} />
        </>
      );
    }
  } else {
    const failedPanels = job.panels.filter((p) => p.status === "failed").map((p) => p.n);
    const lines: React.ReactNode[] = [];
    if (actionError) {
      lines.push(
        <BannerLine key="action" tone="error" icon={<AlertTriangle className="h-4 w-4" />} onDismiss={clearActionError}>
          {actionError}
        </BannerLine>,
      );
    }
    if (pollError) {
      lines.push(
        <BannerLine
          key="poll"
          tone="warning"
          icon={<WifiOff className="h-4 w-4" />}
          action={
            <button type="button" onClick={refresh} className={SMALL_ACTION}>
              Retry now
            </button>
          }
        >
          Lost contact with the server. Retrying…
        </BannerLine>,
      );
    }
    if (!isTerminal(job.status)) {
      lines.push(<JobStatusBar key="status" job={job} etaMs={etaMs} onCancel={cancel} />);
    }
    if (job.status === "failed") {
      lines.push(
        <BannerLine
          key="failed"
          tone="error"
          icon={<AlertTriangle className="h-4 w-4" />}
          action={
            <button type="button" onClick={retry} disabled={retrying} className={SMALL_ACTION}>
              {retrying ? "Starting…" : canResubmit(job) ? "Try again" : "Start over"}
            </button>
          }
        >
          {job.error || "This manga stopped before it was finished."}
          {retryError ? ` (${retryError})` : ""}
        </BannerLine>,
      );
    }
    if (job.status === "cancelled") {
      lines.push(
        <BannerLine key="cancelled" tone="neutral" icon={<CircleSlash className="h-4 w-4" />}>
          Stopped early: {job.panels.filter((p) => p.status === "done").length} of {job.panels.length} panels were drawn.
        </BannerLine>,
      );
    }
    if (job.status === "completed" && failedPanels.length > 0) {
      lines.push(
        <BannerLine
          key="failed-panels"
          tone="warning"
          icon={<AlertTriangle className="h-4 w-4" />}
          action={
            <button type="button" onClick={() => readerRef.current?.goToPanel(failedPanels[0])} className={SMALL_ACTION}>
              Show me
            </button>
          }
        >
          {failedPanels.length === 1
            ? `Panel ${failedPanels[0]} couldn't be drawn. You can regenerate it.`
            : `${failedPanels.length} panels couldn't be drawn. You can regenerate them.`}
        </BannerLine>,
      );
    }
    const banner = lines.length > 0 ? <>{lines}</> : undefined;

    content = (
      <ErrorBoundary onReset={refresh}>
        <Reader key={job.job_id} ref={readerRef} job={job} banner={banner} onRegenerate={regenerate} />
      </ErrorBoundary>
    );
  }

  return (
    <div className="flex min-h-screen flex-col">
      <JobAppBar
        title={title}
        lang={job?.title ? job.language : undefined}
        isDemo={!!demo}
        quiz={job?.quiz}
        quality={job?.quality}
        onHome={onHome}
      />
      <main className="flex-1">{content}</main>
    </div>
  );
}
