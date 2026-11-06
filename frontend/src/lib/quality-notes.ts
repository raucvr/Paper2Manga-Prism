import type { JobQuality } from "./api";

export type QualityNoteKind = "fallback" | "skipped" | "unchecked_art" | "unchecked_idea" | "reader_review";

export interface QualityNoteSection {
  kind: QualityNoteKind;
  heading: string;
  explanation: string;
  items: string[];
}

export interface QualityNotes {
  /** The short name of the disclosure in the top bar. */
  label: string;
  /** Every reason, each kept apart: a backup model did the work, a step did not finish, a picture went unjudged. */
  sections: QualityNoteSection[];
}

/** Stored jobs are old or partial: only a real array counts, whatever the type says. */
function list<T>(value: readonly T[] | null | undefined): readonly T[] {
  return Array.isArray(value) ? value : [];
}

function panelList(value: readonly number[] | null | undefined): string[] {
  const panels = Array.from(new Set(list(value).filter((n) => Number.isFinite(n)))).sort((a, b) => a - b);
  return panels.length ? [`Panel${panels.length > 1 ? "s" : ""} ${panels.join(", ")}`] : [];
}

function skippedLine(entry: { stage?: string; error?: string } | null | undefined): string {
  const stage = entry?.stage?.trim() || "A step";
  const error = entry?.error?.trim();
  return error ? `${stage}: ${error}` : stage;
}

function overallLabel(sections: QualityNoteSection[]): string {
  const kinds = new Set(sections.map((s) => s.kind));
  if (kinds.size === 1 && kinds.has("reader_review")) return "Reading notes";
  if (kinds.has("reader_review")) return "Check details";
  if (kinds.size === 1 && kinds.has("fallback")) return "Backup model used";
  if (kinds.size === 1 && kinds.has("skipped")) return "Some steps incomplete";
  if (!kinds.has("fallback") && !kinds.has("skipped")) return "Some pictures unchecked";
  return "Check details";
}

/** What the reader should be told about a job's quality record; null when there is nothing to disclose (older jobs too). */
export function qualityNotes(quality: JobQuality | null | undefined): QualityNotes | null {
  const fallbacks = list(quality?.fallbacks).filter((note) => typeof note === "string" && note.trim());
  const skipped = list(quality?.skipped).map(skippedLine);
  const art = panelList(quality?.unchecked_art);
  const idea = panelList(quality?.unchecked_idea);

  const sections: QualityNoteSection[] = [];
  const reviews = quality?.reader_reviews;
  if (reviews && typeof reviews === "object" && !Array.isArray(reviews)) {
    const items: string[] = [];
    for (const [language, review] of Object.entries(reviews)) {
      if (!review || typeof review !== "object") continue;
      if (review.status === "reviewed") continue;
      const problems = list(review.problems).filter((p) => p && typeof p.problem === "string" && p.problem.trim());
      const prefix = `${language}: `;
      if (review.status === "needs_revision" && problems.length) {
        for (const problem of problems) {
          const panel = Number.isInteger(problem.n) && problem.n > 0 ? `Panel ${problem.n} — ` : "";
          items.push(`${prefix}${panel}${problem.problem}`);
        }
      } else {
        items.push(`${prefix}${review.status === "needs_revision" ? "Some explanations still need revision." : "The final wording has not received a usable reading review."}`);
      }
    }
    if (items.length) sections.push({
      kind: "reader_review",
      heading: "Explanations to revisit",
      explanation: "An AI read only the manga's words to look for missing explanations. These notes are not a test of what a person understood.",
      items,
    });
  }
  if (fallbacks.length) {
    sections.push({
      kind: "fallback",
      heading: "Backup model used",
      explanation:
        "A model assigned to part of this job could not take some requests, so the writer's model did that work instead. The work still ran, but where it was a check, the writer's model reviewed its own work, so the review may be less independent.",
      items: [...fallbacks],
    });
  }
  if (skipped.length) {
    sections.push({
      kind: "skipped",
      heading: "Steps not fully completed",
      // Checks, fix rounds and translation steps all land here, some of them only in part: say no more than that.
      explanation:
        "These automatic steps hit a problem and may not have been fully completed, so this manga may contain mistakes they would have caught or fixed.",
      items: skipped,
    });
  }
  if (art.length) {
    sections.push({
      kind: "unchecked_art",
      heading: "Art not checked",
      explanation: "These panels were published without a completed check of their art.",
      items: art,
    });
  }
  if (idea.length) {
    sections.push({
      kind: "unchecked_idea",
      heading: "Idea in the picture not checked",
      explanation: "These teaching pictures were published without a verdict on whether the picture shows its idea.",
      items: idea,
    });
  }
  return sections.length ? { label: overallLabel(sections), sections } : null;
}
