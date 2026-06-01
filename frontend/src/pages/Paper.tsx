import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api, API_BASE } from "../api";
import type { PaperPayload } from "../types";
import { Card, Spinner } from "../components/ui";

export default function Paper() {
  const { id } = useParams<{ id: string }>();
  const [paper, setPaper] = useState<PaperPayload | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!id) return;
    api
      .getPaper(id)
      .then(setPaper)
      .catch((e) => setError(e.message));
  }, [id]);

  return (
    <div className="h-full overflow-y-auto scrollbar-thin">
      <div className="sticky top-0 z-10 flex items-center justify-between border-b border-border bg-surface/95 px-6 py-3 backdrop-blur">
        <Link to={`/campaign/${id}`} className="text-sm text-text-secondary hover:text-text-primary">
          ← Back to campaign
        </Link>
        <div className="flex gap-2">
          {paper?.tex_url && (
            <a
              href={paper.tex_url}
              target="_blank"
              rel="noreferrer"
              className="rounded-lg border border-border px-3 py-1.5 text-sm text-text-secondary hover:bg-raised"
            >
              TeX
            </a>
          )}
          {paper?.pdf_url && (
            <a
              href={paper.pdf_url}
              target="_blank"
              rel="noreferrer"
              className="rounded-lg bg-brand px-3 py-1.5 text-sm font-medium text-white hover:bg-brand/90"
            >
              Download PDF
            </a>
          )}
        </div>
      </div>

      <div className="mx-auto max-w-3xl px-8 py-10">
        {error && (
          <Card className="border-warning/40 p-4 text-sm text-warning">
            Paper not ready yet ({error}). It’s generated when the campaign finishes.
          </Card>
        )}
        {!paper && !error && <Spinner />}
        {paper && (
          <article className="space-y-8">
            {paper.abstract_latex && (
              <Section title="Abstract">
                <p className="leading-relaxed text-text-primary">{clean(paper.abstract_latex)}</p>
              </Section>
            )}
            {paper.methods_latex && <LatexSection raw={paper.methods_latex} />}
            {paper.results_latex && <LatexSection raw={paper.results_latex} />}
            {(paper.full_tex_chars || paper.figures_embedded != null) && (
              <p className="border-t border-border pt-4 text-xs text-text-muted">
                {paper.full_tex_chars ? `${paper.full_tex_chars} chars of LaTeX` : ""}
                {paper.figures_embedded != null ? ` · ${paper.figures_embedded} figures` : ""}
                {paper.pdf_url && (
                  <> · PDF/TeX links resolve to the stack’s object store ({hostOf(paper.pdf_url)}).</>
                )}
              </p>
            )}
          </article>
        )}
      </div>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section>
      <h2 className="mb-2 text-lg font-semibold text-text-primary">{title}</h2>
      {children}
    </section>
  );
}

// Render a latex block by splitting on \section/\subsection and cleaning commands.
function LatexSection({ raw }: { raw: string }) {
  const parts = raw.split(/\\section\{([^}]*)\}/);
  // parts: [pre, title1, body1, title2, body2, ...]
  const blocks: { title: string; body: string }[] = [];
  for (let i = 1; i < parts.length; i += 2) {
    blocks.push({ title: parts[i], body: parts[i + 1] ?? "" });
  }
  if (blocks.length === 0) return <p className="leading-relaxed">{clean(raw)}</p>;
  return (
    <>
      {blocks.map((b, i) => (
        <Section key={i} title={clean(b.title)}>
          <div className="space-y-3">
            {b.body
              .split(/\\subsection\{([^}]*)\}/)
              .map((seg, idx) =>
                idx % 2 === 1 ? (
                  <h3 key={idx} className="text-base font-semibold text-text-primary">
                    {clean(seg)}
                  </h3>
                ) : (
                  seg.trim() && (
                    <p key={idx} className="whitespace-pre-wrap leading-relaxed text-text-secondary">
                      {clean(seg)}
                    </p>
                  )
                ),
              )}
          </div>
        </Section>
      ))}
    </>
  );
}

// Minimal LaTeX → readable text. Keeps inline math markers, drops table scaffolding.
function clean(s: string): string {
  return s
    .replace(/\\begin\{[^}]*\}(\[[^\]]*\])?/g, "")
    .replace(/\\end\{[^}]*\}/g, "")
    .replace(/\\(texttt|emph|textbf|textit|mathrm|text)\{([^}]*)\}/g, "$2")
    .replace(/\\caption\{([^}]*)\}/g, "")
    .replace(/\\hline/g, "")
    .replace(/\\\\/g, "\n")
    .replace(/&/g, "  ")
    .replace(/\\%/g, "%")
    .replace(/\\_/g, "_")
    .replace(/\\\$/g, "$")
    .replace(/\\([a-zA-Z]+)/g, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function hostOf(url: string): string {
  try {
    return new URL(url).host;
  } catch {
    return "object store";
  }
}
