import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../api";
import type { PaperPayload } from "../types";

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
    <main className="flex min-w-0 flex-1 flex-col bg-center">
      <div className="flex shrink-0 items-center justify-between border-b border-line px-[26px] py-3">
        <Link to={`/campaign/${id}`} className="text-[12.5px] text-ink-2 hover:text-ink">
          ← Back to campaign
        </Link>
        <div className="flex gap-2">
          {paper?.tex_url && (
            <a
              href={paper.tex_url}
              target="_blank"
              rel="noreferrer"
              className="rounded-lg border border-edge px-3 py-1.5 text-[12.5px] text-ink-2 hover:bg-rowhover"
            >
              TeX
            </a>
          )}
          {paper?.pdf_url && (
            <a
              href={paper.pdf_url}
              target="_blank"
              rel="noreferrer"
              className="rounded-lg px-3 py-1.5 text-[12.5px] font-semibold"
              style={{ background: "var(--text)", color: "var(--centerBg)" }}
            >
              Download PDF
            </a>
          )}
        </div>
      </div>

      <div className="pp-scroll min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto max-w-3xl px-8 py-10">
          {error && (
            <div
              className="rounded-[9px] border px-4 py-3 text-[12.5px] leading-relaxed"
              style={{ borderColor: "var(--border)", color: "var(--text2)" }}
            >
              Paper not ready yet ({error}). It’s generated when the campaign finishes.
            </div>
          )}
          {!paper && !error && <div className="text-[12.5px] text-ink-3">Loading paper…</div>}
          {paper && (
            <article className="space-y-8">
              {paper.abstract_latex && (
                <Section title="Abstract">
                  <p className="text-[14px] leading-[1.7] text-ink">{clean(paper.abstract_latex)}</p>
                </Section>
              )}
              {paper.methods_latex && <LatexSection raw={paper.methods_latex} />}
              {paper.results_latex && <LatexSection raw={paper.results_latex} />}
              {(paper.full_tex_chars || paper.figures_embedded != null) && (
                <p className="border-t border-line pt-4 font-mono text-[11px] text-ink-3">
                  {paper.full_tex_chars ? `${paper.full_tex_chars} chars of LaTeX` : ""}
                  {paper.figures_embedded != null ? ` · ${paper.figures_embedded} figures` : ""}
                </p>
              )}
            </article>
          )}
        </div>
      </div>
    </main>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section>
      <h2 className="mb-2 text-[17px] font-semibold text-ink">{title}</h2>
      {children}
    </section>
  );
}

// Render a latex block by splitting on \section/\subsection and cleaning commands.
function LatexSection({ raw }: { raw: string }) {
  const parts = raw.split(/\\section\{([^}]*)\}/);
  const blocks: { title: string; body: string }[] = [];
  for (let i = 1; i < parts.length; i += 2) {
    blocks.push({ title: parts[i], body: parts[i + 1] ?? "" });
  }
  if (blocks.length === 0) return <p className="text-[14px] leading-[1.7] text-ink-2">{clean(raw)}</p>;
  return (
    <>
      {blocks.map((b, i) => (
        <Section key={i} title={clean(b.title)}>
          <div className="space-y-3">
            {b.body
              .split(/\\subsection\{([^}]*)\}/)
              .map((seg, idx) =>
                idx % 2 === 1 ? (
                  <h3 key={idx} className="text-[15px] font-semibold text-ink">
                    {clean(seg)}
                  </h3>
                ) : (
                  seg.trim() && (
                    <p key={idx} className="whitespace-pre-wrap text-[14px] leading-[1.7] text-ink-2">
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
