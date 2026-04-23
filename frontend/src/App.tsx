import { useCallback, useMemo, useState } from "react";

const apiBase = () => (import.meta.env.VITE_API_URL || "").replace(/\/$/, "");

async function postResearch(question: string) {
  const base = apiBase();
  const res = await fetch(`${base || ""}/research`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  return res.json() as Promise<{ session_id: string; stream_url: string; status: string }>;
}

async function fetchJson<T>(path: string): Promise<T> {
  const base = apiBase();
  const res = await fetch(`${base || ""}${path}`);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  return res.json() as Promise<T>;
}

type Tab = "sse" | "session" | "prior" | "hypotheses" | "trace" | "llm" | "paper";

type PaperApiResponse = {
  session_id: string;
  paper: {
    pdf_url?: string | null;
    tex_url?: string | null;
    methods_latex?: string;
    results_latex?: string;
    full_tex_chars?: number;
    figures_embedded?: number;
  };
};

type LlmCallRow = {
  id: string;
  call_purpose: string | null;
  model: string | null;
  prompt_text: string;
  response_text: string;
  duration_ms: number | null;
};

function PaperDownloadLinks({ raw }: { raw: string }) {
  try {
    const data = JSON.parse(raw) as PaperApiResponse;
    const p = data.paper;
    if (!p) {
      return <p className="muted">No paper payload in response.</p>;
    }
    return (
      <div style={{ marginBottom: "1rem", display: "flex", flexWrap: "wrap", gap: "0.75rem", alignItems: "center" }}>
        {p.pdf_url ? (
          <a className="primary" href={p.pdf_url} target="_blank" rel="noreferrer">
            Open PDF
          </a>
        ) : (
          <span className="muted">PDF not available (pdflatex or MinIO).</span>
        )}
        {p.tex_url ? (
          <a href={p.tex_url} target="_blank" rel="noreferrer">
            Open main.tex
          </a>
        ) : null}
        {typeof p.figures_embedded === "number" ? (
          <span className="muted">
            Figures embedded: {p.figures_embedded} · LaTeX chars: {p.full_tex_chars ?? "—"}
          </span>
        ) : null}
      </div>
    );
  } catch {
    return null;
  }
}

function LlmCallInspector({ raw }: { raw: string }) {
  try {
    const data = JSON.parse(raw) as { llm_calls: LlmCallRow[] };
    const rows = data.llm_calls ?? [];
    if (rows.length === 0) {
      return <p className="muted">No LLM rows for this session.</p>;
    }
    return (
      <div style={{ marginBottom: "1rem", display: "flex", flexDirection: "column", gap: "0.5rem" }}>
        {rows.map((c) => (
          <details key={c.id} className="card" style={{ padding: "0.5rem 0.75rem" }}>
            <summary>
              <strong>{c.call_purpose || "call"}</strong>
              {c.model ? <span className="muted"> · {c.model}</span> : null}
              {c.duration_ms != null ? <span className="muted"> · {c.duration_ms}ms</span> : null}
            </summary>
            <div style={{ marginTop: "0.5rem" }}>
              <div className="muted" style={{ fontSize: "0.85rem" }}>
                Prompt (preview)
              </div>
              <pre className="json-block" style={{ maxHeight: "12rem", overflow: "auto" }}>
                {(c.prompt_text || "").slice(0, 8000)}
              </pre>
              <div className="muted" style={{ fontSize: "0.85rem" }}>
                Response (preview)
              </div>
              <pre className="json-block" style={{ maxHeight: "12rem", overflow: "auto" }}>
                {(c.response_text || "").slice(0, 8000)}
              </pre>
            </div>
          </details>
        ))}
      </div>
    );
  } catch {
    return null;
  }
}

export function App() {
  const [question, setQuestion] = useState(
    "Does transformer attention efficiency degrade non-linearly with sequence length?",
  );
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [events, setEvents] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [tab, setTab] = useState<Tab>("sse");
  const [dataJson, setDataJson] = useState<string>("");
  const [dataLoading, setDataLoading] = useState(false);
  const [dataError, setDataError] = useState<string | null>(null);

  const streamUrl = useMemo(() => {
    if (!sessionId) return null;
    const base = apiBase();
    return `${base || ""}/stream/${sessionId}`;
  }, [sessionId]);

  const startStream = (sid: string) => {
    const base = apiBase();
    const url = `${base || ""}/stream/${sid}`;
    const es = new EventSource(url);
    es.onmessage = (ev) => {
      setEvents((prev) => [...prev, ev.data]);
    };
    es.onerror = () => {
      es.close();
    };
  };

  const loadTab = useCallback(async () => {
    if (!sessionId) return;
    setDataLoading(true);
    setDataError(null);
    try {
      const paths: Record<Exclude<Tab, "sse">, string> = {
        session: `/sessions/${sessionId}`,
        prior: `/sessions/${sessionId}/prior`,
        hypotheses: `/sessions/${sessionId}/hypotheses`,
        trace: `/sessions/${sessionId}/trace`,
        llm: `/sessions/${sessionId}/llm-calls`,
        paper: `/sessions/${sessionId}/paper`,
      };
      if (tab === "sse") {
        setDataJson("");
        return;
      }
      const raw = await fetchJson<unknown>(paths[tab]);
      setDataJson(JSON.stringify(raw, null, 2));
    } catch (e) {
      setDataError(e instanceof Error ? e.message : String(e));
      setDataJson("");
    } finally {
      setDataLoading(false);
    }
  }, [sessionId, tab]);

  const onSubmit = async () => {
    setError(null);
    setEvents([]);
    setDataJson("");
    setBusy(true);
    try {
      const data = await postResearch(question);
      setSessionId(data.session_id);
      startStream(data.session_id);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="layout">
      <h1>Propab</h1>
      <p className="card">Submit a research question. Events stream live from the API (SSE). Use the Session data tabs to inspect stored state (ARCHITECTURE §12).</p>

      <div className="card">
        <label htmlFor="q">Question</label>
        <textarea id="q" rows={4} value={question} onChange={(e) => setQuestion(e.target.value)} />
        <div style={{ marginTop: "0.75rem" }}>
          <button className="primary" type="button" disabled={busy || question.trim().length < 8} onClick={onSubmit}>
            {busy ? "Starting…" : "Start research"}
          </button>
        </div>
        {error ? <p style={{ color: "#b91c1c" }}>{error}</p> : null}
        {sessionId ? (
          <p style={{ marginTop: "0.75rem" }}>
            <strong>Session:</strong> {sessionId}
            <br />
            <strong>Stream:</strong> {streamUrl}
          </p>
        ) : null}
      </div>

      {sessionId ? (
        <div className="card">
          <h2 style={{ marginTop: 0 }}>Session explorer</h2>
          <div className="tabs">
            {(
              [
                ["sse", "Live SSE"],
                ["session", "Session row"],
                ["prior", "Prior JSON"],
                ["hypotheses", "Hypotheses"],
                ["trace", "Experiment trace"],
                ["llm", "LLM calls"],
                ["paper", "Paper (PDF / TeX)"],
              ] as const
            ).map(([id, label]) => (
              <button
                key={id}
                type="button"
                className={tab === id ? "tab active" : "tab"}
                onClick={() => {
                  setTab(id);
                  setDataJson("");
                  setDataError(null);
                }}
              >
                {label}
              </button>
            ))}
            {tab !== "sse" ? (
              <button className="primary" type="button" style={{ marginLeft: "0.5rem" }} disabled={dataLoading} onClick={() => void loadTab()}>
                {dataLoading ? "Loading…" : "Load"}
              </button>
            ) : null}
          </div>
          {tab !== "sse" && dataError ? <p style={{ color: "#b91c1c" }}>{dataError}</p> : null}
          {tab === "paper" && dataJson ? <PaperDownloadLinks raw={dataJson} /> : null}
          {tab === "llm" && dataJson ? <LlmCallInspector raw={dataJson} /> : null}
          {tab !== "sse" && dataJson ? <pre className="json-block">{dataJson}</pre> : null}
          {tab !== "sse" && !dataJson && !dataError && !dataLoading ? (
            <p className="muted">Choose a tab and click Load (paper may 404 until the run finishes).</p>
          ) : null}
        </div>
      ) : null}

      <div className="card">
        <h2 style={{ marginTop: 0 }}>Live events</h2>
        <div>
          {events.length === 0 ? <div className="event">No events yet.</div> : null}
          {events.map((line, i) => (
            <div className="event" key={i}>
              {line}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
