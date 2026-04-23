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
                ["paper", "Paper URLs"],
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
