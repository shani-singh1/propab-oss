import { useMemo, useState } from "react";

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

export function App() {
  const [question, setQuestion] = useState(
    "Does transformer attention efficiency degrade non-linearly with sequence length?",
  );
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [events, setEvents] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

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

  const onSubmit = async () => {
    setError(null);
    setEvents([]);
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
      <p className="card">Submit a research question. Events stream live from the API (SSE).</p>

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
