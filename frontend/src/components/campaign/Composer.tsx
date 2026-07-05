import { useState } from "react";
import { api } from "../../api";
import type { StatusView } from "../../lib/status";

// Steering composer. The only real steering path the backend exposes is
// resuming a (paused/concluded) campaign with a pinned orchestrator directive —
// so while a campaign is actively running, sending is disabled with a note.
export default function Composer({
  campaignId,
  status,
}: {
  campaignId: string;
  status: StatusView;
}) {
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<string | null>(null);

  const disabled = !status.resumable || busy;

  const send = async () => {
    const directive = text.trim();
    if (!directive || disabled) return;
    setBusy(true);
    setNote(null);
    try {
      await api.resumeCampaign(campaignId, { orchestrator_directive: directive });
      setText("");
      setNote("Directive pinned — campaign resumed.");
    } catch (e: any) {
      setNote(e?.message ?? String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="max-w-[680px] rounded-[11px] border border-edge bg-rail px-[13px] py-[11px]">
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
            e.preventDefault();
            send();
          }
        }}
        placeholder={
          status.resumable
            ? "Steer the investigation — pin a directive and resume…"
            : "Campaign is live — steering applies on the next resume"
        }
        className="min-h-[22px] w-full resize-none bg-transparent text-[13.5px] leading-[1.5] text-ink outline-none"
      />
      <div className="mt-2 flex items-center gap-[9px]">
        <span className="font-mono text-[11px] leading-none text-ink-3">
          {note ?? (status.resumable ? "resumes with your directive" : "read-only while running")}
        </span>
        <button
          onClick={send}
          disabled={disabled || !text.trim()}
          title={status.resumable ? "Resume with directive" : "Only available when paused"}
          className="ml-auto flex h-7 w-7 items-center justify-center rounded-[7px] text-[14px] disabled:opacity-30"
          style={{ background: "var(--text)", color: "var(--centerBg)" }}
        >
          ↑
        </button>
      </div>
    </div>
  );
}
