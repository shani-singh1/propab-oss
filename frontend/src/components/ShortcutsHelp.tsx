import { useUIStore } from "../uiStore";

// The `?` overlay: a compact reference of every global shortcut. Self-gates on
// `shortcutsOpen`. Escape / backdrop close it (Escape handled in useKeyboard).

const ROWS: { keys: string[]; label: string }[] = [
  { keys: ["⌘", "K"], label: "Open command palette" },
  { keys: ["/"], label: "Focus campaign search" },
  { keys: ["?"], label: "Toggle this help" },
  { keys: ["g", "w"], label: "Go to Workers" },
  { keys: ["g", "k"], label: "Go to Tasks" },
  { keys: ["g", "t"], label: "Go to Tree" },
  { keys: ["g", "b"], label: "Go to Beliefs" },
  { keys: ["↑", "↓"], label: "Move within the palette" },
  { keys: ["↵"], label: "Run the selected command" },
  { keys: ["esc"], label: "Dismiss palette / help" },
];

export default function ShortcutsHelp() {
  const open = useUIStore((s) => s.shortcutsOpen);
  const setOpen = useUIStore((s) => s.setShortcutsOpen);
  if (!open) return null;

  return (
    <div
      className="pp-overlay fixed inset-0 z-50 flex items-center justify-center px-4"
      onMouseDown={() => setOpen(false)}
      role="dialog"
      aria-modal="true"
      aria-label="Keyboard shortcuts"
    >
      <div
        className="pp-modal w-full max-w-[420px] overflow-hidden rounded-[13px] border border-edge bg-rail shadow-win"
        onMouseDown={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-2 border-b border-line px-[18px] py-[13px]">
          <span className="text-[13.5px] font-semibold text-ink">Keyboard shortcuts</span>
          <button
            onClick={() => setOpen(false)}
            className="ml-auto text-[15px] leading-none text-ink-3 hover:text-ink"
            aria-label="Close"
          >
            ×
          </button>
        </div>
        <div className="px-[18px] py-[10px]">
          {ROWS.map((r) => (
            <div
              key={r.label}
              className="flex items-center justify-between gap-4 border-b border-line py-[9px] last:border-0"
            >
              <span className="text-[12.5px] text-ink-2">{r.label}</span>
              <span className="flex shrink-0 items-center gap-[3px]">
                {r.keys.map((k, i) => (
                  <span key={i} className="pp-kbd">
                    {k}
                  </span>
                ))}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
