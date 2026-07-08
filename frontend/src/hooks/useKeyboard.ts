import { useEffect, useRef } from "react";
import { useUIStore, type RightTab } from "../uiStore";
import { appNavigate } from "../lib/appEvents";

// Global keyboard shortcuts, installed once at the app shell:
//   ⌘K / Ctrl+K  toggle the command palette (works even while typing)
//   /            focus the Navigator search
//   ?            toggle the shortcuts overlay
//   g then t/w/b/k   jump to a right-panel tab (tree / workers / beliefs / tasks)
//   Escape       close whichever overlay is open
//
// We deliberately keep this tiny and dependency-free. Chords ("g t") use a short
// pending-key window. Single-key shortcuts are ignored while the user is typing
// in a field, but ⌘K and Escape always work.

const CHORD_MS = 1200;

function isTypingTarget(el: EventTarget | null): boolean {
  const n = el as HTMLElement | null;
  if (!n || !n.tagName) return false;
  const tag = n.tagName.toUpperCase();
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || n.isContentEditable;
}

const GO_MAP: Record<string, RightTab> = {
  t: "tree",
  w: "workers",
  b: "beliefs",
  k: "tasks",
};

export function useKeyboard() {
  const pending = useRef<{ key: string; at: number } | null>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const store = useUIStore.getState();
      const mod = e.metaKey || e.ctrlKey;

      // ⌘K / Ctrl+K — palette, everywhere.
      if (mod && (e.key === "k" || e.key === "K")) {
        e.preventDefault();
        store.togglePalette();
        return;
      }

      // Escape — dismiss the top overlay.
      if (e.key === "Escape") {
        if (store.paletteOpen) {
          store.closePalette();
          e.preventDefault();
        } else if (store.shortcutsOpen) {
          store.setShortcutsOpen(false);
          e.preventDefault();
        }
        return;
      }

      // Everything below is a bare key — skip while typing or with modifiers.
      if (mod || e.altKey || isTypingTarget(e.target)) {
        pending.current = null;
        return;
      }

      // '/' — focus search.
      if (e.key === "/") {
        const el = document.getElementById("pp-nav-search") as HTMLInputElement | null;
        if (el) {
          e.preventDefault();
          el.focus();
          el.select();
        }
        return;
      }

      // '?' — shortcuts overlay.
      if (e.key === "?") {
        e.preventDefault();
        store.toggleShortcuts();
        return;
      }

      // Chord: g <tab>.
      const now = Date.now();
      const prev = pending.current;
      if (prev && prev.key === "g" && now - prev.at < CHORD_MS) {
        pending.current = null;
        const tab = GO_MAP[e.key.toLowerCase()];
        if (tab) {
          e.preventDefault();
          store.setRightOpen(true);
          store.setRightTab(tab);
          appNavigate({ tab, openRight: true });
        }
        return;
      }
      if (e.key === "g") {
        pending.current = { key: "g", at: now };
        return;
      }
      pending.current = null;
    };

    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
}
