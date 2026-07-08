import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";

// Persisted UI preferences — theme, density, and the right-panel open/tab intent.
//
// This is deliberately SEPARATE from the live event store (`store.ts`): that one
// holds volatile campaign data streamed over SSE and must never be coupled to a
// user's cosmetic choices. Here we keep only durable UI state (localStorage) plus
// a couple of ephemeral overlay flags (palette / shortcuts) that no one needs to
// persist. The right-panel workstream can read `rightOpen`/`rightTab` to adopt the
// persisted intent without importing anything from here beyond the hook.

export type Theme = "dark" | "light";
export type Density = "comfortable" | "compact";
export type RightTab = "workers" | "tasks" | "tree" | "beliefs";

interface UIState {
  theme: Theme;
  density: Density;
  /** persisted intent for the right panel; consumed by the right-panel workstream. */
  rightOpen: boolean;
  rightTab: RightTab;

  // Ephemeral overlay state (never persisted).
  paletteOpen: boolean;
  shortcutsOpen: boolean;

  setTheme: (t: Theme) => void;
  toggleTheme: () => void;
  setDensity: (d: Density) => void;
  toggleDensity: () => void;
  setRightOpen: (v: boolean) => void;
  toggleRight: () => void;
  setRightTab: (t: RightTab) => void;
  openPalette: () => void;
  closePalette: () => void;
  togglePalette: () => void;
  setShortcutsOpen: (v: boolean) => void;
  toggleShortcuts: () => void;
}

// Seed the theme default from the pre-existing signal so returning users don't
// flicker: honor the legacy `pp-theme` key (written by the old ThemeProvider),
// then any attribute already on <html>, then dark.
function seedTheme(): Theme {
  try {
    const legacy = localStorage.getItem("pp-theme");
    if (legacy === "dark" || legacy === "light") return legacy;
  } catch {
    /* no storage */
  }
  const attr = typeof document !== "undefined" && document.documentElement.getAttribute("data-theme");
  if (attr === "dark" || attr === "light") return attr;
  return "dark";
}

export const useUIStore = create<UIState>()(
  persist(
    (set) => ({
      theme: seedTheme(),
      density: "comfortable",
      rightOpen: true,
      rightTab: "workers",
      paletteOpen: false,
      shortcutsOpen: false,

      setTheme: (theme) => set({ theme }),
      toggleTheme: () => set((s) => ({ theme: s.theme === "dark" ? "light" : "dark" })),
      setDensity: (density) => set({ density }),
      toggleDensity: () =>
        set((s) => ({ density: s.density === "comfortable" ? "compact" : "comfortable" })),
      setRightOpen: (rightOpen) => set({ rightOpen }),
      toggleRight: () => set((s) => ({ rightOpen: !s.rightOpen })),
      setRightTab: (rightTab) => set({ rightTab }),
      openPalette: () => set({ paletteOpen: true }),
      closePalette: () => set({ paletteOpen: false }),
      togglePalette: () => set((s) => ({ paletteOpen: !s.paletteOpen })),
      setShortcutsOpen: (shortcutsOpen) => set({ shortcutsOpen }),
      toggleShortcuts: () => set((s) => ({ shortcutsOpen: !s.shortcutsOpen })),
    }),
    {
      name: "pp-ui",
      storage: createJSONStorage(() => localStorage),
      // Persist durable prefs only — never the overlay flags.
      partialize: (s) => ({
        theme: s.theme,
        density: s.density,
        rightOpen: s.rightOpen,
        rightTab: s.rightTab,
      }),
    },
  ),
);

// Reflect theme/density onto the document so the token system (`[data-theme]`)
// and any density-aware CSS pick them up. Applied once on load and on every
// change. Kept here (not in a component) so the attributes are correct before
// first paint and independent of React's mount order.
function applyDom(theme: Theme, density: Density) {
  if (typeof document === "undefined") return;
  const el = document.documentElement;
  el.setAttribute("data-theme", theme);
  el.setAttribute("data-density", density);
  el.style.colorScheme = theme;
  // Keep the legacy key in sync so a rollback still finds the user's choice.
  try {
    localStorage.setItem("pp-theme", theme);
  } catch {
    /* no storage */
  }
}

{
  const s = useUIStore.getState();
  applyDom(s.theme, s.density);
}
useUIStore.subscribe((s) => applyDom(s.theme, s.density));
