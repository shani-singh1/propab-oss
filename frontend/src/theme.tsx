import { type ReactNode } from "react";
import { useUIStore, type Theme } from "./uiStore";

export type { Theme };

// Theme now lives in the persisted UI store (`uiStore.ts`), which also applies
// the `data-theme` attribute to <html>. These exports are kept so existing
// consumers (`main.tsx`, `Navigator.tsx`) need no change: `ThemeProvider` is a
// thin passthrough, and `useTheme` reads/toggles the store.
export function ThemeProvider({ children }: { children: ReactNode }) {
  return <>{children}</>;
}

export function useTheme(): { theme: Theme; toggle: () => void } {
  const theme = useUIStore((s) => s.theme);
  const toggle = useUIStore((s) => s.toggleTheme);
  return { theme, toggle };
}
