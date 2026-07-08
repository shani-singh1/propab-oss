// Light, dependency-free navigation bus.
//
// The command palette and keyboard shortcuts (owned by the global-craft
// workstream) need to reach into the campaign view — select a right-panel tab,
// focus a worker, scroll to a round — WITHOUT importing the center/right-panel
// internals those other workstreams own. So instead of a hard import, they emit a
// single `pp:navigate` DOM event; the panels subscribe with `onAppNavigate` and
// act on whatever fields they understand. Unknown/unhandled fields are simply
// ignored, so this can grow without coordination.

export type RightTab = "workers" | "tasks" | "tree" | "beliefs";

export interface AppNavDetail {
  /** select a right-panel tab */
  tab?: RightTab;
  /** ensure the right panel is open */
  openRight?: boolean;
  /** focus a specific worker (hypothesis id) in the Workers panel */
  worker?: string;
  /** scroll the center narrative to a round number */
  round?: number;
}

export const APP_NAV_EVENT = "pp:navigate";

export function appNavigate(detail: AppNavDetail): void {
  if (typeof window === "undefined") return;
  window.dispatchEvent(new CustomEvent<AppNavDetail>(APP_NAV_EVENT, { detail }));
}

/** Subscribe to navigation requests. Returns an unsubscribe function. */
export function onAppNavigate(cb: (detail: AppNavDetail) => void): () => void {
  const handler = (e: Event) => cb((e as CustomEvent<AppNavDetail>).detail ?? {});
  window.addEventListener(APP_NAV_EVENT, handler);
  return () => window.removeEventListener(APP_NAV_EVENT, handler);
}
