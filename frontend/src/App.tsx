import { Outlet } from "react-router-dom";
import Navigator from "./components/Navigator";
import CommandPalette from "./components/CommandPalette";
import ShortcutsHelp from "./components/ShortcutsHelp";
import { useKeyboard } from "./hooks/useKeyboard";

// The persistent window shell: desk background → rounded framed window with the
// left navigator always visible and the routed view in the center/right. Global
// keyboard shortcuts and the command-palette / shortcuts overlays live here so
// they're available on every route.
export default function App() {
  useKeyboard();
  return (
    <div
      data-pp
      className="flex h-screen w-screen items-stretch justify-center bg-desk p-[18px] font-sans text-ink"
    >
      <div className="flex min-w-0 flex-1 overflow-hidden rounded-xl border border-edge shadow-win">
        <Navigator />
        <Outlet />
      </div>
      <CommandPalette />
      <ShortcutsHelp />
    </div>
  );
}
