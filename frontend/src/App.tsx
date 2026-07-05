import { Outlet } from "react-router-dom";
import Navigator from "./components/Navigator";

// The persistent window shell: desk background → rounded framed window with the
// left navigator always visible and the routed view in the center/right.
export default function App() {
  return (
    <div
      data-pp
      className="flex h-screen w-screen items-stretch justify-center bg-desk p-[18px] font-sans text-ink"
    >
      <div className="flex min-w-0 flex-1 overflow-hidden rounded-xl border border-edge shadow-win">
        <Navigator />
        <Outlet />
      </div>
    </div>
  );
}
