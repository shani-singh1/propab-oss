import React from "react";
import ReactDOM from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import "./index.css";
import App from "./App";
import Dashboard from "./pages/Dashboard";
import NewCampaign from "./pages/NewCampaign";
import Campaign from "./pages/Campaign";
import Paper from "./pages/Paper";

const router = createBrowserRouter([
  {
    path: "/",
    element: <App />,
    children: [
      { index: true, element: <Dashboard /> },
      { path: "new", element: <NewCampaign /> },
      { path: "campaign/:id", element: <Campaign /> },
      { path: "campaign/:id/paper", element: <Paper /> },
    ],
  },
]);

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>,
);
