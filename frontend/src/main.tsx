import React from "react";
import ReactDOM from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import "./index.css";
import App from "./App";
import { ThemeProvider } from "./theme";
import Home from "./pages/Home";
import NewCampaign from "./pages/NewCampaign";
import Campaign from "./pages/Campaign";
import Paper from "./pages/Paper";

const router = createBrowserRouter([
  {
    path: "/",
    element: <App />,
    children: [
      { index: true, element: <Home /> },
      { path: "new", element: <NewCampaign /> },
      { path: "campaign/:id", element: <Campaign /> },
      { path: "campaign/:id/paper", element: <Paper /> },
    ],
  },
]);

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ThemeProvider>
      <RouterProvider router={router} />
    </ThemeProvider>
  </React.StrictMode>,
);
