import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import { installPreloadRecovery, RootErrorBoundary } from "./rootRecovery";
import "katex/dist/katex.min.css";
import "./styles.css";

installPreloadRecovery();

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <RootErrorBoundary>
      <App />
    </RootErrorBoundary>
  </StrictMode>,
);
