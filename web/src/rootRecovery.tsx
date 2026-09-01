import { Component, type ErrorInfo, type ReactNode } from "react";

export const PRELOAD_RECOVERY_KEY = "rcp:preload-recovery-at";
export const PRELOAD_RECOVERY_WINDOW_MS = 30_000;

export function recoverFromPreloadError(
  event: Event,
  storage: Pick<Storage, "getItem" | "setItem">,
  reload: () => void,
  now = Date.now(),
): boolean {
  try {
    const stored = storage.getItem(PRELOAD_RECOVERY_KEY);
    const previous = stored === null ? Number.NaN : Number(stored);
    if (Number.isFinite(previous) && now - previous < PRELOAD_RECOVERY_WINDOW_MS) return false;
    storage.setItem(PRELOAD_RECOVERY_KEY, String(now));
  } catch {
    return false;
  }

  event.preventDefault();
  reload();
  return true;
}

export function installPreloadRecovery(): void {
  window.addEventListener("vite:preloadError", (event) => {
    recoverFromPreloadError(event, window.sessionStorage, () => window.location.reload());
  });
}

export class RootErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state: { error: Error | null } = { error: null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("RCP could not render the window", error, info.componentStack);
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <main className="fatal-state" role="alert">
        <h1>RCP needs to reload</h1>
        <p>The current window could not load part of the application.</p>
        <button className="button secondary" type="button" onClick={() => window.location.reload()}>
          Reload RCP
        </button>
      </main>
    );
  }
}
