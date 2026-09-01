import { LoaderCircle } from "lucide-react";
import { useState } from "react";
import { ApiError } from "../api";

interface Props {
  spaceName: string | null;
  onAuthenticate: (token: string) => Promise<void>;
}

export function TeamLoginBoundary({ spaceName, onAuthenticate }: Props) {
  const [token, setToken] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const resolvedSpaceName = spaceName?.trim() || "Team space";

  const authenticate = async () => {
    const submittedToken = token.trim();
    if (!submittedToken || submitting) return;
    setSubmitting(true);
    setError(null);
    try {
      await onAuthenticate(submittedToken);
      setToken("");
    } catch (caught) {
      setToken("");
      setError(teamLoginFailureMessage(caught));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <main className="team-login-boundary">
      <section
        className="team-login-card"
        aria-labelledby="team-login-title"
        data-team-login="credential-slip"
      >
        <header>
          <span className="team-login-mark" aria-hidden="true">
            RCP
          </span>
          <span className="team-login-space-kind">Team space</span>
        </header>
        <div className="team-login-card-body">
          <h1 id="team-login-title">Sign in to {resolvedSpaceName}</h1>
          <form
            autoComplete="off"
            onSubmit={(event) => {
              event.preventDefault();
              void authenticate();
            }}
          >
            <label htmlFor="team-login-token">Personal team token</label>
            <input
              id="team-login-token"
              name="team-token"
              type="password"
              autoComplete="off"
              autoCapitalize="none"
              autoCorrect="off"
              spellCheck={false}
              value={token}
              onChange={(event) => {
                setToken(event.target.value);
                setError(null);
              }}
              autoFocus
            />
            {error && (
              <p className="team-login-error" role="alert">
                {error}
              </p>
            )}
            <button className="button primary" type="submit" disabled={!token.trim() || submitting}>
              {submitting ? <LoaderCircle className="spin" size={14} /> : null}
              {submitting ? "Signing in" : "Sign in"}
            </button>
          </form>
        </div>
      </section>
    </main>
  );
}

export function teamLoginFailureMessage(error: unknown): string {
  if (error instanceof ApiError && error.status === 401) {
    return "That team token was not accepted. Paste a current token and try again.";
  }
  if (error instanceof ApiError && error.status === 429) {
    return "Too many attempts were made with that token. Wait, then try a current token.";
  }
  return "RCP could not sign in to this team space. Check the connection and try again.";
}
