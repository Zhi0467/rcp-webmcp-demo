import { X } from "lucide-react";
import { useEffect, useState } from "react";
import { api } from "../api";
import { MarkdownAnswer } from "../chatMarkdown";
import type { SkillCatalogEntry, SkillPackageDetail } from "../types";

interface Props {
  entry: SkillCatalogEntry;
  onClose: () => void;
}

/** Read-only view of one official package: no install, edit, or import path. */
export function SkillPackageInspector({ entry, onClose }: Props) {
  const [detail, setDetail] = useState<SkillPackageDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let current = true;
    setDetail(null);
    setError(null);
    api<SkillPackageDetail>(`/api/skills/${entry.kind}/${encodeURIComponent(entry.id)}`)
      .then((next) => {
        if (current) setDetail(next);
      })
      .catch((caught) => {
        if (current) setError(caught instanceof Error ? caught.message : String(caught));
      });
    return () => {
      current = false;
    };
  }, [entry.id, entry.kind]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="skill-inspector-shell" role="dialog" aria-modal="true" aria-label={entry.label}>
      <div className="skill-inspector">
        <header>
          <div>
            <h2>{entry.label}</h2>
            <span className="skill-inspector-meta">
              {entry.kind} · {entry.id} · v{entry.version}
            </span>
          </div>
          <button type="button" className="icon-button" aria-label="Close" onClick={onClose}>
            <X size={15} />
          </button>
        </header>
        {entry.dependencies.length > 0 && (
          <p className="skill-inspector-dependencies">
            Requires {entry.dependencies.map((item) => `${item.id} v${item.version}`).join(", ")}
          </p>
        )}
        <div className="skill-inspector-body">
          {error && <p className="error-text">{error}</p>}
          {detail && <MarkdownAnswer text={detail.body} />}
        </div>
      </div>
    </div>
  );
}
