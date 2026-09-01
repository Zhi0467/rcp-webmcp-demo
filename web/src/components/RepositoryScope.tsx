import { Check, ChevronDown } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type { Repository } from "../types";

interface Props {
  repositories: Repository[];
  projectScope: string[];
  stateRepository: string;
  selected: string[];
  onChange: (aliases: string[]) => void;
}

export function RepositoryScope({
  repositories,
  projectScope,
  stateRepository,
  selected,
  onChange,
}: Props) {
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const close = (event: MouseEvent) => {
      if (!root.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, []);

  const toggle = (alias: string) => {
    const next = selected.includes(alias)
      ? selected.filter((item) => item !== alias)
      : [...selected, alias];
    if (next.length) onChange(next);
  };
  return (
    <div className="scope-picker" ref={root}>
      <button
        className="scope-trigger"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
      >
        <span>
          <span className="eyebrow">Run reads</span>
          <strong>
            {selected.length === projectScope.length
              ? "All repositories"
              : `${selected.length} of ${projectScope.length} repositories`}
          </strong>
        </span>
        <ChevronDown size={15} />
      </button>
      {open && (
        <div className="scope-popover">
          {repositories
            .filter((repo) => projectScope.includes(repo.alias))
            .map((repo) => (
              <button key={repo.alias} className="scope-option" onClick={() => toggle(repo.alias)}>
                <span className={`checkbox ${selected.includes(repo.alias) ? "checked" : ""}`}>
                  {selected.includes(repo.alias) && <Check size={12} />}
                </span>
                <span>
                  <strong>{repo.alias}</strong>
                  <span className="scope-option-meta mono">
                    {repo.machine} · {repo.path}
                  </span>
                </span>
                {repo.alias === stateRepository && <span className="tiny-tag">canonical</span>}
              </button>
            ))}
        </div>
      )}
    </div>
  );
}
