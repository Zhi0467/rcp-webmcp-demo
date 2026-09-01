import { LoaderCircle, Telescope, X } from "lucide-react";
import { useEffect, useId, useRef, useState } from "react";

const MODAL_FOCUSABLE_SELECTOR =
  'a[href], button, input, select, textarea, [tabindex]:not([tabindex="-1"])';

interface ModalKeyEvent {
  key: string;
  shiftKey: boolean;
  preventDefault(): void;
}

export function handleAutoResearchDialogKeyDown(
  event: ModalKeyEvent,
  dialog: HTMLElement,
  activeElement: Element | null,
  busy: boolean,
  onClose: () => void,
): boolean {
  if (event.key === "Escape") {
    if (busy) return false;
    event.preventDefault();
    onClose();
    return true;
  }
  if (event.key !== "Tab") return false;
  const focusable = [...dialog.querySelectorAll<HTMLElement>(MODAL_FOCUSABLE_SELECTOR)].filter(
    (element) =>
      !element.hasAttribute("disabled") &&
      element.getAttribute("aria-hidden") !== "true" &&
      element.tabIndex >= 0,
  );
  if (focusable.length === 0) {
    event.preventDefault();
    dialog.focus();
    return true;
  }
  const first = focusable[0];
  const last = focusable[focusable.length - 1];
  const activeInside = activeElement !== null && dialog.contains(activeElement);
  const destination = event.shiftKey
    ? !activeInside || activeElement === first
      ? last
      : null
    : !activeInside || activeElement === last
      ? first
      : null;
  if (!destination) return false;
  event.preventDefault();
  destination.focus();
  return true;
}

export function makeAutoResearchDialogBackgroundInert(dialog: HTMLElement): () => void {
  const previous = new Map<HTMLElement, boolean>();
  let branch = dialog;
  while (branch.parentElement) {
    const parent = branch.parentElement;
    for (const sibling of parent.children) {
      if (sibling === branch || !("inert" in sibling)) continue;
      const element = sibling as HTMLElement;
      previous.set(element, element.inert);
      element.inert = true;
    }
    branch = parent;
  }
  return () => {
    for (const [element, wasInert] of previous) element.inert = wasInert;
  };
}

export function restoreAutoResearchDialogFocus(element: HTMLElement | null): void {
  if (element?.isConnected) element.focus();
}

interface Props {
  open: boolean;
  busy: boolean;
  error: string | null;
  initialInvocationCeiling: number;
  onClose: () => void;
  onAuthorize: (invocationCeiling: number, startingInstruction: string | null) => void;
}

export function AutoResearchDialog({
  open,
  busy,
  error,
  initialInvocationCeiling,
  onClose,
  onAuthorize,
}: Props) {
  const titleId = useId();
  const dialog = useRef<HTMLFormElement>(null);
  const budgetInput = useRef<HTMLInputElement>(null);
  const [budget, setBudget] = useState(String(initialInvocationCeiling));
  const [instruction, setInstruction] = useState("");

  useEffect(() => {
    if (!open) return;
    setBudget(String(initialInvocationCeiling));
    setInstruction("");
    const returnFocus =
      document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const restoreBackground = dialog.current
      ? makeAutoResearchDialogBackgroundInert(dialog.current)
      : () => undefined;
    const frame = window.requestAnimationFrame(() => budgetInput.current?.focus());
    return () => {
      window.cancelAnimationFrame(frame);
      restoreBackground();
      restoreAutoResearchDialogFocus(returnFocus);
    };
  }, [initialInvocationCeiling, open]);

  useEffect(() => {
    if (!open) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (!dialog.current) return;
      handleAutoResearchDialogKeyDown(event, dialog.current, document.activeElement, busy, onClose);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [busy, onClose, open]);

  if (!open) return null;
  const invocationCeiling = Number(budget);
  const budgetIsValid =
    budget.trim().length > 0 && Number.isSafeInteger(invocationCeiling) && invocationCeiling >= 1;

  return (
    <div
      className="modal-backdrop"
      role="presentation"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget && !busy) onClose();
      }}
    >
      <form
        ref={dialog}
        className="campaign-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        tabIndex={-1}
        onSubmit={(event) => {
          event.preventDefault();
          if (!budgetIsValid || busy) return;
          onAuthorize(invocationCeiling, instruction.trim() || null);
        }}
      >
        <header>
          <span className="campaign-dialog-mark" aria-hidden="true">
            <Telescope size={19} />
          </span>
          <h2 id={titleId}>Authorize auto-research</h2>
          <button
            className="icon-button"
            type="button"
            onClick={onClose}
            disabled={busy}
            aria-label="Close"
          >
            <X size={17} />
          </button>
        </header>
        <div className="campaign-dialog-fields">
          <label className="campaign-budget-field">
            <span>Operational invocation ceiling</span>
            <input
              ref={budgetInput}
              type="number"
              min={1}
              step={1}
              inputMode="numeric"
              value={budget}
              disabled={busy}
              onChange={(event) => setBudget(event.target.value)}
            />
          </label>
          <label>
            <span>Starting instruction (optional)</span>
            <textarea
              rows={5}
              value={instruction}
              disabled={busy}
              onChange={(event) => setInstruction(event.target.value)}
            />
          </label>
        </div>
        {error && (
          <div className="campaign-dialog-error" role="alert">
            {error}
          </div>
        )}
        <footer>
          <button className="button secondary" type="button" onClick={onClose} disabled={busy}>
            Cancel
          </button>
          <button className="button primary" type="submit" disabled={busy || !budgetIsValid}>
            {busy ? <LoaderCircle className="spin" size={14} /> : <Telescope size={14} />}
            {busy ? "Starting…" : "Start auto-research"}
          </button>
        </footer>
      </form>
    </div>
  );
}
