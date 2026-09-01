export const TEXT_SCALE_STORAGE_KEY = "rcp:text-scale";
export const TEXT_SCALE_DEFAULT = 100;
export const TEXT_SCALE_MIN = 80;
export const TEXT_SCALE_MAX = 140;
export const TEXT_SCALE_STEP = 10;

export type TextScaleAction = "decrease" | "increase" | "reset";

export function normalizeTextScale(value: unknown): number {
  if (value === null || value === undefined || value === "") return TEXT_SCALE_DEFAULT;
  const parsed = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(parsed)) return TEXT_SCALE_DEFAULT;
  const stepped = Math.round(parsed / TEXT_SCALE_STEP) * TEXT_SCALE_STEP;
  return Math.min(TEXT_SCALE_MAX, Math.max(TEXT_SCALE_MIN, stepped));
}

export function changeTextScale(current: number, action: TextScaleAction): number {
  if (action === "reset") return TEXT_SCALE_DEFAULT;
  return normalizeTextScale(current + (action === "increase" ? TEXT_SCALE_STEP : -TEXT_SCALE_STEP));
}

export function textScaleShortcut(
  event: Pick<KeyboardEvent, "key" | "metaKey" | "altKey" | "ctrlKey" | "shiftKey">,
): TextScaleAction | null {
  if (!event.metaKey || event.altKey || event.ctrlKey) return null;
  if (event.key === "0") return "reset";
  if (event.key === "+" || event.key === "=") return "increase";
  if (event.key === "-" || event.key === "_") return "decrease";
  return null;
}
