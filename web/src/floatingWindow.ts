export interface Point {
  x: number;
  y: number;
}
export interface Size {
  width: number;
  height: number;
}
export type ResizeCorner = "top-left" | "top-right" | "bottom-left" | "bottom-right";

export interface FloatingRect {
  position: Point;
  size: Size;
}

export type DetailWindowSlot = "original" | "companion";

export const FLOATING_WINDOW_MARGIN = 12;
export const FLOATING_WINDOW_GAP = 12;
export const FLOATING_WINDOW_TOP = 118;
export const DETAIL_WINDOW_OVERLAP_OFFSET = 40;

export function clampFloatingPosition(
  position: Point,
  windowSize: Size,
  viewport: Size,
  margin = FLOATING_WINDOW_MARGIN,
): Point {
  const maxX = Math.max(
    margin,
    viewport.width - Math.min(windowSize.width, viewport.width) - margin,
  );
  const maxY = Math.max(
    margin,
    viewport.height - Math.min(windowSize.height, viewport.height) - margin,
  );
  return {
    x: Math.min(maxX, Math.max(margin, position.x)),
    y: Math.min(maxY, Math.max(margin, position.y)),
  };
}

export function clampFloatingSize(
  size: Size,
  viewport: Size,
  minimum: Size,
  margin = FLOATING_WINDOW_MARGIN,
): Size {
  const maximum = {
    width: Math.max(0, viewport.width - margin * 2),
    height: Math.max(0, viewport.height - margin * 2),
  };
  return {
    width: Math.min(maximum.width, Math.max(minimum.width, size.width)),
    height: Math.min(maximum.height, Math.max(minimum.height, size.height)),
  };
}

export function resizedFloatingRect(
  origin: FloatingRect,
  delta: Point,
  corner: ResizeCorner,
  viewport: Size,
  minimum: Size,
  margin = FLOATING_WINDOW_MARGIN,
): FloatingRect {
  const fromLeft = corner.endsWith("left");
  const fromTop = corner.startsWith("top");
  const fixedX = fromLeft ? origin.position.x + origin.size.width : origin.position.x;
  const fixedY = fromTop ? origin.position.y + origin.size.height : origin.position.y;
  const requested = {
    width: origin.size.width + delta.x * (fromLeft ? -1 : 1),
    height: origin.size.height + delta.y * (fromTop ? -1 : 1),
  };
  const viewportSize = clampFloatingSize(requested, viewport, minimum, margin);
  const size = {
    width: Math.min(
      viewportSize.width,
      Math.max(0, fromLeft ? fixedX - margin : viewport.width - margin - fixedX),
    ),
    height: Math.min(
      viewportSize.height,
      Math.max(0, fromTop ? fixedY - margin : viewport.height - margin - fixedY),
    ),
  };
  return {
    position: {
      x: fromLeft ? fixedX - size.width : fixedX,
      y: fromTop ? fixedY - size.height : fixedY,
    },
    size,
  };
}

export function parseFloatingSize(value: string | null): Size | null {
  if (!value) return null;
  try {
    const parsed: unknown = JSON.parse(value);
    if (
      typeof parsed !== "object" ||
      parsed === null ||
      !("width" in parsed) ||
      !("height" in parsed) ||
      typeof parsed.width !== "number" ||
      typeof parsed.height !== "number" ||
      !Number.isFinite(parsed.width) ||
      !Number.isFinite(parsed.height) ||
      parsed.width <= 0 ||
      parsed.height <= 0
    )
      return null;
    return { width: parsed.width, height: parsed.height };
  } catch {
    return null;
  }
}

export function nodeDetailSizeStorageKey(projectId: string): string {
  return `rcp:node-detail-size:${projectId}`;
}

export function movedPosition(origin: Point, pointerOrigin: Point, pointer: Point): Point {
  return {
    x: origin.x + pointer.x - pointerOrigin.x,
    y: origin.y + pointer.y - pointerOrigin.y,
  };
}

export function floatingWindowSize(kind: "detail" | "chat", viewport: Size): Size {
  const maximumWidth = kind === "detail" ? 590 : 620;
  const sharedWidth = Math.max(
    280,
    (viewport.width - FLOATING_WINDOW_MARGIN * 2 - FLOATING_WINDOW_GAP) / 2,
  );
  return {
    width: Math.min(maximumWidth, sharedWidth),
    height: Math.min(
      720,
      Math.max(240, viewport.height - FLOATING_WINDOW_TOP - FLOATING_WINDOW_MARGIN),
    ),
  };
}

export function defaultFloatingPosition(kind: "detail" | "chat", viewport: Size): Point {
  if (kind === "chat") return { x: FLOATING_WINDOW_MARGIN, y: FLOATING_WINDOW_TOP };
  const width = floatingWindowSize(kind, viewport).width;
  return { x: viewport.width - width - FLOATING_WINDOW_MARGIN, y: FLOATING_WINDOW_TOP };
}

export function detailWindowSlotPosition(
  slot: DetailWindowSlot,
  windowSize: Size,
  viewport: Size,
): Point {
  const right = viewport.width - windowSize.width - FLOATING_WINDOW_MARGIN;
  const fitsSideBySide =
    windowSize.width * 2 + FLOATING_WINDOW_GAP <= viewport.width - FLOATING_WINDOW_MARGIN * 2;
  const companionX = right - DETAIL_WINDOW_OVERLAP_OFFSET;
  const horizontalOffsetFits = companionX >= FLOATING_WINDOW_MARGIN;
  const requested = fitsSideBySide
    ? {
        x: slot === "original" ? right : right - windowSize.width - FLOATING_WINDOW_GAP,
        y: FLOATING_WINDOW_TOP,
      }
    : {
        x: slot === "original" ? right : companionX,
        y:
          !horizontalOffsetFits && slot === "original"
            ? FLOATING_WINDOW_TOP - DETAIL_WINDOW_OVERLAP_OFFSET
            : FLOATING_WINDOW_TOP,
      };
  return clampFloatingPosition(requested, windowSize, viewport);
}
