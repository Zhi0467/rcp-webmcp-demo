import { useEffect, useRef, useState } from "react";
import {
  clampFloatingPosition,
  clampFloatingSize,
  defaultFloatingPosition,
  detailWindowSlotPosition,
  floatingWindowSize,
  movedPosition,
  parseFloatingSize,
  resizedFloatingRect,
  type DetailWindowSlot,
  type Point,
  type ResizeCorner,
  type Size,
} from "../floatingWindow";
import { NODE_DETAIL_RESIZE_MIN_HEIGHT, NODE_DETAIL_RESIZE_MIN_WIDTH } from "../uiConstants";

let topFloatingZIndex = 110;

interface Props {
  children: React.ReactNode;
  className: string;
  kind: "detail" | "chat";
  resizable?: boolean;
  sizeStorageKey?: string;
  detailSlot?: DetailWindowSlot;
  focusRequestToken?: string | number;
}

export function shouldStartWindowDrag(target: Element): boolean {
  if (!target.closest("[data-drag-handle]")) return false;
  if (target.closest("[data-text-selectable]")) return false;
  return !target.closest("button, input, select, textarea, a");
}

const detailMinimumSize: Size = {
  width: NODE_DETAIL_RESIZE_MIN_WIDTH,
  height: NODE_DETAIL_RESIZE_MIN_HEIGHT,
};

export function DraggableWindow({
  children,
  className,
  kind,
  resizable = false,
  sizeStorageKey,
  detailSlot,
  focusRequestToken,
}: Props) {
  const root = useRef<HTMLDivElement>(null);
  const preferredSize = useRef<Size | null>(null);
  const [size, setSize] = useState<Size | null>(() => {
    if (!resizable) return null;
    const viewport = { width: window.innerWidth, height: window.innerHeight };
    let stored: Size | null = null;
    if (sizeStorageKey) {
      try {
        stored = parseFloatingSize(window.localStorage.getItem(sizeStorageKey));
      } catch {
        stored = null;
      }
    }
    preferredSize.current = stored ?? floatingWindowSize(kind, viewport);
    return clampFloatingSize(preferredSize.current, viewport, detailMinimumSize);
  });
  const [position, setPosition] = useState<Point>(() => {
    const viewport = { width: window.innerWidth, height: window.innerHeight };
    const initialSize = size ?? floatingWindowSize(kind, viewport);
    if (kind === "detail" && detailSlot) {
      return detailWindowSlotPosition(detailSlot, initialSize, viewport);
    }
    return clampFloatingPosition(defaultFloatingPosition(kind, viewport), initialSize, viewport);
  });
  const [zIndex, setZIndex] = useState(() => ++topFloatingZIndex);
  const drag = useRef<{ origin: Point; pointer: Point } | null>(null);
  const resize = useRef<{
    corner: ResizeCorner;
    position: Point;
    size: Size;
    pointer: Point;
  } | null>(null);

  const clamp = (next: Point, nextSize?: Size | null) => {
    const bounds = root.current?.getBoundingClientRect();
    return clampFloatingPosition(
      next,
      nextSize ?? { width: bounds?.width ?? 0, height: bounds?.height ?? 0 },
      { width: window.innerWidth, height: window.innerHeight },
    );
  };

  const applyUserResize = (pointer: Point) => {
    if (!resize.current) return;
    const next = resizedFloatingRect(
      { position: resize.current.position, size: resize.current.size },
      {
        x: pointer.x - resize.current.pointer.x,
        y: pointer.y - resize.current.pointer.y,
      },
      resize.current.corner,
      { width: window.innerWidth, height: window.innerHeight },
      detailMinimumSize,
    );
    preferredSize.current = next.size;
    setSize(next.size);
    setPosition(next.position);
    if (sizeStorageKey) {
      try {
        window.localStorage.setItem(sizeStorageKey, JSON.stringify(next.size));
      } catch {
        // Resizing remains usable when browser storage is unavailable.
      }
    }
  };

  useEffect(() => {
    const onResize = () => {
      const viewport = { width: window.innerWidth, height: window.innerHeight };
      const nextSize =
        resizable && preferredSize.current
          ? clampFloatingSize(preferredSize.current, viewport, detailMinimumSize)
          : null;
      if (nextSize) setSize(nextSize);
      if (kind === "detail" && detailSlot) {
        setPosition(
          detailWindowSlotPosition(
            detailSlot,
            nextSize ?? floatingWindowSize(kind, viewport),
            viewport,
          ),
        );
      } else {
        setPosition((current) => clamp(current, nextSize));
      }
    };
    window.addEventListener("resize", onResize);
    onResize();
    return () => window.removeEventListener("resize", onResize);
  }, [detailSlot, kind, resizable]);

  useEffect(() => {
    if (focusRequestToken === undefined) return;
    setZIndex(++topFloatingZIndex);
  }, [focusRequestToken]);

  return (
    <div
      ref={root}
      className={`floating-window ${className}`}
      style={{
        left: position.x,
        top: position.y,
        zIndex,
        ...(size ? { width: size.width, height: size.height } : {}),
      }}
      onPointerDownCapture={() => setZIndex(++topFloatingZIndex)}
      onFocusCapture={() => setZIndex(++topFloatingZIndex)}
      onPointerDown={(event) => {
        const target = event.target as HTMLElement;
        if (!shouldStartWindowDrag(target)) return;
        drag.current = { origin: position, pointer: { x: event.clientX, y: event.clientY } };
        event.currentTarget.setPointerCapture(event.pointerId);
      }}
      onPointerMove={(event) => {
        if (!drag.current) return;
        setPosition(
          clamp(
            movedPosition(drag.current.origin, drag.current.pointer, {
              x: event.clientX,
              y: event.clientY,
            }),
          ),
        );
      }}
      onPointerUp={(event) => {
        drag.current = null;
        if (event.currentTarget.hasPointerCapture(event.pointerId)) {
          event.currentTarget.releasePointerCapture(event.pointerId);
        }
      }}
    >
      {children}
      {resizable &&
        size &&
        (["top-left", "top-right", "bottom-left", "bottom-right"] as const).map((corner) => (
          <div
            key={corner}
            className={`floating-window-resize-corner ${corner}`}
            data-resize-corner={corner}
            onPointerDown={(event) => {
              event.stopPropagation();
              resize.current = {
                corner,
                position,
                size,
                pointer: { x: event.clientX, y: event.clientY },
              };
              event.currentTarget.setPointerCapture(event.pointerId);
            }}
            onPointerMove={(event) => {
              if (!resize.current) return;
              event.stopPropagation();
              applyUserResize({ x: event.clientX, y: event.clientY });
            }}
            onPointerUp={(event) => {
              event.stopPropagation();
              resize.current = null;
              if (event.currentTarget.hasPointerCapture(event.pointerId)) {
                event.currentTarget.releasePointerCapture(event.pointerId);
              }
            }}
            onPointerCancel={() => {
              resize.current = null;
            }}
            onLostPointerCapture={() => {
              resize.current = null;
            }}
          />
        ))}
    </div>
  );
}
