export const DAG_ZOOM_MIN = 0.5;
export const DAG_ZOOM_MAX = 2.5;

export interface DagZoomResult {
  zoom: number;
  scrollLeft: number;
  scrollTop: number;
}

/** Where the DAG was last looked at, so leaving and returning restores the view. */
export type DagViewport = DagZoomResult;

interface DagZoomInput extends DagZoomResult {
  deltaY: number;
  focalX: number;
  focalY: number;
}

export function zoomDagAtPoint({
  zoom,
  deltaY,
  focalX,
  focalY,
  scrollLeft,
  scrollTop,
}: DagZoomInput): DagZoomResult {
  const nextZoom = clamp(zoom * Math.exp(-deltaY * 0.002), DAG_ZOOM_MIN, DAG_ZOOM_MAX);
  const ratio = nextZoom / zoom;
  return {
    zoom: nextZoom,
    scrollLeft: (scrollLeft + focalX) * ratio - focalX,
    scrollTop: (scrollTop + focalY) * ratio - focalY,
  };
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}
