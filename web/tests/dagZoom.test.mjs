import assert from "node:assert/strict";
import test from "node:test";

import { DAG_ZOOM_MAX, DAG_ZOOM_MIN, zoomDagAtPoint } from "../src/hooks/dagZoom.ts";

const base = {
  zoom: 1,
  focalX: 320,
  focalY: 240,
  scrollLeft: 700,
  scrollTop: 400,
};

test("pinch delta zooms in and out in the expected direction", () => {
  assert.ok(zoomDagAtPoint({ ...base, deltaY: -80 }).zoom > base.zoom);
  assert.ok(zoomDagAtPoint({ ...base, deltaY: 80 }).zoom < base.zoom);
});

test("pinch zoom clamps at usable endpoints", () => {
  assert.equal(zoomDagAtPoint({ ...base, deltaY: -100_000 }).zoom, DAG_ZOOM_MAX);
  assert.equal(zoomDagAtPoint({ ...base, deltaY: 100_000 }).zoom, DAG_ZOOM_MIN);
});

test("pinch zoom preserves the graph point beneath the focal point", () => {
  const beforeX = (base.scrollLeft + base.focalX) / base.zoom;
  const beforeY = (base.scrollTop + base.focalY) / base.zoom;
  const result = zoomDagAtPoint({ ...base, deltaY: -120 });

  assert.ok(Math.abs((result.scrollLeft + base.focalX) / result.zoom - beforeX) < 1e-9);
  assert.ok(Math.abs((result.scrollTop + base.focalY) / result.zoom - beforeY) < 1e-9);
});
