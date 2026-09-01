import assert from "node:assert/strict";
import { registerHooks } from "node:module";
import test from "node:test";
import { forceLink, forceManyBody, forceSimulation, forceX, forceY } from "d3-force";

import { resolveRectangleCollisions } from "../src/hooks/dagLayout.ts";

const moduleHooks = registerHooks({
  resolve(specifier, context, nextResolve) {
    if (specifier === "./dagLayout" && context.parentURL?.endsWith("/useForceDag.ts")) {
      return nextResolve("./dagLayout.ts", context);
    }
    return nextResolve(specifier, context);
  },
});
const { DAG_NODE_WIDTH, forceCanvasMetrics, forceLaneX, forceTuning } =
  await import("../src/hooks/useForceDag.ts");
moduleHooks.deregister();

test("repulsion endpoints tune the whole force model from compact to wide", () => {
  const compact = forceTuning(350);
  const wide = forceTuning(1900);

  assert.ok(Math.abs(wide.chargeStrength) > Math.abs(compact.chargeStrength) * 10);
  assert.ok(wide.chargeDistanceMax > compact.chargeDistanceMax * 2);
  assert.ok(wide.linkDistance > compact.linkDistance * 2);
  assert.ok(wide.linkStrength < compact.linkStrength / 2);
  assert.ok(wide.laneStrength < compact.laneStrength / 2);
  assert.ok(wide.centerlineStrength < compact.centerlineStrength / 2);
  assert.ok(wide.collisionPadding > compact.collisionPadding + 50);
  assert.deepEqual(forceTuning(-100), compact);
  assert.deepEqual(forceTuning(5000), wide);
});

test("force canvas is generous and automatic lanes leave manual-layout gutters", () => {
  const nodes = [
    { type: "research_question" },
    { type: "hypothesis" },
    { type: "experiment" },
    { type: "evidence" },
  ];
  const metrics = forceCanvasMetrics(nodes);
  const dragCenterMinimum = 54 + DAG_NODE_WIDTH / 2;
  const dragCenterMaximum = metrics.width - dragCenterMinimum;
  const leftLane = forceLaneX(0, metrics.width);
  const rightLane = forceLaneX(3, metrics.width);

  assert.ok(metrics.width >= 2400);
  assert.ok(metrics.height >= 1300);
  assert.ok(leftLane - dragCenterMinimum >= 350);
  assert.ok(dragCenterMaximum - rightLane >= 350);
});

test("maximum repulsion materially increases settled node separation", () => {
  const compact = settledMeanSeparation(350);
  const wide = settledMeanSeparation(1900);

  assert.ok(
    wide > compact * 1.25,
    `expected ${wide.toFixed(1)} to exceed ${compact.toFixed(1)} by 25%`,
  );
});

function settledMeanSeparation(repulsion) {
  const width = 2400;
  const height = 1300;
  const tuning = forceTuning(repulsion);
  const nodes = [0, 1, 2, 3].flatMap((lane) =>
    [0, 1].map((row) => ({
      id: `${lane}:${row}`,
      lane,
      x: forceLaneX(lane, width),
      y: 540 + row * 220,
    })),
  );
  const edges = [
    ["0:0", "1:0"],
    ["1:0", "2:0"],
    ["2:0", "3:0"],
    ["0:1", "1:1"],
    ["1:1", "2:1"],
    ["2:1", "3:1"],
    ["0:0", "1:1"],
    ["1:0", "2:1"],
    ["2:0", "3:1"],
  ].map(([source, target]) => ({ source, target }));
  const simulation = forceSimulation(nodes)
    .force(
      "links",
      forceLink(edges)
        .id((node) => node.id)
        .distance(tuning.linkDistance)
        .strength(tuning.linkStrength),
    )
    .force(
      "repulsion",
      forceManyBody()
        .strength(tuning.chargeStrength)
        .distanceMin(90)
        .distanceMax(tuning.chargeDistanceMax),
    )
    .force("lanes", forceX((node) => forceLaneX(node.lane, width)).strength(tuning.laneStrength))
    .force("centerline", forceY(height / 2).strength(tuning.centerlineStrength))
    .force("collision", rectangleCollision(tuning.collisionPadding))
    .alpha(0.9)
    .alphaDecay(0.035)
    .velocityDecay(0.34)
    .stop();
  simulation.tick(400);

  let total = 0;
  let pairs = 0;
  for (let left = 0; left < nodes.length; left += 1) {
    for (let right = left + 1; right < nodes.length; right += 1) {
      total += Math.hypot(nodes[right].x - nodes[left].x, nodes[right].y - nodes[left].y);
      pairs += 1;
    }
  }
  return total / pairs;
}

function rectangleCollision(padding) {
  let nodes = [];
  const force = () => resolveRectangleCollisions(nodes, 236 + padding, 96 + padding, 0.94, 4);
  force.initialize = (simulationNodes) => {
    nodes = simulationNodes;
  };
  return force;
}
