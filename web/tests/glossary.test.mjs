import assert from "node:assert/strict";
import test from "node:test";

import { buildGlossaryIndex, segmentGlossaryText } from "../src/glossary.ts";

const glossary = {
  mopd: {
    term: "MOPD",
    plain_definition: "A matched out-of-phase distance.",
  },
  "mopd-lite": {
    term: "MOPD-lite",
    plain_definition: "The lightweight MOPD variant.",
  },
  plasticity: {
    term: "plasticity",
    plain_definition: "Capacity to continue adapting.",
  },
  "plasticity-loss": {
    term: "plasticity loss",
    plain_definition: "A reduction in that capacity.",
  },
};

test("glossary segmentation is case-insensitive, whole-term, longest-first, and lossless", () => {
  const index = buildGlossaryIndex(glossary);
  const text =
    "MOPD-lite differs from mopd during Plasticity Loss, but preMOPD and MOPD_next do not match.";
  const segments = segmentGlossaryText(text, index);
  const matches = segments.filter((segment) => segment.kind === "definition");

  assert.deepEqual(
    matches.map((segment) => [segment.text, segment.term, segment.plainDefinition]),
    [
      ["MOPD-lite", "MOPD-lite", "The lightweight MOPD variant."],
      ["mopd", "MOPD", "A matched out-of-phase distance."],
      ["Plasticity Loss", "plasticity loss", "A reduction in that capacity."],
    ],
  );
  assert.equal(segments.map((segment) => segment.text).join(""), text);
});

test("an empty glossary leaves text as one unchanged segment", () => {
  assert.deepEqual(segmentGlossaryText("Keep this exact.", buildGlossaryIndex({})), [
    { kind: "text", text: "Keep this exact." },
  ]);
  assert.deepEqual(segmentGlossaryText("", buildGlossaryIndex(glossary)), []);
});
