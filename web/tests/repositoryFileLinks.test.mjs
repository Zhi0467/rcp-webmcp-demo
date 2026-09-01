import assert from "node:assert/strict";
import { test } from "node:test";

import {
  isRepositoryFileHrefCandidate,
  repositoryFilePreviewUrl,
  resolveRepositoryFileHref,
} from "../src/repositoryFileLinks.ts";

test("repository file links use path boundaries and preserve the absolute path", () => {
  const repositories = [{ alias: "parent", machine: "local", path: "/work/repo" }];

  assert.deepEqual(
    resolveRepositoryFileHref("/work/repo/packages/core/src/main.py:27:9", repositories),
    {
      kind: "resolved",
      target: { path: "/work/repo/packages/core/src/main.py", line: 27 },
    },
  );
  assert.deepEqual(resolveRepositoryFileHref("/work/repository/main.py", repositories), {
    kind: "error",
    message: "Repository file link does not match a configured repository.",
  });
});

test("repository file links reject every overlapping root, including nested roots", () => {
  const resolution = resolveRepositoryFileHref("/srv/shared/packages/core/src/main.py:8", [
    { alias: "local-copy", machine: "local", path: "/srv/shared" },
    { alias: "nested-copy", machine: "lab", path: "/srv/shared/packages/core" },
  ]);

  assert.deepEqual(resolution, {
    kind: "error",
    message: "Repository file link matches multiple repositories: local-copy, nested-copy.",
  });
});

test("repository file links reject non-absolute and traversal-shaped hrefs", () => {
  const repositories = [{ alias: "repo", machine: "local", path: "/work/repo" }];

  assert.equal(resolveRepositoryFileHref("src/main.py", repositories).kind, "error");
  assert.equal(resolveRepositoryFileHref("/work/repo/../secret.txt", repositories).kind, "error");
  assert.equal(
    resolveRepositoryFileHref("/work/repo/src/main.py?line=2", repositories).kind,
    "error",
  );
});

test("only conservatively parsed absolute filesystem hrefs are candidates", () => {
  assert.equal(isRepositoryFileHrefCandidate("/work/repo/src/main.py:2"), true);
  assert.equal(isRepositoryFileHrefCandidate("/#/projects/project"), false);
  assert.equal(isRepositoryFileHrefCandidate("https://example.test/source.py"), false);
  assert.equal(isRepositoryFileHrefCandidate("src/main.py"), false);
});

test("repository preview URLs preserve the absolute path and optional line", () => {
  assert.equal(
    repositoryFilePreviewUrl("project/id", {
      path: "/srv/lab repo/src/main file.py",
      line: 19,
    }),
    "/api/projects/project%2Fid/repositories/files/preview?path=%2Fsrv%2Flab+repo%2Fsrc%2Fmain+file.py&line=19",
  );
  assert.equal(
    repositoryFilePreviewUrl("project", {
      path: "/work/repo/README",
      line: null,
    }),
    "/api/projects/project/repositories/files/preview?path=%2Fwork%2Frepo%2FREADME",
  );
});
