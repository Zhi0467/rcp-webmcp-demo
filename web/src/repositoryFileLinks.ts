import type { Repository } from "./types";

export interface RepositoryFileTarget {
  path: string;
  line: number | null;
}

export type RepositoryFileResolution =
  { kind: "resolved"; target: RepositoryFileTarget } | { kind: "error"; message: string };

export function isRepositoryFileHrefCandidate(href: string | undefined): boolean {
  return Boolean(href && parseAbsoluteFileHref(href));
}

export function resolveRepositoryFileHref(
  href: string,
  repositories: readonly Repository[],
): RepositoryFileResolution {
  const parsed = parseAbsoluteFileHref(href);
  if (!parsed) return { kind: "error", message: "Repository file link is invalid." };

  const matches = repositories.flatMap((repository) => {
    const root = normalizeRepositoryRoot(repository.path);
    if (!root || !pathIsWithinRoot(parsed.path, root)) return [];
    return [repository];
  });

  if (matches.length === 0) {
    return {
      kind: "error",
      message: "Repository file link does not match a configured repository.",
    };
  }

  if (matches.length > 1) {
    const aliases = [...new Set(matches.map((repository) => repository.alias))].sort();
    return {
      kind: "error",
      message: `Repository file link matches multiple repositories: ${aliases.join(", ")}.`,
    };
  }

  return {
    kind: "resolved",
    target: {
      path: parsed.path,
      line: parsed.line,
    },
  };
}

export function repositoryFilePreviewUrl(projectId: string, target: RepositoryFileTarget): string {
  const query = new URLSearchParams({ path: target.path });
  if (target.line !== null) query.set("line", String(target.line));
  return `/api/projects/${encodeURIComponent(projectId)}/repositories/files/preview?${query}`;
}

function parseAbsoluteFileHref(href: string): { path: string; line: number | null } | null {
  if (!href.startsWith("/") || href.startsWith("//") || href.includes("?") || href.includes("#")) {
    return null;
  }

  let decoded: string;
  try {
    decoded = decodeURIComponent(href);
  } catch {
    return null;
  }
  if (!isConservativeAbsolutePosixPath(decoded)) return null;

  const slash = decoded.lastIndexOf("/");
  const basename = decoded.slice(slash + 1);
  const suffix = /^(.*):([1-9]\d*):[1-9]\d*$/.exec(basename) ?? /^(.*):([1-9]\d*)$/.exec(basename);
  if (!suffix?.[1]) return { path: decoded, line: null };

  const line = Number(suffix[2]);
  if (!Number.isSafeInteger(line)) return { path: decoded, line: null };
  return { path: `${decoded.slice(0, slash + 1)}${suffix[1]}`, line };
}

function normalizeRepositoryRoot(path: string): string | null {
  const normalized = path === "/" ? path : path.replace(/\/+$/, "");
  return isConservativeAbsolutePosixPath(normalized) ? normalized : null;
}

function isConservativeAbsolutePosixPath(path: string): boolean {
  if (
    !path.startsWith("/") ||
    path.startsWith("//") ||
    path.includes("\\") ||
    path.includes("\0")
  ) {
    return false;
  }
  if (path === "/") return true;
  return !path
    .split("/")
    .some((part, index) => index > 0 && (part === "" || part === "." || part === ".."));
}

function pathIsWithinRoot(path: string, root: string): boolean {
  return root === "/" ? path.startsWith("/") : path === root || path.startsWith(`${root}/`);
}
