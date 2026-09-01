# Research Control Panel

This commit is the audited, source-only baseline of Research Control Panel
immediately before the WebMCP Challenge work began. It exists so reviewers can
compare the product that already existed with the challenge implementation.

Internal operational documents, private repository history, machine-specific
fixtures, and credentials are intentionally excluded from this public snapshot.

## Build from source

```bash
npm --prefix web ci
npm --prefix web run build
uv sync --frozen
```

Run the local web application with:

```bash
uv run rcp serve --host 127.0.0.1 --port 8421
```

The WebMCP implementation and hosted challenge experience are added in the next
commit on the default branch. This baseline is tagged `pre-webmcp`.
