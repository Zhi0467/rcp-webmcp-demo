import assert from "node:assert/strict";
import { after, test } from "node:test";
import { renderToStaticMarkup } from "react-dom/server";
import { createServer } from "vite";

import { buildGlossaryIndex } from "../src/glossary.ts";

const server = await createServer({
  root: new URL("..", import.meta.url).pathname,
  configFile: false,
  logLevel: "silent",
  server: { middlewareMode: true, hmr: false },
  optimizeDeps: { noDiscovery: true },
});
const { MarkdownAnswer } = await server.ssrLoadModule("/src/chatMarkdown.ts");

after(() => server.close());

test("chat Markdown renders formatting and unknown fenced languages as inert code", () => {
  const rendered = renderToStaticMarkup(
    MarkdownAnswer({
      text: '**Result**\n\n```made-up-language\n<widget onclick="run()">\n```',
    }),
  );

  assert.match(rendered, /<strong>Result<\/strong>/);
  assert.match(rendered, /<code class="language-made-up-language">/);
  assert.match(rendered, /&lt;widget onclick=&quot;run\(\)&quot;&gt;/);
});

test("chat Markdown does not execute raw HTML", () => {
  const rendered = renderToStaticMarkup(
    MarkdownAnswer({
      text: "<script>globalThis.compromised = true</script>",
    }),
  );

  assert.doesNotMatch(rendered, /<script>/);
  assert.match(rendered, /&lt;script&gt;/);
});

test("chat Markdown links exact graph node ids in prose and inline code", () => {
  const rendered = renderToStaticMarkup(
    MarkdownAnswer({
      text: [
        "See exp/known alongside exp/not-in-the-graph.",
        "",
        "`exp/known` keeps its code styling and becomes a node link.",
        "",
        "```text",
        "exp/known",
        "```",
        "",
        "[A web link](https://example.test/exp/known) and **exp/known**.",
      ].join("\n"),
      nodes: { "exp/known": { id: "exp/known" } },
      onOpenNode() {},
    }),
  );

  assert.match(rendered, /href="#rcp-node=exp%2Fknown"/);
  assert.match(rendered, /class="chat-node-reference"/);
  assert.match(rendered, /aria-label="Open node exp\/known"/);
  assert.match(rendered, /exp\/not-in-the-graph/);
  assert.doesNotMatch(rendered, /rcp-node=exp%2Fnot-in-the-graph/);
  assert.match(
    rendered,
    /<a href="#rcp-node=exp%2Fknown" class="chat-node-reference" aria-label="Open node exp\/known"><code>exp\/known<\/code><\/a>/,
  );
  assert.match(rendered, /<pre><code class="language-text">exp\/known\n<\/code><\/pre>/);
  assert.match(rendered, /href="https:\/\/example\.test\/exp\/known"/);
});

test("chat Markdown marks glossary terms only in prose and preserves node links", () => {
  const glossaryIndex = buildGlossaryIndex({
    mopd: {
      term: "MOPD",
      plain_definition: 'A "matched" distance.',
    },
    known: {
      term: "exp/known",
      plain_definition: "A graph node that is also a glossary term.",
    },
  });
  const rendered = renderToStaticMarkup(
    MarkdownAnswer({
      text: [
        "MOPD and **mopd** appear in prose beside exp/known.",
        "",
        "`MOPD` stays code.",
        "",
        "```text",
        "MOPD",
        "```",
        "",
        "[MOPD](https://example.test) stays a link.",
        "",
        "![MOPD](https://example.test/image.png)",
        "",
        "<span>MOPD</span>",
      ].join("\n"),
      nodes: { "exp/known": { id: "exp/known" } },
      onOpenNode() {},
      glossaryIndex,
    }),
  );

  assert.equal(rendered.match(/<dfn/g)?.length, 2);
  assert.match(rendered, /<dfn[^>]*class="glossary-definition"/);
  assert.match(rendered, /data-definition="A &quot;matched&quot; distance\."/);
  assert.match(rendered, /tabindex="0"/);
  assert.doesNotMatch(rendered, /<dfn[^>]*\stitle=/);
  assert.match(rendered, /<dfn[^>]*>MOPD<\/dfn>/);
  assert.match(rendered, /<strong><dfn[^>]*>mopd<\/dfn><\/strong>/);
  assert.match(rendered, /<code>MOPD<\/code>/);
  assert.match(rendered, /<pre><code class="language-text">MOPD\n<\/code><\/pre>/);
  assert.match(rendered, /<a href="https:\/\/example\.test">MOPD<\/a>/);
  assert.match(rendered, /alt="MOPD"/);
  assert.match(rendered, /&lt;span&gt;MOPD&lt;\/span&gt;/);
  assert.match(rendered, /href="#rcp-node=exp%2Fknown"/);
  assert.doesNotMatch(rendered, /<dfn[^>]*>exp\/known<\/dfn>/);
});

test("project chat Markdown identifies repository file links without changing other links", () => {
  const rendered = renderToStaticMarkup(
    MarkdownAnswer({
      text: [
        "[Source](/Users/example/repo/src/main.py:17:4)",
        "[Web](https://example.test/docs)",
        "[App](/#/projects/example)",
        "[Node](#rcp-node=exp%2Fknown)",
      ].join(" "),
      nodes: { "exp/known": { id: "exp/known" } },
      onOpenNode() {},
      onOpenRepositoryFileLink() {},
    }),
  );

  assert.match(rendered, /class="chat-repository-file-reference"/);
  assert.match(rendered, /aria-label="Open repository file preview"/);
  assert.match(rendered, /href="https:\/\/example\.test\/docs"/);
  assert.doesNotMatch(rendered, /https:\/\/example\.test\/docs" class=/);
  assert.match(rendered, /href="\/#\/projects\/example"/);
  assert.doesNotMatch(rendered, /\/#\/projects\/example" class=/);
  assert.match(rendered, /class="chat-node-reference"/);

  const unchanged = renderToStaticMarkup(
    MarkdownAnswer({ text: "[Source](/Users/example/repo/src/main.py:17)" }),
  );
  assert.doesNotMatch(unchanged, /chat-repository-file-reference/);
});

test("chat Markdown renders dollar and TeX bracket math with KaTeX", () => {
  const rendered = renderToStaticMarkup(
    MarkdownAnswer({
      text: [
        "Inline dollar: $z_{pg}$.",
        "Inline bracket: \\(w_{pg}\\).",
        "",
        "$$",
        "\\bar z_p = \\frac{1}{G} \\sum_{g=1}^{G} z_{pg}",
        "$$",
        "",
        "\\[",
        "s_i = \\frac{\\operatorname{SD}(\\bar z_1, \\ldots, \\bar z_P)}{\\sqrt{P}}",
        "\\]",
      ].join("\n"),
    }),
  );

  assert.equal(rendered.match(/class="katex"/g)?.length, 4);
  assert.equal(rendered.match(/class="katex-display"/g)?.length, 2);
  assert.equal(rendered.match(/<math\b/g)?.length, 4);
  assert.equal(rendered.match(/<math[^>]* display="block"/g)?.length, 2);
  assert.match(rendered, /<p>Inline dollar: <span class="katex">/);
  assert.match(rendered, /Inline bracket: <span class="katex">/);
});

test("chat math delimiters remain literal inside code spans and fences", () => {
  const rendered = renderToStaticMarkup(
    MarkdownAnswer({
      text: ["`$x$` and `\\(y\\)`", "", "```text", "$$ z $$", "\\[w\\]", "```"].join("\n"),
    }),
  );

  assert.doesNotMatch(rendered, /class="katex/);
  assert.match(rendered, /<code>\$x\$<\/code>/);
  assert.match(rendered, /<code>\\\(y\\\)<\/code>/);
  assert.match(
    rendered,
    /<pre><code class="language-text">\$\$ z \$\$\n\\\[w\\\]\n<\/code><\/pre>/,
  );
});

test("chat Markdown leaves unmatched math and a lone currency marker readable", () => {
  const unmatchedOpen = renderToStaticMarkup(
    MarkdownAnswer({
      text: "Price: $5. Unmatched: \\(x and \\[y.",
    }),
  );
  const unmatchedClose = renderToStaticMarkup(
    MarkdownAnswer({ text: "Closing only: \\) and \\]." }),
  );

  assert.doesNotMatch(unmatchedOpen, /class="katex/);
  assert.doesNotMatch(unmatchedClose, /class="katex/);
  assert.match(unmatchedOpen, /<p>Price: \$5\. Unmatched: \(x and \[y\.<\/p>/);
  assert.match(unmatchedClose, /<p>Closing only: \) and \]\.<\/p>/);
});

test("chat math is isolated from node and glossary transforms", () => {
  const glossaryIndex = buildGlossaryIndex({
    mopd: {
      term: "MOPD",
      plain_definition: "Mean orthogonal projection distance.",
    },
  });
  const rendered = renderToStaticMarkup(
    MarkdownAnswer({
      text: "Outside MOPD and exp/known. Inside $\\text{MOPD and exp/known}$.",
      nodes: { "exp/known": { id: "exp/known" } },
      onOpenNode() {},
      glossaryIndex,
    }),
  );

  assert.equal(rendered.match(/<dfn\b/g)?.length, 1);
  assert.equal(rendered.match(/href="#rcp-node=exp%2Fknown"/g)?.length, 1);
  assert.equal(rendered.match(/class="katex"/g)?.length, 1);
  assert.match(
    rendered,
    /<annotation encoding="application\/x-tex">\\text\{MOPD and exp\/known\}<\/annotation>/,
  );
});

test("malformed or untrusted TeX remains visible without breaking the reply", () => {
  const malformed = renderToStaticMarkup(MarkdownAnswer({ text: "Before $\\frac{1}{2$ after." }));
  const untrusted = renderToStaticMarkup(
    MarkdownAnswer({ text: "Unsafe $\\href{javascript:alert(1)}{x}$ stays inert." }),
  );

  assert.match(malformed, /<p>Before <span class="katex-error"/);
  assert.match(malformed, />\\frac\{1\}\{2<\/span> after\.<\/p>/);
  assert.doesNotMatch(untrusted, /href=/);
  assert.match(untrusted, /class="katex"/);
});
