""" Step 1 — load & clean the vLLM docs into a list[Document].

We read the Markdown sources cloned from the vLLM GitHub repo
(`data/vllm_src/docs/**/*.md`) rather than crawling the rendered HTML site,
because the source Markdown is far cleaner (no nav bars, no syntax-highlight
markup to strip).

Each document keeps the metadata that later RAG stages depend on:
  - source  : repo-relative path, used for citations + recall@k matching
  - title   : the first H1 heading (fallback: filename)
  - section : the top-level folder under docs/ (e.g. "getting_started")

The output is a list of LangChain `Document` objects -> fed to the chunker
(Step 2). Nothing is written to disk in this step.
"""

from __future__ import annotations

import re
from pathlib import Path

from langchain_core.documents import Document

# Default location of the cloned vLLM docs (the repo root holds data/vllm_src).
# Resolve relative to this file so it works regardless of the current cwd.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOCS_DIR = REPO_ROOT / "data" / "vllm_src" / "docs"
# `source` paths are stored relative to this so they read like GitHub paths
# (e.g. "docs/getting_started/quickstart.md") and stay stable as citations.
VLLM_SRC_ROOT = REPO_ROOT / "data" / "vllm_src"
# After cleaning, drop docs shorter than this — they're just a heading or an
# unresolved MkDocs `--8<--` include (the `docs/generated/*` targets aren't in
# the clone), so they carry no retrievable content and only add index noise.
MIN_DOC_CHARS = 100

# Matches a leading YAML frontmatter block:  ---\n ... \n---\n  at the very top.
_FRONTMATTER_RE = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)
# Matches the first Markdown H1 heading line: "# Some Title".
_H1_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
# Trailing whitespace on a line (also turns whitespace-only lines — e.g. the
# indent left behind by a removed image — into truly empty lines).
_TRAILING_WS_RE = re.compile(r"[ \t]+$", re.MULTILINE)
# Collapse 3+ consecutive blank lines down to a single blank line.
_EXTRA_BLANKS_RE = re.compile(r"\n{3,}")
# Splits text on fenced code blocks (```...```), KEEPING the fences as
# captured segments so we can leave their contents untouched.
_FENCE_SPLIT_RE = re.compile(r"(```.*?```)", re.DOTALL)
# An HTML comment: <!-- ... -->.
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
# Elements we drop WITH their inner content (not just the tags): <script>/<style>
# carry no prose, and raw <a> anchors here are GitHub badge buttons whose text
# ("Star"/"Watch"/"Fork") is pure social noise. Markdown links [t](u) are NOT
# affected — only literal HTML <a> tags (which appear in README only).
_HTML_DROP_RE = re.compile(
    r"<(script|style|a)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL
)
# An HTML open/close/self-closing tag: <figure ...>, </p>, <br/>.
# Requires a letter or '/' right after '<', so prose like "if x < y" is safe.
_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")
# A Markdown image `![alt](url)`, with an optional trailing attribute-list
# `{ align="center" ... }`. Images carry no text to embed, so we drop both.
_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)(\s*\{[^}\n]*\})?")
# A leftover CommonMark/MkDocs attribute-list containing `=`, e.g.
# `{ align="center" width="60%" }`. The `=` requirement avoids eating prose
# braces like `{x}`.
_ATTR_LIST_RE = re.compile(r"\{[^{}\n]*=[^{}\n]*\}")
# A MkDocs/pymdownx snippet directive line. Two forms appear in vLLM docs:
#   --8<-- "path/to/file"      (file include; target docs/generated/* is built
#                               at docs-build time and absent from the clone)
#   --8<-- [start:name] / [end:name]   (section boundary markers)
# Both are build directives, never prose, so we strip the whole line.
_SNIPPET_RE = re.compile(r"^[ \t]*--8<--.*$", re.MULTILINE)


def _strip_frontmatter(text: str) -> str:
    """Remove a leading YAML frontmatter block, if present.

    MkDocs pages sometimes start with `---\nhide: ...\n---`, which is page
    config, not content. We drop it so it doesn't pollute embeddings.
    """
    return _FRONTMATTER_RE.sub("", text, count=1)


def _extract_title(text: str, fallback: str) -> str:
    """Title = first H1 heading; fall back to a humanized filename."""
    match = _H1_RE.search(text)
    if match:
        return match.group(1).strip()
    # e.g. "data_parallel_deployment" -> "Data Parallel Deployment"
    return fallback.replace("_", " ").replace("-", " ").title()


def _strip_noise(text: str) -> str:
    """Strip HTML + Markdown presentation noise, keeping real prose.

    Order matters: drop script/style/anchor *elements* (with content) BEFORE the
    generic tag-unwrap, otherwise the `<a>` open tag is removed first and we lose
    the ability to delete its "Star"/"Watch" text.

    Callers pass only NON-code segments, so a literal `<...>`, `![..]`, or `{..}`
    inside a code example is never reached here.
    """
    text = _HTML_COMMENT_RE.sub("", text)   # <!-- ... -->
    text = _HTML_DROP_RE.sub("", text)      # <script>/<style>/<a> + their content
    text = _HTML_TAG_RE.sub("", text)       # unwrap <figure>/<p>/<strong>, keep text
    text = _MD_IMAGE_RE.sub("", text)       # ![alt](url){ attrs }
    text = _ATTR_LIST_RE.sub("", text)      # leftover { key="val" } attr-lists
    text = _SNIPPET_RE.sub("", text)        # --8<-- "..." MkDocs includes
    return text


def _clean(text: str) -> str:
    """Clean obvious markdown noise while preserving code blocks.

    Steps:
      1. strip leading YAML frontmatter (page config, not content),
      2. strip HTML + Markdown noise (HTML tags, script/anchor badges, images,
         attribute-lists, `--8<--` includes) — but ONLY outside fenced code
         blocks, so code is preserved verbatim,
      3. collapse runs of blank lines left behind.

    We deliberately keep MkDocs admonitions (`!!! note`) and content tabs
    (`=== "Tab"`) — their prose lives in the indented body, and stripping the
    markers risks corrupting real content. Heavier cleaning is a D5 ablation knob.
    """
    text = _strip_frontmatter(text)
    # re.split with a capturing group keeps the fenced blocks as odd-indexed
    # segments; we only clean the even-indexed (non-code) segments.
    parts = _FENCE_SPLIT_RE.split(text)
    cleaned = [seg if i % 2 else _strip_noise(seg) for i, seg in enumerate(parts)]
    text = "".join(cleaned)
    text = _TRAILING_WS_RE.sub("", text)
    text = _EXTRA_BLANKS_RE.sub("\n\n", text)
    return text.strip()


def load_docs(
    docs_dir: Path | str = DEFAULT_DOCS_DIR,
    min_chars: int = MIN_DOC_CHARS,
) -> list[Document]:
    """Load every Markdown file under `docs_dir` into a list[Document].

    Args:
        docs_dir: root folder to walk for `*.md` files.
        min_chars: drop docs whose cleaned content is shorter than this
            (heading-only / unresolved-include stubs). Pass 0 to keep all.

    Returns:
        One Document per Markdown file kept, with cleaned `page_content` and
        `{source, title, section}` metadata.
    """
    docs_dir = Path(docs_dir)
    if not docs_dir.exists():
        raise FileNotFoundError(
            f"Docs dir not found: {docs_dir}. "
            "Did the `git clone ... data/vllm_src` step run?"
        )

    documents: list[Document] = []
    # rglob is deterministic-enough but we sort for reproducible ordering.
    for md_path in sorted(docs_dir.rglob("*.md")):
        raw = md_path.read_text(encoding="utf-8", errors="replace")
        content = _clean(raw)
        if len(content) < min_chars:
            continue  # skip empty / heading-only / unresolved-include stubs

        # source: path relative to the cloned repo, e.g. docs/serving/foo.md
        source = md_path.relative_to(VLLM_SRC_ROOT).as_posix()
        # section: first path component under docs/ (the doc category).
        rel_to_docs = md_path.relative_to(docs_dir).parts
        section = rel_to_docs[0] if len(rel_to_docs) > 1 else "root"
        title = _extract_title(content, fallback=md_path.stem)

        documents.append(
            Document(
                page_content=content,
                metadata={"source": source, "title": title, "section": section},
            )
        )

    return documents


if __name__ == "__main__":
    #   uv run python -m src.ingestion.loader
    docs = load_docs()
    all_docs = load_docs(min_chars=0)  # for visibility into what got filtered
    dropped = len(all_docs) - len(docs)
    print(
        f"Loaded {len(docs)} documents from {DEFAULT_DOCS_DIR} "
        f"(dropped {dropped} stubs < {MIN_DOC_CHARS} chars)"
    )
    if dropped:
        kept_sources = {d.metadata["source"] for d in docs}
        print("  dropped:")
        for d in sorted(all_docs, key=lambda d: len(d.page_content)):
            if d.metadata["source"] not in kept_sources:
                print(f"    {len(d.page_content):4d}  {d.metadata['source']}")

    # Section distribution — sanity-check that categories look reasonable.
    from collections import Counter

    sections = Counter(d.metadata["section"] for d in docs)
    print("\nDocs per section:")
    for section, count in sections.most_common():
        print(f"  {count:4d}  {section}")

    # Document length distribution (in characters).
    lengths = sorted(len(d.page_content) for d in docs)
    n = len(lengths)
    print(
        f"\nDoc length (chars): min={lengths[0]} "
        f"median={lengths[n // 2]} max={lengths[-1]}"
    )

    # Peek at one sample so we can eyeball cleaning + metadata.
    sample = docs[0]
    print("\n--- sample document ---")
    print("metadata:", sample.metadata)
    print("content[:300]:\n", sample.page_content[:300])
