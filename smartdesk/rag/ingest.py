"""Load the KB (markdown policy docs + FAQ pairs) and turn it into retrieval chunks.

Chunking strategy
-----------------
* Markdown docs are split on their `##` section headers. Policy documents are written
  as self-contained sections ("Sick leave", "Carry-forward"), so a section is the natural
  unit of meaning — splitting mid-section separates a rule from its exceptions.
* Sections longer than ~800 tokens are split again with a recursive splitter
  (overlap 100) so no chunk overwhelms the context window.
* Every chunk is prefixed with "Doc title > Section" — this helps BM25 (titles carry
  key terms) and gives the embedding the topic even for short sections.
* FAQ pairs are kept whole: one Q+A per chunk.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# it -> IT sub-agent; everything people-related -> HR sub-agent
DOMAIN_GROUP = {"it": "IT", "hr": "HR", "onboarding": "HR", "payroll": "HR"}

MAX_CHARS = 3200  # ≈ 800 tokens
_splitter = RecursiveCharacterTextSplitter(chunk_size=MAX_CHARS, chunk_overlap=400)
_header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "title"), ("##", "section")], strip_headers=True
)


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    title: str
    section: str
    domain: str
    group: str  # "IT" | "HR"
    category: str

    def metadata(self) -> dict:
        d = asdict(self)
        d.pop("text")
        return d


def _parse_front_matter(raw: str) -> tuple[dict, str]:
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", raw, re.DOTALL)
    if not match:
        return {}, raw
    meta = {}
    for line in match.group(1).splitlines():
        key, _, value = line.partition(":")
        if value:
            meta[key.strip()] = value.strip()
    return meta, raw[match.end():]


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def chunk_markdown(path: Path, kb_dir: Path) -> list[Chunk]:
    meta, body = _parse_front_matter(path.read_text(encoding="utf-8"))
    domain = meta.get("domain", path.parent.name)
    title = meta.get("title", path.stem.replace("_", " ").title())
    source = path.relative_to(kb_dir).as_posix()

    chunks: list[Chunk] = []
    seen: dict[str, int] = {}  # two sections with the same slug ("Sick leave" / "Sick Leave") need distinct ids
    for section_doc in _header_splitter.split_text(body):
        section = section_doc.metadata.get("section", "Overview")
        content = section_doc.page_content.strip()
        if not content:
            continue
        pieces = _splitter.split_text(content) if len(content) > MAX_CHARS else [content]
        base = f"{path.stem}#{_slug(section)}"
        seen[base] = seen.get(base, 0) + 1
        if seen[base] > 1:
            base = f"{base}--{seen[base]}"  # "--" never occurs in a slug, so it can't clash with a "-2" piece suffix
        for i, piece in enumerate(pieces):
            suffix = f"-{i + 1}" if len(pieces) > 1 else ""
            chunks.append(Chunk(
                chunk_id=f"{base}{suffix}",
                text=f"{title} > {section}\n{piece}",
                source=source,
                title=title,
                section=section,
                domain=domain,
                group=DOMAIN_GROUP.get(domain, "HR"),
                category=meta.get("category", ""),
            ))
    return chunks


def chunk_faq(path: Path, kb_dir: Path) -> list[Chunk]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [
        Chunk(
            chunk_id=row["id"],
            text=f"FAQ > {row['question']}\nQ: {row['question']}\nA: {row['answer']}",
            source=path.relative_to(kb_dir).as_posix(),
            title="FAQ",
            section=row["question"],
            domain=row["domain"],
            group=DOMAIN_GROUP.get(row["domain"], "HR"),
            category=row["category"],
        )
        for row in rows
    ]


def load_kb(kb_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for md in sorted(kb_dir.glob("*/*.md")):
        if md.name.lower() == "readme.md":  # folder guides for developers, not policy documents
            continue
        chunks.extend(chunk_markdown(md, kb_dir))
    faq = kb_dir / "faq.json"
    if faq.exists():
        chunks.extend(chunk_faq(faq, kb_dir))
    return chunks


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    path.write_text(json.dumps([asdict(c) for c in chunks], indent=1), encoding="utf-8")


def read_chunks(path: Path) -> list[Chunk]:
    return [Chunk(**row) for row in json.loads(path.read_text(encoding="utf-8"))]


if __name__ == "__main__":
    from collections import Counter

    kb_dir = Path(__file__).resolve().parents[2] / "kb"
    chunks = load_kb(kb_dir)

    print(f"Total chunks: {len(chunks)}\n")
    print("By group:   ", dict(Counter(c.group for c in chunks)))
    print("By domain:  ", dict(Counter(c.domain for c in chunks)))
    print("By source:")
    for source, n in Counter(c.source for c in chunks).most_common():
        print(f"   {n:>3}  {source}")

    lengths = [len(c.text) for c in chunks]
    print(f"\nChars per chunk: min={min(lengths)} max={max(lengths)} avg={sum(lengths) // len(lengths)}")

    print("\n--- First 3 chunks ---")
    for c in chunks[:3]:
        print(f"\n[{c.chunk_id}]  group={c.group}  category={c.category}")
        print(c.text)
