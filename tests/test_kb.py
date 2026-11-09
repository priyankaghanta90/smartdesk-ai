import json

from smartdesk.config import get_settings
from smartdesk.rag.ingest import load_kb

KB = get_settings().kb_dir
GAP_TERMS = ["flicker", "parking", "printer", "cafeteria", "relocation", "shuttle", "salary advance", "sabbatical"]


def test_kb_has_enough_chunks_in_both_domains():
    chunks = load_kb(KB)
    assert len(chunks) >= 100
    groups = {c.group for c in chunks}
    assert groups == {"IT", "HR"}
    assert len({c.chunk_id for c in chunks}) == len(chunks), "chunk ids must be unique"


def test_chunks_carry_title_prefix_and_metadata():
    chunk = next(c for c in load_kb(KB) if c.chunk_id == "leave_policy#sick-and-safe-time")
    assert chunk.text.startswith("Leave Policy > Sick and safe time")
    assert chunk.group == "HR" and chunk.source == "hr/leave_policy.md"


def test_folder_readmes_are_not_indexed():
    assert not [c.source for c in load_kb(KB) if c.source.lower().endswith("readme.md")]


def test_deliberate_gaps_are_absent_from_kb():
    corpus = " ".join(c.text.lower() for c in load_kb(KB))
    for term in GAP_TERMS:
        assert term not in corpus, f"gap topic {term!r} leaked into the KB"


def test_golden_reference_ids_exist():
    ids = {c.chunk_id for c in load_kb(KB)}
    golden = json.loads((KB.parent / "eval" / "golden.json").read_text(encoding="utf-8"))
    for row in golden:
        for ref in row["reference_source_ids"]:
            assert ref in ids, f"{ref} referenced by {row['user_input']!r} does not exist"
    assert sum(r["answerable"] for r in golden) >= 30
    assert sum(not r["answerable"] for r in golden) >= 10


def test_duplicate_section_slugs_get_distinct_ids(tmp_path):
    from smartdesk.rag.ingest import chunk_markdown

    doc = tmp_path / "it" / "dup.md"
    doc.parent.mkdir()
    doc.write_text("# Doc\n## Sick leave\nOne.\n## Sick Leave\nTwo.\n", encoding="utf-8")
    ids = [c.chunk_id for c in chunk_markdown(doc, tmp_path)]
    assert ids == ["dup#sick-leave", "dup#sick-leave--2"]
