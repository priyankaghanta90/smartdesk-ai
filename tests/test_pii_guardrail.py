"""PII guardrail on semantic-cache writes (Presidio + spaCy; no LLM calls)."""

import pytest

from smartdesk.guardrails.pii import PiiGuard, guarded_store
from smartdesk.rag.semantic_cache import SemanticCache

KB = [
    "Enter your employee ID (format NT followed by five digits, e.g. NT10432). Call +1 612-555-2020.",
    "Email payroll@novatech.com with your employee ID. Enter vpn.novatech.com in GlobalProtect.",
]


@pytest.fixture(scope="module")
def guard():
    return PiiGuard(KB)


@pytest.mark.parametrize("text, entity", [
    ("Hi John Smith, your PTO balance is 12 days.", "PERSON"),
    ("My SSN is 536-22-8143, how do I update my tax form?", "US_SSN"),
    ("My employee ID is NT55821 and I can't log in", "EMPLOYEE_ID"),
    ("Reach me at 612-874-3391", "PHONE_NUMBER"),
    ("email jane.doe@gmail.com", "EMAIL_ADDRESS"),
])
def test_detects_personal_data(guard, text, entity):
    assert entity in guard.find(text)


@pytest.mark.parametrize("text", [
    "You get 15 days of PTO per year, which can be used for vacations [S1].",
    "Enter your employee ID (e.g. NT10432) or call +1 612-555-2020 [S1].",  # public KB values
    "Open GlobalProtect, enter vpn.novatech.com and email payroll@novatech.com [S1].",
])
def test_public_kb_answers_are_clean(guard, text):
    assert guard.find(text) == []


class FakeEmbedder:
    def embed_query(self, text):
        return [1.0, 0.0]


def test_guarded_store_skips_pii_and_stores_clean(guard, tmp_path):
    cache = SemanticCache(tmp_path / "cache.db", FakeEmbedder())
    assert not guarded_store(cache, guard, "How much PTO do I have? I'm John Smith", "HR", "Hi John Smith, 15 days.", [])
    assert cache.lookup("anything", "HR") is None

    assert guarded_store(cache, guard, "How much PTO do I get?", "HR", "15 days per year [S1].", [])
    assert cache.lookup("How much PTO do I get?", "HR").answer == "15 days per year [S1]."
