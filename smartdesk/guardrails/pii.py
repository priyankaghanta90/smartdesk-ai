"""PII guardrail for data that is stored or shown to other employees (Presidio + spaCy).

The LLM provider is trusted, so this does not sit on the path to the LLM. It screens what goes
into the shared semantic cache: a question or answer containing personal data is not cached,
so it can never be served to a colleague who asks a similar question.

A detection whose text already appears in the knowledge base (the example employee ID NT10432,
payroll@novatech.com, vpn.novatech.com, the help-desk phone number…) is public, not personal,
so KB answers still cache.
Detections are reported as entity types only; matched values are never logged.
"""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Iterable

log = logging.getLogger(__name__)

ENTITIES = [
    "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN", "CREDIT_CARD", "US_BANK_NUMBER",
    "IP_ADDRESS", "EMPLOYEE_ID",
]  # LOCATION / DATE_TIME left out: KB answers legitimately mention Minneapolis, "31 December", ...
MIN_SCORE = 0.4  # phone numbers without surrounding context score 0.40
SPACY_LABELS_UNUSED = [
    "CARDINAL", "ORDINAL", "QUANTITY", "PERCENT", "MONEY", "PRODUCT", "FAC", "WORK_OF_ART", "LAW", "LANGUAGE", "EVENT",
]


class PiiGuard:
    def __init__(self, kb_texts: Iterable[str] = ()):
        self._kb = "\n".join(kb_texts).lower()
        self._analyzer = None
        self._lock = threading.Lock()

    def _is_public(self, value: str) -> bool:
        """True if `value` appears as a whole word/phrase in the KB (e.g. vpn.novatech.com, NT10432)."""
        return bool(re.search(rf"(?<!\w){re.escape(value.lower())}(?!\w)", self._kb))

    def _build(self):
        # heavy imports, load lazily
        from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerRegistry
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        nlp = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
            "ner_model_configuration": {"labels_to_ignore": SPACY_LABELS_UNUSED},  # silences "not mapped" logs
        }).create_engine()
        registry = RecognizerRegistry(supported_languages=["en"])
        presidio_log = logging.getLogger("presidio-analyzer")
        level, presidio_log.level = presidio_log.level, logging.ERROR  # it warns about every es/it/pl recognizer it skips
        try:
            registry.load_predefined_recognizers(languages=["en"], nlp_engine=nlp)
        finally:
            presidio_log.setLevel(level)
        analyzer = AnalyzerEngine(nlp_engine=nlp, registry=registry, supported_languages=["en"])
        analyzer.registry.add_recognizer(PatternRecognizer(
            supported_entity="EMPLOYEE_ID", patterns=[Pattern("novatech_id", r"\bNT\d{5}\b", 0.85)],
        ))
        return analyzer

    def warm_up(self) -> None:
        """Load the model once; call from a background thread at start-up."""
        with self._lock:
            if self._analyzer is None:
                self._analyzer = self._build()

    def find(self, text: str) -> list[str]:
        """Entity types of personal data in `text` that isn't already public in the KB."""
        self.warm_up()
        results = self._analyzer.analyze(text, language="en", entities=ENTITIES, score_threshold=MIN_SCORE)
        return sorted({r.entity_type for r in results if not self._is_public(text[r.start:r.end])})


def guarded_store(cache, guard: PiiGuard, question: str, scope: str, answer: str, sources: list[dict]) -> bool:
    """Cache the answer unless the question or answer contains personal data. Returns True if stored.

    Usually runs on a background thread, so failures are logged rather than raised: a missed cache
    write only costs a future cache miss.
    """
    try:
        found = sorted(set(guard.find(question)) | set(guard.find(answer)))
        if found:
            log.info("semantic cache: not storing %s answer, PII detected: %s", scope, ", ".join(found))
            return False
        cache.store(question, scope, answer, sources)
        return True
    except Exception:
        log.exception("semantic cache: guarded store failed")
        return False
