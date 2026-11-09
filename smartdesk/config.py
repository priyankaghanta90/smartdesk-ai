"""Central settings. Every secret comes from the environment / .env — never from code."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    # OpenAI
    openai_api_key: str = ""
    openai_chat_model: str = "gpt-4o-mini"
    openai_fallback_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"
    llm_timeout_s: float = 30.0
    llm_max_retries: int = 3

    # Jira
    ticketing_backend: str = "jira"  # "jira" | "mock"
    jira_base_url: str = ""
    jira_email: str = ""
    jira_api_token: str = ""
    jira_project_key: str = "SD"
    jira_issue_type: str = "Task"
    jira_timeout_s: float = 10.0

    # Retrieval
    retrieval_mode: str = "hybrid_rerank"  # dense | bm25 | hybrid | hybrid_rerank
    candidate_k: int = 10
    final_k: int = 4
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Cross-encoder logits: below this the top chunk is treated as "not about this question".
    # -5.5 is the eval's best escalation F1. Gaps scoring above it (e.g. sabbatical, -3.08) are caught by the
    # LLM's INSUFFICIENT_CONTEXT check; 0.0 wrongly escalated paraphrases like "vacation" vs "PTO".
    rerank_threshold: float = -5.5
    # Used when the reranker is off (mode dense/bm25/hybrid): Chroma relevance in [0, 1].
    dense_threshold: float = 0.35
    cache_similarity: float = 0.95

    # LangSmith (evaluation). Experiments upload prompts and answers to LangSmith, so app tracing stays off;
    # the eval scripts only send the made-up golden data.
    langsmith_api_key: str = ""
    langsmith_project: str = "smartdesk-eval"
    langsmith_endpoint: str = ""  # empty = LangSmith's default (US) cloud; set for EU or self-hosted
    # LLM judge for the evals. Deliberately a different, stronger model than the one answering: gpt-4o-mini
    # gave false fails on retrieval relevance (its own reasoning said "relevant", the score said no).
    eval_judge_model: str = "gpt-4o"

    # Identity. With Google sign-in configured (.streamlit/secrets.toml) the signed-in email is used and typed
    # emails are ignored. Without it the app runs in demo mode: it asks for the email and trusts it.
    # Set True in production to forbid demo mode: typed emails are then never trusted.
    require_verified_email: bool = False
    # Who may sign in to the web UI. Fail closed: with both lists empty, nobody can.
    allowed_email_domains: list[str] = []  # e.g. ["novatech.com"]
    allowed_emails: list[str] = []  # individual accounts, e.g. a personal Gmail for local testing

    def email_allowed(self, email: str) -> bool:
        email = email.strip().lower()
        domain = email.rsplit("@", 1)[-1]
        return email in {e.lower() for e in self.allowed_emails} or domain in {
            d.lower().lstrip("@") for d in self.allowed_email_domains
        }

    # Paths
    kb_dir: Path = ROOT / "kb"
    data_dir: Path = ROOT / "data"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def chunks_path(self) -> Path:
        return self.data_dir / "chunks.json"

    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / "smartdesk.db"


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    return s
