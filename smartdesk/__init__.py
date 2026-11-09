"""SmartDesk AI — first-line IT & HR help-desk agent for NovaTech Solutions.

One subpackage per layer. Imports only go downwards:
agent -> rag / ticketing / guardrails -> config, llm.

rag/: everything about the knowledge base. It knows nothing about agents.

  Phase A (indexing, run once via scripts/build_index.py)
    ingest.py          kb/*.md + faq.json -> Chunk objects
    retrieval.py       build_index(): embed chunks into Chroma + write chunks.json for BM25

  Phase B (answering, on every question)
    retrieval.py       HybridRetriever.retrieve(): dense + BM25 -> RRF -> cross-encoder rerank
    confidence.py      retrieval_gate(): is the best chunk good enough to answer from?
    answer.py          answer_question(): retrieve -> gate -> grounded LLM answer with citations
    prompts.py         the grounded-answer prompt
    semantic_cache.py  SemanticCache: reuse answers to near-identical questions

agent/: LangGraph orchestration on top of the RAG layer.

    flow.py            the complete flow: wires all agents into one graph + SmartDeskAgent.send()
    router.py          decides the intent (KB question / ticket / status / chitchat) and IT vs HR
    it.py              IT specialist: answers from the IT part of the KB
    hr.py              HR specialist: answers from the HR part of the KB
    ticket.py          escalate -> collect -> get_email / get_details -> confirm -> create (Jira)
    status.py          status -> choose_ticket (look up the employee's Jira tickets)
    chitchat.py        greetings and out-of-scope replies
    common.py          shared helpers, AgentDeps, error handling
    state.py           conversation state kept per session
    prompts.py         router, ticket and chitchat prompts

ticketing/: `smartdesk.ticketing` is the entry point (make_ticket_client + re-exports);
base / jira / mock / email_index hold the pieces.

guardrails/pii.py: PII screening for what enters the shared semantic cache.
"""
