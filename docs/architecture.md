# SmartDesk AI — Architecture

## 1. Components and data flow

```mermaid
flowchart LR
    subgraph UI["User interfaces (app/)"]
        ST[Streamlit chat<br/>streamlit_app.py]
        CLI[CLI<br/>cli.py]
        AUTH{Google sign-in<br/>configured?}
    end

    subgraph Agent["LangGraph agent (smartdesk/agent/)"]
        R{{router<br/>intent + IT/HR + query rewrite}}
        IT[it_agent<br/>IT specialist]
        HR[hr_agent<br/>HR specialist]
        ESC[escalate<br/>offer ticket ⏸]
        TF[collect → get_email ⏸ / get_details ⏸ →<br/>confirm ⏸ → create]
        SF[status → get_email ⏸ → choose_ticket ⏸]
        CC[chitchat / out-of-scope]
        MEM[(MemorySaver<br/>per-thread state:<br/>messages, email, draft)]
    end

    subgraph RAG["Knowledge layer (smartdesk/rag/)"]
        CACHE[(Semantic cache<br/>SQLite + float32 vectors)]
        HYB[HybridRetriever<br/>dense + BM25 → RRF]
        CE[Cross-encoder<br/>re-ranker]
        GATE[Confidence gate<br/>score ≥ −5.5]
        GEN[Grounded answer<br/>gpt-4o-mini + fallback<br/>cites S1… or INSUFFICIENT_CONTEXT]
    end

    subgraph Guard["Guardrails (smartdesk/guardrails/)"]
        PII[PII check<br/>Presidio + spaCy<br/>background thread]
    end

    subgraph Data["Index (scripts/build_index.py)"]
        KB[/kb/*.md + faq.json/]
        CH[(Chroma<br/>text-embedding-3-small)]
        BM[(chunks.json<br/>for BM25)]
    end

    subgraph Ext["External services (smartdesk/ticketing/, smartdesk/llm.py)"]
        JIRA[(Jira Cloud<br/>REST v3)]
        EIDX[(EmailIndex<br/>SQLite)]
        OAI[[OpenAI API]]
    end

    ST --> AUTH
    AUTH -- "yes: verified email" --> R
    AUTH -- "no: demo mode, email asked in chat" --> R
    CLI -- "email flag or demo mode" --> R
    R --> IT & HR & TF & SF & CC
    IT & HR -- lookup --> CACHE
    CACHE -- hit --> ST
    CACHE -- miss --> HYB --> CE --> GATE
    GATE -- confident --> GEN
    GATE -- "not confident" --> ESC
    GEN -- INSUFFICIENT_CONTEXT --> ESC
    ESC -- yes --> TF
    GEN -- "answer + sources" --> ST & CLI
    GEN -. first question of a session .-> PII
    PII -. no PII found, store .-> CACHE
    TF -- "create (after yes)" --> JIRA
    SF -- "search by owner label" --> JIRA
    TF -. owner mapping .-> EIDX
    SF -. fallback lookup .-> EIDX
    KB --> CH & BM
    HYB --> CH & BM
    GEN & R & TF --> OAI
    Agent <--> MEM
```

⏸ marks a human-in-the-loop pause (`interrupt()`): the graph stops, the UI shows the question, and the employee's
reply resumes the same node. Identity comes from Google sign-in when it's configured; otherwise the app runs in a
labelled demo mode and asks for the email once per session. The semantic cache is checked before retrieval; only
answers to a session's first question are cached, after a PII check that runs off the response path.

## 2. Agent graph (generated from the compiled LangGraph)

```mermaid
graph TD;
	__start__([start]):::first
	__end__([end]):::last
	__start__ --> router;
	router -.-> it_agent & hr_agent & collect & status & chitchat & __end__;
	it_agent -.-> escalate & __end__;
	hr_agent -.-> escalate & __end__;
	escalate -.-> collect & router & __end__;
	collect -.-> get_email & get_details & confirm & collect & __end__;
	get_details -.-> collect & __end__;
	get_email -.-> collect & status & router & __end__;
	confirm -.-> create & confirm & router & __end__;
	create -.-> __end__;
	status -.-> get_email & choose_ticket & __end__;
	choose_ticket -.-> router & __end__;
	chitchat -.-> __end__;
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```

`python -c "from smartdesk.agent.flow import SmartDeskAgent; print(SmartDeskAgent().mermaid())"` prints this graph
from the code (edges collapsed here for readability). Every edge is a `Command(goto=...)` decided at runtime; the
`Literal[...]` return type on each node lists where it can go. Notable paths:

- `create` is only reachable from `confirm`, after an explicit "yes".
- `collect → collect`: a new ticket request discards an old failed draft and drafts from scratch.
- `get_details → __end__` and `confirm → __end__`: "no, cancel" ends without creating anything.
- `get_email` is only used in demo mode; with sign-in the email is already in the state.

## 3. Sequence: escalation to a Jira ticket (demo mode)

```mermaid
sequenceDiagram
    actor E as Employee
    participant G as LangGraph agent
    participant K as Hybrid retriever + gate
    participant L as OpenAI
    participant J as Jira Cloud

    E->>G: "My monitor has been flickering for two days"
    G->>L: router → intent kb_question, domain IT
    G->>K: retrieve(query, group=IT)
    K-->>G: top re-rank score −10.9 < −5.5 → escalate
    G-->>E: ⏸ "I don't have information on this… raise a ticket with IT Support?"
    E->>G: "yes"
    G-->>E: ⏸ "Could you share your NovaTech email?" (skipped when signed in)
    E->>G: "jane.doe@novatech.com"
    G->>L: draft summary / description / category / priority
    G-->>E: ⏸ ticket preview: "Shall I go ahead?"
    E->>G: "yes"
    G->>J: search label sd-req-… (an earlier attempt may already exist)
    G->>J: POST /rest/api/3/issue (labels: smartdesk, cat-it, requester-…-hash, sd-req-…)
    J-->>G: IT-5
    G-->>E: "Done! Ticket IT-5 … https://<site>.atlassian.net/browse/IT-5"
```

## 4. Evaluation (eval/, LangSmith)

```mermaid
flowchart LR
    subgraph Repo["Repo (source of truth)"]
        GOLD[/golden.json<br/>48 questions/]
        CONV[/conversations.json<br/>16 conversations + accepted paths/]
    end
    subgraph LS["LangSmith"]
        D1[(smartdesk-rag)]
        D2[(smartdesk-router)]
        D3[(smartdesk-conversations)]
        EXP[Experiments<br/>per-row scores, traces,<br/>judge reasoning]
    end
    GOLD -- sync --> D1 & D2
    CONV -- sync --> D3
    D1 --> RAGS[rag suite<br/>answer_question per retriever mode<br/>hit@k · MRR · escalation F1 ·<br/>openevals judges, gpt-4o]
    D2 --> ROUT[router suite<br/>router node only<br/>route accuracy]
    D3 --> AG[agent suite<br/>full SmartDeskAgent, mock tickets, ×3<br/>agentevals path match · outcome · safety]
    RAGS & ROUT & AG --> EXP
    RAGS --> R1[/results.md/]
    ROUT --> R2[/results_router.md/]
    AG --> R3[/results_agent.md/]
```

## 5. Key design decisions

| Decision | Why |
|---|---|
| Header-based chunking + title prefix | Policy sections are self-contained rules, so splitting by section keeps a rule together with its exceptions. The prefix adds the topic for BM25 and embeddings. |
| Hybrid + RRF + cross-encoder | BM25 catches exact terms (GlobalProtect, W-2, YubiKey). Dense search catches paraphrases. The cross-encoder score doubles as the confidence signal: gap topics score about −3 to −11, and the threshold (−5.5) is calibrated by the eval's sweep. |
| Three-layer escalation | No chunks, then the score threshold, then the LLM's `INSUFFICIENT_CONTEXT` self-check (which catches gaps that score above the threshold, e.g. sabbatical at −3.08). A cross-domain retry runs first, so a wrong IT/HR route doesn't cause a false escalation. |
| Router sends every work question to a specialist | Only clearly off-topic messages go to the scope reply. Workplace questions the KB doesn't cover still reach a specialist, which escalates them to a ticket. Found by the router eval (92% → 100%). |
| `interrupt()` for every pending question | Confirmation really blocks ticket creation: `create` is only reachable from `confirm` after an explicit yes. Unambiguous yes/no replies are decided without the LLM. |
| Identity from sign-in, demo mode otherwise | With Google sign-in, the email comes from the login and is re-asserted every turn; typed emails are ignored, so nobody can read a colleague's tickets. Without it, a labelled demo mode trusts the typed email; `REQUIRE_VERIFIED_EMAIL=true` forbids that. |
| Labels + EmailIndex for ownership | Jira has no "requester email" field for non-users. A hashed `requester-…` label makes "my tickets" a JQL query; the local index covers search-index lag. An `sd-req-…` label per draft stops retries creating duplicates. |
| PII-screened, first-turn-only semantic cache | The cache is shared across employees, so answers are cached only when they don't depend on earlier turns and contain no personal data. The check runs after the reply, adding no latency. |
| IT / HR sub-agents | Each has its own persona, a domain-filtered retriever and a cache scope. The router picks one. |
| Three LangSmith eval suites | RAG quality, routing and whole conversations are measured separately, so a router bug can't hide behind good retrieval numbers. Conversations repeat to expose intermittent failures. |
