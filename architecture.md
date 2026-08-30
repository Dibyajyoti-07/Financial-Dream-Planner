# Architecture

## 1. Context Diagram (DFD Level 0)

```mermaid
flowchart LR
    User([User])
    System[["AI-Powered Financial\nDream & Goal Planner"]]
    Gemini[(Gemini Developer API)]
    Groq[(Groq API)]
    Tavily[(Tavily Web Search API)]

    User -- "NL message or structured form" --> System
    System -- "plan / chat reply (streamed or JSON)" --> User
    System -- "tool-calling requests" --> Gemini
    Gemini -- "tool calls + final text" --> System
    System -- "tool-calling requests" --> Groq
    Groq -- "tool calls + final text" --> System
    System -- "web search query" --> Tavily
    Tavily -- "search results" --> System
```

The only external network dependencies are the three optional LLM/search providers, and only `/chat` and `/chat/stream` touch them. Every other endpoint (`/plan`, `/recalculate`, `/health`, `/models/metadata`) is 100% local.

## 2. Level 1 DFD - Request Flow

```mermaid
flowchart TD
    U([User]) -->|"HTTP request"| API[FastAPI api/main.py]

    API -->|"/plan, /recalculate"| Planner[tools/planner.compute_plan]
    API -->|"/chat, /chat/stream"| Agent[LangGraph agent - agent/agent.py]
    API -->|"/health, /models/metadata"| Meta[(model_metadata.json)]

    Agent -->|"tool-calling loop"| LLM{{Gemini or Groq}}
    LLM -->|"decides which tool(s)"| Agent

    Agent --> T1[predict_salary]
    Agent --> T2[future_goal_cost]
    Agent --> T3[investment_required]
    Agent --> T4[feasibility_check]
    Agent --> T5[recommend_category]
    Agent --> T6[knowledge_base_search - RAG]
    Agent --> T7[web_search - Tavily]

    T1 --> Planner
    T2 --> Planner
    T3 --> Planner
    T4 --> Planner
    T5 --> Planner
    Planner --> T1
    Planner --> T2
    Planner --> T3
    Planner --> T4
    Planner --> T5

    T1 --> Model[(salary_model.pkl)]
    T2 --> GoalCSV[(city_goal_costs.csv)]
    T5 --> RAG[(Chroma vector store)]
    T6 --> RAG

    Agent -->|"NL reply + structured plan"| API
    Planner -->|"structured plan"| API
    API -->|"JSON or streamed tokens"| U

    subgraph UI [Optional Streamlit UI]
        ST[app.py]
    end
    ST -->|"HTTP only, no direct import"| API
    U -.->|"or use the UI"| ST
```

**Key architectural rule**: the Streamlit UI (`app.py`) never imports `agent`/`tools` directly - it is only ever a client of the FastAPI HTTP API, exactly like any other caller. This keeps the API the single gateway for every service, so a third-party client (mobile app, curl, another UI) could integrate the same way.

## 3. ML Training Pipeline (offline, one-time)

```mermaid
flowchart LR
    CSV[(data/salary_data.csv)] --> Prep[Drop unnamed column\nAssert no Experience column]
    Prep --> Split[80/20 train/test split\nrandom_state=42]
    Split --> Train["Train 13 regression algorithms\n(Linear, Ridge, Lasso, ElasticNet,\nDecision Tree, Random Forest, Extra Trees,\nAdaBoost, Gradient Boosting, HistGB, KNN, SVR, MLP)"]
    Train --> CV[5-fold cross-validation\nper algorithm]
    CV --> Select["Select lowest Test_MAE\n(tie-break: higher Test_R2)"]
    Select --> Refit[Refit winner on 100% of data]
    Refit --> Save1[(models/salary_model.pkl)]
    Select --> Save2[(models/model_metadata.json)]
```

Run via `python models/train_and_compare.py`. The pipeline (`ColumnTransformer` one-hot encoding City/Education/Job_Role + passthrough scaling of Age, wrapped in `TransformedTargetRegressor`) is baked into the saved `.pkl`, so `tools/salary_tool.py` only needs `model.predict(X)` - no preprocessing duplicated at inference time.

## 4. RAG Pipeline (build-time + query-time)

```mermaid
flowchart LR
    subgraph Build ["Build-time: rag/build_vector_store.py"]
        KB[("rag/knowledge_base/*.txt\n(3 files)")] --> Chunk["RecursiveCharacterTextSplitter\nchunk_size=400, overlap=50"]
        Chunk --> Embed[all-MiniLM-L6-v2\nsentence-transformers]
        Embed --> Store[(Chroma vector store)]
    end

    subgraph Query ["Query-time: rag/retriever.py"]
        Q[Query text] --> Sem[Semantic search\ntop-10 by cosine similarity]
        Q --> BM25[BM25 keyword search\ntop-10]
        Sem --> RRF[Reciprocal Rank Fusion]
        BM25 --> RRF
        RRF --> Rerank[Cross-encoder reranker\nms-marco-MiniLM-L-6-v2]
        Rerank --> Gate{"Any result above\nMIN_SIMILARITY / RERANK_MIN_SCORE?"}
        Gate -->|yes| TopK[Return top-k chunks]
        Gate -->|no| Empty["Return empty list\n-> agent says 'not available in knowledge base'"]
    end

    Store --> Sem
    Store --> BM25
```

## 5. Agent Tool-Calling Sequence (a single `/chat` turn, 3-goal example)

```mermaid
sequenceDiagram
    participant U as User
    participant A as Agent (LLM)
    participant T as Tools

    U->>A: "I am 24, Delhi, MBA, SWE, Home in 5y, save 20%"
    A->>T: predict_salary(age, city, education, job_role)
    T-->>A: {predicted_monthly_salary}
    A->>T: future_goal_cost(city, "Home", 5)
    T-->>A: {current_cost, projected_cost}
    A->>T: investment_required(projected_cost, 5, goal_type="Home")
    T-->>A: {required_monthly_investment}
    A->>T: recommend_category(5, goal_type="Home")
    T-->>A: {category, rationale, guidance (RAG)}
    A->>T: feasibility_check(salary, savings_pct, [goal_requirements])
    T-->>A: {available_capacity, gap, status}
    A-->>U: Streamed natural-language plan + structured JSON
```

The Agent never computes a number itself - every rupee figure in the final reply traces back to one of the five tool calls above. This is the project's "golden rule," enforced by the system prompt (`agent/system_prompt.py`) and verified in tests (`tests/test_edge_cases.py::test_case_21_prompt_injection_ignored_numbers`).

## 6. Resilience: what still works without any LLM

`/plan` and `/recalculate` call `tools/planner.compute_plan` directly - no LLM involved at all. If a user's `/chat` request hits a rate limit or the provider is unreachable, `agent.py`'s fallback path extracts the same fields via a regex-based extractor (`agent/fallback_extractor.py`) and calls the identical `compute_plan` function, so the numeric plan is still produced deterministically even when every LLM provider is down.

## 7. Folder layout

```
data/                    salary_data.csv, city_goal_costs.csv (+ processed exploratory variants)
models/                  train_and_compare.py, salary_model.pkl, model_metadata.json
tools/                   salary_tool, future_cost_tool, investment_tool, feasibility_tool,
                         recommendation_tool, planner (orchestrates all five), constants
rag/                     knowledge_base/*.txt, embeddings.py, build_vector_store.py, retriever.py
agent/                   agent.py (LangGraph + tools + streaming), system_prompt.py,
                         fallback_extractor.py, web_search_tool.py
api/                     main.py (FastAPI routes, Pydantic validation, lifespan warmup)
app.py                   Streamlit UI (Home / Our Analysis / Predictor), HTTP client of api/ only
tests/                   test_edge_cases.py (31 tests)
docs/                    PRD, TRD, Implementation Plan, Edge Cases spec, this file
analysis.ipynb           EDA notebook
model_evaluation.ipynb   model comparison notebook (13 algorithms, 4 targets)
predict.py               original CLI prototype (superseded by the FastAPI + Streamlit app)
```
