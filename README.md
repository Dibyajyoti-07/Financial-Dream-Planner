# AI-Powered Financial Dream & Goal Planner

Local, free-only backend: predicts starting salary, projects future goal costs (Marriage/Car/Home) under inflation, computes required monthly investment, runs feasibility analysis, and recommends broad investment categories. Optional natural-language chat via Google Gemini (free tier).

## Architecture

```
User -> FastAPI (/chat, /plan, /recalculate, /health, /models/metadata)
     -> LangGraph agent (Gemini Developer API or Groq API, tool-calling, user-selectable model)
     -> Tools: predict_salary | future_goal_cost | investment_required
              | feasibility_check | recommend_category
     -> RAG tool (local Chroma vector store) for explanatory questions only
```

`/plan` and `/recalculate` never depend on Gemini - they run the same deterministic pipeline (`tools/planner.compute_plan`) as the agent's fallback path, so the app works fully offline except for `/chat`'s natural-language understanding.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env               # then set GEMINI_API_KEY (free tier: https://aistudio.google.com/apikey)
                                    # and/or GROQ_API_KEY (free tier: https://console.groq.com/keys)

python rag/build_vector_store.py   # one-time: build local vector store from rag/knowledge_base/
python models/train_and_compare.py # one-time: train, compare, save best model

uvicorn api.main:app --reload      # run locally at http://localhost:8000/docs

pytest tests/ -v                   # run edge-case test suite
```

## Models

`/chat`'s natural-language understanding and RAG answer synthesis (never financial calculations) can run on either the Gemini Developer API or the Groq API - the caller chooses among seven free-tier models via `model_id`:

- `gemini-2.5-flash` (default)
- `gemini-3.5-flash`
- `gemini-3.1-flash-lite`
- `groq-gpt-oss-120b` (`openai/gpt-oss-120b`)
- `groq-gpt-oss-20b` (`openai/gpt-oss-20b`)
- `groq-qwen-3.6-27b` (`qwen/qwen3.6-27b`)
- `groq-qwen-3.8-27b` (`qwen/qwen3.8-27b`)

The gpt-oss and Qwen models served by Groq are themselves open-weight; the Gemini models are not (though the Gemini Developer API free tier is, of course, free). If the selected provider is unreachable, rate-limited, or the corresponding key is missing, `/chat` degrades to a regex-based field extractor feeding the same deterministic pipeline `/plan` uses.

## Assumptions (stated explicitly, not guarantees)

- Inflation fixed at 6%/year for all goal-cost projections.
- Investment return assumed by goal horizon: <=3yrs 6%, 4-7yrs 9%, 8+yrs 11%.
- Monthly investment contributions are modeled as growing 10%/year (matching an assumed salary increment) - this is a documented extension beyond the base flat-SIP formula; see `tools/investment_tool.py` docstring.
- Default Area_Type is "Suburban" when not specified.
- Goal timelines capped at 60 years.
- Experience is never a model feature or user input - all users are freshers by design.

## Source of truth

Full requirements live in `docs/PRD.md`, `docs/TRD.md`, `docs/Implementation_Plan.md`, `docs/Edge_Cases_Test.md`.
