# AI-Powered Financial Dream & Goal Planner

Local, free-only backend: predicts starting salary, projects future goal costs (Marriage/Car/Home) under inflation, computes required monthly investment, runs feasibility analysis, and recommends broad investment categories. Optional natural-language chat via Google Gemini (free tier).

## Architecture

```
User -> FastAPI (/chat, /chat/stream, /plan, /recalculate, /health, /models/metadata)
     -> LangGraph agent (Gemini Developer API or Groq API, tool-calling, user-selectable model)
     -> Tools: predict_salary | future_goal_cost | investment_required
              | feasibility_check | recommend_category | web_search
     -> RAG tool (local Chroma vector store) for explanatory questions only
     -> Tavily web search for current, real-world investment research (qualitative only, never a source of financial figures)
```

`/chat/stream` returns the same result as `/chat` but as newline-delimited JSON events (`{"type": "token", "text": ...}` while the answer is generated, then one terminal `{"type": "final"|"fallback", ...}` event) - this is what the Streamlit UI uses for live token-by-token rendering; `/chat` stays as a plain one-shot JSON endpoint for any other caller.

`/plan` and `/recalculate` never depend on Gemini - they run the same deterministic pipeline (`tools/planner.compute_plan`) as the agent's fallback path, so the app works fully offline except for `/chat`'s natural-language understanding.

## Setup

**Every new terminal window must activate the venv first** - `myenv\Scripts\python.exe -m uvicorn ...` (or `python -m ...`) works from any terminal without activating; a bare `uvicorn ...`/`python ...` command only finds these packages if that terminal's venv is active (PowerShell prompt shows `(myenv)` when it is).

```powershell
python -m venv myenv
.\myenv\Scripts\Activate.ps1        # PowerShell; re-run this in every new terminal window
pip install -r requirements.txt
cp .env.example .env                # then set GEMINI_API_KEY (free tier: https://aistudio.google.com/apikey)
                                     # and/or GROQ_API_KEY (free tier: https://console.groq.com/keys)
                                     # and TAVILY_API_KEY for web search (free tier: https://tavily.com)

python rag/build_vector_store.py    # one-time: build local vector store from rag/knowledge_base/
python models/train_and_compare.py  # one-time: train, compare, save best model

uvicorn api.main:app --reload       # run locally at http://localhost:8000/docs

pytest tests/ -v                    # run edge-case test suite
```

## Frontend

`app.py` is a Streamlit UI (Home / Our Analysis / Predictor) that talks to the FastAPI backend over HTTP only - it never imports `agent`/`tools` directly, so the API is the single gateway for every service. Run both, in two **separate, each-activated** terminals:

```powershell
# Terminal 1 - backend, must be running first
.\myenv\Scripts\Activate.ps1
uvicorn api.main:app --reload

# Terminal 2 - the UI
.\myenv\Scripts\Activate.ps1
streamlit run app.py
```

If the API is unreachable, the UI shows a clear message instead of failing silently. Override the API location with `API_BASE_URL` (defaults to `http://localhost:8000`) if running the backend elsewhere.

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


## Project documentation

- `architecture.md` - DFD diagrams and full pipeline explanation (data flow, ML training pipeline, agent tool-calling flow, RAG flow).
- `tests.md` - all 31 automated test cases, what each verifies, and its result.
- `report.md` - project report: the real-world problem, solution walkthrough, assumptions, limitations, and answers to the assignment's viva questions.
