# Project Report — AI-Powered Financial Dream & Goal Planner

## 1. The Real-World Problem

Final-year students and freshers entering the workforce face a specific, recurring problem: they have a starting salary they can only guess at, several major life goals with very different time horizons (marriage, a car, a home), and no simple way to translate "I want X by year Y" into "I need to save/invest Z rupees a month, starting now." Generic financial advice tools assume an existing income and investment history; freshers have neither. They also tend to either underestimate future costs (ignoring inflation entirely) or get discouraged by advice that doesn't account for their specific city, education, and career track.

This project builds a local, free, end-to-end assistant that takes a fresher's profile and goals and answers three concrete questions deterministically:

1. What will I likely earn as a starting salary?
2. What will my goals actually cost by the time I need them, and how much must I invest monthly to get there?
3. Is my planned savings rate actually enough — and if not, by how much am I short?

It does this without requiring any paid service, running entirely on a personal laptop, and without ever letting an AI model invent a financial figure.

## 2. Solution Overview

The system has four layers:

1. **A trained ML model** predicts starting monthly salary from Age, City, Education, and Job Role (never Experience — freshers have none by definition).
2. **Five deterministic Python tools** (never the LLM) compute every financial number: future goal cost under fixed inflation, required monthly investment, feasibility/shortfall, and a broad investment-category recommendation.
3. **A local RAG knowledge base** grounds explanatory questions ("what does medium-term investing mean?") in approved reference text, refusing to answer rather than inventing facts when nothing relevant is found.
4. **An LLM-driven conversational agent** (user's choice of Gemini or Groq models) extracts structured intent from natural language and calls the right tools in the right order — but never computes a number itself.

Everything is exposed through a FastAPI backend (the single integration gateway) and an optional Streamlit UI that talks to that backend over plain HTTP, exactly like any other client would.

Full architecture diagrams: `docs/architecture.md`. Full test results: `docs/tests.md`.

## 3. Key Assumptions (stated explicitly, per project rules)

- **Inflation is fixed at 6% per year** for every goal-cost projection. (The assignment PDF's own tool-responsibility table has a typo — "0.06% annual inflation" — that contradicts the 6% stated everywhere else in the same document and in the worked feasibility example; we implemented the correct 6%, not the typo.)
- **Investment return is assumed by horizon**, not guaranteed: <=3 years = 6%, 4-7 years = 9%, 8+ years = 11%. This is our own documented rule set, as the assignment requires students to define one.
- **Monthly contributions are modeled as growing 10%/year**, matching an assumed salary increment — an explicit, documented extension beyond the assignment's baseline flat-SIP formula (verified in code and tests to collapse exactly to the flat formula when the growth rate is set to 0, so it is a strict generalization, not a different formula).
- **Default Area_Type is "Suburban"** when the user does not specify one; goal timelines are capped at 60 years (an explicit resolution of an ambiguity the assignment leaves open).
- **Experience is never a feature or input**, anywhere in the system — enforced by an `assert` in the training script, not just by omission.
- **All output is educational**, never professional financial advice, and never guarantees returns — stated in the system prompt and in every UI surface.

## 4. Algorithms and Model Selection

Trained and compared **13 regression algorithms** (Linear Regression, Ridge, Lasso, ElasticNet, Decision Tree, Random Forest, Extra Trees, AdaBoost, Gradient Boosting, Hist Gradient Boosting, KNN, SVR, Neural Network/MLP) — well beyond the assignment's minimum of 3 — on an 80/20 train/test split with 5-fold cross-validation, `random_state=42` throughout for reproducibility. Selection rule: **lowest Test MAE**, ties broken by higher Test R². The winning algorithm is refit on 100% of the data before being saved, while the reported metrics stay the honest held-out numbers. Current winner: **AdaBoost** (Test MAE ≈ 7,598), narrowly ahead of Ridge (≈ 7,771) — the exact winner is re-verified by actually running the script, not assumed from a prior run, since results can shift with data changes.

## 5. Limitations (stated honestly)

- **Training data is small** (100 rows for salary, a fixed 10-city × 10-area-type grid for goal costs) — a real deployment would need a much larger, continuously-updated dataset for the salary model to generalize beyond the cities/roles it has seen.
- **The investment-return and salary-growth assumptions are illustrative, not empirical** — they are our own documented rule set, not derived from market data, and are explicitly labeled as such everywhere they appear.
- **RAG retrieval quality is corpus-dependent**: hybrid semantic+BM25+reranking works well, but chunk boundaries in the source documents don't always align with topic boundaries, so a retrieved chunk can occasionally blend two adjacent topics (observed during testing; not a retrieval-logic bug, a corpus-chunking artifact).
- **Free-tier LLM quotas are real constraints**: Gemini's free tier is capped at a small number of requests per day per model, not just per minute — a single multi-goal conversation can consume a meaningful fraction of that daily budget. The system handles this gracefully (deterministic fallback, model-switch suggestions) but it is a genuine operational limitation of relying on free-tier APIs.
- **The web-search tool (Tavily) depends on a third-party paid-model API with a free tier** — like Gemini, it is free to use but not open-source; if the assignment's "no paid API" rule is read strictly as "no third-party API of any kind," this feature is optional and the core system functions fully without it.
- **No persistent user accounts or history** — every plan is stateless by design (matching the assignment's feasibility-recalculation requirement), so there is no saved history across sessions beyond a single browser session's `chat_history`.
- **Single-currency, single-country assumptions** — costs, salaries, and cities are India-specific (INR, Indian city names); the system would need new data and category definitions to generalize to another country's financial products and cost structures.

## 6. Viva Questions — Answered

**Why is salary prediction a Regression problem?**
`Monthly_Salary` is a continuous numeric quantity, not a category — the model must predict a real number on an open-ended scale, which is exactly the definition of a regression task (as opposed to classification, which predicts a label from a fixed set).

**Why did you compare multiple regression models?**
No single algorithm is guaranteed to fit a given dataset's structure best — a linear model, a tree-based model, and an instance-based model make very different assumptions about the relationship between features and target. Rather than assume, we trained 13 candidates (well beyond the assignment's minimum of 3) and let a documented, objective metric decide, so the choice is evidence-based, not arbitrary.

**Why was the final model selected?**
By a documented, reproducible rule: lowest Mean Absolute Error on an 80/20 held-out test split, with ties broken by higher R². This is implemented literally in `models/train_and_compare.py`'s `select_best()` function, not chosen by eye. The current winner (AdaBoost) was re-verified by actually running the script, since an earlier assumption (that Ridge would win) turned out to be wrong when actually measured — a reminder that "documented rule" must mean the rule genuinely decides, not that the expected winner is assumed.

**How did you encode City, Education and Job Role?**
`OneHotEncoder(handle_unknown="ignore")` inside a `ColumnTransformer`, alongside `StandardScaler` on Age, all wrapped in a single `Pipeline`. `handle_unknown="ignore"` is a deliberate choice: an unseen category at prediction time degrades to a zero vector instead of crashing, which is why the system can still predict a salary for a city or job role it never saw during training (tested explicitly).

**Why is Experience excluded?**
The project's fresher assumption means every user is, by definition, at the very start of their career — Experience would be zero or undefined for all of them, carrying no real signal and violating the stated rule that experience-based prediction is disallowed for this system. This is enforced in code (`train_and_compare.py` asserts `"Experience" not in df.columns`), not just by not collecting the field.

**How is future cost calculated?**
`future_cost = current_cost × (1 + 0.06) ^ years`, a pure Python function (`tools/future_cost_tool.py`), where `current_cost` is looked up from `data/city_goal_costs.csv` by City and Area_Type. No LLM is involved in this calculation at any point.

**Why is the project inflation rate 0.06% per year?**
It is not — the correct rate is **6% per year**. The assignment PDF itself contains a typo (its tool-responsibility table says "0.06% annual inflation" while every other mention in the same document, and its own worked example, use 6%). We implemented and consistently used 6%, and flagged this discrepancy rather than silently propagating the typo, since 0.06%/year would be economically meaningless for a multi-year goal projection.

**How is monthly investment calculated?**
Base formula (matching the assignment): `required_monthly = future_cost × r_monthly / ((1 + r_monthly)^n − 1)`, an ordinary annuity solved for payment, where `r_monthly` comes from a horizon-bucket assumed annual return (≤3yrs 6%, 4-7yrs 9%, 8+yrs 11%) and `n` is the number of months. We extended this (with explicit sign-off, documented in `tools/investment_tool.py`'s docstring) to a step-up variant where the monthly contribution grows 10%/year, matching an assumed salary increment — verified mathematically and in a unit test to reduce exactly to the base formula when the growth rate is 0.

**How do you calculate the shortfall in the feasibility analysis?**
`available_capacity = (savings_percentage / 100) × predicted_monthly_salary`. Each goal's allocated share of that capacity is proportional to its own required investment relative to the total required across all selected goals (pro-rata split). `gap = allocated_capacity − required_monthly_investment` — negative means shortfall, positive means surplus, and it is always shown, never hidden, matching the assignment's mandatory shortfall-visibility example, which we reproduce exactly in a test (Home: Rs 35,000 required vs Rs 22,000 capacity → Rs 13,000 shortfall).

**How does changing the goal timeline affect the plan?**
A longer timeline increases the inflation-compounded future cost (more years of 6% compounding), but also gives more months over which to build the required corpus and can shift the goal into a higher assumed-return horizon bucket — both effects usually reduce the *required monthly investment* despite the higher target, because compounding works in the saver's favor over a longer period. It can also change the recommended investment category (short → medium → long-term buckets). The `/recalculate` endpoint lets a user resubmit a changed timeline or savings percentage and get an instantly updated plan; it is fully stateless, so recalculating one scenario never affects another (tested explicitly).

**Why should financial calculations be implemented as tools instead of relying on an LLM?**
LLMs are not deterministic or fully reproducible, and can generate plausible-looking but wrong numbers ("hallucinate"). The assignment mandates deterministic, reproducible calculations — an unacceptable requirement for an LLM-generated number. We enforce this architecturally: the system prompt explicitly forbids the agent from computing figures itself, all five financial functions are pure, testable Python, and a prompt-injection test proves that even when a user explicitly asserts a fake number ("the home costs 1 rupee"), the agent still calls the real tool and returns the real, correctly-computed figure.

**What is RAG and why is it used here?**
Retrieval-Augmented Generation: instead of letting a language model answer from its own (unverifiable, sometimes wrong) internal knowledge, the system first retrieves relevant text chunks from a small, project-approved local knowledge base, then has the model answer strictly from that retrieved text. We use it for explanatory questions (e.g. "what does the medium-term investment category mean?", general savings guidance) where factual grounding matters and invented answers would be actively misleading. Our implementation goes beyond basic single-method retrieval: it combines semantic (embedding) search with BM25 keyword search via reciprocal rank fusion, then reranks the fused results with a cross-encoder model for higher precision.

**What happens when RAG cannot find relevant information?**
The retriever applies a similarity threshold on semantic search and a reranker-score threshold on the final fused candidates; if nothing clears both bars, it returns an empty list — deterministically, not based on the LLM's own judgment. The system prompt then instructs the agent to say exactly "This information is not available in the knowledge base" rather than answer from general knowledge. Verified with a genuinely out-of-scope question (Bitcoin price prediction), which correctly returns zero results and triggers the refusal.

**How does your Agent decide which tool to call?**
The agent is a LangGraph tool-calling ReAct loop: the underlying LLM (the user's choice of Gemini or Groq model) is given seven typed tool schemas with descriptive docstrings and a system prompt describing when each is appropriate (e.g. "use predict_salary whenever a salary figure is needed," "use knowledge_base_search for conceptual questions," "use web_search only for current real-world research, never for figures"). The model reasons over the conversation turn-by-turn, decides which tool(s) to call and in what order, and can chain them — for example, feeding `future_goal_cost`'s output directly into `investment_required`, then into `feasibility_check`. This was verified live across all seven selectable models, with every resulting number cross-checked by hand against the deterministic formulas.

**How did you test hallucination and prompt injection?**
Two dedicated automated tests each: RAG grounding is tested with an in-scope question (must return grounded content) and a genuinely out-of-scope question (must return nothing, triggering refusal). Prompt injection is tested with a message asserting a fake financial fact (verified the real computed number is used, not the injected one) and a direct "reveal your system prompt" attempt (verified it is declined). Beyond the automated suite, these were also verified live against real LLM calls (not just mocks) across multiple models from both providers, confirming the defenses hold in practice, not only in a controlled test double.

## 7. Deliverables Checklist (mapped to this repository)

| Assignment requirement | Location |
|---|---|
| Source code | Entire repository |
| Salary + city/goal datasets | `data/salary_data.csv`, `data/city_goal_costs.csv` |
| Trained model + selection | `models/salary_model.pkl`, `models/model_metadata.json` |
| Preprocessing / model-comparison code | `models/train_and_compare.py`, `model_evaluation.ipynb` |
| API / application code | `api/main.py`, `app.py` |
| RAG knowledge base + retrieval | `rag/knowledge_base/*.txt`, `rag/build_vector_store.py`, `rag/retriever.py` |
| Agent + tool definitions | `agent/agent.py`, `agent/system_prompt.py` |
| README with install/run steps | `README.md` |
| DFD / architecture diagram | `docs/architecture.md` |
| Minimum 15 test cases with results | `docs/tests.md` (31 provided) |
| Project report | this file |
