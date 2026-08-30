import json
import os
from typing import Literal

import groq
from langchain_core.exceptions import ModelRateLimitError
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

RATE_LIMIT_ERRORS = (ModelRateLimitError, groq.RateLimitError)

from agent.fallback_extractor import ExtractionError, extract_plan_request
from agent.system_prompt import SYSTEM_PROMPT
from rag import retriever
from tools import feasibility_tool, future_cost_tool, investment_tool, recommendation_tool, salary_tool
from tools.planner import compute_plan

MODEL_REGISTRY = {
    "gemini-2.5-flash": {"provider": "gemini", "model": "gemini-2.5-flash"},
    "gemini-3.5-flash": {"provider": "gemini", "model": "gemini-3.5-flash"},
    "gemini-3.1-flash-lite": {"provider": "gemini", "model": "gemini-3.1-flash-lite"},
    "groq-gpt-oss-120b": {"provider": "groq", "model": "openai/gpt-oss-120b"},
    "groq-gpt-oss-20b": {"provider": "groq", "model": "openai/gpt-oss-20b"},
    "groq-qwen-3.6-27b": {"provider": "groq", "model": "qwen/qwen3.6-27b"},
    "groq-qwen-3.8-27b": {"provider": "groq", "model": "qwen/qwen3.8-27b"},
}
ALLOWED_MODELS = list(MODEL_REGISTRY.keys())
DEFAULT_MODEL = "gemini-2.5-flash"


@tool
def predict_salary(age: int, city: str, education: str, job_role: str) -> dict:
    """Predict a fresher's monthly salary in INR given age, city, education level, and job role."""
    return salary_tool.predict_salary(age, city, education, job_role)


@tool
def future_goal_cost(city: str, goal_type: Literal["Marriage", "Car", "Home"], years: int, area_type: str = "Suburban") -> dict:
    """Project the future cost (INR) of a Marriage, Car, or Home goal in a given city, N years from now, using the fixed 6% annual inflation assumption. area_type defaults to "Suburban" - only pass a different value (Central, North, South, East, West, Outer, IT Corridor, Premium, Developing) if the user explicitly mentions a specific area."""
    return future_cost_tool.future_goal_cost(city, area_type, goal_type, years)


@tool
def investment_required(future_cost_value: float, years: int, goal_type: Literal["Marriage", "Car", "Home"]) -> dict:
    """Compute the required monthly investment (INR) to reach a future cost target in N years for the given goal_type, using an assumed horizon-based investment return and a 10%/year growing contribution. Always pass the same goal_type used for future_goal_cost so results can be matched to the right goal."""
    result = investment_tool.investment_required(future_cost_value, years)
    result["goal_type"] = goal_type
    return result


@tool
def feasibility_check(predicted_monthly_salary: float, savings_percentage: float, goal_requirements: list[dict]) -> dict:
    """Check feasibility of one or more goals given monthly salary and savings percentage. goal_requirements is a list of {goal_type, required_monthly_investment}."""
    return feasibility_tool.feasibility_check(predicted_monthly_salary, savings_percentage, goal_requirements)


@tool
def recommend_category(years: int, goal_type: Literal["Marriage", "Car", "Home"]) -> dict:
    """Recommend a broad investment category (capital-preservation / balanced / growth-oriented) for a goal N years away, enriched with grounded guidance from the knowledge base. No fund names, no guarantees. Always pass the goal_type this recommendation is for."""
    result = recommendation_tool.recommend_category(years, goal_type)
    result["goal_type"] = goal_type
    return result


@tool
def knowledge_base_search(query: str) -> list[str]:
    """Search the local financial-planning knowledge base for text relevant to a conceptual question. Returns an empty list if nothing relevant is found - in that case, tell the user the information is not available in the knowledge base."""
    return retriever.retrieve(query, k=3)


TOOLS = [predict_salary, future_goal_cost, investment_required, feasibility_check, recommend_category, knowledge_base_search]


class InvalidModelError(ValueError):
    pass


def _build_agent(model_id):
    entry = MODEL_REGISTRY.get(model_id)
    if entry is None:
        raise InvalidModelError(f"Unknown model: {model_id}. Choose one of {ALLOWED_MODELS}")

    if entry["provider"] == "gemini":
        llm = ChatGoogleGenerativeAI(model=entry["model"], google_api_key=os.getenv("GEMINI_API_KEY"), temperature=0)
    else:
        llm = ChatGroq(model=entry["model"], groq_api_key=os.getenv("GROQ_API_KEY"), temperature=0)

    return create_react_agent(llm, TOOLS, prompt=SYSTEM_PROMPT)


def _other_models(model_id):
    return [m for m in ALLOWED_MODELS if m != model_id]


def _unavailable_note(model_id, error_type):
    if error_type == "rate_limit" and model_id:
        others = ", ".join(_other_models(model_id))
        return f"(Model '{model_id}' has hit its API rate/usage limit - this is a deterministic summary instead. Try switching to a different model: {others}.)"
    return "(The selected AI model is currently unavailable - this is a deterministic summary.)"


def _fallback_reply(message, model_id=None, error_type=None):
    note = _unavailable_note(model_id, error_type)
    suggested_models = _other_models(model_id) if error_type == "rate_limit" and model_id else None

    try:
        request = extract_plan_request(message)
    except ExtractionError as exc:
        return {
            "reply": f"{note} I also couldn't extract enough information from your message to compute a plan ({exc}). Please use /plan with structured fields instead.",
            "plan": None,
            "degraded": True,
            "error_type": error_type,
            "suggested_models": suggested_models,
        }

    plan = compute_plan(**request)
    return {
        "reply": (
            f"{note} "
            f"Predicted monthly salary: {plan['predicted_monthly_salary']:.0f}. "
            f"Overall feasibility: {plan['overall_status']}. "
            f"Total gap: {plan['total_gap']:.0f}."
        ),
        "plan": plan,
        "degraded": True,
        "error_type": error_type,
        "suggested_models": suggested_models,
    }


def _extract_plan_from_messages(messages):
    human_indices = [i for i, m in enumerate(messages) if type(m).__name__ == "HumanMessage"]
    recent = messages[human_indices[-1] + 1:] if human_indices else messages

    call_args_by_id = {}
    for msg in recent:
        for call in getattr(msg, "tool_calls", None) or []:
            call_args_by_id[call["id"]] = call["args"]

    salary_result = None
    goals_by_type = {}
    feasibility_result = None
    feasibility_args = None

    for msg in recent:
        if type(msg).__name__ != "ToolMessage":
            continue
        try:
            content = json.loads(msg.content)
        except (json.JSONDecodeError, TypeError):
            continue

        call_args = call_args_by_id.get(msg.tool_call_id, {})
        name = msg.name

        if name == "predict_salary":
            salary_result = content
        elif name == "future_goal_cost":
            goals_by_type.setdefault(content["goal_type"], {}).update({
                "goal_type": content["goal_type"],
                "years": content["years"],
                "current_cost": content["current_cost"],
                "projected_cost": content["projected_cost"],
                "area_type_used": call_args.get("area_type", "Suburban"),
            })
        elif name == "investment_required":
            goal_type = content.get("goal_type")
            if goal_type:
                goals_by_type.setdefault(goal_type, {}).update({
                    "required_monthly_investment": content["required_monthly_investment"],
                    "assumed_annual_return": content["assumed_annual_return"],
                })
        elif name == "recommend_category":
            goal_type = content.get("goal_type")
            if goal_type:
                goals_by_type.setdefault(goal_type, {}).update({
                    "recommended_category": content["category"],
                    "category_rationale": content["rationale"],
                    "guidance": content.get("guidance", []),
                })
        elif name == "feasibility_check":
            feasibility_result = content
            feasibility_args = call_args

    if salary_result is None or feasibility_result is None or not goals_by_type:
        return None

    per_goal_feasibility = {g["goal_type"]: g for g in feasibility_result["per_goal"]}
    goals = []
    for goal_type, goal in goals_by_type.items():
        feas = per_goal_feasibility.get(goal_type, {})
        goals.append({
            **goal,
            "allocated_capacity": feas.get("allocated_capacity"),
            "gap": feas.get("gap"),
            "status": feas.get("status"),
        })

    area_type_used = next((g.get("area_type_used") for g in goals if g.get("area_type_used")), "Suburban")

    return {
        "predicted_monthly_salary": salary_result["predicted_monthly_salary"],
        "salary_low_confidence": salary_result["low_confidence"],
        "savings_percentage": feasibility_args.get("savings_percentage") if feasibility_args else None,
        "available_capacity": feasibility_result["available_capacity"],
        "total_required_monthly": feasibility_result["total_required"],
        "total_gap": feasibility_result["total_gap"],
        "overall_status": feasibility_result["overall_status"],
        "goals": goals,
        "assumptions": {
            "inflation_rate": future_cost_tool.INFLATION_RATE,
            "salary_growth_rate": investment_tool.SALARY_GROWTH_RATE,
            "area_type_used": area_type_used,
        },
    }


def run(message, history=None, model_id=DEFAULT_MODEL):
    try:
        agent = _build_agent(model_id)
        messages = [*(history or []), {"role": "user", "content": message}]
        result = agent.invoke({"messages": messages})
        reply = result["messages"][-1].text
        plan = _extract_plan_from_messages(result["messages"])
        return {"reply": reply, "plan": plan, "degraded": False, "error_type": None, "suggested_models": None}
    except InvalidModelError:
        raise
    except RATE_LIMIT_ERRORS:
        return _fallback_reply(message, model_id=model_id, error_type="rate_limit")
    except Exception:
        return _fallback_reply(message, model_id=model_id, error_type="unavailable")
