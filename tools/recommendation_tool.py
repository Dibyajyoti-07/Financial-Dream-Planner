from rag import retriever
from tools.investment_tool import assumed_annual_return

_CATEGORY_BY_HORIZON = [
    (3, "capital-preservation", "Short horizon - prioritize protecting principal over growth."),
    (7, "balanced", "Medium horizon - a mix of stability and growth can absorb moderate swings."),
    (float("inf"), "growth-oriented", "Long horizon - more time to ride out volatility for higher growth potential."),
]


def recommend_category(years, goal_type=None):
    for max_years, category, rationale in _CATEGORY_BY_HORIZON:
        if years <= max_years:
            query = f"{category} investment category"
            if goal_type:
                query = f"{category} investment category for a {goal_type} goal"
            return {
                "category": category,
                "rationale": rationale,
                "assumed_annual_return": assumed_annual_return(years),
                "guidance": retriever.retrieve(query, k=2),
            }
    raise AssertionError("unreachable")
