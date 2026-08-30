from tools import feasibility_tool, future_cost_tool, investment_tool, recommendation_tool, salary_tool

DEFAULT_AREA_TYPE = "Suburban"


def compute_plan(age, city, education, job_role, savings_percentage, goals, area_type=DEFAULT_AREA_TYPE):
    salary_result = salary_tool.predict_salary(age, city, education, job_role)
    predicted_salary = salary_result["predicted_monthly_salary"]

    goal_results = []
    for goal in goals:
        goal_type = goal["goal_type"]
        years = goal["years"]

        cost = future_cost_tool.future_goal_cost(city, area_type, goal_type, years)
        investment = investment_tool.investment_required(cost["projected_cost"], years)
        category = recommendation_tool.recommend_category(years, goal_type)

        goal_results.append({
            "goal_type": goal_type,
            "years": years,
            "current_cost": cost["current_cost"],
            "projected_cost": cost["projected_cost"],
            "required_monthly_investment": investment["required_monthly_investment"],
            "assumed_annual_return": investment["assumed_annual_return"],
            "recommended_category": category["category"],
            "category_rationale": category["rationale"],
            "guidance": category["guidance"],
        })

    feasibility = feasibility_tool.feasibility_check(
        predicted_salary,
        savings_percentage,
        [{"goal_type": g["goal_type"], "required_monthly_investment": g["required_monthly_investment"]} for g in goal_results],
    )

    feasibility_by_goal = {g["goal_type"]: g for g in feasibility["per_goal"]}
    for g in goal_results:
        matched = feasibility_by_goal[g["goal_type"]]
        g["allocated_capacity"] = matched["allocated_capacity"]
        g["gap"] = matched["gap"]
        g["status"] = matched["status"]

    return {
        "predicted_monthly_salary": predicted_salary,
        "salary_low_confidence": salary_result["low_confidence"],
        "savings_percentage": savings_percentage,
        "available_capacity": feasibility["available_capacity"],
        "total_required_monthly": feasibility["total_required"],
        "total_gap": feasibility["total_gap"],
        "overall_status": feasibility["overall_status"],
        "goals": goal_results,
        "assumptions": {
            "inflation_rate": future_cost_tool.INFLATION_RATE,
            "salary_growth_rate": investment_tool.SALARY_GROWTH_RATE,
            "area_type_used": area_type,
        },
    }
