def _status(required, allocated):
    if allocated <= 0:
        return "Highly Challenging" if required > 0 else "Achievable"
    if required <= allocated:
        return "Achievable"
    over_pct = (required - allocated) / allocated
    if over_pct <= 0.25:
        return "Challenging"
    return "Highly Challenging"


def feasibility_check(predicted_monthly_salary, savings_percentage, goal_requirements):
    available_capacity = (savings_percentage / 100) * predicted_monthly_salary
    total_required = sum(g["required_monthly_investment"] for g in goal_requirements)

    per_goal = []
    for g in goal_requirements:
        required = g["required_monthly_investment"]
        share = (required / total_required) if total_required > 0 else 0.0
        allocated = share * available_capacity
        gap = allocated - required
        per_goal.append({
            "goal_type": g["goal_type"],
            "required_monthly_investment": round(required, 2),
            "allocated_capacity": round(allocated, 2),
            "gap": round(gap, 2),
            "status": _status(required, allocated),
        })

    status_rank = {"Achievable": 0, "Challenging": 1, "Highly Challenging": 2}
    overall_status = max((g["status"] for g in per_goal), key=lambda s: status_rank[s], default="Achievable")

    return {
        "available_capacity": round(available_capacity, 2),
        "total_required": round(total_required, 2),
        "total_gap": round(available_capacity - total_required, 2),
        "per_goal": per_goal,
        "overall_status": overall_status,
    }
