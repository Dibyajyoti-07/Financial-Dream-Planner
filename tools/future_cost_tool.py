from pathlib import Path

import pandas as pd

GOAL_COST_PATH = Path(__file__).parent.parent / "data" / "city_goal_costs.csv"
INFLATION_RATE = 0.06

GOAL_COLUMNS = {
    "Marriage": "Marriage_Cost_Current",
    "Car": "Car_Cost_Current",
    "Home": "Home_Cost_Current",
}

_goal_costs_df = None


class UnknownCityError(ValueError):
    pass


class UnknownAreaTypeError(ValueError):
    pass


def _load():
    global _goal_costs_df
    if _goal_costs_df is None:
        if not GOAL_COST_PATH.exists():
            raise FileNotFoundError(f"{GOAL_COST_PATH} not found")
        _goal_costs_df = pd.read_csv(GOAL_COST_PATH)
    return _goal_costs_df


def is_loaded():
    try:
        _load()
        return True
    except FileNotFoundError:
        return False


def lookup_current_cost(city, area_type, goal_type):
    df = _load()
    goal_column = GOAL_COLUMNS[goal_type]

    city_rows = df[df["City"].str.lower() == city.lower()]
    if city_rows.empty:
        raise UnknownCityError(f"Unknown city: {city}")

    matched = city_rows[city_rows["Area_Type"].str.lower() == area_type.lower()]
    if matched.empty:
        raise UnknownAreaTypeError(f"Unknown area type '{area_type}' for city '{city}'")

    return float(matched[goal_column].iloc[0])


def future_goal_cost(city, area_type, goal_type, years):
    current_cost = lookup_current_cost(city, area_type, goal_type)
    projected_cost = current_cost * ((1 + INFLATION_RATE) ** years)
    return {
        "goal_type": goal_type,
        "years": years,
        "current_cost": round(current_cost, 2),
        "projected_cost": round(projected_cost, 2),
        "inflation_rate": INFLATION_RATE,
    }
