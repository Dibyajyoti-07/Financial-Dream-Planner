import sys
import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "models" / "salary_model.pkl"
GOAL_COST_PATH = Path(__file__).parent / "data" / "city_goal_costs.csv"
INFLATION_RATE = 0.06
SALARY_INCREMENT_RATE = 0.10

KNOWN_CITIES = ["Ahmedabad", "Bangalore", "Chennai", "Delhi", "Hyderabad", "Jaipur", "Kolkata", "Lucknow", "Mumbai", "Pune"]
KNOWN_EDUCATION = ["B.E.", "B.Sc", "B.Tech", "BCA", "M.Sc", "M.Tech", "MBA", "MCA"]
KNOWN_JOB_ROLES = ["Business Analyst", "Data Analyst", "Data Scientist", "DevOps Engineer", "Project Coordinator", "QA Engineer", "Software Engineer", "Technical Support Engineer", "UI UX Designer", "Web Developer"]
KNOWN_AREA_TYPES = ["Central", "North", "South", "East", "West", "Outer", "Suburban", "IT Corridor", "Premium", "Developing"]

GOAL_COLUMNS = {
    "Marriage": "Marriage_Cost_Current",
    "a new Car": "Car_Cost_Current",
    "a new Home": "Home_Cost_Current",
}


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"{MODEL_PATH} not found - run model_evaluation.ipynb first")
    return joblib.load(MODEL_PATH)


def load_goal_costs():
    if not GOAL_COST_PATH.exists():
        raise FileNotFoundError(f"{GOAL_COST_PATH} not found")
    return pd.read_csv(GOAL_COST_PATH)


def predict_salary(model, age, city, education, job_role):
    X = pd.DataFrame([{
        "Age": age,
        "City": city,
        "Education": education,
        "Job_Role": job_role,
    }])
    return float(model.predict(X)[0])


def future_cost(current_cost, years, rate=INFLATION_RATE):
    return current_cost * ((1 + rate) ** years)


def lookup_current_cost(goal_costs_df, city, area_type, goal_column):
    rows = goal_costs_df[
        (goal_costs_df["City"].str.lower() == city.lower())
        & (goal_costs_df["Area_Type"].str.lower() == area_type.lower())
    ]
    if rows.empty:
        return None
    return float(rows[goal_column].iloc[0])


def accumulated_savings(salary, investment_percent, years, increment_rate=SALARY_INCREMENT_RATE):
    total = 0.0
    for year in range(years):
        year_salary = salary * ((1 + increment_rate) ** year)
        total += year_salary * (investment_percent / 100) * 12
    return total


def classify_feasibility(target_savings, required_cost):
    if required_cost <= 0:
        return "Achievable"
    ratio = target_savings / required_cost
    if ratio >= 1.0:
        return "Achievable"
    if ratio >= 0.7:
        return "Challenging"
    return "Highly Challenging"


def ask_text(prompt, allow_empty=False, default=None):
    while True:
        value = input(prompt).strip()
        if value:
            return value
        if allow_empty:
            return default
        print("This field can't be empty. Try again.")


def ask_int(prompt, min_value=None, max_value=None):
    while True:
        raw = input(prompt).strip()
        try:
            value = int(raw)
        except ValueError:
            print("Enter a whole number.")
            continue
        if min_value is not None and value < min_value:
            print(f"Value must be at least {min_value}.")
            continue
        if max_value is not None and value > max_value:
            print(f"Value must be at most {max_value}.")
            continue
        return value


def ask_float(prompt, min_value=0.0, max_value=None):
    while True:
        raw = input(prompt).strip()
        try:
            value = float(raw)
        except ValueError:
            print("Enter a number.")
            continue
        if value < min_value:
            print(f"Value must be at least {min_value}.")
            continue
        if max_value is not None and value > max_value:
            print(f"Value must be at most {max_value}.")
            continue
        return value


def ask_yes_no(prompt):
    while True:
        raw = input(prompt + " (y/n): ").strip().lower()
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("Please answer y or n.")


def normalize_choice(text):
    return "".join(ch for ch in text.lower() if ch.isalnum())


def ask_choice(prompt, options):
    options_display = ", ".join(options)
    normalized_options = {normalize_choice(opt): opt for opt in options}
    while True:
        raw = input(f"{prompt} ({options_display}): ").strip()
        if not raw:
            print("This field can't be empty. Try again.")
            continue
        match = normalized_options.get(normalize_choice(raw))
        if match:
            return match
        print(f"'{raw}' is not a valid option. Please choose one of: {options_display}.")


def collect_goal(goal_costs_df, goal_name, city, area_type):
    if not ask_yes_no(f"Are you planning for {goal_name}?"):
        return None
    years = ask_int(f"In how many years do you want to achieve {goal_name}? ", min_value=1, max_value=50)
    current_cost = lookup_current_cost(goal_costs_df, city, area_type, GOAL_COLUMNS[goal_name])
    if current_cost is None:
        print(f"No cost data available for {city} / {area_type} - skipping {goal_name} projection.")
        return None
    projected = future_cost(current_cost, years)
    return {"goal": goal_name, "years": years, "current_cost": current_cost, "projected_cost": projected}


def main():
    model = load_model()
    goal_costs_df = load_goal_costs()

    print("=== Financial Dream Planner - Salary & Goal Predictor ===")
    name = ask_text("Name: ", allow_empty=True, default="User")
    age = ask_int("Age: ", min_value=18, max_value=70)
    city = ask_choice("City", KNOWN_CITIES)
    area = ask_choice("Area", KNOWN_AREA_TYPES)
    education = ask_choice("Education", KNOWN_EDUCATION)
    job_role = ask_choice("Job Role", KNOWN_JOB_ROLES)

    salary = predict_salary(model, age, city, education, job_role)
    print(f"\nHi {name}, predicted Monthly Salary: {salary:,.0f}")

    goals = [g for g in (collect_goal(goal_costs_df, goal_name, city, area) for goal_name in GOAL_COLUMNS) if g]

    if not goals:
        print("\nNo goals selected - nothing further to project.")
        return

    investment_percent = ask_float(
        "\nWhat percentage of your monthly salary can you currently save/invest? (e.g. 18.5 for 18.5%): ",
        min_value=0.0,
        max_value=100.0,
    )
    starting_monthly = salary * (investment_percent / 100)
    print(f"That's {starting_monthly:,.0f}/month in year 1, growing with a {SALARY_INCREMENT_RATE:.0%} assumed annual salary increment.")

    print("\n=== Goal Projections ===")
    for goal in goals:
        savings = accumulated_savings(salary, investment_percent, goal["years"])
        feasibility = classify_feasibility(savings, goal["projected_cost"])
        print(
            f"{goal['goal']} in {goal['years']} yrs -> current cost {goal['current_cost']:,.0f}, "
            f"projected cost (6% inflation) {goal['projected_cost']:,.0f}, "
            f"savings by then (10% annual salary growth) {savings:,.0f} -> {feasibility}"
        )


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.")
        sys.exit(1)
