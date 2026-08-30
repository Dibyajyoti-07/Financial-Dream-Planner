import re

from tools.constants import KNOWN_CITIES, KNOWN_EDUCATION, KNOWN_JOB_ROLES

EDUCATION_ALIASES = {
    "btech": "B.Tech", "be": "B.E.", "bsc": "B.Sc", "bca": "BCA",
    "msc": "M.Sc", "mtech": "M.Tech", "mba": "MBA", "mca": "MCA",
}


def _normalize(text):
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _find_match(message, options, aliases=None):
    normalized_message = _normalize(message)
    for opt in options:
        if _normalize(opt) in normalized_message:
            return opt
    if aliases:
        for alias, canonical in aliases.items():
            if alias in normalized_message:
                return canonical
    return None


def _find_age(message):
    match = re.search(r"\b(?:age\D{0,5}|i\s*am\s*)(\d{1,3})\b", message, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _find_percentage(message):
    match = re.search(r"(\d+(?:\.\d+)?)\s*%", message)
    return float(match.group(1)) if match else None


def _find_goal_years(message, goal_keyword):
    pattern = rf"\b{goal_keyword}\D{{0,25}}?(\d+)\s*(?:year|yr)"
    match = re.search(pattern, message, re.IGNORECASE)
    return int(match.group(1)) if match else None


class ExtractionError(ValueError):
    pass


def extract_plan_request(message):
    city = _find_match(message, KNOWN_CITIES)
    if city is None:
        raise ExtractionError("Could not identify a known city in the message")

    education = _find_match(message, KNOWN_EDUCATION, EDUCATION_ALIASES)
    job_role = _find_match(message, KNOWN_JOB_ROLES)
    age = _find_age(message)
    savings_percentage = _find_percentage(message)

    if education is None or job_role is None or age is None or savings_percentage is None:
        raise ExtractionError("Could not extract all required fields (age, education, job role, savings %) from the message")

    goals = []
    for keyword, goal_type in [("marriage", "Marriage"), ("car", "Car"), ("home", "Home")]:
        years = _find_goal_years(message, keyword)
        if years is not None:
            goals.append({"goal_type": goal_type, "years": years})

    if not goals:
        raise ExtractionError("Could not identify any goal (Marriage/Car/Home) with a timeline in the message")

    return {
        "age": age,
        "city": city,
        "education": education,
        "job_role": job_role,
        "savings_percentage": savings_percentage,
        "goals": goals,
    }
