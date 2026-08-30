"""
Investment Calculator Tool.

TRD 5.2 baseline: a flat monthly SIP against a horizon-bucket assumed return,
required_monthly = future_cost * r_monthly / ((1 + r_monthly) ** n - 1).

EXTENSION (approved by project owner, documented here as a deliberate deviation,
not a bug): instead of a flat payment, this solves for a STARTING (year-1)
monthly contribution that grows `salary_growth_rate` per year (matching the
project's 10% salary-increment assumption), while still earning the assumed
monthly investment return. At salary_growth_rate=0 this reduces exactly to the
TRD flat formula (see tests/test_edge_cases.py::test_investment_extension_reduces_to_trd_baseline).
Both the return-rate table and the growth-rate assumption are illustrative,
not a guarantee, per project rules.
"""

SALARY_GROWTH_RATE = 0.10


def assumed_annual_return(years):
    if years <= 3:
        return 0.06
    if years <= 7:
        return 0.09
    return 0.11


def investment_required(future_cost_value, years, annual_return_rate=None, salary_growth_rate=SALARY_GROWTH_RATE):
    if years == 0:
        return {
            "required_monthly_investment": round(future_cost_value, 2),
            "assumed_annual_return": None,
            "salary_growth_rate": None,
            "note": "lump sum required now (years=0)",
        }

    r_annual = annual_return_rate if annual_return_rate is not None else assumed_annual_return(years)
    r = r_annual / 12
    n = years * 12
    annuity_factor = ((1 + r) ** 12 - 1) / r if r > 0 else 12

    denom = sum(
        ((1 + salary_growth_rate) ** y) * annuity_factor * ((1 + r) ** (n - (y * 12 + 12)))
        for y in range(years)
    )

    return {
        "required_monthly_investment": round(future_cost_value / denom, 2),
        "assumed_annual_return": r_annual,
        "salary_growth_rate": salary_growth_rate,
    }
