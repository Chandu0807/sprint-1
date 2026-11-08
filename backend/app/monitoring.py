"""
monitoring.py - turns monthly risk scores into early-warning alerts with a recommended action.
Framework-free (no database here), so it can be unit-tested.
"""
from __future__ import annotations

# (feature-name prefixes) -> (alert type, what the loan officer should do)
DRIVERS = [
    (("income_", "sudden_income"),
     ("income_shortfall", "Income is below the seasonal norm. Offer a due-date shift or a temporary EMI reduction, and confirm with a home or business visit.")),
    (("elec_", "recharge_"),
     ("utility_stress", "Utility and recharge behaviour shows cash stress. Call the borrower this week and review household cash flow.")),
    (("new_outside_loan",),
     ("debt_stacking", "A new loan was taken elsewhere. Arrange a debt-counselling visit and review total obligations.")),
    (("missed_payments", "late_or_short", "avg_delay", "delay_change", "max_delay", "paid_to_due"),
     ("repayment_slippage", "Payments are slipping. Offer a restructure or a revised schedule before the next due date.")),
    (("district_shock", "rainfall_", "crop_price"),
     ("weather_shock", "A drought or crop-price shock hit the district. Consider a seasonal schedule for all affected borrowers.")),
    (("missed_group",),
     ("group_signal", "Group meetings are being missed. Ask the group leader to follow up and check on the borrower.")),
]
DEFAULT_DRIVER = ("elevated_risk", "Risk has risen. Review the borrower with the branch manager and plan a check-in visit.")

JUMP_ALERT = 0.12      # risk rising by 12 percentage points in one month triggers an alert
JUMP_WATCH = 0.05      # a smaller rise matters if the borrower is already medium risk


def classify(feature):
    for prefixes, driver in DRIVERS:
        if feature.startswith(prefixes):
            return driver
    return DEFAULT_DRIVER


def build_alert(res, prev_risk):
    """res = one result from ml_core.score_records(). Returns an alert dict, or None if no alert is needed."""
    risk, band = res["risk"], res["band"]
    jump = (risk - prev_risk) if prev_risk is not None else 0.0
    trigger = band == "high" or jump >= JUMP_ALERT or (band == "medium" and jump >= JUMP_WATCH)
    if not trigger:
        return None

    top = res["reasons"][:2]
    alert_type, action = classify(top[0]["feature"]) if top else DEFAULT_DRIVER
    drivers = ", ".join(f"{r['label']} ({r['value']})" for r in top) or "multiple small factors"
    msg = f"Risk {risk:.0%} ({band})"
    if prev_risk is not None and abs(jump) >= 0.01:
        msg += f", {'up' if jump > 0 else 'down'} {abs(jump) * 100:.0f} points since last month"
    msg += f". Main drivers: {drivers}."
    return {"severity": "high" if band == "high" else "medium", "alert_type": alert_type,
            "message": msg, "recommended_action": action}
