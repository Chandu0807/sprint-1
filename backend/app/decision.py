"""
decision.py - the eligibility decision engine (rules + ML). Framework-free, so it is easy to test.

Flow:  application form  ->  build_features()  ->  hard rules  ->  ML risk score  ->  decision + insights
"""
from __future__ import annotations

from datetime import date

from . import ml_core as mc

# ----------------------------------------------------------------------------- policy settings (edit freely)
POLICY = {
    "min_age": 18, "max_age": 60,
    "max_active_lenders": 4,            # bureau loans + declared informal loans
    "max_emi_share_of_household": 0.45, # total EMIs (old + new) must stay under 45% of household income
    "max_share_of_surplus": 0.70,       # new EMI must fit inside 70% of the monthly surplus
    "min_loan": 5000,
    "tier_cap": {"D": 15000, "C": 30000},   # starter-loan limits for thin-file borrowers (graduated later)
    "medium_risk_amount_factor": 0.80,
    "verification_pass_score": 50,
}

# Expected seasonal income level by calendar month (Jan..Dec); same curves the synthetic data was built with
_SEASON = {
    "farmer": [0.9, 0.8, 1.5, 1.6, 0.8, 0.5, 0.4, 0.4, 0.6, 1.6, 1.8, 1.1],
    "daily_wager": [1.0, 1.05, 1.1, 1.1, 1.05, 0.95, 0.8, 0.8, 0.9, 1.0, 1.05, 1.05],
    "vendor": [0.95, 0.9, 1.0, 0.95, 0.95, 0.9, 0.9, 0.95, 1.05, 1.25, 1.2, 1.0],
    "small_shopkeeper": [0.97, 0.95, 1.0, 0.98, 0.98, 0.95, 0.95, 0.98, 1.03, 1.15, 1.1, 1.0],
    "artisan": [1.1, 1.0, 0.9, 0.9, 0.95, 0.8, 0.7, 0.85, 1.0, 1.3, 1.3, 1.2],
    "driver": [1.0, 1.0, 1.05, 1.05, 1.1, 0.95, 0.85, 0.85, 0.95, 1.05, 1.05, 1.1],
    "dairy_livestock": [1.05, 1.05, 1.0, 0.95, 0.9, 0.9, 0.95, 1.0, 1.0, 1.05, 1.05, 1.05],
}
_IRR_AMP = {"rainfed": 1.0, "borewell": 0.7, "canal": 0.6}
_CV = {"low": 0.08, "medium": 0.22, "high": 0.45}
_RECHARGE = {"regular": 0.85, "irregular": 0.45}

# What the officer can tell the borrower, by the biggest risk driver
IMPROVE = {
    "total_emi_burden_known": "Ask for a smaller amount or a longer tenure so the EMI is lighter, or clear an existing EMI first.",
    "emi_to_effective_income": "Ask for a smaller amount or a longer tenure so the EMI is lighter.",
    "loan_amount": "Ask for a smaller amount.",
    "cibil_score": "Repay a small loan on time to build a credit record, then reapply.",
    "has_cibil": "Build a repayment record with a small starter loan.",
    "bureau_overdue_amount": "Clear the overdue amount on the credit file, then reapply.",
    "max_dpd_12m": "Keep every instalment on time for 6 months to repair the repayment record.",
    "loan_gap": "Declare all existing loans honestly; undeclared loans reduce trust.",
    "declared_outside_loans": "Reduce the number of loans running at the same time.",
    "informal_debt_amount_declared": "Repay moneylender or informal debt first.",
    "surplus_income": "Increase monthly surplus (cut expenses or add income) before borrowing more.",
    "elec_on_time_ratio_12m": "Pay electricity bills on time for several months; it counts as proof of discipline.",
    "elec_months_overdue_12m": "Clear overdue utility bills.",
    "recharge_regularity": "Keep a regular mobile recharge pattern.",
    "shg_member": "Join a self-help group; group backing improves approval chances.",
    "references_positive_ratio": "Provide more references who can vouch for you.",
    "verification_score": "Complete the field verification with supporting documents.",
    "hist_income_cv_12m": "Keep a simple record of monthly income to show stability.",
    "data_gaps": "Provide more documents (PAN, bank statement, utility bills) to raise confidence.",
    "data_confidence_score": "Provide more documents (PAN, bank statement, utility bills) to raise confidence.",
}


# ----------------------------------------------------------------------------- helpers
def emi_for(amount, annual_rate_pct, months):
    r = annual_rate_pct / 1200.0
    if r <= 0:
        return amount / months
    k = (1 + r) ** months
    return amount * r * k / (k - 1)


def max_principal(emi, annual_rate_pct, months):
    r = annual_rate_pct / 1200.0
    if emi <= 0:
        return 0.0
    if r <= 0:
        return emi * months
    k = (1 + r) ** months
    return emi * (k - 1) / (r * k)


def lean_flag(occupation, irrigation, month):
    curve = _SEASON.get(occupation)
    if curve is None:
        return 0
    mean = sum(curve) / 12.0
    amp = _IRR_AMP.get(irrigation, 1.0) if occupation == "farmer" else 1.0
    season = 1 + (curve[month - 1] / mean - 1) * amp
    return int(season < 0.85)


def _f(x, default=None):
    return default if x is None else float(x)


def _b(x):
    return None if x is None else float(bool(x))


def build_features(form, today=None):
    """Turn the application form into the exact features the model was trained on.
    Returns (features, extras, notes). `extras` holds derived numbers used by the rules and the UI."""
    today = today or date.today()
    notes = []

    declared = float(form["declared_monthly_income"])
    eff = _f(form.get("effective_income")) or declared * 0.85
    if not form.get("effective_income"):
        notes.append("Income is self-declared: counted at 85% of the declared figure until verified.")

    has_cibil = form.get("cibil_score") is not None
    bureau_loans = _f(form.get("bureau_active_loans"), 0.0) if has_cibil else None
    bureau_emi = _f(form.get("bureau_total_monthly_emi"), 0.0) if has_cibil else None
    declared_formal = int(form.get("declared_formal_loans") or 0)
    declared_informal = int(form.get("declared_informal_loans") or 0)
    loan_gap = (bureau_loans - declared_formal) if has_cibil else None
    if loan_gap is not None and loan_gap > 0:
        notes.append(f"Hidden-loan flag: the credit file shows {int(loan_gap)} more formal loan(s) than the borrower declared.")

    informal_emi = _f(form.get("informal_emi_monthly"), 0.0)
    outside_emi = (bureau_emi or 0.0) + informal_emi
    other_inc = _f(form.get("other_income_amount"), 0.0)
    household = eff + _f(form.get("other_earner_income"), 0.0) + other_inc
    expenses = _f(form.get("monthly_expenses"))
    if expenses is None:
        expenses = 0.65 * household
        notes.append("Household expenses were not given: estimated at 65% of household income.")
    surplus = household - expenses - outside_emi

    amount, months, rate = float(form["loan_amount"]), int(form["tenure_months"]), float(form.get("interest_rate_pct") or 22.0)
    emi = emi_for(amount, rate, months)

    elec_ok = bool(form.get("electricity_available")) and form.get("elec_months_on_time") is not None
    elec_ratio = float(form["elec_months_on_time"]) / 12.0 if elec_ok else None
    elec_overdue = _f(form.get("elec_months_overdue"), 0.0) if elec_ok else None
    refs_total, refs_pos = int(form.get("references_total") or 0), int(form.get("references_positive") or 0)
    shg = bool(form.get("shg_member"))
    years_lender = _f(form.get("years_with_lender"), 0.0)
    proof = form.get("income_proof_type") or "none"

    conf = min(100, 15 * bool(form.get("has_pan")) + 10 * bool(form.get("has_aadhaar")) + 5 * bool(form.get("has_voter_id"))
               + 5 * bool(form.get("has_ration_card")) + 8 * bool(form.get("bank_account_exists")) + 15 * (proof != "none")
               + 20 * has_cibil + 8 * bool(form.get("electricity_available")) + 8 * (years_lender > 0) + 6 * shg
               + 5 * (refs_total >= 2))
    tier = "A" if (form.get("has_pan") and has_cibil) else "B" if conf >= 58 else "C" if conf >= 42 else "D"

    occ, irr = form["occupation"], form.get("irrigation_type")
    feats = {
        "tier": tier, "data_confidence_score": conf, "has_pan": _b(form.get("has_pan")), "has_cibil": float(has_cibil),
        "cibil_score": _f(form.get("cibil_score")), "credit_history_length_years": _f(form.get("credit_history_length_years")) if has_cibil else None,
        "bureau_active_loans": bureau_loans, "bureau_total_monthly_emi": bureau_emi,
        "bureau_overdue_amount": _f(form.get("bureau_overdue_amount"), 0.0) if has_cibil else None,
        "max_dpd_12m": _f(form.get("max_dpd_12m"), 0.0) if has_cibil else None,
        "enquiries_6m": _f(form.get("enquiries_6m"), 0.0) if has_cibil else None,
        "written_off_or_settled": float(bool(form.get("written_off_or_settled"))) if has_cibil else None,
        "declared_outside_loans": float(declared_formal + declared_informal), "loan_gap": loan_gap,
        "informal_debt_amount_declared": _f(form.get("informal_debt_amount_declared"), 0.0),
        "years_with_lender": years_lender, "prev_loans_avg_delay_days": _f(form.get("prev_loans_avg_delay_days")) if years_lender > 0 else None,
        "occupation": occ, "age": float(form["age"]), "family_size": float(form["family_size"]), "dependents": float(form.get("dependents") or 0),
        "declared_monthly_income": declared, "effective_income": eff, "hist_income_cv_12m": _CV.get(form.get("income_variability")),
        "lean_season_at_disbursement": float(lean_flag(occ, irr, today.month)), "other_income_amount": other_inc,
        "household_income": household, "surplus_income": surplus, "owns_land": _b(form.get("owns_land")),
        "land_area_acres": _f(form.get("land_area_acres")) if form.get("owns_land") else None,
        "irrigation_type": irr if form.get("owns_land") else None,
        "elec_on_time_ratio_12m": elec_ratio, "elec_months_overdue_12m": elec_overdue,
        "recharge_regularity": _RECHARGE.get(form.get("recharge_regularity")), "rent_paid_regularly": _b(form.get("rent_paid_regularly")),
        "shg_member": float(shg), "shg_repayment_record": _f(form.get("shg_repayment_record")) if shg else None,
        "references_positive_ratio": (refs_pos / refs_total) if refs_total > 0 else None,
        "verification_score": _f(form.get("verification_score")),
        "loan_amount": amount, "tenure_months": float(months), "emi_to_effective_income": emi / eff,
        "total_emi_burden_known": (emi + outside_emi) / household,
    }
    missing = set(mc.UW_FEATURES) - set(feats)
    assert not missing, f"form is missing model features: {missing}"

    extras = {"emi": emi, "outside_emi": outside_emi, "household": household, "expenses": expenses, "surplus": surplus,
              "rate": rate, "months": months, "amount": amount, "tier": tier, "confidence": conf}
    return feats, extras, notes


def hard_rules(form, feats, extras):
    """Policy rules that apply BEFORE any model. Returns a list of reasons the application must be declined."""
    fails = []
    if not (POLICY["min_age"] <= feats["age"] <= POLICY["max_age"]):
        fails.append(f"Age must be between {POLICY['min_age']} and {POLICY['max_age']}.")
    if feats["written_off_or_settled"]:
        fails.append("A previous loan was written off or settled on the credit file.")
    lenders = (feats["bureau_active_loans"] or 0) + int(form.get("declared_informal_loans") or 0)
    if lenders >= POLICY["max_active_lenders"]:
        fails.append(f"Too many active lenders ({int(lenders)}); the borrower is likely over-indebted.")
    if (feats["max_dpd_12m"] or 0) >= 90:
        fails.append("Credit file shows 90+ days overdue in the last 12 months.")
    return fails


def capacity(extras):
    """How much EMI can this household safely add, and what loan does that support?"""
    room_total = POLICY["max_emi_share_of_household"] * extras["household"] - extras["outside_emi"]
    room_surplus = POLICY["max_share_of_surplus"] * extras["surplus"]
    safe_emi = max(0.0, min(room_total, room_surplus))
    return {"safe_emi": safe_emi, "max_affordable_loan": max_principal(safe_emi, extras["rate"], extras["months"])}


# ----------------------------------------------------------------------------- main entry point
def evaluate(form, uw_artifact, today=None):
    feats, ex, notes = build_features(form, today)
    cap = capacity(ex)
    tier = ex["tier"]
    rules = hard_rules(form, feats, ex)
    ml = mc.score_records(uw_artifact, [feats])[0]
    band, risk = ml["band"], ml["risk"]
    vscore = feats["verification_score"]

    cond, steps = [], []
    decision, rec = None, None
    if rules:
        decision = "not_eligible"
    elif vscore is not None and vscore < POLICY["verification_pass_score"]:
        decision = "not_eligible"
        rules.append(f"Field verification score {vscore:.0f} is below the pass mark ({POLICY['verification_pass_score']}).")
    elif tier in ("C", "D") and vscore is None:
        decision = "manual_verification"
        steps = ["A field officer must visit the home and workplace and complete the 7-point verification checklist.",
                 "Collect at least two references (neighbour, shopkeeper, group leader).",
                 "After supervisor approval the application is re-scored automatically."]
    elif cap["safe_emi"] <= 0:
        decision = "not_eligible"
        rules.append("No repayment capacity: income minus expenses and existing EMIs leaves nothing for a new EMI.")
    elif band == "high":
        decision = "not_eligible"
    else:
        factor = POLICY["medium_risk_amount_factor"] if band == "medium" else 1.0
        limit = min(ex["amount"], cap["max_affordable_loan"] * factor, POLICY["tier_cap"].get(tier, float("inf")))
        rec = int(limit // 1000 * 1000)
        if rec < POLICY["min_loan"]:
            decision = "not_eligible"
            rules.append("Affordable loan amount is below the minimum loan size.")
            rec = None
        else:
            decision = "eligible" if (band == "low" and rec >= ex["amount"]) else "eligible_with_conditions"
            if rec < ex["amount"]:
                cond.append(f"Reduced amount: Rs {rec:,} instead of Rs {int(ex['amount']):,} to keep the EMI affordable.")
            if band == "medium":
                cond += ["Group (SHG) guarantee or a verified co-applicant.", "First three collections are monitored closely."]
            if tier in ("C", "D"):
                cond.append("Starter loan: the limit increases after each on-time repayment cycle.")

    improve = []
    for r in ml["reasons"]:
        tip = IMPROVE.get(r["feature"])
        if tip and tip not in improve:
            improve.append(tip)

    return {
        "decision": decision, "risk_score": round(risk, 4), "risk_band": band, "tier": tier,
        "data_confidence_score": ex["confidence"],
        "reasons_increasing_risk": ml["reasons"], "reasons_reducing_risk": ml["protective"],
        "rules_triggered": rules, "notes": notes, "conditions": cond, "next_steps": steps,
        "how_to_improve": improve[:4] if decision in ("not_eligible", "eligible_with_conditions") else [],
        "requested_amount": int(ex["amount"]), "recommended_amount": rec,
        "recommended_emi": round(emi_for(rec, ex["rate"], ex["months"])) if rec else None,
        "tenure_months": ex["months"], "interest_rate_pct": ex["rate"],
        "capacity": {"household_income": round(ex["household"]), "expenses": round(ex["expenses"]),
                     "existing_emi": round(ex["outside_emi"]), "surplus": round(ex["surplus"]),
                     "safe_emi": round(cap["safe_emi"]), "max_affordable_loan": int(cap["max_affordable_loan"] // 1000 * 1000)},
        "model_features": {k: (None if v is None else v) for k, v in feats.items()},
    }
