"""
ml_core.py - shared, framework-free ML helpers.
Used by BOTH the training script (ml/train.py) and the API (backend/app/main.py),
so the model sees exactly the same feature encoding in training and in production.
No FastAPI / database imports here, so it can be tested on its own.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- feature lists
# Text columns are turned into integer codes using these FIXED lists (order matters).
CATEGORICAL = {
    "occupation": ["farmer", "daily_wager", "vendor", "small_shopkeeper", "artisan",
                   "domestic_worker", "driver", "salaried", "dairy_livestock"],
    "irrigation_type": ["rainfed", "borewell", "canal"],
    "tier": ["A", "B", "C", "D"],
}

# Underwriting model: exactly the things the loan-application form can collect.
UW_FEATURES = [
    "tier", "data_confidence_score", "has_pan", "has_cibil", "cibil_score", "credit_history_length_years",
    "bureau_active_loans", "bureau_total_monthly_emi", "bureau_overdue_amount", "max_dpd_12m", "enquiries_6m",
    "written_off_or_settled", "declared_outside_loans", "loan_gap", "informal_debt_amount_declared",
    "years_with_lender", "prev_loans_avg_delay_days", "occupation", "age", "family_size", "dependents",
    "declared_monthly_income", "effective_income", "hist_income_cv_12m", "lean_season_at_disbursement",
    "other_income_amount", "household_income", "surplus_income", "owns_land", "land_area_acres", "irrigation_type",
    "elec_on_time_ratio_12m", "elec_months_overdue_12m", "recharge_regularity", "rent_paid_regularly",
    "shg_member", "shg_repayment_record", "references_positive_ratio", "verification_score",
    "loan_amount", "tenure_months", "emi_to_effective_income", "total_emi_burden_known",
]

# Monitoring (early-warning) model: signals the system collects every month after disbursement.
MON_FEATURES = [
    "months_on_book", "calendar_month", "late_or_short_payments_3m", "missed_payments_3m", "missed_payments_total",
    "avg_delay_days_3m", "max_delay_days_so_far", "delay_change_vs_prior", "paid_to_due_ratio_3m",
    "income_vs_effective_3m", "income_vs_expected_3m", "income_vs_expected_last", "income_vs_expected_prev",
    "sudden_income_drop_flag", "income_cv_since_disbursal", "elec_days_late_last", "elec_avg_delay_3m",
    "elec_late_months_3m", "elec_overdue_flag", "elec_units_change_vs_12m", "recharge_amount_ratio_3m",
    "recharge_drop_flag", "recharge_lapse_months_3m", "new_outside_loan_last_3m", "new_outside_loans_since_disbursal",
    "district_shock_intensity_2m", "rainfall_deficit_pct", "missed_group_meetings_3m", "lean_season_flag",
    "occupation", "tier", "data_confidence_score", "has_cibil", "cibil_score", "emi_to_effective_income",
    "total_emi_burden_known", "declared_outside_loans", "shg_member", "loan_amount", "age",
]

BOOL_FEATURES = {
    "has_pan", "has_cibil", "written_off_or_settled", "lean_season_at_disbursement", "owns_land", "rent_paid_regularly",
    "shg_member", "sudden_income_drop_flag", "elec_overdue_flag", "recharge_drop_flag", "new_outside_loan_last_3m",
    "lean_season_flag",
}

LABELS = {
    "tier": "Data tier", "data_confidence_score": "Data confidence", "has_pan": "Has PAN", "has_cibil": "Has credit file",
    "cibil_score": "CIBIL score", "credit_history_length_years": "Credit history (years)",
    "bureau_active_loans": "Active loans on bureau", "bureau_total_monthly_emi": "Bureau EMI total",
    "bureau_overdue_amount": "Overdue amount", "max_dpd_12m": "Worst delay, 12m (days)", "enquiries_6m": "Loan enquiries, 6m",
    "written_off_or_settled": "Past write-off/settlement", "declared_outside_loans": "Declared outside loans",
    "loan_gap": "Loans on bureau not declared", "informal_debt_amount_declared": "Informal debt",
    "years_with_lender": "Years with us", "prev_loans_avg_delay_days": "Avg delay on earlier loans (days)",
    "occupation": "Occupation", "age": "Age", "family_size": "Family size", "dependents": "Dependents",
    "declared_monthly_income": "Declared income", "effective_income": "Effective income",
    "hist_income_cv_12m": "Income variability", "lean_season_at_disbursement": "Loan starts in lean season",
    "other_income_amount": "Other income", "household_income": "Household income", "surplus_income": "Monthly surplus",
    "owns_land": "Owns land", "land_area_acres": "Land (acres)", "irrigation_type": "Irrigation",
    "elec_on_time_ratio_12m": "Electricity bills paid on time", "elec_months_overdue_12m": "Electricity months overdue",
    "recharge_regularity": "Mobile recharge regularity", "rent_paid_regularly": "Rent paid regularly",
    "shg_member": "Group (SHG) member", "shg_repayment_record": "Group repayment record",
    "references_positive_ratio": "Positive references", "verification_score": "Field verification score",
    "loan_amount": "Loan amount", "tenure_months": "Tenure (months)", "emi_to_effective_income": "EMI as share of income",
    "total_emi_burden_known": "Total EMI burden",
    "months_on_book": "Months since disbursement", "calendar_month": "Calendar month",
    "late_or_short_payments_3m": "Late/short payments (3m)", "missed_payments_3m": "Missed payments (3m)",
    "missed_payments_total": "Missed payments (total)", "avg_delay_days_3m": "Average delay (3m, days)",
    "max_delay_days_so_far": "Worst delay so far (days)", "delay_change_vs_prior": "Delay trend",
    "paid_to_due_ratio_3m": "Paid vs due (3m)", "income_vs_effective_3m": "Income vs normal (3m)",
    "income_vs_expected_3m": "Income vs seasonal expectation (3m)", "income_vs_expected_last": "Income vs expectation (this month)",
    "income_vs_expected_prev": "Income vs expectation (last month)", "sudden_income_drop_flag": "Sudden income drop",
    "income_cv_since_disbursal": "Income variability since loan", "elec_days_late_last": "Electricity bill delay (days)",
    "elec_avg_delay_3m": "Electricity delay (3m avg)", "elec_late_months_3m": "Late electricity bills (3m)",
    "elec_overdue_flag": "Electricity bill overdue", "elec_units_change_vs_12m": "Electricity usage change",
    "recharge_amount_ratio_3m": "Recharge vs normal", "recharge_drop_flag": "Recharge dropped",
    "recharge_lapse_months_3m": "Months without recharge", "new_outside_loan_last_3m": "New loan elsewhere (3m)",
    "new_outside_loans_since_disbursal": "New outside loans since disbursal", "district_shock_intensity_2m": "District drought/price shock",
    "rainfall_deficit_pct": "Rainfall deficit %", "missed_group_meetings_3m": "Missed group meetings", "lean_season_flag": "Lean season",
}


# --------------------------------------------------------------------------- encoding
def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def to_frame(records, features):
    """List of dicts (or DataFrame) -> numeric DataFrame with columns in `features` order.
    Missing columns/values become NaN (gradient boosting handles NaN natively)."""
    df = records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame(list(records))
    for c in features:
        if c not in df.columns:
            df[c] = np.nan
    X = df[features].copy()
    for c, cats in CATEGORICAL.items():
        if c in X.columns:
            X[c] = X[c].map({v: float(i) for i, v in enumerate(cats)})   # unknown text -> NaN
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = X[c].map(_num)
        X[c] = X[c].astype(float)
    return X.reset_index(drop=True)


def baseline_values(X):
    """'Typical borrower' value per feature (median, or most common category). Used to explain predictions."""
    base = {}
    for c in X.columns:
        s = X[c].dropna()
        if s.empty:
            base[c] = np.nan
        elif c in CATEGORICAL:
            base[c] = float(s.mode().iloc[0])
        else:
            base[c] = float(s.median())
    return base


def make_model():
    """LightGBM if installed (recommended); otherwise scikit-learn's histogram gradient boosting.
    Small, heavily-regularised trees: the dataset is modest in size, so simple models generalise best."""
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=300, learning_rate=0.04, num_leaves=8, min_child_samples=80,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
                              random_state=42, verbose=-1), "lightgbm"
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.04, max_leaf_nodes=8, min_samples_leaf=80,
                                              l2_regularization=5.0, random_state=42), "sklearn-fallback"


# --------------------------------------------------------------------------- explanations
def fmt(feature, v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "not available"
    if feature in CATEGORICAL:
        return CATEGORICAL[feature][int(v)]
    if feature in BOOL_FEATURES:
        return "Yes" if v >= 0.5 else "No"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    return f"{v:.2f}".rstrip("0").rstrip(".")


def occlusion_deltas(model, X, base):
    """For each feature: how much does risk change if that feature is replaced by the typical value?
    delta > 0  => this borrower's value PUSHES RISK UP compared with a typical borrower.
    Works with any model that has predict_proba (no SHAP needed)."""
    p0 = model.predict_proba(X)[:, 1]
    D = np.zeros(X.shape, dtype=float)
    for j, c in enumerate(X.columns):
        Xc = X.copy()
        Xc[c] = base[c]
        D[:, j] = p0 - model.predict_proba(Xc)[:, 1]
    return p0, D


def top_reasons(values, deltas, cols, k_up=5, k_down=3, min_abs=0.002):
    """Pick the biggest risk-raising and risk-lowering factors. Features with no value are not listed one by one
    (e.g. "Bureau EMI: not available" is not actionable); their combined effect becomes ONE reason: limited data."""
    missing = {i for i in range(len(cols)) if np.isnan(values[i])}
    order = np.argsort(-deltas)

    def item(i):
        return {"feature": cols[i], "label": LABELS.get(cols[i], cols[i].replace("_", " ")),
                "value": fmt(cols[i], values[i]), "impact_pp": round(float(deltas[i]) * 100, 1)}
    up = [item(i) for i in order if deltas[i] > min_abs and i not in missing]
    down = [item(i) for i in order[::-1] if deltas[i] < -min_abs and i not in missing][:k_down]
    gap = float(sum(deltas[i] for i in missing))
    if gap > 0.005:
        up.append({"feature": "data_gaps", "label": "Limited verifiable data", "value": f"{len(missing)} details unavailable",
                   "impact_pp": round(gap * 100, 1)})
        up.sort(key=lambda r: -r["impact_pp"])
    return up[:k_up], down


def band_for(p, th):
    return "high" if p >= th["high"] else ("medium" if p >= th["medium"] else "low")


def score_records(art, records):
    """Score borrowers with a saved artifact. Returns one dict per record:
    risk (0-1), band (low/medium/high), reasons (push risk up), protective (push risk down)."""
    X = to_frame(records, art["features"])
    p0, D = occlusion_deltas(art["model"], X, art["baseline"])
    vals, cols = X.to_numpy(float), list(X.columns)
    out = []
    for i in range(len(X)):
        up, down = top_reasons(vals[i], D[i], cols)
        out.append({"risk": float(p0[i]), "band": band_for(p0[i], art["thresholds"]), "reasons": up, "protective": down})
    return out
