"""
Tests for the framework-free logic (no database, no web server needed):

    cd backend && python tests/test_core.py        (run ml/train.py first and put the CSVs in ../data)
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app import decision, ml_core as mc, monitoring  # noqa: E402

UW = joblib.load(ROOT / "ml" / "artifacts" / "underwriting_model.joblib")
MON = joblib.load(ROOT / "ml" / "artifacts" / "monitoring_model.joblib")


def form(**kw):
    """A typical tier-A applicant (farmer with PAN and CIBIL); override fields per scenario."""
    f = dict(full_name="Test", age=38, occupation="farmer", family_size=4, dependents=2,
             has_pan=True, has_aadhaar=True, has_voter_id=True, has_ration_card=True, bank_account_exists=True,
             cibil_score=720, credit_history_length_years=4, bureau_active_loans=1, bureau_total_monthly_emi=1500,
             bureau_overdue_amount=0, max_dpd_12m=0, enquiries_6m=1, written_off_or_settled=False,
             declared_formal_loans=1, declared_informal_loans=0, informal_debt_amount_declared=0, informal_emi_monthly=0,
             years_with_lender=2, prev_loans_avg_delay_days=2,
             declared_monthly_income=14000, effective_income=12500, income_variability="medium", other_income_amount=0,
             other_earner_income=0, monthly_expenses=7000, income_proof_type="bank_statement",
             electricity_available=True, elec_months_on_time=11, elec_months_overdue=0, recharge_regularity="regular",
             rent_paid_regularly=None, shg_member=True, shg_repayment_record=85, references_total=3, references_positive=3,
             owns_land=True, land_area_acres=2.0, irrigation_type="borewell",
             loan_amount=40000, tenure_months=12, interest_rate_pct=22, loan_purpose="crop_inputs")
    f.update(kw)
    return f


def show(title, r):
    print(f"\n--- {title}")
    print(f"decision={r['decision']}  risk={r['risk_score']:.1%} ({r['risk_band']})  tier={r['tier']}  confidence={r['data_confidence_score']}")
    print(f"requested={r['requested_amount']:,}  recommended={r['recommended_amount']}  emi={r['recommended_emi']}  capacity={r['capacity']}")
    if r["rules_triggered"]: print("rules:", r["rules_triggered"])
    if r["reasons_increasing_risk"]: print("risk up:", [(x['label'], x['value'], x['impact_pp']) for x in r['reasons_increasing_risk'][:3]])
    if r["conditions"]: print("conditions:", r["conditions"])
    if r["notes"]: print("notes:", r["notes"])


# 1. strong applicant
r1 = decision.evaluate(form(), UW, today=pd.Timestamp("2026-10-25").date()); show("1 solid tier-A farmer", r1)
assert r1["tier"] == "A" and r1["decision"] in ("eligible", "eligible_with_conditions")

# 2. daily wager, only a ration card, no bank/PAN/CIBIL -> must go to manual verification
r2 = decision.evaluate(form(occupation="daily_wager", has_pan=False, has_aadhaar=False, has_voter_id=False, has_ration_card=True,
        bank_account_exists=False, cibil_score=None, bureau_active_loans=None, bureau_total_monthly_emi=None, declared_formal_loans=0,
        years_with_lender=0, declared_monthly_income=9000, effective_income=None, income_proof_type="none", electricity_available=False,
        elec_months_on_time=None, shg_member=False, references_total=1, references_positive=1, owns_land=False,
        loan_amount=15000, monthly_expenses=5500), UW)
show("2 no-proof daily wager", r2)
assert r2["tier"] in ("C", "D") and r2["decision"] == "manual_verification"

# 3. same person after a good field verification -> scored, starter-loan cap applies
f3 = dict(r2=None)
r3 = decision.evaluate(form(occupation="daily_wager", has_pan=False, has_aadhaar=False, has_voter_id=False, has_ration_card=True,
        bank_account_exists=False, cibil_score=None, bureau_active_loans=None, bureau_total_monthly_emi=None, declared_formal_loans=0,
        years_with_lender=0, declared_monthly_income=9000, effective_income=None, income_proof_type="none", electricity_available=False,
        elec_months_on_time=None, shg_member=False, references_total=1, references_positive=1, owns_land=False,
        loan_amount=15000, monthly_expenses=5500, verification_score=78), UW)
show("3 same person, verified (score 78)", r3)
assert r3["decision"] in ("eligible", "eligible_with_conditions", "not_eligible") and r3["decision"] != "manual_verification"
if r3["recommended_amount"]: assert r3["recommended_amount"] <= decision.POLICY["tier_cap"][r3["tier"]]

# 4. failed verification
r4 = decision.evaluate(form(has_pan=False, cibil_score=None, bureau_active_loans=None, bureau_total_monthly_emi=None, declared_formal_loans=0,
        income_proof_type="none", electricity_available=False, elec_months_on_time=None, verification_score=35), UW)
show("4 failed field verification", r4); assert r4["decision"] == "not_eligible"

# 5. over-indebted, hides a loan
r5 = decision.evaluate(form(bureau_active_loans=4, bureau_total_monthly_emi=6500, declared_formal_loans=2, declared_informal_loans=1, informal_emi_monthly=800), UW)
show("5 over-indebted, hides 2 loans", r5); assert r5["decision"] == "not_eligible"
assert any("Hidden-loan" in n for n in r5["notes"])

# 6. asks for far too much -> amount is reduced to what the household can afford
r6 = decision.evaluate(form(loan_amount=140000), UW); show("6 asks for too much", r6)
assert r6["decision"] == "not_eligible" or (r6["recommended_amount"] or 0) < 140000

# 7. hard rule: written off
r7 = decision.evaluate(form(written_off_or_settled=True), UW); assert r7["decision"] == "not_eligible"

# 8. model sanity: higher EMI burden should not LOWER risk
lo = decision.evaluate(form(loan_amount=20000), UW)["risk_score"]; hi = decision.evaluate(form(loan_amount=90000), UW)["risk_score"]
print(f"\nrisk at Rs20k = {lo:.1%}, at Rs90k = {hi:.1%}"); assert hi >= lo

# 9. emi math round trip
e = decision.emi_for(50000, 22, 12); assert abs(decision.max_principal(e, 22, 12) - 50000) < 1

# 10. monitoring back-test on borrowers the model never saw (test split, month 3)
mon = pd.read_csv(ROOT / "data" / "monitoring_dataset.csv", usecols=["split", "months_on_book", "loan_id", "target_delinquent_next_3m"] + mc.MON_FEATURES)
snap = mon[(mon.split == "test") & (mon.months_on_book == 3)].head(1500).reset_index(drop=True)
res = mc.score_records(MON, snap.to_dict("records"))
alerts = [monitoring.build_alert(r, None) for r in res]
flag = np.array([a is not None for a in alerts]); y = snap.target_delinquent_next_3m.to_numpy()
print(f"\nmonitoring back-test (month 3, {len(snap)} loans): alerts on {flag.mean():.0%} of loans; "
      f"missed-payment rate among alerted {y[flag].mean():.1%} vs {y[~flag].mean():.1%} among not alerted (base {y.mean():.1%})")
assert y[flag].mean() > 1.5 * y[~flag].mean()
sample = next(a for a in alerts if a); print("sample alert:", sample)

# 11. every decision must be strictly JSON-serialisable (it is stored in a PostgreSQL JSON column)
import json
for name, r in [("r1", r1), ("r2", r2), ("r3", r3), ("r4", r4), ("r5", r5), ("r6", r6)]:
    json.dumps(r, allow_nan=False)
print("\nall decision results are valid JSON")

# 12. the exact data path used by the API: rows made by scripts/seed_db.py (JSON, nulls) -> scoring -> alerts
from scripts.seed_db import prepare  # noqa: E402
loan_rows, snap_rows = prepare(ROOT / "data", 400)
m3 = [s for s in snap_rows if s["months_on_book"] == 3]
rs = mc.score_records(MON, [s["features"] for s in m3])
al = [monitoring.build_alert(r, None) for r in rs]
yy = np.array([s["actual_outcome"] for s in m3]); ff = np.array([a is not None for a in al])
print(f"seed-path check: {len(m3)} loans scored, {ff.sum()} alerts, miss rate alerted {yy[ff].mean():.0%} vs rest {yy[~ff].mean():.0%}")
assert len(rs) == len(m3) and all(0 <= r["risk"] <= 1 for r in rs)

print("\nALL TESTS PASSED")
