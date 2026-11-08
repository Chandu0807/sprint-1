"""
main.py - the FastAPI application. Every endpoint is a thin wrapper around the tested logic in
decision.py (eligibility), monitoring.py (alerts) and ml_core.py (scoring).

Run from the backend folder:   uvicorn app.main:app --reload
Interactive API docs:          http://localhost:8000/docs
"""
from contextlib import asynccontextmanager
from typing import Optional

import joblib
import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from . import config, decision, ml_core, models, monitoring, schemas, security
from .db import SessionLocal, engine, get_db

CHECK_WEIGHTS = {"identity": 0.20, "address": 0.15, "occupation": 0.20, "references": 0.15,
                 "group_attestation": 0.10, "household_assessment": 0.10, "informal_signals": 0.10}
ANY_ROLE = ("loan_officer", "field_verifier", "supervisor")


def load_artifact(name):
    path = config.ARTIFACT_DIR / f"{name}_model.joblib"
    if not path.exists():
        print(f"WARNING: {path} not found. Run ml/train.py first.")
        return None
    return joblib.load(path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    models.Base.metadata.create_all(engine)          # creates the tables on first start
    with SessionLocal() as db:
        security.seed_users(db)
    app.state.uw = load_artifact("underwriting")
    app.state.mon = load_artifact("monitoring")
    yield


app = FastAPI(title="Borrower Intelligence Platform", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])


def artifact(request: Request, key: str):
    art = getattr(request.app.state, key)
    if art is None:
        raise HTTPException(status_code=503, detail="Model not trained yet. Run ml/train.py, then restart the API.")
    return art


# ----------------------------------------------------------------------------- auth
@app.get("/health")
def health(request: Request):
    return {"ok": True, "underwriting_model": request.app.state.uw is not None, "monitoring_model": request.app.state.mon is not None}


@app.post("/auth/login")
def login(body: schemas.LoginIn, db: Session = Depends(get_db)):
    user = db.scalar(select(models.User).where(models.User.username == body.username))
    if user is None or not security.verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Wrong username or password")
    return {"access_token": security.create_token(user.username, user.role), "role": user.role, "username": user.username}


# ----------------------------------------------------------------------------- applications
def application_out(a: models.Application):
    return {"id": a.id, "created_at": a.created_at.isoformat(), "applicant_name": a.applicant_name,
            "occupation": a.form.get("occupation"), "requested_amount": a.form.get("loan_amount"), "tier": a.tier,
            "decision": a.decision, "risk_score": a.risk_score, "risk_band": a.risk_band,
            "created_by": a.created_by, "result": a.result}


def case_reason(form):
    parts = []
    if not form.get("has_pan"):
        parts.append("no PAN")
    if form.get("cibil_score") is None:
        parts.append("no credit history")
    if form.get("income_proof_type") == "none":
        parts.append("no income proof")
    return "Thin-file borrower: " + ", ".join(parts or ["low data confidence"])


@app.post("/applications")
def create_application(body: schemas.ApplicationIn, request: Request, db: Session = Depends(get_db),
                       user=Depends(security.require_roles("loan_officer", "supervisor"))):
    form = body.model_dump()
    result = decision.evaluate(form, artifact(request, "uw"))
    row = models.Application(applicant_name=form["full_name"], form=form, result=result, tier=result["tier"],
                             confidence=result["data_confidence_score"], risk_score=result["risk_score"],
                             risk_band=result["risk_band"], decision=result["decision"], created_by=user["username"])
    db.add(row)
    db.flush()                                       # gives `row.id` before commit
    if result["decision"] == "manual_verification":
        db.add(models.VerificationCase(application_id=row.id, applicant_name=form["full_name"], reason=case_reason(form)))
    db.commit()
    return application_out(row)


@app.get("/applications")
def list_applications(limit: int = 50, db: Session = Depends(get_db), user=Depends(security.require_roles(*ANY_ROLE))):
    rows = db.scalars(select(models.Application).order_by(models.Application.id.desc()).limit(limit)).all()
    return [application_out(r) for r in rows]


@app.get("/applications/{app_id}")
def get_application(app_id: int, db: Session = Depends(get_db), user=Depends(security.require_roles(*ANY_ROLE))):
    row = db.get(models.Application, app_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Application not found")
    return application_out(row)


# ----------------------------------------------------------------------------- portfolio + monitoring
@app.get("/portfolio/summary")
def portfolio_summary(request: Request, db: Session = Depends(get_db), user=Depends(security.require_roles(*ANY_ROLE))):
    latest = db.scalar(select(func.max(models.RiskSnapshot.months_on_book)))
    bands = {}
    if latest:
        rows = db.execute(select(models.RiskSnapshot.band, func.count()).where(models.RiskSnapshot.months_on_book == latest)
                          .group_by(models.RiskSnapshot.band)).all()
        bands = {r[0]: r[1] for r in rows}
    decisions = {r[0]: r[1] for r in db.execute(select(models.Application.decision, func.count()).group_by(models.Application.decision)).all()}
    metrics = {}
    for key, short in (("uw", "underwriting"), ("mon", "monitoring")):
        art = getattr(request.app.state, key)
        if art:
            metrics[short] = {"auc_test": art["metrics"]["auc_test"], "model_type": art["metrics"]["model_type"],
                              "bands_on_test": art["metrics"]["bands_on_test"]}
    return {"total_loans": db.scalar(select(func.count()).select_from(models.Loan)), "latest_month_scored": latest,
            "risk_bands": bands, "open_alerts": db.scalar(select(func.count()).select_from(models.Alert).where(models.Alert.status == "open")),
            "application_decisions": decisions, "model_metrics": metrics}


@app.post("/monitoring/run")
def run_monitoring(request: Request, month: int = Query(..., ge=1, le=9), db: Session = Depends(get_db),
                   user=Depends(security.require_roles("supervisor", "loan_officer"))):
    """Re-score every active loan for `month` since disbursement, store the risk, and raise alerts.
    In production a scheduler (e.g. APScheduler) calls this logic on a timer; here a button triggers it."""
    art = artifact(request, "mon")
    snaps = db.scalars(select(models.MonitoringSnapshot).where(models.MonitoringSnapshot.months_on_book == month)).all()
    if not snaps:
        raise HTTPException(status_code=404, detail="No data for that month. Run scripts/seed_db.py first.")
    results = ml_core.score_records(art, [s.features for s in snaps])
    prev = {r.loan_id: r.risk_score for r in db.scalars(select(models.RiskSnapshot).where(models.RiskSnapshot.months_on_book == month - 1))}

    db.execute(delete(models.Alert).where(models.Alert.months_on_book == month))          # re-running a month replaces it
    db.execute(delete(models.RiskSnapshot).where(models.RiskSnapshot.months_on_book == month))
    flagged, outcomes = [], []
    for s, res in zip(snaps, results):
        db.add(models.RiskSnapshot(loan_id=s.loan_id, months_on_book=month, risk_score=res["risk"], band=res["band"], reasons=res["reasons"]))
        alert = monitoring.build_alert(res, prev.get(s.loan_id))
        if alert:
            db.add(models.Alert(loan_id=s.loan_id, months_on_book=month, risk_score=res["risk"], **alert))
        flagged.append(alert is not None)
        outcomes.append(s.actual_outcome)
    db.commit()

    flagged = np.array(flagged)
    known = np.array([o is not None for o in outcomes])
    y = np.array([o if o is not None else 0 for o in outcomes], dtype=float)
    backtest = None
    if known.any() and flagged.any() and (~flagged).any():
        backtest = {"alerted_miss_rate": round(float(y[flagged & known].mean()), 3),
                    "not_alerted_miss_rate": round(float(y[~flagged & known].mean()), 3),
                    "overall_miss_rate": round(float(y[known].mean()), 3)}
    return {"month": month, "loans_scored": len(snaps), "alerts_raised": int(flagged.sum()), "backtest": backtest}


@app.get("/alerts")
def list_alerts(status: str = "open", limit: int = 100, db: Session = Depends(get_db), user=Depends(security.require_roles(*ANY_ROLE))):
    q = (select(models.Alert, models.Loan).join(models.Loan, models.Loan.loan_id == models.Alert.loan_id)
         .where(models.Alert.status == status).order_by(models.Alert.risk_score.desc()).limit(limit))
    return [{"id": a.id, "loan_id": a.loan_id, "occupation": l.occupation, "tier": l.tier, "district": l.district,
             "loan_amount": l.loan_amount, "months_on_book": a.months_on_book, "risk_score": a.risk_score,
             "severity": a.severity, "alert_type": a.alert_type, "message": a.message,
             "recommended_action": a.recommended_action, "status": a.status} for a, l in db.execute(q).all()]


@app.patch("/alerts/{alert_id}")
def update_alert(alert_id: int, body: schemas.AlertPatch, db: Session = Depends(get_db),
                 user=Depends(security.require_roles("loan_officer", "supervisor"))):
    alert = db.get(models.Alert, alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail="Alert not found")
    alert.status = body.status
    db.commit()
    return {"id": alert.id, "status": alert.status}


@app.get("/loans/{loan_id}/risk-history")
def risk_history(loan_id: str, db: Session = Depends(get_db), user=Depends(security.require_roles(*ANY_ROLE))):
    loan = db.get(models.Loan, loan_id)
    if loan is None:
        raise HTTPException(status_code=404, detail="Loan not found")
    hist = db.scalars(select(models.RiskSnapshot).where(models.RiskSnapshot.loan_id == loan_id)
                      .order_by(models.RiskSnapshot.months_on_book)).all()
    return {"loan": {"loan_id": loan.loan_id, "occupation": loan.occupation, "tier": loan.tier, "district": loan.district,
                     "loan_amount": loan.loan_amount, "emi_amount": loan.emi_amount},
            "history": [{"months_on_book": h.months_on_book, "risk_score": round(h.risk_score, 4), "band": h.band,
                         "top_reasons": h.reasons[:3]} for h in hist]}


# ----------------------------------------------------------------------------- manual verification
def case_out(c: models.VerificationCase):
    return {"id": c.id, "application_id": c.application_id, "applicant_name": c.applicant_name, "reason": c.reason,
            "status": c.status, "verification_score": c.verification_score, "verdict": c.verdict,
            "submitted_by": c.submitted_by, "created_at": c.created_at.isoformat()}


@app.get("/verifications")
def list_cases(status: Optional[str] = None, db: Session = Depends(get_db), user=Depends(security.require_roles(*ANY_ROLE))):
    q = select(models.VerificationCase).order_by(models.VerificationCase.id.desc()).limit(200)
    if status:
        q = q.where(models.VerificationCase.status == status)
    return [case_out(c) for c in db.scalars(q).all()]


@app.get("/verifications/{case_id}")
def get_case(case_id: int, db: Session = Depends(get_db), user=Depends(security.require_roles(*ANY_ROLE))):
    case = db.get(models.VerificationCase, case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="Case not found")
    app_row = db.get(models.Application, case.application_id)
    checks = db.scalars(select(models.VerificationCheck).where(models.VerificationCheck.case_id == case_id)).all()
    form = app_row.form
    return {**case_out(case), "checks": [{"check_type": k.check_type, "score": k.score, "notes": k.notes} for k in checks],
            "applicant": {"occupation": form.get("occupation"), "age": form.get("age"), "declared_monthly_income": form.get("declared_monthly_income"),
                          "loan_amount": form.get("loan_amount"), "has_ration_card": form.get("has_ration_card"),
                          "has_aadhaar": form.get("has_aadhaar")},
            "application_decision": app_row.decision, "checklist_weights": CHECK_WEIGHTS}


@app.post("/verifications/{case_id}/submit")
def submit_checks(case_id: int, body: schemas.VerificationSubmit, db: Session = Depends(get_db),
                  user=Depends(security.require_roles("field_verifier", "supervisor"))):
    """Field officer submits the 7 checks (each scored 0-10). The weighted total becomes the 0-100 verification score."""
    case = db.get(models.VerificationCase, case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="Case not found")
    if {c.check_type for c in body.checks} != set(CHECK_WEIGHTS) or len(body.checks) != len(CHECK_WEIGHTS):
        raise HTTPException(status_code=400, detail=f"Submit exactly one score for each of: {', '.join(CHECK_WEIGHTS)}")
    db.execute(delete(models.VerificationCheck).where(models.VerificationCheck.case_id == case_id))
    for c in body.checks:
        db.add(models.VerificationCheck(case_id=case_id, check_type=c.check_type, score=c.score, notes=c.notes))
    score = round(sum(CHECK_WEIGHTS[c.check_type] * c.score for c in body.checks) * 10, 1)
    case.verification_score = score
    case.verdict = "verified" if score >= 70 else "partially_verified" if score >= 50 else "failed"
    case.status = "review"
    case.submitted_by = user["username"]
    db.commit()
    return case_out(case)


@app.post("/verifications/{case_id}/decision")
def decide_case(case_id: int, body: schemas.VerificationDecision, request: Request, db: Session = Depends(get_db),
                user=Depends(security.require_roles("supervisor"))):
    """Supervisor approves or rejects. Approval re-runs the full eligibility decision WITH the verification score."""
    case = db.get(models.VerificationCase, case_id)
    if case is None or case.verification_score is None:
        raise HTTPException(status_code=400, detail="The field officer has not submitted the checks yet")
    app_row = db.get(models.Application, case.application_id)
    if body.approve:
        form = dict(app_row.form)
        form["verification_score"] = case.verification_score
        result = decision.evaluate(form, artifact(request, "uw"))
        app_row.form = form
        case.status = "approved"
    else:
        result = dict(app_row.result)
        result["decision"] = "not_eligible"
        result["rules_triggered"] = list(result.get("rules_triggered", [])) + [f"Rejected by supervisor after field verification. {body.note}".strip()]
        case.status = "rejected"
    app_row.result, app_row.decision = result, result["decision"]
    app_row.risk_score, app_row.risk_band, app_row.tier = result["risk_score"], result["risk_band"], result["tier"]
    db.commit()
    return {"case": case_out(case), "application": application_out(app_row)}
