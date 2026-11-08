"""models.py - the PostgreSQL tables (SQLAlchemy 2.0 style)."""
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def now():
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(50), unique=True)
    password_hash: Mapped[str] = mapped_column(String(200))
    role: Mapped[str] = mapped_column(String(30))          # loan_officer | field_verifier | supervisor


class Application(Base):
    """One loan application and the system's decision. `form` = what the officer entered, `result` = full insight JSON."""
    __tablename__ = "applications"
    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    applicant_name: Mapped[str] = mapped_column(String(120))
    form: Mapped[dict] = mapped_column(JSON)
    result: Mapped[dict] = mapped_column(JSON)
    tier: Mapped[str] = mapped_column(String(2))
    confidence: Mapped[int] = mapped_column(Integer)
    risk_score: Mapped[float] = mapped_column(Float)
    risk_band: Mapped[str] = mapped_column(String(10))
    decision: Mapped[str] = mapped_column(String(30), index=True)
    created_by: Mapped[str] = mapped_column(String(50))


class Loan(Base):
    """Active loans being monitored (seeded from the synthetic dataset for the demo)."""
    __tablename__ = "loans"
    loan_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    occupation: Mapped[str] = mapped_column(String(30))
    tier: Mapped[str] = mapped_column(String(2))
    district: Mapped[str] = mapped_column(String(60))
    state: Mapped[str] = mapped_column(String(60))
    loan_amount: Mapped[float] = mapped_column(Float)
    emi_amount: Mapped[float] = mapped_column(Float)
    tenure_months: Mapped[int] = mapped_column(Integer)
    disbursement_date: Mapped[str] = mapped_column(String(10))


class MonitoringSnapshot(Base):
    """The monthly facts about a loan (payments, income, utilities...). In production these rows are written by your
    collection / income-tracking processes; `actual_outcome` exists only because the demo data is historical."""
    __tablename__ = "monitoring_snapshots"
    __table_args__ = (UniqueConstraint("loan_id", "months_on_book"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    loan_id: Mapped[str] = mapped_column(ForeignKey("loans.loan_id"), index=True)
    months_on_book: Mapped[int] = mapped_column(Integer, index=True)
    features: Mapped[dict] = mapped_column(JSON)
    actual_outcome: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)


class RiskSnapshot(Base):
    __tablename__ = "risk_snapshots"
    id: Mapped[int] = mapped_column(primary_key=True)
    loan_id: Mapped[str] = mapped_column(ForeignKey("loans.loan_id"), index=True)
    months_on_book: Mapped[int] = mapped_column(Integer, index=True)
    risk_score: Mapped[float] = mapped_column(Float)
    band: Mapped[str] = mapped_column(String(10))
    reasons: Mapped[list] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Alert(Base):
    __tablename__ = "alerts"
    id: Mapped[int] = mapped_column(primary_key=True)
    loan_id: Mapped[str] = mapped_column(ForeignKey("loans.loan_id"), index=True)
    months_on_book: Mapped[int] = mapped_column(Integer, index=True)
    risk_score: Mapped[float] = mapped_column(Float)
    severity: Mapped[str] = mapped_column(String(10))
    alert_type: Mapped[str] = mapped_column(String(30))
    message: Mapped[str] = mapped_column(Text)
    recommended_action: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), default="open", index=True)   # open | acknowledged | resolved
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class VerificationCase(Base):
    __tablename__ = "verification_cases"
    id: Mapped[int] = mapped_column(primary_key=True)
    application_id: Mapped[int] = mapped_column(ForeignKey("applications.id"), index=True)
    applicant_name: Mapped[str] = mapped_column(String(120))
    reason: Mapped[str] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(String(20), default="open", index=True)   # open | review | approved | rejected
    verification_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    verdict: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    submitted_by: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class VerificationCheck(Base):
    __tablename__ = "verification_checks"
    id: Mapped[int] = mapped_column(primary_key=True)
    case_id: Mapped[int] = mapped_column(ForeignKey("verification_cases.id"), index=True)
    check_type: Mapped[str] = mapped_column(String(30))
    score: Mapped[float] = mapped_column(Float)
    notes: Mapped[str] = mapped_column(Text, default="")
