"""schemas.py - what the API accepts. FastAPI validates every request against these (wrong types -> clear error)."""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

Occupation = Literal["farmer", "daily_wager", "vendor", "small_shopkeeper", "artisan",
                     "domestic_worker", "driver", "salaried", "dairy_livestock"]
CHECK_TYPES = ["identity", "address", "occupation", "references", "group_attestation", "household_assessment", "informal_signals"]


class LoginIn(BaseModel):
    username: str
    password: str


class ApplicationIn(BaseModel):
    # applicant
    full_name: str = Field(min_length=2, max_length=120)
    age: int = Field(ge=10, le=100)
    gender: Optional[Literal["F", "M", "O"]] = None     # stored for fairness audits only; the model never sees it
    occupation: Occupation
    family_size: int = Field(ge=1, le=20)
    dependents: int = Field(0, ge=0, le=20)
    # documents
    has_pan: bool = False
    has_aadhaar: bool = False
    has_voter_id: bool = False
    has_ration_card: bool = False
    bank_account_exists: bool = False
    # credit file (leave empty if the person has no CIBIL record)
    cibil_score: Optional[int] = Field(None, ge=300, le=900)
    credit_history_length_years: Optional[float] = Field(None, ge=0)
    bureau_active_loans: Optional[int] = Field(None, ge=0)
    bureau_total_monthly_emi: Optional[float] = Field(None, ge=0)
    bureau_overdue_amount: Optional[float] = Field(None, ge=0)
    max_dpd_12m: Optional[int] = Field(None, ge=0)
    enquiries_6m: Optional[int] = Field(None, ge=0)
    written_off_or_settled: bool = False
    # what the borrower says about debts
    declared_formal_loans: int = Field(0, ge=0)
    declared_informal_loans: int = Field(0, ge=0)
    informal_debt_amount_declared: float = Field(0, ge=0)
    informal_emi_monthly: float = Field(0, ge=0)
    years_with_lender: float = Field(0, ge=0)
    prev_loans_avg_delay_days: Optional[float] = Field(None, ge=0)
    # income
    declared_monthly_income: float = Field(gt=0)
    effective_income: Optional[float] = Field(None, gt=0)        # officer's verified/estimated income
    income_variability: Optional[Literal["low", "medium", "high"]] = None
    other_income_amount: float = Field(0, ge=0)
    other_earner_income: float = Field(0, ge=0)
    monthly_expenses: Optional[float] = Field(None, ge=0)
    income_proof_type: Literal["payslip", "bank_statement", "none"] = "none"
    # alternative data
    electricity_available: bool = False
    elec_months_on_time: Optional[int] = Field(None, ge=0, le=12)
    elec_months_overdue: Optional[int] = Field(None, ge=0, le=12)
    recharge_regularity: Optional[Literal["regular", "irregular"]] = None
    rent_paid_regularly: Optional[bool] = None
    # social + assets
    shg_member: bool = False
    shg_repayment_record: Optional[float] = Field(None, ge=0, le=100)
    references_total: int = Field(0, ge=0, le=10)
    references_positive: int = Field(0, ge=0, le=10)
    owns_land: bool = False
    land_area_acres: Optional[float] = Field(None, ge=0)
    irrigation_type: Optional[Literal["rainfed", "borewell", "canal"]] = None
    # the loan
    loan_amount: float = Field(gt=0)
    tenure_months: Literal[12, 18, 24] = 12
    interest_rate_pct: float = Field(22.0, ge=5, le=40)
    loan_purpose: str = "working_capital"


class CheckItem(BaseModel):
    check_type: Literal["identity", "address", "occupation", "references", "group_attestation",
                        "household_assessment", "informal_signals"]
    score: float = Field(ge=0, le=10)
    notes: str = ""


class VerificationSubmit(BaseModel):
    checks: List[CheckItem]


class VerificationDecision(BaseModel):
    approve: bool
    note: str = ""


class AlertPatch(BaseModel):
    status: Literal["open", "acknowledged", "resolved"]
