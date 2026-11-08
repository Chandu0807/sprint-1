# Microfinance AI — Backend Sprint 1

This folder is a backend-only runnable copy of the uploaded `microfinance_ai 2` project. It retains its FastAPI service, application and monitoring logic, database models, authentication, trained model artifacts, training script, and synthetic data used for the demo. The Streamlit frontend and the archived virtual environment are excluded. No private `.env` file was copied; configure your own from `backend/.env.example`.

## What is in the project

| Path | Purpose |
|---|---|
| `backend/app/main.py` | FastAPI entry point, startup setup, and all HTTP routes |
| `backend/app/decision.py` | Application feature building, policy rules, affordability, and eligibility decision |
| `backend/app/ml_core.py` | Shared feature encoding, model training helpers, scoring, risk bands, and explanations |
| `backend/app/monitoring.py` | Monitoring alert trigger and recommended action mapping |
| `backend/app/models.py` | SQLAlchemy tables for users, applications, loans, monitoring, alerts, and verification |
| `backend/app/security.py` | Password hashing, JWT creation, role checks, and local demo users |
| `backend/app/schemas.py` | Request validation models |
| `backend/app/config.py`, `db.py` | Environment configuration and database engine/session |
| `backend/scripts/seed_db.py` | Loads historical synthetic test-split loans and monthly monitoring rows |
| `backend/tests/test_core.py` | Existing backend logic checks, included from the source project |
| `ml/artifacts/` | Packaged underwriting and monitoring model files used at startup |
| `ml/train.py` | Optional retraining script; saves model artifacts under `ml/artifacts/` |
| `data/` | Synthetic underwriting and monitoring CSV datasets |

## Architecture and data flow

```text
HTTP request
    -> FastAPI routes (`backend/app/main.py`)
    -> Pydantic validation + JWT role check (protected routes)
    -> decision / ml_core / monitoring logic
    -> SQLAlchemy session -> SQLite or PostgreSQL

Model training: CSV files -> shared feature code -> LightGBM (or scikit-learn fallback)
                -> ml/artifacts/*_model.joblib -> loaded when the API starts

Monitoring demo: monitoring CSV test split -> seed_db.py -> monthly snapshots in DB
                 -> POST /monitoring/run -> risk snapshots + alerts + historical back-test
```

At startup the service creates missing tables with SQLAlchemy `create_all`, seeds three demo accounts if the users table is empty, and loads both model artifacts. A real migration system is not included. Monitoring is run by an API call in this project; no scheduler or live collection-data integration is configured.

## Backend API

`GET /health` is public. `POST /auth/login` returns a bearer token. The remaining routes require `Authorization: Bearer <token>` and enforce roles defined in `backend/app/security.py`.

| Method and path | Purpose |
|---|---|
| `POST /applications` | Score and save a loan application; thin-file cases create a manual verification case |
| `GET /applications`, `GET /applications/{app_id}` | List or retrieve saved decisions |
| `GET /portfolio/summary` | Counts, risk bands, decisions, and model metrics |
| `POST /monitoring/run?month=1` | Score the seeded month (valid month range: 1–9), save risk snapshots, and raise alerts |
| `GET /alerts`, `PATCH /alerts/{alert_id}` | List and update alert status |
| `GET /loans/{loan_id}/risk-history` | Retrieve a loan’s scored history |
| `GET /verifications`, `GET /verifications/{case_id}` | List or retrieve manual verification cases |
| `POST /verifications/{case_id}/submit` | Submit the seven weighted field checks |
| `POST /verifications/{case_id}/decision` | Supervisor approves/rejects; approval re-scores the application |

FastAPI also serves its generated API explorer at `http://localhost:8000/docs` and OpenAPI JSON at `/openapi.json`. These are framework-provided developer tools; no project frontend is included.

## Requirements

- Python 3.10 or newer.
- `backend/requirements.txt` installs the API runtime: FastAPI, Uvicorn, SQLAlchemy, PostgreSQL driver, PyJWT, bcrypt, dotenv, Pydantic, joblib, LightGBM, pandas, NumPy, and scikit-learn.
- SQLite is the quickest local option and requires no separate database service. PostgreSQL is supported when a database and credentials are configured.
- The requirement versions are lower bounds (`>=`), not a locked/reproducible dependency file.

## Run locally on macOS or Linux

Run these commands from this project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
cp backend/.env.example backend/.env
```

The example config uses SQLite and stores the database at `backend/microfinance.db` when you start the API from `backend/`. Replace the sample `SECRET_KEY` in `backend/.env` with a private random value before use beyond a local demo. For example, generate one with `openssl rand -hex 32` and paste the result into that file.

Start the API in terminal 1:

```bash
source .venv/bin/activate
cd backend
uvicorn app.main:app --reload
```

The first start creates tables and demo users. Check the service and model files from another terminal:

```bash
curl -sS http://localhost:8000/health
```

Expected shape: `{"ok":true,"underwriting_model":true,"monitoring_model":true}`. If either model is false, see “Retrain the models” below, then restart the API.

### Sign in and submit one application

The source project creates these local demo accounts on the first start:

| Username | Password | Role |
|---|---|---|
| `officer` | `officer123` | `loan_officer` |
| `verifier` | `verifier123` | `field_verifier` |
| `supervisor` | `supervisor123` | `supervisor` |

Get a token:

```bash
curl -sS -X POST http://localhost:8000/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"username":"officer","password":"officer123"}'
```

Copy the returned `access_token` and set it in the terminal:

```bash
export TOKEN='paste-access-token-here'
```

Submit a sample application. Optional fields omitted from this body use the defaults in `backend/app/schemas.py`.

```bash
curl -sS -X POST http://localhost:8000/applications \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "full_name":"Sprint One Demo",
    "age":38,
    "occupation":"farmer",
    "family_size":4,
    "dependents":2,
    "has_pan":true,
    "has_aadhaar":true,
    "bank_account_exists":true,
    "cibil_score":720,
    "credit_history_length_years":4,
    "bureau_active_loans":1,
    "bureau_total_monthly_emi":1500,
    "declared_formal_loans":1,
    "declared_monthly_income":14000,
    "effective_income":12500,
    "monthly_expenses":7000,
    "income_proof_type":"bank_statement",
    "loan_amount":20000,
    "tenure_months":12
  }'
```

The response contains the saved application ID, decision, risk score and band, reasons, and affordability calculation. The exact decision depends on the packaged model and the supplied values. Retrieve saved applications with:

```bash
curl -sS http://localhost:8000/applications \
  -H "Authorization: Bearer $TOKEN"
```

### Demonstrate monitoring and alerts

In terminal 2, from the project folder:

```bash
source .venv/bin/activate
cd backend
python scripts/seed_db.py --data ../data --n 1500
```

This replaces existing demo alert/risk/loan seed rows and loads up to 1,500 loans from the synthetic dataset’s `test` split with their available monthly snapshots. Then trigger month 1 from a terminal holding `TOKEN`:

```bash
curl -sS -X POST 'http://localhost:8000/monitoring/run?month=1' \
  -H "Authorization: Bearer $TOKEN"
```

The response reports loans scored, alerts raised, and a back-test when both alerted and non-alerted rows are present. Inspect alerts or the portfolio summary:

```bash
curl -sS 'http://localhost:8000/alerts?status=open' -H "Authorization: Bearer $TOKEN"
curl -sS http://localhost:8000/portfolio/summary -H "Authorization: Bearer $TOKEN"
```

## PostgreSQL option

The source project’s default database is PostgreSQL. To use it, create a PostgreSQL database/user, then replace `DATABASE_URL` in `backend/.env` with the actual connection URL, such as `postgresql+psycopg2://USER:PASSWORD@localhost:5432/microfinance`. Do not use the placeholder credentials as real credentials. Start Uvicorn from `backend/` as above; the tables are created automatically on first start.

## Retrain the models (optional)

Pretrained model files are included, so training is not needed for the standard run. To retrain on the included synthetic CSV files:

```bash
source .venv/bin/activate
cd ml
python train.py --data ../data
cd ..
```

The script trains underwriting and monitoring models with the shared feature definitions in `backend/app/ml_core.py`, chooses risk-band thresholds from the validation split, reports test metrics, and writes updated `.joblib` artifacts into `ml/artifacts/`. It uses LightGBM when available and otherwise falls back to scikit-learn’s histogram gradient boosting implementation.

## What the project does and assumptions

- The underwriting path converts the submitted form into model features, applies fixed eligibility and affordability rules, calculates an ML risk score, and saves the form/result in the database. C/D tier thin-file applicants without a verification score are routed to manual verification.
- A verifier submits seven scored checks. A supervisor decision can re-run the same eligibility logic with the verification score.
- Monitoring scores monthly feature snapshots and stores risk/alert rows. The included monitoring CSV contains synthetic historical outcomes for a demonstration back-test; it is not a live data feed.
- The datasets are explicitly marked synthetic. Bureau fields (including CIBIL-related fields) are supplied in the application request; no bureau, utility, recharge, UPI, or payment-provider integration is present.
- The original archive contained a private `backend/.env`; it is not included here. Use the sanitized `.env.example` to create your own local configuration.
- The original archive also included a machine-specific `.venv`; it is omitted. Create a fresh environment with the commands above.
- Demo passwords are hard-coded in `backend/app/security.py` and are suitable only for a local demonstration. Change/remove them and configure a strong secret before any real deployment.
