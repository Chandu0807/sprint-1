# Sprint 1 Review Preparation — Microfinance AI Backend

Use this as a study guide, not a script to memorize word-for-word. Explain what you understand, and distinguish what the code already provides from what is still a demo or future improvement.

## 1. The 30-second explanation

> “For Sprint 1, I prepared the backend so the loan application and monitoring logic can be demonstrated without the Streamlit interface. FastAPI exposes the API, Pydantic validates requests, the decision module applies the project’s policy and affordability rules, and the saved ML model estimates risk. SQLAlchemy stores applications, decisions, monitoring results, alerts, and verification cases. I can run the API locally with SQLite and demonstrate it with HTTP requests. The included data is synthetic, so this is a prototype and not connected to live lender or bureau systems.”

The uploaded archive already contained these backend source files. This Sprint 1 package keeps that implementation, omits the frontend, bundled virtual environment, and private `.env`, sets the example configuration to SQLite, and adds a backend run/review guide. Describe your contribution accurately: say you built the code only if you were the person who wrote it; otherwise say you selected, configured, integrated, or prepared the backend for the Sprint 1 demo.

## 2. Architecture and request flow

```text
Client (curl / another app)
  -> FastAPI route in backend/app/main.py
  -> Pydantic request validation in schemas.py
  -> JWT authentication and role check in security.py (protected routes)
  -> decision.py or monitoring.py + shared ML functions in ml_core.py
  -> SQLAlchemy models/session -> SQLite or PostgreSQL
  -> JSON response
```

### Application request

1. `POST /applications` receives a borrower form.
2. Pydantic validates field types, allowed categories, and numeric ranges.
3. The token identifies the user; only a loan officer or supervisor can create an application.
4. `decision.build_features()` derives model inputs and business values, including household income, surplus, data confidence, and tier.
5. `decision.evaluate()` calculates the risk score, checks policy rules and repayment capacity, and selects a decision. A C/D thin-file applicant without a verification score is sent to manual verification.
6. The route stores the original form and the result JSON in the `applications` table. For manual verification it also creates a `verification_cases` row.
7. The route returns the saved application, including its ID, risk band, reasons, and decision.

### Monitoring request

1. `backend/scripts/seed_db.py` selects demo loans from the monitoring CSV’s `test` split and loads loan records plus monthly feature snapshots.
2. `POST /monitoring/run?month=N` loads the snapshots for that month and scores them with the monitoring artifact.
3. The API stores `risk_snapshots` and creates alerts when the risk band or month-to-month increase meets the alert rule.
4. Because this is historical synthetic data, the endpoint can compare alerts with the stored future outcome and return a demo back-test.

Monitoring is triggered by an API call. There is no scheduler or live data ingestion service in this version.

## 3. File-by-file map

### API and business logic: `backend/app/`

| File | What it does | How it fits |
|---|---|---|
| `__init__.py` | Marks `app` as a Python package. | Allows imports such as `app.main`. |
| `main.py` | Creates the FastAPI app, startup lifecycle, routes, and database operations. | API entry point: run `uvicorn app.main:app --reload` from `backend/`. |
| `schemas.py` | Defines input models and validation limits with Pydantic. | Rejects invalid request shapes before business logic runs. |
| `config.py` | Loads environment settings such as database URL, secret, token duration, CORS origins, and model path. | Keeps local configuration outside the Python logic. |
| `db.py` | Creates the SQLAlchemy engine and per-request database session. | Gives API routes a consistent connection to SQLite or PostgreSQL. |
| `models.py` | Declares the SQLAlchemy tables and columns. | Defines what is stored in the database. |
| `security.py` | Hashes passwords with bcrypt, issues/verifies JWTs, checks roles, and seeds demo users. | Protects routes and limits actions by role. |
| `decision.py` | Derives application features, applies policy rules, computes affordability, and forms the eligibility result. | Core loan underwriting/business logic. |
| `ml_core.py` | Lists model features, encodes values, constructs model objects, scores records, and prepares explanations. | Shared by model training and the API so feature handling matches. |
| `monitoring.py` | Maps risk drivers to alert types/actions and determines whether to raise an alert. | Core early-warning logic used by `/monitoring/run`. |

### Other backend files

| File or folder | What it does |
|---|---|
| `backend/requirements.txt` | Installs the API dependencies: FastAPI, Uvicorn, SQLAlchemy, PostgreSQL driver, JWT, bcrypt, dotenv, Pydantic, joblib, LightGBM, pandas, NumPy, and scikit-learn. |
| `backend/.env.example` | Sanitized local configuration template. It uses SQLite by default; PostgreSQL is an option. Copy it to `backend/.env`. |
| `backend/scripts/seed_db.py` | Loads demo loans and monthly monitoring snapshots from the synthetic CSV. It clears the previous demo loan, snapshot, risk snapshot, and alert data before seeding. |
| `backend/tests/test_core.py` | Existing tests for decision, scoring, monitoring, JSON output, and the seed-data path. It was included, not run during this preparation. |
| `ml/train.py` | Trains underwriting and monitoring classifiers, evaluates them, and writes model artifacts. |
| `ml/requirements.txt` | Lists the training/model dependencies. The backend requirements already include these in the bundled setup. |
| `ml/artifacts/underwriting_model.joblib` | Saved model used for new applications. |
| `ml/artifacts/monitoring_model.joblib` | Saved model used for monthly monitoring. |
| `data/underwriting_dataset.csv` | Synthetic records used to train/evaluate the underwriting model. |
| `data/monitoring_dataset.csv` | Synthetic loan-month records used to train/evaluate the monitoring model and seed the demo. |
| `data/README.txt` | Identifies the datasets as synthetic and describes their purpose. |

The API creates missing tables at startup with SQLAlchemy `create_all`; there is no migration framework in this project. It seeds demo users only when the users table is empty.

## 4. Data preparation: what the code actually does

This is a key review question. Do not claim there is a full production data-cleaning pipeline. The code performs feature selection, conversion, encoding, and some business-level derivation. It does not implement a separate general-purpose cleaning stage.

### During model training

- `ml/train.py` reads only the `split` field, target label, and the feature columns listed in `ml_core.py`; other CSV columns are ignored for that model.
- It trusts the CSV’s existing `train`, `valid`, and `test` split labels. The project README describes these as time-based splits; the training script does not create or re-check the split itself.
- The target label is kept separately from the feature matrix. Underwriting uses `target_default_12m`; monitoring uses `target_delinquent_next_3m`.
- The training script does not explicitly deduplicate records, clip outliers, normalize inconsistent category spellings, or impute the main tree model’s missing values.

### In shared feature conversion (`ml_core.to_frame`)

- Ensures each required feature column exists; a missing column is added as `NaN`.
- Selects columns in the exact model feature order.
- Maps the known categories (`occupation`, `irrigation_type`, `tier`) to fixed numeric codes. An unknown category becomes `NaN`.
- Converts object-valued features to numeric where possible and then to floating-point values; non-numeric strings become missing values.
- Keeps missing values for the primary LightGBM or scikit-learn histogram tree model, both of which support missing values. The separate logistic baseline uses a median imputer and standard scaler.

### During application feature building (`decision.build_features`)

- If verified/effective income is absent, uses 85% of declared income and adds an explanatory note.
- Uses bureau loan/EMI values only when a credit score exists; compares bureau loans with declared loans to derive a possible `loan_gap`.
- Estimates expenses at 65% of household income when expenses are omitted.
- Derives household income, surplus, EMI affordability, confidence score, tier, and model ratios from submitted values.
- Stores gender in the application form, but gender is not in the underwriting model feature list.

**Good concise answer:** “The project’s preparation is feature-focused rather than a comprehensive cleaning pipeline. It selects the model columns, converts categories and numeric fields consistently, preserves missing values for the tree model, and derives business features from the application form. I would add validation for data quality, duplicate handling, outlier review, and monitoring of missing/unknown categories before using real data.”

## 5. How the decisions work

### Confidence and tiers

`decision.build_features()` calculates a confidence score from available evidence. The source weights are: PAN 15, Aadhaar 10, voter ID 5, ration card 5, bank account 8, income proof 15, credit file 20, electricity data 8, lender history 8, SHG membership 6, and at least two references 5 points; the total is capped at 100.

- Tier A: PAN and a credit file are present.
- Otherwise Tier B: confidence is at least 58.
- Otherwise Tier C: confidence is at least 42.
- Otherwise Tier D.

### Rules and affordability

The source policy requires age 18–60, declines a previous write-off/settlement, declines when the counted active lenders are at least four, and declines a credit file with 90+ days past due in the last 12 months. It also checks repayment capacity.

Safe new EMI is calculated from the smaller of:

- 45% of household income minus existing EMIs; and
- 70% of monthly surplus.

The permitted principal is estimated from the safe EMI, interest rate, and loan tenure. Other source policy values include a minimum loan of Rs 5,000, a Tier C cap of Rs 30,000, a Tier D cap of Rs 15,000, and an 80% amount factor for medium-risk cases. These are prototype policy settings in `decision.py`, not universal lending rules.

High risk or a failed hard rule can result in `not_eligible`. A C/D thin-file applicant without a verification score is routed to `manual_verification`. After field checks, a supervisor can approve and re-score the application using the verification score.

### Risk model and explanations

- The API loads packaged `.joblib` artifacts when it starts. You normally do not need to train models to run the demo.
- `ml/train.py` fits on the training split, chooses band thresholds from validation predictions, and reports metrics on the test split.
- Underwriting thresholds use the 60th and 88th validation percentiles for medium/high bands. Monitoring uses the 80th and 93rd percentiles.
- The primary model is LightGBM when available; the code has a scikit-learn histogram gradient-boosting fallback. A logistic-regression baseline is also trained for comparison.
- Explanations use feature occlusion: replace one value with a typical median/mode and measure the score change. This is a model sensitivity explanation, not proof that the feature caused the outcome, and it is not SHAP.
- The original project README reports approximate AUC values of 0.77 for underwriting and 0.74 for monitoring. Treat these as project-reported results unless you rerun training and confirm them on the included files.

## 6. Database, authentication, and endpoints

### Database tables

- `users`: username, password hash, and role.
- `applications`: applicant form, decision result, tier, confidence, risk, and creator.
- `verification_cases` and `verification_checks`: manual review workflow and its seven check scores.
- `loans` and `monitoring_snapshots`: demo loans and monthly input features.
- `risk_snapshots` and `alerts`: monthly scores, reasons, alert type, status, and recommended action.

Application input/result and monitoring features/reasons are stored in JSON columns. SQLite is easiest for the demo. PostgreSQL is supported by changing `DATABASE_URL`. The service creates tables, but does not migrate existing schemas.

### Authentication

`POST /auth/login` verifies the bcrypt password hash and returns a JWT signed with HS256. Protected calls send it as `Authorization: Bearer <token>`. The demo token duration defaults to eight hours. Roles are `loan_officer`, `field_verifier`, and `supervisor`. Demo passwords are hard-coded in source and must not be used for a real deployment; the sample secret must also be replaced.

### Main routes

- Public: `GET /health`, `POST /auth/login`.
- Applications: `POST /applications`, `GET /applications`, `GET /applications/{id}`.
- Portfolio: `GET /portfolio/summary`.
- Monitoring: `POST /monitoring/run?month=1` (months 1–9).
- Alerts: `GET /alerts`, `PATCH /alerts/{id}`.
- Loans: `GET /loans/{loan_id}/risk-history`.
- Verification: `GET /verifications`, `GET /verifications/{id}`, `POST /verifications/{id}/submit`, `POST /verifications/{id}/decision`.

Role restrictions are enforced in `main.py` through `security.require_roles`. The API does not provide a frontend in this bundle. FastAPI’s generated `/docs` page is an API exploration tool.

## 7. Run and demo checklist (macOS/Linux)

Run from the extracted project root. If it is in Downloads, that may be `cd ~/Downloads/microfinance_ai_backend_sprint1`; adjust if your folder is elsewhere.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r backend/requirements.txt
cp backend/.env.example backend/.env
```

The example config selects a local SQLite file. Keep the API running in terminal 1:

```bash
cd backend
uvicorn app.main:app --reload
```

In terminal 2, run `curl -sS http://localhost:8000/health`. Expected shape: `ok: true` and both model flags `true`. The login returns JSON. Capture only the token in the same shell where you will make the protected calls; do not paste the full JSON response into `export`.

```bash
cd ~/Downloads/microfinance_ai_backend_sprint1/backend
source ../.venv/bin/activate
TOKEN="$(curl -sS -X POST http://localhost:8000/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"username":"officer","password":"officer123"}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')"
echo ${#TOKEN}
```

The token-length check prints a number, not the token. Then demonstrate a saved application:

```bash
curl -sS -X POST http://localhost:8000/applications \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"full_name":"Sprint One Demo","age":38,"occupation":"farmer","family_size":4,"declared_monthly_income":14000,"loan_amount":20000}'

curl -sS http://localhost:8000/applications -H "Authorization: Bearer $TOKEN"
curl -sS http://localhost:8000/verifications -H "Authorization: Bearer $TOKEN"
```

The minimal sample omits documents and credit history, so the current decision logic should create a manual verification case. For monitoring, run the following from a second shell in `backend/`, then call the endpoint from the shell that has `TOKEN`:

```bash
python scripts/seed_db.py --data ../data --n 1500
curl -sS -X POST 'http://localhost:8000/monitoring/run?month=1' \
  -H "Authorization: Bearer $TOKEN"
```

The seed script replaces the demo loans, monitoring snapshots, risk snapshots, and alerts. Use it with the demo database, not a database containing data you need to keep.

Optional included test command (not executed as part of this preparation):

```bash
cd backend
python tests/test_core.py
```

## 8. Tough review questions and accurate answers

**Q: What exactly did you implement for Sprint 1?**  
A: “I prepared the backend workflow for a standalone demo: API routes, request validation, decision and monitoring logic, database persistence, authentication, and model/data assets. The UI is excluded. The source package already contained the backend logic; my Sprint 1 deliverable is the backend-only configuration and demo setup.” Adjust the ownership wording to match your actual contribution.

**Q: What data cleaning did you do?**  
A: “There is no broad cleaning pipeline. The code selects the model columns, converts known categories to fixed numeric values, converts numeric strings, and leaves missing or unknown values as NaN for the tree model. Application features are also derived in the decision module. I would add explicit duplicate, range, outlier, and category-quality checks before real deployment.”

**Q: Is this real borrower data?**  
A: “No. The included CSVs are synthetic and used for a demo. The API accepts typed application fields but has no live CIBIL, payment, utility, recharge, or UPI integration.”

**Q: Why FastAPI?**  
A: “It provides HTTP/JSON routes, request validation, and generated OpenAPI documentation. The same backend can be called by a future web or mobile client without putting the decision rules in that client.”

**Q: Why SQLAlchemy?**  
A: “It maps Python model classes to database tables and manages sessions. The project can use SQLite for a local demo or PostgreSQL with a configured connection URL.”

**Q: Is the database schema migrated?**  
A: “No migration tool is included. On startup SQLAlchemy `create_all` creates missing tables, but it does not provide a versioned migration plan for changing production schemas.”

**Q: How do you avoid train/test leakage?**  
A: “The CSV already has train, valid, and test labels. The script fits on train, selects risk thresholds on valid, and reports final metrics on test. It trusts the supplied split; it does not generate or independently audit that split.”

**Q: How do you handle missing values?**  
A: “At inference/training feature conversion, missing columns and non-numeric values become NaN. The main tree models support missing values. The logistic baseline uses median imputation and scaling.”

**Q: How do you explain a score?**  
A: “The code replaces each feature with a typical median or category mode and observes how the model score changes. It gives an approximate model sensitivity explanation, not a causal explanation.”

**Q: What if a borrower has no formal credit file?**  
A: “The request can omit the credit score. Confidence and tier are calculated from other evidence. A C or D tier applicant without a verification score is sent to field verification rather than automatically being rejected just for having a thin file.”

**Q: How does the affordability check work?**  
A: “It estimates a safe EMI using household income, existing EMI, and monthly surplus. It then estimates the supported principal for the requested rate and tenure and applies the project’s loan limits.”

**Q: How are monitoring alerts triggered?**  
A: “The source logic alerts for a high risk band, a risk increase of at least 12 percentage points, or a medium-band increase of at least 5 points. The main reason feature is mapped to a suggested follow-up action.”

**Q: Does the back-test prove the model works in production?**  
A: “No. It compares alerts with outcomes in the synthetic historical test-split data. It is a demo back-test, not evidence of performance on live borrowers or proof of causal impact.”

**Q: How is access protected?**  
A: “Passwords are bcrypt-hashed, login returns a signed JWT, and protected routes check roles. The demo accounts and example key are not production-ready; production needs unique credentials, secret management, HTTPS, and further security controls.”

**Q: What happens if the model file is missing?**  
A: “Health reports the model as unavailable, and routes requiring it return a service-unavailable response. The packaged artifacts should allow the normal demo to start without training; otherwise retrain from the included data and restart.”

**Q: What are the biggest next steps?**  
A: “Use consented real data and integrations, add explicit data-quality and fairness checks, add database migrations and operational monitoring, replace demo authentication/secrets, and schedule monitoring rather than triggering it manually.”

## 9. What your current terminal result proves

Your screenshot shows a successful login, a captured token, and a successful protected request to `GET /applications`. The returned `[]` means there were no saved applications at that moment. It proves authentication and listing worked; submit a `POST /applications` request to demonstrate scoring and persistence too.

## 10. Final reminders

- Know the end-to-end request flow rather than memorizing every source line.
- Be direct that the data is synthetic and that the backend has no live external integrations.
- Do not describe the back-test or AUC as independent proof of production performance.
- Say the tests are included; only say they passed if you run them and see the passing output yourself.
- If asked something the prototype does not implement, state the boundary and describe it as a future improvement.
