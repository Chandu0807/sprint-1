"""
seed_db.py - loads demo loans + their monthly monitoring data into PostgreSQL.

    python scripts/seed_db.py --data ../data --n 1500

It uses borrowers from the TEST split (the most recent loans, which the model never trained on), so the
monitoring demo is an honest back-test: each snapshot also stores what really happened next.
In a real system these rows would come from your collection and income-tracking processes instead.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import ml_core  # noqa: E402

LOAN_COLS = ["loan_id", "occupation", "tier", "district", "state", "loan_amount", "emi_amount", "tenure_months", "disbursement_date"]


def prepare(data_dir, n):
    """Pure pandas step (no database): returns (loan_rows, snapshot_rows) as lists of dicts."""
    cols = LOAN_COLS + ["split", "months_on_book", "target_delinquent_next_3m"] + ml_core.MON_FEATURES
    mon = pd.read_csv(Path(data_dir) / "monitoring_dataset.csv", usecols=lambda c: c in set(cols))
    test = mon[mon["split"] == "test"]
    keep_ids = test["loan_id"].drop_duplicates().head(n)
    sub = test[test["loan_id"].isin(keep_ids)].copy()

    loans = sub.sort_values("months_on_book").groupby("loan_id", as_index=False).first()[LOAN_COLS]
    loan_rows = json.loads(loans.to_json(orient="records"))          # to_json turns NaN into null (valid JSON)

    feats = json.loads(sub[ml_core.MON_FEATURES].to_json(orient="records"))
    snap_rows = [{"loan_id": lid, "months_on_book": int(m), "features": f, "actual_outcome": int(y)}
                 for lid, m, f, y in zip(sub["loan_id"], sub["months_on_book"], feats, sub["target_delinquent_next_3m"])]
    return loan_rows, snap_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data")
    ap.add_argument("--n", type=int, default=1500, help="how many loans to load")
    a = ap.parse_args()

    from sqlalchemy import delete, insert
    from app import models
    from app.db import SessionLocal, engine

    loan_rows, snap_rows = prepare(a.data, a.n)
    models.Base.metadata.create_all(engine)
    with SessionLocal() as db:
        for table in (models.Alert, models.RiskSnapshot, models.MonitoringSnapshot, models.Loan):   # clear old demo data
            db.execute(delete(table))
        db.execute(insert(models.Loan), loan_rows)
        for i in range(0, len(snap_rows), 5000):                       # insert in chunks
            db.execute(insert(models.MonitoringSnapshot), snap_rows[i:i + 5000])
        db.commit()
    print(f"Loaded {len(loan_rows)} loans and {len(snap_rows)} monthly snapshots.")


if __name__ == "__main__":
    main()
