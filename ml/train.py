"""
train.py - trains the two models and saves them to ml/artifacts/.

    python train.py --data ../data

Model 1  underwriting  : "will this borrower default (90+ days late) within 12 months?"
Model 2  monitoring    : "will this active borrower miss a payment in the next 3 months?"

Uses the TIME-BASED `split` column: fit on train, set risk thresholds on valid, report on test.
"""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# the shared feature code lives in the backend package so training and the API can never drift apart
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app import ml_core as mc  # noqa: E402


def train_one(name, df, features, label, q_medium, q_high, out_dir):
    tr, va, te = (df[df["split"] == s] for s in ("train", "valid", "test"))
    Xtr, Xva, Xte = (mc.to_frame(d, features) for d in (tr, va, te))

    model, kind = mc.make_model()
    model.fit(Xtr, tr[label])
    p_va, p_te = model.predict_proba(Xva)[:, 1], model.predict_proba(Xte)[:, 1]

    # simple baseline, to prove the gradient-boosting model adds value
    base = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000))
    base.fit(Xtr, tr[label])
    p_base = base.predict_proba(Xte)[:, 1]

    # risk bands: thresholds come from the VALIDATION set (never from test)
    th = {"medium": float(np.quantile(p_va, q_medium)), "high": float(np.quantile(p_va, q_high))}
    bands = np.where(p_te >= th["high"], "high", np.where(p_te >= th["medium"], "medium", "low"))
    y_te = te[label].to_numpy()
    band_stats = {b: {"share": round(float((bands == b).mean()), 3),
                      "actual_rate": round(float(y_te[bands == b].mean()), 3) if (bands == b).any() else None}
                  for b in ("low", "medium", "high")}
    metrics = {
        "auc_test": round(float(roc_auc_score(y_te, p_te)), 4),
        "auc_valid": round(float(roc_auc_score(va[label], p_va)), 4),
        "auc_baseline_logistic_test": round(float(roc_auc_score(y_te, p_base)), 4),
        "pr_auc_test": round(float(average_precision_score(y_te, p_te)), 4),
        "positive_rate_test": round(float(y_te.mean()), 4),
        "bands_on_test": band_stats, "model_type": kind,
        "rows": {"train": len(tr), "valid": len(va), "test": len(te)},
    }
    art = {"name": name, "kind": kind, "model": model, "features": features, "baseline": mc.baseline_values(Xtr),
           "thresholds": th, "metrics": metrics, "label": label}
    joblib.dump(art, out_dir / f"{name}_model.joblib")

    print(f"\n=== {name} model ({kind}) ===")
    print(f"AUC test {metrics['auc_test']}  | logistic baseline {metrics['auc_baseline_logistic_test']} | PR-AUC {metrics['pr_auc_test']}")
    print("thresholds:", {k: round(v, 4) for k, v in th.items()})
    for b, s in band_stats.items():
        print(f"  {b:6s} band: {s['share']:.0%} of test borrowers, actual {label} rate {s['actual_rate']}")
    return art


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data", help="folder with underwriting_dataset.csv and monitoring_dataset.csv")
    ap.add_argument("--out", default="artifacts")
    a = ap.parse_args()
    data, out = Path(a.data), Path(a.out)
    out.mkdir(exist_ok=True, parents=True)

    uw = pd.read_csv(data / "underwriting_dataset.csv", usecols=["split", "target_default_12m"] + mc.UW_FEATURES)
    train_one("underwriting", uw, mc.UW_FEATURES, "target_default_12m", 0.60, 0.88, out)

    mon = pd.read_csv(data / "monitoring_dataset.csv", usecols=["split", "target_delinquent_next_3m"] + mc.MON_FEATURES)
    train_one("monitoring", mon, mc.MON_FEATURES, "target_delinquent_next_3m", 0.80, 0.93, out)
    print("\nSaved models to", out.resolve())


if __name__ == "__main__":
    main()
