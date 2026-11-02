"""Anomaly detection for imported orders."""
import numpy as np
import pandas as pd

METHODS = ["Robust z-score", "Isolation Forest"]


def detect(df: pd.DataFrame, method: str = "Robust z-score", z_thr: float = 3.5, contamination: float = 0.03):
    """Return (df with anomaly_score / is_anomaly / reason, note)."""
    out = df.copy()
    out["anomaly_score"], out["is_anomaly"], out["reason"] = 0.0, False, ""
    if len(out) < 30:
        return out, "Need at least 30 rows for anomaly detection."

    price = out["purchase_price"].astype(float)
    med = price.median()
    mad = (price - med).abs().median()
    scale = mad / 0.6745 if mad > 0 else (price.std() or 1.0)
    z = (price - med) / scale
    direction = np.where(z > 0, "High price", "Low price")

    note = ""
    if method == "Isolation Forest":
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            note = "scikit-learn is not installed, so the robust z-score is used instead."
            method = "Robust z-score"

    if method == "Isolation Forest":
        X = pd.DataFrame({
            "price": price,
            "provider": out["cc_provider"].astype("category").cat.codes,
            "am_pm": out["am_pm"].astype("category").cat.codes,
            "language": out["language"].astype("category").cat.codes,
            "hour": out["order_date"].dt.hour.fillna(-1) if out["order_date"].notna().any() else 0,
        })
        model = IsolationForest(n_estimators=150, contamination=contamination, random_state=42).fit(X)
        out["anomaly_score"] = -model.score_samples(X)
        out["is_anomaly"] = model.predict(X) == -1
        out["reason"] = np.where(out["is_anomaly"], np.where(z.abs() > 3, direction, "Unusual pattern"), "")
    else:
        out["anomaly_score"] = z.abs()
        out["is_anomaly"] = z.abs() > z_thr
        out["reason"] = np.where(out["is_anomaly"], direction, "")
    return out, note