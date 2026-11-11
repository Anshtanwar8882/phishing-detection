"""Step 1-2 of the workflow: load the real CEAS-08 emails, clean them, derive event features."""
import re
import numpy as np
import pandas as pd
from .config import WINDOW_START, WINDOW_END, OBS_ID

URGENCY_RE = re.compile(
    r"verify|urgent|suspend|password|account|confirm|security|log ?in|bank|immediately|"
    r"click here|limited|update|expire", re.I)
DOMAIN_RE = re.compile(r"@([\w.\-]+)")


def load_ceas(path, start=WINDOW_START, end=WINDOW_END):
    """Load CEAS_08.csv, parse dates, keep the dense window, drop duplicates, sort by time."""
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    n0 = len(df)
    df = df.dropna(subset=["timestamp"])
    df = df[(df["timestamp"] >= pd.Timestamp(start, tz="UTC")) &
            (df["timestamp"] < pd.Timestamp(end, tz="UTC"))]
    df = df.drop_duplicates(subset=["sender", "subject", "body"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.attrs["rows_raw"] = n0
    return df


def email_features(df):
    """Real-data features -> one of four email event categories (observation ids 1..4)."""
    text = (df["subject"].fillna("") + " " + df["body"].fillna("")).str.lower()
    has_kw = text.str.contains(URGENCY_RE).to_numpy()
    has_url = (df["urls"].to_numpy() == 1)
    cat = np.where(~has_url & ~has_kw, OBS_ID["email_clean"],
          np.where(~has_url & has_kw, OBS_ID["email_urgent_kw"],
          np.where(has_url & ~has_kw, OBS_ID["email_url"], OBS_ID["email_url_urgent"])))
    out = df.copy()
    out["has_url"], out["has_kw"], out["email_obs"] = has_url, has_kw, cat
    out["sender_domain"] = df["sender"].fillna("").str.extract(DOMAIN_RE)[0].fillna("unknown")
    return out
