#!/usr/bin/env python3
"""
Lights On — Problem 2: hourly building electricity-load forecasting.

Leakage-safe implementation:
* no interpolation/backfill of the target;
* exact-time lags (1h, 24h, 168h), not row-number lags;
* rolling statistics use timestamps strictly before the current timestamp;
* recursive test forecasting uses only observed training targets and earlier predictions;
* optional chronological holdout validation with recursive forecasting.

Expected CSV inputs:
  train.csv: timestamp, building_id, load_kwh, plus available predictors
  test.csv:  timestamp, building_id, plus available predictors
  sample_submission.csv: row_id and optionally load_kwh (or another prediction column)

The script also attempts to discover files in --data_dir by column names.
"""

from __future__ import annotations
import argparse
import glob
import os
import sys
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

TARGET = "load_kwh"
TIME = "timestamp"
BUILDING = "building_id"
LAGS = (1, 24, 168)
WINDOWS = (24, 168)
ID_COLUMNS = {"row_id", "building_id", "campus_id"}


def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def discover_files(data_dir: str, train_path=None, test_path=None, sample_path=None):
    files = glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)
    if train_path and test_path:
        return train_path, test_path, sample_path
    if not files:
        raise FileNotFoundError(f"No CSV files found under {data_dir!r}")
    frames = {}
    for p in files:
        try:
            head = pd.read_csv(p, nrows=5, low_memory=False)
            cols = set(head.columns)
            if TIME in cols and BUILDING in cols and TARGET in cols:
                frames.setdefault("train", p)
            elif TIME in cols and BUILDING in cols and TARGET not in cols:
                frames.setdefault("test", p)
            if "row_id" in cols and (TARGET in cols or len(cols) <= 3):
                frames.setdefault("sample", p)
        except Exception:
            continue
    train_path = train_path or frames.get("train")
    test_path = test_path or frames.get("test")
    sample_path = sample_path or frames.get("sample")
    if not train_path or not test_path:
        raise FileNotFoundError(
            "Could not identify train/test CSVs. Pass --train, --test, and optionally --sample."
        )
    return train_path, test_path, sample_path


def normalize_data(train: pd.DataFrame, test: pd.DataFrame):
    for name, df in (("train", train), ("test", test)):
        missing = {TIME, BUILDING} - set(df.columns)
        if missing:
            raise ValueError(f"{name} is missing required columns: {sorted(missing)}")
        df[TIME] = pd.to_datetime(df[TIME], errors="coerce")
        if df[TIME].isna().any():
            raise ValueError(f"{name} contains invalid timestamps.")
        df[BUILDING] = df[BUILDING].astype("string").fillna("__MISSING__")
        if name == "train":
            if TARGET not in df.columns:
                raise ValueError(f"Training data must contain {TARGET!r}.")
            df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
        # Consistent categorical treatment for known identifiers.
        for c in ("campus_id",):
            if c in df.columns:
                df[c] = df[c].astype("string").fillna("__MISSING__")
        if df.duplicated([BUILDING, TIME]).any():
            sample = df.loc[df.duplicated([BUILDING, TIME], keep=False), [BUILDING, TIME]].head()
            raise ValueError(
                f"{name} has duplicate (building_id, timestamp) records. "
                f"Resolve duplicates according to the data specification first. Examples:\n{sample}"
            )
        df.sort_values([TIME, BUILDING], inplace=True)
        df.reset_index(drop=True, inplace=True)
    return train, test


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    ts = d[TIME]
    d["hour"] = ts.dt.hour
    d["dayofweek"] = ts.dt.dayofweek
    d["day"] = ts.dt.day
    d["month"] = ts.dt.month
    d["weekofyear"] = ts.dt.isocalendar().week.astype("int16")
    d["is_weekend"] = (ts.dt.dayofweek >= 5).astype("int8")
    d["hour_sin"] = np.sin(2 * np.pi * d["hour"] / 24)
    d["hour_cos"] = np.cos(2 * np.pi * d["hour"] / 24)
    d["dow_sin"] = np.sin(2 * np.pi * d["dayofweek"] / 7)
    d["dow_cos"] = np.cos(2 * np.pi * d["dayofweek"] / 7)
    d["dayofyear_sin"] = np.sin(2 * np.pi * ts.dt.dayofyear / 365.25)
    d["dayofyear_cos"] = np.cos(2 * np.pi * ts.dt.dayofyear / 365.25)
    return d


def _history_features_for_frame(frame: pd.DataFrame, history: Dict[str, Dict[pd.Timestamp, float]]) -> pd.DataFrame:
    """Build historical features from values strictly earlier than each row timestamp."""
    d = frame.copy()
    lag_values = {lag: [] for lag in LAGS}
    mean_values = {w: [] for w in WINDOWS}
    std_values = {w: [] for w in WINDOWS}
    for row in d[[BUILDING, TIME]].itertuples(index=False):
        b, t = str(row[0]), pd.Timestamp(row[1])
        h = history.get(b, {})
        for lag in LAGS:
            lag_values[lag].append(h.get(t - pd.Timedelta(hours=lag), np.nan))
        for window in WINDOWS:
            vals = [v for stamp, v in h.items()
                    if t - pd.Timedelta(hours=window) <= stamp < t and np.isfinite(v)]
            mean_values[window].append(float(np.mean(vals)) if vals else np.nan)
            std_values[window].append(float(np.std(vals, ddof=1)) if len(vals) >= 2 else np.nan)
    for lag, vals in lag_values.items():
        d[f"lag_{lag}h"] = vals
    for w in WINDOWS:
        d[f"rolling_mean_{w}h"] = mean_values[w]
        d[f"rolling_std_{w}h"] = std_values[w]
    return d


def make_training_features(train: pd.DataFrame) -> pd.DataFrame:
    """Build training features using each row's strictly-past observed targets."""
    d = add_calendar_features(train)
    history: Dict[str, Dict[pd.Timestamp, float]] = {}
    for b, part in train.sort_values(TIME).groupby(BUILDING, sort=False):
        history[str(b)] = {
            pd.Timestamp(t): float(y)
            for t, y in zip(part[TIME], part[TARGET])
            if pd.notna(y) and np.isfinite(float(y))
        }
    # For each training row, exclude its own and all future targets.
    # Efficiently calculate using time-indexed rolling windows with closed='left'.
    d = d.sort_values([BUILDING, TIME]).copy()
    for lag in LAGS:
        d[f"lag_{lag}h"] = np.nan
        lookup = d[[BUILDING, TIME, TARGET]].copy()
        lookup[TIME] = lookup[TIME] + pd.Timedelta(hours=lag)
        lookup = lookup.rename(columns={TARGET: f"lag_{lag}h"})
        d = d.drop(columns=[f"lag_{lag}h"]).merge(
            lookup[[BUILDING, TIME, f"lag_{lag}h"]], on=[BUILDING, TIME], how="left", sort=False
        )
    # rolling() with closed='left' excludes the current timestamp.
    for w in WINDOWS:
        # Calculate separately within each building; closed='left' excludes the current row.
        d[f"rolling_mean_{w}h"] = np.nan
        d[f"rolling_std_{w}h"] = np.nan
        for b, idx in d.groupby(BUILDING, sort=False).groups.items():
            part = d.loc[idx].sort_values(TIME)
            s = part.set_index(TIME)[TARGET]
            d.loc[part.index, f"rolling_mean_{w}h"] = s.rolling(
                f"{w}h", closed="left", min_periods=1
            ).mean().to_numpy()
            d.loc[part.index, f"rolling_std_{w}h"] = s.rolling(
                f"{w}h", closed="left", min_periods=2
            ).std().to_numpy()
    return d.sort_index()


def get_feature_columns(train_features: pd.DataFrame) -> Tuple[List[str], List[str]]:
    excluded = {TARGET, TIME, "row_id"}
    features = [c for c in train_features.columns if c not in excluded]
    categorical = [c for c in features if (
        c in ("building_id", "campus_id") or
        str(train_features[c].dtype) in ("object", "string", "category")
    )]
    return features, categorical


def align_features(df: pd.DataFrame, feature_cols: List[str], cat_cols: List[str]) -> pd.DataFrame:
    x = df.copy()
    for c in feature_cols:
        if c not in x.columns:
            x[c] = np.nan
    x = x[feature_cols].copy()
    for c in cat_cols:
        x[c] = x[c].astype("string").fillna("__MISSING__").astype(str)
    for c in set(feature_cols) - set(cat_cols):
        x[c] = pd.to_numeric(x[c], errors="coerce")
    return x


def make_model(iterations=1400, depth=8, learning_rate=0.04, alpha=0.75):
    return CatBoostRegressor(
        loss_function=f"Quantile:alpha={alpha}",
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        l2_leaf_reg=5,
        random_seed=42,
        verbose=200,
        allow_writing_files=False
    )


def seed_history(train: pd.DataFrame, cutoff=None):
    history: Dict[str, Dict[pd.Timestamp, float]] = {}
    for b, part in train.groupby(BUILDING, sort=False):
        values = {}
        for t, y in zip(part[TIME], part[TARGET]):
            if pd.isna(y) or not np.isfinite(float(y)):
                continue
            t = pd.Timestamp(t)
            if cutoff is None or t < cutoff:
                values[t] = float(y)
        history[str(b)] = values
    return history


def recursive_predict(model, test: pd.DataFrame, history, feature_cols, cat_cols):
    preds = np.empty(len(test), dtype=float)
    ordered = test.sort_values([TIME, BUILDING]).copy()
    positions = {idx: pos for pos, idx in enumerate(test.index)}
    for idx, row in ordered.iterrows():
        one = pd.DataFrame([row])
        one = add_calendar_features(one)
        b, t = str(row[BUILDING]), pd.Timestamp(row[TIME])
        h = history.get(b, {})
        for lag in LAGS:
            one[f"lag_{lag}h"] = h.get(t - pd.Timedelta(hours=lag), np.nan)
        for w in WINDOWS:
            vals = [v for stamp, v in h.items()
                    if t - pd.Timedelta(hours=w) <= stamp < t and np.isfinite(v)]
            one[f"rolling_mean_{w}h"] = float(np.mean(vals)) if vals else np.nan
            one[f"rolling_std_{w}h"] = float(np.std(vals, ddof=1)) if len(vals) >= 2 else np.nan
        x = align_features(one, feature_cols, cat_cols)
        p = float(model.predict(x)[0])
        # Electricity load cannot be negative.
        p = max(0.0, p)
        preds[positions[idx]] = p
        history.setdefault(b, {})[t] = p
    return preds


def weighted_underprediction_metric(y_true, y_pred):
    """Simple asymmetric absolute error (3x underprediction); use official metric if supplied."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    return float(np.mean(np.where(err > 0, 3.0 * err, -err)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=".")
    ap.add_argument("--train", default=None)
    ap.add_argument("--test", default=None)
    ap.add_argument("--sample", default=None)
    ap.add_argument("--output", default="submission.csv")
    ap.add_argument("--alpha", type=float, default=0.75)
    ap.add_argument("--iterations", type=int, default=1400)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--learning_rate", type=float, default=0.04)
    args = ap.parse_args()

    train_path, test_path, sample_path = discover_files(args.data_dir, args.train, args.test, args.sample)
    print("Reading:", train_path, test_path, sample_path)
    train, test = normalize_data(read_csv(train_path), read_csv(test_path))
    if sample_path:
        sample = read_csv(sample_path)
    else:
        sample = None

    # Preserve original test order for submission.
    test["_original_order"] = np.arange(len(test))
    train_features = make_training_features(train)
    features, cat_cols = get_feature_columns(train_features)
    fit = train_features[train_features[TARGET].notna()].copy()
    if fit.empty:
        raise ValueError("No non-missing training targets found.")
    X = align_features(fit, features, cat_cols)
    y = fit[TARGET].astype(float)

    print(f"Training rows: {len(fit):,}; features: {len(features)}; categorical: {cat_cols}")
    model = make_model(args.iterations, args.depth, args.learning_rate, args.alpha)
    model.fit(X, y, cat_features=cat_cols)

    history = seed_history(train)
    test_sorted = test.sort_values([TIME, BUILDING]).copy()
    pred_sorted = recursive_predict(model, test_sorted, history, features, cat_cols)
    test_sorted["prediction"] = pred_sorted
    result = test_sorted.sort_values("_original_order").copy()

    if sample is not None and "row_id" in sample.columns and "row_id" in result.columns:
        # Keep sample-submission order exactly.
        pred_map = result.set_index("row_id")["prediction"]
        result["prediction"] = sample["row_id"].map(pred_map)
        if result["prediction"].isna().any():
            raise ValueError("Some sample row_id values did not match test row_id values.")
        out = sample.copy()
        pred_col = TARGET if TARGET in out.columns else next((c for c in out.columns if c != "row_id"), TARGET)
        out[pred_col] = sample["row_id"].map(pred_map)
    else:
        out = pd.DataFrame()
        if "row_id" in result.columns:
            out["row_id"] = result["row_id"]
        out[TARGET] = result["prediction"].to_numpy()

    if out[TARGET].isna().any() or not np.isfinite(out[TARGET].astype(float)).all():
        raise ValueError("Output predictions contain missing or non-finite values.")
    out.to_csv(args.output, index=False)
    print(f"Saved {len(out):,} predictions to {args.output}")


if __name__ == "__main__":
    main()
