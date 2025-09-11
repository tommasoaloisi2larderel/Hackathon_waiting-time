"""
visualise.py — quick EDA and sanity checks for waiting-time data.

Usage (from repo root):
    python -m Hackathon_waiting-time.visualise 

Outputs go to ./figures/ and console.

This script avoids heavy deps; it uses pandas + matplotlib only.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Iterable, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Local imports
try:
    # We reuse constants and helpers if available
    from .preprocess import split_scenarios, TARGET_COL  # type: ignore
except Exception:  # pragma: no cover - when running standalone
    split_scenarios = None  # will handle gracefully
    TARGET_COL = None

FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_TRAIN = Path("data/waiting_times_train.csv")
DEFAULT_VAL = Path("data/waiting_times_X_test_val.csv")


# ----------------------------
# Utilities
# ----------------------------

def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _cat_cols(df: pd.DataFrame, limit: int = 50) -> list[str]:
    cats = []
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_categorical_dtype(df[c]):
            if df[c].nunique(dropna=True) <= limit:
                cats.append(c)
    return cats


def _ensure_datetime(df: pd.DataFrame, col: str = "DATETIME") -> pd.DataFrame:
    if col in df.columns and not np.issubdtype(df[col].dtype, np.datetime64):
        with pd.option_context("future.no_silent_downcasting", True):
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


# ----------------------------
# Plots
# ----------------------------

def _savefig(name: str):
    path = FIG_DIR / f"{name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"📈 saved: {path}")


def plot_missingness(df: pd.DataFrame, title: str):
    """Bar chart of missingness per column (top 40)."""
    miss = df.isna().mean().sort_values(ascending=False).head(40)
    plt.figure(figsize=(12, max(4, len(miss) * 0.25)))
    miss.plot(kind="barh")
    plt.gca().invert_yaxis()
    plt.title(f"Missingness • {title}")
    plt.xlabel("Fraction missing")
    _savefig(f"missingness_{title}")


def plot_distributions(df: pd.DataFrame, title: str, max_cols: int = 24):
    num_cols = _numeric_cols(df)[:max_cols]
    if not num_cols:
        print("(no numeric columns to plot)")
        return
    n = len(num_cols)
    cols = 3
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(cols * 4, rows * 3))
    for i, c in enumerate(num_cols, 1):
        plt.subplot(rows, cols, i)
        vals = df[c].dropna().values
        if len(vals):
            plt.hist(vals, bins=40)
        plt.title(c)
        plt.xlabel("")
        plt.ylabel("")
    plt.suptitle(f"Distributions • {title}")
    _savefig(f"dists_{title}")


def plot_correlation(df: pd.DataFrame, title: str, max_cols: int = 40):
    num_cols = _numeric_cols(df)[:max_cols]
    if len(num_cols) < 2:
        print("(not enough numeric columns for correlation)")
        return
    corr = df[num_cols].corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    plt.imshow(corr, interpolation="nearest", aspect="auto")
    plt.colorbar(label="Pearson r")
    plt.xticks(range(len(num_cols)), num_cols, rotation=90, fontsize=7)
    plt.yticks(range(len(num_cols)), num_cols, fontsize=7)
    plt.title(f"Correlation (top {len(num_cols)}) • {title}")
    _savefig(f"corr_{title}")


def plot_time_series(df: pd.DataFrame, value_col: str, title: str, by_entity: str | None = "ENTITY_DESCRIPTION_SHORT"):
    if value_col not in df.columns:
        print(f"(skip time series: {value_col} not in df)")
        return
    if "DATETIME" not in df.columns:
        print("(skip time series: DATETIME not in df)")
        return
    df = _ensure_datetime(df)
    plt.figure(figsize=(12, 4))
    if by_entity and by_entity in df.columns:
        # Plot 10 biggest series
        top = df[by_entity].value_counts().head(10).index
        for ent in top:
            s = df.loc[df[by_entity] == ent, ["DATETIME", value_col]].sort_values("DATETIME")
            plt.plot(s["DATETIME"], s[value_col], label=str(ent), linewidth=1)
        plt.legend(loc="upper right", fontsize=7, ncol=2)
    else:
        s = df.sort_values("DATETIME")
        plt.plot(s["DATETIME"], s[value_col], linewidth=1)
    plt.title(f"Time series of {value_col} • {title}")
    plt.xlabel("time")
    plt.ylabel(value_col)
    _savefig(f"ts_{value_col}_{title}")


def plot_category_box(df: pd.DataFrame, value_col: str, cat_col: str, title: str):
    if value_col not in df.columns or cat_col not in df.columns:
        print(f"(skip boxplot: missing {value_col} or {cat_col})")
        return
    # Limit to top 15 categories by count
    counts = df[cat_col].value_counts().head(15).index
    data = [df.loc[df[cat_col] == k, value_col].dropna().values for k in counts]
    plt.figure(figsize=(12, 5))
    plt.boxplot(data, labels=[str(k) for k in counts], showfliers=False)
    plt.xticks(rotation=30, ha="right")
    plt.title(f"{value_col} by {cat_col} • {title}")
    _savefig(f"box_{value_col}_by_{cat_col}_{title}")


def plot_pivot_heatmap(df: pd.DataFrame, value_col: str, row: str, col: str, title: str, agg="median"):
    for need in (value_col, row, col):
        if need not in df.columns:
            print(f"(skip heatmap: missing {need})")
            return
    if agg == "median":
        pivot = df.pivot_table(index=row, columns=col, values=value_col, aggfunc="median")
    else:
        pivot = df.pivot_table(index=row, columns=col, values=value_col, aggfunc="mean")
    plt.figure(figsize=(10, 6))
    plt.imshow(pivot.values, aspect="auto", interpolation="nearest")
    plt.colorbar(label=value_col)
    plt.xticks(range(pivot.shape[1]), pivot.columns.astype(str), rotation=45, ha="right")
    plt.yticks(range(pivot.shape[0]), pivot.index.astype(str))
    plt.title(f"{value_col} {agg} by {row}×{col} • {title}")
    _savefig(f"heat_{value_col}_{row}_x_{col}_{title}")


# ----------------------------
# Sanity / absurdity checks
# ----------------------------

def absurdity_checks(df: pd.DataFrame) -> pd.DataFrame:
    """Return a table of potential absurd rows with reasons.
    This is heuristic and schema-aware for a few known columns.
    """
    issues = []
    n = len(df)
    def add(idx, reason):
        issues.append({"index": idx, "reason": reason})

    # 1) Negative or massive waits / times
    suspect_cols = [
        c for c in df.columns if any(k in c.lower() for k in ["wait", "time_to", "minutes", "queue"]) and pd.api.types.is_numeric_dtype(df[c])
    ]
    for c in suspect_cols:
        s = df[c]
        if not pd.api.types.is_numeric_dtype(s):
            continue
        neg_idx = s.index[s < 0]
        for i in neg_idx:
            add(i, f"negative value in {c}={s.loc[i]}")
        # absurdly large: > 24h in minutes
        big_idx = s.index[s > 24 * 60]
        for i in big_idx:
            add(i, f"unrealistic large value in {c}={s.loc[i]}")

    # 2) Night show without any parade (if helpers available, classify)
    if split_scenarios is not None:
        try:
            splits = split_scenarios(df)
            # rows that didn't fit any known scenario
            p1 = df.get('TIME_TO_PARADE_1')
            p2 = df.get('TIME_TO_PARADE_2')
            ns = df.get('TIME_TO_NIGHT_SHOW')
            if p1 is not None and p2 is not None and ns is not None:
                unmatched = ~(p1.notna() & p2.notna() & ns.notna()) & ~((p1.notna() & ~p2.notna() & ns.notna()) | (~p1.notna() & ~p2.notna() & ~ns.notna()))
                for i in df.index[unmatched]:
                    add(i, "in-between scenario (unknown combination of parade/night show)")
        except Exception:
            pass

    # 3) Duplicates on (DATETIME, ENTITY)
    entity_col = 'ENTITY_DESCRIPTION_SHORT' if 'ENTITY_DESCRIPTION_SHORT' in df.columns else None
    if 'DATETIME' in df.columns:
        df = _ensure_datetime(df)
        keys = ['DATETIME'] + ([entity_col] if entity_col else [])
        if keys:
            dup_mask = df.duplicated(subset=keys, keep=False)
            for i in df.index[dup_mask]:
                add(i, f"duplicate key on {keys}")

    # 4) Constant columns
    const_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    for c in const_cols:
        add(None, f"constant column: {c}")

    # 5) Extreme z-scores for target if known
    if TARGET_COL and TARGET_COL in df.columns and pd.api.types.is_numeric_dtype(df[TARGET_COL]):
        s = df[TARGET_COL].astype(float)
        z = (s - s.mean()) / (s.std(ddof=0) + 1e-9)
        extreme = s.index[np.abs(z) > 6]
        for i in extreme:
            add(i, f"extreme {TARGET_COL} z>|6|: {s.loc[i]:.2f}")

    issues_df = pd.DataFrame(issues)
    if not issues_df.empty:
        # Add context columns if available
        cols = [c for c in ("DATETIME", entity_col, TARGET_COL) if c and c in df.columns]
        if cols:
            issues_df = issues_df.join(df[cols], on="index", how="left")
    print(f"🔎 absurdity checks flagged {len(issues_df)} items out of {n} rows")
    return issues_df


# ----------------------------
# Runner
# ----------------------------

def run_visualisations(df: pd.DataFrame, tag: str):
    # High-level
    plot_missingness(df, tag)
    plot_distributions(df, tag)
    plot_correlation(df, tag)

    # Time series (if target known)
    for col in [TARGET_COL, 'TIME_TO_PARADE_1', 'TIME_TO_PARADE_2', 'TIME_TO_NIGHT_SHOW']:
        if isinstance(col, str) and col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            plot_time_series(df, col, tag)

    # Category views
    for cat in ['ENTITY_DESCRIPTION_SHORT', 'DAY_OF_WEEK', 'PARK_NAME']:
        if cat in df.columns and TARGET_COL in df.columns and pd.api.types.is_numeric_dtype(df[TARGET_COL]):
            plot_category_box(df, TARGET_COL, cat, tag)

    # Calendar heatmap-ish: median by day-of-week × hour
    if 'DATETIME' in df.columns and TARGET_COL in df.columns and pd.api.types.is_numeric_dtype(df[TARGET_COL]):
        tmp = df.copy()
        tmp = _ensure_datetime(tmp)
        tmp['DOW'] = tmp['DATETIME'].dt.day_name()
        tmp['HOUR'] = tmp['DATETIME'].dt.hour
        plot_pivot_heatmap(tmp, TARGET_COL, 'DOW', 'HOUR', tag, agg='median')

    # Absurdities
    issues = absurdity_checks(df)
    if not issues.empty:
        out_csv = FIG_DIR / f"absurdities_{tag}.csv"
        issues.to_csv(out_csv, index=False)
        print(f"🧾 saved: {out_csv}")


def load_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    return df


def main(train_path: Path = DEFAULT_TRAIN, val_path: Path = DEFAULT_VAL):
    print("🎨 Visualising datasets…")
    try:
        train = load_csv_safe(train_path)
        print(f"• train: {train.shape}")
        run_visualisations(train, tag="train")
    except Exception as e:
        print(f"⚠️ train failed: {e}")

    try:
        val = load_csv_safe(val_path)
        print(f"• val: {val.shape}")
        run_visualisations(val, tag="val")
    except Exception as e:
        print(f"⚠️ val failed: {e}")


if __name__ == "__main__":
    main()
