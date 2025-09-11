import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

TARGET_COL = 'WAIT_TIME_IN_2H'
ID_COLS = ['DATETIME', 'ENTITY_DESCRIPTION_SHORT']

# ----------------------------
# Core feature engineering
# ----------------------------

def agreeable_weather(temp, rain, wind):
    """Single scalar 0..1 for weather comfort.
    - Temp: peak around 22°C; gaussian tolerance ~10°C
    - Rain: strong penalty up to 5 mm/h
    - Wind: mild penalty up to 20 m/s
    Missing values -> NaN, handled later.
    """
    if temp is None and rain is None and wind is None:
        return np.nan
    t = temp
    r = rain if rain is not None else 0.0
    w = wind if wind is not None else 0.0
    # soft scores
    t_score = np.exp(-((t - 22.0)/10.0)**2) if t is not None else 0.8
    r_pen = np.clip((r or 0.0)/5.0, 0, 1)
    w_pen = np.clip((w or 0.0)/20.0, 0, 1)
    return float(np.clip(t_score*(1 - 0.7*r_pen)*(1 - 0.3*w_pen), 0, 1))


def create_features(df: pd.DataFrame, weather_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Lean, leak-safe features. Hours are linear (park closes at night)."""
    df = df.copy()
    dt = pd.to_datetime(df['DATETIME'])
    df['DATETIME'] = dt

    # Time features (linear hour)
    df['hour'] = dt.dt.hour.astype(int)
    df['dow'] = dt.dt.dayofweek.astype(int)
    df['month'] = dt.dt.month.astype(int)

    # Ops features
    df['wait_per_capacity'] = df['CURRENT_WAIT_TIME'] / (df['ADJUST_CAPACITY'] + 0.1)
    df['has_downtime'] = (df.get('DOWNTIME', 0) > 0).astype(int)

    # Minimal attraction flags (2 columns for 3 rides; Pirate Ship is 0/0)
    ent = df['ENTITY_DESCRIPTION_SHORT']
    df['ride_water'] = (ent == 'Water Ride').astype(int)
    df['ride_flying'] = (ent == 'Flying Coaster').astype(int)

    # Weather merge and single comfort score
    if weather_df is not None and len(weather_df) > 0:
        w = weather_df.copy()
        w['DATETIME'] = pd.to_datetime(w['DATETIME'])
        df = pd.merge_asof(df.sort_values('DATETIME'), w.sort_values('DATETIME'), on='DATETIME', direction='backward')
        temp = df.get('temp')
        rain = df.get('rain_1h')
        wind = df.get('wind_speed')
        df['weather_agreeable'] = [
            agreeable_weather(t if pd.notna(t) else None,
                              r if pd.notna(r) else None,
                              w if pd.notna(w) else None)
            for t, r, w in zip(temp if temp is not None else [None]*len(df),
                                rain if rain is not None else [None]*len(df),
                                wind if wind is not None else [None]*len(df))
        ]
        # Drop raw weather columns; keep the single scalar
        df.drop(columns=['temp', 'rain_1h', 'wind_speed', 'dew_point', 'pressure', 'humidity', 'snow_1h', 'clouds_all'], errors='ignore', inplace=True)
    else:
        df['weather_agreeable'] = np.nan

    # Single event variable and flag (minutes to next event)
    ev_cols = [c for c in ['TIME_TO_PARADE_1','TIME_TO_PARADE_2','TIME_TO_NIGHT_SHOW'] if c in df.columns]
    if ev_cols:
        tmp = df[ev_cols].copy()
        for c in ev_cols:
            tmp[c] = tmp[c].where(tmp[c] >= 0, np.nan)  # ignore past events
        df['mins_to_event'] = tmp.min(axis=1).fillna(999.0)
        df['event_soon_60'] = (df['mins_to_event'] <= 60).astype(int)
    else:
        df['mins_to_event'] = 999.0
        df['event_soon_60'] = 0

    # 1-step lag per attraction (true past only)
    df['wait_lag_1'] = (
        df.sort_values(['ENTITY_DESCRIPTION_SHORT', 'DATETIME'])
          .groupby('ENTITY_DESCRIPTION_SHORT')['CURRENT_WAIT_TIME']
          .shift(1)
    )
    df['wait_trend_1'] = df['CURRENT_WAIT_TIME'] - df['wait_lag_1']

    return df


# ----------------------------
# Cleaning & selection
# ----------------------------

def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Replace inf with NaN
    for c in df.columns:
        if df[c].dtype.kind in 'biufc':
            df[c] = df[c].replace([np.inf, -np.inf], np.nan)
    # Impute numeric by attraction median -> global median (except target)
    num_cols = [c for c in df.columns if df[c].dtype.kind in 'biufc' and c != TARGET_COL]
    if 'ENTITY_DESCRIPTION_SHORT' in df.columns and num_cols:
        grp = df.groupby('ENTITY_DESCRIPTION_SHORT')
        for c in num_cols:
            df[c] = grp[c].transform(lambda s: s.fillna(s.median()))
    for c in num_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    # Final fallbacks for stubborn NaNs (e.g., columns entirely NaN within a scenario)
    if 'weather_agreeable' in df.columns:
        df['weather_agreeable'] = df['weather_agreeable'].fillna(0.5)  # neutral comfort default
    for c in num_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(0.0)
    # Normalize binaries
    for c in ['has_downtime', 'ride_water', 'ride_flying']:
        if c in df.columns:
            df[c] = df[c].fillna(0).astype(int)
    # Lag: fallback to current if first row
    if 'wait_lag_1' in df.columns:
        df['wait_lag_1'] = df['wait_lag_1'].fillna(df['CURRENT_WAIT_TIME'])
    return df


def select_features(df: pd.DataFrame, k: int = 20, fit: bool = True,
                    selector: SelectKBest | None = None,
                    selected: list[str] | None = None):
    feature_cols = [c for c in df.columns if c not in ['DATETIME', 'ENTITY_DESCRIPTION_SHORT', TARGET_COL]]
    always_keep = [f for f in ['weather_agreeable','mins_to_event','event_soon_60','wait_trend_1'] if f in feature_cols]

    if fit:
        X = df[feature_cols]
        y = df[TARGET_COL]
        selector = SelectKBest(score_func=f_regression, k=k)
        selector.fit(X, y)
        mask = selector.get_support()
        selected = [f for f, keep in zip(feature_cols, mask) if keep]
        selected = [f for f in selected if f not in always_keep]
        selected = (always_keep + selected)[:k]
    else:
        selected = [f for f in (selected or feature_cols) if f in df.columns]
        for f in always_keep:
            if f not in selected:
                selected.insert(0, f)
        selected = selected[:k]

    keep = ['DATETIME', 'ENTITY_DESCRIPTION_SHORT'] + selected + ([TARGET_COL] if TARGET_COL in df.columns else [])
    return df[keep], selected, selector


def scale_features(df: pd.DataFrame, fit: bool = True,
                   scaler: StandardScaler | None = None,
                   scale_cols: list[str] | None = None):
    df = df.copy()
    def is_binary(s: pd.Series) -> bool:
        return s.dropna().isin([0, 1]).all()

    if fit:
        scale_cols = [
            c for c in df.columns
            if c not in ID_COLS + [TARGET_COL] and df[c].dtype.kind in 'fc' and not is_binary(df[c])
        ]
        scaler = StandardScaler()
        if scale_cols:
            df[scale_cols] = scaler.fit_transform(df[scale_cols])
    else:
        scale_cols = scale_cols or []
        if scale_cols:
            for c in scale_cols:
                if c not in df.columns:
                    df[c] = 0.0
            df[scale_cols] = scaler.transform(df[scale_cols])
    return df, scaler, scale_cols


# ----------------------------
# Public helpers for train / eval
# ----------------------------

def split_scenarios(df: pd.DataFrame) -> dict:
    """Split *raw* dataframe into three scenarios BEFORE any normalization/imputation.
    Scenarios are row-level, based on presence of parade time columns:
      1) 'p1_night_only': TIME_TO_PARADE_1 and TIME_TO_NIGHT_SHOW present, TIME_TO_PARADE_2 absent
      2) 'all_three': all three present
      3) 'no_parade': none present
    If parade columns are missing entirely, they are treated as all-NaN.
    Returns a dict mapping scenario key to a *copy* of the subset dataframe.
    """
    df = df.copy()
    p1 = df['TIME_TO_PARADE_1'] if 'TIME_TO_PARADE_1' in df.columns else pd.Series([np.nan] * len(df), index=df.index)
    p2 = df['TIME_TO_PARADE_2'] if 'TIME_TO_PARADE_2' in df.columns else pd.Series([np.nan] * len(df), index=df.index)
    ns = df['TIME_TO_NIGHT_SHOW'] if 'TIME_TO_NIGHT_SHOW' in df.columns else pd.Series([np.nan] * len(df), index=df.index)

    has_p1 = p1.notna()
    has_p2 = p2.notna()
    has_ns = ns.notna()

    mask_all_three = has_p1 & has_p2 & has_ns
    mask_p1_night_only = has_p1 & (~has_p2) & has_ns
    mask_no_parade = (~has_p1) & (~has_p2) & (~has_ns)

    subsets = {
        'p1_night_only': df[mask_p1_night_only].copy(),
        'all_three': df[mask_all_three].copy(),
        'no_parade': df[mask_no_parade].copy(),
    }

    # Optional safety check: rows not matched (should be empty per problem statement)
    unmatched = ~(mask_all_three | mask_p1_night_only | mask_no_parade)
    if unmatched.any():
        # Keep them out of all subsets, but warn via print so the user can investigate
        print(f"⚠️ {unmatched.sum()} rows did not match any scenario and were skipped.")

    return subsets


def preprocess_train(waiting_times_path: str, weather_path: str | None = None, k: int = 15, verbose: bool = True):
    if verbose:
        print("🎯 Preprocessing (TRAIN)")
    df = pd.read_csv(waiting_times_path)
    w = pd.read_csv(weather_path) if weather_path else None
    df = create_features(df, w)
    df = handle_missing(df)
    df, selected, selector = select_features(df, k=k, fit=True)
    if verbose:
        print(f"✓ Selected {len(selected)} features")
    df, scaler, scale_cols = scale_features(df, fit=True)
    if verbose:
        print(f"✓ Scaled {len(scale_cols)} columns\n🚀 Final shape: {df.shape}")
    return df, selected, selector, scaler, scale_cols


def preprocess_eval(waiting_times_path: str, weather_path: str | None,
                    selected: list[str], selector: SelectKBest,
                    scaler: StandardScaler, scale_cols: list[str], verbose: bool = True):
    if verbose:
        print("🎯 Preprocessing (EVAL)")
    df = pd.read_csv(waiting_times_path)
    w = pd.read_csv(weather_path) if weather_path else None
    df = create_features(df, w)
    df = handle_missing(df)
    # Use provided selected list (order preserved)
    keep = ['DATETIME', 'ENTITY_DESCRIPTION_SHORT'] + [f for f in selected if f in df.columns] + ([TARGET_COL] if TARGET_COL in df.columns else [])
    df = df[keep]
    df, _, _ = scale_features(df, fit=False, scaler=scaler, scale_cols=scale_cols)
    if verbose:
        print(f"🚀 Final shape: {df.shape}")
    return df


def preprocess_train_eval_by_scenario(
    train_path: str,
    val_path: str,
    weather_path: str | None = None,
    k: int = 15,
    verbose: bool = True,
    out_prefix_train: str = 'focused_train_',
    out_prefix_val: str = 'focused_val_'
):
    """Run the full pipeline *per scenario*, splitting BEFORE normalization/imputation.
    Produces six files (up to): one train + one val per scenario key.
    Returns a dict of per-scenario artifacts.
    """
    if verbose:
        print("🎯 Preprocessing by scenario")

    # Read raw (unsanitized) so we can split based on emptiness
    raw_train = pd.read_csv(train_path)
    raw_val = pd.read_csv(val_path)
    w = pd.read_csv(weather_path) if weather_path else None

    train_splits = split_scenarios(raw_train)
    val_splits = split_scenarios(raw_val)

    artifacts = {}

    for key, df_train_raw in train_splits.items():
        if verbose:
            print(f"\n— Scenario: {key} (train rows: {len(df_train_raw)})")
        # Run the standard pipeline on the subset
        df_train = create_features(df_train_raw, w)
        df_train = handle_missing(df_train)
        df_train, selected, selector = select_features(df_train, k=k, fit=True)
        df_train, scaler, scale_cols = scale_features(df_train, fit=True)
        # Save
        train_out = f"{out_prefix_train}{key}.csv"
        df_train.to_csv(train_out, index=False)
        if verbose:
            print(f"✓ {key}: saved {train_out} with shape {df_train.shape}")

        # Match validation subset for the same scenario
        df_val_raw = val_splits.get(key, pd.DataFrame(columns=raw_val.columns))
        if verbose:
            print(f"  Validation subset rows: {len(df_val_raw)}")
        if len(df_val_raw) > 0:
            df_val = create_features(df_val_raw, w)
            df_val = handle_missing(df_val)
            # Keep provided selected list and scale with trained scaler
            keep = ['DATETIME', 'ENTITY_DESCRIPTION_SHORT'] + [f for f in selected if f in df_val.columns] + ([TARGET_COL] if TARGET_COL in df_val.columns else [])
            df_val = df_val[keep]
            df_val, _, _ = scale_features(df_val, fit=False, scaler=scaler, scale_cols=scale_cols)
        else:
            # Empty subset -> create an empty frame with the same columns as train
            df_val = pd.DataFrame(columns=df_train.columns)
        val_out = f"{out_prefix_val}{key}.csv"
        df_val.to_csv(val_out, index=False)
        if verbose:
            print(f"✓ {key}: saved {val_out} with shape {df_val.shape}")

        artifacts[key] = {
            'train_df': df_train,
            'val_df': df_val,
            'selected': selected,
            'selector': selector,
            'scaler': scaler,
            'scale_cols': scale_cols,
        }

    return artifacts


# ----------------------------
# Quick demo
# ----------------------------

def quick_analysis():
    artifacts = preprocess_train_eval_by_scenario(
        train_path='data/waiting_times_train.csv',
        val_path='data/waiting_times_X_test_val.csv',
        weather_path='data/weather_data.csv',
        k=20,
        verbose=True,
        out_prefix_train='focused_train_',
        out_prefix_val='focused_val_'
    )
    print("\n✅ Files saved:")
    for key, art in artifacts.items():
        print(f"  - focused_train_{key}.csv  (shape: {art['train_df'].shape})")
        print(f"  - focused_val_{key}.csv    (shape: {art['val_df'].shape})")
        print(f"    Top features ({len(art['selected'])}): {art['selected']}")
    return artifacts


if __name__ == '__main__':
    quick_analysis()