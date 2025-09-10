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

    # Minimal attraction flags (kept because they are highly predictive)
    ent = df['ENTITY_DESCRIPTION_SHORT']
    df['is_water_ride'] = (ent == 'Water Ride').astype(int)
    df['is_flying_coaster'] = (ent == 'Flying Coaster').astype(int)

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
    # Normalize binaries
    for c in ['has_downtime', 'is_water_ride', 'is_flying_coaster']:
        if c in df.columns:
            df[c] = df[c].fillna(0).astype(int)
    # Lag: fallback to current if first row
    if 'wait_lag_1' in df.columns:
        df['wait_lag_1'] = df['wait_lag_1'].fillna(df['CURRENT_WAIT_TIME'])
    return df


def select_features(df: pd.DataFrame, k: int = 20, fit: bool = True,
                    selector: SelectKBest | None = None,
                    selected: list[str] | None = None):
    feature_cols = [c for c in df.columns if c not in ID_COLS + [TARGET_COL]]
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

    keep = ID_COLS + selected + ([TARGET_COL] if TARGET_COL in df.columns else [])
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
    keep = ID_COLS + [f for f in selected if f in df.columns] + ([TARGET_COL] if TARGET_COL in df.columns else [])
    df = df[keep]
    df, _, _ = scale_features(df, fit=False, scaler=scaler, scale_cols=scale_cols)
    if verbose:
        print(f"🚀 Final shape: {df.shape}")
    return df


# ----------------------------
# Quick demo
# ----------------------------

def quick_analysis():
    train_df, selected, selector, scaler, scale_cols = preprocess_train(
        'data/waiting_times_train.csv', 'data/weather_data.csv', k=20, verbose=True
    )
    print(f"Top features ({len(selected)}): {selected}")
    train_df.to_csv('focused_train.csv', index=False)

    val_df = preprocess_eval(
        'data/waiting_times_X_test_val.csv', 'data/weather_data.csv',
        selected, selector, scaler, scale_cols, verbose=True
    )
    val_df.to_csv('focused_val.csv', index=False)
    print("✅ Files saved: focused_train.csv, focused_val.csv")
    return train_df, val_df, selected


if __name__ == '__main__':
    quick_analysis()