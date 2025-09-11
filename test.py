import sys, os, re, textwrap, warnings, math, gc, json, itertools, typing
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

# scikit-learn core
from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV, cross_val_score, KFold, learning_curve
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

# Optional libs
try:
    import shap
    SHAP_AVAILABLE = True
except Exception as e:
    SHAP_AVAILABLE = False

# SciPy is optional for Q-Q plots / KDE fallback
try:
    import scipy
    from scipy import stats
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Print environment/library versions for reproducibility
versions = {
    "python": sys.version.split()[0],
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "matplotlib": matplotlib.__version__,
    "scikit_learn": __import__("sklearn").__version__,
    "shap": getattr(shap, "__version__", "not installed"),
    "scipy": getattr(scipy, "__version__", "not installed"),
}
print("Environment")
print("-" * 40)
for k, v in versions.items():
    print(f"{k:>14}: {v}")
print(f"{'RANDOM_STATE':>14}: {RANDOM_STATE}")

# --- Helper utilities (definitions only; nothing runs until you call run_workflow) ---

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    def clean(col: str) -> str:
        c = col.strip().lower()
        c = re.sub(r"[^a-z0-9]+", "_", c)
        c = re.sub(r"_+", "_", c).strip("_")
        return c
    df = df.copy()
    df.columns = [clean(c) for c in df.columns]
    return df

def load_dataset(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".csv", ".txt"]:
        try:
            df = pd.read_csv(path, engine="python")
        except Exception:
            df = pd.read_csv(path)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    elif ext in [".parquet", ".pq"]:
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    return standardize_columns(df)

def try_parse_duration_to_minutes(s: pd.Series) -> pd.Series:
    """
    Attempt to coerce strings like '1h 30m', '90 min', '00:45:00', '1:30', '5400s' into minutes.
    If numeric, assume minutes already.
    """
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").astype(float)

    s_clean = s.astype(str).str.strip().str.lower()

    def parse_one(x: str) -> float:
        if x in ("", "nan", "none", "na"):
            return np.nan
        # hh:mm[:ss]
        m = re.match(r"^\s*(\d{1,2}):(\d{2})(?::(\d{2}))?\s*$", x)
        if m:
            hh = int(m.group(1))
            mm = int(m.group(2))
            ss = int(m.group(3) or 0)
            return hh * 60 + mm + ss / 60.0
        # iso-like PT#H#M#S
        m = re.match(r"^p?t?(?:(\d+)\s*h(?:ours?)?)?\s*(?:(\d+)\s*m(?:in(?:utes?)?)?)?\s*(?:(\d+)\s*s(?:ec(?:onds?)?)?)?$", x)
        if m and any(g for g in m.groups()):
            h = int(m.group(1) or 0)
            m_ = int(m.group(2) or 0)
            s_ = int(m.group(3) or 0)
            return h * 60 + m_ + s_ / 60.0
        # 90m, 90min, 5400s, 1.5h
        m = re.match(r"^(\d+(\.\d+)?)\s*([smhd]|mins?|minutes?|hours?)?$", x)
        if m:
            val = float(m.group(1))
            unit = (m.group(3) or "m").strip()
            if unit in ["s", "sec", "secs", "second", "seconds"]:
                return val / 60.0
            if unit in ["m", "min", "mins", "minute", "minutes"]:
                return val
            if unit in ["h", "hour", "hours", "d", "day", "days"]:
                if unit.startswith("h"):
                    return val * 60.0
                if unit.startswith("d"):
                    return val * 60.0 * 24.0
        # Fallback numeric
        try:
            return float(x)
        except Exception:
            return np.nan

    return s_clean.map(parse_one).astype(float)

def compact_data_dictionary(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    rows = []
    n = len(df)
    for col in df.columns:
        s = df[col]
        dtype = str(s.dtype)
        n_missing = s.isna().sum()
        pct_missing = 100.0 * n_missing / n if n > 0 else np.nan
        nunique = s.nunique(dropna=True)
        ex_vals = s.dropna().unique()[:3]
        ex_vals = ", ".join(map(lambda x: str(x)[:30], ex_vals))
        constant = nunique <= 1
        # quasi-constant: top value frequency > 99%
        try:
            top_freq = s.value_counts(dropna=True, normalize=True).iloc[0] if nunique > 0 else 0.0
        except Exception:
            top_freq = 0.0
        quasi_constant = (top_freq > 0.99) and (not constant)
        high_card_cat = (dtype in ["object", "string", "category"]) and (nunique > max(100, n * 0.5))
        rows.append({
            "feature": col,
            "dtype": dtype,
            "pct_missing": round(pct_missing, 3),
            "n_unique": int(nunique),
            "example_values": ex_vals,
            "constant": bool(constant),
            "quasi_constant": bool(quasi_constant),
            "high_cardinality_categorical": bool(high_card_cat),
            "is_target": col == target_col,
        })
    dd = pd.DataFrame(rows).sort_values("feature").reset_index(drop=True)
    return dd

def find_datetime_columns(df: pd.DataFrame) -> typing.List[str]:
    dt_cols = []
    for col in df.columns:
        if any(tok in col for tok in ["date", "time", "timestamp", "datetime"]):
            try:
                pd.to_datetime(df[col], errors="raise")
                dt_cols.append(col)
            except Exception:
                continue
    return dt_cols

def leakage_name_heuristic(col: str) -> bool:
    substrs = [
        "wait", "in_2h", "target", "label", "outcome", "future", "after", "post",
        "delay", "response", "lag0", "leak", "horizon", "duration"
    ]
    return any(s in col for s in substrs)

def scan_leakage_columns(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    flags = {}
    y = df[target_col]
    # name-based flags
    for col in df.columns:
        if col == target_col:
            flags[col] = False
            continue
        flags[col] = leakage_name_heuristic(col)
    # correlation-based for numeric
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != target_col]
    for col in num_cols:
        try:
            corr = df[[col, target_col]].dropna().corr().iloc[0,1]
            if abs(corr) > 0.95:
                flags[col] = True
        except Exception:
            pass
    return pd.DataFrame({"feature": list(flags.keys()), "leakage_flag": list(flags.values())})

class RareLabelGrouper:
    """
    Simple transformer to group infrequent categories into 'RARE' based on min_freq threshold (proportion).
    """
    def __init__(self, min_freq: float = 0.01):
        self.min_freq = min_freq
        self.category_maps_ = {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        self.category_maps_ = {}
        for col in X.columns:
            vc = X[col].value_counts(dropna=False, normalize=True)
            keep = set(vc[vc >= self.min_freq].index.tolist())
            self.category_maps_[col] = keep
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            keep = self.category_maps_.get(col, set())
            X[col] = X[col].where(X[col].isin(keep), other="RARE")
        return X

    def get_feature_names_out(self, input_features=None):
        return input_features

def build_preprocessor(df: pd.DataFrame, target_col: str, leakage_df: pd.DataFrame, rare_min_freq: float = 0.01):
    features = [c for c in df.columns if c != target_col]
    leakage_set = set(leakage_df.loc[leakage_df["leakage_flag"], "feature"].tolist())
    features = [f for f in features if f not in leakage_set]

    num_features = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    cat_features = [c for c in features if c not in num_features]

    numeric_pipe = Pipeline(steps=[
        ("imputer", __import__("sklearn").impute.SimpleImputer(strategy="median")),
        ("scaler", RobustScaler())
    ])
    categorical_pipe = Pipeline(steps=[
        ("imputer", __import__("sklearn").impute.SimpleImputer(strategy="most_frequent")),
        ("rare", RareLabelGrouper(min_freq=rare_min_freq)),
        ("ohe", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse=False))
    ])

    pre = ColumnTransformer(transformers=[
        ("num", numeric_pipe, num_features),
        ("cat", categorical_pipe, cat_features),
    ], remainder="drop", verbose_feature_names_out=False)

    feature_info = {
        "all_features": features,
        "numeric": num_features,
        "categorical": cat_features,
        "leakage_removed": list(leakage_set),
    }
    return pre, feature_info

def stratified_regression_split(X: pd.DataFrame, y: pd.Series, test_size=0.2, random_state=RANDOM_STATE):
    # stratify via quantile bins on y (10 bins)
    try:
        bins = pd.qcut(y, q=10, duplicates="drop")
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=bins
        )
    except Exception:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    return X_tr, X_te, y_tr, y_te

def time_based_split(df: pd.DataFrame, dt_cols: typing.List[str], target_col: str, test_size=0.2):
    # use the max across datetime-like columns as a proxy timeline
    ts = None
    for c in dt_cols:
        try:
            ts_c = pd.to_datetime(df[c], errors="coerce")
            ts = ts_c if ts is None else ts.combine_first(ts_c)
        except Exception:
            continue
    if ts is None:
        raise ValueError("Could not form a proper datetime index from the detected datetime columns.")
    order = ts.sort_values().index
    cut = int(math.floor((1 - test_size) * len(order)))
    train_idx = order[:cut]
    test_idx = order[cut:]
    return train_idx, test_idx

def compute_univariate_numeric_stats(X_num: pd.DataFrame, y: pd.Series):
    rows = []
    for col in X_num.columns:
        x = X_num[col]
        dfc = pd.concat([x, y], axis=1).dropna()
        if len(dfc) < 3:
            rows.append({"feature": col, "pearson": np.nan, "spearman": np.nan, "covariance": np.nan})
            continue
        pear = dfc.corr(method="pearson").iloc[0,1]
        spear = dfc.corr(method="spearman").iloc[0,1]
        cov = np.cov(dfc.iloc[:,0], dfc.iloc[:,1])[0,1]
        rows.append({"feature": col, "pearson": pear, "spearman": spear, "covariance": cov})
    return pd.DataFrame(rows)

def fit_preprocessor_on_train(pre, X_train):
    pre_fit = pre.fit(X_train)
    feat_names = None
    try:
        feat_names = pre_fit.get_feature_names_out()
    except Exception:
        feat_names = [f"f{i}" for i in range(pre_fit.transform(X_train).shape[1])]
    return pre_fit, list(feat_names)

def compute_mutual_information(pre_fit, X_train, y_train, feature_names):
    Xtr = pre_fit.transform(X_train)
    mi = mutual_info_regression(Xtr, y_train, random_state=RANDOM_STATE)
    return pd.DataFrame({"feature": feature_names, "mutual_information": mi})

def compute_vif_iterative(X_num_imputed_scaled: pd.DataFrame, max_vif=10.0):
    """
    Compute VIF by regressing each feature on the rest using LinearRegression.
    Iteratively drop the feature with the highest VIF > max_vif.
    """
    cols = list(X_num_imputed_scaled.columns)
    dropped = []
    def _vif_for_matrix(Xmat: np.ndarray):
        vifs = []
        for j in range(Xmat.shape[1]):
            yj = Xmat[:, j]
            Xj = np.delete(Xmat, j, axis=1)
            if Xj.shape[1] == 0:
                vifs.append(1.0)
                continue
            lr = LinearRegression()
            lr.fit(Xj, yj)
            r2 = lr.score(Xj, yj)
            vif = np.inf if (1 - r2) <= 1e-12 else 1.0 / (1.0 - r2)
            vifs.append(vif)
        return np.array(vifs)

    Xmat = X_num_imputed_scaled.values
    vifs = _vif_for_matrix(Xmat)
    history = [dict(zip(cols, vifs))]

    while True:
        max_idx = int(np.nanargmax(vifs))
        max_val = vifs[max_idx]
        if max_val <= max_vif or len(cols) <= 1:
            break
        drop_col = cols[max_idx]
        dropped.append((drop_col, float(max_val)))
        # drop the column and recompute
        keep_mask = np.ones(len(cols), dtype=bool)
        keep_mask[max_idx] = False
        cols = [c for i, c in enumerate(cols) if keep_mask[i]]
        Xmat = Xmat[:, keep_mask]
        vifs = _vif_for_matrix(Xmat)
        history.append(dict(zip(cols, vifs)))

    final_vifs = pd.DataFrame([history[0]]).T.reset_index()
    final_vifs.columns = ["feature", "vif_initial"]
    last = pd.DataFrame([history[-1]]).T.reset_index()
    last.columns = ["feature", "vif_final"]
    out = pd.merge(final_vifs, last, on="feature", how="outer")
    out["dropped_by_vif"] = ~out["feature"].isin(cols)
    out = out.sort_values("vif_initial", ascending=False).reset_index(drop=True)
    return out, dropped, cols

def build_full_linear_pipeline(preprocessor, model):
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("post_scale", StandardScaler(with_mean=False)),
        ("model", model)
    ])

def evaluate_models(X_train, y_train, X_test, y_test, preprocessor, feature_names_pre):
    results = []
    models = {
        "OLS": LinearRegression(),
        "RidgeCV": RidgeCV(alphas=np.logspace(-3, 3, 25), cv=5),
        "LassoCV": LassoCV(alphas=np.logspace(-3, 3, 25), cv=5, random_state=RANDOM_STATE, max_iter=5000),
        "ElasticNetCV": ElasticNetCV(alphas=np.logspace(-3, 3, 25), l1_ratio=[0.1, 0.5, 0.9], cv=5, random_state=RANDOM_STATE, max_iter=5000),
    }
    scoring = {
        "rmse": make_scorer(lambda yt, yp: mean_squared_error(yt, yp, squared=False)),
        "mae": make_scorer(mean_absolute_error),
        "r2": make_scorer(r2_score),
    }
    coef_tables = {}

    for name, est in models.items():
        pipe = build_full_linear_pipeline(preprocessor, est)
        cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=RANDOM_STATE)
        # cross-validated metrics on training
        cv_scores = {m: cross_val_score(pipe, X_train, y_train, cv=cv, scoring=sc, n_jobs=None) for m, sc in scoring.items()}
        # fit on train, eval on test
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        test_rmse = mean_squared_error(y_test, y_pred, squared=False)
        test_mae = mean_absolute_error(y_test, y_pred)
        test_r2 = r2_score(y_test, y_pred)

        results.append({
            "model": name,
            "cv_rmse_mean": float(np.mean(cv_scores["rmse"])),
            "cv_rmse_std": float(np.std(cv_scores["rmse"])),
            "cv_mae_mean": float(np.mean(cv_scores["mae"])),
            "cv_mae_std": float(np.std(cv_scores["mae"])),
            "cv_r2_mean": float(np.mean(cv_scores["r2"])),
            "cv_r2_std": float(np.std(cv_scores["r2"])),
            "test_rmse": float(test_rmse),
            "test_mae": float(test_mae),
            "test_r2": float(test_r2),
        })

        # standardized coefficients (after post_scale)
        try:
            # Recover feature names after preprocessor + post_scale
            pipe_pre = Pipeline(steps=[("preprocessor", preprocessor), ("post_scale", StandardScaler(with_mean=False))])
            pipe_pre.fit(X_train, y_train)
            Xtr = pipe_pre.transform(X_train)
            if hasattr(pipe_pre, "get_feature_names_out"):
                feat_names = list(pipe_pre.get_feature_names_out())
            else:
                feat_names = feature_names_pre
            coef = getattr(pipe.named_steps["model"], "coef_", None)
            if coef is not None:
                coef_tables[name] = pd.DataFrame({
                    "feature": feat_names,
                    "coef": coef.ravel()
                }).assign(model=name).sort_values("coef", key=lambda s: np.abs(s), ascending=False)
        except Exception:
            pass

    metrics_df = pd.DataFrame(results).sort_values("test_rmse").reset_index(drop=True)
    return metrics_df, coef_tables

def nonlinear_and_permutation(X_train, y_train, X_test, y_test, preprocessor):
    grid = {
        "rf__n_estimators": [200],
        "rf__max_depth": [None, 6, 12],
        "rf__min_samples_leaf": [1, 5],
    }
    rf = RandomForestRegressor(random_state=RANDOM_STATE, n_estimators=200, n_jobs=None)
    rf_pipe = Pipeline(steps=[("preprocessor", preprocessor), ("rf", rf)])
    rf_gs = GridSearchCV(rf_pipe, param_grid=grid, cv=5, scoring="neg_root_mean_squared_error")
    rf_gs.fit(X_train, y_train)

    gbdt_grid = {
        "gb__n_estimators": [200],
        "gb__max_depth": [3, 6],
        "gb__min_samples_leaf": [1, 5],
        "gb__learning_rate": [0.05, 0.1]
    }
    gb = GradientBoostingRegressor(random_state=RANDOM_STATE)
    gb_pipe = Pipeline(steps=[("preprocessor", preprocessor), ("gb", gb)])
    gb_gs = GridSearchCV(gb_pipe, param_grid=gbdt_grid, cv=5, scoring="neg_root_mean_squared_error")
    gb_gs.fit(X_train, y_train)

    # choose best by CV
    best_est = rf_gs if rf_gs.best_score_ >= gb_gs.best_score_ else gb_gs
    best_name = "RandomForestRegressor" if best_est is rf_gs else "GradientBoostingRegressor"

    # Permutation importance on held-out
    best_model = best_est.best_estimator_
    best_model.fit(X_train, y_train)
    perm = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, scoring="neg_root_mean_squared_error")
    # Get feature names after preprocessing
    try:
        feat_names = best_model.named_steps["preprocessor"].get_feature_names_out()
    except Exception:
        Xtr = best_model.named_steps["preprocessor"].transform(X_train)
        feat_names = [f"f{i}" for i in range(Xtr.shape[1])]

    perm_df = pd.DataFrame({
        "feature": feat_names,
        "perm_importance_mean": perm.importances_mean,
        "perm_importance_std": perm.importances_std,
    }).sort_values("perm_importance_mean", ascending=False).reset_index(drop=True)

    return best_name, best_model, perm_df, rf_gs, gb_gs

def consolidated_ranking(corr_df, mi_df, coef_tables, perm_df, vif_df, leakage_df):
    # Prepare signals dictionary: ranks normalized to [0,1] with 1 = best (most predictive)
    signals = {}

    # |Pearson|, |Spearman|, |Covariance| (abs)
    if corr_df is not None and len(corr_df):
        tmp = corr_df.set_index("feature").copy()
        for col in ["pearson", "spearman", "covariance"]:
            if col in tmp:
                s = tmp[col].abs().rank(method="average", ascending=False)
                signals[col] = (s - s.min()) / (s.max() - s.min() + 1e-12)

    # MI
    if mi_df is not None and len(mi_df):
        s = mi_df.set_index("feature")["mutual_information"].rank(method="average", ascending=False)
        signals["mutual_information"] = (s - s.min()) / (s.max() - s.min() + 1e-12)

    # Ridge |coef|
    if "RidgeCV" in coef_tables:
        s = coef_tables["RidgeCV"].set_index("feature")["coef"].abs().rank(method="average", ascending=False)
        signals["ridge_coef_abs"] = (s - s.min()) / (s.max() - s.min() + 1e-12)

    # Lasso presence (binary): 1 if non-zero
    if "LassoCV" in coef_tables:
        ct = coef_tables["LassoCV"].copy()
        ct = ct.set_index("feature")["coef"].abs()
        presence = (ct > 1e-12).astype(float)
        signals["lasso_presence"] = presence

    # Permutation importance
    if perm_df is not None and len(perm_df):
        s = perm_df.set_index("feature")["perm_importance_mean"].rank(method="average", ascending=False)
        signals["permutation"] = (s - s.min()) / (s.max() - s.min() + 1e-12)

    # Merge all signal series
    all_features = set()
    for s in signals.values():
        all_features.update(s.index.tolist())
    df = pd.DataFrame({"feature": sorted(list(all_features))}).set_index("feature")
    for name, s in signals.items():
        df[name] = s.reindex(df.index)

    # Add VIF flags
    if vif_df is not None and len(vif_df):
        df = df.merge(vif_df.set_index("feature")[["vif_initial", "vif_final", "dropped_by_vif"]], left_index=True, right_index=True, how="left")

    # Add leakage flags
    if leakage_df is not None and len(leakage_df):
        df = df.merge(leakage_df.set_index("feature")[["leakage_flag"]], left_index=True, right_index=True, how="left")

    # Weights (renormalize if some signals missing)
    weights = {
        # correlations total 0.25
        "pearson": 0.125,
        "spearman": 0.125,
        "covariance": 0.0,  # covariance scale varies; keep 0 by default (still listed individually)
        "mutual_information": 0.15,
        "ridge_coef_abs": 0.15,
        "lasso_presence": 0.10,
        "permutation": 0.25,
        # SHAP would be 0.10 if present (skipped in this consolidated helper)
    }
    present = [k for k in weights if k in df.columns]
    wsum = sum(weights[k] for k in present)
    norm_weights = {k: weights[k] / (wsum if wsum > 0 else 1.0) for k in present}

    df["consolidated_score"] = 0.0
    for k, w in norm_weights.items():
        df["consolidated_score"] = df["consolidated_score"] + df[k].fillna(0.0) * w

    # Penalize features dropped by VIF slightly
    if "dropped_by_vif" in df.columns:
        df.loc[df["dropped_by_vif"] == True, "consolidated_score"] *= 0.9

    df = df.sort_values("consolidated_score", ascending=False).reset_index()
    return df

def save_figure(fig, name):
    os.makedirs("/mnt/data/figures", exist_ok=True)
    path = f"/mnt/data/figures/{name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=160)
    plt.close(fig)
    return path

def plot_bar(sorted_df, feature_col, value_col, top_n=20, title="", xlabel="", ylabel=""):
    dfp = sorted_df[[feature_col, value_col]].dropna().sort_values(value_col, ascending=False).head(top_n)
    fig = plt.figure()
    plt.barh(dfp[feature_col][::-1], dfp[value_col][::-1])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    path = save_figure(fig, re.sub(r'[^a-z0-9_]+', '_', title.lower()) or value_col)
    return path

def run_workflow(file_path: str, target_col: str = "wait_time_in_2h"):
    # 1) Load
    df_raw = load_dataset(file_path)
    if target_col not in df_raw.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset. Available columns: {list(df_raw.columns)[:30]}...")

    # target coercion
    y_raw = try_parse_duration_to_minutes(df_raw[target_col])
    if y_raw.isna().all() or y_raw.nunique(dropna=True) <= 1:
        raise ValueError("Target is missing or constant after coercion; please verify the WAIT_TIME_IN_2H column.")
    df = df_raw.copy()
    df[target_col] = y_raw

    # 2) Data audit
    dd = compact_data_dictionary(df, target_col=target_col)

    # Deduplicate rows / columns
    n_before = len(df)
    df = df.drop_duplicates()
    n_dup_rows = n_before - len(df)
    # Perfect duplicate columns
    dup_cols = []
    cols = df.columns.tolist()
    seen = {}
    for c in cols:
        v = tuple(df[c].fillna("__NA__").astype(str).tolist())
        if v in seen:
            dup_cols.append(c)
        else:
            seen[v] = c
    df = df.drop(columns=dup_cols) if dup_cols else df

    # leakage scan
    leakage_df = scan_leakage_columns(df, target_col)

    # target distribution plot + stats
    y = df[target_col]
    fig = plt.figure()
    plt.hist(y.dropna().values, bins=50, density=True)
    try:
        y.dropna().plot(kind="kde")
    except Exception:
        pass
    plt.title(f"Target distribution: {target_col}")
    plt.xlabel("minutes")
    plt.ylabel("density")
    path_target_hist = save_figure(fig, "target_distribution")

    target_stats = y.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_frame(name="value")

    # 3) Train/validation protocol
    dt_cols = find_datetime_columns(df.drop(columns=[target_col]))
    if len(dt_cols):
        train_idx, test_idx = time_based_split(df, dt_cols, target_col, test_size=0.2)
        train_df = df.loc[train_idx]
        test_df = df.loc[test_idx]
    else:
        X = df.drop(columns=[target_col])
        y = df[target_col]
        X_train, X_test, y_train, y_test = stratified_regression_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    # 4) Preprocessing pipelines
    preprocessor, feat_info = build_preprocessor(train_df, target_col, leakage_df)
    pre_fit, pre_feat_names = fit_preprocessor_on_train(preprocessor, X_train)

    # 5) Univariate screening (numeric only for correlations/covariance)
    num_cols = feat_info["numeric"]
    # Impute+scale numeric only for this step
    num_imputer = __import__("sklearn").impute.SimpleImputer(strategy="median")
    scaler = RobustScaler()
    Xnum_tr = pd.DataFrame(scaler.fit_transform(num_imputer.fit_transform(X_train[num_cols])), columns=num_cols, index=X_train.index) if len(num_cols) else pd.DataFrame(index=X_train.index)

    corr_df = compute_univariate_numeric_stats(Xnum_tr, y_train) if len(num_cols) else pd.DataFrame(columns=["feature","pearson","spearman","covariance"])

    # MI on preprocessed features
    mi_df = compute_mutual_information(pre_fit, X_train, y_train, pre_feat_names)

    # Visuals for univariate
    figs = []
    if len(corr_df):
        figs.append(plot_bar(corr_df.assign(abs_pearson=corr_df["pearson"].abs()).sort_values("abs_pearson", ascending=False),
                             "feature", "abs_pearson", 20, title="Top |Pearson| correlations", xlabel="|Pearson r|", ylabel="feature"))
        figs.append(plot_bar(corr_df.assign(abs_spearman=corr_df["spearman"].abs()).sort_values("abs_spearman", ascending=False),
                             "feature", "abs_spearman", 20, title="Top |Spearman| correlations", xlabel="|Spearman ρ|", ylabel="feature"))
    if len(mi_df):
        figs.append(plot_bar(mi_df.sort_values("mutual_information", ascending=False),
                             "feature", "mutual_information", 30, title="Top Mutual Information (train)", xlabel="MI", ylabel="feature"))

    # 6) Multicollinearity diagnostics (numeric subset)
    vif_df = pd.DataFrame(columns=["feature","vif_initial","vif_final","dropped_by_vif"])
    vif_dropped_seq = []
    vif_kept = num_cols
    if len(num_cols) >= 2:
        # iterative VIF
        vif_df, vif_dropped_seq, vif_kept = compute_vif_iterative(Xnum_tr, max_vif=10.0)
        # Visuals
        fig = plt.figure()
        plt.barh(vif_df["feature"][::-1], vif_df["vif_initial"][::-1])
        plt.title("VIF before pruning (numeric)")
        plt.xlabel("VIF")
        plt.ylabel("feature")
        figs.append(save_figure(fig, "vif_before"))
        fig = plt.figure()
        plt.barh(vif_df["feature"][::-1], vif_df["vif_final"].fillna(0)[::-1])
        plt.title("VIF after pruning (numeric)")
        plt.xlabel("VIF")
        plt.ylabel("feature")
        figs.append(save_figure(fig, "vif_after"))

    # 7) Baseline & regularized linear models
    metrics_df, coef_tables = evaluate_models(X_train, y_train, X_test, y_test, preprocessor, pre_feat_names)

    # Coefficient magnitude plots for top 30 (Ridge & Lasso)
    coef_figs = []
    for m in ["RidgeCV", "LassoCV"]:
        if m in coef_tables:
            top = coef_tables[m].head(30)
            fig = plt.figure()
            plt.barh(top["feature"][::-1], top["coef"][::-1])
            plt.title(f"{m}: Top |standardized coefficients|")
            plt.xlabel("coefficient")
            plt.ylabel("feature")
            coef_figs.append(save_figure(fig, f"{m.lower()}_top_coefs"))
    figs.extend(coef_figs)

    # 8) Nonlinear + permutation importance
    best_tree_name, best_tree_model, perm_df, rf_gs, gb_gs = nonlinear_and_permutation(X_train, y_train, X_test, y_test, preprocessor)
    figs.append(plot_bar(perm_df, "feature", "perm_importance_mean", 25, title=f"Permutation importance ({best_tree_name})", xlabel="mean importance", ylabel="feature"))

    # SHAP (optional)
    shap_df = None
    shap_fig = None
    if SHAP_AVAILABLE:
        try:
            # Extract tree model and preprocessor
            pre = best_tree_model.named_steps["preprocessor"]
            model = best_tree_model.named_steps.get("rf") or best_tree_model.named_steps.get("gb")
            Xtr = pre.transform(X_train)
            feat_names = pre.get_feature_names_out()
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(Xtr)
            # Beeswarm plot
            shap_fig = plt.figure()
            shap.summary_plot(shap_vals, Xtr, feature_names=feat_names, show=False)
            shap_path = save_figure(shap_fig, "shap_beeswarm")
            # Bar plot
            shap_fig2 = plt.figure()
            shap.summary_plot(shap_vals, Xtr, feature_names=feat_names, plot_type="bar", show=False)
            shap_bar_path = save_figure(shap_fig2, "shap_bar")
            # Aggregate mean |SHAP|
            shap_abs = np.abs(shap_vals).mean(axis=0)
            shap_df = pd.DataFrame({"feature": feat_names, "shap_mean_abs": shap_abs}).sort_values("shap_mean_abs", ascending=False)
        except Exception:
            SHAP_AVAILABLE_LOCAL = False

    # 9) Consolidated ranking
    ranking_df = consolidated_ranking(corr_df, mi_df, coef_tables, perm_df, vif_df, leakage_df)

    # 10) Model performance snapshot
    # Already have metrics_df; select best by test_rmse
    best_linear = metrics_df.sort_values("test_rmse").iloc[0]["model"]
    # Residuals vs fitted (best linear)
    best_linear_est = build_full_linear_pipeline(preprocessor, {
        "OLS": LinearRegression(),
        "RidgeCV": RidgeCV(alphas=np.logspace(-3, 3, 25), cv=5),
        "LassoCV": LassoCV(alphas=np.logspace(-3, 3, 25), cv=5, random_state=RANDOM_STATE, max_iter=5000),
        "ElasticNetCV": ElasticNetCV(alphas=np.logspace(-3, 3, 25), l1_ratio=[0.1, 0.5, 0.9], cv=5, random_state=RANDOM_STATE, max_iter=5000),
    }[best_linear])
    best_linear_est.fit(X_train, y_train)
    yhat_train = best_linear_est.predict(X_train)
    yhat_test = best_linear_est.predict(X_test)

    # Residual plots
    fig = plt.figure()
    plt.scatter(yhat_test, y_test - yhat_test, s=10, alpha=0.7)
    plt.axhline(0, linestyle="--")
    plt.title(f"Residuals vs Fitted ({best_linear})")
    plt.xlabel("Fitted values")
    plt.ylabel("Residuals")
    figs.append(save_figure(fig, "residuals_vs_fitted_test"))

    if SCIPY_AVAILABLE:
        fig = plt.figure()
        stats.probplot((y_test - yhat_test), dist="norm", plot=plt)
        plt.title("Q-Q plot of residuals (test)")
        figs.append(save_figure(fig, "qq_plot_residuals_test"))

    # Learning curve for best tree model
    train_sizes, train_scores, valid_scores = learning_curve(best_tree_model, pd.concat([X_train, X_test]), pd.concat([y_train, y_test]), cv=5, scoring="neg_root_mean_squared_error", train_sizes=np.linspace(0.1, 1.0, 5), random_state=RANDOM_STATE)
    fig = plt.figure()
    plt.plot(train_sizes, -np.mean(train_scores, axis=1), marker="o", label="train RMSE")
    plt.plot(train_sizes, -np.mean(valid_scores, axis=1), marker="o", label="CV RMSE")
    plt.title(f"Learning curve ({best_tree_name})")
    plt.xlabel("Training examples")
    plt.ylabel("RMSE")
    plt.legend()
    figs.append(save_figure(fig, "learning_curve_best_tree"))

    # 11) Partial dependence for top 3 numeric features
    top_features = ranking_df["feature"].tolist()
    # Map back to original numeric features (pre-OHE)
    numeric_candidates = [f for f in feat_info["numeric"] if f in X_train.columns]
    pdp_targets = [f for f in top_features if f in numeric_candidates][:3]
    pdp_paths = []
    if len(pdp_targets):
        fig = plt.figure()
        PartialDependenceDisplay.from_estimator(best_tree_model, X_test, features=pdp_targets, kind="both", grid_resolution=50)
        plt.suptitle("Partial Dependence (with ICE) - top numeric drivers")
        pdp_paths.append(save_figure(fig, "pdp_ice_top_numeric"))

    # 12) Exports
    os.makedirs("/mnt/data", exist_ok=True)
    ranked_path = "/mnt/data/ranked_features.csv"
    metrics_path = "/mnt/data/model_metrics.csv"
    dd_path = "/mnt/data/data_dictionary.csv"

    ranking_df.to_csv(ranked_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    dd.to_csv(dd_path, index=False)

    artifacts = {
        "data_dictionary": dd_path,
        "target_hist": path_target_hist,
        "univariate_figs": figs,
        "ranked_features_csv": ranked_path,
        "model_metrics_csv": metrics_path,
        "vif_drops_sequence": vif_dropped_seq,
        "leakage_removed": feat_info["leakage_removed"],
        "dup_rows_removed": int(n_dup_rows),
        "dup_cols_removed": dup_cols,
    }
    return {
        "ranking_df": ranking_df.head(30),
        "drop_list": {
            "leakage": feat_info["leakage_removed"],
            "constant_or_quasi": dd.query("constant or quasi_constant")["feature"].tolist(),
            "high_cardinality_categorical": dd.query("high_cardinality_categorical and not is_target")["feature"].tolist(),
            "vif_dropped": vif_df.query("dropped_by_vif")["feature"].tolist() if len(vif_df) else [],
        },
        "watchlist": ranking_df.tail(int(min(30, len(ranking_df) // 10)))["feature"].tolist(),
        "metrics_df": metrics_df,
        "artifacts": artifacts
    }

print("\nSetup complete. Call run_workflow(<path_to_your_dataset>) after you upload your file.")

result = run_workflow("/mnt/data/your_file.ext")