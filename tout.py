import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


import pandas as pd
from pathlib import Path

# POUR LE TRAIN
# chemin du fichier source
src = Path("waiting_times_train.csv")   # <— remplace par ton chemin si besoin
# dossier de sortie
out_dir = src.parent

# --- Chargement ---
df = pd.read_csv(src)

# --- Construction des masques ---
has_parade_1   = df["TIME_TO_PARADE_1"].notna()
has_parade_2   = df["TIME_TO_PARADE_2"].notna()
has_night_show = df["TIME_TO_NIGHT_SHOW"].notna()

# 1) Fichier : Parade 1 et/ou Night Show, mais PAS Parade 2
df_parade1_night = df[(has_parade_1 | has_night_show) & (~has_parade_2)]

# 2) Fichier : Parade 1 + Parade 2 + Night Show (tous présents)
df_all_three = df[has_parade_1 & has_parade_2 & has_night_show]

# 3) Fichier : Aucun des trois événements présents
df_none = df[~(has_parade_1 | has_parade_2 | has_night_show)]

# --- Sauvegarde ---
(df_parade1_night
 .to_csv(out_dir / "with_parade1_nightshow.csv", index=False))
(df_all_three
 .to_csv(out_dir / "with_all_events.csv", index=False))
(df_none
 .to_csv(out_dir / "with_no_events.csv", index=False))

print("Fichiers créés :")
print(out_dir / "with_parade1_nightshow.csv")
print(out_dir / "with_all_events.csv")
print(out_dir / "with_no_events.csv")

# chemin du fichier source
src = Path("waiting_times_X_test_val.csv")   # <— remplace par ton chemin si besoin
# dossier de sortie
out_dir = src.parent

# --- Chargement ---
df = pd.read_csv(src)

# --- Construction des masques ---
has_parade_1   = df["TIME_TO_PARADE_1"].notna()
has_parade_2   = df["TIME_TO_PARADE_2"].notna()
has_night_show = df["TIME_TO_NIGHT_SHOW"].notna()

# 1) Fichier : Parade 1 et/ou Night Show, mais PAS Parade 2
df_parade1_night = df[(has_parade_1 | has_night_show) & (~has_parade_2)]

# 2) Fichier : Parade 1 + Parade 2 + Night Show (tous présents)
df_all_three = df[has_parade_1 & has_parade_2 & has_night_show]

# 3) Fichier : Aucun des trois événements présents
df_none = df[~(has_parade_1 | has_parade_2 | has_night_show)]

# --- Sauvegarde ---
(df_parade1_night
 .to_csv(out_dir / "with_parade1_nightshow_test.csv", index=False))
(df_all_three
 .to_csv(out_dir / "with_all_events_test.csv", index=False))
(df_none
 .to_csv(out_dir / "with_no_events_test.csv", index=False))

print("Fichiers créés :")
print(out_dir / "with_parade1_nightshow_test.csv")
print(out_dir / "with_all_events_test.csv")
print(out_dir / "with_no_events_test.csv")



# ======================
#    LinearGD Model (conservé si besoin)
# ======================
class LinearGD:
    def __init__(self, lr=0.01, n_iter=1000, batch_size=None, l2=0.0, tol=1e-6, verbose=False):
        self.lr = lr
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.l2 = l2
        self.tol = tol
        self.verbose = verbose
        self.w = None
        self.b = None
        self.history = []

    def fit(self, X, y, X_val=None, y_val=None, early_stopping_rounds=None):
        n, d = X.shape
        rng = np.random.default_rng(42)

        self.w = rng.normal(scale=0.01, size=(d, 1))
        self.b = 0.0
        if self.batch_size is None:
            self.batch_size = n

        best_val_loss = np.inf
        rounds_no_improve = 0
        best_weights = (self.w.copy(), self.b)

        for it in range(self.n_iter):
            indices = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                batch_idx = indices[start:start + self.batch_size]
                Xb = X[batch_idx]
                yb = y[batch_idx]

                preds = Xb.dot(self.w) + self.b
                error = preds - yb

                grad_w = (2.0 / Xb.shape[0]) * (Xb.T.dot(error)) + 2.0 * self.l2 * self.w
                grad_b = (2.0 / Xb.shape[0]) * np.sum(error)

                self.w -= self.lr * grad_w
                self.b -= self.lr * grad_b

            preds_all = X.dot(self.w) + self.b
            train_loss = np.mean((preds_all - y) ** 2) + self.l2 * np.sum(self.w ** 2)
            self.history.append(train_loss)

            if X_val is not None and y_val is not None:
                val_preds = X_val.dot(self.w) + self.b
                val_loss = np.mean((val_preds - y_val) ** 2)

                if val_loss < best_val_loss - self.tol:
                    best_val_loss = val_loss
                    rounds_no_improve = 0
                    best_weights = (self.w.copy(), self.b)
                else:
                    rounds_no_improve += 1

                if early_stopping_rounds is not None and rounds_no_improve >= early_stopping_rounds:
                    if self.verbose:
                        print(f"Early stopping at iter {it}, best val_mse={best_val_loss:.6f}")
                    self.w, self.b = best_weights
                    return

    def predict(self, X):
        return X.dot(self.w) + self.b


# ======================
#    Utils & Features
# ======================
def safe_merge_on_datetime(df_left, df_right):
    df_left = df_left.copy()
    df_right = df_right.copy()
    df_left["DATETIME"] = pd.to_datetime(df_left["DATETIME"])
    df_right["DATETIME"] = pd.to_datetime(df_right["DATETIME"])
    return pd.merge(df_left, df_right, on="DATETIME", how="left")

def feature_engineer(df):
    df = df.copy()

    # datetime features
    df["DATETIME"] = pd.to_datetime(df["DATETIME"])
    df["year"] = df["DATETIME"].dt.year
    df["month"] = df["DATETIME"].dt.month
    df["dayofweek"] = df["DATETIME"].dt.weekday
    df["hour"] = df["DATETIME"].dt.hour
    df["minute"] = df["DATETIME"].dt.minute
    df["time_min"] = df["hour"] * 60 + df["minute"]

    # cyclic features
    df["time_sin"] = np.sin(2 * np.pi * df["time_min"] / 1440.0)
    df["time_cos"] = np.cos(2 * np.pi * df["time_min"] / 1440.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)

    # attraction encoding
    if "ENTITY_DESCRIPTION_SHORT" in df.columns:
        dummies = pd.get_dummies(df["ENTITY_DESCRIPTION_SHORT"], prefix="ENT")
        df = pd.concat([df.drop(columns=["ENTITY_DESCRIPTION_SHORT"]), dummies], axis=1)

    # fill weather columns if present
    for c in ["rain_1h", "pressure", "temp", "humidity", "wind_speed",
              "dew_point", "feels_like", "clouds_all"]:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    df = df.drop(columns=["hour", "minute"], errors="ignore")
    df = df.fillna(0)
    return df

def standardize_train_test(X_train, X_test):
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0).replace(0, 1)
    X_train_s = (X_train - mu) / sigma
    X_test_s = (X_test - mu) / sigma
    return X_train_s.values, X_test_s.values, mu, sigma

def align_columns(df, columns):
    """Assure que df possède exactement 'columns', en ajoutant les manquants à 0 et en ordonnant."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        for c in missing:
            df[c] = 0.0
    return df[columns]

def drop_event_columns(df, keep=("TIME_TO_PARADE_1","TIME_TO_PARADE_2","TIME_TO_NIGHT_SHOW")):
    """Supprime des features les colonnes d'événements qui ne doivent PAS être utilisées."""
    all_evt = {"TIME_TO_PARADE_1","TIME_TO_PARADE_2","TIME_TO_NIGHT_SHOW"}
    to_drop = list(all_evt - set(keep))
    return df.drop(columns=to_drop, errors="ignore")

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# AJOUT : utilitaire pour retirer des features non désirées
def drop_unwanted_features(X, unwanted=(), drop_prefixes=(), drop_contains=()):
    """
    Supprime de X :
      - toute colonne listée dans `unwanted`
      - toute colonne dont le nom commence par l'un des `drop_prefixes`
      - toute colonne contenant l'un des fragments de `drop_contains`
    """
    cols = set(X.columns)
    to_drop = set([c for c in unwanted if c in cols])

    for p in drop_prefixes:
        to_drop.update([c for c in cols if c.startswith(p)])

    for frag in drop_contains:
        to_drop.update([c for c in cols if frag in c])

    return X.drop(columns=sorted(to_drop), errors="ignore")
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# ======================
#    Run one scenario
# ======================
def run_scenario(
    train_path,
    test_path,
    keep_event_cols,     # tuple des colonnes d'event à conserver comme features
    key_label,           # étiquette pour la sortie
    weather_path="weather_data.csv",
    target="WAIT_TIME_IN_2H",
    lr=0.01, n_iter=2000, batch_size=256, l2=1e-3, early_stopping_rounds=50, verbose=True
):
    print(f"\n=== Scenario: {key_label} ===")
    train_raw = pd.read_csv(train_path)
    test_raw  = pd.read_csv(test_path)
    weather   = pd.read_csv(weather_path)

    # Merge météo
    train_df = safe_merge_on_datetime(train_raw, weather)
    test_df  = safe_merge_on_datetime(test_raw, weather)

    # Features de base
    train_fe = feature_engineer(train_df)
    test_fe  = feature_engineer(test_df)

    # Cible
    if target not in train_fe.columns:
        raise ValueError(f"Colonne cible {target} absente du train: {train_path}")

    # y 1D pour RandomForest
    y_all = train_fe[target].astype(float).values.ravel()

    # Gestion des colonnes d'events selon le scénario
    train_fe_use = drop_event_columns(train_fe, keep=keep_event_cols)
    test_fe_use  = drop_event_columns(test_fe,  keep=keep_event_cols)

    # Sélection des features numériques hors target
    X_all = train_fe_use.drop(columns=[target], errors="ignore").select_dtypes(include=[np.number])
    X_test = test_fe_use.select_dtypes(include=[np.number])

    # >>>>>>>>> Choisis ici ce que tu veux enlever (exemples) <<<<<<<<<
    UNWANTED = {
        # exemples météo :
         "feels_like", "clouds_all", "pressure",
        # exemples autres :
        # "ADJUST_CAPACITY", "DOWNTIME", "CURRENT_WAIT_TIME",
    }
    DROP_PREFIXES = (
        # exemple : enlever tous les one-hot d’attractions ENT_*
        # "ENT_",
    )
    DROP_CONTAINS = (
        # exemple : enlever toutes les features contenant ces fragments
        # "_lag", "_rolling",
    )

    # Appliquer le drop au train et au test
    X_all  = drop_unwanted_features(X_all,  unwanted=UNWANTED, drop_prefixes=DROP_PREFIXES, drop_contains=DROP_CONTAINS)
    X_test = drop_unwanted_features(X_test, unwanted=UNWANTED, drop_prefixes=DROP_PREFIXES, drop_contains=DROP_CONTAINS)

    # IMPORTANT : aligne le test sur les colonnes finales du train
    X_test = align_columns(X_test, X_all.columns)

    # Split train/val
    X_tr, X_val, y_tr, y_val = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

    # Standardisation (inutile pour RF mais on garde l’API cohérente)
    X_tr_s, X_val_s, mu, sigma = standardize_train_test(X_tr, X_val)

    # Train (RandomForest)
    model = XGBRegressor(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
    model.fit(X_tr_s, y_tr)

    # Eval
    preds_val = model.predict(X_val_s)
    rmse_val = np.sqrt(mean_squared_error(y_val, preds_val))
    print(f"[{key_label}] Validation RMSE: {rmse_val:.4f}")

    # Test pred
    X_test_s = (X_test - mu) / sigma
    preds_test = model.predict(X_test_s.values).ravel()

    # Préparer sortie test
    out = test_raw.copy()
    out["y_pred"] = preds_test
    out["KEY"] = "Validation"

    # Optionnel: retirer quelques colonnes pour un CSV plus "léger"
    cols_to_drop = [
        "ADJUST_CAPACITY", "DOWNTIME", "CURRENT_WAIT_TIME",
        "TIME_TO_PARADE_1", "TIME_TO_PARADE_2", "TIME_TO_NIGHT_SHOW"
    ]
    out_light = out.drop(columns=cols_to_drop, errors="ignore")

    # Sauvegarde par scénario
    scen_out_path = Path(f"predictions_{key_label}.csv")
    out_light.to_csv(scen_out_path, index=False)
    print(f"[{key_label}] ✅ Sauvegardé: {scen_out_path}")

    return out_light


# ======================
#    Main: 3 scénarios + concat finale
# ======================
def main():
    # Chemins des fichiers par scénario
    scenarios = [
        # (train_path, test_path, keep_event_cols, key_label)
        ("with_no_events.csv",              "with_no_events_test.csv",              tuple(),                                "NO_EVENTS"),
        ("with_all_events.csv",             "with_all_events_test.csv",             ("TIME_TO_PARADE_1","TIME_TO_PARADE_2","TIME_TO_NIGHT_SHOW"), "ALL_EVENTS"),
        ("with_parade1_nightshow.csv",      "with_parade1_nightshow_test.csv",      ("TIME_TO_PARADE_1","TIME_TO_NIGHT_SHOW"),                   "P1_NIGHT"),
    ]

    outputs = []
    for train_path, test_path, keep_cols, key in scenarios:
        out_df = run_scenario(
            train_path=train_path,
            test_path=test_path,
            keep_event_cols=keep_cols,
            key_label=key,
            weather_path="weather_data.csv",
            target="WAIT_TIME_IN_2H",
            lr=0.01, n_iter=2000, batch_size=256, l2=1e-3, early_stopping_rounds=50, verbose=True
        )
        outputs.append(out_df)

    # Concat finale
    final_df = pd.concat(outputs, axis=0, ignore_index=True)
    final_path = Path("waiting_times_predictions_ALL_forest.csv")
    final_df.to_csv(final_path, index=False)
    print(f"\n🎉 Fichier final concaténé: {final_path} ({len(final_df)} lignes)")


if __name__ == "__main__":
    main()


