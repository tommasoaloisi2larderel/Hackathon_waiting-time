import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBRegressor



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


def add_season_column(X):
    """
    Ajoute une colonne 'season' au DataFrame X selon le mois (X['month']):
        - 0 = hiver (déc, janv, fév)
        - 1 = automne (sept, oct, nov)
        - 2 = printemps (mars, avr, mai)
        - 3 = été (juin, juil, août)

    Paramètres :
        X (pd.DataFrame) : doit contenir une colonne 'month' (1 à 12)

    Retour :
        X (pd.DataFrame) avec une nouvelle colonne 'season'
    """
    X = X.copy()

    if "month" not in X.columns:
        raise ValueError("La colonne 'month' est absente du DataFrame.")

    # Mapping mois -> saison
    def month_to_season(month):
        if month in [12, 1, 2]:
            return 0  # hiver
        elif month in [9, 10, 11]:
            return 1  # automne
        elif month in [3, 4, 5]:
            return 2  # printemps
        elif month in [6, 7, 8]:
            return 3  # été
        else:
            return -1  # cas invalide

    X["season"] = X["month"].apply(month_to_season)
    return X

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# ======================
#    Run one scenario
# ======================
def add_covid_column_from_year_month(X):
    """
    Ajoute une colonne 'covid' selon 'year' et 'month':
        - 1 si dans période COVID : 2020, 2021, ou janv-mai 2022
        - 0 sinon
    """
    X = X.copy()
    
    if "year" not in X.columns or "month" not in X.columns:
        raise ValueError("Les colonnes 'year' et 'month' sont requises.")

    def is_covid_period(y, m):
        if y in [2020, 2021]:
            return 1
        if y == 2022 and m in [1, 2, 3, 4, 5]:
            return 1
        return 0

    X["covid"] = X.apply(lambda row: is_covid_period(row["year"], row["month"]), axis=1)
    return X

def add_vacances_column(X):
    """
    Ajoute une colonne 'vacances' au DataFrame X selon le mois (X['month']):
        - 1 = période de vacances scolaires (approximatif)
        - 0 = hors vacances

    Les mois considérés comme vacances :
        - février (2) → vacances d'hiver
        - avril (4)   → vacances de printemps
        - juillet (7), août (8) → vacances d'été
        - décembre (12) → vacances de Noël

    Paramètres :
        X (pd.DataFrame) : doit contenir une colonne 'month'

    Retour :
        X (pd.DataFrame) avec une nouvelle colonne 'vacances'
    """
    X = X.copy()

    if "month" not in X.columns:
        raise ValueError("La colonne 'month' est absente du DataFrame.")

    VACANCES_MOIS = {2, 4, 7, 8, 12}

    X["vacances"] = X["month"].apply(lambda m: 1 if m in VACANCES_MOIS else 0)
    return X


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
         "feels_like", "clouds_all","pressure","humidity"
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
    X_all=add_season_column(X_all)
    X_test=add_season_column(X_test)
    add_covid_column_from_year_month(X_all)
    add_covid_column_from_year_month(X_test)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # IMPORTANT : aligne le test sur les colonnes finales du train
    X_test = align_columns(X_test, X_all.columns)

    # Split train/val
    X_tr, X_val, y_tr, y_val = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

    # Standardisation (inutile pour RF mais on garde l’API cohérente)
    X_tr_s, X_val_s, mu, sigma = standardize_train_test(X_tr, X_val)
    # Train (RandomForest)
    A=[]
   
    for k in range (35,70):
      model = XGBRegressor(
      n_estimators=300,
      learning_rate=0.05,
      max_depth=7,
      min_child_weight=3,
      subsample=0.9,
      colsample_bytree=0.7,
      gamma=0.1,
      reg_alpha=1,
      reg_lambda=1,
      random_state=k,
      n_jobs=-1
)
      model.fit(X_tr_s, y_tr)


      
      



    # Test pred
      X_test_s = (X_test - mu) / sigma
      preds_test = model.predict(X_test_s.values).ravel()
      A.append(preds_test)


          

    s=0
    FIN=[]
    for i in range (len(A[0])):
      for j in range (len(A)):
        s=s+A[j][i]
      FIN.append(s/len(A))
      s=0
    preds_test=FIN
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

    final_path = Path("ok.csv")
    final_df.to_csv(final_path, index=False)
    print(f"\n🎉 Fichier final concaténé: {final_path} ({len(final_df)} lignes)")


if __name__ == "__main__":
    main()


