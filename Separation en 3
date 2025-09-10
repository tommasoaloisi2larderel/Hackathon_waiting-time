import pandas as pd
from pathlib import Path

# --- Réglages ---
# chemin du fichier source
src = Path("/mnt/data/waiting_times_train.csv")   # <— remplace par ton chemin si besoin
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
