"""
Entrena modelos de "picks conjuntos" - los que necesitan saber las carreras
de AMBOS equipos en el mismo juego, no solo lo que permitio un pitcher:

  - favorito 1-3 entradas: que equipo anota mas en las primeras 3 entradas
    (equivale al lado +0.5/-0.5 de un handicap de esas entradas)
  - favorito 1-5 entradas: lo mismo para las primeras 5 entradas
  - total 1er inning / 1-3 / 1-5: over/under de carreras COMBINADAS de
    ambos equipos, contra una linea de referencia calculada de los datos
    (la mediana historica, redondeada a .5 para que nunca empate)

No hace falta recolectar nada nuevo de la API: training_data.csv ya trae,
por cada gamePk, una fila del abridor local y otra del visitante, y
"label_runs_1to3"/"label_runs_1to5" de la fila del abridor X = carreras que
anoto el equipo RIVAL de X en esas entradas. Cruzando esas dos filas por
gamePk se arman las carreras reales de ambos equipos en ese juego, sin
llamadas nuevas.

Uso:
    python ml_train_matchup.py --data training_data.csv
"""

import argparse
import json
import math
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml_train import FEATURES, FEATURES_1TO3, FEATURES_1TO5, evaluate, time_split

ALL_PITCHER_FEATURES = sorted(set(FEATURES) | set(FEATURES_1TO3) | set(FEATURES_1TO5))


class ConstantBaselineClassifier:
    """'Sin modelo' para los targets nuevos: no existe un baseline ingenuo
    previo (como scoreless_rate_trailing) para 'quien anota mas' o 'total de
    carreras combinado', asi que el punto de comparacion es predecir siempre
    la tasa historica fija de la clase positiva."""

    def __init__(self, p):
        self.p = p

    def predict_proba(self, X):
        p = np.full(len(X), self.p)
        return np.column_stack([1 - p, p])


ConstantBaselineClassifier.__module__ = "ml_train_matchup"


def build_matchup_dataset(df):
    """Empareja las dos filas (abridor local / visitante) de cada gamePk en
    una sola fila por juego, con las features de ambos pitchers prefijadas
    (home_pitcher_*/away_pitcher_*) y las carreras reales de ambos equipos
    derivadas de los labels ya existentes."""
    has_full = "label_runs_full" in df.columns

    rows = []
    ties_1to3 = ties_1to5 = ties_full = 0
    for gamePk, g in df.groupby("gamePk"):
        home = g[g["is_home"] == 1]
        away = g[g["is_home"] == 0]
        if len(home) != 1 or len(away) != 1:
            continue
        home = home.iloc[0]
        away = away.iloc[0]

        row = {"gamePk": gamePk, "date": home["date"]}
        for f in ALL_PITCHER_FEATURES:
            row[f"home_pitcher_{f}"] = home.get(f)
            row[f"away_pitcher_{f}"] = away.get(f)

        # El abridor visitante enfrenta a los bateadores locales: lo que el
        # sufrio ES lo que el equipo local anoto, y viceversa.
        home_runs_1st, away_runs_1st = away["label_runs_1st"], home["label_runs_1st"]
        home_runs_1to3, away_runs_1to3 = away["label_runs_1to3"], home["label_runs_1to3"]
        home_runs_1to5, away_runs_1to5 = away["label_runs_1to5"], home["label_runs_1to5"]

        row["label_1st_total"] = home_runs_1st + away_runs_1st
        row["label_1to3_total"] = home_runs_1to3 + away_runs_1to3
        row["label_1to5_total"] = home_runs_1to5 + away_runs_1to5

        if home_runs_1to3 == away_runs_1to3:
            row["label_1to3_favorite"] = None
            ties_1to3 += 1
        else:
            row["label_1to3_favorite"] = int(home_runs_1to3 > away_runs_1to3)

        if home_runs_1to5 == away_runs_1to5:
            row["label_1to5_favorite"] = None
            ties_1to5 += 1
        else:
            row["label_1to5_favorite"] = int(home_runs_1to5 > away_runs_1to5)

        if has_full:
            home_runs_full, away_runs_full = away["label_runs_full"], home["label_runs_full"]
            if home_runs_full == away_runs_full:
                # Un juego de MLB no puede terminar en empate (se juegan
                # entradas extra hasta que alguien gane) - si esto pasa es
                # un dato incompleto/juego suspendido, se descarta igual.
                row["label_full_favorite"] = None
                ties_full += 1
            else:
                row["label_full_favorite"] = int(home_runs_full > away_runs_full)

        rows.append(row)

    print(f"Juegos emparejados (ambos abridores presentes en el dataset): {len(rows)} "
          f"de {df['gamePk'].nunique()} gamePks originales")
    print(f"Empates/incompletos descartados - favorito 1-3: {ties_1to3} | favorito 1-5: {ties_1to5}"
          + (f" | ganador del juego: {ties_full}" if has_full else ""))
    return pd.DataFrame(rows)


def add_over_labels(df):
    """Agrega label_<rango>_over (1 si el total combinado supero la linea)
    calculando la linea como la mediana historica redondeada al .5 mas
    cercano hacia abajo, para que nunca caiga justo en un valor posible
    (evita 'push')."""
    thresholds = {}
    for total_col in ["label_1st_total", "label_1to3_total", "label_1to5_total"]:
        median = df[total_col].median()
        threshold = math.floor(median) + 0.5
        thresholds[total_col] = threshold
        over_col = total_col.replace("_total", "_over")
        df[over_col] = (df[total_col] > threshold).astype(int)
    return df, thresholds


def train_matchup_classifier(df, test_frac, model_out, features, label, title, extra_bundle=None):
    d = df.dropna(subset=features + [label]).copy()
    if len(d) < 30:
        print(f"\n{title}: muy pocos juegos emparejados ({len(d)}), se omite este modelo.")
        return None

    train_df, test_df = time_split(d, test_frac)
    X_train, y_train = train_df[features], train_df[label]
    X_test, y_test = test_df[features], test_df[label]

    logreg = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    logreg.fit(X_train, y_train)
    prob_logreg = logreg.predict_proba(X_test)[:, 1]

    hgb = HistGradientBoostingClassifier(max_depth=3, random_state=0)
    hgb.fit(X_train, y_train)
    prob_hgb = hgb.predict_proba(X_test)[:, 1]

    base_rate = train_df[label].mean()
    baseline = ConstantBaselineClassifier(base_rate)
    prob_naive = baseline.predict_proba(X_test)[:, 1]

    results = [
        evaluate("Baseline constante (tasa historica)", y_test, prob_naive),
        evaluate("Regresion logistica", y_test, prob_logreg),
        evaluate("Gradient Boosting", y_test, prob_hgb),
    ]
    results_df = pd.DataFrame(results)
    print(f"\n--- {title}: {len(d)} juegos ({len(train_df)} train / {len(test_df)} test) ---")
    print(results_df.to_string(index=False))

    best_name = results_df.sort_values("brier").iloc[0]["modelo"]
    print(f"Mejor modelo por Brier score: {best_name}")

    if best_name == "Gradient Boosting":
        model_to_save = hgb
    elif best_name.startswith("Baseline"):
        model_to_save = baseline
    else:
        model_to_save = logreg

    bundle = {"model": model_to_save, "features": features, "model_name": best_name}
    if extra_bundle:
        bundle.update(extra_bundle)
    joblib.dump(bundle, model_out)
    print(f"Modelo guardado en {model_out}")

    with open(model_out + ".metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    best_row = results_df.sort_values("brier").iloc[0]
    return {"title": title, "model_out": model_out, "best_name": best_name,
            "brier": best_row["brier"], "n_games": len(d)}


def log_matchup_history(n_games_total, results_list, thresholds, history_path="training_history_matchup.csv"):
    row = {"timestamp": datetime.now().isoformat(timespec="seconds"), "n_games_emparejados": n_games_total}
    for r in results_list:
        if r is None:
            continue
        key = os.path.splitext(os.path.basename(r["model_out"]))[0]
        row[f"modelo_{key}"] = r["best_name"]
        row[f"brier_{key}"] = r["brier"]
        row[f"n_{key}"] = r["n_games"]
    for col, threshold in thresholds.items():
        row[f"linea_{col}"] = threshold
    file_exists = os.path.exists(history_path)
    pd.DataFrame([row]).to_csv(history_path, mode="a", header=not file_exists, index=False)
    return row


def main():
    parser = argparse.ArgumentParser(
        description="Entrena los modelos de picks conjuntos (favorito y total de carreras 1-3/1-5, total 1er inning)")
    parser.add_argument("--data", default="training_data.csv")
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--history-path", default="training_history_matchup.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    if "label_runs_1st" not in df.columns:
        print("ADVERTENCIA: el dataset no tiene la columna 'label_runs_1st' (se agrego recientemente a "
              "ml_data.py) - hace falta recolectar de nuevo (sin --skip-collect) antes de poder entrenar "
              "estos modelos. Se omite este entrenamiento.")
        return

    matchup_df = build_matchup_dataset(df)
    if matchup_df.empty:
        print("No se pudieron emparejar juegos (se necesitan ambas filas home/away por gamePk). Nada que entrenar.")
        return
    matchup_df, thresholds = add_over_labels(matchup_df)

    feats_1st = [f"home_pitcher_{f}" for f in FEATURES] + [f"away_pitcher_{f}" for f in FEATURES]
    feats_1to3 = [f"home_pitcher_{f}" for f in FEATURES_1TO3] + [f"away_pitcher_{f}" for f in FEATURES_1TO3]
    feats_1to5 = [f"home_pitcher_{f}" for f in FEATURES_1TO5] + [f"away_pitcher_{f}" for f in FEATURES_1TO5]

    results = [
        train_matchup_classifier(matchup_df, args.test_frac, "model_1to3_favorite.joblib", feats_1to3,
                                  "label_1to3_favorite", "Favorito 1-3 entradas (equipo local anota mas)"),
        train_matchup_classifier(matchup_df, args.test_frac, "model_1to3_total.joblib", feats_1to3,
                                  "label_1to3_over", f"Total de carreras 1-3 (over {thresholds['label_1to3_total']})",
                                  extra_bundle={"threshold": thresholds["label_1to3_total"]}),
        train_matchup_classifier(matchup_df, args.test_frac, "model_1to5_favorite.joblib", feats_1to5,
                                  "label_1to5_favorite", "Favorito 1-5 entradas (equipo local anota mas)"),
        train_matchup_classifier(matchup_df, args.test_frac, "model_1to5_total.joblib", feats_1to5,
                                  "label_1to5_over", f"Total de carreras 1-5 (over {thresholds['label_1to5_total']})",
                                  extra_bundle={"threshold": thresholds["label_1to5_total"]}),
        train_matchup_classifier(matchup_df, args.test_frac, "model_1st_total.joblib", feats_1st,
                                  "label_1st_over", f"Total de carreras 1er inning (over {thresholds['label_1st_total']})",
                                  extra_bundle={"threshold": thresholds["label_1st_total"]}),
    ]

    if "label_full_favorite" in matchup_df.columns:
        results.append(train_matchup_classifier(
            matchup_df, args.test_frac, "model_full_favorite.joblib", feats_1to5,
            "label_full_favorite", "Money line (equipo local gana el juego completo)"))
    else:
        print("\nADVERTENCIA: el dataset no tiene 'label_runs_full' (se agrego recientemente a ml_data.py) - "
              "hace falta recolectar de nuevo (sin --skip-collect) para poder entrenar el money line. "
              "Se omite ese modelo esta vez, los demas se entrenan igual.")

    log_matchup_history(len(matchup_df), results, thresholds, args.history_path)
    print(f"\nGuardado en {args.history_path} para ver la tendencia entre reentrenamientos.")


if __name__ == "__main__":
    import ml_train_matchup
    ml_train_matchup.main()
