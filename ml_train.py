"""
Entrena y evalua un modelo que predice la probabilidad de que un abridor deje
0 carreras en el 1er inning en su PROXIMA apertura (el bet "-0.5 carreras
1er inning"), usando el dataset de ml_data.py.

Compara el modelo contra un baseline ingenuo: usar directamente la tasa de
"0 carreras" del pitcher en sus ultimas aperturas (scoreless_rate_trailing)
como si fuera la probabilidad, SIN ningun modelo. Esto es exactamente el
numero que ya mostraba el programa original (ej. "9/10 = 90%"). Si el modelo
de ML no le gana a este baseline, no vale la pena usarlo.

Uso:
    python ml_train.py --data training_data.csv --model-out model.joblib
"""

import argparse
import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (brier_score_loss, roc_auc_score, accuracy_score,
                              mean_absolute_error, mean_squared_error)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "n_prior_starts", "scoreless_rate_trailing", "era_to_date", "whip_to_date",
    "fip_to_date", "ip_per_start_to_date", "is_home", "opponent_avg",
    "similar_avg_runs_1to3", "similar_avg_n",
    "opponent_form_avg_1to3", "opponent_form_runs_1to3",
    "game_temp_f", "game_wind_mph", "game_wind_effect", "game_is_indoor",
    "days_rest", "opponent_vs_hand_avg", "opponent_vs_hand_ops", "park_factor",
    "own_bullpen_ip_3d", "h2h_avg_runs_1to3", "h2h_n_games",
    "tto1_runs_trailing_avg", "tto1_pa_trailing_avg",
]
LABEL = "label_scoreless_1st"

FEATURES_1TO3 = FEATURES + ["runs_1to3_trailing_avg"]
LABEL_1TO3 = "label_runs_1to3"

FEATURES_1TO5 = FEATURES + ["runs_1to5_trailing_avg", "similar_avg_runs_1to5",
                            "opponent_form_avg_1to5", "opponent_form_runs_1to5"]
LABEL_1TO5 = "label_runs_1to5"


class NaiveBaselineRegressor:
    """Envoltorio para poder 'guardar' el baseline ingenuo como si fuera un
    modelo, cuando de verdad es el que gana (evita guardar por error un
    modelo real cuando en realidad el ganador fue el promedio simple)."""

    def __init__(self, col):
        self.col = col

    def predict(self, X):
        return X[self.col].values


class NaiveBaselineClassifier:
    def __init__(self, col):
        self.col = col

    def predict_proba(self, X):
        p = X[self.col].values
        return np.column_stack([1 - p, p])


# Fuerza el modulo "real" (ml_train), aunque este script se corra como
# __main__ - si no, joblib.load() desde otro script (mlb_first_inning_report.py,
# ml_predict.py) no encuentra la clase y truena con AttributeError.
NaiveBaselineRegressor.__module__ = "ml_train"
NaiveBaselineClassifier.__module__ = "ml_train"


def time_split(df, test_frac=0.2):
    df = df.sort_values("date").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_frac))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def log_training_history(n_rows, results_1st, results_1to3, results_1to5,
                          history_path="training_history.csv"):
    """Guarda una fila por cada corrida de entrenamiento, para poder ver con
    el tiempo si el modelo mejora al ir acumulando mas juegos/features (el
    'aprendizaje continuo' es esto: reentrenar periodicamente con mas datos,
    no que el modelo se actualice solo en tiempo real)."""
    def best_row(results, metric, minimize=True):
        df = pd.DataFrame(results)
        best = df.sort_values(metric, ascending=minimize).iloc[0]
        naive = df[df["modelo"].str.startswith("Baseline")].iloc[0]
        return best["modelo"], best[metric], naive[metric]

    best_1st, brier_1st, brier_1st_naive = best_row(results_1st, "brier")
    best_1to3, mae_1to3, mae_1to3_naive = best_row(results_1to3, "mae")
    best_1to5, mae_1to5, mae_1to5_naive = best_row(results_1to5, "mae")

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_rows": n_rows,
        "modelo_1er_inning": best_1st, "brier_1er_inning": brier_1st, "brier_1er_inning_baseline": brier_1st_naive,
        "modelo_1to3": best_1to3, "mae_1to3": mae_1to3, "mae_1to3_baseline": mae_1to3_naive,
        "modelo_1to5": best_1to5, "mae_1to5": mae_1to5, "mae_1to5_baseline": mae_1to5_naive,
    }
    file_exists = os.path.exists(history_path)
    df_row = pd.DataFrame([row])
    df_row.to_csv(history_path, mode="a", header=not file_exists, index=False)
    return row


def evaluate(name, y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "modelo": name,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "auc": round(roc_auc_score(y_true, y_prob), 4) if len(set(y_true)) > 1 else None,
        "brier": round(brier_score_loss(y_true, y_prob), 4),
    }


def evaluate_regression(name, y_true, y_pred):
    return {
        "modelo": name,
        "mae": round(mean_absolute_error(y_true, y_pred), 4),
        "rmse": round(mean_squared_error(y_true, y_pred) ** 0.5, 4),
    }


def print_feature_importance(model, X_test, y_test, features, scoring, title, n_repeats=10):
    """Importancia por permutacion (no el .feature_importances_ nativo, que
    HistGradientBoosting ni siquiera expone): mide cuanto empeora la metrica
    real del modelo al revolver cada columna, una a la vez, en el set de
    prueba. Funciona igual para el modelo lineal, el de gradient boosting o
    el baseline (aunque el baseline sale con todo en 0 porque ignora las
    demas columnas)."""
    if isinstance(model, (NaiveBaselineClassifier, NaiveBaselineRegressor)):
        print(f"\n{title}: se omite (el modelo ganador fue el baseline ingenuo, "
              f"no usa las demas features).")
        return
    try:
        result = permutation_importance(model, X_test, y_test, scoring=scoring,
                                         n_repeats=n_repeats, random_state=0)
    except Exception as e:
        print(f"\n{title}: no se pudo calcular ({e}).")
        return
    order = result.importances_mean.argsort()[::-1]
    print(f"\n{title} (caida en '{scoring}' al revolver cada columna; mas alto = pesa mas):")
    for i in order:
        print(f"  {features[i]:<28} {result.importances_mean[i]:+.4f}  (+/- {result.importances_std[i]:.4f})")


def print_calibration_table(y_true, y_prob, title, n_bins=5):
    """Divide las predicciones del set de prueba en quintiles (por probabilidad
    predicha) y compara la probabilidad promedio predicha vs. la tasa real
    observada en cada grupo. Un modelo bien calibrado tiene 'predicho' ~=
    'real' en cada fila; si un bin se desvia mucho, el modelo esta
    sobre/sub-estimando sistematicamente en ese rango."""
    df = pd.DataFrame({"y": y_true.values if hasattr(y_true, "values") else y_true, "p": y_prob})
    try:
        df["bin"] = pd.qcut(df["p"], q=min(n_bins, df["p"].nunique()), duplicates="drop")
    except ValueError:
        print(f"\n{title}: muy pocos valores distintos para armar bins de calibracion.")
        return
    grouped = df.groupby("bin", observed=True).agg(n=("y", "size"), predicho=("p", "mean"), real=("y", "mean"))
    print(f"\n{title} (predicho vs. real por quintil de probabilidad, set de prueba):")
    print(grouped.round(4).to_string())


def print_regression_calibration_table(y_true, y_pred, title, n_bins=5):
    df = pd.DataFrame({"y": y_true.values if hasattr(y_true, "values") else y_true, "p": y_pred})
    try:
        df["bin"] = pd.qcut(df["p"], q=min(n_bins, df["p"].nunique()), duplicates="drop")
    except ValueError:
        print(f"\n{title}: muy pocos valores distintos para armar bins de calibracion.")
        return
    grouped = df.groupby("bin", observed=True).agg(n=("y", "size"), predicho=("p", "mean"), real=("y", "mean"))
    print(f"\n{title} (carreras predichas vs. reales por quintil, set de prueba):")
    print(grouped.round(3).to_string())


def train_runs_regression(df, test_frac, model_out, features, label, naive_col, range_label):
    train_df, test_df = time_split(df, test_frac)
    X_train, y_train = train_df[features], train_df[label]
    X_test, y_test = test_df[features], test_df[label]

    linreg = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("reg", LinearRegression()),
    ])
    linreg.fit(X_train, y_train)
    pred_linreg = linreg.predict(X_test)

    hgb = HistGradientBoostingRegressor(max_depth=3, random_state=0)
    hgb.fit(X_train, y_train)
    pred_hgb = hgb.predict(X_test)

    pred_naive = test_df[naive_col].values

    results = [
        evaluate_regression("Baseline ingenuo (promedio trailing)", y_test, pred_naive),
        evaluate_regression("Regresion lineal", y_test, pred_linreg),
        evaluate_regression("Gradient Boosting", y_test, pred_hgb),
    ]
    results_df = pd.DataFrame(results)
    print(f"\n--- Carreras esperadas en entradas {range_label}: comparacion en el set de prueba ---")
    print(results_df.to_string(index=False))
    print("\nNota: 'mae'/'rmse' mas bajo = mejor (error promedio en carreras, 0 = perfecto).")

    best_name = results_df.sort_values("mae").iloc[0]["modelo"]
    print(f"\nMejor modelo (carreras {range_label}) por MAE: {best_name}")

    if best_name == "Gradient Boosting":
        model_to_save, pred_to_save = hgb, pred_hgb
    elif best_name.startswith("Baseline"):
        model_to_save, pred_to_save = NaiveBaselineRegressor(naive_col), pred_naive
    else:
        model_to_save, pred_to_save = linreg, pred_linreg
    joblib.dump({"model": model_to_save, "features": features, "model_name": best_name}, model_out)
    print(f"Modelo guardado en {model_out}")

    print_feature_importance(model_to_save, X_test, y_test, features, scoring="neg_mean_absolute_error",
                              title=f"Importancia de variables (carreras {range_label})")
    print_regression_calibration_table(y_test, pred_to_save, f"Calibracion (carreras {range_label})")
    return results


def main():
    parser = argparse.ArgumentParser(description="Entrena el modelo de 1er inning")
    parser.add_argument("--data", default="training_data.csv")
    parser.add_argument("--model-out", default="model.joblib")
    parser.add_argument("--model-1to3-out", default="model_1to3.joblib")
    parser.add_argument("--model-1to5-out", default="model_1to5.joblib")
    parser.add_argument("--test-frac", type=float, default=0.2)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    print(f"Dataset: {len(df)} filas, {df[LABEL].mean():.1%} tasa de 0-carreras-1er-inning")

    train_df, test_df = time_split(df, args.test_frac)
    print(f"Train: {len(train_df)} filas (hasta {train_df['date'].max()}) | "
          f"Test: {len(test_df)} filas (desde {test_df['date'].min()})")

    X_train, y_train = train_df[FEATURES], train_df[LABEL]
    X_test, y_test = test_df[FEATURES], test_df[LABEL]

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

    prob_naive = test_df["scoreless_rate_trailing"].values

    results = [
        evaluate("Baseline ingenuo (tasa trailing)", y_test, prob_naive),
        evaluate("Regresion logistica", y_test, prob_logreg),
        evaluate("Gradient Boosting", y_test, prob_hgb),
    ]
    results_df = pd.DataFrame(results)
    print("\n--- Comparacion en el set de prueba (mas reciente, no visto en entrenamiento) ---")
    print(results_df.to_string(index=False))
    print("\nNota: 'brier' mas bajo = mejor calibracion de probabilidad (0 = perfecto). "
          "'auc' mas alto = mejor separando casos 0-carreras vs. no.")

    best_name = results_df.sort_values("brier").iloc[0]["modelo"]
    print(f"\nMejor modelo por Brier score: {best_name}")

    if best_name == "Gradient Boosting":
        model_to_save, prob_to_save = hgb, prob_hgb
    elif best_name.startswith("Baseline"):
        model_to_save, prob_to_save = NaiveBaselineClassifier("scoreless_rate_trailing"), prob_naive
    else:
        model_to_save, prob_to_save = logreg, prob_logreg
    joblib.dump({"model": model_to_save, "features": FEATURES, "model_name": best_name}, args.model_out)
    print(f"Modelo guardado en {args.model_out}")

    print_feature_importance(model_to_save, X_test, y_test, FEATURES, scoring="neg_brier_score",
                              title="Importancia de variables (0 carreras 1er inning)")
    print_calibration_table(y_test, prob_to_save, "Calibracion (0 carreras 1er inning)")

    with open(args.model_out + ".metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    results_1to3 = train_runs_regression(df, args.test_frac, args.model_1to3_out,
                                          FEATURES_1TO3, LABEL_1TO3, "runs_1to3_trailing_avg", "1-3")
    with open(args.model_1to3_out + ".metrics.json", "w") as f:
        json.dump(results_1to3, f, indent=2)

    results_1to5 = train_runs_regression(df, args.test_frac, args.model_1to5_out,
                                          FEATURES_1TO5, LABEL_1TO5, "runs_1to5_trailing_avg", "1-5")
    with open(args.model_1to5_out + ".metrics.json", "w") as f:
        json.dump(results_1to5, f, indent=2)

    history_row = log_training_history(len(df), results, results_1to3, results_1to5)
    print("\n--- Guardado en training_history.csv (para ver progreso entre reentrenamientos) ---")
    print(history_row)


if __name__ == "__main__":
    # Se reimporta como modulo "ml_train" (en vez de correr como __main__)
    # para que las clases NaiveBaseline* se puedan des-picklear despues desde
    # otros scripts (joblib/pickle exige que el modulo del objeto sea
    # importable con ese mismo nombre, no solo "__main__").
    import ml_train
    ml_train.main()
