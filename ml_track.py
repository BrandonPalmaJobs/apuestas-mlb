"""
Guarda cada prediccion de ML que se hace (al correr el reporte) en un log
(predictions_log.csv), y despues permite evaluarlas contra el resultado
REAL del juego una vez que termino - para juzgar al modelo con muchos
juegos en vez de con una sola observacion.

Uso:
    # Se llama automaticamente desde mlb_first_inning_report.py, no hace
    # falta correrlo a mano para loguear.

    # Para evaluar lo logueado hasta ahora:
    python ml_track.py --evaluate
"""

import argparse
import os

import pandas as pd

import mlb_first_inning_report as m
from ml_data import runs_allowed_first_cached, runs_allowed_1to3_cached, runs_allowed_1to5_cached

LOG_COLUMNS = [
    "logged_date", "gamePk", "game_date", "team_name", "pitcher_id", "pitcher_name", "is_home",
    "pred_prob_1st", "model_1st", "baseline_prob_1st",
    "pred_runs_1to3", "model_1to3", "baseline_runs_1to3",
    "pred_runs_1to5", "model_1to5", "baseline_runs_1to5",
    "actual_scoreless_1st", "actual_runs_1to3", "actual_runs_1to5", "evaluated",
]


def log_prediction(row, log_path="predictions_log.csv"):
    """Agrega una fila al log, evitando duplicados por (gamePk, pitcher_id)."""
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        already = ((df["gamePk"] == row["gamePk"]) & (df["pitcher_id"] == row["pitcher_id"])).any()
        if already:
            return
    else:
        df = pd.DataFrame(columns=LOG_COLUMNS)

    full_row = {col: row.get(col) for col in LOG_COLUMNS}
    df = pd.concat([df, pd.DataFrame([full_row])], ignore_index=True)
    df.to_csv(log_path, index=False)


def build_log_row(team_name, pitcher_id, pitcher_name, is_home, gamePk, game_date, pitcher_report):
    if not pitcher_report:
        return None
    mlp = pitcher_report.get("ml_prediction") or {}
    mlp3 = pitcher_report.get("ml_prediction_1to3") or {}
    mlp5 = pitcher_report.get("ml_prediction_1to5") or {}
    feats = pitcher_report.get("ml_features") or {}
    return {
        "logged_date": m.date.today().isoformat(),
        "gamePk": gamePk, "game_date": game_date, "team_name": team_name,
        "pitcher_id": pitcher_id, "pitcher_name": pitcher_name, "is_home": int(is_home),
        "pred_prob_1st": mlp.get("prob"), "model_1st": mlp.get("model_name"),
        "baseline_prob_1st": feats.get("scoreless_rate_trailing"),
        "pred_runs_1to3": mlp3.get("runs"), "model_1to3": mlp3.get("model_name"),
        "baseline_runs_1to3": feats.get("runs_1to3_trailing_avg"),
        "pred_runs_1to5": mlp5.get("runs"), "model_1to5": mlp5.get("model_name"),
        "baseline_runs_1to5": feats.get("runs_1to5_trailing_avg"),
        "actual_scoreless_1st": None, "actual_runs_1to3": None, "actual_runs_1to5": None,
        "evaluated": False,
    }


def is_game_final(gamePk):
    data = m.get_json("/schedule", {"sportId": 1, "gamePk": gamePk})
    for d in data.get("dates", []):
        for g in d["games"]:
            if g["gamePk"] == gamePk:
                return g["status"]["detailedState"] == "Final"
    return False


def evaluate_log(log_path="predictions_log.csv"):
    if not os.path.exists(log_path):
        print(f"No existe {log_path} todavia - corre el reporte para algunos juegos primero.")
        return

    df = pd.read_csv(log_path)
    pending = df[df["evaluated"].fillna(False) != True]
    updated = 0
    for idx, row in pending.iterrows():
        gamePk = int(row["gamePk"])
        if not is_game_final(gamePk):
            continue
        is_home = bool(row["is_home"])
        actual_1st = runs_allowed_first_cached(gamePk, is_home)
        actual_1to3 = runs_allowed_1to3_cached(gamePk, is_home)
        actual_1to5 = runs_allowed_1to5_cached(gamePk, is_home)
        df.loc[idx, "actual_scoreless_1st"] = int(actual_1st == 0) if actual_1st is not None else None
        df.loc[idx, "actual_runs_1to3"] = actual_1to3
        df.loc[idx, "actual_runs_1to5"] = actual_1to5
        df.loc[idx, "evaluated"] = True
        updated += 1
    df.to_csv(log_path, index=False)
    print(f"Juegos recien evaluados: {updated}")

    done = df[df["evaluated"] == True]
    if done.empty:
        print("Todavia no hay juegos terminados en el log para evaluar.")
        return

    print(f"\n--- Resultados acumulados: {len(done)} aperturas evaluadas ---\n")

    d1 = done.dropna(subset=["pred_prob_1st", "actual_scoreless_1st"])
    if len(d1):
        pred_class = (d1["pred_prob_1st"] >= 0.5).astype(int)
        acc = (pred_class == d1["actual_scoreless_1st"]).mean()
        avg_pred = d1["pred_prob_1st"].mean()
        avg_actual = d1["actual_scoreless_1st"].mean()
        print(f"1er inning (0 carreras): {len(d1)} aperturas | accuracy {acc:.1%} | "
              f"prob. promedio predicha {avg_pred:.1%} vs. tasa real {avg_actual:.1%}")

    for label, pred_col, actual_col in [("1-3 entradas", "pred_runs_1to3", "actual_runs_1to3"),
                                          ("1-5 entradas", "pred_runs_1to5", "actual_runs_1to5")]:
        d = done.dropna(subset=[pred_col, actual_col])
        if len(d):
            mae = (d[pred_col] - d[actual_col]).abs().mean()
            print(f"{label} (carreras permitidas): {len(d)} aperturas | MAE {mae:.3f} | "
                  f"promedio predicho {d[pred_col].mean():.2f} vs. real {d[actual_col].mean():.2f}")

    print("\n--- Detalle ---")
    cols = ["game_date", "pitcher_name", "pred_prob_1st", "actual_scoreless_1st",
            "pred_runs_1to3", "actual_runs_1to3", "pred_runs_1to5", "actual_runs_1to5"]
    print(done[cols].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Evalua las predicciones logueadas contra el resultado real")
    parser.add_argument("--log-path", default="predictions_log.csv")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    if args.evaluate:
        evaluate_log(args.log_path)
    else:
        print("Usa --evaluate para comparar el log contra los resultados reales.")


if __name__ == "__main__":
    main()
