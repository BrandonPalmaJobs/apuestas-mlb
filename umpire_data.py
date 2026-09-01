"""
Construye un indice de tendencia por umpire de home plate: para cada umpire
que ha dirigido esta temporada, el promedio de carreras/ponches/bases por
bola POR JUEGO en los partidos que le tocaron, comparado contra el promedio
de la liga en el mismo periodo.

Es un PROXY, no la geometria real de zona de strike: una zona angosta
produce mas bases por bola y mas carreras; una zona amplia produce mas
ponches y menos carreras. No existe una fuente publica y verificada (sin
depender de scrapear un sitio de terceros) para la geometria exacta de cada
umpire, asi que este indice se calcula 100% con datos oficiales de MLB
Stats API (boxscore de cada juego: quien fue el umpire de home plate, y las
carreras/ponches/BB de picheo de ambos equipos en ese juego). El resultado
tambien refleja la calidad de los pitchers/bateadores de esos juegos en
particular, no solo al umpire - por eso se marca siempre como proxy en el
reporte, nunca como un numero definitivo.

Corre una sola vez por temporada (o cuando quieras refrescarlo): recorre
TODOS los juegos Final del rango de fechas pedido, con un request de
boxscore por juego. Para la temporada completa (~2000+ juegos) tarda un
buen rato - usa --start-date/--end-date para una prueba rapida.

Uso:
    python umpire_data.py --season 2026 --out umpire_tendency.csv
    python umpire_data.py --start-date 2026-07-01 --end-date 2026-07-05 --out sample.csv
"""

import argparse
import sys
import time
from datetime import date

import pandas as pd

import mlb_first_inning_report as m


def collect_umpire_games(season, start_date=None, end_date=None, verbose=True):
    start = start_date or f"{season}-01-01"
    end = end_date or date.today().isoformat()
    data = m.get_json("/schedule", {"sportId": 1, "startDate": start, "endDate": end})

    gamePks = []
    for d in data.get("dates", []):
        for g in d["games"]:
            if g["status"]["detailedState"] == "Final":
                gamePks.append(g["gamePk"])

    rows = []
    skipped_errors = 0
    t0 = time.time()
    for i, gamePk in enumerate(gamePks):
        # Un try/except por juego: de vez en cuando un boxscore viejo o
        # incompleto trae el umpire sin 'fullName' (o alguna otra forma
        # inesperada) - sin esto, un solo juego raro tumba una corrida de
        # toda la temporada que tarda varios minutos.
        try:
            box = m.get_json(f"/game/{gamePk}/boxscore")
            hp = next((o for o in box.get("officials", []) if o.get("officialType") == "Home Plate"), None)
            if not hp or not hp.get("official", {}).get("fullName"):
                continue
            home_pitch = box["teams"]["home"].get("teamStats", {}).get("pitching", {})
            away_pitch = box["teams"]["away"].get("teamStats", {}).get("pitching", {})
            rows.append({
                "gamePk": gamePk,
                "umpire_id": hp["official"]["id"],
                "umpire_name": hp["official"]["fullName"],
                "total_runs": (home_pitch.get("runs", 0) or 0) + (away_pitch.get("runs", 0) or 0),
                "total_k": (home_pitch.get("strikeOuts", 0) or 0) + (away_pitch.get("strikeOuts", 0) or 0),
                "total_bb": (home_pitch.get("baseOnBalls", 0) or 0) + (away_pitch.get("baseOnBalls", 0) or 0),
            })
        except Exception as e:
            skipped_errors += 1
            if verbose:
                print(f"  [salteado] gamePk {gamePk}: {e}", file=sys.stderr)
            continue
        if verbose and (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(gamePks)} juegos revisados ({time.time()-t0:.0f}s)...", file=sys.stderr)
    if skipped_errors and verbose:
        print(f"  Total de juegos salteados por error: {skipped_errors}", file=sys.stderr)
    return pd.DataFrame(rows)


def aggregate_umpire_tendency(games_df, min_games=3):
    if games_df.empty:
        return games_df
    league_avg_runs = games_df["total_runs"].mean()
    league_avg_k = games_df["total_k"].mean()
    league_avg_bb = games_df["total_bb"].mean()
    agg = games_df.groupby(["umpire_id", "umpire_name"]).agg(
        n_games=("gamePk", "count"),
        avg_runs=("total_runs", "mean"),
        avg_k=("total_k", "mean"),
        avg_bb=("total_bb", "mean"),
    ).reset_index()
    agg["runs_vs_league"] = agg["avg_runs"] / league_avg_runs
    agg["k_vs_league"] = agg["avg_k"] / league_avg_k
    agg["bb_vs_league"] = agg["avg_bb"] / league_avg_bb
    agg = agg[agg["n_games"] >= min_games]
    return agg.sort_values("n_games", ascending=False)


def main():
    parser = argparse.ArgumentParser(
        description="Indice de tendencia por umpire de home plate (proxy con datos oficiales de MLB)")
    parser.add_argument("--season", type=int, default=date.today().year)
    parser.add_argument("--start-date", default=None, help="Default: 1-enero de --season")
    parser.add_argument("--end-date", default=None, help="Default: hoy")
    parser.add_argument("--min-games", type=int, default=3,
                         help="Umpires con menos juegos que esto se omiten del CSV final (muestra muy chica)")
    parser.add_argument("--out", default="umpire_tendency.csv")
    args = parser.parse_args()

    games_df = collect_umpire_games(args.season, args.start_date, args.end_date)
    print(f"Juegos con umpire de home plate identificado: {len(games_df)}")

    tendency_df = aggregate_umpire_tendency(games_df, min_games=args.min_games)
    tendency_df.to_csv(args.out, index=False)
    print(f"Guardado {len(tendency_df)} umpires (con >= {args.min_games} juegos) en {args.out}")
    if len(tendency_df):
        print(tendency_df.head(15).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
