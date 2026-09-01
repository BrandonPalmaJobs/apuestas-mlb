"""
Usa el modelo entrenado (ml_train.py) para predecir la probabilidad de que
el abridor probable de cada equipo deje 0 carreras en el 1er inning en su
PROXIMA apertura (el juego de hoy/proximo entre estos dos equipos).

Calcula las features exactamente igual que en el entrenamiento (ml_data.py):
usando solo las ultimas hasta 10 aperturas de cada pitcher, sin usar nada
del juego que se esta prediciendo (todavia no ha pasado).

Uso:
    python ml_predict.py "Reds" "Brewers" --model model.joblib
"""

import argparse
import os

import joblib
import pandas as pd

import mlb_first_inning_report as m
from ml_data import (runs_allowed_1to3_cached, runs_allowed_1to5_cached,
                      runs_allowed_first_cached, trailing_team_offense, live_weather_features,
                      bullpen_workload_asof, head_to_head_asof, runs_and_pa_in_tto1)


def compute_current_features(pitcher_id, opponent_team_id, is_home_next, season,
                              window=10, avg_similarity_threshold=0.015, weather_features=None,
                              opponent_avg_override=None, team_id=None, game_date=None):
    starts = m.get_all_starts(pitcher_id, season)
    prior = starts[-window:]
    if len(prior) < 3:
        return None, f"Muy pocas aperturas registradas esta temporada ({len(prior)}), no se puede predecir con confianza."

    sum_ip = sum_er = sum_h = sum_bb = sum_k = sum_hr = sum_hbp = 0.0
    scoreless = 0
    runs_1to3_list = []
    runs_1to5_list = []
    tto1_runs_list = []
    tto1_pa_list = []
    for ps in prior:
        st = ps["stat"]
        ip = m.ip_to_decimal(st.get("inningsPitched", "0.0"))
        sum_ip += ip
        sum_er += st.get("earnedRuns", 0)
        sum_h += st.get("hits", 0)
        sum_bb += st.get("baseOnBalls", 0)
        sum_k += st.get("strikeOuts", 0)
        sum_hr += st.get("homeRuns", 0)
        sum_hbp += st.get("hitBatsmen", 0)
        r1 = runs_allowed_first_cached(ps["game"]["gamePk"], ps["isHome"])
        if r1 == 0:
            scoreless += 1
        runs_1to3_list.append(runs_allowed_1to3_cached(ps["game"]["gamePk"], ps["isHome"]))
        runs_1to5_list.append(runs_allowed_1to5_cached(ps["game"]["gamePk"], ps["isHome"]))
        tto1_runs, tto1_pa = runs_and_pa_in_tto1(ps["game"]["gamePk"], pitcher_id, ps["isHome"])
        if tto1_runs is not None:
            tto1_runs_list.append(tto1_runs)
            tto1_pa_list.append(tto1_pa)

    n_prior = len(prior)
    era_to_date = (9 * sum_er / sum_ip) if sum_ip > 0 else None
    whip_to_date = ((sum_h + sum_bb) / sum_ip) if sum_ip > 0 else None
    fip_to_date = ((13 * sum_hr + 3 * (sum_bb + sum_hbp) - 2 * sum_k) / sum_ip +
                    m.DEFAULT_FIP_CONSTANT) if sum_ip > 0 else None
    ip_per_start_to_date = sum_ip / n_prior
    scoreless_rate = scoreless / n_prior
    runs_1to3_trailing_avg = sum(runs_1to3_list) / len(runs_1to3_list)
    runs_1to5_trailing_avg = sum(runs_1to5_list) / len(runs_1to5_list)
    tto1_runs_trailing_avg = (sum(tto1_runs_list) / len(tto1_runs_list)) if tto1_runs_list else None
    tto1_pa_trailing_avg = (sum(tto1_pa_list) / len(tto1_pa_list)) if tto1_pa_list else None

    opp_avg = opponent_avg_override if opponent_avg_override is not None else \
        m.get_team_season_avg(opponent_team_id, season)

    sim_runs_1to3 = []
    sim_runs_1to5 = []
    for ps in prior:
        if ps["isHome"] != is_home_next:
            continue
        p_opp_avg = m.get_team_season_avg(ps["opponent"]["id"], season)
        if opp_avg is None or p_opp_avg is None or abs(p_opp_avg - opp_avg) > avg_similarity_threshold:
            continue
        sim_runs_1to3.append(runs_allowed_1to3_cached(ps["game"]["gamePk"], ps["isHome"]))
        sim_runs_1to5.append(runs_allowed_1to5_cached(ps["game"]["gamePk"], ps["isHome"]))
    similar_avg_runs = (sum(sim_runs_1to3) / len(sim_runs_1to3)) if sim_runs_1to3 else None
    similar_avg_runs_1to5 = (sum(sim_runs_1to5) / len(sim_runs_1to5)) if sim_runs_1to5 else None
    similar_avg_n = len(sim_runs_1to3)

    opp_form_avg, opp_form_runs = trailing_team_offense(
        opponent_team_id, m.date.today().isoformat(), season, n=5, inning_end=3
    )
    opp_form_avg_1to5, opp_form_runs_1to5 = trailing_team_offense(
        opponent_team_id, m.date.today().isoformat(), season, n=5, inning_end=5
    )

    ref_date = game_date or m.date.today().isoformat()
    days_rest = m.get_days_rest(pitcher_id, season, ref_date)

    pitcher_hand = m.get_pitcher_hand(pitcher_id)
    vs_hand = m.get_team_vs_hand_split(opponent_team_id, season, pitcher_hand) if pitcher_hand else None
    opp_vs_hand_avg = vs_hand["avg"] if vs_hand else None
    opp_vs_hand_ops = vs_hand["ops"] if vs_hand else None

    park_factor = None
    home_team_for_park = team_id if is_home_next else opponent_team_id
    if home_team_for_park is not None:
        park_factor = m.compute_park_factor(home_team_for_park, season)

    own_bullpen_ip_3d = None
    if team_id is not None:
        own_bullpen_ip_3d = bullpen_workload_asof(team_id, ref_date, days=3)["bullpen_ip"]
    h2h = head_to_head_asof(opponent_team_id, pitcher_id, ref_date, [season, season - 1], inning_end=3)

    features = {
        "n_prior_starts": n_prior,
        "scoreless_rate_trailing": scoreless_rate,
        "runs_1to3_trailing_avg": runs_1to3_trailing_avg,
        "runs_1to5_trailing_avg": runs_1to5_trailing_avg,
        "tto1_runs_trailing_avg": tto1_runs_trailing_avg, "tto1_pa_trailing_avg": tto1_pa_trailing_avg,
        "era_to_date": era_to_date,
        "whip_to_date": whip_to_date,
        "fip_to_date": fip_to_date,
        "ip_per_start_to_date": ip_per_start_to_date,
        "is_home": int(is_home_next),
        "opponent_avg": opp_avg,
        "opponent_form_avg_1to3": opp_form_avg,
        "opponent_form_runs_1to3": opp_form_runs,
        "opponent_form_avg_1to5": opp_form_avg_1to5,
        "opponent_form_runs_1to5": opp_form_runs_1to5,
        "similar_avg_runs_1to3": similar_avg_runs,
        "similar_avg_runs_1to5": similar_avg_runs_1to5,
        "similar_avg_n": similar_avg_n,
        "days_rest": days_rest,
        "opponent_vs_hand_avg": opp_vs_hand_avg,
        "opponent_vs_hand_ops": opp_vs_hand_ops,
        "park_factor": park_factor,
        "own_bullpen_ip_3d": own_bullpen_ip_3d,
        "h2h_avg_runs_1to3": h2h["avg_runs"], "h2h_n_games": h2h["n_games"],
    }
    features.update(weather_features or live_weather_features(None, None))
    return features, None


def predict_with_bundle(bundle, features):
    X = pd.DataFrame([features])[bundle["features"]]
    model = bundle["model"]
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[0, 1]
    return model.predict(X)[0]


def predict_matchup(features_home, features_away, bundle):
    """Sirve cualquiera de los 5 modelos conjuntos entrenados por
    ml_train_matchup.py (favorito y total de carreras 1-3/1-5, total 1er
    inning): arma la fila home_pitcher_*/away_pitcher_* a partir de los
    dicts de features que compute_current_features ya calcula por
    separado para cada pitcher, y regresa la probabilidad de la clase
    positiva (home favorito / total sobre la linea, segun el bundle)."""
    row = {}
    row.update({f"home_pitcher_{k}": v for k, v in features_home.items()})
    row.update({f"away_pitcher_{k}": v for k, v in features_away.items()})
    X = pd.DataFrame([row])[bundle["features"]]
    return bundle["model"].predict_proba(X)[0, 1]


def main():
    parser = argparse.ArgumentParser(description="Predice con ML: 0 carreras 1er inning y carreras esperadas 1-3")
    parser.add_argument("equipo_a")
    parser.add_argument("equipo_b")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--model", default="model.joblib", help="Modelo de clasificacion (0 carreras 1er inning)")
    parser.add_argument("--model-1to3", default="model_1to3.joblib", help="Modelo de regresion (carreras 1-3)")
    parser.add_argument("--model-1to5", default="model_1to5.joblib", help="Modelo de regresion (carreras 1-5)")
    args = parser.parse_args()

    bundle_1st = joblib.load(args.model)
    print(f"Modelo 1er inning cargado: {bundle_1st['model_name']} ({args.model})")

    bundle_1to3 = None
    if os.path.exists(args.model_1to3):
        bundle_1to3 = joblib.load(args.model_1to3)
        print(f"Modelo 1-3 entradas cargado: {bundle_1to3['model_name']} ({args.model_1to3})")

    bundle_1to5 = None
    if os.path.exists(args.model_1to5):
        bundle_1to5 = joblib.load(args.model_1to5)
        print(f"Modelo 1-5 entradas cargado: {bundle_1to5['model_name']} ({args.model_1to5})")
    print()

    teams = m.get_teams()
    team_a = m.resolve_team(args.equipo_a, teams)
    team_b = m.resolve_team(args.equipo_b, teams)

    matchup_info = m.find_matchup_game(team_a["id"], team_b["id"]) or \
        m.find_matchup_game(team_b["id"], team_a["id"])
    if not matchup_info:
        print("No encontre un juego programado entre estos dos equipos.")
        return

    pid_a, pname_a = matchup_info["pitchers"].get(team_a["id"], (None, None))
    pid_b, pname_b = matchup_info["pitchers"].get(team_b["id"], (None, None))

    is_home_a = matchup_info["home_team_id"] == team_a["id"]
    is_home_b = matchup_info["home_team_id"] == team_b["id"]

    venue = m.get_venue_details(matchup_info["venue_id"])
    weather = m.get_weather_forecast(venue["lat"], venue["lon"], matchup_info["date"])
    weather_features = live_weather_features(venue, weather)
    print(f"Clima del juego (pronostico): {weather['temp_c']:.0f}C, "
          f"viento {weather_features['game_wind_mph']} mph "
          f"({'adentro' if weather_features['game_wind_effect'] == -1 else 'afuera' if weather_features['game_wind_effect'] == 1 else 'cruzado/neutral'}), "
          f"{'techo cerrado' if weather_features['game_is_indoor'] else 'aire libre/techo abierto'}\n"
          if weather else "Clima: N/D\n")

    avg_a_adj, injured_a, avg_a_raw = m.team_avg_adjusted_for_injuries(team_a["id"], args.season)
    avg_b_adj, injured_b, avg_b_raw = m.team_avg_adjusted_for_injuries(team_b["id"], args.season)
    for team, raw, adj, inj in ((team_a, avg_a_raw, avg_a_adj, injured_a), (team_b, avg_b_raw, avg_b_adj, injured_b)):
        print(f"{team['name']}: AVG temporada {raw:.3f} | AVG ajustado sin lesionados: "
              f"{adj:.3f}" if raw is not None and adj is not None else f"{team['name']}: AVG N/D")
        for p in inj:
            avg_txt = f"{p['avg']:.3f}" if p["avg"] is not None else "N/D"
            print(f"    LESIONADO: {p['name']} ({p['position']}, {p['status']}) - AVG {avg_txt}")
    print()

    opp_avg_by_team = {team_a["id"]: avg_b_adj, team_b["id"]: avg_a_adj}

    for pid, pname, team, opp, is_home in (
        (pid_a, pname_a, team_a, team_b, is_home_a),
        (pid_b, pname_b, team_b, team_a, is_home_b),
    ):
        if pid is None:
            print(f"{team['name']}: abridor no disponible todavia.")
            continue
        features, error = compute_current_features(pid, opp["id"], is_home, args.season,
                                                     weather_features=weather_features,
                                                     opponent_avg_override=opp_avg_by_team[team["id"]],
                                                     team_id=team["id"], game_date=matchup_info["date"])
        print(f"{team['name']} - {pname}:")
        if features is None:
            print(f"  {error}")
            print()
            continue

        prob_1st = predict_with_bundle(bundle_1st, features)
        print(f"  Prediccion ({bundle_1st['model_name']}): {prob_1st*100:.1f}% de que EL PITCHER "
              f"deje 0 carreras PERMITIDAS en el 1er inning "
              f"(baseline ingenuo: {features['scoreless_rate_trailing']*100:.1f}%)")

        if bundle_1to3:
            runs_1to3 = predict_with_bundle(bundle_1to3, features)
            print(f"  Prediccion ({bundle_1to3['model_name']}): {runs_1to3:.2f} carreras "
                  f"PERMITIDAS POR EL PITCHER en entradas 1-3 (carreras que le anota el rival a el) "
                  f"[baseline ingenuo: {features['runs_1to3_trailing_avg']:.2f}]")
        if bundle_1to5:
            runs_1to5 = predict_with_bundle(bundle_1to5, features)
            print(f"  Prediccion ({bundle_1to5['model_name']}): {runs_1to5:.2f} carreras "
                  f"PERMITIDAS POR EL PITCHER en entradas 1-5 (carreras que le anota el rival a el) "
                  f"[baseline ingenuo: {features['runs_1to5_trailing_avg']:.2f}]")
        print()


if __name__ == "__main__":
    main()
