"""
Recolecta un dataset historico de aperturas de pitcher en MLB, con features
"point-in-time" (calculadas solo con datos anteriores a cada apertura, para
evitar fuga de informacion del futuro), para entrenar un modelo que prediga
si el abridor dejara 0 carreras en el 1er inning (el bet "-0.5 carreras 1er
inning") en su PROXIMA apertura.

Cada fila = una apertura de un pitcher, con:
  - features calculadas SOLO con sus aperturas anteriores (hasta 10 previas)
  - la etiqueta (label) = si esa apertura en particular fue 0 carreras en el 1er inning

Uso:
    python ml_data.py --season 2026 --out training_data.csv
    python ml_data.py --season 2026 --teams "Reds,Brewers,Yankees" --out sample.csv   # prueba rapida

Simplificacion conocida (documentada, no oculta):
  - "opponent_avg" usa el AVG de temporada ACTUAL del rival (no el AVG que
    tenia el rival justo antes de ese juego especifico). Como el AVG de un
    equipo cambia poco de un juego a otro, el sesgo es pequeno, pero no es
    cero. Para un dataset 100% libre de fuga habria que recalcular el AVG
    del rival juego por juego, lo cual es mucho mas caro en llamadas a la API.
"""

import argparse
import concurrent.futures
import re
import sys
import time
from datetime import datetime, timedelta

import pandas as pd

import mlb_first_inning_report as m

_LINESCORE_CACHE = {}
_PBP_CACHE = {}
_TEAM_GAMELOG_CACHE = {}
_WEATHER_CACHE = {}
_BOXSCORE_CACHE = {}


def parse_wind(wind_str):
    """'5 mph, In From RF' -> (5.0, -1). Direcciones: In=-1 (adentro, favorece
    pitchers), Out=+1 (afuera, favorece bateadores), cualquier otra cosa
    (cruzado, Varies, Calm, None) = 0."""
    if not wind_str:
        return None, 0
    match = re.search(r"(\d+)\s*mph", wind_str, re.IGNORECASE)
    mph = float(match.group(1)) if match else None
    text = wind_str.lower()
    if "in from" in text:
        effect = -1
    elif "out to" in text:
        effect = 1
    else:
        effect = 0
    return mph, effect


def get_game_weather_cached(gamePk):
    """Clima REAL registrado por MLB para ese juego especifico (no un pronostico
    retroactivo): temperatura, viento (mph + direccion en texto plano ya
    relativa al terreno, ej. 'In From RF'), condicion, y tipo de techo del
    estadio. Se usa tal cual para entrenamiento (es informacion conocida
    antes del primer pitch, no una fuga del resultado del juego)."""
    if gamePk in _WEATHER_CACHE:
        return _WEATHER_CACHE[gamePk]
    url = f"https://statsapi.mlb.com/api/v1.1/game/{gamePk}/feed/live"
    last_err = None
    data = None
    for attempt in range(3):
        try:
            r = m.requests.get(url, timeout=15)
            r.raise_for_status()
            data = r.json()
            break
        except (m.requests.exceptions.ConnectionError, m.requests.exceptions.Timeout) as e:
            last_err = e
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
    if data is None:
        raise last_err
    gd = data.get("gameData", {})
    weather = gd.get("weather", {}) or {}
    field_info = gd.get("venue", {}).get("fieldInfo", {}) or {}

    temp_f = None
    try:
        temp_f = float(weather.get("temp"))
    except (TypeError, ValueError):
        pass

    wind_mph, wind_effect = parse_wind(weather.get("wind"))
    condition = (weather.get("condition") or "").lower()
    roof_type = (field_info.get("roofType") or "").lower()
    is_indoor = 1 if ("roof" in condition or "dome" in condition or roof_type == "dome") else 0

    result = {
        "game_temp_f": temp_f,
        "game_wind_mph": wind_mph,
        "game_wind_effect": wind_effect,
        "game_is_indoor": is_indoor,
    }
    _WEATHER_CACHE[gamePk] = result
    return result


def live_weather_features(venue, weather):
    """Igual que get_game_weather_cached pero para un juego FUTURO, a partir
    del pronostico (Open-Meteo) en vez del clima real ya ocurrido. Usa el
    mismo formato de columnas para que el modelo reciba features consistentes
    en entrenamiento (clima real) y en prediccion en vivo (pronostico)."""
    if not weather or not venue:
        return {"game_temp_f": None, "game_wind_mph": None, "game_wind_effect": None, "game_is_indoor": None}

    temp_f = weather["temp_c"] * 9 / 5 + 32
    wind_desc = m.classify_wind(weather["wind_dir_from"], venue["azimuth"])
    if "adentro" in wind_desc:
        wind_effect = -1
    elif "afuera" in wind_desc:
        wind_effect = 1
    else:
        wind_effect = 0
    is_indoor = 1 if "CERRADO" in m.roof_status_heuristic(weather) else 0

    return {
        "game_temp_f": temp_f,
        "game_wind_mph": weather["wind_mph"],
        "game_wind_effect": wind_effect,
        "game_is_indoor": is_indoor,
    }


def get_linescore_cached(gamePk):
    if gamePk not in _LINESCORE_CACHE:
        _LINESCORE_CACHE[gamePk] = m.get_json(f"/game/{gamePk}/linescore")
    return _LINESCORE_CACHE[gamePk]


def get_pbp_cached(gamePk):
    if gamePk not in _PBP_CACHE:
        _PBP_CACHE[gamePk] = m.get_json(f"/game/{gamePk}/playByPlay")
    return _PBP_CACHE[gamePk]


def runs_allowed_first_cached(gamePk, is_home):
    ls = get_linescore_cached(gamePk)
    innings = ls.get("innings", [])
    if not innings:
        return None
    side = "away" if is_home else "home"
    return innings[0].get(side, {}).get("runs")


def runs_allowed_innings_cached(gamePk, is_home, inning_end):
    ls = get_linescore_cached(gamePk)
    innings = ls.get("innings", [])
    side = "away" if is_home else "home"
    return sum((inn.get(side, {}).get("runs", 0) or 0) for inn in innings[:inning_end])


def runs_allowed_1to3_cached(gamePk, is_home):
    return runs_allowed_innings_cached(gamePk, is_home, 3)


def runs_allowed_1to5_cached(gamePk, is_home):
    return runs_allowed_innings_cached(gamePk, is_home, 5)


_TTO1_CACHE = {}


def runs_and_pa_in_tto1(gamePk, pitcher_id, pitcher_is_home):
    """Carreras permitidas y bateadores enfrentados por EL ABRIDOR durante su
    'primera vuelta al orden' (times through the order 1): desde su primer
    bateador enfrentado hasta justo antes de que enfrente a un bateador que
    ya habia visto antes en este juego, o hasta que sale del juego - lo que
    pase primero. Es una medida mas fina que 'carreras en entradas 1-3',
    porque no depende de cortes de entrada fijos (un abridor eficiente puede
    completar su 1ra vuelta en 2 entradas; uno que se mete en problemas
    puede seguir en su 1ra vuelta entrando a la 3ra), y es justo lo que se
    apuesta en la practica: como le va a un abridor viendo al lineup rival
    por PRIMERA vez, que suele ser su mejor version segun las estadisticas
    de MLB de 'times through the order penalty'.

    Usa result.awayScore/homeScore de cada jugada (marcador acumulado
    despues de esa jugada) para las carreras - como el abridor siempre
    empieza el juego 0-0, el marcador del RIVAL despues de la ultima
    jugada de su 1ra vuelta ES directamente las carreras permitidas."""
    key = (gamePk, pitcher_id)
    if key in _TTO1_CACHE:
        return _TTO1_CACHE[key]
    pbp = get_pbp_cached(gamePk)
    opp_score_key = "away" if pitcher_is_home else "home"
    seen_batters = set()
    runs = 0
    pa = 0
    for p in pbp.get("allPlays", []):
        if p["matchup"]["pitcher"]["id"] != pitcher_id:
            continue
        batter_id = p["matchup"]["batter"]["id"]
        if batter_id in seen_batters:
            break
        seen_batters.add(batter_id)
        pa += 1
        runs = p["result"].get(f"{opp_score_key}Score", runs) or 0
    result = (runs, pa) if pa > 0 else (None, None)
    _TTO1_CACHE[key] = result
    return result


def team_batting_innings_cached(gamePk, team_is_home, inning_end):
    ls = get_linescore_cached(gamePk)
    innings = ls.get("innings", [])
    side = "home" if team_is_home else "away"
    runs = sum((inn.get(side, {}).get("runs", 0) or 0) for inn in innings[:inning_end])

    pbp = get_pbp_cached(gamePk)
    hits = at_bats = 0
    for p in pbp.get("allPlays", []):
        about = p["about"]
        if about["inning"] > inning_end:
            continue
        batting_is_home = not about["isTopInning"]
        if batting_is_home != team_is_home:
            continue
        event_type = p["result"].get("eventType", "")
        if event_type in m.NON_AB_EVENTS:
            continue
        at_bats += 1
        if event_type in m.HIT_EVENTS:
            hits += 1
    return runs, hits, at_bats


def get_team_game_log(team_id, season):
    key = (team_id, season)
    if key in _TEAM_GAMELOG_CACHE:
        return _TEAM_GAMELOG_CACHE[key]
    data = m.get_json(
        "/schedule",
        {"sportId": 1, "teamId": team_id, "startDate": f"{season}-01-01",
         "endDate": m.date.today().isoformat()},
    )
    games = []
    for d in data.get("dates", []):
        for g in d["games"]:
            if g["status"]["detailedState"] != "Final":
                continue
            for side in ("away", "home"):
                if g["teams"][side]["team"]["id"] == team_id:
                    games.append({"date": d["date"], "gamePk": g["gamePk"], "is_home": side == "home"})
    games.sort(key=lambda x: x["date"])
    _TEAM_GAMELOG_CACHE[key] = games
    return games


def trailing_team_offense(team_id, as_of_date, season, n=5, inning_end=3):
    games = get_team_game_log(team_id, season)
    prior = [g for g in games if g["date"] < as_of_date][-n:]
    if not prior:
        return None, None
    total_runs = total_hits = total_ab = 0
    for g in prior:
        runs, hits, ab = team_batting_innings_cached(g["gamePk"], g["is_home"], inning_end)
        total_runs += runs
        total_hits += hits
        total_ab += ab
    avg = (total_hits / total_ab) if total_ab else None
    avg_runs = total_runs / len(prior)
    return avg, avg_runs


def get_boxscore_cached(gamePk):
    if gamePk not in _BOXSCORE_CACHE:
        _BOXSCORE_CACHE[gamePk] = m.get_json(f"/game/{gamePk}/boxscore")
    return _BOXSCORE_CACHE[gamePk]


_BULLPEN_WORKLOAD_CACHE = {}


def bullpen_workload_asof(team_id, as_of_date, days=3):
    """Igual que m.bullpen_recent_workload (carga reciente del bullpen, sin
    contar al abridor de cada juego), pero anclada en as_of_date en vez de
    'hoy'. Usa la ventana [as_of_date - days, as_of_date - 1] (nunca incluye
    el dia mismo de la apertura que se esta prediciendo/entrenando) para que
    sea point-in-time y se pueda reusar tal cual en entrenamiento (sin fuga)
    y en prediccion en vivo (as_of_date = fecha del juego de hoy).

    Cacheado por (team_id, as_of_date, days): muchos pitchers del MISMO
    equipo comparten fechas de apertura parecidas, asi que sin cache esto
    repite el mismo request de schedule una y otra vez para practicamente
    la misma ventana de 3 dias - es lo que hacia que recolectar la
    temporada completa se volviera impracticamente lento (horas en vez de
    minutos)."""
    key = (team_id, as_of_date, days)
    if key in _BULLPEN_WORKLOAD_CACHE:
        return _BULLPEN_WORKLOAD_CACHE[key]
    as_of = datetime.strptime(as_of_date, "%Y-%m-%d").date()
    start = (as_of - timedelta(days=days)).isoformat()
    end = (as_of - timedelta(days=1)).isoformat()
    data = m.get_json("/schedule", {"sportId": 1, "teamId": team_id, "startDate": start, "endDate": end})
    total_ip = total_pitches = 0.0
    games_included = 0
    for d in data.get("dates", []):
        for g in d["games"]:
            if g["status"]["detailedState"] != "Final":
                continue
            gamePk = g["gamePk"]
            for side in ("away", "home"):
                if g["teams"][side]["team"]["id"] != team_id:
                    continue
                box = get_boxscore_cached(gamePk)
                pitcher_ids = box["teams"][side].get("pitchers", [])
                if not pitcher_ids:
                    continue
                games_included += 1
                for pid in pitcher_ids[1:]:  # [0] es el abridor de ESE juego
                    player = box["teams"][side]["players"].get(f"ID{pid}", {})
                    stats = player.get("stats", {}).get("pitching", {})
                    total_ip += m.ip_to_decimal(stats.get("inningsPitched", "0.0"))
                    total_pitches += stats.get("numberOfPitches", 0) or 0
    result = {"bullpen_ip": total_ip, "bullpen_pitches": total_pitches, "games_included": games_included}
    _BULLPEN_WORKLOAD_CACHE[key] = result
    return result


def head_to_head_asof(batting_team_id, pitcher_id, as_of_date, seasons, inning_end=3):
    """Historial REAL de batting_team_id bateando contra pitcher_id, sumando
    solo aperturas ANTERIORES a as_of_date - la misma idea que
    m.head_to_head_history pero cortada en el tiempo, para poder usarse en
    el dataset de entrenamiento sin fuga de informacion (ninguna apertura
    futura respecto a la fila que se esta construyendo cuenta aqui)."""
    total_runs = total_hits = total_ab = 0
    n_games = 0
    for season in seasons:
        starts = m.get_all_starts(pitcher_id, season)
        for s in starts:
            if s["opponent"]["id"] != batting_team_id or s["date"] >= as_of_date:
                continue
            gamePk = s["game"]["gamePk"]
            team_is_home = not s["isHome"]
            runs, hits, ab = team_batting_innings_cached(gamePk, team_is_home, inning_end)
            total_runs += runs
            total_hits += hits
            total_ab += ab
            n_games += 1
    avg = (total_hits / total_ab) if total_ab else None
    avg_runs = (total_runs / n_games) if n_games else None
    return {"avg": avg, "avg_runs": avg_runs, "n_games": n_games}


def collect_pitcher_rows(pitcher_id, pitcher_name, team_id, season,
                          window=10, min_prior=3, avg_similarity_threshold=0.015):
    starts = m.get_all_starts(pitcher_id, season)
    rows = []
    skipped_errors = 0
    for i, s in enumerate(starts):
        prior = starts[max(0, i - window):i]
        if len(prior) < min_prior:
            continue

        # Un try/except por apertura: de vez en cuando la API trae un juego o
        # jugada con datos incompletos/inesperados (ya paso con un umpire sin
        # 'fullName' y con un team_id sin pagina de stats en otros proyectos
        # hermanos). Sin esto, UNA apertura rara tumba la recoleccion de la
        # temporada completa de los 30 equipos.
        try:
            row = _build_pitcher_row(pitcher_id, pitcher_name, team_id, season, s, prior,
                                      avg_similarity_threshold)
            if row is not None:
                rows.append(row)
        except Exception as e:
            skipped_errors += 1
            print(f"  [salteado] {pitcher_name} {s.get('date', '?')}: {e}", file=sys.stderr)

    if skipped_errors:
        print(f"  {pitcher_name}: {skipped_errors} apertura(s) salteada(s) por error", file=sys.stderr)
    return rows


def _build_pitcher_row(pitcher_id, pitcher_name, team_id, season, s, prior, avg_similarity_threshold):
    """Arma una fila del dataset para UNA apertura (s), usando solo sus
    aperturas anteriores (prior). Regresa None si falta el resultado del
    juego (label). Separado de collect_pitcher_rows para poder envolverlo en
    un try/except por apertura sin liar el control de flujo del for."""
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

    cur = s
    opp_id = cur["opponent"]["id"]
    opp_avg = m.get_team_season_avg(opp_id, season)
    is_home_cur = cur["isHome"]

    sim_runs_1to3 = []
    sim_runs_1to5 = []
    for ps in prior:
        if ps["isHome"] != is_home_cur:
            continue
        p_opp_avg = m.get_team_season_avg(ps["opponent"]["id"], season)
        if opp_avg is None or p_opp_avg is None or abs(p_opp_avg - opp_avg) > avg_similarity_threshold:
            continue
        sim_runs_1to3.append(runs_allowed_1to3_cached(ps["game"]["gamePk"], ps["isHome"]))
        sim_runs_1to5.append(runs_allowed_1to5_cached(ps["game"]["gamePk"], ps["isHome"]))
    similar_avg_runs = (sum(sim_runs_1to3) / len(sim_runs_1to3)) if sim_runs_1to3 else None
    similar_avg_runs_1to5 = (sum(sim_runs_1to5) / len(sim_runs_1to5)) if sim_runs_1to5 else None
    similar_avg_n = len(sim_runs_1to3)

    opp_form_avg, opp_form_runs = trailing_team_offense(opp_id, cur["date"], season, n=5, inning_end=3)
    opp_form_avg_1to5, opp_form_runs_1to5 = trailing_team_offense(opp_id, cur["date"], season, n=5, inning_end=5)

    days_rest = None
    last_prior_date = prior[-1]["date"] if prior else None
    if last_prior_date:
        d1 = datetime.strptime(last_prior_date, "%Y-%m-%d")
        d2 = datetime.strptime(cur["date"], "%Y-%m-%d")
        days_rest = (d2 - d1).days

    pitcher_hand = m.get_pitcher_hand(pitcher_id)
    opp_vs_hand = m.get_team_vs_hand_split(opp_id, season, pitcher_hand) if pitcher_hand else None
    opp_vs_hand_avg = opp_vs_hand["avg"] if opp_vs_hand else None
    opp_vs_hand_ops = opp_vs_hand["ops"] if opp_vs_hand else None

    home_team_id_this_game = team_id if is_home_cur else opp_id
    park_factor = m.compute_park_factor(home_team_id_this_game, season)

    # El bullpen relevante aqui es el DEL PROPIO EQUIPO del abridor (team_id),
    # no el del rival: si el bullpen propio esta cargado, el manager tiene mas
    # incentivo a dejar al abridor mas tiempo (o a sacarlo rapido si no hay
    # brazos frescos), lo que afecta cuantas carreras se le acumulan en
    # entradas 1-3/1-5. El bullpen del RIVAL (que batea, no lanza, en estas
    # entradas) no tiene relacion causal con las carreras que permite EL
    # abridor - por eso NO se usa opp_id aqui.
    own_bullpen = bullpen_workload_asof(team_id, cur["date"], days=3)
    h2h = head_to_head_asof(opp_id, pitcher_id, cur["date"], [season, season - 1], inning_end=3)

    label_runs = runs_allowed_first_cached(cur["game"]["gamePk"], is_home_cur)
    label_runs_1to3 = runs_allowed_1to3_cached(cur["game"]["gamePk"], is_home_cur)
    label_runs_1to5 = runs_allowed_1to5_cached(cur["game"]["gamePk"], is_home_cur)
    if label_runs is None or label_runs_1to3 is None or label_runs_1to5 is None:
        return None

    return {
        "pitcher_id": pitcher_id, "pitcher_name": pitcher_name, "team_id": team_id,
        "date": cur["date"], "gamePk": cur["game"]["gamePk"],
        "n_prior_starts": n_prior,
        "scoreless_rate_trailing": scoreless_rate,
        "runs_1to3_trailing_avg": runs_1to3_trailing_avg,
        "runs_1to5_trailing_avg": runs_1to5_trailing_avg,
        "tto1_runs_trailing_avg": tto1_runs_trailing_avg, "tto1_pa_trailing_avg": tto1_pa_trailing_avg,
        "era_to_date": era_to_date, "whip_to_date": whip_to_date,
        "fip_to_date": fip_to_date, "ip_per_start_to_date": ip_per_start_to_date,
        "is_home": int(is_home_cur), "opponent_avg": opp_avg,
        "similar_avg_runs_1to3": similar_avg_runs, "similar_avg_runs_1to5": similar_avg_runs_1to5,
        "similar_avg_n": similar_avg_n,
        "opponent_form_avg_1to3": opp_form_avg, "opponent_form_runs_1to3": opp_form_runs,
        "opponent_form_avg_1to5": opp_form_avg_1to5, "opponent_form_runs_1to5": opp_form_runs_1to5,
        "days_rest": days_rest,
        "opponent_vs_hand_avg": opp_vs_hand_avg, "opponent_vs_hand_ops": opp_vs_hand_ops,
        "park_factor": park_factor,
        "own_bullpen_ip_3d": own_bullpen["bullpen_ip"],
        "h2h_avg_runs_1to3": h2h["avg_runs"], "h2h_n_games": h2h["n_games"],
        "label_scoreless_1st": int(label_runs == 0),
        "label_runs_1st": label_runs,
        "label_runs_1to3": label_runs_1to3,
        "label_runs_1to5": label_runs_1to5,
    }


def collect_league_dataset(season, team_names=None, verbose=True, max_workers=8):
    """Recolecta todo el dataset. Los pitchers de un mismo equipo son
    independientes entre si (solo comparten los caches de solo-lectura de
    ml_data.py/mlb_first_inning_report.py), asi que se procesan en paralelo
    con un thread pool: como el cuello de botella es esperar respuestas de
    la API (I/O), no CPU, varios hilos a la vez reducen el tiempo total de
    horas a minutos sin cambiar que datos se calculan ni como."""
    teams = m.get_teams()
    if team_names:
        wanted = {t.strip().lower() for t in team_names}
        teams = [t for t in teams if t["name"].lower() in wanted or t["teamName"].lower() in wanted]

    all_rows = []
    t0 = time.time()
    for ti, team in enumerate(teams):
        roster = m.get_json(f"/teams/{team['id']}/roster", {"rosterType": "fullSeason", "season": season})
        pitchers = [p for p in roster.get("roster", []) if p["position"]["abbreviation"] == "P"]
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(collect_pitcher_rows, p["person"]["id"], p["person"]["fullName"],
                                 team["id"], season): p
                for p in pitchers
            }
            for future in concurrent.futures.as_completed(futures):
                p = futures[future]
                try:
                    all_rows.extend(future.result())
                except Exception as e:
                    print(f"  [salteado] pitcher {p['person']['fullName']}: {e}", file=sys.stderr)
        if verbose:
            print(f"[{ti+1}/{len(teams)}] {team['name']}: {len(pitchers)} pitchers revisados, "
                  f"{len(all_rows)} filas acumuladas ({time.time()-t0:.0f}s)", file=sys.stderr)
        # _PBP_CACHE y _BOXSCORE_CACHE guardan el JSON completo (jugada por
        # jugada / boxscore) de cada juego que se toca - para UN equipo esto
        # es manejable, pero acumulado sin limpiar para las 30 franquicias
        # de la temporada completa puede llegar a varios cientos de MB, mas
        # de lo que da el contenedor gratuito de Streamlit Cloud (se vio
        # tronar la app entera, sin traceback, a media recoleccion). El
        # costo es volver a pedir el juego si otro equipo lo comparte mas
        # adelante (partido entre dos equipos ya procesados) - mas lento,
        # pero no se queda sin memoria.
        _PBP_CACHE.clear()
        _BOXSCORE_CACHE.clear()
    return pd.DataFrame(all_rows)


def enrich_with_weather(df, verbose=True):
    """Agrega columnas de clima real (game_temp_f, game_wind_mph, game_wind_effect,
    game_is_indoor) a un DataFrame que ya tiene una columna 'gamePk', sin
    recalcular nada mas. Un fetch por gamePk unico (se puede repetir entre
    filas si varios pitchers de un mismo equipo comparten aperturas... en la
    practica cada gamePk es unico por fila, asi que es 1 llamada por fila)."""
    unique_pks = df["gamePk"].unique()
    t0 = time.time()
    weather_by_pk = {}
    skipped = 0
    for i, pk in enumerate(unique_pks):
        # Un try/except por juego: con ~1800+ requests seguidos a la API en
        # este paso, es cuestion de tiempo que uno se quede sin respuesta
        # aunque ya tenga reintentos - sin esto, ese juego tumbaba todo el
        # paso y se perdian los otros 1000+ ya descargados.
        try:
            weather_by_pk[pk] = get_game_weather_cached(int(pk))
        except Exception as e:
            skipped += 1
            weather_by_pk[pk] = {"game_temp_f": None, "game_wind_mph": None,
                                  "game_wind_effect": None, "game_is_indoor": None}
            print(f"  [salteado] gamePk {pk}: {e}", file=sys.stderr)
        if verbose and (i + 1) % 200 == 0:
            print(f"  clima: {i+1}/{len(unique_pks)} juegos ({time.time()-t0:.0f}s)", file=sys.stderr)
    if skipped:
        print(f"  Total de juegos salteados por error: {skipped} (se quedan sin clima, el resto de "
              f"columnas no se ve afectado)", file=sys.stderr)
    weather_df = pd.DataFrame.from_dict(weather_by_pk, orient="index")
    weather_df.index.name = "gamePk"
    return df.merge(weather_df, on="gamePk", how="left")


def main():
    parser = argparse.ArgumentParser(description="Recolecta dataset historico para el modelo de 1er inning")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--teams", default=None,
                         help="Lista separada por comas para una corrida de prueba, ej. 'Reds,Brewers'")
    parser.add_argument("--out", default="training_data.csv")
    parser.add_argument("--enrich-weather", default=None,
                         help="En vez de recolectar todo de nuevo, solo agrega clima real a un CSV existente "
                              "(pasa la ruta del CSV de entrada; se sobrescribe --out con el resultado)")
    args = parser.parse_args()

    if args.enrich_weather:
        df = pd.read_csv(args.enrich_weather)
        print(f"Enriqueciendo {len(df)} filas ({df['gamePk'].nunique()} juegos unicos) con clima real...",
              file=sys.stderr)
        df = enrich_with_weather(df)
        df.to_csv(args.out, index=False)
        print(f"Guardado {len(df)} filas con clima en {args.out}")
        return

    team_names = [t.strip() for t in args.teams.split(",")] if args.teams else None
    df = collect_league_dataset(args.season, team_names=team_names)
    df.to_csv(args.out, index=False)
    print(f"Guardado {len(df)} filas en {args.out}")
    if len(df):
        print(df["label_scoreless_1st"].value_counts(normalize=True))


if __name__ == "__main__":
    main()
