"""
Reporte pre-apuesta para un juego de MLB entre dos equipos.

Responde con datos reales (APIs publicas, sin key) todas estas preguntas:

  Sobre el 1er inning (datos exactos via MLB Stats API):
    1) Veces que el abridor del equipo A dejo -0.5 carreras (0 carreras) en el 1er
       inning en sus ultimas 10 aperturas.
    2) Lo mismo para el equipo B.
    3) Average de bateo del equipo A en el 1er inning en sus ultimos 5 juegos.
    4) Lo mismo para el equipo B.
    5) Carreras promedio anotadas por el equipo A en el 1er inning, ultimos 5 juegos.
    6) Lo mismo para el equipo B.

  Sobre el juego y los abridores (MLB Stats API + Open-Meteo, sin key):
    - Quien es el abridor de cada equipo.
    - Donde juegan (estadio) y si es domo/techo retractil/aire libre.
    - Clima pronosticado para el juego (temperatura, prob. de lluvia).
    - Direccion y velocidad del viento, y si sopla de los jardines al plato
      (adentro) o del plato a los jardines (afuera), usando el azimuth oficial
      del estadio que publica MLB.
    - ERA, WHIP, FIP y entradas promedio por apertura de cada abridor, temporada actual.

Uso:
    python mlb_first_inning_report.py "Reds" "Brewers"
    python mlb_first_inning_report.py "Reds" "Brewers" --season 2026

    # Escribir el reporte en un Google Sheet (sobrescribe la misma plantilla cada vez):
    python mlb_first_inning_report.py "Reds" "Brewers" --sheet-id TU_SPREADSHEET_ID \
        --credentials credentials.json --sheet-tab "Reporte MLB"
    # TU_SPREADSHEET_ID es la parte de la URL entre /d/ y /edit, ej.
    # https://docs.google.com/spreadsheets/d/ESTE_ES_EL_ID/edit
    # Ver setup_google_sheets.md para la guia paso a paso de credenciales.

Notas:
  - No requiere API key para nada del reporte principal: MLB Stats API y
    Open-Meteo son publicas y gratuitas.
  - El "techo abierto/cerrado" del dia se calcula con una heuristica simple
    (clima + umbral de 15.6C / 60F) para estadios de techo retractil; no es la
    decision oficial del equipo. Para eso llama al roof hotline del estadio.
  - FIP se calcula con la formula estandar y una constante aproximada
    (ajustable con --fip-constant); MLB Stats API no publica FIP oficial.
  - Gemini es opcional: si defines GEMINI_API_KEY, --use-gemini lo usa como
    respaldo narrativo SOLO para preguntas fuera de este reporte (lesiones,
    noticias de roster, etc). Nunca se usa para los numeros de arriba.
"""

import argparse
import csv
import os
import sys
import time
from datetime import date, datetime, timedelta

import requests

BASE = "https://statsapi.mlb.com/api/v1"

HIT_EVENTS = {"single", "double", "triple", "home_run"}
NON_AB_EVENTS = {
    "walk", "intent_walk", "hit_by_pitch", "sac_fly", "sac_bunt",
    "sac_fly_double_play", "sac_bunt_double_play", "catcher_interf",
}

# Tipo de techo por equipo (dato fijo de cada estadio). "retractil" significa
# que el equipo decide dia a dia si abrirlo o cerrarlo.
ROOF_TYPES = {
    "Arizona Diamondbacks": "techo retractil",
    "Athletics": "aire libre",
    "Atlanta Braves": "aire libre",
    "Baltimore Orioles": "aire libre",
    "Boston Red Sox": "aire libre",
    "Chicago Cubs": "aire libre",
    "Chicago White Sox": "aire libre",
    "Cincinnati Reds": "aire libre",
    "Cleveland Guardians": "aire libre",
    "Colorado Rockies": "aire libre",
    "Detroit Tigers": "aire libre",
    "Houston Astros": "techo retractil",
    "Kansas City Royals": "aire libre",
    "Los Angeles Angels": "aire libre",
    "Los Angeles Dodgers": "aire libre",
    "Miami Marlins": "techo retractil",
    "Milwaukee Brewers": "techo retractil",
    "Minnesota Twins": "aire libre",
    "New York Mets": "aire libre",
    "New York Yankees": "aire libre",
    "Philadelphia Phillies": "aire libre",
    "Pittsburgh Pirates": "aire libre",
    "San Diego Padres": "aire libre (con techado parcial)",
    "San Francisco Giants": "aire libre",
    "Seattle Mariners": "techo retractil",
    "St. Louis Cardinals": "aire libre",
    "Tampa Bay Rays": "domo/estadio temporal (verificar, en transicion)",
    "Texas Rangers": "techo retractil",
    "Toronto Blue Jays": "techo retractil",
    "Washington Nationals": "aire libre",
}

DEFAULT_FIP_CONSTANT = 3.10
DEFAULT_WHIP_THRESHOLD = 1.1


def classify_whip(whip, threshold=DEFAULT_WHIP_THRESHOLD):
    """IF(WHIP <= threshold, "Aprobada", "Riesgosa")"""
    if whip is None:
        return "N/D"
    try:
        w = float(whip)
    except (TypeError, ValueError):
        return "N/D"
    return "Aprobada" if w <= threshold else "Riesgosa"


CONFIDENCE_LOW = 4
CONFIDENCE_HIGH = 8


def confidence_label(n, low=CONFIDENCE_LOW, high=CONFIDENCE_HIGH):
    """Marca que tan chica es la muestra detras de un numero (aperturas
    revisadas, juegos con AVG similar, enfrentamientos historicos, etc.).
    No cambia el numero, solo avisa cuando hay que confiar menos en el
    porque viene de pocos juegos - el ruido de muestra chica es el error
    mas facil de cometer al leer estos reportes como si fueran certezas."""
    if n is None:
        return "N/D"
    if n < low:
        return f"confianza BAJA, n={n}"
    if n < high:
        return f"confianza MEDIA, n={n}"
    return f"confianza ALTA, n={n}"


def get_json(path, params=None, retries=3):
    """Un reporte hace docenas de llamadas seguidas a la API de MLB - sin
    reintentos, un solo timeout momentaneo (comun al correr desde un
    servidor compartido como Streamlit Cloud) tumba todo el reporte."""
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(f"{BASE}{path}", params=params, timeout=15)
            r.raise_for_status()
            return r.json()
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
    raise last_err


def get_teams():
    data = get_json("/teams", {"sportId": 1})
    return data["teams"]


def resolve_team(query, teams):
    q = query.strip().lower()
    for t in teams:
        if q == t["name"].lower() or q == t["teamName"].lower() or q == t["abbreviation"].lower():
            return t
    for t in teams:
        if q in t["name"].lower() or q in t["teamName"].lower():
            return t
    raise ValueError(f"No encontre un equipo de MLB que coincida con '{query}'")


def resolve_pitcher_on_team(team_id, name_query):
    """Busca un pitcher por nombre en el roster activo del equipo. Sirve para
    forzar manualmente el abridor de hoy cuando el 'probablePitcher' que
    reporta la API de MLB todavia no refleja un cambio de rotacion que las
    casas de apuestas ya tienen confirmado (la API de MLB a veces se tarda
    horas en actualizarse)."""
    q = name_query.strip().lower()
    data = get_json(f"/teams/{team_id}/roster", {"rosterType": "active"})
    candidates = [p for p in data.get("roster", []) if p["position"]["abbreviation"] == "P"]
    for p in candidates:
        if q == p["person"]["fullName"].lower():
            return p["person"]["id"], p["person"]["fullName"]
    for p in candidates:
        if q in p["person"]["fullName"].lower():
            return p["person"]["id"], p["person"]["fullName"]
    raise ValueError(f"No encontre un pitcher en el roster activo que coincida con '{name_query}'")


def find_matchup_game(team_a_id, team_b_id, days_ahead=10):
    """Busca el proximo juego programado especificamente entre estos dos equipos."""
    today = date.today()
    start = today.isoformat()
    end = (today + timedelta(days=days_ahead)).isoformat()
    data = get_json(
        "/schedule",
        {
            "sportId": 1,
            "teamId": team_a_id,
            "opponentId": team_b_id,
            "startDate": start,
            "endDate": end,
            "hydrate": "probablePitcher,venue,team",
        },
    )
    for d in data.get("dates", []):
        for g in d["games"]:
            if g["status"]["detailedState"] not in ("Scheduled", "Pre-Game", "Warmup"):
                continue
            info = {"gamePk": g["gamePk"], "date": d["date"], "venue_id": g["venue"]["id"],
                     "venue_name": g["venue"]["name"], "pitchers": {},
                     "home_team_id": g["teams"]["home"]["team"]["id"]}
            for side in ("away", "home"):
                team = g["teams"][side]
                pp = team.get("probablePitcher")
                info["pitchers"][team["team"]["id"]] = (pp["id"], pp["fullName"]) if pp else (None, None)
            return info
    return None


def find_last_rotation_starter(team_id):
    """Respaldo: si no hay juego programado entre ambos equipos, usa el ultimo
    abridor que aparecio en el roster activo del equipo."""
    today = date.today()
    start = (today - timedelta(days=14)).isoformat()
    end = today.isoformat()
    data = get_json(
        "/schedule",
        {"sportId": 1, "teamId": team_id, "startDate": start, "endDate": end},
    )
    games = []
    for d in data.get("dates", []):
        for g in d["games"]:
            if g["status"]["detailedState"] == "Final":
                games.append((d["date"], g["gamePk"]))
    games.sort(reverse=True)
    for gdate, gamePk in games:
        box = get_json(f"/game/{gamePk}/boxscore")
        for side in ("away", "home"):
            team_box = box["teams"][side]
            if team_box["team"]["id"] == team_id:
                pitchers = team_box.get("pitchers", [])
                if pitchers:
                    pid = pitchers[0]
                    name = box["teams"][side]["players"][f"ID{pid}"]["person"]["fullName"]
                    return pid, name
    return None, None


def get_last_n_starts(pitcher_id, season, n=10):
    data = get_json(
        f"/people/{pitcher_id}/stats",
        {"stats": "gameLog", "group": "pitching", "season": season},
    )
    stats = data.get("stats", [])
    if not stats:
        return []
    splits = stats[0].get("splits", [])
    starts = [s for s in splits if s["stat"].get("gamesStarted") == 1]
    starts.sort(key=lambda s: s["date"])
    return starts[-n:]


def runs_allowed_in_first_by_pitcher(gamePk, pitcher_is_home):
    linescore = get_json(f"/game/{gamePk}/linescore")
    innings = linescore.get("innings", [])
    if not innings:
        return None
    inn1 = innings[0]
    if pitcher_is_home:
        return inn1.get("away", {}).get("runs")
    return inn1.get("home", {}).get("runs")


def count_scoreless_first_innings(pitcher_id, season, n=10):
    starts = get_last_n_starts(pitcher_id, season, n=n)
    results = []
    for s in starts:
        gamePk = s["game"]["gamePk"]
        is_home = s["isHome"]
        runs = runs_allowed_in_first_by_pitcher(gamePk, is_home)
        if runs is not None:
            results.append((s["date"], runs))
    scoreless = sum(1 for _, r in results if r == 0)
    return scoreless, len(results), results


def get_team_last_n_games(team_id, n=5):
    today = date.today()
    start = (today - timedelta(days=30)).isoformat()
    end = today.isoformat()
    data = get_json(
        "/schedule",
        {"sportId": 1, "teamId": team_id, "startDate": start, "endDate": end},
    )
    games = []
    for d in data.get("dates", []):
        for g in d["games"]:
            if g["status"]["detailedState"] != "Final":
                continue
            for side in ("away", "home"):
                if g["teams"][side]["team"]["id"] == team_id:
                    games.append((d["date"], g["gamePk"], side == "home"))
    games.sort(key=lambda x: x[0])
    return games[-n:]


def innings_range_batting_for_team(gamePk, team_is_home, inning_end=1):
    """Runs/hits/at-bats de este equipo en las entradas 1..inning_end de este juego."""
    linescore = get_json(f"/game/{gamePk}/linescore")
    innings = linescore.get("innings", [])
    side = "home" if team_is_home else "away"
    runs = sum((inn.get(side, {}).get("runs", 0) or 0) for inn in innings[:inning_end])

    pbp = get_json(f"/game/{gamePk}/playByPlay")
    hits = 0
    at_bats = 0
    for p in pbp.get("allPlays", []):
        about = p["about"]
        if about["inning"] > inning_end:
            continue
        batting_is_home = not about["isTopInning"]
        if batting_is_home != team_is_home:
            continue
        event_type = p["result"].get("eventType", "")
        if event_type in NON_AB_EVENTS:
            continue
        at_bats += 1
        if event_type in HIT_EVENTS:
            hits += 1
    return runs, hits, at_bats


def runs_allowed_innings_1to3(gamePk, pitcher_is_home):
    linescore = get_json(f"/game/{gamePk}/linescore")
    innings = linescore.get("innings", [])
    side = "away" if pitcher_is_home else "home"
    return sum((inn.get(side, {}).get("runs", 0) or 0) for inn in innings[:3])


SPLIT_RANGES = {"1to3": 3, "1to5": 5, "full": 9}


_TEAM_AVG_CACHE = {}


def get_team_season_avg(team_id, season):
    key = (team_id, season)
    if key in _TEAM_AVG_CACHE:
        return _TEAM_AVG_CACHE[key]
    avg = None
    try:
        data = get_json(
            f"/teams/{team_id}/stats",
            {"stats": "season", "group": "hitting", "season": season},
        )
        avg = float(data["stats"][0]["splits"][0]["stat"]["avg"])
    except (KeyError, IndexError, ValueError, TypeError):
        avg = None
    _TEAM_AVG_CACHE[key] = avg
    return avg


_TEAM_OPS_CACHE = {}


def get_team_season_hitting_stats(team_id, season):
    """AVG/OBP/SLG/OPS de temporada del equipo. OPS (on-base + slugging) es un
    mejor indicador de calidad ofensiva que solo el AVG, porque tambien
    cuenta bases por bola y poder."""
    key = (team_id, season)
    if key in _TEAM_OPS_CACHE:
        return _TEAM_OPS_CACHE[key]
    result = {"avg": None, "obp": None, "slg": None, "ops": None}
    try:
        data = get_json(f"/teams/{team_id}/stats", {"stats": "season", "group": "hitting", "season": season})
        s = data["stats"][0]["splits"][0]["stat"]
        result = {"avg": float(s["avg"]), "obp": float(s["obp"]),
                  "slg": float(s["slg"]), "ops": float(s["ops"])}
    except (KeyError, IndexError, ValueError, TypeError):
        pass
    _TEAM_OPS_CACHE[key] = result
    return result


def get_injured_position_players(team_id, season):
    """Jugadores de POSICION (no pitchers) actualmente en lista de lesionados
    (10/15/60 dias), con su AVG/AB/H de temporada. Usa el roster de 40-man
    con status, que MLB expone publicamente."""
    data = get_json(f"/teams/{team_id}/roster", {"rosterType": "40Man", "season": season})
    injured = []
    for p in data.get("roster", []):
        status = p.get("status", {})
        code = status.get("code", "")
        position = p.get("position", {}).get("abbreviation", "")
        if not code.startswith("D") or position == "P":
            continue
        pid = p["person"]["id"]
        try:
            stat_data = get_json(
                f"/people/{pid}/stats", {"stats": "season", "group": "hitting", "season": season}
            )
            stat = stat_data["stats"][0]["splits"][0]["stat"]
            avg = float(stat.get("avg", 0) or 0)
            ab = int(stat.get("atBats", 0) or 0)
            hits = int(stat.get("hits", 0) or 0)
        except (KeyError, IndexError, ValueError, TypeError):
            avg, ab, hits = None, 0, 0
        injured.append({
            "id": pid, "name": p["person"]["fullName"], "position": position,
            "status": status.get("description", code), "avg": avg, "ab": ab, "hits": hits,
        })
    return injured


def team_avg_adjusted_for_injuries(team_id, season):
    """AVG del equipo si se le restan los turnos/hits de los jugadores de
    posicion actualmente lesionados - una mejor foto de la ofensiva REAL
    disponible hoy que el AVG de temporada completa (que incluye los juegos
    que jugaron antes de lesionarse)."""
    injured = get_injured_position_players(team_id, season)
    raw_avg = get_team_season_avg(team_id, season)
    try:
        data = get_json(f"/teams/{team_id}/stats", {"stats": "season", "group": "hitting", "season": season})
        stat = data["stats"][0]["splits"][0]["stat"]
        team_ab = int(stat.get("atBats", 0) or 0)
        team_hits = int(stat.get("hits", 0) or 0)
    except (KeyError, IndexError, ValueError, TypeError):
        return raw_avg, injured, raw_avg

    injured_ab = sum(p["ab"] for p in injured)
    injured_hits = sum(p["hits"] for p in injured)
    adj_ab = team_ab - injured_ab
    adj_hits = team_hits - injured_hits
    adjusted_avg = (adj_hits / adj_ab) if adj_ab > 0 else raw_avg
    return adjusted_avg, injured, raw_avg


def get_confirmed_lineup(gamePk):
    """Lineup titular (9 bateadores) que MLB ya confirmo para este gamePk
    especifico, si esta disponible. Normalmente se publica entre 1 y 3 horas
    antes del primer lanzamiento; antes de eso regresa listas vacias (no es
    un error, todavia no existe la info). Es mas preciso que el AVG ajustado
    solo por lesionados en la lista de 10/15/60 dias, porque tambien capta a
    un regular que descansa ese dia SIN estar lesionado (cosa que
    team_avg_adjusted_for_injuries no puede ver)."""
    data = get_json("/schedule", {"sportId": 1, "gamePk": gamePk, "hydrate": "lineups"})
    for d in data.get("dates", []):
        for g in d["games"]:
            if g["gamePk"] == gamePk:
                lu = g.get("lineups", {}) or {}
                return {"home": lu.get("homePlayers", []), "away": lu.get("awayPlayers", [])}
    return {"home": [], "away": []}


def lineup_batting_avg(players, season):
    """AVG combinado (hits/AB reales, no promedio de promedios) de los
    bateadores confirmados en el lineup, con UNA sola llamada a la API para
    los 9 (endpoint /people con varios personIds), en vez de 9 llamadas
    sueltas."""
    if not players:
        return None, []
    ids = ",".join(str(p["id"]) for p in players)
    data = get_json("/people", {"personIds": ids,
                                 "hydrate": f"stats(group=hitting,type=season,season={season})"})
    total_ab = total_hits = 0
    detail = []
    for person in data.get("people", []):
        stats = person.get("stats", [])
        stat = stats[0]["splits"][0]["stat"] if stats and stats[0].get("splits") else {}
        ab = int(stat.get("atBats", 0) or 0)
        hits = int(stat.get("hits", 0) or 0)
        avg = float(stat["avg"]) if stat.get("avg") not in (None, "") else None
        total_ab += ab
        total_hits += hits
        detail.append({"id": person.get("id"), "name": person.get("fullName"), "avg": avg, "ab": ab})
    avg = (total_hits / total_ab) if total_ab else None
    return avg, detail


def get_home_plate_umpire(gamePk):
    """Umpire de home plate asignado a este gamePk, si MLB ya lo publico.
    MLB confirma la tripleta de umpires normalmente horas antes del primer
    pitch (a veces el mismo dia), no con dias de anticipacion - por eso esto
    es best-effort y regresa (None, None) la mayoria de las veces que se
    corre el reporte con mucha anticipacion."""
    try:
        box = get_json(f"/game/{gamePk}/boxscore")
    except requests.HTTPError:
        return None, None
    hp = next((o for o in box.get("officials", []) if o.get("officialType") == "Home Plate"), None)
    if not hp:
        return None, None
    return hp["official"]["id"], hp["official"]["fullName"]


def get_umpire_tendency(umpire_name, tendency_path="umpire_tendency.csv"):
    """Busca la tendencia historica de este umpire en el indice generado por
    umpire_data.py (carreras/ponches/BB PROMEDIO en los juegos que dirigio
    esta temporada, comparado contra el promedio de liga). Es un PROXY
    construido solo con datos oficiales de MLB Stats API (boxscore de cada
    juego) - NO es la geometria real de zona de strike (no hay una fuente
    publica y verificada para eso sin depender de un scraper de terceros),
    asi que se marca como tal en el reporte. Regresa None si el archivo
    todavia no existe (no se ha corrido umpire_data.py) o el umpire no
    aparece en el indice."""
    if not os.path.exists(tendency_path):
        return None
    with open(tendency_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["umpire_name"] == umpire_name:
                return {
                    "n_games": int(row["n_games"]), "avg_runs": float(row["avg_runs"]),
                    "runs_vs_league": float(row["runs_vs_league"]),
                    "avg_k": float(row["avg_k"]), "k_vs_league": float(row["k_vs_league"]),
                }
    return None


_ALL_STARTS_CACHE = {}


def get_all_starts(pitcher_id, season):
    """Cacheado en memoria por (pitcher_id, season): dentro de una sola
    corrida del script esto se vuelve a pedir muchas veces para el mismo
    pitcher (similar_avg_runs_splits, head_to_head_asof, etc.) y la lista de
    aperturas de un pitcher no cambia a media corrida, asi que recalcularla
    cada vez es puro desperdicio de llamadas a la API."""
    key = (pitcher_id, season)
    if key in _ALL_STARTS_CACHE:
        return _ALL_STARTS_CACHE[key]
    data = get_json(
        f"/people/{pitcher_id}/stats",
        {"stats": "gameLog", "group": "pitching", "season": season},
    )
    stats = data.get("stats", [])
    if not stats:
        starts = []
    else:
        splits = stats[0].get("splits", [])
        starts = [s for s in splits if s["stat"].get("gamesStarted") == 1]
        starts.sort(key=lambda s: s["date"])
    _ALL_STARTS_CACHE[key] = starts
    return starts


def similar_avg_runs_splits(pitcher_id, season, target_avg, threshold=0.015):
    """Carreras promedio permitidas (1er inning, 1-3, 1-5, y juego completo),
    local y visitante, SOLO en aperturas contra rivales cuyo AVG de temporada
    esta a +/- threshold del target_avg (el AVG del equipo contrario en el
    juego que se esta analizando). Regresa un dict con una entrada por rango:
    result["1to3"], result["1to5"], result["full"], cada uno con
    home_avg_runs/home_n/home_games/away_avg_runs/away_n/away_games."""
    empty = {"home_avg_runs": None, "home_n": 0, "home_games": [],
             "away_avg_runs": None, "away_n": 0, "away_games": []}
    if target_avg is None:
        return {key: dict(empty) for key in SPLIT_RANGES}

    starts = get_all_starts(pitcher_id, season)
    buckets = {key: {"home_runs": [], "away_runs": [], "home_games": [], "away_games": []}
               for key in SPLIT_RANGES}
    for s in starts:
        opp_id = s["opponent"]["id"]
        opp_avg = get_team_season_avg(opp_id, season)
        if opp_avg is None or abs(opp_avg - target_avg) > threshold:
            continue
        gamePk = s["game"]["gamePk"]
        is_home = s["isHome"]
        linescore = get_json(f"/game/{gamePk}/linescore")
        innings = linescore.get("innings", [])
        side = "away" if is_home else "home"
        for key, inning_end in SPLIT_RANGES.items():
            runs = sum((inn.get(side, {}).get("runs", 0) or 0) for inn in innings[:inning_end])
            entry = (s["date"], s["opponent"]["name"], opp_avg, runs)
            bucket = buckets[key]
            if is_home:
                bucket["home_runs"].append(runs)
                bucket["home_games"].append(entry)
            else:
                bucket["away_runs"].append(runs)
                bucket["away_games"].append(entry)

    def avg_or_none(lst):
        return sum(lst) / len(lst) if lst else None

    result = {}
    for key, b in buckets.items():
        result[key] = {
            "home_avg_runs": avg_or_none(b["home_runs"]), "home_n": len(b["home_runs"]), "home_games": b["home_games"],
            "away_avg_runs": avg_or_none(b["away_runs"]), "away_n": len(b["away_runs"]), "away_games": b["away_games"],
        }
    return result


def head_to_head_history(team_id, pitcher_id, seasons, inning_end=9):
    """Como ha bateado REALMENTE la ofensiva de este equipo contra ESTE
    pitcher especifico (no una aproximacion por AVG similar) - juntando
    varias temporadas para tener mas muestra, ya que un equipo normalmente
    solo enfrenta a un abridor rival unas pocas veces por temporada."""
    total_runs = total_hits = total_ab = 0
    per_game = []
    for season in seasons:
        starts = get_all_starts(pitcher_id, season)
        for s in starts:
            if s["opponent"]["id"] != team_id:
                continue
            gamePk = s["game"]["gamePk"]
            pitcher_is_home = s["isHome"]
            team_is_home = not pitcher_is_home
            runs, hits, ab = innings_range_batting_for_team(gamePk, team_is_home, inning_end=inning_end)
            total_runs += runs
            total_hits += hits
            total_ab += ab
            per_game.append((season, s["date"], runs, hits, ab))

    avg = (total_hits / total_ab) if total_ab else None
    avg_runs = (total_runs / len(per_game)) if per_game else None
    return {"avg": avg, "avg_runs": avg_runs, "n_games": len(per_game), "per_game": per_game}


def bullpen_recent_workload(team_id, days=3):
    """Cuantas entradas/pitcheos ha lanzado el BULLPEN (sin contar al abridor
    de cada juego) en los ultimos N dias - una carga alta sugiere posible
    fatiga, mas alla de lo que digan sus stats de temporada completa."""
    today = date.today()
    start = (today - timedelta(days=days)).isoformat()
    end = today.isoformat()
    data = get_json("/schedule", {"sportId": 1, "teamId": team_id, "startDate": start, "endDate": end})
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
                box = get_json(f"/game/{gamePk}/boxscore")
                pitcher_ids = box["teams"][side].get("pitchers", [])
                if not pitcher_ids:
                    continue
                games_included += 1
                for pid in pitcher_ids[1:]:  # [0] es el abridor de ESE juego, el resto es bullpen
                    player = box["teams"][side]["players"].get(f"ID{pid}", {})
                    stats = player.get("stats", {}).get("pitching", {})
                    total_ip += ip_to_decimal(stats.get("inningsPitched", "0.0"))
                    total_pitches += stats.get("numberOfPitches", 0) or 0
    return {"bullpen_ip": total_ip, "bullpen_pitches": total_pitches,
            "games_included": games_included, "days": days}


_PITCHER_HAND_CACHE = {}


def get_pitcher_hand(pitcher_id):
    if pitcher_id in _PITCHER_HAND_CACHE:
        return _PITCHER_HAND_CACHE[pitcher_id]
    hand = None
    try:
        data = get_json(f"/people/{pitcher_id}")
        hand = data["people"][0]["pitchHand"]["code"]
    except (KeyError, IndexError):
        hand = None
    _PITCHER_HAND_CACHE[pitcher_id] = hand
    return hand


_VS_HAND_CACHE = {}


def get_team_vs_hand_split(team_id, season, pitcher_hand):
    """AVG/OPS del equipo bateando especificamente contra pitcheo de esa mano
    (L o R) en toda la temporada - mas relevante que el AVG general cuando el
    abridor de hoy es zurdo (poco comun) y el equipo rival tiene un perfil
    marcado contra zurdos/derechos."""
    if pitcher_hand not in ("L", "R"):
        return None
    key = (team_id, season, pitcher_hand)
    if key in _VS_HAND_CACHE:
        return _VS_HAND_CACHE[key]
    sit_code = "vl" if pitcher_hand == "L" else "vr"
    result = None
    try:
        data = get_json(f"/teams/{team_id}/stats",
                         {"stats": "statSplits", "group": "hitting", "season": season, "sitCodes": sit_code})
        s = data["stats"][0]["splits"][0]["stat"]
        result = {"avg": float(s["avg"]), "ops": float(s["ops"])}
    except (KeyError, IndexError, ValueError, TypeError):
        result = None
    _VS_HAND_CACHE[key] = result
    return result


_PLAYER_VS_HAND_CACHE = {}


def get_player_vs_hand_split(player_id, season, pitcher_hand):
    """Version por jugador de get_team_vs_hand_split: AVG/OPS de ESTE
    bateador especificamente contra pitcheo de esa mano en toda la
    temporada. Solo tiene sentido pedirlo para el lineup CONFIRMADO de
    hoy (9 llamadas por equipo), no para el equipo completo."""
    if pitcher_hand not in ("L", "R"):
        return None
    key = (player_id, season, pitcher_hand)
    if key in _PLAYER_VS_HAND_CACHE:
        return _PLAYER_VS_HAND_CACHE[key]
    sit_code = "vl" if pitcher_hand == "L" else "vr"
    result = None
    try:
        data = get_json(f"/people/{player_id}/stats",
                         {"stats": "statSplits", "group": "hitting", "season": season, "sitCodes": sit_code})
        s = data["stats"][0]["splits"][0]["stat"]
        result = {"avg": float(s["avg"]), "ops": float(s["ops"])}
    except (KeyError, IndexError, ValueError, TypeError):
        result = None
    _PLAYER_VS_HAND_CACHE[key] = result
    return result


def get_player_last_n_games(player_id, season, n=5):
    """Ultimos n juegos de bateo de este jugador en la temporada actual
    (fecha, turnos al bat, hits) - para ver si viene 'caliente' o 'frio'
    mas alla de su AVG de toda la temporada (momentum)."""
    empty = {"avg": None, "hits": 0, "ab": 0, "n_games": 0, "games": []}
    try:
        data = get_json(f"/people/{player_id}/stats",
                         {"stats": "gameLog", "group": "hitting", "season": season})
    except requests.HTTPError:
        return empty
    stats = data.get("stats", [])
    if not stats or not stats[0].get("splits"):
        return empty
    splits = sorted(stats[0]["splits"], key=lambda s: s["date"])[-n:]
    total_ab = total_hits = 0
    games = []
    for s in splits:
        st = s["stat"]
        ab = int(st.get("atBats", 0) or 0)
        hits = int(st.get("hits", 0) or 0)
        total_ab += ab
        total_hits += hits
        games.append({"date": s["date"], "ab": ab, "hits": hits})
    avg = (total_hits / total_ab) if total_ab else None
    return {"avg": avg, "hits": total_hits, "ab": total_ab, "n_games": len(splits), "games": games}


def lineup_batter_detail(players, season, opposing_pitcher_hand):
    """Para cada bateador del lineup ya CONFIRMADO: AVG/OPS contra la mano
    del abridor rival de hoy, y como viene en sus ultimos 5 juegos.
    Regresa un dict {player_id: {...}} para cruzarlo despues con el AVG de
    temporada que ya trae lineup_batting_avg (evita pedirlo dos veces).
    Solo se debe llamar cuando el lineup esta confirmado - antes de eso no
    hay 9 titulares reales que analizar, solo se estaria adivinando."""
    detail = {}
    for p in players:
        player_id = p["id"]
        vs_hand = get_player_vs_hand_split(player_id, season, opposing_pitcher_hand)
        last5 = get_player_last_n_games(player_id, season, n=5)
        detail[player_id] = {
            "vs_hand_avg": vs_hand["avg"] if vs_hand else None,
            "vs_hand_ops": vs_hand["ops"] if vs_hand else None,
            "last5_avg": last5["avg"], "last5_hits": last5["hits"], "last5_ab": last5["ab"],
        }
    return detail


def get_days_rest(pitcher_id, season, game_date):
    """Dias de descanso del abridor antes de esta apertura (dias desde su
    ultima apertura previa a game_date). None si no hay una apertura previa
    esta temporada (ej. es su primer inicio del año)."""
    starts = get_all_starts(pitcher_id, season)
    previous = [s for s in starts if s["date"] < game_date]
    if not previous:
        return None
    last_start_date = previous[-1]["date"]
    d1 = datetime.strptime(last_start_date, "%Y-%m-%d")
    d2 = datetime.strptime(game_date, "%Y-%m-%d")
    return (d2 - d1).days


_PARK_FACTOR_CACHE = {}


def compute_park_factor(team_id, season):
    """Factor de estadio BASICO (metodo simplificado, estilo Baseball-Reference):
    compara cuantas carreras totales (anotadas + permitidas) hace este equipo
    por juego EN CASA vs DE VISITA esta temporada. >1.0 = el parque favorece
    la anotacion (tipo Coors Field), <1.0 = la reduce (parque de pitcheo).
    Se calcula con datos reales de la temporada, no una tabla fija - por eso
    puede ser ruidoso con pocos juegos jugados."""
    if (team_id, season) in _PARK_FACTOR_CACHE:
        return _PARK_FACTOR_CACHE[(team_id, season)]

    data = get_json("/schedule", {"sportId": 1, "teamId": team_id,
                                   "startDate": f"{season}-01-01", "endDate": date.today().isoformat()})
    home_runs_total = home_games = away_runs_total = away_games = 0
    for d in data.get("dates", []):
        for g in d["games"]:
            if g["status"]["detailedState"] != "Final":
                continue
            for side in ("away", "home"):
                if g["teams"][side]["team"]["id"] != team_id:
                    continue
                gamePk = g["gamePk"]
                ls = get_json(f"/game/{gamePk}/linescore")
                innings = ls.get("innings", [])
                team_runs = sum((inn.get(side, {}).get("runs", 0) or 0) for inn in innings)
                other_side = "home" if side == "away" else "away"
                opp_runs = sum((inn.get(other_side, {}).get("runs", 0) or 0) for inn in innings)
                total_runs = team_runs + opp_runs
                if side == "home":
                    home_runs_total += total_runs
                    home_games += 1
                else:
                    away_runs_total += total_runs
                    away_games += 1

    if not home_games or not away_games:
        result = None
    else:
        home_rate = home_runs_total / home_games
        away_rate = away_runs_total / away_games
        result = (home_rate / away_rate) if away_rate else None
    _PARK_FACTOR_CACHE[(team_id, season)] = result
    return result


def team_innings_report(team_id, n=5, inning_end=1):
    games = get_team_last_n_games(team_id, n=n)
    total_runs = 0
    total_hits = 0
    total_ab = 0
    per_game = []
    for gdate, gamePk, is_home in games:
        runs, hits, ab = innings_range_batting_for_team(gamePk, is_home, inning_end=inning_end)
        total_runs += runs
        total_hits += hits
        total_ab += ab
        per_game.append((gdate, runs, hits, ab))
    avg = (total_hits / total_ab) if total_ab else None
    avg_runs = (total_runs / len(games)) if games else None
    return avg, avg_runs, per_game


def ip_to_decimal(ip_str):
    """Convierte '91.2' (91 innings + 2 outs) a decimal (91.667)."""
    whole, _, frac = str(ip_str).partition(".")
    outs = int(whole) * 3 + (int(frac) if frac else 0)
    return outs / 3


def get_pitcher_season_stats(pitcher_id, season, fip_constant=DEFAULT_FIP_CONSTANT):
    data = get_json(
        f"/people/{pitcher_id}/stats",
        {"stats": "season", "group": "pitching", "season": season},
    )
    stats = data.get("stats", [])
    if not stats or not stats[0].get("splits"):
        return None
    s = stats[0]["splits"][0]["stat"]
    ip_decimal = ip_to_decimal(s.get("inningsPitched", "0.0"))
    gs = s.get("gamesStarted", 0)
    hr = s.get("homeRuns", 0)
    bb = s.get("baseOnBalls", 0)
    hbp = s.get("hitBatsmen", 0)
    k = s.get("strikeOuts", 0)
    fip = None
    if ip_decimal > 0:
        fip = (13 * hr + 3 * (bb + hbp) - 2 * k) / ip_decimal + fip_constant
    return {
        "era": s.get("era"),
        "whip": s.get("whip"),
        "ip": s.get("inningsPitched"),
        "ip_decimal": ip_decimal,
        "gs": gs,
        "ip_per_start": (ip_decimal / gs) if gs else None,
        "fip": fip,
    }


def _rate_stats_from_totals(ip_decimal, er, h, bb, hr, k, hbp, fip_constant=DEFAULT_FIP_CONSTANT):
    if ip_decimal <= 0:
        return {"era": None, "whip": None, "fip": None, "ip_decimal": 0}
    era = 9 * er / ip_decimal
    whip = (h + bb) / ip_decimal
    fip = (13 * hr + 3 * (bb + hbp) - 2 * k) / ip_decimal + fip_constant
    return {"era": era, "whip": whip, "fip": fip, "ip_decimal": ip_decimal}


def get_team_season_pitching_stats(team_id, season, fip_constant=DEFAULT_FIP_CONSTANT):
    """ERA/WHIP/FIP agregados de TODO el pitcheo del equipo en la temporada."""
    data = get_json(f"/teams/{team_id}/stats", {"stats": "season", "group": "pitching", "season": season})
    try:
        s = data["stats"][0]["splits"][0]["stat"]
    except (KeyError, IndexError):
        return None
    ip_decimal = ip_to_decimal(s.get("inningsPitched", "0.0"))
    return _rate_stats_from_totals(
        ip_decimal, s.get("earnedRuns", 0), s.get("hits", 0), s.get("baseOnBalls", 0),
        s.get("homeRuns", 0), s.get("strikeOuts", 0), s.get("hitBatsmen", 0), fip_constant,
    )


def get_bullpen_stats(team_id, season, exclude_pitcher_id=None, fip_constant=DEFAULT_FIP_CONSTANT):
    """ERA/WHIP/FIP agregados SOLO de los relevistas puros del equipo (pitchers
    con 0 aperturas esta temporada) - una aproximacion de "el resto del staff
    de pitcheo" que va a tomar la bola despues del abridor."""
    roster = get_json(f"/teams/{team_id}/roster", {"rosterType": "40Man", "season": season})
    total_ip = total_er = total_h = total_bb = total_hr = total_k = total_hbp = 0.0
    n_relievers = 0
    for p in roster.get("roster", []):
        if p["position"]["abbreviation"] != "P":
            continue
        pid = p["person"]["id"]
        if pid == exclude_pitcher_id:
            continue
        data = get_json(f"/people/{pid}/stats", {"stats": "season", "group": "pitching", "season": season})
        try:
            raw_stat = data["stats"][0]["splits"][0]["stat"]
        except (KeyError, IndexError):
            continue
        if raw_stat.get("gamesStarted", 0) > 0:
            continue  # tiene aperturas esta temporada, no es un relevista puro
        ip_decimal = ip_to_decimal(raw_stat.get("inningsPitched", "0.0"))
        if not ip_decimal:
            continue
        n_relievers += 1
        total_ip += ip_decimal
        total_er += raw_stat.get("earnedRuns", 0)
        total_h += raw_stat.get("hits", 0)
        total_bb += raw_stat.get("baseOnBalls", 0)
        total_hr += raw_stat.get("homeRuns", 0)
        total_k += raw_stat.get("strikeOuts", 0)
        total_hbp += raw_stat.get("hitBatsmen", 0)

    result = _rate_stats_from_totals(total_ip, total_er, total_h, total_bb, total_hr, total_k,
                                      total_hbp, fip_constant)
    result["n_relievers"] = n_relievers
    return result


_LEAGUE_AVG_CACHE = {}


def get_league_averages(season, fip_constant=DEFAULT_FIP_CONSTANT):
    """Promedios de ERA/WHIP/FIP/AVG de TODA la liga (30 equipos), para poder
    normalizar que tan buena/mala es una rotacion+bullpen especifica en
    comparacion. Se calcula una sola vez por corrida del programa (cache)."""
    if season in _LEAGUE_AVG_CACHE:
        return _LEAGUE_AVG_CACHE[season]
    teams = get_teams()
    eras, whips, fips, avgs = [], [], [], []
    for t in teams:
        pitching = get_team_season_pitching_stats(t["id"], season, fip_constant=fip_constant)
        if pitching and pitching["era"] is not None:
            eras.append(pitching["era"])
            whips.append(pitching["whip"])
            fips.append(pitching["fip"])
        avg = get_team_season_avg(t["id"], season)
        if avg is not None:
            avgs.append(avg)
    result = {
        "era": sum(eras) / len(eras) if eras else None,
        "whip": sum(whips) / len(whips) if whips else None,
        "fip": sum(fips) / len(fips) if fips else None,
        "avg": sum(avgs) / len(avgs) if avgs else None,
    }
    _LEAGUE_AVG_CACHE[season] = result
    return result


def project_team_runs(team_recent_avg_runs_per_game, starter_stats, bullpen_stats, league_avg,
                       assumed_game_innings=9):
    """Proyecta cuantas carreras anotaria la OFENSIVA de un equipo, ajustando
    su ritmo reciente de anotacion (ultimos 5 juegos, juego completo) segun
    que tan buena es la mezcla abridor+bullpen del RIVAL comparada con el
    promedio de la liga. Formula transparente (no un modelo entrenado):

        indice_calidad = promedio( mezcla_ERA/liga_ERA, mezcla_FIP/liga_FIP,
                                    mezcla_WHIP/liga_WHIP )
        carreras_proyectadas = carreras_recientes_del_equipo * indice_calidad

    indice > 1 = pitcheo rival mas debil que el promedio de la liga (favorece
    mas carreras). indice < 1 = pitcheo rival mas fuerte (menos carreras)."""
    if not starter_stats or not bullpen_stats or not league_avg:
        return None
    if starter_stats.get("era") is None or bullpen_stats.get("era") is None:
        return None
    if not all(league_avg.get(k) for k in ("era", "whip", "fip")):
        return None

    starter_ip = starter_stats.get("ip_per_start") or (assumed_game_innings * 0.6)
    starter_ip = min(starter_ip, assumed_game_innings)
    bullpen_ip = assumed_game_innings - starter_ip

    def blend(stat_name):
        s_val = starter_stats.get(stat_name)
        b_val = bullpen_stats.get(stat_name)
        if s_val is None or b_val is None:
            return None
        return (float(s_val) * starter_ip + float(b_val) * bullpen_ip) / assumed_game_innings

    mezcla_era = blend("era")
    mezcla_whip = blend("whip")
    mezcla_fip = blend("fip")
    if not mezcla_era or not mezcla_whip or not mezcla_fip:
        return None

    # ERA/FIP/WHIP: mientras MAS BAJOS, MEJOR pitcheo (menos carreras esperadas).
    # Por eso el cociente va mezcla/liga (no al reves): si el rival esta POR
    # DEBAJO del promedio de liga (pitcheo bueno), el indice sale <1 y REDUCE
    # la proyeccion; si esta por ENCIMA (pitcheo malo), el indice sale >1 y la
    # aumenta.
    ratios = [mezcla_era / league_avg["era"], mezcla_fip / league_avg["fip"],
              mezcla_whip / league_avg["whip"]]
    indice_calidad = sum(ratios) / len(ratios)
    carreras_proyectadas = team_recent_avg_runs_per_game * indice_calidad

    return {
        "carreras_proyectadas": carreras_proyectadas,
        "indice_calidad_rival": indice_calidad,
        "mezcla_era": mezcla_era, "mezcla_whip": mezcla_whip, "mezcla_fip": mezcla_fip,
        "starter_ip_asumido": starter_ip, "bullpen_ip_asumido": bullpen_ip,
    }


def build_team_report(team_id, team_name, pitcher_id, pitcher_name, season,
                       opponent_avg=None, avg_similarity_threshold=0.015,
                       fip_constant=DEFAULT_FIP_CONSTANT,
                       opponent_team_id=None, is_home=None, ml_bundle=None,
                       ml_bundle_1to3=None, ml_bundle_1to5=None, weather_features=None,
                       opponent_pitcher_id=None, game_date=None):
    pitcher_report = None
    if pitcher_id:
        scoreless, total, details = count_scoreless_first_innings(pitcher_id, season, n=10)
        season_stats = get_pitcher_season_stats(pitcher_id, season, fip_constant=fip_constant)
        similar_avg_splits = similar_avg_runs_splits(
            pitcher_id, season, opponent_avg, threshold=avg_similarity_threshold
        )
        days_rest = get_days_rest(pitcher_id, season, game_date) if game_date else None
        ml_prediction = None
        ml_prediction_1to3 = None
        ml_prediction_1to5 = None
        features = None
        if (ml_bundle or ml_bundle_1to3 or ml_bundle_1to5) and opponent_team_id is not None and is_home is not None:
            from ml_predict import compute_current_features
            import pandas as pd
            features, error = compute_current_features(pitcher_id, opponent_team_id, is_home, season,
                                                         weather_features=weather_features,
                                                         opponent_avg_override=opponent_avg,
                                                         team_id=team_id, game_date=game_date)
            if features is not None:
                if ml_bundle:
                    X = pd.DataFrame([features])[ml_bundle["features"]]
                    prob = ml_bundle["model"].predict_proba(X)[0, 1]
                    ml_prediction = {"prob": prob, "model_name": ml_bundle["model_name"]}
                if ml_bundle_1to3:
                    X3 = pd.DataFrame([features])[ml_bundle_1to3["features"]]
                    runs_1to3 = ml_bundle_1to3["model"].predict(X3)[0]
                    ml_prediction_1to3 = {"runs": runs_1to3, "model_name": ml_bundle_1to3["model_name"]}
                if ml_bundle_1to5:
                    X5 = pd.DataFrame([features])[ml_bundle_1to5["features"]]
                    runs_1to5 = ml_bundle_1to5["model"].predict(X5)[0]
                    ml_prediction_1to5 = {"runs": runs_1to5, "model_name": ml_bundle_1to5["model_name"]}
            else:
                ml_prediction = {"prob": None, "error": error}
                ml_prediction_1to3 = {"runs": None, "error": error}
                ml_prediction_1to5 = {"runs": None, "error": error}
        pitcher_report = {
            "id": pitcher_id,
            "name": pitcher_name,
            "scoreless": scoreless,
            "total_checked": total,
            "details": details,
            "season_stats": season_stats,
            "similar_avg_splits": similar_avg_splits,
            "ml_prediction": ml_prediction,
            "ml_prediction_1to3": ml_prediction_1to3,
            "ml_prediction_1to5": ml_prediction_1to5,
            "ml_features": features,
            "days_rest": days_rest,
        }

    avg_1st, avg_runs_1st, per_game_1st = team_innings_report(team_id, n=5, inning_end=1)
    avg_1to3, avg_runs_1to3, per_game_1to3 = team_innings_report(team_id, n=5, inning_end=3)
    avg_full, avg_runs_full, per_game_full = team_innings_report(team_id, n=5, inning_end=9)

    ofensiva_proyectada = None
    head_to_head = None
    opp_bullpen_workload = None
    if opponent_team_id is not None and opponent_pitcher_id is not None and avg_runs_full is not None:
        league_avg = get_league_averages(season, fip_constant=fip_constant)
        opp_starter_stats = get_pitcher_season_stats(opponent_pitcher_id, season, fip_constant=fip_constant)
        opp_bullpen_stats = get_bullpen_stats(opponent_team_id, season,
                                               exclude_pitcher_id=opponent_pitcher_id,
                                               fip_constant=fip_constant)
        ofensiva_proyectada = project_team_runs(avg_runs_full, opp_starter_stats, opp_bullpen_stats, league_avg)
        if ofensiva_proyectada:
            ofensiva_proyectada["bullpen_n_relievers"] = opp_bullpen_stats.get("n_relievers")
            ofensiva_proyectada["carreras_recientes_base"] = avg_runs_full

        head_to_head = head_to_head_history(team_id, opponent_pitcher_id, [season, season - 1])
        opp_bullpen_workload = bullpen_recent_workload(opponent_team_id, days=3)

    vs_hand_split = None
    if opponent_pitcher_id is not None:
        opp_hand = get_pitcher_hand(opponent_pitcher_id)
        if opp_hand:
            split = get_team_vs_hand_split(team_id, season, opp_hand)
            if split:
                vs_hand_split = {"hand": opp_hand, "avg": split["avg"], "ops": split["ops"]}

    hitting_stats = get_team_season_hitting_stats(team_id, season)

    return {
        "team_name": team_name,
        "team_id": team_id,
        "pitcher": pitcher_report,
        "avg_1st": avg_1st,
        "avg_runs_1st": avg_runs_1st,
        "per_game": per_game_1st,
        "avg_1to3": avg_1to3,
        "avg_runs_1to3": avg_runs_1to3,
        "per_game_1to3": per_game_1to3,
        "avg_full": avg_full,
        "avg_runs_full": avg_runs_full,
        "per_game_full": per_game_full,
        "ofensiva_proyectada": ofensiva_proyectada,
        "head_to_head": head_to_head,
        "opp_bullpen_workload": opp_bullpen_workload,
        "hitting_stats": hitting_stats,
        "vs_hand_split": vs_hand_split,
    }


def get_venue_details(venue_id):
    data = get_json(f"/venues/{venue_id}", {"hydrate": "location"})
    v = data["venues"][0]
    loc = v.get("location", {})
    return {
        "name": v["name"],
        "city": loc.get("city"),
        "state": loc.get("stateAbbrev"),
        "lat": loc.get("defaultCoordinates", {}).get("latitude"),
        "lon": loc.get("defaultCoordinates", {}).get("longitude"),
        "azimuth": loc.get("azimuthAngle"),
    }


def get_weather_forecast(lat, lon, game_date, hour=19):
    if lat is None or lon is None:
        return None
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,precipitation_probability,windspeed_10m,winddirection_10m",
        "temperature_unit": "celsius",
        "windspeed_unit": "mph",
        "timezone": "auto",
        "start_date": game_date,
        "end_date": game_date,
    }
    r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    target = f"{game_date}T{hour:02d}:00"
    idx = None
    for i, t in enumerate(times):
        if t == target:
            idx = i
            break
    if idx is None and times:
        idx = len(times) // 2
    if idx is None:
        return None
    return {
        "temp_c": hourly["temperature_2m"][idx],
        "precip_prob": hourly["precipitation_probability"][idx],
        "wind_mph": hourly["windspeed_10m"][idx],
        "wind_dir_from": hourly["winddirection_10m"][idx],
        "time": times[idx],
    }


def classify_wind(wind_dir_from, azimuth):
    if wind_dir_from is None or azimuth is None:
        return "N/D (falta azimuth del estadio o direccion de viento)"
    diff = (wind_dir_from - azimuth + 180) % 360 - 180
    if abs(diff) <= 45:
        return "sopla DE los jardines HACIA el plato (adentro, favorece pitchers)"
    if abs(diff) >= 135:
        return "sopla DEL plato HACIA los jardines (afuera, favorece bateadores/jonrones)"
    return "viento cruzado (lateral, ni claramente adentro ni afuera)"


ROOF_TEMP_THRESHOLD_C = 15.6  # equivalente a 60F, umbral que usan los Brewers para el techo


def roof_status_heuristic(weather):
    if not weather:
        return "N/D"
    if weather["precip_prob"] >= 40 or weather["temp_c"] < ROOF_TEMP_THRESHOLD_C:
        return "probablemente CERRADO (estimado por clima, no es decision oficial del equipo)"
    return "probablemente ABIERTO (estimado por clima, no es decision oficial del equipo)"


def print_report(matchup, report_a, report_b, weather, whip_threshold=DEFAULT_WHIP_THRESHOLD,
                  injuries_info=None, park_factor=None, lineup_info=None, umpire_info=None):
    def pct(scoreless, total):
        if not total:
            return "N/D"
        return f"{scoreless}/{total} ({scoreless/total*100:.1f}%, {confidence_label(total)})"

    def fmt_avg(v):
        if v is None:
            return "N/D"
        return f"{v:.3f}".lstrip("0") if v < 1 else f"{v:.3f}"

    a, b = report_a, report_b
    print()
    print("=" * 72)
    print(f"REPORTE PRE-JUEGO: {a['team_name']} @ {b['team_name']}" if matchup else
          f"REPORTE: {a['team_name']} vs {b['team_name']}")
    print("=" * 72)

    if matchup:
        print()
        print(f"Fecha:              {matchup['date']}")
        print(f"Estadio:            {matchup['venue']['name']} ({matchup['venue']['city']}, {matchup['venue']['state']})")
        print(f"Tipo de techo:      {ROOF_TYPES.get(b['team_name'], 'N/D')}")
        if weather:
            print(f"Clima pronosticado: {weather['temp_c']}C (~{weather['time']}), "
                  f"prob. lluvia {weather['precip_prob']}%")
            print(f"Viento:             {weather['wind_mph']} mph, direccion origen {weather['wind_dir_from']} "
                  f"grados -> {classify_wind(weather['wind_dir_from'], matchup['venue']['azimuth'])}")
            print(f"Techo (heuristica): {roof_status_heuristic(weather)}")
        else:
            print("Clima:              N/D (no se pudo obtener pronostico)")
        if park_factor is not None:
            tendencia = "favorece anotacion" if park_factor > 1.05 else \
                "reduce anotacion" if park_factor < 0.95 else "neutral"
            print(f"Factor de estadio:  {park_factor:.3f} ({tendencia}, calculado con datos reales "
                  f"local vs. visitante de esta temporada)")
        if umpire_info:
            t = umpire_info.get("tendency")
            if t:
                print(f"Umpire home plate:  {umpire_info['name']} - "
                      f"proxy carreras/juego dirigido: {t['runs_vs_league']:.2f}x liga "
                      f"({confidence_label(t['n_games'])}, PROXY: no es zona de strike real, "
                      f"ver umpire_data.py)")
            else:
                print(f"Umpire home plate:  {umpire_info['name']} (sin tendencia calculada todavia - "
                      f"corre umpire_data.py para generar umpire_tendency.csv)")
        else:
            print("Umpire home plate:  N/D (MLB todavia no lo confirma, normalmente sale horas antes del juego)")

    if injuries_info:
        print()
        print("-" * 72)
        print("LESIONADOS EN LA OFENSIVA (afecta AVG usado en similar_avg y ML)")
        print("-" * 72)
        for team_name, info in injuries_info.items():
            print()
            raw, adj = info["raw_avg"], info["adj_avg"]
            if raw is not None and adj is not None:
                print(f"{team_name}: AVG temporada {raw:.3f}  ->  AVG ajustado sin lesionados: {adj:.3f}")
            else:
                print(f"{team_name}: AVG N/D")
            if not info["injured"]:
                print("    (sin lesionados de posicion actualmente)")
            for p in info["injured"]:
                avg_txt = f"{p['avg']:.3f}" if p["avg"] is not None else "N/D"
                print(f"    LESIONADO: {p['name']:<22} ({p['position']}, {p['status']}) - AVG {avg_txt} ({p['ab']} AB)")

    if lineup_info:
        print()
        print("-" * 72)
        print("LINEUP TITULAR CONFIRMADO (solo en vivo; MLB lo publica ~1-3h antes del primer pitch)")
        print("-" * 72)
        for team_name, info in lineup_info.items():
            print()
            if not info["confirmed"]:
                print(f"{team_name}: lineup TODAVIA NO CONFIRMADO - se usa AVG ajustado por lesionados como respaldo.")
                continue
            avg_txt = f"{info['avg']:.3f}" if info["avg"] is not None else "N/D"
            print(f"{team_name}: lineup CONFIRMADO, AVG combinado de los 9 titulares: {avg_txt}")
            for p in info["detail"]:
                avg_p = f"{p['avg']:.3f}" if p["avg"] is not None else "N/D"
                print(f"    {p['name']:<22} AVG {avg_p} ({p['ab']} AB)")

    def fmt_runs(v, n):
        if v is None:
            return f"N/D (0 juegos que cumplan el filtro de AVG similar)"
        return f"{round(v, 3)} (muestra: {n} juego{'s' if n != 1 else ''}, {confidence_label(n)})"

    def print_pitcher_block(idx, rep, rival_name):
        p = rep["pitcher"]
        print()
        print(f"{idx}) Abridor {rep['team_name']}: {p['name'] if p else 'N/D'}")
        if p:
            ss = p["season_stats"]
            if ss:
                print(f"     ERA: {ss['era']}  |  WHIP: {ss['whip']} "
                      f"({classify_whip(ss['whip'], threshold=whip_threshold)})  |  "
                      f"FIP (aprox.): {round(ss['fip'], 2) if ss['fip'] is not None else 'N/D'}")
                print(f"     Entradas por apertura (temporada): "
                      f"{round(ss['ip_per_start'], 2) if ss['ip_per_start'] else 'N/D'} "
                      f"({ss['ip']} IP en {ss['gs']} aperturas)")
            dr = p.get("days_rest")
            if dr is not None:
                alerta = " (descanso corto, posible fatiga)" if dr < 4 else ""
                print(f"     Dias de descanso antes de esta apertura: {dr}{alerta}")
            print(f"     Veces con -0.5 carreras (0 carreras) en el 1er inning, "
                  f"ultimas {p['total_checked']} aperturas: "
                  f"{pct(p['scoreless'], p['total_checked'])}")
            n_prior = (p.get("ml_features") or {}).get("n_prior_starts")
            ml_conf = f" [{confidence_label(n_prior)}]" if n_prior is not None else ""
            mlp = p.get("ml_prediction")
            if mlp:
                if mlp.get("prob") is not None:
                    print(f"     Prediccion ML ({mlp['model_name']}): "
                          f"{mlp['prob']*100:.1f}% de que EL PITCHER deje 0 carreras PERMITIDAS en el 1er inning"
                          f"{ml_conf}")
                else:
                    print(f"     Prediccion ML: N/D ({mlp.get('error')})")
            mlp3 = p.get("ml_prediction_1to3")
            if mlp3:
                if mlp3.get("runs") is not None:
                    print(f"     Prediccion ML ({mlp3['model_name']}): "
                          f"{mlp3['runs']:.2f} carreras PERMITIDAS POR EL PITCHER en entradas 1-3 "
                          f"(carreras que le anota el rival a el){ml_conf}")
                else:
                    print(f"     Prediccion ML (1-3 entradas): N/D ({mlp3.get('error')})")
            mlp5 = p.get("ml_prediction_1to5")
            if mlp5:
                if mlp5.get("runs") is not None:
                    print(f"     Prediccion ML ({mlp5['model_name']}): "
                          f"{mlp5['runs']:.2f} carreras PERMITIDAS POR EL PITCHER en entradas 1-5 "
                          f"(carreras que le anota el rival a el){ml_conf}")
                else:
                    print(f"     Prediccion ML (1-5 entradas): N/D ({mlp5.get('error')})")
            sas = p["similar_avg_splits"]
            range_labels = [("1to3", "entradas 1-3"), ("1to5", "entradas 1-5"), ("full", "el juego completo")]
            for key, label_txt in range_labels:
                s = sas[key]
                print(f"     Carreras permitidas ({label_txt}) de LOCAL, vs rivales con AVG "
                      f"similar al de {rival_name}: {fmt_runs(s['home_avg_runs'], s['home_n'])}")
                print(f"     Carreras permitidas ({label_txt}) de VISITANTE, vs rivales con AVG "
                      f"similar al de {rival_name}: {fmt_runs(s['away_avg_runs'], s['away_n'])}")

    def print_ofensiva_proyectada(idx, rep, rival_pitcher_name):
        op = rep.get("ofensiva_proyectada")
        print()
        print(f"{idx}) Ofensiva proyectada de {rep['team_name']} (formula, no modelo entrenado)")
        if not op:
            print("     N/D (falta el abridor rival o datos suficientes)")
            return
        print(f"     Carreras recientes del equipo (ultimos 5 juegos, juego completo): "
              f"{round(op['carreras_recientes_base'], 2)}")
        print(f"     Pitcheo rival ({rival_pitcher_name} + bullpen, {op['bullpen_n_relievers']} relevistas), "
              f"mezcla ponderada por innings esperados ({round(op['starter_ip_asumido'], 1)} abridor + "
              f"{round(op['bullpen_ip_asumido'], 1)} bullpen):")
        print(f"       ERA mezcla: {round(op['mezcla_era'], 2)}  |  "
              f"WHIP mezcla: {round(op['mezcla_whip'], 2)}  |  "
              f"FIP mezcla: {round(op['mezcla_fip'], 2)}")
        print(f"     Indice de calidad del pitcheo rival vs. promedio de liga: "
              f"{round(op['indice_calidad_rival'], 3)} "
              f"({'mas debil que la liga, favorece carreras' if op['indice_calidad_rival'] > 1 else 'mas fuerte que la liga, reduce carreras'})")
        print(f"     >>> Carreras proyectadas: {round(op['carreras_proyectadas'], 2)}")

        hs = rep.get("hitting_stats")
        if hs and hs.get("ops") is not None:
            print(f"     OPS de temporada de {rep['team_name']} (OBP+SLG, mejor indicador que solo AVG): "
                  f"{hs['ops']:.3f}  (OBP {hs['obp']:.3f} + SLG {hs['slg']:.3f})")

        vhs = rep.get("vs_hand_split")
        if vhs:
            mano_txt = "zurdos" if vhs["hand"] == "L" else "derechos"
            print(f"     {rep['team_name']} bateando vs. pitcheo {mano_txt} esta temporada "
                  f"(el abridor rival es {mano_txt}): AVG {vhs['avg']:.3f}, OPS {vhs['ops']:.3f}")

        h2h = rep.get("head_to_head")
        if h2h and h2h["n_games"] > 0:
            print(f"     Historial REAL de {rep['team_name']} vs {rival_pitcher_name} "
                  f"(temporada actual + anterior, {h2h['n_games']} juego(s), {confidence_label(h2h['n_games'])}): "
                  f"AVG {h2h['avg']:.3f}, {round(h2h['avg_runs'], 2)} carreras/juego")
            for season_yr, gdate, runs, hits, ab in h2h["per_game"]:
                print(f"       {gdate} ({season_yr}): {runs} carreras, {hits}-{ab}")
        elif h2h is not None:
            print(f"     Historial REAL de {rep['team_name']} vs {rival_pitcher_name}: "
                  f"sin enfrentamientos previos en temporada actual/anterior")

        wl = rep.get("opp_bullpen_workload")
        if wl and wl["games_included"] > 0:
            fatiga = "ALTA - posible fatiga" if wl["bullpen_ip"] >= 9 else "normal"
            print(f"     Carga reciente del bullpen rival (ultimos {wl['days']} dias, "
                  f"{wl['games_included']} juego(s)): {round(wl['bullpen_ip'], 1)} IP, "
                  f"{int(wl['bullpen_pitches'])} pitcheos -> {fatiga}")

    print_pitcher_block(1, a, b["team_name"])
    print_pitcher_block(2, b, a["team_name"])

    print()
    print("-" * 72)
    print("OFENSIVA PROYECTADA (considera abridor + bullpen rival)")
    print("-" * 72)
    print_ofensiva_proyectada(1, a, b["pitcher"]["name"] if b["pitcher"] else "N/D")
    print_ofensiva_proyectada(2, b, a["pitcher"]["name"] if a["pitcher"] else "N/D")

    print()
    print("-" * 72)
    print("AVERAGE Y CARRERAS DE LA OFENSIVA (ultimos 5 juegos)")
    print("-" * 72)

    print(f"\n3) Average de bateo DE LA OFENSIVA de {a['team_name']} en el 1er inning "
          f"(ultimos {len(a['per_game'])} juegos): {fmt_avg(a['avg_1st'])}")
    print(f"4) Average de bateo DE LA OFENSIVA de {b['team_name']} en el 1er inning "
          f"(ultimos {len(b['per_game'])} juegos): {fmt_avg(b['avg_1st'])}")

    print(f"\n5) Carreras ANOTADAS POR LA OFENSIVA de {a['team_name']} en el 1er inning "
          f"(ultimos {len(a['per_game'])} juegos): "
          f"{'N/D' if a['avg_runs_1st'] is None else round(a['avg_runs_1st'], 3)}")
    print(f"6) Carreras ANOTADAS POR LA OFENSIVA de {b['team_name']} en el 1er inning "
          f"(ultimos {len(b['per_game'])} juegos): "
          f"{'N/D' if b['avg_runs_1st'] is None else round(b['avg_runs_1st'], 3)}")

    print(f"\n7) Average de bateo DE LA OFENSIVA de {a['team_name']} en entradas 1ra-3ra "
          f"(ultimos {len(a['per_game_1to3'])} juegos): {fmt_avg(a['avg_1to3'])}")
    print(f"8) Average de bateo DE LA OFENSIVA de {b['team_name']} en entradas 1ra-3ra "
          f"(ultimos {len(b['per_game_1to3'])} juegos): {fmt_avg(b['avg_1to3'])}")

    print(f"\n9) Carreras ANOTADAS POR LA OFENSIVA de {a['team_name']} en entradas 1ra-3ra "
          f"(ultimos {len(a['per_game_1to3'])} juegos): "
          f"{'N/D' if a['avg_runs_1to3'] is None else round(a['avg_runs_1to3'], 3)}")
    print(f"10) Carreras ANOTADAS POR LA OFENSIVA de {b['team_name']} en entradas 1ra-3ra "
          f"(ultimos {len(b['per_game_1to3'])} juegos): "
          f"{'N/D' if b['avg_runs_1to3'] is None else round(b['avg_runs_1to3'], 3)}")

    print(f"\n11) Average de bateo DE LA OFENSIVA de {a['team_name']} en el juego completo "
          f"(ultimos {len(a['per_game_full'])} juegos): {fmt_avg(a['avg_full'])}")
    print(f"12) Average de bateo DE LA OFENSIVA de {b['team_name']} en el juego completo "
          f"(ultimos {len(b['per_game_full'])} juegos): {fmt_avg(b['avg_full'])}")

    print(f"\n13) Carreras ANOTADAS POR LA OFENSIVA de {a['team_name']} en el juego completo "
          f"(ultimos {len(a['per_game_full'])} juegos): "
          f"{'N/D' if a['avg_runs_full'] is None else round(a['avg_runs_full'], 3)}")
    print(f"14) Carreras ANOTADAS POR LA OFENSIVA de {b['team_name']} en el juego completo "
          f"(ultimos {len(b['per_game_full'])} juegos): "
          f"{'N/D' if b['avg_runs_full'] is None else round(b['avg_runs_full'], 3)}")

    print()
    print("-" * 72)
    print("DETALLE POR JUEGO (1er inning)")
    print("-" * 72)
    for rep in (a, b):
        print(f"\n{rep['team_name']}:")
        for gdate, runs, hits, ab in rep["per_game"]:
            print(f"    {gdate}:  {runs} carreras,  {hits}-{ab} bateando en el 1er inning")

    print()
    print("-" * 72)
    print("DETALLE POR JUEGO (entradas 1ra-3ra)")
    print("-" * 72)
    for rep in (a, b):
        print(f"\n{rep['team_name']}:")
        for gdate, runs, hits, ab in rep["per_game_1to3"]:
            print(f"    {gdate}:  {runs} carreras,  {hits}-{ab} bateando en entradas 1-3")

    print()
    print("-" * 72)
    print("DETALLE POR JUEGO (juego completo)")
    print("-" * 72)
    for rep in (a, b):
        print(f"\n{rep['team_name']}:")
        for gdate, runs, hits, ab in rep["per_game_full"]:
            print(f"    {gdate}:  {runs} carreras,  {hits}-{ab} bateando en el juego")

    print()
    print("-" * 72)
    print("DETALLE SPLITS AVG SIMILAR (1er inning / 1-3 / 1-5 / juego completo, por apertura)")
    print("-" * 72)
    for rep, rival_name in ((a, b["team_name"]), (b, a["team_name"])):
        p = rep["pitcher"]
        if not p:
            continue
        sas = p["similar_avg_splits"]
        print(f"\n{p['name']} ({rep['team_name']}) - rivales con AVG similar a {rival_name}:")
        for side_key, side_label in (("home_games", "Local"), ("away_games", "Visitante")):
            print(f"  {side_label}:")
            games_1to3 = sas["1to3"][side_key]
            games_1to5 = sas["1to5"][side_key]
            games_full = sas["full"][side_key]
            for (gdate, opp_name, opp_avg, r1to3), (_, _, _, r1to5), (_, _, _, rfull) in \
                    zip(games_1to3, games_1to5, games_full):
                print(f"    {gdate} vs {opp_name} (AVG {opp_avg:.3f}): "
                      f"{r1to3} carreras (1-3)  |  {r1to5} carreras (1-5)  |  {rfull} carreras (juego)")
    print()


def report_to_rows(matchup, report_a, report_b, weather, whip_threshold=DEFAULT_WHIP_THRESHOLD,
                    injuries_info=None, park_factor=None):
    """Convierte el reporte a pares [etiqueta, valor] en un orden fijo, para
    escribirlos siempre en las mismas celdas de la hoja (plantilla fija)."""
    a, b = report_a, report_b

    def fmt_avg(v):
        if v is None:
            return "N/D"
        return f"{v:.3f}".lstrip("0") if v < 1 else f"{v:.3f}"

    def pct(scoreless, total):
        return f"{scoreless}/{total}" if total else "N/D"

    rows = [
        ["Reporte MLB", f"{a['team_name']} vs {b['team_name']}"],
        ["Fecha", matchup["date"] if matchup else "N/D"],
        ["Estadio", matchup["venue"]["name"] if matchup else "N/D"],
        ["Tipo de techo", ROOF_TYPES.get(b["team_name"], "N/D")],
        ["Clima temp (C)", weather["temp_c"] if weather else "N/D"],
        ["Prob. lluvia (%)", weather["precip_prob"] if weather else "N/D"],
        ["Viento (mph)", weather["wind_mph"] if weather else "N/D"],
        ["Viento direccion (grados)", weather["wind_dir_from"] if weather else "N/D"],
        ["Viento relativo", classify_wind(weather["wind_dir_from"], matchup["venue"]["azimuth"])
         if weather and matchup else "N/D"],
        ["Estatus techo (heuristica)", roof_status_heuristic(weather) if weather else "N/D"],
        ["Factor de estadio", round(park_factor, 3) if park_factor is not None else "N/D"],
    ]

    if injuries_info:
        for team_name, info in injuries_info.items():
            rows.append([f"{team_name} - AVG temporada (ofensiva)",
                         round(info["raw_avg"], 3) if info["raw_avg"] is not None else "N/D"])
            rows.append([f"{team_name} - AVG ajustado sin lesionados",
                         round(info["adj_avg"], 3) if info["adj_avg"] is not None else "N/D"])
            rows.append([f"{team_name} - Lesionados (ofensiva)",
                         "; ".join(f"{p['name']} ({p['position']}, AVG {p['avg']:.3f})"
                                   if p["avg"] is not None else f"{p['name']} ({p['position']})"
                                   for p in info["injured"]) or "Ninguno"])

    for label, rep in ((f"Equipo A ({a['team_name']})", a), (f"Equipo B ({b['team_name']})", b)):
        p = rep["pitcher"]
        ss = p["season_stats"] if p else None
        sas = p["similar_avg_splits"] if p else None
        rows.extend([
            [f"{label} - Abridor", p["name"] if p else "N/D"],
            [f"{label} - ERA", ss["era"] if ss else "N/D"],
            [f"{label} - WHIP", ss["whip"] if ss else "N/D"],
            [f"{label} - WHIP Clasificacion",
             classify_whip(ss["whip"], threshold=whip_threshold) if ss else "N/D"],
            [f"{label} - FIP (aprox.)", round(ss["fip"], 2) if ss and ss["fip"] is not None else "N/D"],
            [f"{label} - IP por apertura", round(ss["ip_per_start"], 2) if ss and ss["ip_per_start"] else "N/D"],
            [f"{label} - Dias de descanso", p["days_rest"] if p and p.get("days_rest") is not None else "N/D"],
            [f"{label} - 0 carreras 1er inning (ult. aperturas)",
             pct(p["scoreless"], p["total_checked"]) if p else "N/D"],
            [f"{label} - Prediccion ML (prob. 0 carreras 1er inning)",
             round(p["ml_prediction"]["prob"] * 100, 1) if p and p.get("ml_prediction") and
             p["ml_prediction"].get("prob") is not None else "N/D"],
            [f"{label} - Prediccion ML (carreras PERMITIDAS POR EL PITCHER, entradas 1-3)",
             round(p["ml_prediction_1to3"]["runs"], 2) if p and p.get("ml_prediction_1to3") and
             p["ml_prediction_1to3"].get("runs") is not None else "N/D"],
            [f"{label} - Prediccion ML (carreras PERMITIDAS POR EL PITCHER, entradas 1-5)",
             round(p["ml_prediction_1to5"]["runs"], 2) if p and p.get("ml_prediction_1to5") and
             p["ml_prediction_1to5"].get("runs") is not None else "N/D"],
        ])
        for range_key, range_label in (("1to3", "1-3"), ("1to5", "1-5"), ("full", "juego completo")):
            s = sas[range_key] if sas else None
            rows.extend([
                [f"{label} - Carreras PERMITIDAS POR EL PITCHER, {range_label} LOCAL vs AVG similar",
                 round(s["home_avg_runs"], 3) if s and s["home_avg_runs"] is not None else "N/D"],
                [f"{label} - Muestra LOCAL vs AVG similar ({range_label})", s["home_n"] if s else "N/D"],
                [f"{label} - Carreras PERMITIDAS POR EL PITCHER, {range_label} VISITANTE vs AVG similar",
                 round(s["away_avg_runs"], 3) if s and s["away_avg_runs"] is not None else "N/D"],
                [f"{label} - Muestra VISITANTE vs AVG similar ({range_label})", s["away_n"] if s else "N/D"],
            ])
        rows.extend([
            [f"{label} - AVG bateo DE LA OFENSIVA, 1er inning (5 juegos)", fmt_avg(rep["avg_1st"])],
            [f"{label} - Carreras ANOTADAS POR LA OFENSIVA, 1er inning (5 juegos)",
             round(rep["avg_runs_1st"], 3) if rep["avg_runs_1st"] is not None else "N/D"],
            [f"{label} - AVG bateo DE LA OFENSIVA, entradas 1-3 (5 juegos)", fmt_avg(rep["avg_1to3"])],
            [f"{label} - Carreras ANOTADAS POR LA OFENSIVA, entradas 1-3 (5 juegos)",
             round(rep["avg_runs_1to3"], 3) if rep["avg_runs_1to3"] is not None else "N/D"],
            [f"{label} - AVG bateo DE LA OFENSIVA, juego completo (5 juegos)", fmt_avg(rep["avg_full"])],
            [f"{label} - Carreras ANOTADAS POR LA OFENSIVA, juego completo (5 juegos)",
             round(rep["avg_runs_full"], 3) if rep["avg_runs_full"] is not None else "N/D"],
            [f"{label} - Ofensiva proyectada (formula, considera bullpen rival)",
             round(rep["ofensiva_proyectada"]["carreras_proyectadas"], 2)
             if rep.get("ofensiva_proyectada") else "N/D"],
            [f"{label} - Indice calidad pitcheo rival (formula)",
             round(rep["ofensiva_proyectada"]["indice_calidad_rival"], 3)
             if rep.get("ofensiva_proyectada") else "N/D"],
            [f"{label} - OPS de temporada", round(rep["hitting_stats"]["ops"], 3)
             if rep.get("hitting_stats") and rep["hitting_stats"].get("ops") is not None else "N/D"],
            [f"{label} - Historial real vs abridor rival (AVG)",
             round(rep["head_to_head"]["avg"], 3)
             if rep.get("head_to_head") and rep["head_to_head"]["avg"] is not None else "N/D"],
            [f"{label} - Historial real vs abridor rival (carreras/juego)",
             round(rep["head_to_head"]["avg_runs"], 2)
             if rep.get("head_to_head") and rep["head_to_head"]["avg_runs"] is not None else "N/D"],
            [f"{label} - Muestra historial real vs abridor rival",
             rep["head_to_head"]["n_games"] if rep.get("head_to_head") else "N/D"],
            [f"{label} - Carga bullpen rival ultimos 3 dias (IP)",
             round(rep["opp_bullpen_workload"]["bullpen_ip"], 1)
             if rep.get("opp_bullpen_workload") else "N/D"],
            [f"{label} - AVG vs mano del abridor rival",
             round(rep["vs_hand_split"]["avg"], 3) if rep.get("vs_hand_split") else "N/D"],
            [f"{label} - OPS vs mano del abridor rival",
             round(rep["vs_hand_split"]["ops"], 3) if rep.get("vs_hand_split") else "N/D"],
        ])

    return rows


def write_to_google_sheet(rows, spreadsheet_id, sheet_tab, credentials_path, start_cell="A1"):
    try:
        import gspread
    except ImportError:
        print("Falta instalar gspread: pip install gspread google-auth", file=sys.stderr)
        raise

    if not os.path.exists(credentials_path):
        raise FileNotFoundError(
            f"No encontre el archivo de credenciales '{credentials_path}'. "
            f"Revisa la guia de configuracion (README) para crear una cuenta de "
            f"servicio de Google Cloud y descargar su JSON."
        )

    gc = gspread.service_account(filename=credentials_path)
    sh = gc.open_by_key(spreadsheet_id)
    try:
        ws = sh.worksheet(sheet_tab)
    except gspread.exceptions.WorksheetNotFound:
        ws = sh.add_worksheet(title=sheet_tab, rows=max(len(rows) + 5, 20), cols=5)

    ws.update(range_name=start_cell, values=rows)
    print(f"Escrito en Google Sheet: pestana '{sheet_tab}', {len(rows)} filas desde {start_cell}.")


def maybe_ask_gemini(question):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[Gemini] GEMINI_API_KEY no esta configurada, se omite.", file=sys.stderr)
        return None
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.5-flash:generateContent?key={api_key}"
    )
    payload = {"contents": [{"parts": [{"text": question}]}]}
    r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        return None


def main():
    parser = argparse.ArgumentParser(description="Reporte pre-apuesta MLB para dos equipos")
    parser.add_argument("equipo_a", help="Nombre o abreviacion del equipo A (visitante o cualquiera), ej. 'Reds'")
    parser.add_argument("equipo_b", help="Nombre o abreviacion del equipo B, ej. 'Brewers'")
    parser.add_argument("--season", type=int, default=date.today().year)
    parser.add_argument("--fip-constant", type=float, default=DEFAULT_FIP_CONSTANT)
    parser.add_argument("--avg-similarity-threshold", type=float, default=0.015,
                         help="Diferencia maxima de AVG para considerar un rival 'similar' (default 0.015)")
    parser.add_argument("--whip-threshold", type=float, default=DEFAULT_WHIP_THRESHOLD,
                         help=f"WHIP <= este valor se clasifica 'Aprobada', si no 'Riesgosa' (default {DEFAULT_WHIP_THRESHOLD})")
    parser.add_argument("--use-gemini", action="store_true",
                         help="Permite usar Gemini como respaldo narrativo (requiere GEMINI_API_KEY)")
    parser.add_argument("--sheet-id", default=os.environ.get("MLB_SHEET_ID"),
                         help="ID de tu Google Spreadsheet (o define MLB_SHEET_ID). Si se da, escribe el reporte ahi.")
    parser.add_argument("--sheet-tab", default="Reporte MLB",
                         help="Nombre de la pestana donde escribir (default 'Reporte MLB')")
    parser.add_argument("--sheet-start-cell", default="A1")
    parser.add_argument("--credentials", default=os.environ.get(
        "GOOGLE_SHEETS_CREDENTIALS", "credentials.json"),
        help="Ruta al JSON de la cuenta de servicio de Google (o define GOOGLE_SHEETS_CREDENTIALS)")
    parser.add_argument("--ml-model", default="model.joblib",
                         help="Ruta al modelo de clasificacion (0 carreras 1er inning). Si existe, se agrega "
                              "al reporte. Si no existe el archivo, se omite sin error.")
    parser.add_argument("--ml-model-1to3", default="model_1to3.joblib",
                         help="Ruta al modelo de regresion (carreras esperadas entradas 1-3). Igual, opcional.")
    parser.add_argument("--ml-model-1to5", default="model_1to5.joblib",
                         help="Ruta al modelo de regresion (carreras esperadas entradas 1-5). Igual, opcional.")
    parser.add_argument("--no-ml", action="store_true", help="No incluir ninguna prediccion de ML aunque existan los modelos")
    parser.add_argument("--log-path", default="predictions_log.csv",
                         help="Donde guardar el historial de predicciones para evaluarlas despues (ml_track.py)")
    parser.add_argument("--no-log", action="store_true", help="No guardar esta prediccion en el log de seguimiento")
    parser.add_argument("--no-park-factor", action="store_true",
                         help="Omite el calculo de factor de estadio (tarda ~20s por revisar toda la temporada)")
    parser.add_argument("--pitcher-a", default=None,
                         help="Forzar manualmente el abridor del Equipo A (nombre o parte del nombre), "
                              "por si la API de MLB aun no refleja un cambio de rotacion reciente")
    parser.add_argument("--pitcher-b", default=None,
                         help="Forzar manualmente el abridor del Equipo B (nombre o parte del nombre)")
    args = parser.parse_args()

    ml_bundle = None
    ml_bundle_1to3 = None
    ml_bundle_1to5 = None
    if not args.no_ml:
        import joblib
        if args.ml_model and os.path.exists(args.ml_model):
            ml_bundle = joblib.load(args.ml_model)
        if args.ml_model_1to3 and os.path.exists(args.ml_model_1to3):
            ml_bundle_1to3 = joblib.load(args.ml_model_1to3)
        if args.ml_model_1to5 and os.path.exists(args.ml_model_1to5):
            ml_bundle_1to5 = joblib.load(args.ml_model_1to5)

    teams = get_teams()
    team_a = resolve_team(args.equipo_a, teams)
    team_b = resolve_team(args.equipo_b, teams)

    if team_a["id"] == team_b["id"]:
        print(f"ERROR: '{args.equipo_a}' y '{args.equipo_b}' se resolvieron al mismo equipo "
              f"({team_a['name']}). Revisa los nombres (ej. no pongas 'Colorado' y 'Rockies', "
              f"son el mismo equipo) y vuelve a correr el reporte con el rival correcto.")
        return

    matchup_info = find_matchup_game(team_a["id"], team_b["id"]) or \
        find_matchup_game(team_b["id"], team_a["id"])

    pid_a = pname_a = pid_b = pname_b = None
    weather = None
    matchup = None
    is_home_a = is_home_b = None
    weather_features = None
    umpire_info = None

    if matchup_info:
        pid_a, pname_a = matchup_info["pitchers"].get(team_a["id"], (None, None))
        pid_b, pname_b = matchup_info["pitchers"].get(team_b["id"], (None, None))
        venue = get_venue_details(matchup_info["venue_id"])
        weather = get_weather_forecast(venue["lat"], venue["lon"], matchup_info["date"])
        matchup = {"date": matchup_info["date"], "venue": venue}
        is_home_a = matchup_info["home_team_id"] == team_a["id"]
        is_home_b = matchup_info["home_team_id"] == team_b["id"]
        if ml_bundle or ml_bundle_1to3 or ml_bundle_1to5:
            from ml_data import live_weather_features
            weather_features = live_weather_features(venue, weather)
        ump_id, ump_name = get_home_plate_umpire(matchup_info["gamePk"])
        if ump_name:
            umpire_info = {"name": ump_name, "tendency": get_umpire_tendency(ump_name)}

    if pid_a is None:
        pid_a, pname_a = find_last_rotation_starter(team_a["id"])
    if pid_b is None:
        pid_b, pname_b = find_last_rotation_starter(team_b["id"])

    if args.pitcher_a:
        pid_a, pname_a = resolve_pitcher_on_team(team_a["id"], args.pitcher_a)
        print(f"(Abridor de {team_a['name']} forzado manualmente a {pname_a}, "
              f"la API de MLB tenia otro nombre)\n")
    if args.pitcher_b:
        pid_b, pname_b = resolve_pitcher_on_team(team_b["id"], args.pitcher_b)
        print(f"(Abridor de {team_b['name']} forzado manualmente a {pname_b}, "
              f"la API de MLB tenia otro nombre)\n")

    avg_a_adj, injured_a, avg_a_raw = team_avg_adjusted_for_injuries(team_a["id"], args.season)
    avg_b_adj, injured_b, avg_b_raw = team_avg_adjusted_for_injuries(team_b["id"], args.season)

    # Enriquecimiento SOLO EN VIVO (no se reconstruye para el entrenamiento
    # historico): si MLB ya publico el lineup titular de hoy (normalmente
    # 1-3h antes del primer pitch), usa el AVG real de esos 9 bateadores en
    # vez del AVG ajustado solo por lesionados - capta tambien a un regular
    # que descansa sin estar lesionado.
    lineup_avg_a = lineup_avg_b = None
    lineup_detail_a = lineup_detail_b = []
    lineup_confirmed_a = lineup_confirmed_b = False
    if matchup_info:
        lineup_data = get_confirmed_lineup(matchup_info["gamePk"])
        players_a = lineup_data["home"] if is_home_a else lineup_data["away"]
        players_b = lineup_data["home"] if is_home_b else lineup_data["away"]
        if players_a:
            lineup_avg_a, lineup_detail_a = lineup_batting_avg(players_a, args.season)
            lineup_confirmed_a = True
        if players_b:
            lineup_avg_b, lineup_detail_b = lineup_batting_avg(players_b, args.season)
            lineup_confirmed_b = True

    final_avg_a = lineup_avg_a if lineup_avg_a is not None else avg_a_adj
    final_avg_b = lineup_avg_b if lineup_avg_b is not None else avg_b_adj

    game_date = matchup_info["date"] if matchup_info else date.today().isoformat()
    park_factor = None
    if not args.no_park_factor:
        park_team_id = matchup_info["home_team_id"] if matchup_info else team_b["id"]
        park_factor = compute_park_factor(park_team_id, args.season)

    report_a = build_team_report(team_a["id"], team_a["name"], pid_a, pname_a, args.season,
                                  opponent_avg=final_avg_b, avg_similarity_threshold=args.avg_similarity_threshold,
                                  fip_constant=args.fip_constant,
                                  opponent_team_id=team_b["id"], is_home=is_home_a,
                                  ml_bundle=ml_bundle, ml_bundle_1to3=ml_bundle_1to3, ml_bundle_1to5=ml_bundle_1to5,
                                  weather_features=weather_features, opponent_pitcher_id=pid_b,
                                  game_date=game_date)
    report_b = build_team_report(team_b["id"], team_b["name"], pid_b, pname_b, args.season,
                                  opponent_avg=final_avg_a, avg_similarity_threshold=args.avg_similarity_threshold,
                                  fip_constant=args.fip_constant,
                                  opponent_team_id=team_a["id"], is_home=is_home_b,
                                  ml_bundle=ml_bundle, ml_bundle_1to3=ml_bundle_1to3, ml_bundle_1to5=ml_bundle_1to5,
                                  weather_features=weather_features, opponent_pitcher_id=pid_a,
                                  game_date=game_date)

    injuries_info = {
        team_a["name"]: {"raw_avg": avg_a_raw, "adj_avg": avg_a_adj, "injured": injured_a},
        team_b["name"]: {"raw_avg": avg_b_raw, "adj_avg": avg_b_adj, "injured": injured_b},
    }
    lineup_info = {
        team_a["name"]: {"confirmed": lineup_confirmed_a, "avg": lineup_avg_a, "detail": lineup_detail_a},
        team_b["name"]: {"confirmed": lineup_confirmed_b, "avg": lineup_avg_b, "detail": lineup_detail_b},
    }
    print_report(matchup, report_a, report_b, weather, whip_threshold=args.whip_threshold,
                 injuries_info=injuries_info, park_factor=park_factor, lineup_info=lineup_info,
                 umpire_info=umpire_info)

    if matchup_info and not args.no_ml and not args.no_log:
        from ml_track import build_log_row, log_prediction
        for team, pid, is_h, rep in ((team_a, pid_a, is_home_a, report_a), (team_b, pid_b, is_home_b, report_b)):
            row = build_log_row(team["name"], pid, rep["pitcher"]["name"] if rep["pitcher"] else None,
                                 is_h, matchup_info["gamePk"], matchup_info["date"],
                                 rep["pitcher"])
            if row:
                log_prediction(row, args.log_path)

    if args.sheet_id:
        rows = report_to_rows(matchup, report_a, report_b, weather, whip_threshold=args.whip_threshold,
                               injuries_info=injuries_info, park_factor=park_factor)
        write_to_google_sheet(rows, args.sheet_id, args.sheet_tab, args.credentials, args.sheet_start_cell)

    if args.use_gemini:
        extra = maybe_ask_gemini(
            f"Dame contexto adicional (lesiones recientes, noticias de bullpen) para el "
            f"proximo juego entre {report_a['team_name']} y {report_b['team_name']} en MLB."
        )
        if extra:
            print("--- Info adicional (Gemini) ---")
            print(extra)


if __name__ == "__main__":
    main()
