"""
Interfaz web (Streamlit) para el sistema de apuestas MLB - reemplaza a los
.bat: se corre desde cualquier navegador (incluido el del celular) una vez
publicada en Streamlit Community Cloud.

No reimplementa el analisis: llama exactamente a las mismas funciones que
ya usan mlb_first_inning_report.py / ml_predict.py / ml_track.py /
umpire_data.py / ml_train.py. Este archivo solo se encarga de la parte
visual (formularios, tarjetas, tablas) y de que el reentrenamiento se
guarde de forma permanente en GitHub (ver git_sync.py).

Correr localmente para probar:
    streamlit run streamlit_app.py
"""

import json
import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from datetime import date, datetime

import pandas as pd
import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import git_sync
import mlb_first_inning_report as m
import ml_track
import umpire_data
from ml_data import live_weather_features
from ml_predict import compute_current_features, predict_matchup, predict_with_bundle

MATCHUP_MODELS = [
    ("model_1st_total.joblib", "Primer inning - Total", "over"),
    ("model_1to3_favorite.joblib", "Innings 1 a 3 - Hándicap", "favorite"),
    ("model_1to3_total.joblib", "Innings 1 a 3 - Total", "over"),
    ("model_1to5_favorite.joblib", "Innings 1 a 5 - Hándicap", "favorite"),
    ("model_1to5_total.joblib", "Innings 1 a 5 - Total", "over"),
]

st.set_page_config(page_title="MLB Apuestas", page_icon="⚾", layout="wide")


# ---------------------------------------------------------------------------
# Utilidades compartidas
# ---------------------------------------------------------------------------

class _LiveLog:
    """Redirige stdout/stderr a un st.empty() para ver el progreso de una
    funcion larga en vivo, en vez de esperar a que termine sin feedback."""

    def __init__(self, placeholder, max_lines=200):
        self.placeholder = placeholder
        self.lines = []
        self.max_lines = max_lines

    def write(self, s):
        for part in s.splitlines():
            if part.strip():
                self.lines.append(part)
        self.placeholder.code("\n".join(self.lines[-self.max_lines:]) or " ")

    def flush(self):
        pass


@contextmanager
def live_log(placeholder):
    log = _LiveLog(placeholder)
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = log, log
    try:
        yield log
    finally:
        sys.stdout, sys.stderr = old_out, old_err


def check_password():
    """Si se configuro APP_PASSWORD en Secrets, pide un PIN antes de mostrar
    la app. Si no se configuro, no bloquea nada (util para probar local)."""
    if "APP_PASSWORD" not in st.secrets:
        return True
    if st.session_state.get("_authed"):
        return True
    st.title("⚾ MLB Apuestas")
    with st.form("login"):
        pw = st.text_input("PIN de acceso", type="password")
        submitted = st.form_submit_button("Entrar")
    if submitted:
        if pw == st.secrets["APP_PASSWORD"]:
            st.session_state["_authed"] = True
            st.rerun()
        else:
            st.error("PIN incorrecto.")
    return False


@st.cache_data(ttl=6 * 3600)
def get_teams_cached():
    return m.get_teams()


@st.cache_resource
def _load_bundle_cached(path, mtime):
    import joblib
    return joblib.load(path)


def load_bundle(path):
    """Carga un modelo .joblib, cacheado por ruta de archivo + fecha de
    modificacion - si se reentrena y el archivo cambia, se vuelve a cargar
    solo (sin necesidad de reiniciar la app ni limpiar cache a mano)."""
    if not os.path.exists(path):
        return None
    return _load_bundle_cached(path, os.path.getmtime(path))


def _materialize_google_credentials():
    raw = st.secrets["GOOGLE_CREDENTIALS_JSON"]
    content = raw if isinstance(raw, str) else json.dumps(dict(raw))
    path = os.path.join(tempfile.gettempdir(), "mlb_app_google_credentials.json")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def _stream_subprocess(cmd, log_placeholder):
    process = subprocess.Popen(
        cmd, cwd=APP_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, encoding="utf-8", errors="replace",
    )
    lines = []
    for line in process.stdout:
        lines.append(line.rstrip("\n"))
        log_placeholder.code("\n".join(lines[-300:]))
    process.wait()
    return process.returncode


# ---------------------------------------------------------------------------
# Reporte del juego (reemplaza correr_reporte.bat)
# ---------------------------------------------------------------------------

def _merge_batter_detail(lineup_detail, players, season, opposing_pitcher_hand):
    """Le agrega a cada dict de lineup_detail (ya trae id/name/avg/ab de
    lineup_batting_avg) los splits vs. mano del abridor rival y la forma
    reciente (ultimos 5 juegos), mutando la lista in-place."""
    extra_by_id = m.lineup_batter_detail(players, season, opposing_pitcher_hand)
    for player in lineup_detail:
        extra = extra_by_id.get(player["id"], {})
        player.update(extra)


def compute_picks(report_a, report_b, is_home_a):
    """Junta las predicciones YA calculadas de 0-carreras-1er-inning por
    pitcher con las de los 5 modelos conjuntos (favorito y total de
    carreras 1-3/1-5, total 1er inning), en una sola lista de picks
    candidatos ordenada de mas a menos confianza. El money line (quien
    gana el juego completo) se calcula aparte con la formula de ofensiva
    proyectada - no es un modelo entrenado, por eso no se mezcla en el
    mismo ranking de confianza calibrada."""
    picks = []

    for rep in (report_a, report_b):
        p = rep.get("pitcher")
        mlp = p.get("ml_prediction") if p else None
        if mlp and mlp.get("prob") is not None:
            prob = mlp["prob"]
            picks.append({
                "label": f"{p['name']} ({rep['team_name']}) - 0 carreras 1er inning",
                "pick": "Si, 0 carreras" if prob >= 0.5 else "No, permite carreras",
                "confidence": prob if prob >= 0.5 else 1 - prob,
            })

    features_a = (report_a.get("pitcher") or {}).get("ml_features")
    features_b = (report_b.get("pitcher") or {}).get("ml_features")
    if features_a and features_b and is_home_a is not None:
        features_home, features_away = (features_a, features_b) if is_home_a else (features_b, features_a)
        team_home_name = report_a["team_name"] if is_home_a else report_b["team_name"]
        team_away_name = report_b["team_name"] if is_home_a else report_a["team_name"]

        for filename, label, kind in MATCHUP_MODELS:
            bundle = load_bundle(os.path.join(APP_DIR, filename))
            if not bundle:
                continue
            try:
                prob = predict_matchup(features_home, features_away, bundle)
            except Exception:
                continue
            if kind == "favorite":
                over = prob >= 0.5
                pick_text = f"{team_home_name if over else team_away_name} anota mas"
            else:
                threshold = bundle.get("threshold")
                over = prob >= 0.5
                linea = f" {threshold}" if threshold is not None else ""
                pick_text = f"{'Over' if over else 'Under'}{linea}"
            picks.append({"label": label, "pick": pick_text, "confidence": prob if over else 1 - prob})

    picks.sort(key=lambda x: x["confidence"], reverse=True)

    moneyline_lean = None
    op_a, op_b = report_a.get("ofensiva_proyectada"), report_b.get("ofensiva_proyectada")
    if op_a and op_b:
        runs_a, runs_b = op_a["carreras_proyectadas"], op_b["carreras_proyectadas"]
        moneyline_lean = {
            "favorite": report_a["team_name"] if runs_a > runs_b else report_b["team_name"],
            "team_a": report_a["team_name"], "runs_a": runs_a,
            "team_b": report_b["team_name"], "runs_b": runs_b,
        }

    return picks, moneyline_lean


def build_full_report(equipo_a, equipo_b, season, pitcher_a_override, pitcher_b_override,
                       whip_threshold, fip_constant, avg_similarity_threshold,
                       no_park_factor, write_sheet, sheet_id, log_predictions):
    ml_bundle = load_bundle(os.path.join(APP_DIR, "model.joblib"))
    ml_bundle_1to3 = load_bundle(os.path.join(APP_DIR, "model_1to3.joblib"))
    ml_bundle_1to5 = load_bundle(os.path.join(APP_DIR, "model_1to5.joblib"))

    teams = get_teams_cached()
    team_a = m.resolve_team(equipo_a, teams)
    team_b = m.resolve_team(equipo_b, teams)
    if team_a["id"] == team_b["id"]:
        raise ValueError(f"'{equipo_a}' y '{equipo_b}' son el mismo equipo, elige dos equipos distintos.")

    matchup_info = m.find_matchup_game(team_a["id"], team_b["id"]) or \
        m.find_matchup_game(team_b["id"], team_a["id"])

    pid_a = pname_a = pid_b = pname_b = None
    weather = None
    matchup = None
    is_home_a = is_home_b = None
    weather_features = None
    umpire_info = None
    pitcher_override_msgs = []

    if matchup_info:
        pid_a, pname_a = matchup_info["pitchers"].get(team_a["id"], (None, None))
        pid_b, pname_b = matchup_info["pitchers"].get(team_b["id"], (None, None))
        venue = m.get_venue_details(matchup_info["venue_id"])
        weather = m.get_weather_forecast(venue["lat"], venue["lon"], matchup_info["date"])
        matchup = {"date": matchup_info["date"], "venue": venue}
        is_home_a = matchup_info["home_team_id"] == team_a["id"]
        is_home_b = matchup_info["home_team_id"] == team_b["id"]
        if ml_bundle or ml_bundle_1to3 or ml_bundle_1to5:
            weather_features = live_weather_features(venue, weather)
        ump_id, ump_name = m.get_home_plate_umpire(matchup_info["gamePk"])
        if ump_name:
            umpire_info = {"name": ump_name, "tendency": m.get_umpire_tendency(ump_name)}

    if pid_a is None:
        pid_a, pname_a = m.find_last_rotation_starter(team_a["id"])
    if pid_b is None:
        pid_b, pname_b = m.find_last_rotation_starter(team_b["id"])

    if pitcher_a_override:
        pid_a, pname_a = m.resolve_pitcher_on_team(team_a["id"], pitcher_a_override)
        pitcher_override_msgs.append(f"Abridor de {team_a['name']} forzado manualmente a {pname_a}.")
    if pitcher_b_override:
        pid_b, pname_b = m.resolve_pitcher_on_team(team_b["id"], pitcher_b_override)
        pitcher_override_msgs.append(f"Abridor de {team_b['name']} forzado manualmente a {pname_b}.")

    avg_a_adj, injured_a, avg_a_raw = m.team_avg_adjusted_for_injuries(team_a["id"], season)
    avg_b_adj, injured_b, avg_b_raw = m.team_avg_adjusted_for_injuries(team_b["id"], season)

    lineup_avg_a = lineup_avg_b = None
    lineup_detail_a = lineup_detail_b = []
    lineup_confirmed_a = lineup_confirmed_b = False
    if matchup_info:
        lineup_data = m.get_confirmed_lineup(matchup_info["gamePk"])
        players_a = lineup_data["home"] if is_home_a else lineup_data["away"]
        players_b = lineup_data["home"] if is_home_b else lineup_data["away"]
        if players_a:
            lineup_avg_a, lineup_detail_a = m.lineup_batting_avg(players_a, season)
            lineup_confirmed_a = True
            _merge_batter_detail(lineup_detail_a, players_a, season, m.get_pitcher_hand(pid_b) if pid_b else None)
        if players_b:
            lineup_avg_b, lineup_detail_b = m.lineup_batting_avg(players_b, season)
            lineup_confirmed_b = True
            _merge_batter_detail(lineup_detail_b, players_b, season, m.get_pitcher_hand(pid_a) if pid_a else None)

    final_avg_a = lineup_avg_a if lineup_avg_a is not None else avg_a_adj
    final_avg_b = lineup_avg_b if lineup_avg_b is not None else avg_b_adj

    game_date = matchup_info["date"] if matchup_info else date.today().isoformat()
    park_factor = None
    if not no_park_factor:
        park_team_id = matchup_info["home_team_id"] if matchup_info else team_b["id"]
        park_factor = m.compute_park_factor(park_team_id, season)

    report_a = m.build_team_report(
        team_a["id"], team_a["name"], pid_a, pname_a, season,
        opponent_avg=final_avg_b, avg_similarity_threshold=avg_similarity_threshold, fip_constant=fip_constant,
        opponent_team_id=team_b["id"], is_home=is_home_a,
        ml_bundle=ml_bundle, ml_bundle_1to3=ml_bundle_1to3, ml_bundle_1to5=ml_bundle_1to5,
        weather_features=weather_features, opponent_pitcher_id=pid_b, game_date=game_date,
    )
    report_b = m.build_team_report(
        team_b["id"], team_b["name"], pid_b, pname_b, season,
        opponent_avg=final_avg_a, avg_similarity_threshold=avg_similarity_threshold, fip_constant=fip_constant,
        opponent_team_id=team_a["id"], is_home=is_home_b,
        ml_bundle=ml_bundle, ml_bundle_1to3=ml_bundle_1to3, ml_bundle_1to5=ml_bundle_1to5,
        weather_features=weather_features, opponent_pitcher_id=pid_a, game_date=game_date,
    )

    injuries_info = {
        team_a["name"]: {"raw_avg": avg_a_raw, "adj_avg": avg_a_adj, "injured": injured_a},
        team_b["name"]: {"raw_avg": avg_b_raw, "adj_avg": avg_b_adj, "injured": injured_b},
    }
    lineup_info = {
        team_a["name"]: {"confirmed": lineup_confirmed_a, "avg": lineup_avg_a, "detail": lineup_detail_a},
        team_b["name"]: {"confirmed": lineup_confirmed_b, "avg": lineup_avg_b, "detail": lineup_detail_b},
    }

    sheet_status = None
    if write_sheet and sheet_id:
        try:
            rows = m.report_to_rows(matchup, report_a, report_b, weather, whip_threshold=whip_threshold,
                                     injuries_info=injuries_info, park_factor=park_factor)
            creds_path = _materialize_google_credentials()
            m.write_to_google_sheet(rows, sheet_id, "Reporte MLB", creds_path)
            sheet_status = ("ok", "Reporte escrito en Google Sheets.")
        except Exception as e:
            sheet_status = ("error", f"No se pudo escribir en Google Sheets: {e}")

    log_status = None
    if matchup_info and log_predictions and (ml_bundle or ml_bundle_1to3 or ml_bundle_1to5):
        from ml_track import build_log_row, log_prediction
        log_path = os.path.join(APP_DIR, "predictions_log.csv")
        for team, pid, is_h, rep in ((team_a, pid_a, is_home_a, report_a), (team_b, pid_b, is_home_b, report_b)):
            row = build_log_row(team["name"], pid, rep["pitcher"]["name"] if rep["pitcher"] else None,
                                 is_h, matchup_info["gamePk"], matchup_info["date"], rep["pitcher"])
            if row:
                log_prediction(row, log_path)
        log_status = git_sync.commit_and_push(
            ["predictions_log.csv"], f"Log de predicciones {game_date}", st.secrets, APP_DIR,
        )

    picks, moneyline_lean = compute_picks(report_a, report_b, is_home_a)

    return {
        "matchup": matchup, "report_a": report_a, "report_b": report_b, "weather": weather,
        "whip_threshold": whip_threshold, "injuries_info": injuries_info, "park_factor": park_factor,
        "lineup_info": lineup_info, "umpire_info": umpire_info,
        "pitcher_override_msgs": pitcher_override_msgs, "sheet_status": sheet_status, "log_status": log_status,
        "picks": picks, "moneyline_lean": moneyline_lean,
    }


def render_top_info(matchup, weather, park_factor, umpire_info, team_b_name):
    if not matchup:
        st.info("No hay un juego programado entre estos equipos en los proximos dias - "
                 "se muestran datos generales de cada equipo/abridor.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Fecha", matchup["date"])
    c1.caption(f"📍 {matchup['venue']['name']} — {matchup['venue']['city']}, {matchup['venue']['state']}")
    c1.caption(f"Techo: {m.ROOF_TYPES.get(team_b_name, 'N/D')}")

    if weather:
        wind_class = m.classify_wind(weather["wind_dir_from"], matchup["venue"]["azimuth"])
        c2.metric("Clima", f"{weather['temp_c']}°C", help=f"~{weather['time']} · prob. lluvia {weather['precip_prob']}%")
        c2.caption(f"💨 {weather['wind_mph']} mph — {wind_class}")
        c2.caption(f"Techo (heuristica): {m.roof_status_heuristic(weather)}")
    else:
        c2.metric("Clima", "N/D")

    if park_factor is not None:
        tendencia = "favorece anotacion" if park_factor > 1.05 else "reduce anotacion" if park_factor < 0.95 else "neutral"
        c3.metric("Factor de estadio", f"{park_factor:.3f}", help=tendencia)
    if umpire_info:
        t = umpire_info.get("tendency")
        if t:
            c3.caption(f"⚖️ {umpire_info['name']} — {t['runs_vs_league']:.2f}x carreras/liga "
                       f"({m.confidence_label(t['n_games'])}, proxy)")
        else:
            c3.caption(f"⚖️ {umpire_info['name']} (sin tendencia calculada todavia)")
    else:
        c3.caption("⚖️ Umpire: N/D todavia")


def render_pitcher_card(rep, rival_name, whip_threshold):
    p = rep["pitcher"]
    st.markdown(f"#### {rep['team_name']}")
    st.caption(p["name"] if p else "Abridor N/D")
    if not p:
        st.warning("Abridor no disponible todavia.")
        return

    ss = p["season_stats"]
    if ss:
        c1, c2, c3 = st.columns(3)
        c1.metric("ERA", ss["era"] or "N/D")
        whip_class = m.classify_whip(ss["whip"], threshold=whip_threshold)
        c2.metric("WHIP", ss["whip"] or "N/D")
        c2.caption(("🟢 " if whip_class == "Aprobada" else "🔴 " if whip_class == "Riesgosa" else "") + whip_class)
        c3.metric("FIP (aprox.)", f"{ss['fip']:.2f}" if ss["fip"] is not None else "N/D")
        if ss.get("ip_per_start"):
            st.caption(f"{ss['ip_per_start']:.2f} entradas/apertura ({ss['ip']} IP en {ss['gs']} aperturas)")

    dr = p.get("days_rest")
    if dr is not None:
        st.caption(f"Descanso: {dr} dia(s)" + (" ⚠️ descanso corto" if dr < 4 else ""))

    total = p["total_checked"]
    scoreless = p["scoreless"]
    st.metric(
        f"0 carreras 1er inning (ultimas {total} aperturas)" if total else "0 carreras 1er inning",
        f"{scoreless/total*100:.1f}%" if total else "N/D",
        help=f"{scoreless}/{total} — {m.confidence_label(total)}" if total else None,
    )

    feats = p.get("ml_features") or {}
    n_prior = feats.get("n_prior_starts")
    ml_conf = m.confidence_label(n_prior) if n_prior is not None else None

    mlp = p.get("ml_prediction")
    if mlp and mlp.get("prob") is not None:
        baseline = feats.get("scoreless_rate_trailing")
        delta = f"{(mlp['prob']-baseline)*100:+.1f} pp vs. baseline" if baseline is not None else None
        st.metric(f"Prediccion ML ({mlp['model_name']})", f"{mlp['prob']*100:.1f}%", delta=delta, help=ml_conf)
    elif mlp:
        st.caption(f"Prediccion ML: N/D ({mlp.get('error')})")

    mlp3 = p.get("ml_prediction_1to3")
    if mlp3 and mlp3.get("runs") is not None:
        baseline3 = feats.get("runs_1to3_trailing_avg")
        delta3 = f"{mlp3['runs']-baseline3:+.2f} vs. baseline" if baseline3 is not None else None
        st.metric("Carreras permitidas — entradas 1-3", f"{mlp3['runs']:.2f}", delta=delta3, help=ml_conf)

    mlp5 = p.get("ml_prediction_1to5")
    if mlp5 and mlp5.get("runs") is not None:
        baseline5 = feats.get("runs_1to5_trailing_avg")
        delta5 = f"{mlp5['runs']-baseline5:+.2f} vs. baseline" if baseline5 is not None else None
        st.metric("Carreras permitidas — entradas 1-5", f"{mlp5['runs']:.2f}", delta=delta5, help=ml_conf)

    with st.expander(f"Splits detallados vs. rivales con AVG similar a {rival_name}"):
        sas = p["similar_avg_splits"]
        rows = []
        for key, label in [("1to3", "1-3 entradas"), ("1to5", "1-5 entradas"), ("full", "Juego completo")]:
            s = sas[key]
            rows.append({
                "Rango": label,
                "Local: carreras": round(s["home_avg_runs"], 3) if s["home_avg_runs"] is not None else None,
                "Local: n": s["home_n"],
                "Visitante: carreras": round(s["away_avg_runs"], 3) if s["away_avg_runs"] is not None else None,
                "Visitante: n": s["away_n"],
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def render_ofensiva_proyectada(rep, rival_pitcher_name):
    st.markdown(f"**{rep['team_name']}**")
    op = rep.get("ofensiva_proyectada")
    if not op:
        st.caption("N/D (falta el abridor rival o datos suficientes).")
        return
    st.metric(
        "Carreras proyectadas", f"{op['carreras_proyectadas']:.2f}",
        help=f"Base: {op['carreras_recientes_base']:.2f} carreras recientes x indice "
             f"{op['indice_calidad_rival']:.3f} (rival {'mas debil' if op['indice_calidad_rival'] > 1 else 'mas fuerte'} que la liga)",
    )
    st.caption(f"vs. {rival_pitcher_name} + bullpen ({op['bullpen_n_relievers']} relevistas) — "
               f"ERA {op['mezcla_era']:.2f} · WHIP {op['mezcla_whip']:.2f} · FIP {op['mezcla_fip']:.2f}")

    hs = rep.get("hitting_stats")
    if hs and hs.get("ops") is not None:
        st.caption(f"OPS de temporada: {hs['ops']:.3f} (OBP {hs['obp']:.3f} + SLG {hs['slg']:.3f})")

    vhs = rep.get("vs_hand_split")
    if vhs:
        mano = "zurdos" if vhs["hand"] == "L" else "derechos"
        st.caption(f"Bateando vs. {mano} esta temporada: AVG {vhs['avg']:.3f}, OPS {vhs['ops']:.3f}")

    h2h = rep.get("head_to_head")
    if h2h and h2h["n_games"] > 0:
        st.caption(f"Historial real vs {rival_pitcher_name}: {h2h['n_games']} juego(s) — "
                   f"AVG {h2h['avg']:.3f}, {h2h['avg_runs']:.2f} carreras/juego")
    elif h2h is not None:
        st.caption(f"Sin enfrentamientos previos vs {rival_pitcher_name} en temporada actual/anterior.")

    wl = rep.get("opp_bullpen_workload")
    if wl and wl["games_included"] > 0:
        fatiga = "ALTA ⚠️" if wl["bullpen_ip"] >= 9 else "normal"
        st.caption(f"Carga bullpen rival (ultimos {wl['days']} dias): {wl['bullpen_ip']:.1f} IP — fatiga {fatiga}")


def render_offense_table(a, b):
    def r(v, nd=3):
        return round(v, nd) if v is not None else None

    rows = []
    for label, avg_key, runs_key in [
        ("1er inning", "avg_1st", "avg_runs_1st"),
        ("Entradas 1-3", "avg_1to3", "avg_runs_1to3"),
        ("Juego completo", "avg_full", "avg_runs_full"),
    ]:
        rows.append({
            "Rango": label,
            f"{a['team_name']} AVG": r(a[avg_key]), f"{a['team_name']} carreras": r(a[runs_key]),
            f"{b['team_name']} AVG": r(b[avg_key]), f"{b['team_name']} carreras": r(b[runs_key]),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def render_picks_section(picks, moneyline_lean):
    st.markdown("### 🏆 Picks recomendados")
    if not picks:
        st.caption("No hay picks disponibles todavia - hace falta un juego programado entre estos equipos "
                   "y los modelos conjuntos entrenados (pestana 'Reentrenar modelos').")
    else:
        rows = []
        for p in picks[:5]:
            badge = "🟢" if p["confidence"] >= 0.65 else "🟡" if p["confidence"] >= 0.55 else "🔴"
            rows.append({"": badge, "Pick": p["label"], "Recomendacion": p["pick"],
                         "Confianza": f"{p['confidence']*100:.1f}%"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("Ordenados de mas a menos confianza segun modelos entrenados con el historial de la "
                   "temporada. No es garantia de nada - revisa el tamano de muestra de cada dato en las "
                   "secciones de abajo antes de decidir.")

    if moneyline_lean:
        st.caption(
            f"💰 Money line (estimado por formula de ofensiva proyectada, **no** es un modelo entrenado "
            f"como los picks de arriba): favorito **{moneyline_lean['favorite']}** — "
            f"{moneyline_lean['team_a']} {moneyline_lean['runs_a']:.2f} carreras proyectadas vs. "
            f"{moneyline_lean['team_b']} {moneyline_lean['runs_b']:.2f}."
        )


def render_report(matchup, report_a, report_b, weather, whip_threshold, injuries_info, park_factor,
                   lineup_info, umpire_info, pitcher_override_msgs=None, sheet_status=None, log_status=None,
                   picks=None, moneyline_lean=None):
    a, b = report_a, report_b
    st.subheader(f"{a['team_name']} @ {b['team_name']}" if matchup else f"{a['team_name']} vs {b['team_name']}")

    for msg in (pitcher_override_msgs or []):
        st.caption(f"ℹ️ {msg}")
    if sheet_status:
        (st.success if sheet_status[0] == "ok" else st.error)(sheet_status[1])
    if log_status:
        ok, msg = log_status
        st.caption(("✅ " if ok else "⚠️ ") + msg)

    render_top_info(matchup, weather, park_factor, umpire_info, b["team_name"])

    st.divider()
    render_picks_section(picks, moneyline_lean)

    if injuries_info:
        with st.expander("🩹 Lesionados en la ofensiva"):
            for team_name, info in injuries_info.items():
                raw, adj = info["raw_avg"], info["adj_avg"]
                if raw is not None and adj is not None:
                    st.markdown(f"**{team_name}**: AVG {raw:.3f} → ajustado sin lesionados {adj:.3f}")
                else:
                    st.markdown(f"**{team_name}**: AVG N/D")
                if info["injured"]:
                    st.dataframe(pd.DataFrame(info["injured"])[["name", "position", "status", "avg", "ab"]],
                                 hide_index=True, use_container_width=True)
                else:
                    st.caption("Sin lesionados de posicion actualmente.")

    if lineup_info:
        with st.expander("🧢 Lineup titular confirmado"):
            for team_name, info in lineup_info.items():
                if not info["confirmed"]:
                    st.caption(f"{team_name}: lineup todavia no confirmado (se uso AVG ajustado por lesionados).")
                    continue
                avg_txt = f"{info['avg']:.3f}" if info["avg"] is not None else "N/D"
                st.markdown(f"**{team_name}** — AVG combinado de titulares: {avg_txt}")
                df = pd.DataFrame(info["detail"]).drop(columns=["id"], errors="ignore")
                df = df.rename(columns={
                    "name": "Nombre", "avg": "AVG temporada", "ab": "AB",
                    "vs_hand_avg": "AVG vs. rival hoy", "vs_hand_ops": "OPS vs. rival hoy",
                    "last5_avg": "AVG ult. 5", "last5_hits": "Hits ult. 5", "last5_ab": "AB ult. 5",
                })
                st.dataframe(df.round(3), hide_index=True, use_container_width=True)
                if "AVG vs. rival hoy" in df.columns:
                    st.caption("'vs. rival hoy' = contra la mano (zurdo/derecho) del abridor rival de este juego. "
                               "'ult. 5' = ultimos 5 juegos jugados, sin importar rival (forma reciente).")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        render_pitcher_card(a, b["team_name"], whip_threshold)
    with col2:
        render_pitcher_card(b, a["team_name"], whip_threshold)

    st.divider()
    st.markdown("### 🎯 Ofensiva proyectada (considera abridor + bullpen rival)")
    col1, col2 = st.columns(2)
    with col1:
        render_ofensiva_proyectada(a, b["pitcher"]["name"] if b["pitcher"] else "N/D")
    with col2:
        render_ofensiva_proyectada(b, a["pitcher"]["name"] if a["pitcher"] else "N/D")

    st.divider()
    st.markdown("### 📊 Ofensiva reciente (ultimos 5 juegos)")
    render_offense_table(a, b)

    with st.expander("Detalle por juego"):
        for label, key in [("1er inning", "per_game"), ("Entradas 1-3", "per_game_1to3"),
                            ("Juego completo", "per_game_full")]:
            st.markdown(f"**{label}**")
            frames = []
            for rep in (a, b):
                if rep[key]:
                    df = pd.DataFrame(rep[key], columns=["Fecha", "Carreras", "Hits", "AB"])
                    df.insert(0, "Equipo", rep["team_name"])
                    frames.append(df)
            if frames:
                st.dataframe(pd.concat(frames, ignore_index=True), hide_index=True, use_container_width=True)


def render_reporte_tab():
    st.header("📋 Reporte del juego")
    st.caption("Analisis completo pre-apuesta: abridores, clima, lesionados, lineup, umpire, "
               "ofensiva proyectada y predicciones ML. Tarda 1-2 minutos.")

    teams = get_teams_cached()
    team_names = sorted(t["name"] for t in teams)

    with st.form("form_reporte"):
        c1, c2 = st.columns(2)
        equipo_a = c1.selectbox("Equipo A", team_names, index=0)
        equipo_b = c2.selectbox("Equipo B", team_names, index=min(1, len(team_names) - 1))

        with st.expander("Opciones avanzadas"):
            season = st.number_input("Temporada", value=date.today().year, step=1)
            c3, c4 = st.columns(2)
            pitcher_a_override = c3.text_input(
                "Forzar abridor Equipo A", help="Solo si el reporte muestra un abridor desactualizado.")
            pitcher_b_override = c4.text_input("Forzar abridor Equipo B")
            whip_threshold = st.number_input("Umbral WHIP para 'Aprobada'",
                                              value=m.DEFAULT_WHIP_THRESHOLD, step=0.05, format="%.2f")
            fip_constant = st.number_input("Constante FIP", value=m.DEFAULT_FIP_CONSTANT, step=0.05, format="%.2f")
            avg_similarity_threshold = st.number_input("Umbral de AVG similar", value=0.015, step=0.005, format="%.3f")
            no_park_factor = st.checkbox("Omitir factor de estadio (mas rapido)", value=False)
            log_predictions = st.checkbox("Guardar esta prediccion en el historial de seguimiento", value=True)

            sheets_ready = bool(st.secrets.get("GOOGLE_CREDENTIALS_JSON"))
            write_sheet = False
            sheet_id = st.secrets.get("MLB_SHEET_ID", "")
            if sheets_ready:
                write_sheet = st.checkbox("Tambien escribir en Google Sheets", value=False)
                sheet_id = st.text_input("ID del Google Spreadsheet", value=sheet_id)
            else:
                st.caption("Google Sheets no configurado (ver pestana Ajustes).")

        submitted = st.form_submit_button("Generar reporte", type="primary", use_container_width=True)

    if submitted:
        with st.spinner("Generando reporte (clima, lesionados, umpire, lineup, historial)... 1-2 min"):
            try:
                result = build_full_report(
                    equipo_a, equipo_b, int(season), pitcher_a_override, pitcher_b_override,
                    float(whip_threshold), float(fip_constant), float(avg_similarity_threshold),
                    no_park_factor, write_sheet, sheet_id, log_predictions,
                )
                st.session_state["report_result"] = result
                st.session_state["report_error"] = None
            except Exception as e:
                st.session_state["report_result"] = None
                st.session_state["report_error"] = f"{type(e).__name__}: {e}"

    if st.session_state.get("report_error"):
        st.error(st.session_state["report_error"])
    if st.session_state.get("report_result"):
        render_report(**st.session_state["report_result"])


# ---------------------------------------------------------------------------
# Prediccion rapida (reemplaza correr_prediccion_ml.bat)
# ---------------------------------------------------------------------------

def render_prediccion_rapida():
    st.header("🤖 Prediccion rapida")
    st.caption("Solo la prediccion ML del abridor probable de cada equipo, sin clima detallado, "
               "lesionados, lineup, umpire ni ofensiva proyectada (para eso usa 'Reporte del juego'). "
               "Mas rapida.")

    teams = get_teams_cached()
    team_names = sorted(t["name"] for t in teams)
    with st.form("form_prediccion_rapida"):
        c1, c2 = st.columns(2)
        equipo_a = c1.selectbox("Equipo A", team_names, index=0, key="qp_a")
        equipo_b = c2.selectbox("Equipo B", team_names, index=min(1, len(team_names) - 1), key="qp_b")
        season = st.number_input("Temporada", value=date.today().year, step=1, key="qp_season")
        submitted = st.form_submit_button("Predecir", type="primary", use_container_width=True)

    if not submitted:
        return

    bundle_1st = load_bundle(os.path.join(APP_DIR, "model.joblib"))
    bundle_1to3 = load_bundle(os.path.join(APP_DIR, "model_1to3.joblib"))
    bundle_1to5 = load_bundle(os.path.join(APP_DIR, "model_1to5.joblib"))
    if not bundle_1st:
        st.error("No encontre model.joblib - reentrena los modelos primero (pestana 'Reentrenar modelos').")
        return

    try:
        team_a = m.resolve_team(equipo_a, teams)
        team_b = m.resolve_team(equipo_b, teams)
        matchup_info = m.find_matchup_game(team_a["id"], team_b["id"]) or \
            m.find_matchup_game(team_b["id"], team_a["id"])
        if not matchup_info:
            st.warning("No encontre un juego programado entre estos dos equipos en los proximos dias.")
            return

        pid_a, pname_a = matchup_info["pitchers"].get(team_a["id"], (None, None))
        pid_b, pname_b = matchup_info["pitchers"].get(team_b["id"], (None, None))
        is_home_a = matchup_info["home_team_id"] == team_a["id"]
        is_home_b = matchup_info["home_team_id"] == team_b["id"]

        venue = m.get_venue_details(matchup_info["venue_id"])
        weather = m.get_weather_forecast(venue["lat"], venue["lon"], matchup_info["date"])
        weather_features = live_weather_features(venue, weather)

        with st.spinner("Calculando..."):
            avg_a_adj, _, _ = m.team_avg_adjusted_for_injuries(team_a["id"], int(season))
            avg_b_adj, _, _ = m.team_avg_adjusted_for_injuries(team_b["id"], int(season))
            opp_avg_by_team = {team_a["id"]: avg_b_adj, team_b["id"]: avg_a_adj}

            cols = st.columns(2)
            for col, pid, pname, team, opp, is_home in (
                (cols[0], pid_a, pname_a, team_a, team_b, is_home_a),
                (cols[1], pid_b, pname_b, team_b, team_a, is_home_b),
            ):
                with col:
                    st.markdown(f"#### {team['name']}")
                    st.caption(pname or "N/D")
                    if pid is None:
                        st.warning("Abridor no disponible todavia.")
                        continue
                    features, error = compute_current_features(
                        pid, opp["id"], is_home, int(season), weather_features=weather_features,
                        opponent_avg_override=opp_avg_by_team[team["id"]], team_id=team["id"],
                        game_date=matchup_info["date"],
                    )
                    if features is None:
                        st.warning(error)
                        continue
                    prob = predict_with_bundle(bundle_1st, features)
                    st.metric("0 carreras 1er inning", f"{prob*100:.1f}%",
                              delta=f"{(prob-features['scoreless_rate_trailing'])*100:+.1f} pp vs. baseline")
                    if bundle_1to3:
                        r3 = predict_with_bundle(bundle_1to3, features)
                        st.metric("Carreras permitidas 1-3", f"{r3:.2f}",
                                  delta=f"{r3-features['runs_1to3_trailing_avg']:+.2f} vs. baseline")
                    if bundle_1to5:
                        r5 = predict_with_bundle(bundle_1to5, features)
                        st.metric("Carreras permitidas 1-5", f"{r5:.2f}",
                                  delta=f"{r5-features['runs_1to5_trailing_avg']:+.2f} vs. baseline")
    except Exception as e:
        st.error(f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Evaluar predicciones (reemplaza evaluar_predicciones.bat)
# ---------------------------------------------------------------------------

def render_evaluar():
    st.header("📈 Evaluar predicciones ML vs. resultado real")
    st.caption("Compara cada prediccion guardada en el historial contra lo que realmente paso "
               "en el juego, una vez que termino.")

    log_path = os.path.join(APP_DIR, "predictions_log.csv")

    if st.button("Evaluar ahora", type="primary"):
        placeholder = st.empty()
        with live_log(placeholder):
            ml_track.evaluate_log(log_path)
        ok, msg = git_sync.commit_and_push(
            ["predictions_log.csv"], f"Evalua predicciones {date.today().isoformat()}", st.secrets, APP_DIR,
        )
        st.caption(("✅ " if ok else "⚠️ ") + msg)

    if not os.path.exists(log_path):
        st.info("Todavia no hay historial - genera algunos reportes primero (pestana 'Reporte del juego').")
        return

    df = pd.read_csv(log_path)
    done = df[df["evaluated"] == True]  # noqa: E712
    if done.empty:
        st.info(f"{len(df)} prediccion(es) en el historial, ninguna evaluada todavia "
                "(los juegos no han terminado, o falta presionar 'Evaluar ahora').")
        return

    c1, c2, c3 = st.columns(3)
    d1 = done.dropna(subset=["pred_prob_1st", "actual_scoreless_1st"])
    if len(d1):
        pred_class = (d1["pred_prob_1st"] >= 0.5).astype(int)
        acc = (pred_class == d1["actual_scoreless_1st"]).mean()
        c1.metric("Accuracy 0 carreras 1er inning", f"{acc:.1%}", help=f"{len(d1)} aperturas evaluadas")
    for col, label, pred_col, actual_col in [
        (c2, "MAE carreras 1-3", "pred_runs_1to3", "actual_runs_1to3"),
        (c3, "MAE carreras 1-5", "pred_runs_1to5", "actual_runs_1to5"),
    ]:
        d = done.dropna(subset=[pred_col, actual_col])
        if len(d):
            mae = (d[pred_col] - d[actual_col]).abs().mean()
            col.metric(label, f"{mae:.3f}", help=f"{len(d)} aperturas evaluadas")

    st.dataframe(
        done[["game_date", "team_name", "pitcher_name", "pred_prob_1st", "actual_scoreless_1st",
              "pred_runs_1to3", "actual_runs_1to3", "pred_runs_1to5", "actual_runs_1to5"]]
        .sort_values("game_date", ascending=False),
        hide_index=True, use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Reentrenar modelos (reemplaza reentrenar_modelos.bat)
# ---------------------------------------------------------------------------

def _run_retrain(skip_collect, skip_enrich):
    season = date.today().year
    data_changed = False

    with st.status("Paso 1/5 · Evaluando predicciones pasadas...", expanded=True) as status:
        log = st.empty()
        rc = _stream_subprocess([sys.executable, "-u", "ml_track.py", "--evaluate"], log)
        status.update(label="Paso 1/5 listo" if rc == 0 else "Paso 1/5 con advertencias",
                       state="complete" if rc == 0 else "error")

    if not skip_collect:
        with st.status("Paso 2/5 · Recolectando historial completo de la temporada (20-45 min)...",
                        expanded=True) as status:
            log = st.empty()
            rc = _stream_subprocess(
                [sys.executable, "-u", "ml_data.py", "--season", str(season), "--out", "training_data.csv"], log)
            if rc != 0:
                status.update(label="Paso 2/5 fallo", state="error")
                st.error("La recoleccion de datos fallo (revisa el log de arriba). Se detiene el reentrenamiento.")
                return
            status.update(label="Paso 2/5 listo", state="complete")
            data_changed = True
    else:
        st.info("Paso 2/5 omitido (se usa el training_data.csv que ya existe).")

    if not skip_enrich:
        with st.status("Paso 3/5 · Agregando clima real de cada juego...", expanded=True) as status:
            log = st.empty()
            rc = _stream_subprocess(
                [sys.executable, "-u", "ml_data.py", "--enrich-weather", "training_data.csv",
                 "--out", "training_data.csv"], log)
            status.update(label="Paso 3/5 listo" if rc == 0 else "Paso 3/5 con advertencias",
                           state="complete" if rc == 0 else "error")
            data_changed = True
    else:
        st.info("Paso 3/5 omitido (se usa el clima que ya tiene training_data.csv).")

    with st.status("Paso 4/5 · Reentrenando los 3 modelos por pitcher...", expanded=True) as status:
        log = st.empty()
        rc = _stream_subprocess([sys.executable, "-u", "ml_train.py", "--data", "training_data.csv"], log)
        if rc != 0:
            status.update(label="Paso 4/5 fallo", state="error")
            st.error("El entrenamiento fallo, revisa el log de arriba.")
            return
        status.update(label="Paso 4/5 listo", state="complete")

    with st.status("Paso 5/5 · Reentrenando los picks conjuntos (favorito y total 1-3/1-5, total 1er inning)...",
                    expanded=True) as status:
        log = st.empty()
        rc = _stream_subprocess([sys.executable, "-u", "ml_train_matchup.py", "--data", "training_data.csv"], log)
        status.update(label="Paso 5/5 listo" if rc == 0 else "Paso 5/5 con advertencias",
                       state="complete" if rc == 0 else "error")

    st.success("Reentrenamiento terminado. Los modelos nuevos ya se usan en esta app.")

    files_to_save = [
        "model.joblib", "model.joblib.metrics.json",
        "model_1to3.joblib", "model_1to3.joblib.metrics.json",
        "model_1to5.joblib", "model_1to5.joblib.metrics.json",
        "training_history.csv",
        "model_1st_total.joblib", "model_1st_total.joblib.metrics.json",
        "model_1to3_favorite.joblib", "model_1to3_favorite.joblib.metrics.json",
        "model_1to3_total.joblib", "model_1to3_total.joblib.metrics.json",
        "model_1to5_favorite.joblib", "model_1to5_favorite.joblib.metrics.json",
        "model_1to5_total.joblib", "model_1to5_total.joblib.metrics.json",
        "training_history_matchup.csv",
    ]
    if data_changed:
        files_to_save.append("training_data.csv")
    ok, msg = git_sync.commit_and_push(
        files_to_save, f"Reentrenamiento automatico {date.today().isoformat()}", st.secrets, APP_DIR,
    )
    (st.success if ok else st.warning)(msg)


def render_reentrenar():
    st.header("🔁 Reentrenar modelos")
    st.success(
        "El reentrenamiento ya corre solo cada 3 dias en GitHub Actions (servidor de GitHub, no "
        "depende de esta pestana ni de tu conexion) - no hace falta que uses el boton de abajo a "
        "menos que quieras forzar un reentrenamiento ahora mismo. Revisa el progreso o dispáralo a "
        "mano en github.com/BrandonPalmaJobs/apuestas-mlb → pestaña **Actions**."
    )
    with st.expander("Reentrenar manualmente desde aqui (no recomendado - ver de arriba)"):
        st.write(
            "Vuelve a entrenar los modelos con **todos** los juegos jugados hasta hoy. Tarda entre "
            "**20 y 45 minutos** porque recolecta el historial completo de la temporada desde la API "
            "de MLB, y necesita que esta pestaña se quede conectada TODO ese tiempo sin cortes - "
            "cualquier corte de red pierde el progreso. Por eso el reentrenamiento automatico de "
            "GitHub Actions (arriba) es la forma recomendada."
        )

        if not git_sync.is_configured(st.secrets):
            st.warning(
                "GITHUB_TOKEN / GITHUB_REPO no configurados en Secrets: los modelos reentrenados solo "
                "van a durar hasta que la app se reinicie o se duerma por inactividad."
            )

        skip_collect = st.checkbox(
            "Omitir recoleccion de datos (usar el training_data.csv que ya existe - marca esto si la "
            "recoleccion ya termino bien la ultima vez y solo fallo un paso de despues)", value=False,
        )
        skip_enrich = st.checkbox(
            "Omitir tambien el clima real (usar training_data.csv tal cual, sin volver a agregar "
            "clima) - normalmente déjalo SIN marcar", value=False,
        )
        tiempo_msg = "unos minutos" if skip_collect else "20-45 minutos"
        confirmado = st.checkbox(
            f"Entiendo que esto puede tardar {tiempo_msg} y no voy a cerrar la app mientras corre.")

        if st.button("Iniciar reentrenamiento", type="primary", disabled=not confirmado):
            _run_retrain(skip_collect, skip_enrich)

    hist_path = os.path.join(APP_DIR, "training_history.csv")
    if os.path.exists(hist_path):
        st.subheader("Historial de reentrenamientos")
        st.dataframe(pd.read_csv(hist_path).tail(10), hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------
# Tendencia de umpires (reemplaza correr_umpires.bat)
# ---------------------------------------------------------------------------

def render_umpires():
    st.header("⚖️ Tendencia de umpires (home plate)")
    st.caption(
        "Recorre los juegos ya jugados y calcula que tan 'generoso' o 'apretado' ha sido cada "
        "umpire de home plate (carreras/ponches/bases por bola por juego que dirigio, comparado "
        "contra el promedio de la liga). Es un PROXY, no la zona de strike real. Puede tardar "
        "varios minutos para la temporada completa."
    )

    umpire_csv = os.path.join(APP_DIR, "umpire_tendency.csv")
    c1, c2 = st.columns(2)
    season = c1.number_input("Temporada", value=date.today().year, step=1)
    min_games = c2.number_input("Minimo de juegos para incluir al umpire", value=3, step=1)

    if st.button("Generar / actualizar indice de umpires", type="primary"):
        placeholder = st.empty()
        with st.spinner("Recorriendo juegos de la temporada..."):
            with live_log(placeholder):
                games_df = umpire_data.collect_umpire_games(int(season))
                tendency_df = umpire_data.aggregate_umpire_tendency(games_df, min_games=int(min_games))
                tendency_df.to_csv(umpire_csv, index=False)
        st.success(f"Guardados {len(tendency_df)} umpires (de {len(games_df)} juegos revisados).")
        ok, msg = git_sync.commit_and_push(
            ["umpire_tendency.csv"], f"Actualiza tendencia de umpires {date.today().isoformat()}",
            st.secrets, APP_DIR,
        )
        (st.success if ok else st.warning)(msg)
        st.session_state["_umpire_df"] = tendency_df

    df = st.session_state.get("_umpire_df")
    if df is None and os.path.exists(umpire_csv):
        df = pd.read_csv(umpire_csv)
    if df is not None and not df.empty:
        st.dataframe(df.sort_values("n_games", ascending=False).round(3),
                     hide_index=True, use_container_width=True)
    elif df is None:
        st.info("Todavia no se ha generado umpire_tendency.csv.")


# ---------------------------------------------------------------------------
# Ajustes
# ---------------------------------------------------------------------------

def render_ajustes():
    st.header("⚙️ Ajustes y estado")
    st.caption("Que integraciones estan configuradas via Secrets de Streamlit (sin mostrar los valores).")

    def badge(ok):
        return "🟢 configurado" if ok else "⚪ no configurado"

    st.write(f"**Guardado automatico en GitHub** (para que el reentrenamiento sea permanente): "
             f"{badge(git_sync.is_configured(st.secrets))}")
    st.write(f"**Google Sheets**: "
             f"{badge(bool(st.secrets.get('GOOGLE_CREDENTIALS_JSON')) and bool(st.secrets.get('MLB_SHEET_ID')))}")
    st.write(f"**PIN de acceso**: {badge('APP_PASSWORD' in st.secrets)}")

    st.divider()
    st.caption("Archivos de datos/modelos actuales:")
    for fname in ["model.joblib", "model_1to3.joblib", "model_1to5.joblib",
                  "model_1st_total.joblib", "model_1to3_favorite.joblib", "model_1to3_total.joblib",
                  "model_1to5_favorite.joblib", "model_1to5_total.joblib",
                  "training_data.csv", "training_history.csv", "training_history_matchup.csv",
                  "predictions_log.csv", "umpire_tendency.csv"]:
        path = os.path.join(APP_DIR, fname)
        if os.path.exists(path):
            mtime = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M")
            st.caption(f"✅ {fname} — actualizado {mtime}")
        else:
            st.caption(f"⚪ {fname} — no existe todavia")


# ---------------------------------------------------------------------------
# Navegacion
# ---------------------------------------------------------------------------

def main():
    if not check_password():
        return

    st.sidebar.title("⚾ MLB Apuestas")
    section = st.sidebar.radio("Seccion", [
        "📋 Reporte del juego",
        "🤖 Prediccion rapida",
        "📈 Evaluar predicciones",
        "🔁 Reentrenar modelos",
        "⚖️ Tendencia de umpires",
        "⚙️ Ajustes",
    ])

    if section == "📋 Reporte del juego":
        render_reporte_tab()
    elif section == "🤖 Prediccion rapida":
        render_prediccion_rapida()
    elif section == "📈 Evaluar predicciones":
        render_evaluar()
    elif section == "🔁 Reentrenar modelos":
        render_reentrenar()
    elif section == "⚖️ Tendencia de umpires":
        render_umpires()
    elif section == "⚙️ Ajustes":
        render_ajustes()


if __name__ == "__main__":
    main()
