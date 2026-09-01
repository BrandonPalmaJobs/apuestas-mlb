"""
Guarda archivos (modelos reentrenados, datos, logs) de vuelta al repo de
GitHub del proyecto, para que sobrevivan un reinicio del contenedor de
Streamlit Cloud.

Por que hace falta esto: Streamlit Community Cloud NO tiene disco
persistente garantizado. Cuando la app "se duerme" por inactividad y
alguien la vuelve a abrir, el contenedor se recrea clonando el repo de
GitHub de nuevo - cualquier archivo generado localmente (model.joblib
reentrenado, training_data.csv actualizado, predictions_log.csv) que no se
haya subido a GitHub se pierde.

Requiere dos secrets configurados en Streamlit (Settings -> Secrets):
    GITHUB_TOKEN = "token con permiso de escritura sobre el repo"
    GITHUB_REPO  = "usuario/nombre-del-repo"
"""

import subprocess


def is_configured(secrets):
    return bool(secrets.get("GITHUB_TOKEN")) and bool(secrets.get("GITHUB_REPO"))


def _run(args, cwd):
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True)


def commit_and_push(paths, message, secrets, cwd):
    """Hace commit y push de `paths` (rutas relativas a `cwd`) al repo
    configurado en los secrets. No toca la config global de git ni el
    remote 'origin' - arma la URL de push con el token solo para este
    comando. Regresa (ok: bool, mensaje: str)."""
    if not is_configured(secrets):
        return False, "GITHUB_TOKEN / GITHUB_REPO no configurados en Secrets - el cambio no se guardo en GitHub."

    token = secrets["GITHUB_TOKEN"]
    repo = secrets["GITHUB_REPO"].strip().strip("/")
    remote = f"https://x-access-token:{token}@github.com/{repo}.git"

    existing = [p for p in paths if p]
    add = _run(["git", "add", "--", *existing], cwd)
    if add.returncode != 0:
        return False, f"git add fallo: {add.stderr.strip()}"

    status = _run(["git", "status", "--porcelain", "--", *existing], cwd)
    if not status.stdout.strip():
        return True, "Sin cambios nuevos que guardar (ya estaba al dia en GitHub)."

    commit = _run(
        ["git", "-c", "user.email=mlb-apuestas-app@local",
         "-c", "user.name=MLB Apuestas App", "commit", "-m", message],
        cwd,
    )
    if commit.returncode != 0:
        return False, f"git commit fallo: {commit.stderr.strip()}"

    branch_res = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd)
    branch = branch_res.stdout.strip() or "main"

    push = _run(["git", "push", remote, f"HEAD:{branch}"], cwd)
    if push.returncode != 0:
        return False, f"git push fallo: {push.stderr.strip()}"

    return True, f"Guardado en GitHub ({repo}, rama {branch})."
