# MLB Apuestas — app web

Interfaz web (Streamlit) para correr el sistema desde el celular o cualquier
navegador, sin usar los `.bat`. No cambia el analisis: la app llama a las
mismas funciones que ya usaban `mlb_first_inning_report.py`, `ml_predict.py`,
`ml_track.py`, `umpire_data.py` y `ml_train.py`; solo le pone una interfaz
encima y hace que el reentrenamiento se pueda disparar desde el celular.

Secciones de la app (siguen exactamente lo que hacia cada `.bat`):

| Antes (.bat)                  | Ahora (pestana en la app) |
|--------------------------------|---------------------------|
| `correr_reporte.bat`           | 📋 Reporte del juego |
| `correr_prediccion_ml.bat`     | 🤖 Prediccion rapida |
| `evaluar_predicciones.bat`     | 📈 Evaluar predicciones |
| `reentrenar_modelos.bat`       | 🔁 Reentrenar modelos |
| `correr_umpires.bat`           | ⚖️ Tendencia de umpires |

Los `.bat` se pueden dejar como respaldo para correr todo en la compu si
hace falta, pero ya no son necesarios para el uso diario.

## Por que hace falta un paso extra para el reentrenamiento

Streamlit Community Cloud (el hosting gratuito) no garantiza disco
persistente: cuando la app se "duerme" por inactividad y alguien la vuelve a
abrir, se recrea clonando el repo de GitHub de nuevo. Si reentrenas los
modelos y ese resultado no se sube a GitHub, se pierde en el siguiente
reinicio. Por eso la app, despues de un reentrenamiento exitoso, hace commit
y push de los modelos actualizados al repo automaticamente — para eso
necesita un `GITHUB_TOKEN` (Paso 3 de abajo). Si no lo configuras, la app
funciona igual, solo que el reentrenamiento no sobrevive un reinicio.

## Paso 1 — Repo de GitHub ✅ (ya hecho)

Ya cree y subi el codigo a un repositorio **privado**:
https://github.com/BrandonPalmaJobs/apuestas-mlb

## Paso 2 — Token para que el reentrenamiento se guarde solo

1. Ve a https://github.com/settings/tokens?type=beta (Fine-grained tokens)
2. "Generate new token"
3. Repository access: **Only select repositories** → `apuestas-mlb`
4. Permissions → Repository permissions → **Contents: Read and write**
5. Generate token, copia el valor (empieza con `github_pat_...`) —
   solo se muestra una vez

(Este paso lo tienes que hacer tu desde el navegador — es una credencial
personal, no algo que yo deba generar o ver por ti.)

## Paso 3 — Desplegar en Streamlit Community Cloud

1. Ve a https://share.streamlit.io e inicia sesion con tu cuenta de GitHub
2. "Create app" → "Deploy a public app from a repo" (el repo privado se ve
   igual una vez autorizado el acceso de Streamlit a GitHub)
3. Repository: `BrandonPalmaJobs/apuestas-mlb`
4. Branch: `main`
5. Main file path: `streamlit_app.py`
6. Antes de darle "Deploy", abre "Advanced settings" y pega esto en el
   cuadro de **Secrets** (ajusta los valores):

   ```toml
   GITHUB_TOKEN = "github_pat_xxxxxxxxxxxx"
   GITHUB_REPO = "BrandonPalmaJobs/apuestas-mlb"

   # Opcional: pide un PIN antes de dejar entrar a la app
   APP_PASSWORD = "elige-un-pin"

   # Opcional: solo si vas a usar la integracion con Google Sheets
   # (ver setup_google_sheets.md para como generar el credentials.json)
   MLB_SHEET_ID = "id-de-tu-spreadsheet"

   [GOOGLE_CREDENTIALS_JSON]
   type = "service_account"
   project_id = "..."
   private_key_id = "..."
   private_key = "..."
   client_email = "..."
   client_id = "..."
   # ... (el resto de campos del credentials.json que descargaste de Google)
   ```

7. Deploy. La primera vez tarda 1-3 minutos en instalar las dependencias.

## Paso 4 — Usarla desde el celular

Abre la URL que te da Streamlit (algo como
`https://apuestas-mlb-xxxx.streamlit.app`) en el navegador del celular. Para
que se sienta como una app:

- **iPhone (Safari):** boton de compartir → "Agregar a pantalla de inicio"
- **Android (Chrome):** menu (⋮) → "Agregar a pantalla principal"

## Notas

- El reentrenamiento tarda 20-45 minutos igual que antes — no cierres la
  pestana ni bloquees el celular mientras corre.
- Puedes cambiar los secrets despues del deploy en cualquier momento desde
  el dashboard de Streamlit Cloud (⋮ sobre tu app → Settings → Secrets).
- Si en algun momento quieres correr la app en tu compu para probar cambios
  antes de subirlos: `pip install -r requirements.txt` y luego
  `streamlit run streamlit_app.py`.
