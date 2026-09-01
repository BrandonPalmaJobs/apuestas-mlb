@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo   REPORTE MLB - analisis pre-apuesta (1er inning y mas)
echo ============================================================
echo.
echo Este programa te va a pedir un par de datos y despues genera
echo un reporte completo del juego: abridores, clima, lesionados,
echo lineup, umpire, predicciones, etc. Solo sigue las preguntas.
echo.
echo ------------------------------------------------------------
echo PASO 1 de 3: Que equipos juegan
echo ------------------------------------------------------------
echo Puedes escribir el nombre completo, la ciudad, el nombre del
echo equipo o su abreviacion. Ejemplos validos: "Reds", "Cincinnati",
echo "Yankees", "NYY". No importa mayusculas/minusculas.
echo.
set /p EQUIPO_A="  Equipo A: "
set /p EQUIPO_B="  Equipo B: "

echo.
echo ------------------------------------------------------------
echo PASO 2 de 3: Forzar el abridor (opcional, casi nunca hace falta)
echo ------------------------------------------------------------
echo El reporte YA detecta solo quien es el abridor probable de cada
echo equipo. Usa esto UNICAMENTE si ves en las casas de apuestas que
echo hubo un cambio de rotacion muy reciente que el reporte todavia
echo no reconoce (la API de MLB a veces tarda unas horas en
echo actualizarse).
echo.
echo Si el abridor que muestre el reporte esta bien, deja estas dos
echo preguntas VACIAS y solo presiona Enter.
echo.
set /p PITCHER_A="  Forzar abridor del Equipo A (Enter para omitir): "
set /p PITCHER_B="  Forzar abridor del Equipo B (Enter para omitir): "

set OVERRIDES=
if not "%PITCHER_A%"=="" set OVERRIDES=%OVERRIDES% --pitcher-a "%PITCHER_A%"
if not "%PITCHER_B%"=="" set OVERRIDES=%OVERRIDES% --pitcher-b "%PITCHER_B%"

echo.
echo ------------------------------------------------------------
echo PASO 3 de 3: Guardarlo tambien en Google Sheets (opcional)
echo ------------------------------------------------------------
echo Si dices que si, el reporte se escribe ademas en una hoja de
echo calculo de Google (sobrescribe siempre la misma plantilla).
echo Para esto necesitas tener ya configurado el archivo
echo credentials.json en esta misma carpeta - si no lo tienes o no
echo sabes que es esto, responde "n" y seguimos sin problema
echo (revisa setup_google_sheets.md si mas adelante quieres activarlo).
echo.
set /p USAR_SHEET="  Escribir tambien en Google Sheets? (s = si / n = no): "

echo.
echo ============================================================
echo   Generando el reporte, esto puede tardar 1-2 minutos...
echo   (revisa clima, lesionados, umpire, lineup, historial, etc.)
echo ============================================================
echo.

if /i "%USAR_SHEET%"=="s" (
    echo El ID es la parte de la URL de tu hoja entre "/d/" y "/edit".
    set /p SHEET_ID="  ID de tu Google Spreadsheet: "
    python mlb_first_inning_report.py "%EQUIPO_A%" "%EQUIPO_B%" --sheet-id "%SHEET_ID%" --credentials credentials.json%OVERRIDES%
) else (
    python mlb_first_inning_report.py "%EQUIPO_A%" "%EQUIPO_B%"%OVERRIDES%
)

echo.
echo ============================================================
echo   LISTO. Si algo salio mal, revisa arriba el mensaje de error
echo   (los mas comunes: nombre de equipo mal escrito, o no hay
echo   internet). Presiona una tecla para cerrar esta ventana.
echo ============================================================
pause >nul
