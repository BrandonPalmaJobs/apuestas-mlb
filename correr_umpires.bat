@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo   INDICE DE TENDENCIA DE UMPIRES (home plate) - MLB
echo ============================================================
echo.
echo Esto recorre los juegos ya jugados de la temporada y arma un
echo archivo (umpire_tendency.csv) con que tan "generoso" o
echo "apretado" ha sido cada umpire de home plate (carreras/
echo ponches/base por bolas por juego que dirigio, comparado
echo contra el promedio de la liga).
echo.
echo Es un PROXY, no la zona de strike real - se explica bien en el
echo reporte principal cuando aparece. Una vez que generes este
echo archivo, mlb_first_inning_report.py lo usa automaticamente
echo para mostrar la tendencia del umpire asignado a cada juego.
echo.
echo Recorrer la TEMPORADA COMPLETA puede tardar bastante (son
echo cientos de juegos, un request por juego). No hace falta
echo correrlo seguido - una vez, y despues cada semana o dos para
echo refrescarlo con los juegos mas recientes.
echo.
set /p SEASON="Temporada a analizar (Enter para usar la actual): "

echo.
echo ============================================================
echo   Generando el indice, esto puede tardar varios minutos...
echo   No cierres esta ventana.
echo ============================================================
echo.

if "%SEASON%"=="" (
    python umpire_data.py --out umpire_tendency.csv
) else (
    python umpire_data.py --season %SEASON% --out umpire_tendency.csv
)

echo.
echo ============================================================
echo   LISTO. Si se genero umpire_tendency.csv sin errores arriba,
echo   el reporte principal (mlb_first_inning_report.py o
echo   correr_reporte.bat) ya lo va a usar automaticamente.
echo   Presiona una tecla para cerrar esta ventana.
echo ============================================================
pause >nul
