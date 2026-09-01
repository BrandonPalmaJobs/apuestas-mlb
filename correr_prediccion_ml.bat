@echo off
setlocal
cd /d "%~dp0"

echo ================================================
echo   Prediccion ML - 0 carreras en el 1er inning
echo ================================================
echo.

set /p EQUIPO_A="Nombre del Equipo A (ej. Reds): "
set /p EQUIPO_B="Nombre del Equipo B (ej. Brewers): "

echo.
echo Calculando prediccion, espera un momento...
echo.

python ml_predict.py "%EQUIPO_A%" "%EQUIPO_B%" --model model.joblib

echo.
echo ================================================
echo   Listo. Presiona una tecla para cerrar.
echo ================================================
pause >nul
