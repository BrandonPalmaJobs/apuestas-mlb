@echo off
setlocal
cd /d "%~dp0"

echo ================================================
echo   Evaluar predicciones ML vs resultado real
echo ================================================
echo.

python ml_track.py --evaluate

echo.
echo ================================================
echo   Listo. Presiona una tecla para cerrar.
echo ================================================
pause >nul
