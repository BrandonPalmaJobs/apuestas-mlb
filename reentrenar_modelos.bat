@echo off
setlocal
cd /d "%~dp0"

echo ================================================
echo   Reentrenar modelos con datos mas recientes
echo   (esto puede tardar 20-40 minutos, no cierres la ventana)
echo ================================================
echo.

python retrain_pipeline.py

echo.
echo ================================================
echo   Listo. Presiona una tecla para cerrar.
echo ================================================
pause >nul
