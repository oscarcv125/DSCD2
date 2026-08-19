@echo off

cd /d "%~dp0"

if not exist "models\bank_marketing_pipeline.joblib" (
    echo No hay modelo entrenado. Corre primero setup.bat
    pause
    goto :eof
)

call ".venv\Scripts\activate.bat"

start "" http://127.0.0.1:8000

echo.
echo API corriendo en http://127.0.0.1:8000
echo Para detenerla, presiona Ctrl+C
echo.
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
