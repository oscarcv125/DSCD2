@echo off

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo === Creando entorno virtual ===
    python -m venv .venv
    if errorlevel 1 goto error
)

call ".venv\Scripts\activate.bat"

echo.
echo === Instalando dependencias ===
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 goto error

echo.
echo === Descargando el dataset ===
python training\get_data.py

echo.
echo === Entrenando el modelo ===
python training\train.py
if errorlevel 1 goto error

echo.
echo ================================================
echo  Todo listo. Ahora abre run.bat
echo ================================================
pause
goto :eof

:error
echo.
echo Algo fallo. Revisa el mensaje de arriba.
pause
