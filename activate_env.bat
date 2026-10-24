@echo off
REM ============================================================
REM  Activate the project-local Geology RAG environment.
REM  Opens a CMD where `python` points into .conda-env.
REM  Run this, then use init_demo.py / streamlit inside that window.
REM ============================================================

set "PROJECT_DIR=%~dp0"
set "PROJECT_DIR=%PROJECT_DIR:~0,-1%"
set "ENV_DIR=%PROJECT_DIR%\.conda-env"

if not exist "%ENV_DIR%\python.exe" (
    echo Environment not found at:
    echo   %ENV_DIR%
    echo.
    echo Run setup_env.bat first.
    pause
    exit /b 1
)

REM Locate conda
set "CONDA_EXE="
if exist "%USERPROFILE%\miniconda3\Scripts\conda.exe" set "CONDA_EXE=%USERPROFILE%\miniconda3\Scripts\conda.exe"
if not defined CONDA_EXE if exist "%USERPROFILE%\anaconda3\Scripts\conda.exe" set "CONDA_EXE=%USERPROFILE%\anaconda3\Scripts\conda.exe"
if not defined CONDA_EXE (
    echo conda.exe not found. Activate manually with:
    echo   call %ENV_DIR%\activate.bat
    pause
    exit /b 1
)

call "%CONDA_EXE%" activate "%ENV_DIR%"

echo.
echo Active environment: %PREFIX%
echo Python: 
python --version
echo CUDA available:
python -c "import torch; print(torch.cuda.is_available())" 2>nul
echo.
echo Ready. Try:
echo   python init_demo.py
echo   streamlit run src\app\streamlit_app.py
echo.
cmd /k
