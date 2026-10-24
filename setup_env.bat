@echo off
REM ============================================================
REM  Geology RAG Demo - Environment Installer
REM  Installs everything INSIDE this project folder.
REM
REM  Usage (run in CMD):
REM     setup_env.bat              CPU-only torch  (smallest, ~3-4 GB)
REM     setup_env.bat GPU          CUDA torch for RTX 4060 (~6-8 GB)
REM     setup_env.bat GPU MINERU   also install MinerU parser (+4-6 GB)
REM
REM  Created at:  D:\RAG\geology-rag-demo\.conda-env
REM  Log file:    D:\RAG\geology-rag-demo\setup_env.log
REM ============================================================

setlocal enabledelayedexpansion

set "PROJECT_DIR=%~dp0"
set "PROJECT_DIR=%PROJECT_DIR:~0,-1%"
set "ENV_DIR=%PROJECT_DIR%\.conda-env"
set "MODEL_DIR=%PROJECT_DIR%\models"
set "LOG_FILE=%PROJECT_DIR%\setup_env.log"
set "PIP_MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple"
set "PYTORCH_CU=https://download.pytorch.org/whl/cu121"
set "ALIYUN_CU=https://mirrors.aliyun.com/pytorch-wheels/cu121/torch_stable.html"

set "USE_GPU=0"
set "USE_MINERU=0"
for %%A in (%*) do (
    if /I "%%A"=="GPU"    set "USE_GPU=1"
    if /I "%%A"=="MINERU" set "USE_MINERU=1"
)

echo ============================================================ > "%LOG_FILE%"
echo  Geology RAG Demo install started >> "%LOG_FILE%"
echo ============================================================ >> "%LOG_FILE%"

echo ============================================================
echo  Geology RAG Demo - Environment Installer
echo ============================================================
echo   Project : %PROJECT_DIR%
echo   Env     : %ENV_DIR%
echo   Models  : %MODEL_DIR%
echo   Torch   :
if "%USE_GPU%"=="1" (echo               CUDA 12.1 ^(GPU, RTX 4060^) - big install) else (echo               CPU only - small install, rerun with GPU later)
echo   MinerU  :
if "%USE_MINERU%"=="1" (echo               YES - heavy, +4-6 GB) else (echo               no - pymupdf fallback only)
echo.

REM ------------------------------------------------------------
REM [1/6] Locate conda
REM ------------------------------------------------------------
echo [1/6] Locating conda...
set "CONDA_EXE="
if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\..\Scripts\conda.exe" set "CONDA_EXE=%CONDA_PREFIX%\..\Scripts\conda.exe"
if not defined CONDA_EXE if exist "%USERPROFILE%\miniconda3\Scripts\conda.exe" set "CONDA_EXE=%USERPROFILE%\miniconda3\Scripts\conda.exe"
if not defined CONDA_EXE if exist "%USERPROFILE%\anaconda3\Scripts\conda.exe" set "CONDA_EXE=%USERPROFILE%\anaconda3\Scripts\conda.exe"
if not defined CONDA_EXE (
    echo   ERROR: conda not found in default locations.
    echo   Install Miniconda first, or edit CONDA_EXE in this script.
    pause & exit /b 1
)
echo   Found: %CONDA_EXE%
for %%F in ("%CONDA_EXE%") do set "CONDA_BASE=%%~dpF.."
echo.

REM ------------------------------------------------------------
REM [2/6] Accept conda channel ToS (needed on conda >= 23.7)
REM ------------------------------------------------------------
echo [2/6] Accepting conda channel Terms of Service...
call "%CONDA_EXE%" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main   >> "%LOG_FILE%" 2>&1 || echo   ^(ToS step skipped - older conda, ok^)
call "%CONDA_EXE%" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r      >> "%LOG_FILE%" 2>&1
call "%CONDA_EXE%" config --set always_yes true                                                   >> "%LOG_FILE%" 2>&1
echo   Done.
echo.

REM ------------------------------------------------------------
REM [3/6] Create project-local conda environment
REM ------------------------------------------------------------
echo [3/6] Creating environment at %ENV_DIR% ...
if exist "%ENV_DIR%\python.exe" (
    echo   Environment already exists. Reusing it.
    echo   ^(delete the .conda-env folder for a clean reinstall^)
) else (
    call "%CONDA_EXE%" create --prefix "%ENV_DIR%" python=3.10 -y >> "%LOG_FILE%" 2>&1
    if errorlevel 1 (
        echo   ERROR: conda create failed. See %LOG_FILE%
        echo   Last lines of log:
        powershell -NoProfile -Command "Get-Content '%LOG_FILE%' -Tail 20"
        pause & exit /b 1
    )
    echo   Created.
)

set "PY=%ENV_DIR%\python.exe"
set "PIP=%PY% -m pip"
"%PY%" --version
echo.

REM ------------------------------------------------------------
REM [4/6] Install PyTorch (must come before FlagEmbedding)
REM ------------------------------------------------------------
echo [4/6] Installing PyTorch...
if "%USE_GPU%"=="1" (
    echo   Trying official CUDA 12.1 wheel index...
    %PIP% install torch torchvision --index-url %PYTORCH_CU% >> "%LOG_FILE%" 2>&1
    if errorlevel 1 (
        echo   Official index failed, trying Aliyun mirror...
        %PIP% install torch torchvision -f %ALIYUN_CU% -i %PIP_MIRROR% >> "%LOG_FILE%" 2>&1
    )
) else (
    echo   Installing CPU build from Tsinghua mirror...
    %PIP% install torch torchvision -i %PIP_MIRROR% >> "%LOG_FILE%" 2>&1
)
if errorlevel 1 (
    echo   ERROR: PyTorch install failed. See %LOG_FILE%
    pause & exit /b 1
)
%PIP% show torch 2>nul | findstr /I "^Version" 
echo.

REM ------------------------------------------------------------
REM [5/6] Install project dependencies
REM ------------------------------------------------------------
echo [5/6] Installing core dependencies...
%PIP% install -r "%PROJECT_DIR%\requirements.txt" -i %PIP_MIRROR% >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    echo   Core install had issues. Retrying with default index...
    %PIP% install -r "%PROJECT_DIR%\requirements.txt" >> "%LOG_FILE%" 2>&1
)

if "%USE_MINERU%"=="1" (
    echo.
    echo   Installing MinerU ^(heavy, this takes a while^)...
    %PIP% install -r "%PROJECT_DIR%\requirements-mineru.txt" -i %PIP_MIRROR% >> "%LOG_FILE%" 2>&1
)
echo   Done.
echo.

REM ------------------------------------------------------------
REM [6/6] Local config + HuggingFace cache + verification
REM ------------------------------------------------------------
echo [6/6] Finalizing local configuration...
if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%"

REM Keep model weights inside the project so C: drive never fills up.
call "%CONDA_EXE%" env config vars set --prefix "%ENV_DIR%" ^
    HF_HOME="%MODEL_DIR%\hf-cache" ^
    HUGGINGFACE_HUB_CACHE="%MODEL_DIR%\hf-cache\hub" ^
    TORCH_HOME="%MODEL_DIR%\torch" ^
    HF_ENDPOINT="https://hf-mirror.com" >> "%LOG_FILE%" 2>&1
echo   Model cache pinned to %MODEL_DIR%\hf-cache

REM Create .env from template if missing
if not exist "%PROJECT_DIR%\.env" (
    copy "%PROJECT_DIR%\.env.example" "%PROJECT_DIR%\.env" >nul
    echo   Created .env  --  EDIT IT and put in your DEEPSEEK_API_KEY
) else (
    echo   .env already exists, left untouched
)

REM Import smoke test
echo.
echo   Running import check...
"%PY%" -c "import sys; sys.path.insert(0,'%PROJECT_DIR%'); import torch, chromadb, neo4j, FlagEmbedding, openai, streamlit, ragas, datasets; from transformers.utils import is_torch_available as f; print('   OK  torch', torch.__version__, '| cuda available:', torch.cuda.is_available(), '| transformers sees torch:', f())" 2>>"%LOG_FILE%"
if errorlevel 1 (
    echo   WARNING: some imports failed. Check %LOG_FILE% - often a CUDA torch mismatch.
)

echo.
echo ============================================================
echo  Install finished.
echo ============================================================
echo.
echo  Activate the environment (any CMD window):
echo      %CONDA_EXE% activate "%ENV_DIR%"
echo    ...or just double-click:  activate_env.bat
echo.
echo  Then build the index:
echo      python init_demo.py
echo.
echo  Then start the web UI:
echo      streamlit run src\app\streamlit_app.py
echo.
echo  IMPORTANT: first run downloads bge-m3 + reranker ^(about 4.6 GB^)
echo  into %MODEL_DIR%\hf-cache. Download goes through hf-mirror.com.
echo    - VPN OFF  = correct for hf-mirror.com  ^(default^)
echo    - VPN ON   = remove HF_ENDPOINT if you want to pull from huggingface.co
echo.
echo  Full log: %LOG_FILE%
echo ============================================================
pause
endlocal
