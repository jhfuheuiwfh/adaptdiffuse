@echo off
setlocal EnableExtensions
chcp 65001 >nul
title AdaptDiffuse - Plug and Play
cd /d "%~dp0"

echo.
echo  ================================================
echo   AdaptDiffuse  -  Stable Diffusion Engine
echo   sd-cli first: CUDA - ROCm - Vulkan - CPU
echo   Fallback: DirectML (torch)
echo  ================================================
echo.

:: ---------- Python ----------
set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY (
  where python >nul 2>nul && set "PY=python"
)
if not defined PY (
  echo [ERROR] Python 3.10+ not found. Install from https://www.python.org/downloads/
  echo         Enable "Add python.exe to PATH" during setup.
  pause
  exit /b 1
)

%PY% -c "import sys; raise SystemExit(0 if sys.version_info>=(3,10) else 1)" 2>nul
if errorlevel 1 (
  echo [ERROR] Python 3.10+ is required.
  pause
  exit /b 1
)

:: ---------- VENV ----------
if not exist ".venv\Scripts\python.exe" (
  echo [1/4] Creating virtual environment...
  %PY% -m venv .venv
  if errorlevel 1 (
    echo [ERROR] Failed to create venv.
    pause
    exit /b 1
  )
)

set "VPY=.venv\Scripts\python.exe"
set "VPIP=.venv\Scripts\python.exe -m pip"

:: ---------- First-run marker ----------
if not exist ".setup_done" (
  echo [2/4] Installing core packages (first run only)...
  "%VPY%" -m pip install -q --upgrade pip
  "%VPY%" -m pip install -q -r requirements.txt
  if errorlevel 1 (
    echo [WARN] Some packages failed; continuing with backend setup...
  )

  echo [3/4] Detecting GPU / installing best backend...
  "%VPY%" -m app.backend
  if errorlevel 1 (
    echo [WARN] Backend auto-detect returned an error; trying defaults...
  )
  "%VPY%" -m app.main --setup
  if errorlevel 1 (
    echo [WARN] Backend setup reported errors; WebUI will still try CPU mode.
  )
  echo done> ".setup_done"
) else (
  echo [2/4] Setup already completed - skipping install.
)

echo [4/4] Launching WebUI...
echo.
echo  Open in browser:  http://127.0.0.1:7860
echo  Press Ctrl+C in this window to stop.
echo.

"%VPY%" -m app.main --host 127.0.0.1 --port 7860

echo.
echo AdaptDiffuse stopped.
pause
exit /b 0
