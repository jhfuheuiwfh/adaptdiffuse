@echo off
setlocal EnableExtensions EnableDelayedExpansion
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

:: ---------- Python: detect ALL installs, use 3.10 ----------
set "PY="
set "PF86=%ProgramFiles(x86)%"
set "PYCAND=%TEMP%\adaptdiffuse_pycand_%RANDOM%%RANDOM%.txt"

echo [..] Detecting installed Pythons...
echo.
type nul > "%PYCAND%" 2>nul

:: --- inventory: Windows py launcher ---
where py >nul 2>nul
if not errorlevel 1 (
  echo  [py launcher]
  for /f "tokens=1,*" %%A in ('py -0p 2^>nul') do (
    echo    %%A  %%B
    if not "%%B"=="" >>"%PYCAND%" echo %%B
  )
)

:: --- inventory: python on PATH ---
where python >nul 2>nul
if not errorlevel 1 (
  echo  [PATH]
  for /f "delims=" %%P in ('where python 2^>nul') do (
    set "PVER="
    for /f "delims=" %%V in ('"%%P" --version 2^>nul') do set "PVER=%%V"
    if defined PVER (echo    !PVER!  %%P) else (echo    [unknown]  %%P)
    >>"%PYCAND%" echo %%P
  )
)

:: --- inventory: registry InstallPath (3.10, all hives) ---
echo  [registry]
for %%H in (
  "HKLM\SOFTWARE\Python\PythonCore\3.10\InstallPath"
  "HKLM\SOFTWARE\WOW6432Node\Python\PythonCore\3.10\InstallPath"
  "HKCU\SOFTWARE\Python\PythonCore\3.10\InstallPath"
) do (
  for /f "tokens=2,*" %%A in ('reg query %%H 2^>nul ^| findstr /i "REG_"') do (
    if /i "%%~xA"==".exe" (
      if exist "%%B" if not "%%B"=="" (
        echo    %%B
        >>"%PYCAND%" echo %%B
      )
    ) else if exist "%%Bpython.exe" (
      echo    %%Bpython.exe
      >>"%PYCAND%" echo %%Bpython.exe
    ) else if exist "%%B\python.exe" (
      echo    %%B\python.exe
      >>"%PYCAND%" echo %%B\python.exe
    )
  )
)

:: --- inventory: well-known install locations ---
echo  [common locations]
for %%P in (
  "%LocalAppData%\Programs\Python\Python310\python.exe"
  "%LocalAppData%\Programs\Python\Python310-32\python.exe"
  "%ProgramFiles%\Python310\python.exe"
  "%PF86%\Python310\python.exe"
  "%SystemDrive%\Python310\python.exe"
  "%SystemDrive%\Python310-32\python.exe"
) do (
  if exist "%%P" (
    echo    %%~P
    >>"%PYCAND%" echo %%~P
  )
)
for /d %%D in ("%LocalAppData%\Python\pythoncore-3.10*") do (
  if exist "%%D\python.exe" (
    echo    %%~D\python.exe
    >>"%PYCAND%" echo %%~D\python.exe
  )
)
echo.

:: --- selection: prefer the py launcher's 3.10 ---
where py >nul 2>nul
if not errorlevel 1 (
  py -3.10 -c "import sys; raise SystemExit(0 if sys.version_info[:2]==(3,10) else 1)" >nul 2>nul
  if not errorlevel 1 set "PY=py -3.10"
)

:: --- selection: exact -V:3.10 entry from the launcher list ---
if not defined PY (
  where py >nul 2>nul
  if not errorlevel 1 (
    for /f "tokens=1,*" %%A in ('py -0p 2^>nul') do (
      if not defined PY if /i "%%A"=="-V:3.10" if exist "%%B" set PY="%%B"
    )
  )
)

:: --- selection: first real 3.10 executable from the inventory ---
if not defined PY (
  for /f "usebackq delims=" %%P in ("%PYCAND%") do (
    if not defined PY (
      echo %%P | findstr /i /c:"\WindowsApps\" >nul
      if errorlevel 1 (
        "%%P" -c "import sys; raise SystemExit(0 if sys.version_info[:2]==(3,10) else 1)" >nul 2>nul
        if not errorlevel 1 set PY="%%P"
      )
    )
  )
)

if exist "%PYCAND%" del /q "%PYCAND%" >nul 2>nul

if not defined PY (
  echo [ERROR] Python 3.10 not found on this system.
  echo         Install it from https://www.python.org/downloads/release/python-3100/
  echo         ^(or any 3.10.x build; enable "Add python.exe to PATH"^).
  echo         Re-run launch.bat after installing.
  pause
  exit /b 1
)

%PY% -c "import sys; raise SystemExit(0 if sys.version_info[:2]==(3,10) else 1)" >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Selected interpreter is not Python 3.10: %PY%
  pause
  exit /b 1
)

set "PYVER=Python 3.10"
for /f "delims=" %%V in ('%PY% --version 2^>nul') do set "PYVER=%%V"
echo [OK] Using %PYVER%  ^(%PY%^)
echo.

:: ---------- VENV ----------
set "VPY=.venv\Scripts\python.exe"
if exist "%VPY%" (
  "%VPY%" -c "import sys; raise SystemExit(0 if sys.version_info[:2]==(3,10) else 1)" >nul 2>nul
  if errorlevel 1 (
    echo [i] Existing .venv is not Python 3.10 - recreating it...
    rmdir /s /q ".venv" >nul 2>nul
    if exist ".setup_done" del /q ".setup_done" >nul 2>nul
  )
)

if not exist ".venv\Scripts\python.exe" (
  echo [1/4] Creating virtual environment...
  %PY% -m venv .venv
  if errorlevel 1 (
    echo [ERROR] Failed to create venv.
    pause
    exit /b 1
  )
)

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
