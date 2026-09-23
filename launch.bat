@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
title AdaptDiffuse - Plug and Play
cd /d "%~dp0"

set "LOGDIR=%~dp0logs"
if not exist "%LOGDIR%" mkdir "%LOGDIR%" 2>nul
set "LOG=%LOGDIR%\launch.log"
>>"%LOG%" echo ================================================
>>"%LOG%" echo START  %date% %time%  cmd=%ComSpec%
>>"%LOG%" echo CWD=%CD%
>>"%LOG%" echo ARGS=%*
>>"%LOG%" echo ================================================
echo [log] %LOG%

echo.
echo  ================================================
echo   AdaptDiffuse  -  Stable Diffusion Engine
echo   sd-cli first: CUDA - ROCm - Vulkan - CPU
echo   Fallback: DirectML (torch)
echo  ================================================
echo.

set "PY="
set "PF86=%ProgramFiles(x86)%"
set "PYCAND=%TEMP%\adaptdiffuse_pycand_%RANDOM%%RANDOM%.txt"

call :log "[..] Detecting installed Pythons"
echo [..] Detecting installed Pythons
echo.

type nul > "%PYCAND%" 2>nul

where py >nul 2>nul
if errorlevel 1 goto :no_py_inv
call :log " [py launcher]"
echo  [py launcher]
for /f "tokens=1,*" %%A in ('py -0p 2^>nul') do (
  call :log "   %%A  %%B"
  echo    %%A  %%B
  if not "%%B"=="" >>"%PYCAND%" echo %%B
)
:no_py_inv

where python >nul 2>nul
if errorlevel 1 goto :no_path_inv
call :log " [PATH python]"
echo  [PATH]
for /f "delims=" %%P in ('where python 2^>nul') do (
  set "PVER="
  for /f "delims=" %%V in ('"%%P" --version 2^>nul') do set "PVER=%%V"
  if defined PVER (
    call :log "   !PVER!  %%P"
    echo    !PVER!  %%P
  ) else (
    call :log "   [unknown]  %%P"
    echo    [unknown]  %%P
  )
  >>"%PYCAND%" echo %%P
)
:no_path_inv

call :log " [registry]"
echo  [registry]
for %%H in (
  "HKLM\SOFTWARE\Python\PythonCore\3.10\InstallPath"
  "HKLM\SOFTWARE\WOW6432Node\Python\PythonCore\3.10\InstallPath"
  "HKCU\SOFTWARE\Python\PythonCore\3.10\InstallPath"
) do (
  for /f "tokens=2,*" %%A in ('reg query %%H 2^>nul ^| findstr /i "REG_"') do (
    if /i "%%~xA"==".exe" (
      if exist "%%B" if not "%%B"=="" (
        call :log "   %%B"
        echo    %%B
        >>"%PYCAND%" echo %%B
      )
    ) else if exist "%%Bpython.exe" (
      call :log "   %%Bpython.exe"
      echo    %%Bpython.exe
      >>"%PYCAND%" echo %%Bpython.exe
    ) else if exist "%%B\python.exe" (
      call :log "   %%B\python.exe"
      echo    %%B\python.exe
      >>"%PYCAND%" echo %%B\python.exe
    )
  )
)

call :log " [common locations]"
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
    call :log "   %%~P"
    echo    %%~P
    >>"%PYCAND%" echo %%~P
  )
)
for /d %%D in ("%LocalAppData%\Python\pythoncore-3.10*") do (
  if exist "%%D\python.exe" (
    call :log "   %%~D\python.exe"
    echo    %%~D\python.exe
    >>"%PYCAND%" echo %%D\python.exe
  )
)
echo.

where py >nul 2>nul
if errorlevel 1 goto :sel_inv
py -3.10 -c "import sys; raise SystemExit(0 if sys.version_info[:2]==(3,10) else 1)" >nul 2>nul
if errorlevel 1 goto :sel_inv
set "PY=py -3.10"
call :log "selected via py launcher: py -3.10"
goto :sel_done

:sel_inv
where py >nul 2>nul
if errorlevel 1 goto :sel_list
for /f "tokens=1,*" %%A in ('py -0p 2^>nul') do (
  if defined PY goto :sel_list
  if /i "%%A"=="-V:3.10" if exist "%%B" (
    set PY="%%B"
    call :log "selected via py -0p -V:3.10: %%B"
  )
)

:sel_list
if defined PY goto :sel_done
for /f "usebackq delims=" %%P in ("%PYCAND%") do (
  if defined PY goto :sel_done
  echo %%P | findstr /i /c:"\WindowsApps\" >nul
  if errorlevel 1 (
    "%%P" -c "import sys; raise SystemExit(0 if sys.version_info[:2]==(3,10) else 1)" >nul 2>nul
    if not errorlevel 1 (
      set PY="%%P"
      call :log "selected from inventory: %%P"
    )
  ) else (
    call :log "skip WindowsApps stub: %%P"
  )
)

:sel_done
if exist "%PYCAND%" del /q "%PYCAND%" >nul 2>nul

if defined PY goto :py_ok
call :log "[ERROR] Python 3.10 not found"
echo [ERROR] Python 3.10 not found on this system.
echo         Install from https://www.python.org/downloads/release/python-3100/
echo         Re-run launch.bat after installing.
call :fail 1

:py_ok
%PY% -c "import sys; raise SystemExit(0 if sys.version_info[:2]==(3,10) else 1)" >nul 2>nul
if not errorlevel 1 goto :py_ver_ok
call :log "[ERROR] Selected interpreter is not Python 3.10: %PY%"
echo [ERROR] Selected interpreter is not Python 3.10: %PY%
call :fail 1

:py_ver_ok
set "PYVER=Python 3.10"
for /f "delims=" %%V in ('%PY% --version 2^>nul') do set "PYVER=%%V"
call :log "[OK] Using %PYVER% via %PY%"
echo [OK] Using %PYVER%  ^(%PY%^)
echo.

set "VPY=.venv\Scripts\python.exe"
if not exist "%VPY%" goto :venv_create
"%VPY%" -c "import sys; raise SystemExit(0 if sys.version_info[:2]==(3,10) else 1)" >nul 2>nul
if not errorlevel 1 goto :venv_ok
call :log "[i] Existing venv is not Python 3.10 - recreating"
echo [i] Existing .venv is not Python 3.10 - recreating it
rmdir /s /q ".venv" >nul 2>nul
if exist ".setup_done" del /q ".setup_done" >nul 2>nul

:venv_create
call :log "[1/4] Creating virtual environment"
echo [1/4] Creating virtual environment...
%PY% -m venv .venv >>"%LOG%" 2>&1
if not errorlevel 1 goto :venv_ok
call :log "[ERROR] Failed to create venv"
echo [ERROR] Failed to create venv.
call :fail 1

:venv_ok
if exist "%VPY%" goto :venv_have
call :log "[ERROR] venv python missing after create"
echo [ERROR] .venv\Scripts\python.exe missing
call :fail 1

:venv_have
call :log "[1/4] venv ready"
if exist ".setup_done" goto :setup_skip

call :log "[2/4] Installing core packages - first run"
echo [2/4] Installing core packages - first run only...
"%VPY%" -m pip install -q --upgrade pip >>"%LOG%" 2>&1
call :log "pip upgrade exit=!errorlevel!"
"%VPY%" -m pip install -q -r requirements.txt >>"%LOG%" 2>&1
if errorlevel 1 (
  call :log "[WARN] Some packages failed - continuing"
  echo [WARN] Some packages failed; continuing with backend setup
) else (
  call :log "[2/4] requirements.txt OK"
)

call :log "[3/4] Detecting GPU / installing backend"
echo [3/4] Detecting GPU / installing best backend...
"%VPY%" -m app.backend >>"%LOG%" 2>&1
if errorlevel 1 (
  call :log "[WARN] app.backend error - trying defaults"
  echo [WARN] Backend auto-detect returned an error; trying defaults
) else (
  call :log "[3/4] app.backend OK"
)
"%VPY%" -m app.main --setup >>"%LOG%" 2>&1
if errorlevel 1 (
  call :log "[WARN] app.main --setup errors - WebUI will try CPU"
  echo [WARN] Backend setup reported errors; WebUI will still try CPU mode
) else (
  call :log "[3/4] app.main --setup OK"
)
echo done> ".setup_done"
call :log "wrote .setup_done"
goto :launch

:setup_skip
call :log "[2/4] Setup already completed - skip install"
echo [2/4] Setup already completed - skipping install.

:launch
call :log "[4/4] Launching WebUI"
echo [4/4] Launching WebUI...
echo.
echo  Open in browser:  http://127.0.0.1:7860
echo  Press Ctrl+C in this window to stop.
echo.

"%VPY%" -m app.main --host 127.0.0.1 --port 7860
set "WEXIT=!errorlevel!"
call :log "WebUI exited with code !WEXIT!"

echo.
if not "!WEXIT!"=="0" (
  echo [ERROR] WebUI exited with code !WEXIT! - see log:
  echo   %LOG%
) else (
  echo AdaptDiffuse stopped.
)
echo.
call :log "END %date% %time%"
echo Press any key to close...
pause >nul
exit /b 0

:log
>>"%LOG%" echo [%date% %time%] %~1
exit /b 0

:fail
set "EC=%~1"
>>"%LOG%" echo [%date% %time%] FATAL exit code %EC%
echo.
echo [FATAL] exit code %EC%  - log: %LOG%
echo Press any key to close...
pause >nul
exit /b %EC%
