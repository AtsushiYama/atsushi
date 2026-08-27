@echo off
setlocal
cd /d "%~dp0"
if not exist "runtime" mkdir "runtime"
if not exist ".venv\Scripts\python.exe" (
  echo .venv\Scripts\python.exe not found>>"runtime\task.log"
  exit /b 2
)
".venv\Scripts\python.exe" "strength_local_notifier.py" >>"runtime\task.log" 2>&1
exit /b %ERRORLEVEL%
