
@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ==========================================
REM Config
REM ==========================================
set "DOWNLOAD_ROOT=%CD%\downloads"
set "LOG_ROOT=%CD%\logs"

REM Create output directories
if not exist "%DOWNLOAD_ROOT%" mkdir "%DOWNLOAD_ROOT%"
if not exist "%LOG_ROOT%" mkdir "%LOG_ROOT%"

REM Check wget availability
where wget >nul 2>&1
if errorlevel 1 (
  echo [ERROR] wget.exe not found in PATH. Install it (e.g., "choco install wget") and retry.
  exit /b 1
)

REM Timestamp (locale-agnostic-ish): YYYYMMDD_HHMMSS
set "YY=%date:~0,4%"
set "MM=%date:~5,2%"
set "DD=%date:~8,2%"
set "HH=%time:~0,2%"
set "MN=%time:~3,2%"
set "SS=%time:~6,2%"
REM Trim leading space in hour if present
if "%HH:~0,1%"==" " set "HH=0%HH:~1,1%"
set "TS=%YY%%MM%%DD%_%HH%%MN%%SS%"

echo [INFO] Download root: "%DOWNLOAD_ROOT%"
echo [INFO] Logs root     : "%LOG_ROOT%"
echo [INFO] Timestamp     : %TS%
echo.

REM ==========================================
REM Main loop over inline URLs
REM Add/remove URLs in the list below (double-quoted entries)
REM ==========================================
for %%U in (
  "https://ov-share-05.iotg.sclab.intel.com/cv_bench_cache/WW50_llm-optimum_2025.4.1-20426-RC1/phi-3-mini-128k-instruct/pytorch/ov/OV_FP16-4BIT_DEFAULT/"
  "https://ov-share-05.iotg.sclab.intel.com/cv_bench_cache/WW51_llm-optimum_2026.0.0-20670/phi-3-mini-128k-instruct/pytorch/ov/OV_FP16-4BIT_DEFAULT/"
) do (
  set "URL=%%~U"

  REM ===== Sanitize a name for logs (replace forbidden filename chars) =====
  set "SAFE_NAME=!URL!"
  set "SAFE_NAME=!SAFE_NAME::=_!"
  set "SAFE_NAME=!SAFE_NAME:/=_!"
  set "SAFE_NAME=!SAFE_NAME:\=_!"
  set "SAFE_NAME=!SAFE_NAME:?=_!"
  set "SAFE_NAME=!SAFE_NAME:*=_!"
  set "SAFE_NAME=!SAFE_NAME:"=_!"
  set "SAFE_NAME=!SAFE_NAME:^<=_!"
  set "SAFE_NAME=!SAFE_NAME:^>=_!"
  set "SAFE_NAME=!SAFE_NAME:^|=_!"
  set "SAFE_NAME=!SAFE_NAME:^&=_!"
  set "SAFE_NAME=!SAFE_NAME: =_!"

  REM ===== Derive a tail folder from the last non-empty segment of the URL =====
  set "TAIL="
  for %%p in (!URL:/= !) do (
    set "TAIL=%%p"
  )
  if "!TAIL!"=="" set "TAIL=download"

  REM ===== Output folder and log file names =====
  set "OUT_DIR=%DOWNLOAD_ROOT%\!TAIL!"
  set "LOG_FILE=%LOG_ROOT%\wget_!SAFE_NAME!_%TS%.log"

  if not exist "!OUT_DIR!" mkdir "!OUT_DIR!"

  echo [INFO] Downloading:
  echo   URL      : !URL!
  echo   Out dir  : !OUT_DIR!
  echo   Log file : !LOG_FILE!
  echo.

  REM ===== WGET command (sequential) =====
  wget -r -l 0 ^
    --no-parent ^
    -nH --cut-dirs=1 ^
    --reject "index.html*" ^
    --no-check-certificate ^
    --directory-prefix="!OUT_DIR!" ^
    -o "!LOG_FILE!" ^
    "!URL!"

  if errorlevel 1 (
    echo [ERROR] wget failed for !URL!
  ) else (
    echo [DONE] !URL!
  )
  echo.
)

echo [ALL COMPLETED]
endlocal
exit /b 0
