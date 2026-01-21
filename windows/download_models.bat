@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM === Optional: clear proxies ===
set "https_proxy="
set "http_proxy="

REM === Base settings ===
set "GT_ROOT=https://ov-share-04.sclab.intel.com/cv_bench_cache/AC_llm/wwb_ref_gt_data_cache/2026.0.0-20769-87b915269ed_nat_ref/CPU_ICX/default_data_wwb_long_prompt/cache_nat_refs_cli___long_prompt/"

REM === Change your own download dir ===
set "OUT_DIR=WW02_llm-optimum_2026.0.0-20769"

REM === List of ref dirs (space-separated for cmd FOR) ===
set "REF_DIRS=qwen3-8b__NAT minicpm4-8b__NAT phi-3-mini-128k-instruct__NAT"

REM === Prepare output directory ===
if not exist "%OUT_DIR%" (
    mkdir "%OUT_DIR%"
)

REM === Detect downloader: prefer wget, fallback to curl ===
where /q wget
if not errorlevel 1 (
    set "DOWNLOADER=wget"
) else (
    where /q curl
    if not errorlevel 1 (
        set "DOWNLOADER=curl"
    ) else (
        echo ERROR: Neither 'wget' nor 'curl' found in PATH.
        echo Please install one of them or run this script where they are available.
        exit /b 1
    )
)

echo Using downloader: %DOWNLOADER%
echo GT_ROOT: %GT_ROOT%
echo.

REM === Loop over REF_DIRS and download reference.csv ===
for %%D in (%REF_DIRS%) do (
    set "ref_dir=%%D"
    REM Build URL and output paths using delayed expansion
    set "URL=!GT_ROOT!!ref_dir!/reference.csv"
    set "OUT_FILE_DIR=%OUT_DIR%\!ref_dir!"
  
    if not exist "!OUT_FILE_DIR!" (
        mkdir "!OUT_FILE_DIR!"
    )
    set "OUT_FILE=!OUT_FILE_DIR!\reference.csv"

    echo Downloading: !URL!
    if /i "%DOWNLOADER%"=="wget" (
        REM -q for quiet; --no-check-certificate if internal CA is not trusted
        wget -q -O "!OUT_FILE!" --no-check-certificate "!URL!"
        if errorlevel 1 (
            echo FAILED: wget could not download "!URL!"
        ) else (
            echo Saved to "!OUT_FILE!"
        )
    ) else (
        REM curl: -f to fail on HTTP errors; -L to follow redirects; -s for quiet; -S to show errors; -k to skip cert checks if needed
        REM Remove -k if your certificate chain is trusted.
        curl -f -L -s -S -k -o "!OUT_FILE!" "!URL!"
        if errorlevel 1 (
            echo FAILED: curl could not download "!URL!"
        ) else (
            echo Saved to "!OUT_FILE!"
        )
    )
    echo.
)

echo All done.
endlocal

REM ------------------------------------------------------------
REM Examples (kept as comments)
REM wget -r -l 0 ^
REM   --no-parent ^
REM   -nH --cut-dirs=1 ^
REM   --reject "index.html*" ^
REM   --no-check-certificate ^
REM   https://ov-share-13.iotg.sclab.intel.com/cv_bench_cache/WW02_llm-optimum_2026.0.0-20769/phi-3-mini-128k-instruct/pytorch/ov/OV_FP16-4BIT_DEFAULT/
REM ------------------------------------------------------------




@REM wget -r -l 0 ^
@REM   --no-parent ^
@REM   -nH --cut-dirs=1 ^
@REM   --reject "index.html*" ^
@REM   --no-check-certificate ^
@REM   https://ov-share-13.iotg.sclab.intel.com/cv_bench_cache/WW02_llm-optimum_2026.0.0-20769/phi-3-mini-128k-instruct/pytorch/ov/OV_FP16-4BIT_DEFAULT/

@REM wget -r -l 0 ^
@REM   --no-parent ^
@REM   -nH --cut-dirs=1 ^
@REM   --reject "index.html*" ^
@REM   --no-check-certificate ^
@REM   https://ov-share-13.iotg.sclab.intel.com/cv_bench_cache/WW02_llm-optimum_2026.0.0-20769/minicpm4-8b/pytorch/ov/OV_FP16-4BIT_DEFAULT/


@REM   wget -r -l 0 ^
@REM   --no-parent ^
@REM   -nH --cut-dirs=1 ^
@REM   --reject "index.html*" ^
@REM   --no-check-certificate ^
@REM   https://ov-share-13.iotg.sclab.intel.com/cv_bench_cache/WW02_llm-optimum_2026.0.0-20769/qwen3-8b/pytorch/ov/OV_FP16-4BIT_DEFAULT/


