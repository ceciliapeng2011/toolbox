
@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM User-provided lists (adapted from your base script)
REM ============================================================

set "MODEL_DIRS_0=%HOME%\WW02_llm-optimum_2026.0.0-20769\qwen3-8b\pytorch\ov\OV_FP16-4BIT_DEFAULT"
set "MODEL_DIRS_1=%HOME%\WW02_llm-optimum_2026.0.0-20769\minicpm4-8b\pytorch\ov\OV_FP16-4BIT_DEFAULT"
set "MODEL_DIRS_2=%HOME%\WW02_llm-optimum_2026.0.0-20769\phi-3-mini-128k-instruct\pytorch\ov\OV_FP16-4BIT_DEFAULT"
set "MODEL_DIRS_COUNT=3"

set "GT_DIR=%HOME%\WW02_llm-optimum_2026.0.0-20769"
set "GT_FILES_0=%GT_DIR%\qwen3-8b__NAT\reference.csv"
set "GT_FILES_1=%GT_DIR%\minicpm4-8b__NAT\reference.csv"
set "GT_FILES_2=%GT_DIR%\phi-3-mini-128k-instruct__NAT\reference.csv"
set "GT_FILES_COUNT=3"

set "MODEL_TYPE=text"

REM No spaces in JSON strings to keep cmd quoting simple
set "CB_CONFIGS_0={\"enable_prefix_caching\":false,\"max_num_batched_tokens\":4096,\"use_sparse_attention\":false}"
set "CB_CONFIGS_1={\"enable_prefix_caching\":false,\"max_num_batched_tokens\":4096,\"use_sparse_attention\":true,\"sparse_attention_config\":{\"mode\":\"XATTENTION\",\"xattention_threshold\":100,\"xattention_block_size\":128}}"
set "CB_CONFIGS_2={\"enable_prefix_caching\":false,\"max_num_batched_tokens\":4096,\"use_sparse_attention\":true,\"sparse_attention_config\":{\"mode\":\"XATTENTION\",\"xattention_threshold\":100,\"xattention_block_size\":256}}"
set "CB_CONFIGS_3={\"enable_prefix_caching\":false,\"max_num_batched_tokens\":4096,\"use_sparse_attention\":true,\"sparse_attention_config\":{\"mode\":\"XATTENTION\",\"xattention_threshold\":0.9,\"xattention_block_size\":128}}"
set "CB_CONFIGS_4={\"enable_prefix_caching\":false,\"max_num_batched_tokens\":4096,\"use_sparse_attention\":true,\"sparse_attention_config\":{\"mode\":\"XATTENTION\",\"xattention_threshold\":0.9,\"xattention_block_size\":256}}"
set "CB_CONFIGS_5={\"enable_prefix_caching\":false,\"max_num_batched_tokens\":4096,\"use_sparse_attention\":true,\"sparse_attention_config\":{\"mode\":\"XATTENTION\",\"xattention_threshold\":0.1,\"xattention_block_size\":128}}"
set "CB_CONFIGS_6={\"enable_prefix_caching\":false,\"max_num_batched_tokens\":4096,\"use_sparse_attention\":true,\"sparse_attention_config\":{\"mode\":\"XATTENTION\",\"xattention_threshold\":0.1,\"xattention_block_size\":256}}"
set "CB_CONFIGS_COUNT=7"

set "OV_CONFIGS_0={\"KV_CACHE_PRECISION\":\"i8\",\"KEY_CACHE_QUANT_MODE\":\"BY_TOKEN\"}"
set "OV_CONFIGS_1={\"KV_CACHE_PRECISION\":\"f16\"}"
set "OV_CONFIGS_COUNT=2"

REM ============================================================
REM Environment (as in your base script)
REM ============================================================
set "OPENVINO_LOG_LEVEL=4"

REM ============================================================
REM Output roots
REM ============================================================
set "OUTPUT_ROOT=.\wwb_outputs"
set "LOG_ROOT=.\wwb_logs"

if not exist "%OUTPUT_ROOT%" mkdir "%OUTPUT_ROOT%"
if not exist "%LOG_ROOT%" mkdir "%LOG_ROOT%"

REM Timestamp (yyyyMMdd_HHmmss)
for /f "usebackq delims=" %%T in (`powershell -NoProfile -Command "(Get-Date).ToString('yyyyMMdd_HHmmss')"`) do set "RUN_TS=%%T"

REM ============================================================
REM Helpers (PowerShell one-liners invoked from cmd)
REM ============================================================

REM Extract model name: segment immediately before 'pytorch' in a forward-slash path
set "PS_GET_MODEL=$p = [Console]::In.ReadLine(); $parts = $p -split '/'; $idx = [Array]::IndexOf($parts, 'pytorch'); if($idx -gt 0){ $parts[$idx-1] }"

REM Build a readable OV tag from JSON
set "PS_OV_TAG=$j = [Console]::In.ReadLine(); $o = $j | ConvertFrom-Json; $kv = $o.KV_CACHE_PRECISION; $kq = $o.KEY_CACHE_QUANT_MODE; $tag = 'ov_kv-' + $kv; if($kq){ $tag += '_kq-' + ($kq.ToLower()) }; $tag = $tag -replace '[^A-Za-z0-9._-]+','-'; $tag = $tag.Trim('-'); $tag"

REM Build a readable CB tag from JSON
set "PS_CB_TAG=$j = [Console]::In.ReadLine(); $o = $j | ConvertFrom-Json; $dense = -not ($o.use_sparse_attention); $mtoks = $o.max_num_batched_tokens; if($dense){ $tag = 'cb_dense' } else { $mode = $o.sparse_attention_config.mode; $thr = $o.sparse_attention_config.xattention_threshold; $bsz = $o.sparse_attention_config.xattention_block_size; if($mode){ $mode = $mode.ToLower() } else { $mode = 'unknown' }; $tag = 'cb_sparse-' + $mode + '_thr-' + $thr + '_bs-' + $bsz }; if($mtoks){ $tag += '_mtoks-' + $mtoks }; $tag = $tag -replace '[^A-Za-z0-9._-]+','-'; $tag = $tag.Trim('-'); $tag"

REM ============================================================
REM Main loop
REM ============================================================

set /a MODEL_MAX=MODEL_DIRS_COUNT-1
set /a GT_MAX=GT_FILES_COUNT-1
set /a OV_MAX=OV_CONFIGS_COUNT-1
set /a CB_MAX=CB_CONFIGS_COUNT-1

for /l %%I in (0,1,%MODEL_MAX%) do (
  set "model_dir=!MODEL_DIRS_%%I!"

  REM Derive model_name from model_dir
  for /f "usebackq delims=" %%M in (`echo !model_dir! ^| powershell -NoProfile -Command "!PS_GET_MODEL!"`) do set "model_name=%%M"

  if "!model_name!"=="" (
    echo [WARN] Skip: could not extract model name from !model_dir!
    echo.
    goto :continue_model
  )

  REM Find matching GT file that contains "/<model_name>__"
  set "gt_file="
  for /l %%G in (0,1,%GT_MAX%) do (
    set "candidate=!GT_FILES_%%G!"
    echo "!candidate!" | findstr /C:"/!model_name!__" >nul
    if !errorlevel! == 0 (
      set "gt_file=!candidate!"
    )
  )
  if "!gt_file!"=="" (
    echo [WARN] Skip: no GT file matched for model: !model_name!
    echo.
    goto :continue_model
  )

  REM Create model+timestamp roots
  set "model_root=%OUTPUT_ROOT%\!model_name!\%RUN_TS%"
  set "log_root=%LOG_ROOT%\!model_name!\%RUN_TS%"
  if not exist "!model_root!" mkdir "!model_root!"
  if not exist "!log_root!" mkdir "!log_root!"

  for /l %%O in (0,1,%OV_MAX%) do (
    set "ov_config=!OV_CONFIGS_%%O!"
    REM Build ov_tag via PowerShell
    for /f "usebackq delims=" %%T in (`echo !ov_config! ^| powershell -NoProfile -Command "!PS_OV_TAG!"`) do set "ov_tag=%%T"

    for /l %%C in (0,1,%CB_MAX%) do (
      set "cb_config=!CB_CONFIGS_%%C!"
      REM Build cb_tag via PowerShell
      for /f "usebackq delims=" %%U in (`echo !cb_config! ^| powershell -NoProfile -Command "!PS_CB_TAG!"`) do set "cb_tag=%%U"

      set "out_dir=!model_root!\!ov_tag!\!cb_tag!"
      if not exist "!out_dir!" mkdir "!out_dir!"

      set "out_log=!log_root!\wwb_!ov_tag!__!cb_tag!.log"

      echo ===========================================================
      echo Running:
      echo   model_dir = !model_dir!
      echo   model     = !model_name!
      echo   gt_file   = !gt_file!
      echo   ov_tag    = !ov_tag!
      echo   cb_tag    = !cb_tag!
      echo   output    = !out_dir!
      echo   log       = !out_log!
      echo -----------------------------------------------------------

      REM Save exact JSON configs for full traceability
      > "!out_dir!\ov_config.json"  echo(!ov_config!
      > "!out_dir!\cb_config.json"  echo(!cb_config!

      REM Execute (JSON args have no spaces; safe to pass unquoted)
      wwb ^
        --target-model "!model_dir!" ^
        --model-type "%MODEL_TYPE%" ^
        --long-prompt ^
        --genai ^
        --gt-data "!gt_file!" ^
        --device GPU ^
        --cb-config !cb_config! ^
        --ov-config !ov_config! ^
        --output "!out_dir!" ^
        > "!out_log!" 2>&1

      if errorlevel 1 (
        echo [ERR ] Failed: !model_name! | !ov_tag! | !cb_tag!
      ) else (
        echo [OK  ] Done: !model_name! | !ov_tag! | !cb_tag!
      )
      echo.
    )
  )

  :continue_model
)

echo [DONE] All runs completed.
echo   Outputs: %OUTPUT_ROOT%
echo   Logs   : %LOG_ROOT%

endlocal
