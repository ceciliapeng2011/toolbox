
@echo off
setlocal enabledelayedexpansion
set "count=0"
set "result="

for /f "usebackq tokens=6 delims=:" %%A in (`findstr /C:"Execute pagedattentionextension:PagedAttentionExtension_" log.temp`) do (
  for /f "tokens=1" %%B in ("%%~A") do (
    if !count! lss 48 (
      set /a count+=1
      if defined result (
        set "result=!result! %%B"
      ) else (
        set "result=%%B"
      )
    )
  )
)

echo %result%
endlocal
