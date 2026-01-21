
@echo off
:: Run as Administrator
echo === Applying GraphicsDrivers MemoryManager Registry Setting ===

:: 1. Backup current key
echo Backing up registry key...
reg export "HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers\MemoryManager" "%~dp0GraphicsDrivers_Backup.reg"
echo Backup saved to %~dp0GraphicsDrivers_Backup.reg

:: 2. Apply new value
echo Applying new value: SystemPartitionCommitLimitPercentage = 66
reg add "HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers\MemoryManager" /v SystemPartitionCommitLimitPercentage /t REG_DWORD /d 66 /f

:: 3. Verify
echo Verifying the change...
reg query "HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers\MemoryManager" /v SystemPartitionCommitLimitPercentage

echo === Done! Please restart your computer for changes to take effect. ===
pause
