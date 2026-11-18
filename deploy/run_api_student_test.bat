@echo off
REM ===== Laptop 1: serve the 10 student_test trips (rehearsal) =====
cd /d "%~dp0.."
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
set PATHSNAP_TRIPS_ROOT=data\raw\Student folder
set PATHSNAP_SPLIT=student_test
echo.
echo ===== Laptop 1 IP address - type the Wi-Fi one on Laptop 2 =====
powershell -NoProfile -Command "Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.254.*' } | ForEach-Object { '  {0,-25} {1}' -f $_.InterfaceAlias, $_.IPAddress }"
echo   (ignore 192.168.56.x - that is the VirtualBox adapter)
echo.
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
