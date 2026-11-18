@echo off
REM ===== Laptop 2: evaluator web UI =====
REM Usage:  deploy\run_ui.bat 192.168.1.23      (Laptop 1's IPv4 address)
cd /d "%~dp0.."
set API_IP=%1
if "%API_IP%"=="" set /p API_IP=Enter Laptop 1 IP address:
start "" "http://localhost:5500/?api=http://%API_IP%:8000"
python -m http.server 5500 --directory ui\web
