@echo off
REM ===== Laptop 1, run ONCE as Administrator: allow other laptops to reach the API on port 8000 =====
netsh advfirewall firewall add rule name="PathSnap API 8000" dir=in action=allow protocol=TCP localport=8000
pause
