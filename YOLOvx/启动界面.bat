@echo off
cd /d "%~dp0"
echo Starting YOLOv11 Classroom Behavior Detection UI...
echo.
"C:/ProgramData/anaconda3/python.exe" ui_main.py
if errorlevel 1 (
echo.
echo Start failed. Check error message above.
pause
)
