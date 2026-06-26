@echo off
setlocal
cd /d "%~dp0"
.venv\Scripts\python.exe gradio_app_sam2_ben2_route_a_for_Movie.py %*
endlocal
