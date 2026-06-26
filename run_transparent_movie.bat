@echo off
setlocal
cd /d "%~dp0"
.venv\Scripts\python.exe gradio_app_sam2_transparent_BG_haystack_for_Movie.py %*
endlocal
