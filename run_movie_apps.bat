@echo off
setlocal
cd /d "%~dp0"

echo Starting RouteA movie app on port 7862...
start "RouteA Movie App" cmd /k ".venv\Scripts\python.exe gradio_app_sam2_ben2_route_a_for_Movie.py"

echo Starting transparent-background movie app...
start "Transparent BG Movie App" cmd /k ".venv\Scripts\python.exe gradio_app_sam2_transparent_BG_haystack_for_Movie.py"

echo.
echo Both launchers were started in separate windows.
endlocal
