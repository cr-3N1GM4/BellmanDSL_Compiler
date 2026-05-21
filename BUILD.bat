@echo off
REM ============================================================================
REM BUILD.bat — one-command build for MDP-DSL v3.0 Phase 3B on Windows.
REM Requires g++ on PATH (MSYS2, MinGW-w64, or w64devkit recommended).
REM Produces:
REM   bin\mdp_compiler.exe — main compiler with all phase flags
REM   bin\mdp_autopsy.exe  — failure-log backwards solver (Phase 3B)
REM ============================================================================

cd /d "%~dp0"

echo === MDP-DSL v3.0 Phase 3B build ===
echo Working directory: %CD%
echo.

REM ---- preflight ----
for %%F in (src\mdp_compiler.cpp src\mdp_autopsy.cpp ^
            src\visualizer.hpp src\visualizer.cpp ^
            src\html_output.hpp src\html_output.cpp) do (
    if not exist "%%F" (
        echo [ERROR] Missing %%F
        pause
        exit /b 1
    )
)

where g++ >nul 2>&1
if errorlevel 1 (
    echo [ERROR] g++ not found on PATH.
    echo         Install MSYS2 ^(https://msys2.org^) and run:
    echo           pacman -S mingw-w64-x86_64-gcc
    pause
    exit /b 1
)

if not exist bin mkdir bin

REM ---- build mdp_compiler ----
pushd src
g++ -std=c++17 -O2 -DMDP_VIZ_ENABLED -o ..\bin\mdp_compiler.exe mdp_compiler.cpp
if errorlevel 1 ( popd & echo [ERROR] mdp_compiler build failed & pause & exit /b 1 )
popd
echo [OK] Built: bin\mdp_compiler.exe

REM ---- build mdp_autopsy (Phase 3B) ----
pushd src
g++ -std=c++17 -O2 -o ..\bin\mdp_autopsy.exe mdp_autopsy.cpp
if errorlevel 1 ( popd & echo [ERROR] mdp_autopsy build failed & pause & exit /b 1 )
popd
echo [OK] Built: bin\mdp_autopsy.exe
echo.

echo Quick checks (should each show VERIFY PASS):
echo.
bin\mdp_compiler.exe examples\tiger.mdp     | findstr "VERIFY"
echo.
bin\mdp_compiler.exe examples\robot_nav.mdp | findstr "VERIFY"
echo.
bin\mdp_compiler.exe examples\portfolio.mdp | findstr "VERIFY"
echo.
echo Phase 3A autopsy smoke tests (each should show ERROR or WARN):
echo.
bin\mdp_compiler.exe examples\autopsy_demo\myopic_corridor.mdp --diagnose             2>&1 | findstr /R "ERROR/ WARN/"
bin\mdp_compiler.exe examples\autopsy_demo\dead_state_demo.mdp --diagnose=structural   2>&1 | findstr /R "ERROR/ WARN/"
bin\mdp_compiler.exe examples\autopsy_demo\broken_for_repair.mdp --repair "pi(Start) == Move" 2>&1 | findstr "Minimal fix"
echo.
echo Phase 3B mdp_autopsy smoke tests:
echo.
bin\mdp_autopsy.exe examples\autopsy_demo\medical_dose.mdp examples\autopsy_demo\medical_clean.log 2>&1 | findstr "MODEL_"
bin\mdp_autopsy.exe examples\autopsy_demo\medical_dose.mdp examples\autopsy_demo\medical_run.log   2>&1 | findstr "MODEL_"
bin\mdp_autopsy.exe examples\autopsy_demo\robot_model.mdp  examples\autopsy_demo\robot_run.log    2>&1 | findstr "MODEL_"

echo.
echo Try these next:
echo   bin\mdp_compiler.exe examples\portfolio.mdp --show-hmm
echo   bin\mdp_compiler.exe examples\autopsy_demo\broken_for_repair.mdp --repair "pi(Start) == Move"
echo   bin\mdp_autopsy.exe  examples\autopsy_demo\robot_model.mdp examples\autopsy_demo\robot_run.log
echo   python portfolio\portfolio_server.py             ^&^& visit http://localhost:8421
echo.
pause
