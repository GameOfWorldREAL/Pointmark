@echo off
SETLOCAL EnableDelayedExpansion
set KEYFRAMER_PATH=%1
set VIDEO_PATH=%2
set PROJECT_PATH=%3
set KEYFRAMER=%4

for %%I in ("%VIDEO_PATH%") do set "VIDEO_NAME=%%~nxI"

echo VIDEO_PATH: %VIDEO_PATH%
echo PROJECT_PATH: %PROJECT_PATH%
echo KEYFRAMER: %KEYFRAMER%

if %KEYFRAMER%==1 (
    copy "!VIDEO_PATH!" "!KEYFRAMER_PATH!\jkeyframer\videos\"
    echo run jkeyframer:
    cd "!KEYFRAMER_PATH!\jkeyframer"
    call "process_jkeyframer.bat"
    del ".\videos\!VIDEO_NAME!"
    move ".\keyframes\*.*" "!PROJECT_PATH!\images" >nul
    rmdir /s /q ".\extracted"
    rmdir /s /q ".\keyframes"
    rmdir /s /q ".\logs"
    cd "..\..\src"
)