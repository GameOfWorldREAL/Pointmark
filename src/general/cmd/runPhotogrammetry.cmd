@echo off
SETLOCAL EnableDelayedExpansion
rem iamges expected to be in PROJECT_PATH/images

rem ======== SETUP ========
set RECONST_PIPELINE=%1
set PROJECT_PATH=%2
for %%I in ("%PROJECT_PATH%") do set "PROJECT_NAME=%%~nxI"
set PG_CACHE_PATH=%3

rem ======== RUN MESHROOM ========
if %RECONST_PIPELINE%==1 (

    rem === Meshroom setup ===:
    set MROOM_PATH=%4
    set TEMPLATE_PATH=%5

    rem === run ===:
    set IMAGES_PATH=!PROJECT_PATH!\images
    set OUTPUT_PATH=!PROJECT_PATH!\Meshroom

    echo =======================================
    echo Project: !PROJECT_NAME!
    echo start Meshroom with parameters:
    echo Meshroom path: !MROOM_PATH!
    echo input images: !IMAGES_PATH!
    echo template: !TEMPLATE_PATH!
    echo output: !OUTPUT_PATH!
    echo =======================================

    copy !TEMPLATE_PATH! !OUTPUT_PATH!\!PROJECT_NAME!.mg
    "!MROOM_PATH!\meshroom_batch" --pipeline "!OUTPUT_PATH!\!PROJECT_NAME!.mg" --input "!IMAGES_PATH!" --cache !PG_CACHE_PATH! --verbose "error"
)

rem ======== RUN COLMAP ========
if %RECONST_PIPELINE%==2 (

    rem === Colmap setup ===:
    set COLMAP_PATH=%4

    rem === run ===:

    set IMAGES_PATH=%PROJECT_PATH%\images

    echo =======================================
    echo Project: !PROJECT_NAME!
    echo start Colmap with parameters:
    echo Colmap path: !COLMAP_PATH!
    echo input images: !IMAGES_PATH!
    echo output: !OUTPUT_PATH!
    echo =======================================

    %COLMAP_PATH%\colmap automatic_reconstructor --workspace_path !PG_CACHE_PATH! --image_path "!IMAGES_PATH!"
)