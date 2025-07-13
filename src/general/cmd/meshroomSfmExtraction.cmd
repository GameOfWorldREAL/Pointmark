@echo off
rem ======== SETUP ========
set PROJECT_PATH=%1
set PROJECT_NAME=%2
set PG_CACHE_PATH=%3

rem === Meshroom setup ===:
set MROOM_PATH=%4
set PIPELINE=%5
set IMAGES_PATH=%PROJECT_PATH%\images

rem === run ===:
echo =======================================
echo Project: %PROJECT_NAME%
echo start Meshroom with parameters:
echo Meshroom path: %MROOM_PATH%
echo input images: %IMAGES_PATH%
echo cache: %PG_CACHE_PATH%
echo pipeline: %PIPELINE%\%PROJECT_NAME%.mg
echo =======================================

"%MROOM_PATH%\meshroom_batch" --pipeline "%PIPELINE%\%PROJECT_NAME%.mg" --input "%IMAGES_PATH%" --cache %PG_CACHE_PATH% --verbose "info"
