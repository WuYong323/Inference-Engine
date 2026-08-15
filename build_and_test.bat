@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "PATH=%CUDA_HOME%\bin;%PATH%"
cd /d C:\Users\donk\PycharmProjects\PythonProject1
REM 用法: build_and_test.bat [脚本名.py]   不传参数默认跑 test_cuda_ext.py
if "%~1"=="" (set "SCRIPT=test_cuda_ext.py") else (set "SCRIPT=%~1")
.venv\Scripts\python.exe %SCRIPT%
