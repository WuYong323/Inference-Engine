:: 这个批处理文件是一个一键构建并测试 CUDA 扩展的便捷工具。它依次完成：
::      配置 Visual Studio 编译环境（x64）。
::      设置 CUDA 工具链路径。
::      切换到项目目录。
::      根据传入参数或默认值，运行对应的 Python 脚本（通常会调用 PyTorch 的 CUDA 扩展编译流程，并运行单元测试）。
:: 典型的应用场景是：
::      开发 PyTorch 自定义 CUDA 算子，每次修改完 .cu 或 .cpp 文件，
::      运行此脚本即可重新编译并验证功能，而无需手动设置环境变量或打开多个命令行窗口。




REM 关闭命令回显
@echo off

REM 调用 Visual Studio 开发者命令提示符
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64

REM 设置 CUDA 环境
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "PATH=%CUDA_HOME%\bin;%PATH%"

REM 切换当前工作目录
cd /d C:\Users\donk\PycharmProjects\PythonProject1

REM 用法: build_kernel.bat [脚本名.py]   不传参数默认跑 cuda_ext.py
if "%~1"=="" (set "SCRIPT=cuda_ext.py") else (set "SCRIPT=%~1")

REM 运行 Python 脚本
.venv\Scripts\python.exe %SCRIPT%
