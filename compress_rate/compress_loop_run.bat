:: ============================================================
:: LAMMPS batch runner
:: Params: 
::    loop_number values: 100、1000
:: 
:: Output folders:
::   100  -> compress_rate_0.01/
::   1000 -> compress_rate_0.001/
::
:: Autor: Shentongtong
:: Date: 2024-06-20
:: ============================================================



:: 关闭命令回显,会让终端更干净
@echo off
:: 开启变量延迟扩展,以便在循环内"动态"更改变量
setlocal enabledelayedexpansion  


:: 设置输入文件路径
set "input_file=compress_rate/in.compress_loop"


:: 循环运行命令, loop_rate值为0.01、0.001
for %%L in (0.001 0.0001) do (
    :: 1.设置保存路径 comcompress_rate_xx
    set "output_folder=compress_rate/compress_rate_%%L"
    set "log_file=compress_rate/compress_rate_%%L/running_rate_%%L.log"

    :: 2.如果存在则删除
    if exist "!output_folder!" (
        echo Folder !output_folder! already exists. Deleting it...
        rmdir /s /q "!output_folder!"
    )

    :: 3.重新创建文件夹
    mkdir "!output_folder!"

    :: 4.运行模拟
    echo Running with compress_rate %%L ...
    mpiexec -np 8 lmp -in %input_file% -var compress_rate %%L -log none > "!log_file!" 2>&1
    echo experiment %%L has completed.
)


echo All experiments runs completed.
endlocal