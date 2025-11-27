@echo off
echo ========================================
echo 多因子系统 - 每日数据更新
echo ========================================
echo.

REM 设置Python路径（根据实际情况修改）
set PYTHON_PATH=python

REM 运行每日更新
echo 正在执行每日数据更新...
%PYTHON_PATH% update_data.py --daily

if %ERRORLEVEL% EQU 0 (
    echo.
    echo [成功] 每日数据更新完成！
) else (
    echo.
    echo [错误] 数据更新失败，请查看日志文件
)

echo.
echo 按任意键退出...
pause >nul