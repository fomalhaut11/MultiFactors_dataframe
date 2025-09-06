@echo off
REM Windows 终端编码配置脚本
REM 解决Unicode编码问题

echo 正在配置Windows终端编码设置...

REM 设置控制台代码页为UTF-8
chcp 65001

REM 设置环境变量
set PYTHONIOENCODING=utf-8
set LANG=zh_CN.UTF-8
set PYTHONUTF8=1

REM 创建永久环境变量
setx PYTHONIOENCODING "utf-8"
setx LANG "zh_CN.UTF-8" 
setx PYTHONUTF8 "1"

echo.
echo 编码配置完成！
echo.
echo 重要提醒：
echo 1. 请重新启动您的IDE（VSCode/PyCharm等）
echo 2. 重新打开终端/命令提示符
echo 3. 如果使用PowerShell，请设置执行策略
echo.

pause