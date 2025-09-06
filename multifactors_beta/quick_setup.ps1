[Environment]::SetEnvironmentVariable("PYTHONIOENCODING", "utf-8", "User")
[Environment]::SetEnvironmentVariable("LANG", "zh_CN.UTF-8", "User")  
[Environment]::SetEnvironmentVariable("PYTHONUTF8", "1", "User")
Write-Host "环境变量已永久配置完成！"