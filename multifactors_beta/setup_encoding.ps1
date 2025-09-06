# PowerShell 编码配置脚本
# 解决Unicode编码问题

Write-Host "正在配置PowerShell编码设置..." -ForegroundColor Green

# 设置控制台输出编码为UTF-8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding = [System.Text.Encoding]::UTF8

# 设置PowerShell默认编码
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'
$PSDefaultParameterValues['*:Encoding'] = 'utf8'

# 设置环境变量
$env:PYTHONIOENCODING = "utf-8"
$env:LANG = "zh_CN.UTF-8"
$env:PYTHONUTF8 = "1"

# 创建永久环境变量
[Environment]::SetEnvironmentVariable("PYTHONIOENCODING", "utf-8", "User")
[Environment]::SetEnvironmentVariable("LANG", "zh_CN.UTF-8", "User")  
[Environment]::SetEnvironmentVariable("PYTHONUTF8", "1", "User")

Write-Host ""
Write-Host "编码配置完成！" -ForegroundColor Green
Write-Host ""
Write-Host "重要提醒：" -ForegroundColor Yellow
Write-Host "1. 请重新启动您的IDE（VSCode/PyCharm等）" -ForegroundColor Yellow
Write-Host "2. 重新打开PowerShell终端" -ForegroundColor Yellow
Write-Host "3. 建议将此脚本添加到PowerShell配置文件中" -ForegroundColor Yellow
Write-Host ""

# 询问是否添加到PowerShell配置文件
$addToProfile = Read-Host "是否将编码设置添加到PowerShell配置文件？(y/n)"
if ($addToProfile -eq 'y' -or $addToProfile -eq 'Y') {
    $profileContent = @"
# 自动编码配置
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding = [System.Text.Encoding]::UTF8
`$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'
`$PSDefaultParameterValues['*:Encoding'] = 'utf8'
`$env:PYTHONIOENCODING = "utf-8"
`$env:LANG = "zh_CN.UTF-8"
`$env:PYTHONUTF8 = "1"
"@

    if (!(Test-Path $PROFILE)) {
        New-Item -ItemType File -Path $PROFILE -Force
    }
    
    Add-Content -Path $PROFILE -Value $profileContent
    Write-Host "已添加到PowerShell配置文件: $PROFILE" -ForegroundColor Green
}

Read-Host "按任意键继续"