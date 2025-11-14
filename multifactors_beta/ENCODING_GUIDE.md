# 编码问题解决指南

> ⚠️ **文档说明：简化版本**
>
> 本文档是快速参考版本，完整解决方案请参考：
> - [编码问题完整解决方案](ENCODING_SOLUTION.md) - 详细技术方案
>
> 最后更新：2025-09-07

## 问题现象
- Windows系统下Python输出中文乱码
- 文件读写编码错误
- Unicode字符显示异常

## 根本原因
Windows系统默认使用GBK编码，而项目使用UTF-8编码，导致编码不匹配。

## 最佳解决方案

### 1. 环境变量设置（推荐）

**永久解决方案**：设置系统环境变量
```bash
# 运行 setup_encoding_env.bat 
# 或手动设置以下环境变量：
PYTHONIOENCODING=utf-8
PYTHONUTF8=1
LANG=zh_CN.UTF-8
```

**临时解决方案**：在命令行中设置
```bash
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
python your_script.py
```

### 2. IDE配置
- **PyCharm**: File → Settings → Editor → File Encodings → UTF-8
- **VSCode**: 设置编码为UTF-8
- **Jupyter**: 确保notebook编码为UTF-8

### 3. 使用更好的终端
- 使用 **Windows Terminal** 而不是 cmd
- 或使用 **PowerShell**
- 配置终端默认编码为UTF-8

## 项目级简单修复

项目已在 `core/__init__.py` 中添加了最简单的编码修复：
```python
import os
import sys
if sys.platform.startswith('win') and os.environ.get('PYTHONIOENCODING') != 'utf-8':
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
```

## 检测编码环境

运行 `python check_encoding_env.py` 检查当前编码设置。

## 注意事项

1. **不要在代码中强制修改sys.stdout** - 这会导致其他问题
2. **环境变量设置后需要重启命令行/IDE** 才能生效  
3. **优先使用环境解决方案** 而不是代码修复
4. **避免在代码中输出调试信息** 污染正常输出

## 验证方法

设置环境变量后，运行：
```python
import sys
print(f"标准输出编码: {sys.stdout.encoding}")  # 应该是 utf-8
print("测试中文: 你好世界")  # 应该正常显示
```

## 如果问题仍然存在

1. 检查IDE/终端编码设置
2. 确认环境变量已正确设置并生效
3. 重启Python进程和IDE
4. 使用Windows Terminal替代cmd