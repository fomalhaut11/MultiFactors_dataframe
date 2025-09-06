#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的编码解决方案
只包含必要的编码修复功能，没有自动执行和调试输出
"""

import sys
import os
import io
import locale


def fix_windows_encoding():
    """
    修复Windows下的编码问题
    只在需要时手动调用，不自动执行，不产生任何输出
    """
    if sys.platform != 'win32':
        return  # 非Windows系统不需要修复
    
    # 1. 设置环境变量
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    
    # 2. 设置控制台代码页（静默）
    try:
        os.system('chcp 65001 >nul 2>&1')
    except:
        pass
    
    # 3. 重新包装标准流（只在必要时）
    if hasattr(sys.stdout, 'buffer') and not hasattr(sys.stdout, '_encoding_fixed'):
        try:
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer, 
                encoding='utf-8', 
                errors='replace',
                line_buffering=True
            )
            sys.stdout._encoding_fixed = True
        except:
            pass
    
    if hasattr(sys.stderr, 'buffer') and not hasattr(sys.stderr, '_encoding_fixed'):
        try:
            sys.stderr = io.TextIOWrapper(
                sys.stderr.buffer, 
                encoding='utf-8', 
                errors='replace',
                line_buffering=True
            )
            sys.stderr._encoding_fixed = True
        except:
            pass


def safe_print(*args, **kwargs):
    """
    安全的print函数，处理编码问题
    """
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        # 替换无法编码的字符
        safe_args = []
        for arg in args:
            if isinstance(arg, str):
                safe_args.append(arg.encode('utf-8', errors='replace').decode('utf-8'))
            else:
                safe_args.append(str(arg))
        print(*safe_args, **kwargs)


def clean_emoji_text(text):
    """
    清理文本中的emoji，替换为文字描述
    """
    if not isinstance(text, str):
        text = str(text)
    
    # 简单的emoji替换映射
    replacements = {
        '✓': '[成功]',
        '✗': '[失败]', 
        '⚠': '[警告]',
        '📊': '[图表]',
        '🎯': '[目标]',
        '📝': '[文档]',
        '🔧': '[工具]',
        '🚀': '[启动]',
    }
    
    for emoji, replacement in replacements.items():
        text = text.replace(emoji, replacement)
    
    # 移除其他可能的特殊字符
    import re
    text = re.sub(r'[^\u0000-\u007F\u4e00-\u9fff]', '', text)
    
    return text


# 简单的全局修复函数，只在明确调用时执行
def initialize_encoding():
    """
    初始化编码设置
    只在程序启动时手动调用一次
    """
    fix_windows_encoding()


# 不自动执行任何操作
if __name__ == "__main__":
    # 测试功能
    initialize_encoding()
    print("编码修复功能测试")
    safe_print("测试中文：你好世界")
    safe_print("测试emoji：✓ 成功")
    print(clean_emoji_text("测试清理：✓ 成功 ✗ 失败 📊 图表"))