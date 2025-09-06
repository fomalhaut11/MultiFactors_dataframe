#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
编码工具模块
解决Windows GBK编码兼容性问题，统一字符处理
"""

import sys
import locale
import os
import codecs
import io
from typing import Union

# 强制设置编码
def setup_encoding():
    """设置系统编码为UTF-8"""
    # 设置环境变量
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['LANG'] = 'zh_CN.UTF-8'
    
    # Windows系统特殊处理
    if sys.platform.startswith('win'):
        try:
            # 强制设置控制台代码页为UTF-8
            os.system('chcp 65001 >nul 2>&1')
            
            # 重新配置标准输出流
            if hasattr(sys.stdout, 'buffer'):
                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            if hasattr(sys.stderr, 'buffer'):    
                sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
                
        except Exception as e:
            print(f"警告: 编码设置失败 - {e}")
            
    # 设置默认编码
    if hasattr(sys, 'setdefaultencoding'):
        sys.setdefaultencoding('utf-8')
        
    # 设置locale
    try:
        if sys.platform.startswith('win'):
            locale.setlocale(locale.LC_ALL, 'Chinese_China.utf8')
        else:
            locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')
    except:
        try:
            locale.setlocale(locale.LC_ALL, '')
        except:
            pass

def clean_emoji_text(text: str) -> str:
    """清理文本中的emoji和特殊Unicode字符"""
    if not isinstance(text, str):
        return str(text)
    
    # emoji替换映射
    emoji_replacements = {
        '🎯': '[目标]',
        '📊': '[图表]',
        '📋': '[清单]',
        '⚡': '[快速]',
        '🎉': '[成功]',
        '✅': '[通过]',
        '❌': '[失败]',
        '⚠️': '[警告]',
        '🔬': '[测试]',
        '🧪': '[实验]',
        '🚀': '[启动]',
        '🧠': '[智能]',
        '🤖': '[AI]',
        '💡': '[提示]',
        '→': '->',
        '←': '<-',
        '↑': '^',
        '↓': 'v',
        '★': '*',
        '☆': '*'
    }
    
    # 替换已知emoji
    for emoji, replacement in emoji_replacements.items():
        text = text.replace(emoji, replacement)
    
    # 移除其他非GBK兼容字符
    try:
        # 尝试GBK编码，失败的字符用?替代
        text.encode('gbk')
        return text
    except UnicodeEncodeError:
        # 逐字符检查，替换不兼容字符
        clean_chars = []
        for char in text:
            try:
                char.encode('gbk')
                clean_chars.append(char)
            except UnicodeEncodeError:
                clean_chars.append('?')
        return ''.join(clean_chars)

def safe_print(*args, **kwargs):
    """安全的打印函数，自动清理emoji"""
    cleaned_args = []
    for arg in args:
        if isinstance(arg, str):
            cleaned_args.append(clean_emoji_text(arg))
        else:
            cleaned_args.append(arg)
    
    print(*cleaned_args, **kwargs)

def format_status(success: bool) -> str:
    """格式化状态文本，不使用emoji"""
    return "成功" if success else "失败"

def format_warning() -> str:
    """格式化警告文本"""
    return "警告"

def format_check_result(passed: bool) -> str:
    """格式化检查结果"""
    return "通过" if passed else "失败"

# 在模块导入时自动设置编码
setup_encoding()

if __name__ == "__main__":
    # 测试编码清理功能
    test_text = "[目标] 测试文本 [通过] 成功 [失败] 失败 [警告] 警告"
    safe_print(f"测试文本: {test_text}")
    
    # 测试状态格式化
    safe_print(f"成功状态: {format_status(True)}")
    safe_print(f"失败状态: {format_status(False)}")
    safe_print(f"警告文本: {format_warning()}")
    
    safe_print("编码工具模块测试完成")