#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
安全字符串处理模块
提供简单实用的字符串处理函数，避免编码问题
"""


def safe_str(text):
    """
    将任何内容转换为安全的字符串
    
    Parameters:
    -----------
    text : any
        任何需要转换的内容
        
    Returns:
    --------
    str : 安全的字符串
    """
    if text is None:
        return ""
    
    if isinstance(text, bytes):
        return text.decode('utf-8', errors='replace')
    
    try:
        return str(text)
    except:
        return repr(text)


def remove_special_chars(text):
    """
    移除可能导致编码问题的特殊字符
    
    Parameters:
    -----------
    text : str
        输入文本
        
    Returns:
    --------
    str : 清理后的文本
    """
    # 替换常见的特殊字符
    replacements = {
        '[OK]': '[OK]',
        '[FAIL]': '[FAIL]',
        '[WARN]': '[WARN]',
        '[TIP]': '[TIP]',
        '[DATA]': '[DATA]',
        '[START]': '[START]',
        '[TARGET]': '[TARGET]',
        '[DOC]': '[DOC]',
        '[TOOL]': '[TOOL]',
        '[v]': '[v]',
        '[x]': '[x]',
        '->': '->',
        '←': '<-',
        '↑': '^',
        '↓': 'v',
        '•': '*',
        '·': '.',
        '…': '...',
        '—': '-',
        '–': '-',
        '"': '"',
        '"': '"',
        ''': "'",
        ''': "'",
    }
    
    result = str(text)
    for old, new in replacements.items():
        result = result.replace(old, new)
    
    # 移除其他非ASCII字符
    result = ''.join(char if ord(char) < 128 else '?' for char in result)
    
    return result


def format_message(msg_type, message):
    """
    格式化消息
    
    Parameters:
    -----------
    msg_type : str
        消息类型 (success, error, warning, info)
    message : str
        消息内容
        
    Returns:
    --------
    str : 格式化后的消息
    """
    prefixes = {
        'success': '[成功]',
        'error': '[错误]',
        'warning': '[警告]',
        'info': '[信息]',
        'debug': '[调试]',
    }
    
    prefix = prefixes.get(msg_type, '[信息]')
    clean_message = remove_special_chars(safe_str(message))
    
    return f"{prefix} {clean_message}"


# 便捷函数
def success(message):
    """打印成功消息"""
    print(format_message('success', message))


def error(message):
    """打印错误消息"""
    print(format_message('error', message))


def warning(message):
    """打印警告消息"""
    print(format_message('warning', message))


def info(message):
    """打印信息消息"""
    print(format_message('info', message))


def section(title, width=60, char='='):
    """打印分节标题"""
    clean_title = remove_special_chars(safe_str(title))
    print(char * width)
    print(clean_title)
    print(char * width)


# 测试代码
if __name__ == "__main__":
    print("测试安全字符串处理...")
    
    # 测试各种消息
    success("操作成功 [OK]")
    error("操作失败 [FAIL]")
    warning("警告信息 [WARN]")
    info("提示信息 [TIP]")
    
    # 测试分节
    section("数据处理 [DATA]")
    
    # 测试特殊字符
    test_strings = [
        "正常文本",
        "包含emoji: [OK] [FAIL] [TIP] [DATA]",
        "包含特殊符号: -> ← ↑ ↓",
        "包含引号: "双引号" '单引号'",
        "包含省略号: 等等…",
    ]
    
    print("\n原始字符串 vs 清理后:")
    for s in test_strings:
        print(f"原始: {s}")
        print(f"清理: {remove_special_chars(s)}")
        print()