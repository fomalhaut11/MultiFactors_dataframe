#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
编码问题修复模块
解决Windows环境下的UTF-8编码问题
"""
import sys
import io
import os
import locale


def fix_encoding():
    """
    修复Windows环境下的编码问题
    
    这个函数应该在程序启动时调用，可以解决：
    1. print中文乱码问题
    2. emoji字符无法显示问题
    3. 文件读写编码问题
    """
    # 1. 设置环境变量
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    # 2. 修复标准输出编码 - 安全版本
    if sys.platform == 'win32':
        # 检查是否已经被修改过，避免重复包装
        if not hasattr(sys.stdout, '_original_buffer_wrapped'):
            try:
                # 保存原始状态标记
                original_stdout = sys.stdout
                if hasattr(original_stdout, 'buffer') and hasattr(original_stdout.buffer, 'raw'):
                    new_stdout = io.TextIOWrapper(
                        original_stdout.buffer, 
                        encoding='utf-8', 
                        errors='replace',
                        line_buffering=True
                    )
                    new_stdout._original_buffer_wrapped = True
                    sys.stdout = new_stdout
            except Exception:
                # 如果出现任何问题，保持原状态
                pass
                
        if not hasattr(sys.stderr, '_original_buffer_wrapped'):
            try:
                # 保存原始状态标记
                original_stderr = sys.stderr
                if hasattr(original_stderr, 'buffer') and hasattr(original_stderr.buffer, 'raw'):
                    new_stderr = io.TextIOWrapper(
                        original_stderr.buffer,
                        encoding='utf-8',
                        errors='replace',
                        line_buffering=True
                    )
                    new_stderr._original_buffer_wrapped = True
                    sys.stderr = new_stderr
            except Exception:
                # 如果出现任何问题，保持原状态
                pass
    
    # 3. 设置默认编码
    if hasattr(sys, 'setdefaultencoding'):
        sys.setdefaultencoding('utf-8')
    
    # 4. 尝试设置控制台代码页为UTF-8
    if sys.platform == 'win32':
        try:
            import subprocess
            # 设置控制台代码页为65001 (UTF-8)
            subprocess.run(['chcp', '65001'], shell=True, capture_output=True)
        except:
            pass
    
    # 5. 设置locale
    try:
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except:
        try:
            locale.setlocale(locale.LC_ALL, 'C.UTF-8')
        except:
            pass


def safe_print(*args, **kwargs):
    """
    安全的print函数，自动处理编码问题
    
    使用方法：
        from core.utils.encoding_fix import safe_print
        safe_print("中文内容", "[OK] emoji内容")
    """
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        # 如果直接print失败，尝试编码后输出
        encoded_args = []
        for arg in args:
            if isinstance(arg, str):
                # 替换无法编码的字符
                encoded_args.append(arg.encode('utf-8', errors='replace').decode('utf-8', errors='replace'))
            else:
                encoded_args.append(str(arg))
        print(*encoded_args, **kwargs)


def remove_emojis(text):
    """
    移除文本中的emoji字符
    
    Parameters:
    -----------
    text : str
        包含emoji的文本
        
    Returns:
    --------
    str : 移除emoji后的文本
    """
    import re
    
    # emoji的Unicode范围
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF"  # 补充符号
        "\U00002600-\U000027BF"  # 杂项符号
        "\U0001F004-\U0001F0CF"  # 麻将牌等
        "\U00002B50"             # 星星
        "\U00002705"             # 白色重勾号
        "\U0001F4A1"             # 灯泡
        "\U0001F4CA"             # 条形图
        "\U0001F680"             # 火箭
        "\U0001F3AF"             # 靶心
        "]+", flags=re.UNICODE
    )
    
    return emoji_pattern.sub('', text)


def replace_emojis(text, replacements=None):
    """
    替换文本中的emoji为文字描述
    
    Parameters:
    -----------
    text : str
        包含emoji的文本
    replacements : dict, optional
        emoji到文字的映射字典
        
    Returns:
    --------
    str : 替换后的文本
    """
    if replacements is None:
        replacements = {
            '[OK]': '[完成]',
            '[FAIL]': '[错误]',
            '[WARN]': '[警告]',
            '[TIP]': '[提示]',
            '[DATA]': '[图表]',
            '[START]': '[启动]',
            '[TARGET]': '[目标]',
            '[DOC]': '[文档]',
            '[TOOL]': '[工具]',
            '[UP]': '[上升]',
            '[DOWN]': '[下降]',
            '[v]': '[对]',
            '[x]': '[错]',
            '->': '->',
            '←': '<-',
            '↑': '^',
            '↓': 'v',
        }
    
    result = text
    for emoji, replacement in replacements.items():
        result = result.replace(emoji, replacement)
    
    # 移除其他未映射的emoji
    result = remove_emojis(result)
    
    return result


class EncodingSafePrinter:
    """
    编码安全的打印器类
    
    使用方法：
        printer = EncodingSafePrinter()
        printer.print("包含emoji的内容 [OK]")
        printer.success("操作成功")
        printer.error("操作失败")
        printer.warning("警告信息")
        printer.info("提示信息")
    """
    
    def __init__(self, remove_emoji=True, fix_encoding_on_init=True):
        self.remove_emoji = remove_emoji
        if fix_encoding_on_init:
            fix_encoding()
    
    def _process_text(self, text):
        """处理文本，移除或替换emoji"""
        if self.remove_emoji:
            return replace_emojis(str(text))
        return str(text)
    
    def print(self, *args, **kwargs):
        """安全打印"""
        processed_args = [self._process_text(arg) for arg in args]
        safe_print(*processed_args, **kwargs)
    
    def success(self, message):
        """打印成功消息"""
        self.print(f"[成功] {message}")
    
    def error(self, message):
        """打印错误消息"""
        self.print(f"[错误] {message}")
    
    def warning(self, message):
        """打印警告消息"""
        self.print(f"[警告] {message}")
    
    def info(self, message):
        """打印信息消息"""
        self.print(f"[信息] {message}")
    
    def section(self, title, width=60, char='='):
        """打印分节标题"""
        self.print(char * width)
        self.print(title)
        self.print(char * width)


# 创建全局打印器实例
printer = EncodingSafePrinter()


# 在模块导入时自动修复编码
fix_encoding()


if __name__ == "__main__":
    # 测试代码
    print("测试编码修复功能...")
    
    # 测试中文
    print("测试中文：你好世界")
    
    # 测试emoji
    test_texts = [
        "[OK] 完成",
        "[FAIL] 错误", 
        "[TIP] 提示",
        "[DATA] 数据分析",
        "[START] 项目启动"
    ]
    
    print("\n原始print测试:")
    for text in test_texts:
        try:
            print(text)
        except UnicodeEncodeError as e:
            print(f"编码错误: {e}")
    
    print("\n使用safe_print:")
    for text in test_texts:
        safe_print(text)
    
    print("\n使用EncodingSafePrinter:")
    printer = EncodingSafePrinter()
    for text in test_texts:
        printer.print(text)
    
    print("\n测试工具方法:")
    printer.success("操作成功")
    printer.error("操作失败")
    printer.warning("这是警告")
    printer.info("这是提示")
    
    printer.section("测试分节标题")