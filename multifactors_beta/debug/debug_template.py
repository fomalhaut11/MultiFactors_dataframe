#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试脚本模板
使用说明：复制此文件到temp目录并修改
"""
import sys
import os
from pathlib import Path
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 设置输出编码（Windows）
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def setup_output(script_name):
    """设置输出文件"""
    # 创建带时间戳的输出文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = project_root / "debug" / "results" / f"{script_name}_{timestamp}.txt"
    
    return output_file


def main():
    """主函数"""
    # 设置输出
    output_file = setup_output("debug_template")
    
    # 同时输出到控制台和文件
    with open(output_file, 'w', encoding='utf-8') as f:
        def log(msg):
            print(msg)
            f.write(msg + '\n')
            
        log("=" * 60)
        log(f"调试脚本运行时间：{datetime.now()}")
        log("=" * 60)
        
        try:
            # ========== 在这里添加你的调试代码 ==========
            log("\n1. 加载数据...")
            # 示例：加载数据
            # from factors.financial.fundamental_factors import BPFactor
            # bp_factor = BPFactor()
            
            log("\n2. 执行计算...")
            # 示例：执行计算
            
            log("\n3. 输出结果...")
            # 示例：输出结果
            
            # ========================================
            
            log("\n" + "=" * 60)
            log("调试完成！")
            
        except Exception as e:
            log(f"\n错误：{e}")
            import traceback
            log(traceback.format_exc())
    
    print(f"\n结果已保存至：{output_file}")


if __name__ == "__main__":
    main()