"""
快速测试板块估值计算是否能生成PE、PB等指标
"""

import sys
import pandas as pd
from pathlib import Path
import logging

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.processor.sector_valuation_processor import SectorValuationProcessor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def quick_test():
    print("=" * 60)
    print("快速测试板块PE、PB计算")
    print("=" * 60)

    # 创建处理器
    processor = SectorValuationProcessor()

    # 运行计算
    print("\n开始计算...")
    result = processor.process()

    print(f"\n✅ 计算完成!")
    print(f"结果形状: {result.shape}")
    print(f"列名: {list(result.columns)}")

    # 检查是否有PE、PB等列
    valuation_cols = [col for col in result.columns if col in ['PE_TTM', 'PB', 'PS_TTM', 'PCF_TTM']]
    if valuation_cols:
        print(f"\n✅ 成功计算的估值指标: {valuation_cols}")

        # 显示部分数据
        sample = result[result['PE_TTM'].notna() if 'PE_TTM' in result.columns else result.index].head(5)
        print(f"\n样本数据:")
        for col in ['Sector', 'TotalMarketCap', 'PE_TTM', 'PB']:
            if col in sample.columns:
                print(f"{col}: {sample[col].values[:3]}")
    else:
        print(f"\n⚠️ 未找到估值指标列，只有: {list(result.columns)}")

    # 保存路径
    print(f"\n数据已保存至: E:\\Documents\\PythonProject\\StockProject\\StockData\\SectorData\\")

if __name__ == "__main__":
    quick_test()