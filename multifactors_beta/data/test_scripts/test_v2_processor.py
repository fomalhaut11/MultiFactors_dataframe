"""
测试V2版本的板块估值处理器（带中间数据保存）
"""

import sys
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.processor.sector_valuation_processor_v2 import SectorValuationProcessorV2

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

def test_v2():
    print("=" * 60)
    print("测试V2板块估值处理器（带中间数据保存）")
    print("=" * 60)

    # 创建处理器，启用中间数据保存
    processor = SectorValuationProcessorV2(save_intermediate=True)

    # 第一次运行 - 会计算并保存中间数据
    print("\n第一次运行（计算并保存中间数据）...")
    result = processor.process(force_recalc=False, date_range=30)  # 只计算30天用于测试

    print(f"\n✅ 计算完成!")
    print(f"结果形状: {result.shape}")
    print(f"列名: {list(result.columns)}")

    # 检查是否有估值指标
    if 'PE_TTM' in result.columns:
        pe_data = result[result['PE_TTM'].notna()]
        print(f"\nPE_TTM数据: {len(pe_data)}条")

    if 'PB' in result.columns:
        pb_data = result[result['PB'].notna()]
        print(f"PB数据: {len(pb_data)}条")

    # 检查中间数据是否保存
    intermediate_path = Path("E:/Documents/PythonProject/StockProject/StockData/SectorData/intermediate")
    if intermediate_path.exists():
        files = list(intermediate_path.glob("*.pkl"))
        print(f"\n中间数据文件: {len(files)}个")
        for f in files:
            print(f"  - {f.name}: {f.stat().st_size / 1024 / 1024:.2f}MB")

    print("\n" + "=" * 60)

    # 第二次运行 - 应该直接加载中间数据
    print("\n第二次运行（加载已有中间数据）...")
    processor2 = SectorValuationProcessorV2(save_intermediate=True)
    result2 = processor2.process(force_recalc=False, date_range=30)

    print(f"\n✅ 第二次运行完成!")
    print(f"结果形状: {result2.shape}")

if __name__ == "__main__":
    test_v2()