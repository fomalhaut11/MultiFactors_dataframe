"""
测试V3版本板块估值处理器
使用个股PE反推净利润的高效版本
"""

import sys
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.processor.sector_valuation_processor_v3 import SectorValuationProcessorV3

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

def test_v3():
    print("=" * 60)
    print("测试V3版本板块估值处理器")
    print("使用个股PE反推净利润")
    print("=" * 60)

    # 创建处理器
    processor = SectorValuationProcessorV3()

    # 运行计算（先测试30天）
    result = processor.process(date_range=30)

    if result.empty:
        print("\n⚠️ 未生成结果")
        return

    print(f"\n✅ 计算完成!")
    print(f"结果形状: {result.shape}")
    print(f"列名: {list(result.columns)}")

    # 显示估值指标统计
    if 'PE_TTM' in result.columns:
        pe_data = result['PE_TTM'].dropna()
        if not pe_data.empty:
            print(f"\nPE_TTM统计:")
            print(f"  有效记录: {len(pe_data)}/{len(result)}")
            print(f"  均值: {pe_data.mean():.2f}")
            print(f"  中位数: {pe_data.median():.2f}")
            print(f"  范围: {pe_data.min():.2f} - {pe_data.max():.2f}")

    if 'PB' in result.columns:
        pb_data = result['PB'].dropna()
        if not pb_data.empty:
            print(f"\nPB统计:")
            print(f"  有效记录: {len(pb_data)}/{len(result)}")
            print(f"  均值: {pb_data.mean():.2f}")
            print(f"  中位数: {pb_data.median():.2f}")
            print(f"  范围: {pb_data.min():.2f} - {pb_data.max():.2f}")

    # 显示最新日期的板块估值
    if not result.empty:
        latest_date = result['TradingDate'].max()
        latest_data = result[result['TradingDate'] == latest_date]

        print(f"\n最新日期 ({latest_date}) 的板块估值:")
        if 'PE_TTM' in latest_data.columns:
            # 筛选有PE数据的行业
            pe_sectors = latest_data[latest_data['PE_TTM'].notna()].copy()
            if not pe_sectors.empty:
                # 简化行业名称
                pe_sectors['SectorName'] = pe_sectors['Sector'].str.replace('concept_name_', '').str.replace('(申万)', '')
                pe_sectors = pe_sectors.sort_values('TotalMarketCap', ascending=False).head(10)

                print("\nTop10行业（按市值）:")
                for _, row in pe_sectors.iterrows():
                    print(f"  {row['SectorName']:12} - "
                          f"市值: {row['TotalMarketCap']/1e10:.0f}万亿, "
                          f"PE: {row['PE_TTM']:.1f}, "
                          f"股票数: {row['ValidStocks']}")

    # 检查输出文件
    output_path = Path("E:/Documents/PythonProject/StockProject/StockData/SectorData")
    if output_path.exists():
        files = list(output_path.glob("*v3*"))
        if files:
            print(f"\n输出文件:")
            for f in files:
                print(f"  - {f.name}: {f.stat().st_size / 1024:.1f}KB")

if __name__ == "__main__":
    test_v3()