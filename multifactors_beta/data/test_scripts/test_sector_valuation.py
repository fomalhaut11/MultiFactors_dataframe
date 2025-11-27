"""
测试行业板块估值计算器
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 导入板块估值处理器
from data.processor.sector_valuation_processor import SectorValuationProcessor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_sector_valuation():
    """测试板块估值计算"""

    print("=" * 60)
    print("测试行业板块估值计算器")
    print("=" * 60)

    # 创建处理器实例
    processor = SectorValuationProcessor()

    try:
        # 运行主处理流程
        print("\n开始计算板块估值指标...")
        sector_valuation = processor.process()

        print(f"\n✅ 计算完成!")
        print(f"数据形状: {sector_valuation.shape}")
        print(f"列名: {list(sector_valuation.columns)}")

        # 显示数据基本信息
        if not sector_valuation.empty:
            print(f"\n日期范围: {sector_valuation['TradingDate'].min()} 到 {sector_valuation['TradingDate'].max()}")
            print(f"行业数量: {sector_valuation['Sector'].nunique()}")
            print(f"记录总数: {len(sector_valuation)}")

            # 获取最新估值数据
            print("\n" + "=" * 40)
            print("最新板块估值（按市值排序前10）")
            print("=" * 40)
            latest_valuation = processor.get_latest_valuation()

            if not latest_valuation.empty:
                # 清理行业名称（去除前缀）
                latest_valuation['SectorName'] = latest_valuation['Sector'].str.replace('concept_name_', '').str.replace('(申万)', '').str.replace('(退市)', '')

                # 选择显示的列
                display_cols = ['SectorName', 'TotalMarketCap', 'StockCount', 'PE_TTM', 'PB', 'PS_TTM']
                available_cols = [col for col in display_cols if col in latest_valuation.columns]

                # 格式化输出
                display_df = latest_valuation[available_cols].head(10).copy()

                # 格式化市值（亿元）
                if 'TotalMarketCap' in display_df.columns:
                    display_df['TotalMarketCap'] = (display_df['TotalMarketCap'] / 1e8).round(2)
                    display_df.rename(columns={'TotalMarketCap': '总市值(亿元)'}, inplace=True)

                # 格式化估值指标
                for col in ['PE_TTM', 'PB', 'PS_TTM']:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].round(2)

                print(display_df.to_string(index=False))

            # 显示某个特定板块的历史估值
            print("\n" + "=" * 40)
            print("示例：查看某板块历史PE走势")
            print("=" * 40)

            # 找一个数据比较全的板块
            sector_counts = sector_valuation.groupby('Sector').size()
            if not sector_counts.empty:
                example_sector = sector_counts.idxmax()

                history = processor.get_sector_history(example_sector, 'PE_TTM')
                if not history.empty:
                    sector_name = example_sector.replace('concept_name_', '').replace('(申万)', '')
                    print(f"\n板块: {sector_name}")
                    print(f"历史数据点数: {len(history)}")

                    # 显示最近10个交易日
                    recent_history = history.tail(10).copy()
                    recent_history['PE_TTM'] = recent_history['PE_TTM'].round(2)
                    recent_history['TotalMarketCap'] = (recent_history['TotalMarketCap'] / 1e8).round(2)
                    recent_history.rename(columns={'TotalMarketCap': '总市值(亿元)'}, inplace=True)

                    print("\n最近10个交易日:")
                    print(recent_history.to_string(index=False))

            # 显示估值统计汇总
            print("\n" + "=" * 40)
            print("板块估值统计汇总")
            print("=" * 40)

            summary = processor.get_valuation_summary()
            if not summary.empty:
                # 显示PE分布
                pe_stats = summary['PE_TTM'].dropna()
                if len(pe_stats) > 0:
                    print(f"\nPE_TTM分布:")
                    print(f"  最小值: {pe_stats.min():.2f}")
                    print(f"  25分位: {pe_stats.quantile(0.25):.2f}")
                    print(f"  中位数: {pe_stats.median():.2f}")
                    print(f"  75分位: {pe_stats.quantile(0.75):.2f}")
                    print(f"  最大值: {pe_stats.max():.2f}")

                # 显示PB分布
                pb_stats = summary['PB'].dropna()
                if len(pb_stats) > 0:
                    print(f"\nPB分布:")
                    print(f"  最小值: {pb_stats.min():.2f}")
                    print(f"  25分位: {pb_stats.quantile(0.25):.2f}")
                    print(f"  中位数: {pb_stats.median():.2f}")
                    print(f"  75分位: {pb_stats.quantile(0.75):.2f}")
                    print(f"  最大值: {pb_stats.max():.2f}")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)

if __name__ == "__main__":
    test_sector_valuation()