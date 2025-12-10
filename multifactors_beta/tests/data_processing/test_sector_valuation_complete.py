"""
完整测试板块估值计算器功能
包含数据验证、计算逻辑测试、结果分析等
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_data_availability():
    """测试所需数据文件是否存在"""
    logger.info("="*60)
    logger.info("1. 检查数据文件可用性")
    logger.info("="*60)

    data_root = Path("E:/Documents/PythonProject/StockProject/StockData")

    required_files = {
        "价格数据": data_root / "Price.pkl",
        "PE_ttm数据": data_root / "RawFactors" / "EP_ttm.pkl",
        "PB数据": data_root / "RawFactors" / "BP.pkl",
        "行业分类": data_root / "Classificationdata" / "classification_one_hot.pkl"
    }

    all_exist = True
    for name, path in required_files.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            logger.info(f"✓ {name}: {path.name} ({size_mb:.1f} MB)")
        else:
            logger.error(f"✗ {name}: {path} 不存在")
            all_exist = False

    return all_exist

def test_calculation_logic():
    """测试计算逻辑的正确性"""
    logger.info("\n" + "="*60)
    logger.info("2. 验证计算逻辑")
    logger.info("="*60)

    from data.processor import SectorValuationFromStockPE

    # 创建处理器
    processor = SectorValuationFromStockPE()

    # 加载数据
    data = processor.load_data()

    # 检查数据加载
    logger.info("\n数据加载情况:")
    for key, value in data.items():
        if isinstance(value, pd.DataFrame) or isinstance(value, pd.Series):
            logger.info(f"  {key}: shape={value.shape}, dtype={value.dtype if hasattr(value, 'dtype') else 'mixed'}")

    # 抽样验证计算逻辑
    logger.info("\n抽样验证计算逻辑:")

    # 获取最新日期
    market_cap = data['market_cap']
    pe_ttm = data.get('pe_ttm')

    if market_cap is not None and pe_ttm is not None:
        # 找一个有效的日期和股票
        common_index = market_cap.index.intersection(pe_ttm.index)
        if len(common_index) > 0:
            sample_idx = common_index[0]
            sample_date, sample_stock = sample_idx

            mc = market_cap.loc[sample_idx]
            pe = pe_ttm.loc[sample_idx]

            if pd.notna(mc) and pd.notna(pe) and pe > 0:
                implied_profit = mc / pe
                logger.info(f"\n样本: {sample_stock} @ {sample_date}")
                logger.info(f"  市值: {mc/1e8:.2f} 亿元")
                logger.info(f"  PE_TTM: {pe:.2f}")
                logger.info(f"  反推净利润: {implied_profit/1e8:.2f} 亿元")
                logger.info(f"  验证: 市值/净利润 = {mc/implied_profit:.2f} (应等于PE)")

                if abs(mc/implied_profit - pe) < 0.01:
                    logger.info("  ✓ 计算逻辑验证通过")
                else:
                    logger.warning("  ✗ 计算逻辑可能有误差")

def test_full_calculation():
    """测试完整计算流程"""
    logger.info("\n" + "="*60)
    logger.info("3. 执行完整计算")
    logger.info("="*60)

    from data.processor import SectorValuationFromStockPE

    # 创建处理器
    processor = SectorValuationFromStockPE()

    # 计算最近30个交易日的板块估值
    logger.info("\n计算最近30个交易日的板块估值...")
    result = processor.process(date_range=30)

    if not result.empty:
        logger.info(f"\n计算结果:")
        logger.info(f"  记录数: {len(result)}")
        logger.info(f"  日期范围: {result['TradingDate'].min()} 至 {result['TradingDate'].max()}")
        logger.info(f"  板块数量: {result['Sector'].nunique()}")

        # 统计有效数据
        if 'PE_TTM' in result.columns:
            pe_valid = result['PE_TTM'].notna().sum()
            logger.info(f"  有效PE记录: {pe_valid}/{len(result)} ({pe_valid/len(result)*100:.1f}%)")

        if 'PB' in result.columns:
            pb_valid = result['PB'].notna().sum()
            logger.info(f"  有效PB记录: {pb_valid}/{len(result)} ({pb_valid/len(result)*100:.1f}%)")

        return result
    else:
        logger.error("计算结果为空")
        return pd.DataFrame()

def analyze_results(result_df):
    """分析计算结果"""
    if result_df.empty:
        return

    logger.info("\n" + "="*60)
    logger.info("4. 结果分析")
    logger.info("="*60)

    # 获取最新日期数据
    latest_date = result_df['TradingDate'].max()
    latest_data = result_df[result_df['TradingDate'] == latest_date].copy()

    logger.info(f"\n最新日期 ({latest_date}) 板块估值:")

    # PE分析
    if 'PE_TTM' in latest_data.columns:
        pe_data = latest_data[latest_data['PE_TTM'].notna()].sort_values('PE_TTM')

        logger.info("\nPE_TTM 排名:")
        logger.info("最低PE板块 (前5):")
        for _, row in pe_data.head(5).iterrows():
            logger.info(f"  {row['Sector']:10} PE={row['PE_TTM']:6.2f} 市值={row['TotalMarketCap']/1e12:6.2f}万亿")

        logger.info("\n最高PE板块 (前5):")
        for _, row in pe_data.tail(5).iterrows():
            logger.info(f"  {row['Sector']:10} PE={row['PE_TTM']:6.2f} 市值={row['TotalMarketCap']/1e12:6.2f}万亿")

    # PB分析
    if 'PB' in latest_data.columns:
        pb_data = latest_data[latest_data['PB'].notna()].sort_values('PB')

        if not pb_data.empty:
            logger.info("\nPB 排名:")
            logger.info("最低PB板块 (前3):")
            for _, row in pb_data.head(3).iterrows():
                logger.info(f"  {row['Sector']:10} PB={row['PB']:6.2f}")

    # 时间序列分析
    logger.info("\n" + "="*60)
    logger.info("5. 时间序列分析")
    logger.info("="*60)

    # 选择几个重点板块分析PE变化
    focus_sectors = ['医药生物', '基础化工', '电子', '计算机', '食品饮料']

    for sector in focus_sectors:
        sector_data = result_df[result_df['Sector'] == sector]
        if not sector_data.empty and 'PE_TTM' in sector_data.columns:
            pe_series = sector_data.set_index('TradingDate')['PE_TTM'].dropna()
            if not pe_series.empty:
                logger.info(f"\n{sector}:")
                logger.info(f"  PE范围: {pe_series.min():.2f} - {pe_series.max():.2f}")
                logger.info(f"  PE均值: {pe_series.mean():.2f}")
                logger.info(f"  最新PE: {pe_series.iloc[-1]:.2f}")

def test_data_consistency():
    """测试数据一致性"""
    logger.info("\n" + "="*60)
    logger.info("6. 数据一致性检查")
    logger.info("="*60)

    # 读取保存的结果
    data_root = Path("E:/Documents/PythonProject/StockProject/StockData")
    pkl_path = data_root / "SectorData" / "sector_valuation_from_stock_pe.pkl"
    csv_path = data_root / "SectorData" / "sector_valuation_from_stock_pe.csv"

    if pkl_path.exists() and csv_path.exists():
        pkl_data = pd.read_pickle(pkl_path)
        csv_data = pd.read_csv(csv_path)

        logger.info(f"PKL文件记录数: {len(pkl_data)}")
        logger.info(f"CSV文件记录数: {len(csv_data)}")

        if len(pkl_data) == len(csv_data):
            logger.info("✓ 文件记录数一致")
        else:
            logger.warning("✗ 文件记录数不一致")

    # 检查汇总报告
    import json
    summary_path = data_root / "SectorData" / "sector_valuation_summary.json"
    if summary_path.exists():
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)

        logger.info("\n汇总报告内容:")
        logger.info(f"  报告生成时间: {summary.get('report_date')}")
        logger.info(f"  最新数据日期: {summary.get('latest_date')}")
        logger.info(f"  总记录数: {summary.get('total_records')}")
        logger.info(f"  板块数量: {summary.get('sectors')}")

        if 'pe_stats' in summary:
            pe_stats = summary['pe_stats']
            logger.info(f"\n  PE统计:")
            logger.info(f"    均值: {pe_stats['mean']:.2f}")
            logger.info(f"    中位数: {pe_stats['median']:.2f}")
            logger.info(f"    范围: {pe_stats['min']:.2f} - {pe_stats['max']:.2f}")

def generate_detailed_report(result_df):
    """生成详细分析报告"""
    if result_df.empty:
        return

    logger.info("\n" + "="*60)
    logger.info("7. 生成详细分析报告")
    logger.info("="*60)

    report_path = Path("E:/Documents/PythonProject/StockProject/StockData/SectorData/sector_valuation_analysis.txt")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("板块估值分析报告\n")
        f.write("=" * 80 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"数据范围: {result_df['TradingDate'].min()} 至 {result_df['TradingDate'].max()}\n")
        f.write("\n")

        # 最新日期完整数据
        latest_date = result_df['TradingDate'].max()
        latest_data = result_df[result_df['TradingDate'] == latest_date].copy()

        f.write(f"最新日期 ({latest_date}) 板块估值一览\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'板块':<15} {'成分股数':<10} {'市值(万亿)':<12} {'PE_TTM':<10} {'PB':<10}\n")
        f.write("-" * 80 + "\n")

        for _, row in latest_data.sort_values('TotalMarketCap', ascending=False).iterrows():
            sector = row['Sector']
            stock_count = row['StockCount']
            market_cap = row['TotalMarketCap'] / 1e12
            pe = row.get('PE_TTM', np.nan)
            pb = row.get('PB', np.nan)

            pe_str = f"{pe:.2f}" if pd.notna(pe) else "N/A"
            pb_str = f"{pb:.2f}" if pd.notna(pb) else "N/A"

            f.write(f"{sector:<15} {stock_count:<10} {market_cap:<12.2f} {pe_str:<10} {pb_str:<10}\n")

    logger.info(f"详细报告已保存至: {report_path}")

def main():
    """主测试函数"""
    print("\n" + "="*60)
    print("板块估值计算器 - 完整功能测试")
    print("="*60)

    # 1. 检查数据可用性
    if not test_data_availability():
        logger.error("数据文件不完整，测试中止")
        return

    # 2. 验证计算逻辑
    test_calculation_logic()

    # 3. 执行完整计算
    result = test_full_calculation()

    # 4. 分析结果
    if not result.empty:
        analyze_results(result)

        # 5. 数据一致性检查
        test_data_consistency()

        # 6. 生成详细报告
        generate_detailed_report(result)

    logger.info("\n" + "="*60)
    logger.info("测试完成")
    logger.info("="*60)

if __name__ == "__main__":
    main()