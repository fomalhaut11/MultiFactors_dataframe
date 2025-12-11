#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
复合盈利能力因子简化分析
绕过复杂的测试框架，直接进行基本的因子分析
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def load_factor_data():
    """加载复合因子数据"""
    try:
        factor_file = Path('E:/Documents/PythonProject/StockProject/StockData/factors/mixed/ComplexProfitability.pkl')
        factor_data = pd.read_pickle(factor_file)
        logger.info(f"因子数据加载成功: {factor_data.shape}")
        return factor_data
    except Exception as e:
        logger.error(f"因子数据加载失败: {e}")
        return None

def load_return_data():
    """加载收益率数据"""
    try:
        # 尝试加载收益率数据
        return_file = Path('E:/Documents/PythonProject/StockProject/StockData/Returns_C2C.pkl')
        if return_file.exists():
            return_data = pd.read_pickle(return_file)
            logger.info(f"收益率数据加载成功: {return_data.shape}")
            return return_data
        else:
            logger.warning("收益率数据文件不存在")
            return None
    except Exception as e:
        logger.error(f"收益率数据加载失败: {e}")
        return None

def basic_factor_analysis(factor_data, return_data=None):
    """基本因子分析"""
    try:
        logger.info("=" * 60)
        logger.info("复合盈利能力因子基本分析")
        logger.info("=" * 60)
        
        # 1. 数据基本统计
        logger.info("1. 数据基本统计")
        logger.info(f"   总数据点: {len(factor_data):,}")
        logger.info(f"   有效数据点: {factor_data.notna().sum():,}")
        logger.info(f"   有效率: {factor_data.notna().sum() / len(factor_data) * 100:.1f}%")
        
        # 2. 数值分布统计
        logger.info("\n2. 数值分布统计")
        desc_stats = factor_data.describe()
        for stat, value in desc_stats.items():
            if stat in ['mean', 'std', 'min', '25%', '50%', '75%', 'max']:
                logger.info(f"   {stat}: {value:,.4f}")
        
        # 3. 极端值分析
        logger.info("\n3. 极端值分析")
        q1 = factor_data.quantile(0.01)
        q99 = factor_data.quantile(0.99)
        extreme_count = ((factor_data < q1) | (factor_data > q99)).sum()
        logger.info(f"   1%-99%分位数范围: [{q1:.4f}, {q99:.4f}]")
        logger.info(f"   极端值数量 (1%和99%之外): {extreme_count:,}")
        logger.info(f"   极端值占比: {extreme_count / len(factor_data) * 100:.2f}%")
        
        # 4. 时间分布分析
        logger.info("\n4. 时间分布分析")
        if isinstance(factor_data.index, pd.MultiIndex):
            dates = factor_data.index.get_level_values(0).unique()
            stocks = factor_data.index.get_level_values(1).unique()
            logger.info(f"   时间范围: {dates.min()} 到 {dates.max()}")
            logger.info(f"   交易日数量: {len(dates):,}")
            logger.info(f"   股票数量: {len(stocks):,}")
            
            # 按年统计
            yearly_stats = factor_data.groupby(factor_data.index.get_level_values(0).year).agg(['count', 'mean', 'std'])
            logger.info(f"   年度统计:")
            for year in yearly_stats.index[-5:]:  # 显示最近5年
                count = yearly_stats.loc[year, 'count']
                mean = yearly_stats.loc[year, 'mean']
                std = yearly_stats.loc[year, 'std']
                logger.info(f"     {year}: 数量={count:,}, 均值={mean:.4f}, 标准差={std:.4f}")
        
        # 5. 截面分析（选择最近一个交易日）
        logger.info("\n5. 截面分析 (最近交易日)")
        if isinstance(factor_data.index, pd.MultiIndex):
            latest_date = factor_data.index.get_level_values(0).max()
            latest_cross_section = factor_data[factor_data.index.get_level_values(0) == latest_date]
            
            if len(latest_cross_section) > 0:
                logger.info(f"   最近日期: {latest_date}")
                logger.info(f"   股票数量: {len(latest_cross_section):,}")
                logger.info(f"   截面均值: {latest_cross_section.mean():.4f}")
                logger.info(f"   截面标准差: {latest_cross_section.std():.4f}")
                
                # 分位数分析
                logger.info(f"   分位数分布:")
                for pct in [10, 25, 50, 75, 90]:
                    value = latest_cross_section.quantile(pct/100)
                    logger.info(f"     {pct}%: {value:.4f}")
        
        # 6. 简单的IC分析（如果有收益率数据）
        if return_data is not None:
            logger.info("\n6. 简单IC分析")
            try:
                # 对齐数据
                aligned_data = pd.concat([factor_data, return_data], axis=1, join='inner')
                if not aligned_data.empty and len(aligned_data.columns) == 2:
                    correlation = aligned_data.iloc[:, 0].corr(aligned_data.iloc[:, 1])
                    logger.info(f"   因子与收益率相关系数: {correlation:.4f}")
                    
                    # 按日期计算IC
                    if isinstance(aligned_data.index, pd.MultiIndex):
                        daily_ic = aligned_data.groupby(level=0).apply(
                            lambda x: x.iloc[:, 0].corr(x.iloc[:, 1])
                        ).dropna()
                        
                        if not daily_ic.empty:
                            logger.info(f"   日均IC: {daily_ic.mean():.4f}")
                            logger.info(f"   IC标准差: {daily_ic.std():.4f}")
                            logger.info(f"   ICIR: {daily_ic.mean() / daily_ic.std():.4f}")
                            logger.info(f"   IC胜率: {(daily_ic > 0).mean():.2%}")
                else:
                    logger.warning("   数据对齐失败，无法计算IC")
            except Exception as e:
                logger.warning(f"   IC计算失败: {e}")
        
        logger.info("\n" + "=" * 60)
        logger.info("基本分析完成")
        
        return True
        
    except Exception as e:
        logger.error(f"基本因子分析失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_simple_report(factor_data):
    """生成简单的分析报告"""
    try:
        logger.info("生成简单分析报告...")
        
        # 创建报告目录
        report_dir = Path('reports')
        report_dir.mkdir(exist_ok=True)
        
        # 生成简单报告
        report_file = report_dir / 'ComplexProfitability_Simple_Analysis.txt'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("复合盈利能力因子简单分析报告\n")
            f.write("=" * 50 + "\n\n")
            
            # 基本统计
            f.write("基本统计信息:\n")
            f.write(f"  数据点总数: {len(factor_data):,}\n")
            f.write(f"  有效数据点: {factor_data.notna().sum():,}\n")
            f.write(f"  有效率: {factor_data.notna().sum() / len(factor_data) * 100:.1f}%\n\n")
            
            # 数值分布
            f.write("数值分布:\n")
            desc_stats = factor_data.describe()
            for stat, value in desc_stats.items():
                f.write(f"  {stat}: {value:,.4f}\n")
            
            # 时间信息
            if isinstance(factor_data.index, pd.MultiIndex):
                dates = factor_data.index.get_level_values(0).unique()
                stocks = factor_data.index.get_level_values(1).unique()
                f.write(f"\n时间范围: {dates.min()} 到 {dates.max()}\n")
                f.write(f"交易日数: {len(dates):,}\n")
                f.write(f"股票数: {len(stocks):,}\n")
            
            # 因子特征
            f.write(f"\n因子特征:\n")
            f.write(f"  因子名称: ComplexProfitability\n")
            f.write(f"  因子类型: 复合因子 (财务+技术+市场)\n")
            f.write(f"  计算公式: {{(TTM利润-TTM财务费用)-单季度存货}}/短期债务 / 5日收益率截面z-score\n")
            f.write(f"  数据频率: 日频\n")
            
            # 初步评价
            f.write(f"\n初步评价:\n")
            std_dev = factor_data.std()
            if std_dev > 1000:
                f.write(f"  因子值变化较大，标准差={std_dev:.0f}，可能需要进一步标准化处理\n")
            else:
                f.write(f"  因子值分布相对稳定，标准差={std_dev:.4f}\n")
                
            extreme_pct = ((factor_data < factor_data.quantile(0.01)) | (factor_data > factor_data.quantile(0.99))).mean()
            if extreme_pct > 0.05:
                f.write(f"  极端值比例较高({extreme_pct:.2%})，建议进行去极值处理\n")
            else:
                f.write(f"  极端值比例正常({extreme_pct:.2%})\n")
        
        logger.info(f"简单分析报告已保存到: {report_file}")
        return True
        
    except Exception as e:
        logger.error(f"生成简单报告失败: {e}")
        return False

def main():
    """主函数"""
    print("复合盈利能力因子简化分析")
    print("=" * 60)
    
    # 1. 加载因子数据
    factor_data = load_factor_data()
    if factor_data is None:
        print("因子数据加载失败！")
        return False
    
    # 2. 加载收益率数据（可选）
    return_data = load_return_data()
    
    # 3. 执行基本分析
    analysis_success = basic_factor_analysis(factor_data, return_data)
    
    # 4. 生成简单报告
    report_success = generate_simple_report(factor_data)
    
    if analysis_success and report_success:
        print("\n" + "=" * 60)
        print("复合盈利能力因子简化分析完成！")
        print("分析结果已保存到 reports/ 目录")
        print("=" * 60)
        return True
    else:
        print("\n" + "=" * 60)
        print("分析过程出现问题，请检查日志")
        print("=" * 60)
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)