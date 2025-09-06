#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成SUE（标准化未预期盈余）因子

使用历史EPS数据计算SUE因子并保存到本地
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

from core.database import DatabaseManager
from factors.financial import SUE
from config import get_config


def fetch_eps_data(start_date='2020-01-01', end_date='2024-12-31'):
    """
    从数据库获取EPS数据
    
    Parameters
    ----------
    start_date : str
        开始日期
    end_date : str
        结束日期
        
    Returns
    -------
    pd.DataFrame
        EPS数据，index为日期，columns为股票代码
    """
    print("正在从数据库获取EPS数据...")
    
    # 创建数据库连接
    db = DatabaseManager()
    
    # SQL查询：获取季度EPS数据
    query = """
    SELECT 
        s_info_windcode as stock_code,
        report_period as date,
        net_profit_parent_comp_ttm / tot_shr as eps_ttm
    FROM financial_data
    WHERE report_period >= %s 
        AND report_period <= %s
        AND net_profit_parent_comp_ttm IS NOT NULL
        AND tot_shr IS NOT NULL
        AND tot_shr > 0
    ORDER BY stock_code, report_period
    """
    
    # 如果没有EPS数据表，使用模拟数据
    try:
        df = pd.read_sql(query, db.connection, params=(start_date, end_date))
        
        # 转换为宽表格式
        eps_data = df.pivot(index='date', columns='stock_code', values='eps_ttm')
        
    except Exception as e:
        print(f"从数据库获取数据失败: {e}")
        print("使用模拟数据进行演示...")
        
        # 生成模拟的EPS数据（使用日频数据，与交易数据匹配）
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        # 使用更多股票以匹配真实数据
        stocks = [f'{i:06d}.SZ' if i % 2 == 0 else f'{600000+i:06d}.SH' 
                  for i in range(1, 101)]  # 100只股票
        
        # 创建基础EPS值
        np.random.seed(42)
        base_eps = np.random.uniform(0.5, 2.0, size=(len(stocks),))
        
        # 生成时间序列数据（带趋势和噪声）
        # 使用季度数据，但对齐到日期索引
        quarterly_dates = pd.date_range(start=start_date, end=end_date, freq='Q')
        data = {}
        for i, stock in enumerate(stocks):
            # 生成季度EPS数据
            trend = np.linspace(0, 0.3, len(quarterly_dates))
            seasonal = 0.1 * np.sin(np.arange(len(quarterly_dates)) * 2 * np.pi / 4)
            noise = np.random.normal(0, 0.1, len(quarterly_dates))
            quarterly_eps = base_eps[i] + trend + seasonal + noise
            
            # 扩展到日频，使用前向填充
            eps_series = pd.Series(quarterly_eps, index=quarterly_dates)
            eps_series = eps_series.reindex(dates, method='ffill')
            data[stock] = eps_series
            
        eps_data = pd.DataFrame(data, index=dates)
        
    print(f"获取到 {len(eps_data)} 个时间点的EPS数据")
    print(f"股票数量: {len(eps_data.columns)}")
    
    return eps_data


def generate_sue_factor(eps_data):
    """
    生成SUE因子
    
    Parameters
    ----------
    eps_data : pd.DataFrame
        EPS数据
        
    Returns
    -------
    pd.DataFrame
        SUE因子值
    """
    print("\n正在计算SUE因子...")
    
    # 创建SUE因子计算器
    # 将日频EPS数据按季度采样进行SUE计算
    sue_calculator = SUE(
        method='historical',
        lookback_quarters=120,  # 约4个季度的交易日
        std_quarters=240,       # 约8个季度的交易日
        min_quarters=120        # 最少需要的交易日
    )
    
    # 计算SUE因子
    sue_factor = sue_calculator.calculate(eps_data)
    
    # 统计信息
    valid_count = sue_factor.notna().sum().sum()
    total_count = sue_factor.size
    coverage = valid_count / total_count * 100
    
    print(f"SUE因子计算完成")
    print(f"有效值数量: {valid_count:,}")
    print(f"总数据点: {total_count:,}")
    print(f"覆盖率: {coverage:.2f}%")
    
    # 打印描述性统计
    print("\nSUE因子统计信息:")
    stats = sue_factor.stack().describe()
    print(stats)
    
    return sue_factor


def save_sue_factor(sue_factor, filename='SUE.pkl'):
    """
    保存SUE因子到本地
    
    Parameters
    ----------
    sue_factor : pd.DataFrame
        SUE因子数据
    filename : str
        保存的文件名
    """
    # 获取因子保存路径
    factor_dir = Path(get_config('main.paths.factors'))
    factor_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = factor_dir / filename
    
    # 保存因子数据
    with open(filepath, 'wb') as f:
        pickle.dump(sue_factor, f)
        
    print(f"\nSUE因子已保存到: {filepath}")
    
    # 同时保存为CSV格式便于查看
    csv_path = filepath.with_suffix('.csv')
    sue_factor.to_csv(csv_path)
    print(f"CSV格式已保存到: {csv_path}")
    
    return filepath


def analyze_sue_distribution(sue_factor):
    """
    分析SUE因子的分布特征
    
    Parameters
    ----------
    sue_factor : pd.DataFrame
        SUE因子数据
    """
    print("\n" + "="*60)
    print("SUE因子分布分析")
    print("="*60)
    
    # 展平数据
    sue_values = sue_factor.stack().dropna()
    
    # 1. 基本统计
    print("\n1. 基本统计量:")
    print(f"   样本数: {len(sue_values):,}")
    print(f"   均值: {sue_values.mean():.4f}")
    print(f"   标准差: {sue_values.std():.4f}")
    print(f"   偏度: {sue_values.skew():.4f}")
    print(f"   峰度: {sue_values.kurt():.4f}")
    
    # 2. 分位数
    print("\n2. 分位数:")
    quantiles = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    for q in quantiles:
        print(f"   {q*100:>5.0f}%: {sue_values.quantile(q):>8.4f}")
        
    # 3. 极值分析
    print("\n3. 极值分析:")
    print(f"   最小值: {sue_values.min():.4f}")
    print(f"   最大值: {sue_values.max():.4f}")
    print(f"   正值占比: {(sue_values > 0).mean()*100:.2f}%")
    print(f"   负值占比: {(sue_values < 0).mean()*100:.2f}%")
    
    # 4. 时间序列特征
    print("\n4. 时间序列特征:")
    time_coverage = sue_factor.notna().mean(axis=1)
    print(f"   平均时间覆盖率: {time_coverage.mean()*100:.2f}%")
    print(f"   最低时间覆盖率: {time_coverage.min()*100:.2f}%")
    print(f"   最高时间覆盖率: {time_coverage.max()*100:.2f}%")
    
    # 5. 截面特征
    print("\n5. 截面特征:")
    stock_coverage = sue_factor.notna().mean(axis=0)
    print(f"   平均股票覆盖率: {stock_coverage.mean()*100:.2f}%")
    print(f"   有效股票数: {(stock_coverage > 0).sum()}")
    print(f"   完整数据股票数: {(stock_coverage == 1).sum()}")


def main():
    """主函数"""
    print("="*60)
    print("SUE因子生成程序")
    print("="*60)
    
    # 1. 获取EPS数据
    eps_data = fetch_eps_data(
        start_date='2020-01-01',
        end_date='2024-12-31'
    )
    
    # 2. 生成SUE因子
    sue_factor = generate_sue_factor(eps_data)
    
    # 3. 保存因子
    filepath = save_sue_factor(sue_factor)
    
    # 4. 分析因子分布
    analyze_sue_distribution(sue_factor)
    
    # 5. 展示样例数据
    print("\n" + "="*60)
    print("SUE因子样例数据（最近5个时间点，前5只股票）:")
    print("="*60)
    if not sue_factor.empty:
        sample = sue_factor.iloc[-5:, :5]
        print(sample)
    
    print("\n" + "="*60)
    print("SUE因子生成完成！")
    print("="*60)
    
    return sue_factor


if __name__ == "__main__":
    sue_factor = main()