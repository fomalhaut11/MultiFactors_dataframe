#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析5日收益率因子的极值情况

查找最大亏损和最大收益的股票和日期
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def load_returns_5d_data():
    """加载5日收益率数据"""
    try:
        # 从存储目录加载
        data_root = Path('E:/Documents/PythonProject/StockProject/StockData')
        factor_file = data_root / 'factors' / 'technical' / 'Returns_5D_C2C.pkl'
        
        if not factor_file.exists():
            raise FileNotFoundError(f"5日收益率因子文件不存在: {factor_file}")
        
        logger.info(f"加载5日收益率数据: {factor_file}")
        returns_data = pd.read_pickle(factor_file)
        
        logger.info(f"数据形状: {returns_data.shape}")
        logger.info(f"数据范围: [{returns_data.min():.6f}, {returns_data.max():.6f}]")
        
        return returns_data
        
    except Exception as e:
        logger.error(f"加载5日收益率数据失败: {e}")
        raise


def find_extreme_returns(returns_data):
    """查找极值收益率"""
    try:
        # 找到最小值（最大亏损）
        min_return = returns_data.min()
        min_idx = returns_data.idxmin()
        
        # 找到最大值（最大收益）
        max_return = returns_data.max()
        max_idx = returns_data.idxmax()
        
        logger.info("=" * 60)
        logger.info("5日收益率极值分析")
        logger.info("=" * 60)
        
        # 最大亏损分析
        logger.info("📉 最大亏损情况:")
        logger.info(f"对数收益率: {min_return:.6f}")
        logger.info(f"实际收益率: {(np.exp(min_return) - 1) * 100:.2f}%")
        
        if isinstance(min_idx, tuple) and len(min_idx) == 2:
            date, stock = min_idx
            logger.info(f"发生日期: {date}")
            logger.info(f"股票代码: {stock}")
        else:
            logger.info(f"索引信息: {min_idx}")
        
        logger.info("-" * 40)
        
        # 最大收益分析
        logger.info("📈 最大收益情况:")
        logger.info(f"对数收益率: {max_return:.6f}")
        logger.info(f"实际收益率: {(np.exp(max_return) - 1) * 100:.2f}%")
        
        if isinstance(max_idx, tuple) and len(max_idx) == 2:
            date, stock = max_idx
            logger.info(f"发生日期: {date}")
            logger.info(f"股票代码: {stock}")
        else:
            logger.info(f"索引信息: {max_idx}")
        
        # 统计分析
        logger.info("-" * 40)
        logger.info("📊 统计摘要:")
        logger.info(f"数据总量: {len(returns_data):,}")
        logger.info(f"均值: {returns_data.mean():.6f}")
        logger.info(f"标准差: {returns_data.std():.6f}")
        logger.info(f"中位数: {returns_data.median():.6f}")
        
        # 分位数分析
        percentiles = [1, 5, 10, 90, 95, 99]
        logger.info("分位数分析:")
        for p in percentiles:
            value = returns_data.quantile(p/100)
            actual_return = (np.exp(value) - 1) * 100
            logger.info(f"  {p:2d}%分位: {value:.6f} ({actual_return:+6.2f}%)")
        
        # 极值计数
        logger.info("-" * 40)
        logger.info("极值统计:")
        
        # 大于100%收益的情况
        extreme_positive = returns_data[returns_data > np.log(2.0)]  # 收益率>100%
        logger.info(f"收益率>100%的情况: {len(extreme_positive):,} 次")
        
        # 小于-50%收益的情况
        extreme_negative = returns_data[returns_data < np.log(0.5)]  # 收益率<-50%
        logger.info(f"收益率<-50%的情况: {len(extreme_negative):,} 次")
        
        # 小于-80%收益的情况  
        very_extreme_negative = returns_data[returns_data < np.log(0.2)]  # 收益率<-80%
        logger.info(f"收益率<-80%的情况: {len(very_extreme_negative):,} 次")
        
        return {
            'min_return': min_return,
            'min_idx': min_idx,
            'max_return': max_return, 
            'max_idx': max_idx,
            'extreme_positive': extreme_positive,
            'extreme_negative': extreme_negative,
            'very_extreme_negative': very_extreme_negative
        }
        
    except Exception as e:
        logger.error(f"查找极值失败: {e}")
        raise


def analyze_extreme_cases(returns_data, extremes):
    """分析极值案例的详细情况"""
    try:
        logger.info("=" * 60)
        logger.info("极值案例详细分析")
        logger.info("=" * 60)
        
        # 分析最大亏损的前10个案例
        logger.info("🔍 最大亏损TOP10:")
        worst_returns = returns_data.nsmallest(10)
        for i, (idx, value) in enumerate(worst_returns.items(), 1):
            if isinstance(idx, tuple) and len(idx) == 2:
                date, stock = idx
                actual_return = (np.exp(value) - 1) * 100
                logger.info(f"{i:2d}. {date} {stock}: {value:.6f} ({actual_return:+6.2f}%)")
            else:
                actual_return = (np.exp(value) - 1) * 100
                logger.info(f"{i:2d}. {idx}: {value:.6f} ({actual_return:+6.2f}%)")
        
        logger.info("-" * 40)
        
        # 分析最大收益的前10个案例
        logger.info("🔍 最大收益TOP10:")
        best_returns = returns_data.nlargest(10)
        for i, (idx, value) in enumerate(best_returns.items(), 1):
            if isinstance(idx, tuple) and len(idx) == 2:
                date, stock = idx
                actual_return = (np.exp(value) - 1) * 100
                logger.info(f"{i:2d}. {date} {stock}: {value:.6f} ({actual_return:+6.2f}%)")
            else:
                actual_return = (np.exp(value) - 1) * 100
                logger.info(f"{i:2d}. {idx}: {value:.6f} ({actual_return:+6.2f}%)")
        
        # 按年份统计极值分布
        if isinstance(returns_data.index, pd.MultiIndex):
            logger.info("-" * 40)
            logger.info("📅 按年份统计极值分布:")
            
            dates = returns_data.index.get_level_values(0)
            years = dates.year
            
            # 极端负收益按年份分布
            extreme_neg_by_year = extremes['very_extreme_negative'].groupby(
                extremes['very_extreme_negative'].index.get_level_values(0).year
            ).size()
            
            logger.info("收益率<-80%的年份分布:")
            for year, count in extreme_neg_by_year.items():
                logger.info(f"  {year}: {count:,} 次")
        
        return True
        
    except Exception as e:
        logger.error(f"分析极值案例失败: {e}")
        return False


if __name__ == "__main__":
    print("5日收益率因子极值分析")
    print("=" * 60)
    
    try:
        # 1. 加载数据
        returns_data = load_returns_5d_data()
        
        # 2. 查找极值
        extremes = find_extreme_returns(returns_data)
        
        # 3. 详细分析
        analyze_extreme_cases(returns_data, extremes)
        
        print("\n" + "=" * 60)
        print("分析完成！")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"分析失败: {e}")
        import traceback
        traceback.print_exc()