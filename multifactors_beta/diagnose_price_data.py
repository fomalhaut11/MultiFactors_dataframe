#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断Price.pkl原始数据
专门分析600770在2024-06-05附近的异常情况
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_diagnose_price_data():
    """加载并诊断价格数据"""
    try:
        # 加载原始价格数据
        price_file = Path('E:/Documents/PythonProject/StockProject/StockData/Price.pkl')
        logger.info(f"加载原始价格数据: {price_file}")
        
        price_data = pd.read_pickle(price_file)
        logger.info(f"价格数据形状: {price_data.shape}")
        logger.info(f"价格数据列: {list(price_data.columns)}")
        logger.info(f"价格数据索引类型: {type(price_data.index)}")
        
        if isinstance(price_data.index, pd.MultiIndex):
            logger.info(f"MultiIndex层级: {price_data.index.names}")
            dates = price_data.index.get_level_values(0)
            stocks = price_data.index.get_level_values(1)
            logger.info(f"日期范围: {dates.min()} - {dates.max()}")
            logger.info(f"股票数量: {stocks.nunique()}")
        
        return price_data
        
    except Exception as e:
        logger.error(f"加载价格数据失败: {e}")
        raise

def analyze_stock_600770(price_data):
    """分析600770的具体数据"""
    try:
        logger.info("=" * 60)
        logger.info("分析600770的价格数据")
        logger.info("=" * 60)
        
        # 筛选600770的数据
        stock_data = price_data.loc[(slice(None), '600770'), :]
        logger.info(f"600770数据量: {len(stock_data)}")
        
        # 重置索引以便于分析
        stock_data = stock_data.reset_index()
        stock_data = stock_data.sort_values('TradingDates')
        
        # 分析2024-06-05前后的数据
        target_date = pd.Timestamp('2024-06-05')
        
        # 找到目标日期前后5天的数据
        date_mask = (stock_data['TradingDates'] >= target_date - pd.Timedelta(days=10)) & \
                   (stock_data['TradingDates'] <= target_date + pd.Timedelta(days=10))
        
        around_target = stock_data[date_mask].copy()
        logger.info(f"目标日期前后数据量: {len(around_target)}")
        
        if len(around_target) > 0:
            logger.info("\n600770在2024-06-05前后的原始数据:")
            logger.info("日期           收盘价    复权因子    复权收盘价")
            logger.info("-" * 50)
            
            for _, row in around_target.iterrows():
                date = row['TradingDates']
                close = row['c']
                adjfactor = row['adjfactor']
                adj_close = close * adjfactor
                logger.info(f"{date}  {close:8.2f}  {adjfactor:8.4f}  {adj_close:10.2f}")
            
            # 计算手工验证的收益率
            logger.info("\n手工计算日收益率验证:")
            logger.info("日期           日收益率   累计5日收益率")
            logger.info("-" * 40)
            
            around_target['adj_close'] = around_target['c'] * around_target['adjfactor']
            around_target['daily_return'] = np.log(around_target['adj_close'] / around_target['adj_close'].shift(1))
            around_target['rolling_5d'] = around_target['daily_return'].rolling(window=5, min_periods=5).sum()
            
            for _, row in around_target.iterrows():
                date = row['TradingDates'] 
                daily_ret = row['daily_return']
                roll_5d = row['rolling_5d']
                if not pd.isna(daily_ret):
                    daily_pct = (np.exp(daily_ret) - 1) * 100
                    if not pd.isna(roll_5d):
                        roll_pct = (np.exp(roll_5d) - 1) * 100
                        logger.info(f"{date}  {daily_pct:8.2f}%     {roll_pct:8.2f}%")
                    else:
                        logger.info(f"{date}  {daily_pct:8.2f}%")
                else:
                    logger.info(f"{date}      --")
            
            # 检查异常
            max_daily = around_target['daily_return'].max()
            min_daily = around_target['daily_return'].min()
            
            if not pd.isna(max_daily):
                max_daily_pct = (np.exp(max_daily) - 1) * 100
                logger.info(f"\n最大日收益率: {max_daily_pct:.2f}%")
            
            if not pd.isna(min_daily):
                min_daily_pct = (np.exp(min_daily) - 1) * 100
                logger.info(f"最小日收益率: {min_daily_pct:.2f}%")
            
            # 检查复权因子是否有突变
            adj_factor_diff = around_target['adjfactor'].diff()
            max_adj_change = adj_factor_diff.abs().max()
            logger.info(f"\n复权因子最大变化: {max_adj_change:.6f}")
            
            if max_adj_change > 0.1:  # 如果复权因子变化超过0.1
                logger.warning("发现复权因子大幅变化，可能是除权除息导致的")
                big_changes = around_target[adj_factor_diff.abs() > 0.1]
                for _, row in big_changes.iterrows():
                    logger.warning(f"  {row['TradingDates']}: 复权因子变化 {adj_factor_diff.loc[row.name]:.6f}")
        
        else:
            logger.warning("在指定日期范围内没有找到600770的数据")
            
        return around_target
        
    except Exception as e:
        logger.error(f"分析600770数据失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def check_data_integrity(price_data):
    """检查数据完整性"""
    try:
        logger.info("=" * 60) 
        logger.info("数据完整性检查")
        logger.info("=" * 60)
        
        # 检查关键列是否存在
        required_cols = ['c', 'adjfactor']
        missing_cols = [col for col in required_cols if col not in price_data.columns]
        
        if missing_cols:
            logger.error(f"缺少必需列: {missing_cols}")
            return False
        
        # 检查空值
        for col in required_cols:
            null_count = price_data[col].isnull().sum()
            logger.info(f"{col}列空值数量: {null_count:,}")
        
        # 检查复权因子的分布
        adj_stats = price_data['adjfactor'].describe()
        logger.info("\n复权因子分布统计:")
        logger.info(adj_stats)
        
        # 检查异常的复权因子值
        extreme_adj = price_data[(price_data['adjfactor'] < 0.01) | (price_data['adjfactor'] > 100)]
        logger.info(f"\n极端复权因子(<0.01或>100)数量: {len(extreme_adj):,}")
        
        if len(extreme_adj) > 0:
            logger.info("极端复权因子样本:")
            logger.info(extreme_adj.head(10))
        
        return True
        
    except Exception as e:
        logger.error(f"数据完整性检查失败: {e}")
        return False

if __name__ == "__main__":
    print("Price.pkl数据诊断")
    print("=" * 60)
    
    try:
        # 1. 加载数据
        price_data = load_and_diagnose_price_data()
        
        # 2. 检查数据完整性
        if check_data_integrity(price_data):
            logger.info("数据完整性检查通过")
        
        # 3. 专门分析600770
        stock_data = analyze_stock_600770(price_data)
        
        logger.info("\n诊断完成！")
        
    except Exception as e:
        logger.error(f"诊断失败: {e}")
        import traceback
        traceback.print_exc()