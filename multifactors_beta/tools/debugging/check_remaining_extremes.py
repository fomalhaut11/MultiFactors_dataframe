#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查修正后数据中的剩余极值问题
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_remaining_extremes():
    """检查修正后数据中剩余的极值"""
    try:
        # 加载修正后的数据
        data_file = Path('E:/Documents/PythonProject/StockProject/StockData/factors/technical/Returns_5D_C2C.pkl')
        logger.info("加载修正后的5日收益率数据...")
        returns_data = pd.read_pickle(data_file)
        
        logger.info(f"数据形状: {returns_data.shape}")
        
        # 找极值
        min_return = returns_data.min()
        max_return = returns_data.max()
        
        min_pct = (np.exp(min_return) - 1) * 100
        max_pct = (np.exp(max_return) - 1) * 100
        
        logger.info(f"整体范围: [{min_pct:.2f}%, {max_pct:.2f}%]")
        
        # 找到最极端的情况
        min_idx = returns_data.idxmin()  
        max_idx = returns_data.idxmax()
        
        logger.info(f"\n最大亏损:")
        logger.info(f"股票: {min_idx[1]}, 日期: {min_idx[0]}, 收益率: {min_pct:.2f}%")
        
        logger.info(f"\n最大收益:")
        logger.info(f"股票: {max_idx[1]}, 日期: {max_idx[0]}, 收益率: {max_pct:.2f}%")
        
        # 统计极值数量
        extreme_negative = returns_data[returns_data < np.log(0.1)]  # <-90%
        extreme_positive = returns_data[returns_data > np.log(5.0)]  # >400%
        
        logger.info(f"\n极值统计:")
        logger.info(f"收益率<-90%: {len(extreme_negative):,} 次")
        logger.info(f"收益率>400%: {len(extreme_positive):,} 次")
        
        # 找到最极端的几个案例进行分析
        logger.info(f"\n最大亏损TOP5:")
        worst_5 = returns_data.nsmallest(5)
        for i, (idx, value) in enumerate(worst_5.items(), 1):
            actual_return = (np.exp(value) - 1) * 100
            logger.info(f"{i}. {idx[0]} {idx[1]}: {actual_return:+6.2f}%")
        
        logger.info(f"\n最大收益TOP5:")
        best_5 = returns_data.nlargest(5) 
        for i, (idx, value) in enumerate(best_5.items(), 1):
            actual_return = (np.exp(value) - 1) * 100
            logger.info(f"{i}. {idx[0]} {idx[1]}: {actual_return:+6.2f}%")
            
        return worst_5, best_5
        
    except Exception as e:
        logger.error(f"检查极值失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def investigate_extreme_case(stock_code, date):
    """深入调查特定的极值案例"""
    try:
        logger.info(f"\n深入调查 {stock_code} 在 {date} 的极值情况...")
        
        # 加载原始价格数据
        price_file = Path('E:/Documents/PythonProject/StockProject/StockData/Price.pkl')
        price_data = pd.read_pickle(price_file)
        
        # 筛选该股票的数据
        stock_data = price_data.loc[(slice(None), stock_code), :].copy()
        stock_data = stock_data.reset_index()
        stock_data = stock_data.sort_values('TradingDates')
        
        # 查看目标日期前后的数据
        target_date = pd.Timestamp(date)
        mask = (stock_data['TradingDates'] >= target_date - pd.Timedelta(days=10)) & \
               (stock_data['TradingDates'] <= target_date + pd.Timedelta(days=10))
        
        around_data = stock_data[mask].copy()
        
        if len(around_data) > 0:
            # 计算复权价格和收益率
            around_data['adj_close'] = around_data['c'] * around_data['adjfactor']
            around_data['daily_return'] = np.log(around_data['adj_close'] / around_data['adj_close'].shift(1))
            around_data['returns_5d'] = around_data['daily_return'].rolling(window=5, min_periods=5).sum()
            
            logger.info(f"\n{stock_code} 在 {date} 前后的详细数据:")
            logger.info("日期           收盘价  复权因子   复权价格    日收益率   5日收益率")
            logger.info("-" * 70)
            
            for _, row in around_data.iterrows():
                date_str = row['TradingDates'].strftime('%Y-%m-%d')
                close = row['c']
                adjfactor = row['adjfactor']
                adj_close = row['adj_close']
                daily_ret = row['daily_return']
                ret_5d = row['returns_5d']
                
                daily_str = f"{(np.exp(daily_ret)-1)*100:7.2f}%" if not pd.isna(daily_ret) else "    --"
                ret5d_str = f"{(np.exp(ret_5d)-1)*100:8.2f}%" if not pd.isna(ret_5d) else "     --"
                
                logger.info(f"{date_str}  {close:6.2f}  {adjfactor:8.4f}  {adj_close:8.2f}  {daily_str}  {ret5d_str}")
                
            # 检查复权因子变化
            adj_changes = around_data['adjfactor'].diff().abs()
            max_change = adj_changes.max()
            
            if max_change > 0.01:
                logger.info(f"\n发现复权因子大变化: 最大变化 {max_change:.6f}")
                big_changes = around_data[adj_changes > 0.01]
                for _, row in big_changes.iterrows():
                    logger.info(f"  {row['TradingDates']}: 复权因子变化")
                    
        else:
            logger.info("没有找到相关数据")
            
    except Exception as e:
        logger.error(f"调查极值案例失败: {e}")

if __name__ == "__main__":
    print("检查修正后数据的剩余极值问题")
    print("=" * 60)
    
    # 1. 检查整体极值情况
    worst_cases, best_cases = check_remaining_extremes()
    
    if worst_cases is not None and len(worst_cases) > 0:
        # 2. 深入调查最极端的案例
        worst_idx = worst_cases.index[0]  # 最大亏损的案例
        stock_code = worst_idx[1]
        date = worst_idx[0]
        
        investigate_extreme_case(stock_code, date)
        
        # 也调查最大收益的案例
        if best_cases is not None and len(best_cases) > 0:
            best_idx = best_cases.index[0]
            best_stock = best_idx[1]
            best_date = best_idx[0]
            
            investigate_extreme_case(best_stock, best_date)
    
    print("\n分析完成！")