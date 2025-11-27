#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修正版5日收益率计算逻辑
专门验证600770的计算结果
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_returns_5d_fixed():
    """使用修正后的方法计算5日收益率"""
    try:
        # 1. 加载价格数据
        price_file = Path('E:/Documents/PythonProject/StockProject/StockData/Price.pkl')
        logger.info(f"加载价格数据: {price_file}")
        price_data = pd.read_pickle(price_file)
        
        # 2. 计算复权收盘价
        price_data['adj_close'] = price_data['c'] * price_data['adjfactor']
        
        # 3. 重置索引，更容易处理
        df = price_data.reset_index()
        
        # 4. 专门测试600770
        logger.info("专门测试600770...")
        stock_600770 = df[df['StockCodes'] == '600770'].copy()
        stock_600770 = stock_600770.sort_values('TradingDates')
        
        # 5. 计算日收益率
        stock_600770['daily_return'] = np.log(stock_600770['adj_close'] / stock_600770['adj_close'].shift(1))
        
        # 6. 计算5日滚动收益率  
        stock_600770['returns_5d'] = stock_600770['daily_return'].rolling(window=5, min_periods=5).sum()
        
        # 7. 检查2024-06-05附近的数据
        target_date = pd.Timestamp('2024-06-05')
        mask = (stock_600770['TradingDates'] >= target_date - pd.Timedelta(days=5)) & \
               (stock_600770['TradingDates'] <= target_date + pd.Timedelta(days=5))
        
        test_data = stock_600770[mask][['TradingDates', 'adj_close', 'daily_return', 'returns_5d']].copy()
        
        logger.info("\n600770修正算法计算结果:")
        logger.info("日期           复权价格    日收益率    5日收益率")
        logger.info("-" * 50)
        
        for _, row in test_data.iterrows():
            date = row['TradingDates']
            adj_price = row['adj_close']
            daily_ret = row['daily_return']
            roll_5d = row['returns_5d']
            
            daily_str = f"{(np.exp(daily_ret) - 1) * 100:6.2f}%" if not pd.isna(daily_ret) else "   --"
            roll_str = f"{(np.exp(roll_5d) - 1) * 100:6.2f}%" if not pd.isna(roll_5d) else "   --"
            
            logger.info(f"{date}  {adj_price:8.2f}     {daily_str}     {roll_str}")
        
        # 8. 具体检查2024-06-05的值
        target_row = test_data[test_data['TradingDates'] == target_date]
        if not target_row.empty:
            target_value = target_row['returns_5d'].iloc[0]
            if not pd.isna(target_value):
                target_pct = (np.exp(target_value) - 1) * 100
                logger.info(f"\n【关键验证】600770在2024-06-05的5日收益率:")
                logger.info(f"对数收益率: {target_value:.6f}")
                logger.info(f"实际收益率: {target_pct:.2f}%")
                logger.info(f"预期结果: 约-9% (与手工计算一致)")
                
                if abs(target_pct - (-9.03)) < 1.0:  # 允许1%的误差
                    logger.info("修正算法验证成功！")
                    return True
                else:
                    logger.error(f"修正算法验证失败！预期约-9%，实际{target_pct:.2f}%")
                    return False
            else:
                logger.error("目标日期的5日收益率为NaN")
                return False
        else:
            logger.error("未找到目标日期的数据")
            return False
            
    except Exception as e:
        logger.error(f"修正算法测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def batch_test_corrected_algorithm():
    """批量测试修正后的算法"""
    try:
        logger.info("开始批量测试修正算法...")
        
        # 1. 加载数据
        price_file = Path('E:/Documents/PythonProject/StockProject/StockData/Price.pkl')
        price_data = pd.read_pickle(price_file)
        
        # 2. 计算复权价格
        price_data['adj_close'] = price_data['c'] * price_data['adjfactor']
        
        # 3. 选择几只代表性股票进行测试
        test_stocks = ['600770', '000001', '000002', '002509']  # 包括之前的极值股票
        
        results = []
        
        for stock_code in test_stocks:
            logger.info(f"测试股票 {stock_code}...")
            
            # 获取股票数据
            stock_data = price_data.loc[(slice(None), stock_code), :].copy()
            stock_data = stock_data.reset_index()
            stock_data = stock_data.sort_values('TradingDates')
            
            # 计算收益率
            stock_data['daily_return'] = np.log(stock_data['adj_close'] / stock_data['adj_close'].shift(1))
            stock_data['returns_5d'] = stock_data['daily_return'].rolling(window=5, min_periods=5).sum()
            
            # 统计
            valid_returns = stock_data['returns_5d'].dropna()
            if len(valid_returns) > 0:
                min_ret = valid_returns.min()
                max_ret = valid_returns.max()
                min_pct = (np.exp(min_ret) - 1) * 100
                max_pct = (np.exp(max_ret) - 1) * 100
                
                results.append({
                    'stock': stock_code,
                    'count': len(valid_returns),
                    'min_log': min_ret,
                    'max_log': max_ret,
                    'min_pct': min_pct,
                    'max_pct': max_pct
                })
                
                logger.info(f"  {stock_code}: 数据点{len(valid_returns):,}, 范围[{min_pct:.2f}%, {max_pct:.2f}%]")
        
        # 汇总结果
        logger.info("\n批量测试汇总:")
        for result in results:
            logger.info(f"{result['stock']}: 收益率范围 [{result['min_pct']:6.2f}%, {result['max_pct']:6.2f}%]")
        
        # 检查是否还有异常极值
        extreme_results = [r for r in results if r['min_pct'] < -90 or r['max_pct'] > 900]
        
        if extreme_results:
            logger.error("仍然发现异常极值:")
            for r in extreme_results:
                logger.error(f"  {r['stock']}: [{r['min_pct']:.2f}%, {r['max_pct']:.2f}%]")
            return False
        else:
            logger.info("批量测试通过，没有发现异常极值!")
            return True
            
    except Exception as e:
        logger.error(f"批量测试失败: {e}")
        return False

if __name__ == "__main__":
    print("修正版5日收益率算法测试")
    print("=" * 60)
    
    try:
        # 1. 单独测试600770
        logger.info("Phase 1: 测试600770修正结果...")
        if calculate_returns_5d_fixed():
            logger.info("600770测试通过!")
            
            # 2. 批量测试
            logger.info("\nPhase 2: 批量测试修正算法...")
            if batch_test_corrected_algorithm():
                logger.info("批量测试通过!")
                print("\n修正算法验证成功！现在可以重新生成因子数据。")
            else:
                print("\n批量测试失败，需要进一步调查。")
        else:
            print("\n600770测试失败，修正算法仍有问题。")
            
    except Exception as e:
        logger.error(f"测试过程失败: {e}")
        print("测试过程出现错误。")