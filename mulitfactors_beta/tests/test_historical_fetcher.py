#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试历史数据获取器
先获取2024年的数据作为测试
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
from datetime import datetime
from get_historical_price_2014 import HistoricalPriceFetcher

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_historical_fetcher():
    """测试历史数据获取器（仅2024年）"""
    logger.info("开始测试历史数据获取器")
    
    try:
        # 创建获取器
        fetcher = HistoricalPriceFetcher()
        
        # 测试获取2024年的数据
        logger.info("测试获取2024年数据...")
        success = fetcher.fetch_year_data(2024)
        
        if success:
            logger.info("2024年数据获取成功")
            
            # 测试合并功能（只有一年的数据）
            logger.info("测试数据合并...")
            combine_success = fetcher.combine_yearly_data()
            
            if combine_success:
                logger.info("数据合并成功")
                
                # 验证生成的文件
                from core.config_manager import get_path
                import pandas as pd
                
                price_file = os.path.join(get_path('data_root'), "Price.pkl")
                if os.path.exists(price_file):
                    data = pd.read_pickle(price_file)
                    logger.info(f"验证Price.pkl: 形状 {data.shape}")
                    logger.info(f"索引: {data.index.names}")
                    logger.info(f"日期范围: {data.index.get_level_values('TradingDates').min()} - {data.index.get_level_values('TradingDates').max()}")
                    
                    return True
                else:
                    logger.error("Price.pkl文件未生成")
                    return False
            else:
                logger.error("数据合并失败")
                return False
        else:
            logger.error("2024年数据获取失败")
            return False
            
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("测试历史数据获取器（2024年数据）")
    print("=" * 60)
    
    success = test_historical_fetcher()
    
    if success:
        print("[OK] 历史数据获取器测试通过")
        print("现在可以运行完整的历史数据获取了")
    else:
        print("[FAIL] 历史数据获取器测试失败")