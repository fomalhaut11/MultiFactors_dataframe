#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小范围测试分块获取功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
from data.fetcher.chunked_price_fetcher import ChunkedPriceFetcher

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_small_range():
    """测试小范围数据获取"""
    logger.info("开始小范围测试分块获取功能")
    
    # 创建分块获取器（小块测试）
    fetcher = ChunkedPriceFetcher(chunk_days=30, chunk_stocks=500)
    
    try:
        # 测试2025年6月份的数据（约30天）
        begin_date = 20250601
        end_date = 20250630
        
        logger.info(f"测试获取2025年6月数据：{begin_date} - {end_date}")
        
        # 按日期分块获取
        data = fetcher.fetch_price_data_chunked(
            begin_date=begin_date,
            end_date=end_date,
            by_date=True,
            save_intermediate=False  # 小测试不需要中间文件
        )
        
        if not data.empty:
            logger.info(f"测试成功！获取数据形状: {data.shape}")
            logger.info(f"股票数量: {data['code'].nunique()}")
            logger.info(f"日期范围: {data['tradingday'].min()} - {data['tradingday'].max()}")
            
            # 保存测试结果
            file_path = fetcher.save_price_data(data, "Price_small_test.pkl")
            logger.info(f"测试数据已保存到: {file_path}")
            
            # 验证保存的数据
            logger.info("验证保存的数据格式...")
            import pandas as pd
            saved_data = pd.read_pickle(file_path)
            logger.info(f"保存的数据形状: {saved_data.shape}")
            logger.info(f"索引级别: {saved_data.index.names}")
            logger.info(f"列名: {list(saved_data.columns)}")
            
            return True
        else:
            logger.error("测试失败：没有获取到数据")
            return False
            
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_small_range()
    if success:
        print("[OK] 小范围分块获取测试通过")
    else:
        print("[FAIL] 小范围分块获取测试失败")