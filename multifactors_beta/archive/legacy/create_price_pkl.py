#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建完整的Price.pkl文件
使用分块获取方式从2022年开始获取数据
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


def create_price_pkl():
    """创建Price.pkl文件（从2022年开始）"""
    logger.info("开始创建Price.pkl文件")
    
    # 创建分块获取器
    # 使用较小的分块以确保稳定性
    fetcher = ChunkedPriceFetcher(chunk_days=60, chunk_stocks=1000)
    
    try:
        # 从2022年开始获取数据（约3年的数据量）
        begin_date = 20220101
        end_date = 0  # 到今天
        
        logger.info(f"开始获取价格数据：{begin_date} - 今天")
        logger.info("注意：这个过程可能需要10-30分钟，请耐心等待...")
        
        # 使用按日期分块的方式
        data = fetcher.fetch_price_data_chunked(
            begin_date=begin_date,
            end_date=end_date,
            by_date=True,  # 按日期分块
            save_intermediate=True  # 保存中间结果
        )
        
        if not data.empty:
            logger.info(f"数据获取成功！")
            logger.info(f"总记录数: {len(data):,}")
            logger.info(f"股票数量: {data['code'].nunique():,}")
            logger.info(f"日期范围: {data['tradingday'].min()} - {data['tradingday'].max()}")
            
            # 保存为Price.pkl文件
            file_path = fetcher.save_price_data(data, "Price.pkl")
            logger.info(f"Price.pkl已成功创建: {file_path}")
            
            # 验证文件
            logger.info("验证Price.pkl文件...")
            import pandas as pd
            saved_data = pd.read_pickle(file_path)
            logger.info(f"验证成功 - 文件大小: {os.path.getsize(file_path) / 1024 / 1024:.1f} MB")
            logger.info(f"数据形状: {saved_data.shape}")
            
            return True
        else:
            logger.error("没有获取到任何数据")
            return False
            
    except KeyboardInterrupt:
        logger.info("用户中断了数据获取过程")
        logger.info("中间文件已保存，可以运行 get_full_price_data.py 选择恢复功能")
        return False
        
    except Exception as e:
        logger.error(f"获取数据失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("创建Price.pkl文件")
    print("=" * 60)
    print("此操作将从2022年开始获取所有股票价格数据")
    print("预计耗时：10-30分钟")
    print("数据量：约300-500万条记录")
    print("=" * 60)
    
    confirm = input("确认开始获取数据？(y/N): ").strip().lower()
    
    if confirm == 'y':
        print("开始获取数据，请耐心等待...")
        success = create_price_pkl()
        
        if success:
            print("\n" + "=" * 60)
            print("[SUCCESS] Price.pkl文件创建成功！")
            print("现在可以使用完整的价格数据进行因子分析了。")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("[FAILED] Price.pkl文件创建失败")
            print("请检查日志信息或尝试恢复中间文件")
            print("=" * 60)
    else:
        print("操作已取消")