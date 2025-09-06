#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
获取完整的价格数据
使用分块获取方式，避免内存溢出和数据库超时
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
from datetime import datetime
from data.fetcher.chunked_price_fetcher import ChunkedPriceFetcher

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def get_full_price_data():
    """获取完整的历史价格数据"""
    logger.info("开始获取完整的历史价格数据")
    
    # 创建分块获取器
    # 调整参数：每次获取90天数据，避免单次查询过大
    fetcher = ChunkedPriceFetcher(chunk_days=90, chunk_stocks=1000)
    
    try:
        # 获取从2020年开始的所有数据
        begin_date = 20200101
        end_date = 0  # 0表示到今天
        
        logger.info(f"开始分块获取价格数据：{begin_date} - 今天")
        
        # 使用按日期分块的方式
        data = fetcher.fetch_price_data_chunked(
            begin_date=begin_date,
            end_date=end_date,
            by_date=True,  # 按日期分块，更稳定
            save_intermediate=True  # 保存中间结果，防止中断
        )
        
        if not data.empty:
            logger.info(f"数据获取成功！总形状: {data.shape}")
            logger.info(f"股票数量: {data['code'].nunique()}")
            logger.info(f"日期范围: {data['tradingday'].min()} - {data['tradingday'].max()}")
            
            # 保存为最终的Price.pkl文件
            file_path = fetcher.save_price_data(data, "Price.pkl")
            logger.info(f"完整价格数据已保存到: {file_path}")
            
            return True
        else:
            logger.error("没有获取到任何数据")
            return False
            
    except KeyboardInterrupt:
        logger.info("用户中断了数据获取过程")
        logger.info("注意：中间结果已保存，可以稍后继续处理")
        return False
        
    except Exception as e:
        logger.error(f"获取数据失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def resume_from_chunks():
    """从中间文件恢复数据（如果获取过程中断）"""
    logger.info("尝试从中间文件恢复数据")
    
    from config import get_config
    import pandas as pd
    
    data_root = get_config('main.paths.data_root')
    
    # 查找所有临时文件
    temp_files = []
    for file in os.listdir(data_root):
        if file.startswith('temp_price_chunk_') and file.endswith('.pkl'):
            temp_files.append(os.path.join(data_root, file))
    
    if not temp_files:
        logger.info("没有找到中间文件")
        return False
    
    logger.info(f"找到 {len(temp_files)} 个中间文件")
    
    try:
        # 加载所有中间文件
        all_data = []
        for temp_file in temp_files:
            logger.info(f"加载: {temp_file}")
            chunk_data = pd.read_pickle(temp_file)
            all_data.append(chunk_data)
        
        # 合并数据
        logger.info("合并中间数据...")
        combined_data = pd.concat(all_data, ignore_index=True)
        combined_data = combined_data.drop_duplicates(['code', 'tradingday'])
        combined_data = combined_data.sort_values(['tradingday', 'code'])
        
        logger.info(f"合并完成，总形状: {combined_data.shape}")
        
        # 保存最终数据
        fetcher = ChunkedPriceFetcher()
        file_path = fetcher.save_price_data(combined_data, "Price.pkl")
        logger.info(f"数据已保存到: {file_path}")
        
        # 清理中间文件
        for temp_file in temp_files:
            os.remove(temp_file)
        logger.info("中间文件已清理")
        
        return True
        
    except Exception as e:
        logger.error(f"恢复数据失败: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("多因子股票研究系统 - 价格数据获取")
    print("=" * 60)
    print("选择操作:")
    print("1. 获取完整价格数据（分块方式）")
    print("2. 从中间文件恢复数据")
    print("3. 退出")
    print("=" * 60)
    
    try:
        choice = input("请选择操作 (1-3): ").strip()
        
        if choice == "1":
            success = get_full_price_data()
            if success:
                print("\n[OK] 价格数据获取成功！")
            else:
                print("\n[FAIL] 价格数据获取失败")
                
        elif choice == "2":
            success = resume_from_chunks()
            if success:
                print("\n[OK] 数据恢复成功！")
            else:
                print("\n[FAIL] 数据恢复失败")
                
        elif choice == "3":
            print("退出程序")
            
        else:
            print("无效选择")
            
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n[ERROR] 程序执行失败: {e}")