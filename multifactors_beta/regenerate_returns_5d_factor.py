#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用修正后的算法重新生成5日收益率因子数据

修正了MultiIndex处理错误，确保计算准确性
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def regenerate_returns_5d_factor_corrected():
    """用修正算法重新生成5日收益率因子"""
    try:
        logger.info("=" * 60)
        logger.info("开始用修正算法重新生成5日收益率因子")
        logger.info("=" * 60)
        
        # 1. 加载价格数据
        price_file = Path('E:/Documents/PythonProject/StockProject/StockData/Price.pkl')
        logger.info(f"加载价格数据: {price_file}")
        price_data = pd.read_pickle(price_file)
        logger.info(f"价格数据形状: {price_data.shape}")
        
        # 2. 计算复权收盘价
        logger.info("计算复权收盘价...")
        price_data['adj_close'] = price_data['c'] * price_data['adjfactor']
        
        # 3. 重置索引以便处理
        logger.info("重新整理数据结构...")
        df = price_data.reset_index()
        df = df.sort_values(['StockCodes', 'TradingDates'])
        
        # 4. 分批计算以优化内存使用
        logger.info("开始按股票分批计算收益率...")
        
        def calc_stock_returns(group):
            """计算单只股票的收益率"""
            # 确保按日期排序
            group = group.sort_values('TradingDates')
            
            # 计算日收益率
            group['daily_return'] = np.log(group['adj_close'] / group['adj_close'].shift(1))
            
            # 计算5日滚动收益率
            group['returns_5d'] = group['daily_return'].rolling(window=5, min_periods=5).sum()
            
            return group[['TradingDates', 'StockCodes', 'returns_5d']].dropna()
        
        # 获取所有股票
        unique_stocks = df['StockCodes'].unique()
        total_stocks = len(unique_stocks)
        logger.info(f"总股票数: {total_stocks}")
        
        # 分批处理
        batch_size = 200  # 每批200只股票
        all_results = []
        
        for i in range(0, total_stocks, batch_size):
            batch_stocks = unique_stocks[i:i+batch_size]
            batch_data = df[df['StockCodes'].isin(batch_stocks)]
            
            logger.info(f"处理第{i//batch_size + 1}批，股票 {i+1}-{min(i+batch_size, total_stocks)}")
            
            # 计算这批股票的收益率
            batch_results = batch_data.groupby('StockCodes').apply(calc_stock_returns)
            
            if not batch_results.empty:
                # 重置索引
                batch_results = batch_results.reset_index(drop=True)
                all_results.append(batch_results)
            
            # 进度报告
            if (i + batch_size) % 1000 == 0 or i + batch_size >= total_stocks:
                progress = min(i + batch_size, total_stocks) / total_stocks * 100
                logger.info(f"进度: {progress:.1f}% ({min(i + batch_size, total_stocks)}/{total_stocks})")
        
        # 5. 合并所有结果
        logger.info("合并所有计算结果...")
        final_result = pd.concat(all_results, ignore_index=True)
        
        logger.info(f"合并后数据量: {len(final_result):,}")
        
        # 6. 设置正确的MultiIndex
        logger.info("设置MultiIndex格式...")
        final_result = final_result.set_index(['TradingDates', 'StockCodes'])
        returns_5d_series = final_result['returns_5d']
        returns_5d_series.name = "Returns_5D_C2C_Corrected"
        
        # 7. 数据质量检查
        logger.info("进行数据质量检查...")
        total_count = len(returns_5d_series)
        null_count = returns_5d_series.isna().sum()
        
        if total_count > 0:
            min_val = returns_5d_series.min()
            max_val = returns_5d_series.max()
            mean_val = returns_5d_series.mean()
            std_val = returns_5d_series.std()
            
            # 转换为实际收益率百分比
            min_pct = (np.exp(min_val) - 1) * 100
            max_pct = (np.exp(max_val) - 1) * 100
            
            logger.info(f"数据质量统计:")
            logger.info(f"  总数据点: {total_count:,}")
            logger.info(f"  NaN数量: {null_count:,}")
            logger.info(f"  对数收益率范围: [{min_val:.6f}, {max_val:.6f}]")
            logger.info(f"  实际收益率范围: [{min_pct:.2f}%, {max_pct:.2f}%]")
            logger.info(f"  均值: {mean_val:.6f}")
            logger.info(f"  标准差: {std_val:.6f}")
            
            # 验证600770的值
            if ('2024-06-05', '600770') in returns_5d_series.index:
                test_value = returns_5d_series.loc[('2024-06-05', '600770')]
                test_pct = (np.exp(test_value) - 1) * 100
                logger.info(f"验证: 600770在2024-06-05的5日收益率 = {test_pct:.2f}%")
        
        # 8. 保存修正后的因子数据
        logger.info("保存修正后的因子数据...")
        
        # 准备存储目录
        data_root = Path('E:/Documents/PythonProject/StockProject/StockData')
        factors_dir = data_root / 'factors' / 'technical'
        factors_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存数据文件（替换原有的错误数据）
        factor_file = factors_dir / "Returns_5D_C2C.pkl"
        backup_file = factors_dir / "Returns_5D_C2C_backup_wrong.pkl"
        
        # 备份错误数据
        if factor_file.exists():
            logger.info("备份原错误数据...")
            import shutil
            shutil.move(str(factor_file), str(backup_file))
        
        # 保存修正数据
        returns_5d_series.to_pickle(factor_file)
        logger.info(f"修正数据已保存: {factor_file}")
        
        # 检查文件大小
        file_size_mb = factor_file.stat().st_size / (1024 * 1024)
        logger.info(f"文件大小: {file_size_mb:.1f} MB")
        
        # 9. 保存修正的元数据
        logger.info("保存修正的元数据...")
        metadata = {
            "name": "Returns_5D_C2C",
            "category": "technical",
            "description": "5日close-to-close滚动收益率 (修正版)",
            "data_requirements": [
                "Price.pkl (包含close price和adjustment factor)"
            ],
            "calculation_method": "close-to-close 5-day rolling log returns (corrected algorithm)",
            "time_direction": "历史数据，向前滚动",
            "output_format": "MultiIndex Series [TradingDates, StockCodes]",
            "frequency": "日频",
            "min_periods": 5,
            "calculation_date": datetime.now().isoformat(),
            "data_shape": [total_count],
            "file_path": str(factor_file),
            "file_size_mb": round(file_size_mb, 2),
            "version": "corrected_v1.0",
            "verification": {
                "600770_2024_06_05": f"{test_pct:.2f}%" if '600770' in str(returns_5d_series.index) else "N/A",
                "data_range_pct": f"[{min_pct:.2f}%, {max_pct:.2f}%]"
            }
        }
        
        metadata_file = factors_dir / "Returns_5D_C2C_metadata.json"
        import json
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"修正元数据已保存: {metadata_file}")
        
        # 10. 验证保存的数据
        logger.info("验证保存的数据...")
        loaded_data = pd.read_pickle(factor_file)
        if loaded_data.equals(returns_5d_series):
            logger.info("数据保存验证通过")
        else:
            logger.warning("数据保存验证失败")
        
        logger.info("=" * 60)
        logger.info("修正版5日收益率因子重新生成完成！")
        logger.info("=" * 60)
        logger.info(f"因子文件: {factor_file}")
        logger.info(f"元数据文件: {metadata_file}")
        logger.info(f"备份文件: {backup_file}")
        logger.info(f"数据点数: {total_count:,}")
        logger.info(f"收益率范围: [{min_pct:.2f}%, {max_pct:.2f}%]")
        logger.info("=" * 60)
        
        return factor_file, metadata_file, returns_5d_series
        
    except Exception as e:
        logger.error(f"重新生成5日收益率因子失败: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    print("修正版5日收益率因子重新生成")
    print("=" * 60)
    print("此脚本将:")
    print("1. 用修正后的算法重新计算5日收益率")
    print("2. 备份原错误数据")
    print("3. 保存修正后的正确数据")
    print("4. 更新元数据")
    print("=" * 60)
    print("预计耗时: 5-10分钟")
    print("=" * 60)
    
    try:
        factor_file, metadata_file, factor_data = regenerate_returns_5d_factor_corrected()
        
        print("\n" + "=" * 60)
        print("成功: 修正版5日收益率因子重新生成完成!")
        print("=" * 60)
        print(f"数据文件: {factor_file}")
        print(f"元数据文件: {metadata_file}")
        print(f"数据点数: {len(factor_data):,}")
        print("=" * 60)
        print("现在可以安全地使用修正后的5日收益率因子了！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n失败: 重新生成过程出现错误: {e}")
        sys.exit(1)