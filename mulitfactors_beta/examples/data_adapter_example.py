#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据适配器使用示例
演示如何使用数据适配器加载原始数据并计算因子
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from factors.utils import DataAdapter, FactorCalculator

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demo_data_loading():
    """演示数据加载和准备"""
    print("\n" + "="*60)
    print("数据加载和准备示例")
    print("="*60)
    
    # 设置数据路径
    data_path = Path(r"E:\Documents\PythonProject\StockProject\StockData")
    
    try:
        # 使用数据适配器加载所有数据
        print("\n1. 使用DataAdapter加载数据...")
        prepared_data = DataAdapter.load_and_prepare_data(data_path)
        
        print("\n2. 检查加载的数据:")
        for key, value in prepared_data.items():
            if isinstance(value, pd.DataFrame):
                print(f"   - {key}: DataFrame 形状 {value.shape}")
                if isinstance(value.index, pd.MultiIndex):
                    print(f"     索引: {value.index.names}")
            elif isinstance(value, pd.Series):
                print(f"   - {key}: Series 长度 {len(value)}")
                if isinstance(value.index, pd.MultiIndex):
                    print(f"     索引: {value.index.names}")
            elif isinstance(value, pd.DatetimeIndex):
                print(f"   - {key}: {len(value)} 个交易日")
                print(f"     日期范围: {value[0]} 到 {value[-1]}")
        
        return prepared_data
        
    except Exception as e:
        print(f"\n[FAIL] 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def demo_bp_factor_calculation(prepared_data):
    """演示BP因子计算"""
    print("\n" + "="*60)
    print("BP因子计算示例")
    print("="*60)
    
    if not prepared_data:
        print("[FAIL] 没有准备好的数据")
        return
    
    try:
        # 创建因子计算器
        print("\n1. 创建因子计算器...")
        calculator = FactorCalculator()
        
        # 检查必要的数据
        financial_data = prepared_data.get('financial_data')
        market_cap = prepared_data.get('market_cap')
        release_dates = prepared_data.get('release_dates')
        trading_dates = prepared_data.get('trading_dates')
        
        if financial_data is None or market_cap is None:
            print("[FAIL] 缺少必要的数据")
            return
        
        print("\n2. 计算BP因子...")
        print(f"   财务数据形状: {financial_data.shape}")
        print(f"   市值数据长度: {len(market_cap)}")
        
        # 计算BP因子
        results = calculator.calculate_factors(
            factor_names=['BP'],
            financial_data=financial_data,
            market_cap=market_cap,
            release_dates=release_dates,
            trading_dates=trading_dates
        )
        
        if 'BP' in results:
            bp_factor = results['BP']
            print(f"\n3. BP因子计算成功!")
            print(f"   因子数据点: {len(bp_factor)}")
            print(f"   非空值: {bp_factor.notna().sum()}")
            print(f"   数值范围: [{bp_factor.min():.6f}, {bp_factor.max():.6f}]")
            
            # 查看2024年1月的数据样本
            if isinstance(bp_factor.index, pd.MultiIndex):
                dates = bp_factor.index.get_level_values(0)
                mask = (dates >= pd.Timestamp('2024-01-01')) & (dates <= pd.Timestamp('2024-01-31'))
                bp_jan = bp_factor[mask]
                
                if len(bp_jan) > 0:
                    print(f"\n4. 2024年1月数据样本:")
                    print(f"   数据点数: {len(bp_jan)}")
                    print(f"   均值: {bp_jan.mean():.6f}")
                    
                    # 显示前5个数据点
                    print("\n   前5个数据点:")
                    for i in range(min(5, len(bp_jan))):
                        idx = bp_jan.index[i]
                        value = bp_jan.iloc[i]
                        print(f"   {idx[0].strftime('%Y-%m-%d')} {idx[1]}: {value:.6f}")
        else:
            print("\n[FAIL] BP因子计算失败")
            
    except Exception as e:
        print(f"\n[FAIL] 因子计算出错: {e}")
        import traceback
        traceback.print_exc()


def demo_manual_data_preparation():
    """演示手动数据准备（当DataAdapter遇到问题时）"""
    print("\n" + "="*60)
    print("手动数据准备示例")
    print("="*60)
    
    data_path = Path(r"E:\Documents\PythonProject\StockProject\StockData")
    
    try:
        # 1. 加载原始数据
        print("\n1. 加载原始数据文件...")
        lrb = pd.read_pickle(data_path / 'lrb.pkl')
        xjlb = pd.read_pickle(data_path / 'xjlb.pkl')
        fzb = pd.read_pickle(data_path / 'fzb.pkl')
        price = pd.read_pickle(data_path / 'Price.pkl')
        market_cap = pd.read_pickle(data_path / 'MarketCap.pkl')
        
        print(f"   利润表: {lrb.shape}")
        print(f"   现金流量表: {xjlb.shape}")
        print(f"   资产负债表: {fzb.shape}")
        print(f"   价格数据: {price.shape}")
        print(f"   市值数据: {market_cap.shape}")
        
        # 2. 手动准备财务数据
        print("\n2. 准备财务数据...")
        financial_data, release_dates = DataAdapter.prepare_financial_data(lrb, xjlb, fzb)
        
        # 3. 手动准备价格数据
        print("\n3. 准备价格数据...")
        price_data = DataAdapter.prepare_price_data(price)
        
        # 4. 手动准备市值数据
        print("\n4. 准备市值数据...")
        market_cap_series = DataAdapter.prepare_market_cap(market_cap)
        
        # 5. 提取交易日期
        print("\n5. 提取交易日期...")
        trading_dates = DataAdapter.extract_trading_dates(price_data)
        
        print("\n[OK] 数据准备完成!")
        
        return {
            'financial_data': financial_data,
            'release_dates': release_dates,
            'price_data': price_data,
            'market_cap': market_cap_series,
            'trading_dates': trading_dates
        }
        
    except Exception as e:
        print(f"\n[FAIL] 手动数据准备失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主函数"""
    print("数据适配器和因子计算演示")
    print("="*80)
    
    # 方法1：使用DataAdapter自动加载
    print("\n方法1：使用DataAdapter自动加载数据")
    prepared_data = demo_data_loading()
    
    if prepared_data:
        # 计算BP因子
        demo_bp_factor_calculation(prepared_data)
    else:
        # 方法2：手动准备数据
        print("\n方法2：手动准备数据")
        prepared_data = demo_manual_data_preparation()
        
        if prepared_data:
            demo_bp_factor_calculation(prepared_data)
    
    print("\n" + "="*80)
    print("演示完成!")
    
    print("\n[TIP] 关键要点:")
    print("1. DataAdapter可以自动加载和转换数据格式")
    print("2. 财务数据需要转换为MultiIndex格式")
    print("3. 发布日期信息对增量更新很重要")
    print("4. 市值数据需要与价格数据的索引对齐")


if __name__ == "__main__":
    main()