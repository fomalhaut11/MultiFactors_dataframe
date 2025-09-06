#!/usr/bin/env python3
"""
简化的自定义因子工作流 - 直接使用现有数据
生成因子 → 单因子分析 → 正交化处理
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from factors.generator.mixed.custom_mixed_factors import (
    create_cashflow_efficiency_ratio,
    register_factor_metadata
)
from config import get_config
from factors.tester.core.pipeline import SingleFactorTestPipeline

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_mock_financial_data():
    """创建模拟财务数据用于演示"""
    
    # 创建日期范围（季度报告日期）
    dates = pd.date_range('2020-03-31', '2023-12-31', freq='Q')
    stocks = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH', '000858.SZ']
    
    # 创建MultiIndex
    index = pd.MultiIndex.from_product([dates, stocks], names=['date', 'stock_code'])
    
    # 生成模拟数据
    np.random.seed(42)  # 确保结果可重复
    n_samples = len(index)
    
    financial_data = pd.DataFrame({
        'FIN_EXP_CS': np.random.lognormal(15, 1, n_samples),           # 财务费用
        'DEPR_FA_COGA_DPBA': np.random.lognormal(16, 1, n_samples),   # 折旧费用
        'CASH_RECP_SG_AND_RS': np.random.lognormal(18, 1, n_samples), # 销售收现
    }, index=index)
    
    logger.info(f"创建模拟财务数据: {financial_data.shape}")
    return financial_data


def create_mock_bp_data():
    """创建模拟BP数据"""
    
    # 创建交易日数据
    dates = pd.bdate_range('2020-01-01', '2023-12-31')
    stocks = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH', '000858.SZ']
    
    # 创建MultiIndex
    index = pd.MultiIndex.from_product([dates, stocks], names=['date', 'stock_code'])
    
    # 生成模拟BP数据
    np.random.seed(123)
    n_samples = len(index)
    
    # BP通常在0.2-3.0之间
    bp_data = pd.Series(
        np.random.lognormal(-0.5, 0.8, n_samples),
        index=index,
        name='BP'
    )
    
    # 确保在合理范围内
    bp_data = bp_data.clip(0.1, 5.0)
    
    logger.info(f"创建模拟BP数据: {len(bp_data)}")
    return bp_data


def step1_generate_factor():
    """步骤1：生成自定义因子"""
    print("\n" + "="*60)
    print("📊 步骤1：生成 CashflowEfficiencyRatio 因子")
    print("="*60)
    
    try:
        # 注册因子元数据
        logger.info("注册因子元数据...")
        register_factor_metadata()
        
        # 准备模拟数据
        logger.info("准备模拟数据...")
        financial_data = create_mock_financial_data()
        bp_data = create_mock_bp_data()
        
        # 组织数据
        data = {
            'financial_data': financial_data,
            'bp_data': bp_data.to_frame('BP')
        }
        
        # 创建因子实例并计算
        logger.info("计算自定义因子...")
        factor = create_cashflow_efficiency_ratio()
        result = factor.calculate(data)
        
        if result.empty:
            print("❌ 因子计算失败")
            return pd.Series()
        
        # 保存因子
        raw_factors_path = get_config('main.paths.raw_factors')
        factors_path = get_config('main.paths.factors')
        
        # 确保目录存在
        os.makedirs(raw_factors_path, exist_ok=True)
        os.makedirs(factors_path, exist_ok=True)
        
        factor_name = "CashflowEfficiencyRatio"
        
        # 保存到两个位置
        raw_file = os.path.join(raw_factors_path, f"{factor_name}.pkl")
        factors_file = os.path.join(factors_path, f"{factor_name}.pkl")
        
        result.to_pickle(raw_file)
        result.to_pickle(factors_file)
        
        print(f"✅ 因子生成完成！")
        print(f"   有效样本数: {result.notna().sum()}")
        print(f"   保存位置: {raw_file}")
        
        # 显示统计信息
        valid_data = result.dropna()
        if len(valid_data) > 0:
            print(f"\n📊 因子统计:")
            print(f"   均值: {valid_data.mean():.6f}")
            print(f"   标准差: {valid_data.std():.6f}")
            print(f"   最小值: {valid_data.min():.6f}")
            print(f"   最大值: {valid_data.max():.6f}")
        
        return result
        
    except Exception as e:
        logger.error(f"因子生成失败: {e}")
        print(f"❌ 因子生成失败: {e}")
        return pd.Series()


def step2_single_factor_test():
    """步骤2：单因子测试"""
    print("\n" + "="*60)
    print("🔍 步骤2：单因子分析测试")
    print("="*60)
    
    try:
        pipeline = SingleFactorTestPipeline()
        
        # 配置测试参数
        test_config = {
            'factor_version': 'raw',
            'group_nums': 5,  # 减少分组数加快测试
            'begin_date': '2020-01-01',
            'end_date': '2023-12-31',
            'netral_base': True,
            'use_industry': True,
            'backtest_type': 'daily'
        }
        
        print("正在运行单因子测试...")
        result = pipeline.run(
            factor_name='CashflowEfficiencyRatio',
            save_result=True,
            **test_config
        )
        
        if result and not result.errors:
            print("✅ 单因子测试完成！")
            
            if result.performance_metrics:
                metrics = result.performance_metrics
                print(f"\n📈 性能指标:")
                print(f"   IC均值: {metrics.get('ic_mean', 'N/A'):.6f}")
                print(f"   ICIR: {metrics.get('ic_ir', 'N/A'):.6f}")
                print(f"   年化收益: {metrics.get('annual_return', 'N/A'):.4f}")
                print(f"   夏普比率: {metrics.get('sharpe_ratio', 'N/A'):.4f}")
            
            return result
        else:
            error_msg = result.errors if result else "未知错误"
            print(f"❌ 单因子测试失败: {error_msg}")
            return None
            
    except Exception as e:
        logger.error(f"单因子测试失败: {e}")
        print(f"❌ 单因子测试失败: {e}")
        return None


def step3_orthogonalize():
    """步骤3：正交化处理"""
    print("\n" + "="*60)
    print("🔀 步骤3：正交化处理")
    print("="*60)
    
    try:
        print("运行正交化因子生成...")
        
        # 直接调用正交化脚本
        result = os.system('python generate_orthogonal_factors.py --factors CashflowEfficiencyRatio')
        
        if result == 0:
            print("✅ 正交化处理完成！")
            
            # 检查输出文件
            orth_path = get_config('main.paths.orthogonalization_factors')
            orth_file = os.path.join(orth_path, 'CashflowEfficiencyRatio_orth.pkl')
            
            if os.path.exists(orth_file):
                print(f"   正交化因子已保存: {orth_file}")
                
                # 读取并显示基本信息
                orth_data = pd.read_pickle(orth_file)
                print(f"   正交化样本数: {orth_data.notna().sum()}")
                
                return True
            else:
                print("⚠️  正交化文件未找到")
                return False
        else:
            print("❌ 正交化处理失败")
            return False
            
    except Exception as e:
        logger.error(f"正交化处理失败: {e}")
        print(f"❌ 正交化处理失败: {e}")
        return False


def step4_test_orthogonal():
    """步骤4：测试正交化因子"""
    print("\n" + "="*60)
    print("🧪 步骤4：测试正交化因子")
    print("="*60)
    
    try:
        pipeline = SingleFactorTestPipeline()
        
        test_config = {
            'factor_version': 'orthogonal',
            'group_nums': 5,
            'begin_date': '2020-01-01',
            'netral_base': False,  # 正交化因子不需要再次中性化
            'use_industry': False,
            'backtest_type': 'daily'
        }
        
        print("正在测试正交化因子...")
        result = pipeline.run(
            factor_name='CashflowEfficiencyRatio',
            save_result=True,
            **test_config
        )
        
        if result and not result.errors:
            print("✅ 正交化因子测试完成！")
            
            if result.performance_metrics:
                metrics = result.performance_metrics
                print(f"\n📈 正交化因子性能:")
                print(f"   IC均值: {metrics.get('ic_mean', 'N/A'):.6f}")
                print(f"   ICIR: {metrics.get('ic_ir', 'N/A'):.6f}")
                print(f"   年化收益: {metrics.get('annual_return', 'N/A'):.4f}")
            
            return result
        else:
            error_msg = result.errors if result else "未知错误"
            print(f"❌ 正交化因子测试失败: {error_msg}")
            return None
            
    except Exception as e:
        logger.error(f"正交化因子测试失败: {e}")
        print(f"❌ 正交化因子测试失败: {e}")
        return None


def main():
    """主函数 - 运行完整工作流"""
    print("🚀 CashflowEfficiencyRatio 因子完整工作流")
    print("包含：生成 → 测试 → 正交化 → 对比")
    print("="*60)
    
    start_time = datetime.now()
    
    try:
        # 步骤1：生成因子
        factor_data = step1_generate_factor()
        if factor_data.empty:
            print("❌ 工作流终止：因子生成失败")
            return 1
        
        # 步骤2：单因子测试
        raw_result = step2_single_factor_test()
        if not raw_result:
            print("⚠️  单因子测试失败，但继续执行")
        
        # 步骤3：正交化处理
        orth_success = step3_orthogonalize()
        if not orth_success:
            print("❌ 正交化失败，跳过后续测试")
            return 1
        
        # 步骤4：测试正交化因子
        orth_result = step4_test_orthogonal()
        
        # 工作流完成总结
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "="*60)
        print("🎉 工作流执行完成！")
        print("="*60)
        print(f"⏱️  总耗时: {duration}")
        
        # 文件位置总结
        print(f"\n📁 生成的文件:")
        print(f"   原始因子: {get_config('main.paths.raw_factors')}/CashflowEfficiencyRatio.pkl")
        print(f"   正交化因子: {get_config('main.paths.orthogonalization_factors')}/CashflowEfficiencyRatio_orth.pkl")
        print(f"   测试结果: 保存在测试结果目录")
        
        # 性能对比
        if raw_result and orth_result and raw_result.performance_metrics and orth_result.performance_metrics:
            print(f"\n📊 性能对比:")
            raw_ic = raw_result.performance_metrics.get('ic_mean', 0)
            orth_ic = orth_result.performance_metrics.get('ic_mean', 0)
            print(f"   原始因子IC: {raw_ic:.6f}")
            print(f"   正交化因子IC: {orth_ic:.6f}")
            
            if abs(orth_ic) > abs(raw_ic):
                print("   ✅ 正交化后IC绝对值提升")
            else:
                print("   ⚠️  正交化后IC绝对值下降")
        
        print(f"\n🔧 后续使用:")
        print("1. python factor_manager.py show CashflowEfficiencyRatio")
        print("2. 在策略中使用正交化版本的因子")
        
        return 0
        
    except Exception as e:
        logger.error(f"工作流执行失败: {e}")
        print(f"❌ 工作流执行失败: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())