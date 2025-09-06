#!/usr/bin/env python3
"""
自定义因子计算示例
演示如何使用factors模块快速实现新因子
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入自定义因子
from factors.generator.mixed.custom_mixed_factors import (
    CashflowEfficiencyRatio, 
    create_cashflow_efficiency_ratio,
    register_factor_metadata
)

# 导入系统组件
from config import get_config
from data.fetcher.data_fetcher import DataFetcherManager
from factors.meta import get_factor_registry

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CustomFactorCalculator:
    """自定义因子计算器"""
    
    def __init__(self):
        self.data_fetcher = DataFetcherManager()
        logger.info("自定义因子计算器初始化完成")
    
    def prepare_data(self, start_date='2020-01-01', end_date='2023-12-31'):
        """
        准备计算所需数据
        
        Parameters
        ----------
        start_date : str
            开始日期
        end_date : str  
            结束日期
            
        Returns
        -------
        Dict[str, pd.DataFrame]
            包含所需数据的字典
        """
        logger.info(f"准备数据: {start_date} 到 {end_date}")
        
        try:
            # 获取财务数据
            logger.info("获取财务数据...")
            financial_tables = self.data_fetcher.fetch_data(
                'stock', 'financial',
                begin_date=int(start_date.replace('-', '')),
                end_date=int(end_date.replace('-', ''))
            )
            
            # 处理财务数据字典，合并需要的字段
            if isinstance(financial_tables, dict):
                # 从利润表(lrb)获取财务费用
                lrb_data = financial_tables.get('lrb', pd.DataFrame())
                # 从现金流量表(xjlb)获取销售收现和折旧
                xjlb_data = financial_tables.get('xjlb', pd.DataFrame())
                
                # 合并需要的字段到一个DataFrame
                financial_data = pd.DataFrame()
                
                if not lrb_data.empty:
                    fin_expense_cols = [col for col in lrb_data.columns if 'FIN_EXP' in col or '财务费用' in col]
                    if fin_expense_cols:
                        financial_data['FIN_EXP_CS'] = lrb_data[fin_expense_cols[0]]
                        
                if not xjlb_data.empty:
                    # 查找折旧相关字段
                    depr_cols = [col for col in xjlb_data.columns if 'DEPR' in col or '折旧' in col]
                    cash_cols = [col for col in xjlb_data.columns if 'CASH_RECP_SG' in col or '销售商品' in col]
                    
                    if depr_cols:
                        financial_data['DEPR_FA_COGA_DPBA'] = xjlb_data[depr_cols[0]]
                    if cash_cols:
                        financial_data['CASH_RECP_SG_AND_RS'] = xjlb_data[cash_cols[0]]
                
                logger.info(f"合并财务数据: {financial_data.shape}")
            else:
                financial_data = financial_tables
            
            # 获取BP因子数据
            logger.info("获取BP因子数据...")
            bp_data = self._load_bp_factor()
            
            logger.info(f"数据准备完成:")
            logger.info(f"  财务数据形状: {financial_data.shape}")
            if hasattr(bp_data, 'shape'):
                logger.info(f"  BP数据形状: {bp_data.shape}")
            else:
                logger.info(f"  BP数据长度: {len(bp_data) if bp_data is not None else 0}")
            
            return {
                'financial_data': financial_data,
                'bp_data': bp_data
            }
            
        except Exception as e:
            logger.error(f"数据准备失败: {e}")
            return {}
    
    def _load_bp_factor(self):
        """加载BP因子数据"""
        try:
            # 尝试从已有的因子文件加载BP
            raw_factors_path = get_config('main.paths.raw_factors')
            bp_file_path = os.path.join(raw_factors_path, 'BP.pkl')
            
            if os.path.exists(bp_file_path):
                bp_data = pd.read_pickle(bp_file_path)
                logger.info(f"从文件加载BP数据: {bp_file_path}")
                return bp_data
            
            # 如果没有现成的BP文件，从混合因子管理器计算
            logger.info("计算BP因子...")
            from factors.generator.mixed import get_mixed_factor_manager
            
            manager = get_mixed_factor_manager()
            
            # 获取所需数据
            financial_data = self.data_fetcher.fetch_data('stock', 'financial')
            market_cap = self.data_fetcher.fetch_data('market', 'market_cap')
            
            data = {
                'financial_data': financial_data,
                'market_cap': market_cap
            }
            
            bp = manager.calculate_factor('BP', data)
            return bp.to_frame('BP') if isinstance(bp, pd.Series) else bp
            
        except Exception as e:
            logger.warning(f"加载BP因子失败: {e}")
            # 返回空DataFrame，让验证函数处理
            return pd.DataFrame()
    
    def calculate_factor(self, data=None, save_result=True):
        """
        计算自定义因子
        
        Parameters
        ----------
        data : Dict, optional
            预准备的数据，如果为None则自动获取
        save_result : bool
            是否保存计算结果
            
        Returns
        -------
        pd.Series
            计算结果
        """
        try:
            # 数据准备
            if data is None:
                logger.info("自动准备数据...")
                data = self.prepare_data()
            
            if not data:
                logger.error("数据准备失败")
                return pd.Series()
            
            # 创建因子实例
            logger.info("创建因子计算器...")
            factor = create_cashflow_efficiency_ratio()
            
            # 计算因子
            logger.info("开始计算现金流效率比率因子...")
            result = factor.calculate(data)
            
            if result.empty:
                logger.error("因子计算失败")
                return pd.Series()
            
            # 保存结果
            if save_result:
                self._save_factor_result(result)
            
            # 显示统计信息
            self._print_factor_statistics(result)
            
            logger.info("✅ 自定义因子计算完成")
            return result
            
        except Exception as e:
            logger.error(f"因子计算失败: {e}")
            return pd.Series()
    
    def _save_factor_result(self, factor_data: pd.Series):
        """保存因子结果"""
        try:
            # 保存到raw_factors目录
            raw_factors_path = get_config('main.paths.raw_factors')
            os.makedirs(raw_factors_path, exist_ok=True)
            
            output_file = os.path.join(raw_factors_path, 'CashflowEfficiencyRatio.pkl')
            factor_data.to_pickle(output_file)
            
            logger.info(f"因子结果已保存: {output_file}")
            
            # 同时保存到factors目录（新的统一存储位置）
            factors_path = get_config('main.paths.factors') 
            if factors_path:
                os.makedirs(factors_path, exist_ok=True)
                factors_output_file = os.path.join(factors_path, 'CashflowEfficiencyRatio.pkl')
                factor_data.to_pickle(factors_output_file)
                logger.info(f"因子结果已保存: {factors_output_file}")
            
        except Exception as e:
            logger.error(f"保存因子结果失败: {e}")
    
    def _print_factor_statistics(self, factor_data: pd.Series):
        """打印因子统计信息"""
        try:
            print("\n" + "="*60)
            print("现金流效率比率因子 - 计算结果统计")
            print("="*60)
            
            # 基本统计
            print(f"样本总数: {len(factor_data)}")
            print(f"有效样本数: {factor_data.notna().sum()}")
            print(f"缺失率: {factor_data.isna().sum() / len(factor_data):.2%}")
            
            valid_data = factor_data.dropna()
            if len(valid_data) > 0:
                print(f"\n数据分布:")
                print(f"  均值: {valid_data.mean():.6f}")
                print(f"  标准差: {valid_data.std():.6f}")
                print(f"  中位数: {valid_data.median():.6f}")
                print(f"  最小值: {valid_data.min():.6f}")
                print(f"  最大值: {valid_data.max():.6f}")
                
                # 分位数
                print(f"\n分位数分布:")
                for q in [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]:
                    print(f"  {q*100:5.1f}%: {valid_data.quantile(q):.6f}")
                
                # 时间分布
                if isinstance(factor_data.index, pd.MultiIndex):
                    dates = factor_data.index.get_level_values(0).unique()
                    print(f"\n时间范围:")
                    print(f"  开始日期: {dates.min()}")
                    print(f"  结束日期: {dates.max()}")
                    print(f"  时间点数: {len(dates)}")
            
            print("="*60)
            
        except Exception as e:
            logger.error(f"打印统计信息失败: {e}")
    
    def test_factor_with_pipeline(self, save_test_result=True):
        """使用测试流水线测试因子"""
        try:
            logger.info("使用测试流水线测试因子...")
            
            from factors.tester.core.pipeline import SingleFactorTestPipeline
            
            # 创建测试流水线
            pipeline = SingleFactorTestPipeline()
            
            # 运行测试
            result = pipeline.run(
                factor_name='CashflowEfficiencyRatio',
                save_result=save_test_result,
                factor_version='raw',  # 使用原始版本
                group_nums=10,         # 10分组
                begin_date='2020-01-01'
            )
            
            if result and not result.errors:
                logger.info("✅ 因子测试通过")
                
                # 打印关键指标
                if result.performance_metrics:
                    print(f"\n因子测试结果:")
                    print(f"  IC均值: {result.performance_metrics.get('ic_mean', 'N/A')}")
                    print(f"  IC标准差: {result.performance_metrics.get('ic_std', 'N/A')}")
                    print(f"  ICIR: {result.performance_metrics.get('ic_ir', 'N/A')}")
                    print(f"  年化收益: {result.performance_metrics.get('annual_return', 'N/A')}")
                
                return result
            else:
                logger.error(f"因子测试失败: {result.errors if result else 'Unknown error'}")
                return None
                
        except Exception as e:
            logger.error(f"因子测试失败: {e}")
            return None


def main():
    """主函数，演示完整流程"""
    print("🚀 开始计算自定义因子: 现金流效率比率")
    print("公式: ((FIN_EXP_CS + DEPR_FA_COGA_DPBA) / CASH_RECP_SG_AND_RS) / BP")
    
    try:
        # 1. 注册因子元数据
        print("\n1. 注册因子元数据...")
        register_factor_metadata()
        
        # 2. 创建计算器
        print("\n2. 初始化计算器...")
        calculator = CustomFactorCalculator()
        
        # 3. 计算因子
        print("\n3. 计算因子...")
        factor_result = calculator.calculate_factor(save_result=True)
        
        if factor_result.empty:
            print("❌ 因子计算失败")
            return
        
        # 4. 测试因子
        print("\n4. 测试因子性能...")
        test_result = calculator.test_factor_with_pipeline(save_test_result=True)
        
        # 5. 查看因子注册信息
        print("\n5. 查看因子注册信息...")
        try:
            registry = get_factor_registry()
            metadata = registry.get_factor('CashflowEfficiencyRatio')
            if metadata:
                print(f"  因子名称: {metadata.name}")
                print(f"  因子类型: {metadata.type.value if metadata.type else 'N/A'}")
                print(f"  因子描述: {metadata.description}")
                print(f"  计算公式: {metadata.formula}")
        except Exception as e:
            print(f"查看注册信息失败: {e}")
        
        print("\n✅ 自定义因子开发完成！")
        print(f"因子文件保存位置:")
        print(f"  - {get_config('main.paths.raw_factors')}/CashflowEfficiencyRatio.pkl")
        print(f"  - {get_config('main.paths.factors')}/CashflowEfficiencyRatio.pkl")
        
        # 使用建议
        print(f"\n📖 后续使用建议:")
        print(f"1. 生成正交化版本:")
        print(f"   python generate_orthogonal_factors.py --factors CashflowEfficiencyRatio")
        print(f"2. 查看因子详情:")
        print(f"   python factor_manager.py show CashflowEfficiencyRatio")
        print(f"3. 在组合中使用:")
        print(f"   从factors目录加载并在投资策略中应用")
        
    except Exception as e:
        logger.error(f"主程序执行失败: {e}")
        print(f"❌ 执行失败: {e}")


if __name__ == "__main__":
    main()