#!/usr/bin/env python3
"""
完整的自定义因子生成、分析和正交化流程
1. 生成 CashflowEfficiencyRatio 因子
2. 进行单因子分析
3. 对行业和对数市值进行正交化处理
4. 按照配置存储所有结果
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
    create_cashflow_efficiency_ratio,
    register_factor_metadata
)

# 导入系统组件
from config import get_config, get_config
from data.fetcher.data_fetcher import DataFetcherManager
from factors.tester.core.pipeline import SingleFactorTestPipeline
from factors.meta import get_factor_registry, FactorType, NeutralizationCategory

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CustomFactorWorkflow:
    """自定义因子完整工作流"""
    
    def __init__(self):
        self.factor_name = "CashflowEfficiencyRatio"
        self.data_fetcher = DataFetcherManager()
        self.pipeline = SingleFactorTestPipeline()
        self.factor_registry = get_factor_registry()
        
        # 获取配置路径
        self.raw_factors_path = get_config('main.paths.raw_factors')
        self.factors_path = get_config('main.paths.factors')
        self.orth_factors_path = get_config('main.paths.orthogonalization_factors')
        
        logger.info("自定义因子工作流初始化完成")
    
    def step1_generate_factor(self, start_date='2020-01-01', end_date='2023-12-31'):
        """
        步骤1：生成自定义因子
        
        Returns
        -------
        pd.Series
            生成的因子数据
        """
        print("\n" + "="*60)
        print("📊 步骤1：生成 CashflowEfficiencyRatio 因子")
        print("="*60)
        
        try:
            # 1.1 注册因子元数据
            logger.info("注册因子元数据...")
            register_factor_metadata()
            
            # 1.2 准备数据
            logger.info(f"准备数据: {start_date} 到 {end_date}")
            
            # 获取财务数据
            financial_data = self.data_fetcher.get_data(
                'stock',
                start_date=start_date,
                end_date=end_date,
                fields=[
                    'FIN_EXP_CS',           # 财务费用
                    'DEPR_FA_COGA_DPBA',    # 折旧费用
                    'CASH_RECP_SG_AND_RS',  # 销售商品收现
                ]
            )
            
            # 获取BP因子数据
            bp_data = self._load_bp_factor_data()
            
            if financial_data.empty or bp_data.empty:
                logger.error("必要数据获取失败")
                return pd.Series()
            
            logger.info(f"数据准备完成 - 财务数据: {financial_data.shape}, BP数据: {len(bp_data)}")
            
            # 1.3 计算因子
            logger.info("计算自定义因子...")
            
            data = {
                'financial_data': financial_data,
                'bp_data': bp_data.to_frame('BP') if isinstance(bp_data, pd.Series) else bp_data
            }
            
            factor = create_cashflow_efficiency_ratio()
            result = factor.calculate(data)
            
            if result.empty:
                logger.error("因子计算失败")
                return pd.Series()
            
            # 1.4 保存因子到配置路径
            logger.info("保存因子到存储路径...")
            
            # 保存到raw_factors（原始因子）
            raw_file = os.path.join(self.raw_factors_path, f"{self.factor_name}.pkl")
            result.to_pickle(raw_file)
            logger.info(f"原始因子已保存: {raw_file}")
            
            # 保存到factors（统一存储）
            factors_file = os.path.join(self.factors_path, f"{self.factor_name}.pkl")
            result.to_pickle(factors_file)
            logger.info(f"因子已保存: {factors_file}")
            
            # 1.5 显示因子统计
            self._print_factor_statistics(result)
            
            print(f"✅ 因子生成完成！生成 {result.notna().sum()} 个有效数据点")
            
            return result
            
        except Exception as e:
            logger.error(f"因子生成失败: {e}")
            print(f"❌ 因子生成失败: {e}")
            return pd.Series()
    
    def step2_single_factor_analysis(self):
        """
        步骤2：进行单因子分析
        
        Returns
        -------
        TestResult
            测试结果
        """
        print("\n" + "="*60)
        print("🔍 步骤2：单因子分析测试")
        print("="*60)
        
        try:
            logger.info("开始单因子分析...")
            
            # 配置测试参数
            test_config = {
                'factor_version': 'raw',          # 使用原始因子
                'group_nums': 10,                 # 10分组
                'begin_date': '2020-01-01',       # 测试开始日期
                'end_date': '2023-12-31',         # 测试结束日期
                'netral_base': True,              # 使用基准中性化
                'use_industry': True,             # 使用行业中性化
                'backtest_type': 'daily',         # 日频回测
                'back_test_trading_price': 'o2o'  # 开盘到开盘价格
            }
            
            # 运行测试
            result = self.pipeline.run(
                factor_name=self.factor_name,
                save_result=True,  # 保存测试结果
                **test_config
            )
            
            if result and not result.errors:
                print("✅ 单因子分析完成！")
                
                # 显示关键指标
                if result.performance_metrics:
                    print(f"\n📈 关键性能指标:")
                    metrics = result.performance_metrics
                    print(f"  IC均值: {metrics.get('ic_mean', 'N/A'):.6f}")
                    print(f"  IC标准差: {metrics.get('ic_std', 'N/A'):.6f}")
                    print(f"  ICIR: {metrics.get('ic_ir', 'N/A'):.6f}")
                    print(f"  年化收益: {metrics.get('annual_return', 'N/A'):.4f}")
                    print(f"  夏普比率: {metrics.get('sharpe_ratio', 'N/A'):.4f}")
                    print(f"  最大回撤: {metrics.get('max_drawdown', 'N/A'):.4f}")
                
                return result
            else:
                error_msg = result.errors if result else "未知错误"
                logger.error(f"单因子分析失败: {error_msg}")
                print(f"❌ 单因子分析失败: {error_msg}")
                return None
                
        except Exception as e:
            logger.error(f"单因子分析过程失败: {e}")
            print(f"❌ 单因子分析过程失败: {e}")
            return None
    
    def step3_orthogonalize_factor(self):
        """
        步骤3：对行业因子和对数市值因子进行正交化处理
        
        Returns
        -------
        bool
            是否成功
        """
        print("\n" + "="*60)
        print("🔀 步骤3：正交化处理（回归行业和对数市值）")
        print("="*60)
        
        try:
            # 使用现有的正交化生成器
            from generate_orthogonal_factors import OrthogonalFactorGenerator
            
            logger.info("初始化正交化生成器...")
            generator = OrthogonalFactorGenerator()
            
            # 准备控制变量
            logger.info("准备控制变量...")
            control_vars = generator.prepare_control_variables()
            
            if control_vars.empty:
                logger.error("控制变量准备失败")
                print("❌ 控制变量准备失败")
                return False
            
            logger.info(f"控制变量准备完成: {control_vars.shape}")
            print(f"📊 控制变量包含: {list(control_vars.columns)}")
            
            # 执行正交化
            logger.info(f"对因子 {self.factor_name} 执行正交化...")
            result = generator.generate_single_factor(
                factor_name=self.factor_name,
                control_vars=control_vars,
                force=True  # 强制执行，即使分类为跳过
            )
            
            if result['status'] == 'success':
                print("✅ 因子正交化完成！")
                print(f"📁 正交化因子保存至: {result['output_file']}")
                
                # 显示统计信息
                if result.get('stats'):
                    stats = result['stats']
                    print(f"\n📊 正交化统计:")
                    print(f"  原始样本数: {stats.get('original_count', 'N/A')}")
                    print(f"  有效样本数: {stats.get('valid_count', 'N/A')}")
                    print(f"  正交化样本数: {stats.get('orthogonal_count', 'N/A')}")
                    print(f"  使用方法: {stats.get('method_used', 'N/A')}")
                
                # 更新因子注册表
                logger.info("更新因子注册表...")
                self.factor_registry.mark_orthogonalized(
                    name=self.factor_name,
                    orthogonal_path=result['output_file'],
                    control_factors=['LogMarketCap', 'BP', 'industry'],
                    method='OLS'
                )
                
                return True
                
            else:
                logger.error(f"正交化失败: {result['message']}")
                print(f"❌ 正交化失败: {result['message']}")
                return False
                
        except Exception as e:
            logger.error(f"正交化处理失败: {e}")
            print(f"❌ 正交化处理失败: {e}")
            return False
    
    def step4_test_orthogonal_factor(self):
        """
        步骤4：测试正交化后的因子
        
        Returns
        -------
        TestResult
            测试结果
        """
        print("\n" + "="*60)
        print("🧪 步骤4：测试正交化因子")
        print("="*60)
        
        try:
            logger.info("测试正交化因子...")
            
            # 配置测试参数
            test_config = {
                'factor_version': 'orthogonal',   # 使用正交化因子
                'group_nums': 10,
                'begin_date': '2020-01-01',
                'netral_base': False,             # 正交化因子不需要再次中性化
                'use_industry': False,            # 已经中性化过了
                'backtest_type': 'daily'
            }
            
            # 运行测试
            result = self.pipeline.run(
                factor_name=self.factor_name,
                save_result=True,
                **test_config
            )
            
            if result and not result.errors:
                print("✅ 正交化因子测试完成！")
                
                # 显示关键指标
                if result.performance_metrics:
                    print(f"\n📈 正交化因子性能指标:")
                    metrics = result.performance_metrics
                    print(f"  IC均值: {metrics.get('ic_mean', 'N/A'):.6f}")
                    print(f"  ICIR: {metrics.get('ic_ir', 'N/A'):.6f}")
                    print(f"  年化收益: {metrics.get('annual_return', 'N/A'):.4f}")
                    print(f"  夏普比率: {metrics.get('sharpe_ratio', 'N/A'):.4f}")
                
                return result
            else:
                error_msg = result.errors if result else "未知错误"
                print(f"❌ 正交化因子测试失败: {error_msg}")
                return None
                
        except Exception as e:
            logger.error(f"正交化因子测试失败: {e}")
            print(f"❌ 正交化因子测试失败: {e}")
            return None
    
    def step5_compare_results(self, raw_result, orth_result):
        """
        步骤5：对比原始因子和正交化因子的表现
        
        Parameters
        ----------
        raw_result : TestResult
            原始因子测试结果
        orth_result : TestResult
            正交化因子测试结果
        """
        print("\n" + "="*60)
        print("📊 步骤5：原始vs正交化因子对比分析")
        print("="*60)
        
        if not raw_result or not orth_result:
            print("❌ 缺少测试结果，无法进行对比")
            return
        
        try:
            raw_metrics = raw_result.performance_metrics or {}
            orth_metrics = orth_result.performance_metrics or {}
            
            print(f"\n{'指标':<15} {'原始因子':<12} {'正交化因子':<12} {'变化':<10}")
            print("-" * 55)
            
            metrics_to_compare = [
                ('IC均值', 'ic_mean'),
                ('IC标准差', 'ic_std'), 
                ('ICIR', 'ic_ir'),
                ('年化收益', 'annual_return'),
                ('夏普比率', 'sharpe_ratio'),
                ('最大回撤', 'max_drawdown')
            ]
            
            for display_name, key in metrics_to_compare:
                raw_val = raw_metrics.get(key, 0)
                orth_val = orth_metrics.get(key, 0)
                
                if raw_val != 0:
                    change_pct = (orth_val - raw_val) / abs(raw_val) * 100
                    change_str = f"{change_pct:+.1f}%"
                else:
                    change_str = "N/A"
                
                print(f"{display_name:<15} {raw_val:<12.4f} {orth_val:<12.4f} {change_str:<10}")
            
            # 结论
            print(f"\n📝 分析结论:")
            
            ic_raw = raw_metrics.get('ic_mean', 0)
            ic_orth = orth_metrics.get('ic_mean', 0)
            
            if abs(ic_orth) > abs(ic_raw):
                print("  ✅ 正交化后IC绝对值提升，去除噪音效果良好")
            else:
                print("  ⚠️  正交化后IC绝对值下降，可能去除了有效信号")
            
            icir_raw = raw_metrics.get('ic_ir', 0)
            icir_orth = orth_metrics.get('ic_ir', 0)
            
            if icir_orth > icir_raw:
                print("  ✅ 正交化后ICIR提升，稳定性改善")
            else:
                print("  ⚠️  正交化后ICIR下降，稳定性可能受影响")
            
        except Exception as e:
            logger.error(f"对比分析失败: {e}")
            print(f"❌ 对比分析失败: {e}")
    
    def _load_bp_factor_data(self):
        """加载BP因子数据"""
        try:
            # 尝试从已有文件加载
            bp_file = os.path.join(self.raw_factors_path, 'BP.pkl')
            if os.path.exists(bp_file):
                bp_data = pd.read_pickle(bp_file)
                logger.info(f"从文件加载BP数据: {bp_file}")
                return bp_data
            
            # 如果文件不存在，从混合因子管理器获取
            logger.info("计算BP因子数据...")
            from factors.generator.mixed import get_mixed_factor_manager
            
            manager = get_mixed_factor_manager()
            
            # 获取必要数据
            financial_data = self.data_fetcher.get_data(
                'stock',
                fields=['TOTAL_EQUITY']
            )
            
            market_cap = self.data_fetcher.get_data(
                'market',
                fields=['market_cap']
            )
            
            data = {
                'financial_data': financial_data,
                'market_cap': market_cap
            }
            
            bp = manager.calculate_factor('BP', data)
            return bp
            
        except Exception as e:
            logger.error(f"加载BP因子数据失败: {e}")
            return pd.Series()
    
    def _print_factor_statistics(self, factor_data: pd.Series):
        """打印因子统计信息"""
        try:
            valid_data = factor_data.dropna()
            if len(valid_data) == 0:
                print("  ⚠️  因子数据全为空")
                return
            
            print(f"\n📊 因子统计信息:")
            print(f"  样本总数: {len(factor_data)}")
            print(f"  有效样本: {len(valid_data)}")
            print(f"  缺失率: {factor_data.isna().sum() / len(factor_data):.2%}")
            print(f"  均值: {valid_data.mean():.6f}")
            print(f"  标准差: {valid_data.std():.6f}")
            print(f"  偏度: {valid_data.skew():.4f}")
            print(f"  峰度: {valid_data.kurtosis():.4f}")
            
            # 分位数
            quantiles = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
            print(f"  分位数分布:")
            for q in quantiles:
                print(f"    {q*100:5.1f}%: {valid_data.quantile(q):8.6f}")
            
        except Exception as e:
            logger.error(f"打印统计信息失败: {e}")
    
    def run_complete_workflow(self):
        """运行完整工作流"""
        print("🚀 开始 CashflowEfficiencyRatio 因子完整工作流")
        print("包含：生成→分析→正交化→对比")
        print("="*60)
        
        start_time = datetime.now()
        
        try:
            # 步骤1：生成因子
            factor_data = self.step1_generate_factor()
            if factor_data.empty:
                print("❌ 因子生成失败，流程终止")
                return False
            
            # 步骤2：单因子分析
            raw_result = self.step2_single_factor_analysis()
            if not raw_result:
                print("❌ 单因子分析失败，但继续执行后续步骤")
            
            # 步骤3：正交化处理
            orth_success = self.step3_orthogonalize_factor()
            if not orth_success:
                print("❌ 正交化失败，跳过后续对比")
                return False
            
            # 步骤4：测试正交化因子
            orth_result = self.step4_test_orthogonal_factor()
            
            # 步骤5：对比分析
            if raw_result and orth_result:
                self.step5_compare_results(raw_result, orth_result)
            
            # 最终总结
            end_time = datetime.now()
            duration = end_time - start_time
            
            print("\n" + "="*60)
            print("🎉 完整工作流执行完成！")
            print("="*60)
            print(f"⏱️  总耗时: {duration}")
            print(f"📁 原始因子: {self.raw_factors_path}/{self.factor_name}.pkl")
            print(f"📁 正交化因子: {self.orth_factors_path}/{self.factor_name}_orth.pkl")
            print(f"🔍 测试结果: 已保存到测试结果目录")
            print("\n🔧 后续操作建议:")
            print("1. 使用 factor_manager.py show CashflowEfficiencyRatio 查看元数据")
            print("2. 在投资策略中使用正交化版本的因子")
            print("3. 定期重新计算以获得最新数据")
            
            return True
            
        except Exception as e:
            logger.error(f"完整工作流执行失败: {e}")
            print(f"❌ 完整工作流执行失败: {e}")
            return False


def main():
    """主函数"""
    try:
        workflow = CustomFactorWorkflow()
        success = workflow.run_complete_workflow()
        
        if success:
            print("\n✅ 所有步骤执行成功！")
        else:
            print("\n❌ 部分步骤执行失败，请检查日志")
            
    except Exception as e:
        print(f"❌ 程序执行失败: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())