"""
动量因子族集成示例 - 完整的开发、测试和部署流程
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
import time
from datetime import datetime

# 导入因子模块
from factors.generator.technical.price_factors import (
    MomentumFactor, 
    MultiPeriodMomentumFactory,
    MomentumFactorBase
)
from factors.tester.core.pipeline import SingleFactorTestPipeline
from factors.tester.core.data_manager import DataManager
from factors.tester.base.test_result import TestResult
from config import get_config, config_managernfig

logger = logging.getLogger(__name__)


class MomentumFactorIntegrator:
    """动量因子族集成器 - 统一管理动量因子的生成、测试和部署"""
    
    def __init__(self, 
                 periods: List[int] = [5, 10, 20, 60, 120],
                 factor_types: List[str] = ['standard', 'residual', 'risk_adjusted']):
        """
        Parameters:
        -----------
        periods : 动量周期列表（交易日）
        factor_types : 因子类型列表
        """
        self.periods = periods
        self.factor_types = factor_types
        self.factory = MultiPeriodMomentumFactory(periods)
        self.generated_factors: Dict[str, pd.Series] = {}
        self.test_results: Dict[str, TestResult] = {}
        
    def generate_all_factors(self, 
                           price_data: pd.DataFrame,
                           save_factors: bool = True) -> Dict[str, pd.Series]:
        """
        批量生成所有类型的动量因子
        
        Parameters:
        -----------
        price_data : DataFrame with MultiIndex [TradingDates, StockCodes]
            必须包含：close, open, high, low, volume, adjfactor
        save_factors : 是否保存因子到磁盘
        
        Returns:
        --------
        Dict[str, pd.Series] : 所有生成的因子
        """
        logger.info("开始批量生成动量因子...")
        start_time = time.time()
        
        all_factors = {}
        
        # 按类型生成因子
        for factor_type in self.factor_types:
            logger.info(f"生成{factor_type}类型动量因子...")
            
            type_factors = self.factory.generate_momentum_factors(
                price_data, 
                factor_type=factor_type
            )
            
            # 添加类型前缀
            for name, factor in type_factors.items():
                full_name = f"{name}_{factor_type}" if factor_type != 'standard' else name
                all_factors[full_name] = factor
                
                if save_factors:
                    # 保存因子
                    save_path = config.get_config('main.paths.factors') / f"{full_name}.pkl"
                    factor.to_pickle(save_path)
                    logger.info(f"因子 {full_name} 已保存到 {save_path}")
        
        self.generated_factors.update(all_factors)
        
        elapsed_time = time.time() - start_time
        logger.info(f"动量因子生成完成！共生成 {len(all_factors)} 个因子，耗时 {elapsed_time:.2f} 秒")
        
        return all_factors
    
    def run_comprehensive_tests(self, 
                              price_data: pd.DataFrame,
                              factors_to_test: Optional[List[str]] = None,
                              test_config: Optional[Dict] = None) -> Dict[str, TestResult]:
        """
        对动量因子进行全面测试
        
        Parameters:
        -----------
        price_data : 价格数据
        factors_to_test : 要测试的因子名称列表，None表示测试所有因子
        test_config : 测试配置
        
        Returns:
        --------
        Dict[str, TestResult] : 测试结果
        """
        if factors_to_test is None:
            factors_to_test = list(self.generated_factors.keys())
        
        logger.info(f"开始测试 {len(factors_to_test)} 个动量因子...")
        
        # 默认测试配置
        if test_config is None:
            test_config = {
                'holding_periods': [5, 10, 20],
                'quantile_num': 10,
                'weight_method': 'equal',
                'benchmark': None,
                'neutralize_industry': True,
                'neutralize_market_cap': True
            }
        
        test_results = {}
        
        for factor_name in factors_to_test:
            if factor_name not in self.generated_factors:
                logger.warning(f"因子 {factor_name} 未找到，跳过测试")
                continue
                
            logger.info(f"测试因子: {factor_name}")
            
            try:
                # 创建测试管道
                pipeline = SingleFactorTestPipeline(
                    factor_name=factor_name,
                    **test_config
                )
                
                # 运行测试
                result = pipeline.run_test(
                    factor_data=self.generated_factors[factor_name],
                    price_data=price_data
                )
                
                test_results[factor_name] = result
                logger.info(f"因子 {factor_name} 测试完成")
                
            except Exception as e:
                logger.error(f"因子 {factor_name} 测试失败: {str(e)}")
                continue
        
        self.test_results.update(test_results)
        logger.info(f"动量因子测试完成！共测试 {len(test_results)} 个因子")
        
        return test_results
    
    def analyze_factor_performance(self) -> pd.DataFrame:
        """
        分析因子表现并生成汇总报告
        
        Returns:
        --------
        pd.DataFrame : 因子表现汇总
        """
        if not self.test_results:
            logger.warning("无测试结果，请先运行测试")
            return pd.DataFrame()
        
        logger.info("分析因子表现...")
        
        performance_metrics = []
        
        for factor_name, result in self.test_results.items():
            try:
                # 提取关键指标（假设TestResult有这些属性）
                metrics = {
                    'factor_name': factor_name,
                    'ic_mean': getattr(result, 'ic_mean', np.nan),
                    'ic_std': getattr(result, 'ic_std', np.nan),
                    'ic_ir': getattr(result, 'ic_ir', np.nan),
                    'rank_ic_mean': getattr(result, 'rank_ic_mean', np.nan),
                    'annual_return': getattr(result, 'annual_return', np.nan),
                    'annual_volatility': getattr(result, 'annual_volatility', np.nan),
                    'sharpe_ratio': getattr(result, 'sharpe_ratio', np.nan),
                    'max_drawdown': getattr(result, 'max_drawdown', np.nan)
                }
                
                # 解析因子类型和周期
                if '_' in factor_name:
                    parts = factor_name.split('_')
                    metrics['period'] = int(parts[1].replace('d', ''))
                    metrics['factor_type'] = parts[2] if len(parts) > 2 else 'standard'
                else:
                    metrics['period'] = np.nan
                    metrics['factor_type'] = 'unknown'
                
                performance_metrics.append(metrics)
                
            except Exception as e:
                logger.warning(f"分析因子 {factor_name} 时出错: {str(e)}")
                continue
        
        if not performance_metrics:
            return pd.DataFrame()
        
        df = pd.DataFrame(performance_metrics)
        
        # 按IC IR排序
        df = df.sort_values('ic_ir', ascending=False)
        
        logger.info("因子表现分析完成")
        
        return df
    
    def get_top_factors(self, 
                       metric: str = 'ic_ir',
                       top_n: int = 10) -> List[str]:
        """
        根据指定指标获取表现最好的因子
        
        Parameters:
        -----------
        metric : 评价指标
        top_n : 返回的因子数量
        
        Returns:
        --------
        List[str] : 表现最好的因子名称列表
        """
        if not self.test_results:
            logger.warning("无测试结果")
            return []
        
        performance_df = self.analyze_factor_performance()
        
        if performance_df.empty:
            return []
        
        top_factors = performance_df.head(top_n)['factor_name'].tolist()
        
        logger.info(f"基于{metric}指标，表现最好的{top_n}个因子: {top_factors}")
        
        return top_factors
    
    def save_results(self, output_dir: Optional[str] = None):
        """保存所有结果到指定目录"""
        if output_dir is None:
            output_dir = config.get_config('main.paths.results') / 'momentum_factors'
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存因子数据
        factors_dir = os.path.join(output_dir, 'factors')
        os.makedirs(factors_dir, exist_ok=True)
        
        for name, factor in self.generated_factors.items():
            factor.to_pickle(os.path.join(factors_dir, f"{name}.pkl"))
        
        # 保存测试结果
        results_dir = os.path.join(output_dir, 'test_results')
        os.makedirs(results_dir, exist_ok=True)
        
        for name, result in self.test_results.items():
            # 假设TestResult有save方法
            if hasattr(result, 'save'):
                result.save(os.path.join(results_dir, f"{name}_result.pkl"))
        
        # 保存汇总分析
        performance_df = self.analyze_factor_performance()
        if not performance_df.empty:
            performance_df.to_csv(os.path.join(output_dir, 'momentum_factor_performance.csv'), 
                                index=False)
        
        logger.info(f"所有结果已保存到 {output_dir}")


def momentum_factor_workflow_example():
    """
    完整的动量因子开发工作流示例
    """
    logger.info("开始动量因子开发工作流...")
    
    try:
        # 1. 数据准备
        logger.info("1. 准备数据...")
        data_manager = DataManager()
        price_data = data_manager.load_price_data()  # 假设这个方法存在
        
        # 检查数据完整性
        required_columns = ['close', 'open', 'high', 'low', 'volume', 'adjfactor']
        missing_columns = [col for col in required_columns if col not in price_data.columns]
        if missing_columns:
            raise ValueError(f"价格数据缺少列: {missing_columns}")
        
        # 2. 初始化集成器
        logger.info("2. 初始化动量因子集成器...")
        integrator = MomentumFactorIntegrator(
            periods=[5, 10, 20, 60, 120],
            factor_types=['standard', 'residual', 'risk_adjusted']
        )
        
        # 3. 批量生成因子
        logger.info("3. 批量生成动量因子...")
        all_factors = integrator.generate_all_factors(price_data, save_factors=True)
        
        print(f"成功生成 {len(all_factors)} 个动量因子:")
        for name in sorted(all_factors.keys()):
            factor_data = all_factors[name]
            print(f"  - {name}: {factor_data.count()} 个有效观测值")
        
        # 4. 运行测试
        logger.info("4. 运行因子测试...")
        test_results = integrator.run_comprehensive_tests(price_data)
        
        # 5. 分析结果
        logger.info("5. 分析因子表现...")
        performance_df = integrator.analyze_factor_performance()
        
        if not performance_df.empty:
            print("\n动量因子表现汇总（按IC IR排序）:")
            print(performance_df[['factor_name', 'period', 'factor_type', 
                                'ic_ir', 'sharpe_ratio', 'annual_return']].head(10))
        
        # 6. 获取最佳因子
        top_factors = integrator.get_top_factors(metric='ic_ir', top_n=5)
        print(f"\n表现最佳的5个动量因子: {top_factors}")
        
        # 7. 保存结果
        logger.info("6. 保存结果...")
        integrator.save_results()
        
        logger.info("动量因子开发工作流完成！")
        return integrator
        
    except Exception as e:
        logger.error(f"工作流执行失败: {str(e)}")
        raise


def quick_momentum_test():
    """快速动量因子测试示例"""
    logger.info("运行快速动量因子测试...")
    
    # 创建单个动量因子
    momentum_20d = MomentumFactor(window=20)
    
    # 模拟价格数据
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    stocks = [f'stock_{i:03d}' for i in range(100)]
    
    # 创建MultiIndex
    index = pd.MultiIndex.from_product([dates, stocks], 
                                     names=['TradingDates', 'StockCodes'])
    
    # 模拟价格数据
    np.random.seed(42)
    n_obs = len(index)
    
    price_data = pd.DataFrame({
        'close': 100 * (1 + np.random.normal(0, 0.02, n_obs)).cumprod(),
        'open': np.random.normal(99, 1, n_obs),
        'high': np.random.normal(101, 1, n_obs),
        'low': np.random.normal(98, 1, n_obs),
        'volume': np.random.randint(1000, 10000, n_obs),
        'adjfactor': 1.0  # 简化的复权因子
    }, index=index)
    
    # 确保价格数据合理性
    price_data['high'] = np.maximum(price_data['high'], price_data['close'])
    price_data['low'] = np.minimum(price_data['low'], price_data['close'])
    price_data['open'] = np.clip(price_data['open'], 
                                price_data['low'], price_data['high'])
    
    # 计算动量因子
    momentum_values = momentum_20d.calculate(price_data)
    
    print(f"动量因子统计信息:")
    print(f"  - 有效观测值: {momentum_values.count()}")
    print(f"  - 均值: {momentum_values.mean():.4f}")
    print(f"  - 标准差: {momentum_values.std():.4f}")
    print(f"  - 最小值: {momentum_values.min():.4f}")
    print(f"  - 最大值: {momentum_values.max():.4f}")
    
    # 检查数据质量
    nan_ratio = momentum_values.isna().mean()
    print(f"  - NaN比例: {nan_ratio:.2%}")
    
    if nan_ratio < 0.1:  # 少于10%的NaN
        print("✓ 动量因子计算成功！")
    else:
        print("⚠ 动量因子存在较多NaN值，请检查数据质量")
    
    return momentum_values


if __name__ == '__main__':
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 运行快速测试
    print("=== 快速动量因子测试 ===")
    quick_momentum_test()
    
    print("\n=== 完整工作流示例 ===")
    # 注意：完整工作流需要真实的数据，这里只演示结构
    try:
        momentum_factor_workflow_example()
    except Exception as e:
        print(f"完整工作流需要真实数据支持，当前演示结构: {str(e)}")