#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试协调器模块

协调实验因子与现有测试框架的集成，提供专门的测试管理
强制使用factors.tester的SingleFactorTestPipeline
"""

import pandas as pd
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
from pathlib import Path
import json

from factors.tester import SingleFactorTestPipeline, TestResult, ResultManager
from factors.tester.core import DataManager
from .factor_registry import ExperimentalFactorRegistry, FactorStatus
from .calculation_engine import CalculationResult
from config import get_config

logger = logging.getLogger(__name__)


class ExperimentalTestConfig:
    """实验因子测试配置"""
    
    DEFAULT_CONFIG = {
        # 测试参数
        'group_nums': 10,
        'outlier_method': 'IQR',
        'outlier_param': 5,
        'normalization_method': 'zscore',
        'backtest_type': 'all',
        'ic_decay_periods': 20,
        'turnover_cost_rate': 0.0015,
        
        # 实验专用参数
        'auto_promote_threshold': {
            'ic_mean': 0.05,
            'icir': 0.5,
            'sharpe': 1.0
        },
        'auto_archive_threshold': {
            'ic_mean': -0.02,
            'icir': -0.3
        },
        'test_timeout_minutes': 30,
        'parallel_test_limit': 3
    }
    
    def __init__(self, config_override: Dict[str, Any] = None):
        self.config = self.DEFAULT_CONFIG.copy()
        if config_override:
            self._deep_update(self.config, config_override)
    
    def _deep_update(self, base_dict: dict, update_dict: dict):
        """深度更新字典"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get(self, key: str = None, default=None):
        """获取配置值"""
        if key is None:
            return self.config.copy()
        
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value


class TestCoordinator:
    """
    测试协调器
    
    功能：
    1. 协调实验因子与现有测试框架的集成
    2. 自动化测试流程和状态管理
    3. 测试结果分析和自动评估
    4. 批量测试和性能监控
    """
    
    def __init__(self, registry: Optional[ExperimentalFactorRegistry] = None,
                 config_override: Dict[str, Any] = None):
        """
        初始化测试协调器
        
        Parameters:
        -----------
        registry : ExperimentalFactorRegistry, optional
            因子注册表实例
        config_override : Dict[str, Any], optional
            测试配置覆盖
        """
        self.registry = registry or ExperimentalFactorRegistry()
        self.config = ExperimentalTestConfig(config_override)
        
        # 初始化测试管道
        self.test_pipeline = SingleFactorTestPipeline()
        self.result_manager = ResultManager()
        
        # 测试路径
        try:
            base_path = Path(get_config('main.paths.single_factor_test'))
            self.test_results_path = base_path / "experimental_lab"
        except:
            self.test_results_path = Path("data/test_results/experimental_lab")
        
        self.test_results_path.mkdir(parents=True, exist_ok=True)
        
        # 测试统计
        self.test_stats = {
            'total_tests': 0,
            'successful_tests': 0,
            'failed_tests': 0,
            'auto_promoted': 0,
            'auto_archived': 0
        }
        
        logger.info(f"测试协调器初始化完成，结果路径: {self.test_results_path}")
    
    def test_factor(self, factor_name: str, calculation_result: CalculationResult = None,
                   test_params: Dict[str, Any] = None, auto_evaluate: bool = True) -> TestResult:
        """
        测试单个实验因子
        
        Parameters:
        -----------
        factor_name : str
            因子名称
        calculation_result : CalculationResult, optional
            预计算的因子数据
        test_params : Dict[str, Any], optional
            测试参数覆盖
        auto_evaluate : bool
            是否自动评估并更新状态
            
        Returns:
        --------
        TestResult
            测试结果
        """
        logger.info(f"开始测试实验因子: {factor_name}")
        
        try:
            # 获取因子信息
            factor = self.registry.get_factor(factor_name)
            if not factor:
                raise ValueError(f"因子 {factor_name} 未在注册表中找到")
            
            # 更新因子状态为测试中
            self.registry.update_factor_status(factor_name, FactorStatus.TESTING)
            
            # 准备测试参数
            final_test_params = self.config.get()
            if test_params:
                final_test_params.update(test_params)
            
            # 准备因子数据
            if calculation_result and calculation_result.success:
                factor_data = calculation_result.factor_data
            else:
                # 如果没有提供计算结果，需要重新计算
                logger.warning(f"未提供因子 {factor_name} 的计算结果，将跳过测试")
                raise ValueError("需要有效的因子计算结果才能进行测试")
            
            # 执行测试
            test_result = self._execute_test(
                factor_name=factor_name,
                factor_data=factor_data,
                test_params=final_test_params
            )
            
            # 保存测试结果到实验专用路径
            self._save_experimental_test_result(factor_name, test_result)
            
            # 更新注册表中的测试结果
            self._update_registry_test_results(factor_name, test_result)
            
            # 自动评估
            if auto_evaluate:
                self._auto_evaluate_factor(factor_name, test_result)
            else:
                # 只更新状态为已测试
                self.registry.update_factor_status(factor_name, FactorStatus.TESTED)
            
            # 更新统计
            self.test_stats['total_tests'] += 1
            self.test_stats['successful_tests'] += 1
            
            logger.info(f"因子 {factor_name} 测试完成")
            return test_result
            
        except Exception as e:
            logger.error(f"测试因子 {factor_name} 失败: {e}")
            
            # 更新状态为失败
            self.registry.update_factor_status(
                factor_name, FactorStatus.FAILED, 
                f"测试失败: {str(e)}"
            )
            
            # 更新统计
            self.test_stats['total_tests'] += 1
            self.test_stats['failed_tests'] += 1
            
            raise
    
    def batch_test_factors(self, factor_names: List[str] = None,
                          status_filter: List[FactorStatus] = None,
                          test_params: Dict[str, Any] = None) -> Dict[str, TestResult]:
        """
        批量测试因子
        
        Parameters:
        -----------
        factor_names : List[str], optional
            要测试的因子名称列表，为空则根据状态筛选
        status_filter : List[FactorStatus], optional
            状态筛选，默认为[CALCULATED]
        test_params : Dict[str, Any], optional
            测试参数
            
        Returns:
        --------
        Dict[str, TestResult]
            测试结果字典
        """
        # 确定要测试的因子列表
        if factor_names is None:
            if status_filter is None:
                status_filter = [FactorStatus.CALCULATED]
            
            factors_to_test = self.registry.list_factors(status=status_filter[0])
            for status in status_filter[1:]:
                factors_to_test.extend(self.registry.list_factors(status=status))
            
            factor_names = [f.name for f in factors_to_test]
        
        if not factor_names:
            logger.info("没有找到需要测试的因子")
            return {}
        
        logger.info(f"开始批量测试 {len(factor_names)} 个因子")
        
        results = {}
        successful_count = 0
        
        for i, factor_name in enumerate(factor_names, 1):
            logger.info(f"批量测试进度: {i}/{len(factor_names)} - {factor_name}")
            
            try:
                # 这里假设因子已经计算完成，实际使用时可能需要先计算
                result = self.test_factor(
                    factor_name=factor_name,
                    test_params=test_params,
                    auto_evaluate=True
                )
                results[factor_name] = result
                successful_count += 1
                
            except Exception as e:
                logger.error(f"批量测试中处理因子 {factor_name} 失败: {e}")
                results[factor_name] = None
        
        logger.info(f"批量测试完成: {successful_count}/{len(factor_names)} 个因子成功")
        return results
    
    def get_test_summary(self, factor_names: List[str] = None) -> pd.DataFrame:
        """
        获取测试汇总表
        
        Parameters:
        -----------
        factor_names : List[str], optional
            要汇总的因子名称，为空则汇总所有已测试因子
            
        Returns:
        --------
        pd.DataFrame
            测试汇总表
        """
        if factor_names is None:
            # 获取所有已测试的因子
            tested_factors = self.registry.list_factors()
            tested_factors = [f for f in tested_factors 
                            if f.status in [FactorStatus.TESTED, FactorStatus.VALIDATED, 
                                          FactorStatus.FAILED]]
            factor_names = [f.name for f in tested_factors]
        
        summary_data = []
        
        for factor_name in factor_names:
            factor = self.registry.get_factor(factor_name)
            if not factor:
                continue
            
            row = {
                'factor_name': factor_name,
                'status': factor.status.value,
                'category': factor.category,
                'test_time': factor.last_updated,
            }
            
            # 添加性能指标
            if factor.performance_metrics:
                row.update(factor.performance_metrics)
            
            summary_data.append(row)
        
        df = pd.DataFrame(summary_data)
        
        # 排序
        if not df.empty:
            df = df.sort_values('test_time', ascending=False)
        
        return df
    
    def promote_factor(self, factor_name: str, target_category: str) -> bool:
        """
        提升因子到正式repository
        
        Parameters:
        -----------
        factor_name : str
            因子名称
        target_category : str
            目标分类
            
        Returns:
        --------
        bool
            是否成功提升
        """
        try:
            factor = self.registry.get_factor(factor_name)
            if not factor:
                raise ValueError(f"因子 {factor_name} 不存在")
            
            if factor.status != FactorStatus.VALIDATED:
                logger.warning(f"因子 {factor_name} 状态为 {factor.status.value}，建议先验证通过")
            
            # 这里需要实现实际的代码生成和文件创建逻辑
            # 暂时只更新状态
            self.registry.update_factor_status(
                factor_name, FactorStatus.PROMOTED,
                f"已提升到 {target_category} 分类"
            )
            
            self.test_stats['auto_promoted'] += 1
            logger.info(f"因子 {factor_name} 成功提升到 {target_category}")
            return True
            
        except Exception as e:
            logger.error(f"提升因子 {factor_name} 失败: {e}")
            return False
    
    def archive_factor(self, factor_name: str, reason: str = "") -> bool:
        """
        归档因子
        
        Parameters:
        -----------
        factor_name : str
            因子名称
        reason : str
            归档原因
            
        Returns:
        --------
        bool
            是否成功归档
        """
        try:
            self.registry.update_factor_status(
                factor_name, FactorStatus.ARCHIVED, 
                f"归档原因: {reason}"
            )
            
            self.test_stats['auto_archived'] += 1
            logger.info(f"因子 {factor_name} 已归档: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"归档因子 {factor_name} 失败: {e}")
            return False
    
    def get_test_stats(self) -> Dict[str, Any]:
        """获取测试统计信息"""
        return self.test_stats.copy()
    
    def _execute_test(self, factor_name: str, factor_data: pd.Series,
                     test_params: Dict[str, Any]) -> TestResult:
        """执行具体的测试"""
        # 使用现有的测试管道
        # 注意：这里需要将factor_data转换为测试管道期望的格式
        
        # 创建临时因子数据文件（如果测试管道需要从文件加载）
        temp_factor_file = self.test_results_path / f"temp_{factor_name}.pkl"
        factor_data.to_pickle(temp_factor_file)
        
        try:
            # 运行测试（这里需要根据实际的SingleFactorTestPipeline接口调整）
            result = self.test_pipeline.run(
                factor_name=factor_name,
                factor_data=factor_data,
                save_result=False,  # 我们自己管理保存
                **test_params
            )
            return result
            
        finally:
            # 清理临时文件
            if temp_factor_file.exists():
                temp_factor_file.unlink()
    
    def _save_experimental_test_result(self, factor_name: str, test_result: TestResult):
        """保存实验测试结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{factor_name}_experimental_{timestamp}.pkl"
        file_path = self.test_results_path / filename
        
        self.result_manager.save(test_result, str(file_path))
        logger.debug(f"实验测试结果已保存: {file_path}")
    
    def _update_registry_test_results(self, factor_name: str, test_result: TestResult):
        """更新注册表中的测试结果"""
        # 提取关键性能指标
        performance_metrics = {}
        
        if hasattr(test_result, 'performance_metrics') and test_result.performance_metrics:
            performance_metrics.update(test_result.performance_metrics)
        
        # 从IC结果中提取指标
        if hasattr(test_result, 'ic_result') and test_result.ic_result:
            ic_result = test_result.ic_result
            performance_metrics.update({
                'ic_mean': getattr(ic_result, 'ic_mean', 0),
                'icir': getattr(ic_result, 'icir', 0),
                'rank_ic_mean': getattr(ic_result, 'rank_ic_mean', 0)
            })
        
        # 更新注册表
        test_results_summary = {
            'test_completion_time': datetime.now().isoformat(),
            'test_success': True,
            'key_metrics': performance_metrics
        }
        
        self.registry.update_test_results(
            factor_name, 
            test_results_summary, 
            performance_metrics
        )
    
    def _auto_evaluate_factor(self, factor_name: str, test_result: TestResult):
        """自动评估因子并更新状态"""
        try:
            # 获取评估阈值
            promote_threshold = self.config.get('auto_promote_threshold', {})
            archive_threshold = self.config.get('auto_archive_threshold', {})
            
            # 提取性能指标
            factor = self.registry.get_factor(factor_name)
            metrics = factor.performance_metrics
            
            if not metrics:
                logger.warning(f"因子 {factor_name} 缺少性能指标，无法自动评估")
                self.registry.update_factor_status(factor_name, FactorStatus.TESTED)
                return
            
            # 检查是否达到提升标准
            meets_promote_criteria = True
            for metric, threshold in promote_threshold.items():
                if metric in metrics:
                    if metrics[metric] < threshold:
                        meets_promote_criteria = False
                        break
            
            if meets_promote_criteria:
                self.registry.update_factor_status(
                    factor_name, FactorStatus.VALIDATED,
                    "自动评估：达到提升标准"
                )
                logger.info(f"因子 {factor_name} 自动验证通过")
                return
            
            # 检查是否需要归档
            should_archive = False
            for metric, threshold in archive_threshold.items():
                if metric in metrics:
                    if metrics[metric] < threshold:
                        should_archive = True
                        break
            
            if should_archive:
                self.registry.update_factor_status(
                    factor_name, FactorStatus.FAILED,
                    "自动评估：性能不达标"
                )
                logger.info(f"因子 {factor_name} 自动标记为失败")
            else:
                # 既不提升也不归档，标记为已测试
                self.registry.update_factor_status(
                    factor_name, FactorStatus.TESTED,
                    "自动评估：等待人工评估"
                )
                logger.info(f"因子 {factor_name} 等待人工评估")
        
        except Exception as e:
            logger.error(f"自动评估因子 {factor_name} 失败: {e}")
            self.registry.update_factor_status(factor_name, FactorStatus.TESTED)


if __name__ == "__main__":
    # 测试代码
    from .factor_registry import ExperimentalFactorRegistry
    from .calculation_engine import CalculationResult
    import numpy as np
    
    # 创建测试协调器
    coordinator = TestCoordinator()
    
    # 创建测试数据
    test_factor_data = pd.Series(
        data=np.random.randn(1000),
        index=pd.MultiIndex.from_product(
            [pd.date_range('2024-01-01', periods=100), 
             ['000001', '000002', '000003', '300001', '600000']],
            names=['TradingDates', 'StockCodes']
        )
    )
    
    test_calc_result = CalculationResult(
        factor_name="test_experimental_factor",
        factor_data=test_factor_data,
        success=True
    )
    
    print("测试协调器创建成功")
    print(f"测试统计: {coordinator.get_test_stats()}")