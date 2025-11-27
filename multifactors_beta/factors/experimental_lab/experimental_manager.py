#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验因子统一管理器

整合所有experimental_lab子组件，提供统一的用户接口
这是experimental_lab模块的门面类，简化用户使用
"""

import pandas as pd
from typing import Dict, List, Optional, Any, Callable, Tuple
import logging
from datetime import datetime
from pathlib import Path

from .factor_registry import ExperimentalFactorRegistry, FactorStatus
from .calculation_engine import FactorCalculationEngine, CalculationResult
from .test_coordinator import TestCoordinator
from .results_tracker import ResultsTracker
from factors.tester.base import TestResult

logger = logging.getLogger(__name__)


class ExperimentalFactorManager:
    """
    实验因子统一管理器
    
    这是experimental_lab模块的主要用户接口，整合了：
    - ExperimentalFactorRegistry: 因子注册和状态管理
    - FactorCalculationEngine: 因子计算引擎
    - TestCoordinator: 测试协调器
    - ResultsTracker: 结果跟踪器
    
    功能：
    1. 新因子的完整生命周期管理
    2. 简化的API接口
    3. 自动化工作流程
    4. 与现有系统的无缝集成
    """
    
    def __init__(self, base_path: Optional[str] = None, config_override: Dict[str, Any] = None):
        """
        初始化实验因子管理器
        
        Parameters:
        -----------
        base_path : str, optional
            基础路径，所有数据将存储在此路径下
        config_override : Dict[str, Any], optional
            配置覆盖参数
        """
        # 初始化各个组件
        self.registry = ExperimentalFactorRegistry(base_path)
        self.calculation_engine = FactorCalculationEngine()
        self.test_coordinator = TestCoordinator(self.registry, config_override)
        self.results_tracker = ResultsTracker(self.registry)
        
        # 管理器统计
        self.manager_stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'started_at': datetime.now()
        }
        
        logger.info("实验因子管理器初始化完成")
    
    # ========== 因子注册相关 ==========
    
    def register_factor(self, name: str, calculation_func: Callable, 
                       description: str = "", category: str = "experimental", 
                       author: str = "", **metadata) -> bool:
        """
        注册新的实验因子
        
        Parameters:
        -----------
        name : str
            因子名称（唯一标识）
        calculation_func : Callable
            因子计算函数，必须接受context参数
        description : str
            因子描述
        category : str
            因子分类
        author : str
            作者
        **metadata : dict
            其他元数据
            
        Returns:
        --------
        bool
            是否成功注册
        """
        try:
            logger.info(f"注册新因子: {name}")
            
            # 注册到注册表
            factor = self.registry.register_factor(
                name=name,
                calculation_func=calculation_func,
                description=description,
                category=category,
                author=author,
                **metadata
            )
            
            # 更新跟踪器
            self.results_tracker.update_factor_lifecycle(
                factor_name=name,
                event_type="registration",
                timestamp=factor.created_time
            )
            
            self._update_stats(True)
            logger.info(f"因子 {name} 注册成功")
            return True
            
        except Exception as e:
            logger.error(f"注册因子 {name} 失败: {e}")
            self._update_stats(False)
            return False
    
    def list_factors(self, status: Optional[FactorStatus] = None, 
                    category: Optional[str] = None) -> pd.DataFrame:
        """
        列出因子
        
        Parameters:
        -----------
        status : FactorStatus, optional
            按状态筛选
        category : str, optional
            按分类筛选
            
        Returns:
        --------
        pd.DataFrame
            因子列表
        """
        factors = self.registry.list_factors(status, category)
        
        if not factors:
            return pd.DataFrame()
        
        # 转换为DataFrame
        data = []
        for factor in factors:
            row = {
                'name': factor.name,
                'description': factor.description,
                'category': factor.category,
                'status': factor.status.value,
                'author': factor.author,
                'created_time': factor.created_time,
                'last_updated': factor.last_updated
            }
            
            # 添加性能指标
            if factor.performance_metrics:
                for metric, value in factor.performance_metrics.items():
                    row[f'perf_{metric}'] = value
            
            data.append(row)
        
        df = pd.DataFrame(data)
        return df.sort_values('last_updated', ascending=False)
    
    def get_factor_info(self, name: str) -> Optional[Dict[str, Any]]:
        """获取因子详细信息"""
        factor = self.registry.get_factor(name)
        if not factor:
            return None
        
        lifecycle = self.results_tracker.get_factor_lifecycle(name)
        
        info = {
            'basic_info': {
                'name': factor.name,
                'description': factor.description,
                'category': factor.category,
                'author': factor.author,
                'status': factor.status.value
            },
            'timestamps': {
                'created_time': factor.created_time,
                'last_updated': factor.last_updated
            },
            'performance_metrics': factor.performance_metrics,
            'test_results': factor.test_results,
            'status_history': factor.status_history
        }
        
        if lifecycle:
            info['lifecycle'] = {
                'calculation_success': lifecycle.calculation_success,
                'test_success': lifecycle.test_success,
                'calculation_time_cost': lifecycle.calculation_time_cost,
                'test_time_cost': lifecycle.test_time_cost,
                'promotion_decision': lifecycle.promotion_decision
            }
        
        return info
    
    # ========== 因子计算相关 ==========
    
    def calculate_factor(self, name: str, calculation_params: Dict[str, Any] = None, 
                        save_result: bool = True) -> CalculationResult:
        """
        计算因子
        
        Parameters:
        -----------
        name : str
            因子名称
        calculation_params : Dict[str, Any], optional
            计算参数
        save_result : bool
            是否保存结果
            
        Returns:
        --------
        CalculationResult
            计算结果
        """
        try:
            logger.info(f"开始计算因子: {name}")
            
            # 获取计算函数
            calculation_func = self.registry.get_calculation_function(name)
            if not calculation_func:
                raise ValueError(f"因子 {name} 的计算函数未找到")
            
            # 执行计算
            result = self.calculation_engine.calculate_factor(
                factor_name=name,
                calculation_func=calculation_func,
                calculation_params=calculation_params,
                save_result=save_result
            )
            
            # 更新注册表状态
            if result.success:
                self.registry.update_factor_status(
                    name, FactorStatus.CALCULATED,
                    f"计算成功，耗时 {result.calculation_time:.2f}秒"
                )
                
                # 更新跟踪器
                self.results_tracker.update_factor_lifecycle(
                    factor_name=name,
                    event_type="calculation",
                    timestamp=result.calculated_at,
                    success=True,
                    time_cost=result.calculation_time,
                    quality_score=self._calculate_data_quality_score(result.factor_data)
                )
            else:
                self.registry.update_factor_status(
                    name, FactorStatus.FAILED,
                    f"计算失败: {result.error_msg}"
                )
                
                self.results_tracker.update_factor_lifecycle(
                    factor_name=name,
                    event_type="calculation",
                    timestamp=result.calculated_at,
                    success=False,
                    time_cost=result.calculation_time
                )
            
            self._update_stats(result.success)
            return result
            
        except Exception as e:
            logger.error(f"计算因子 {name} 失败: {e}")
            self._update_stats(False)
            raise
    
    def batch_calculate(self, factor_names: List[str] = None, 
                       status_filter: List[FactorStatus] = None,
                       calculation_params: Dict[str, Any] = None) -> Dict[str, CalculationResult]:
        """
        批量计算因子
        
        Parameters:
        -----------
        factor_names : List[str], optional
            要计算的因子名称列表
        status_filter : List[FactorStatus], optional
            状态筛选，默认为[REGISTERED]
        calculation_params : Dict[str, Any], optional
            计算参数
            
        Returns:
        --------
        Dict[str, CalculationResult]
            计算结果字典
        """
        # 确定要计算的因子列表
        if factor_names is None:
            if status_filter is None:
                status_filter = [FactorStatus.REGISTERED]
            
            factors_to_calculate = []
            for status in status_filter:
                factors_to_calculate.extend(self.registry.list_factors(status=status))
            
            factor_names = [f.name for f in factors_to_calculate]
        
        if not factor_names:
            logger.info("没有找到需要计算的因子")
            return {}
        
        logger.info(f"开始批量计算 {len(factor_names)} 个因子")
        
        results = {}
        for factor_name in factor_names:
            try:
                result = self.calculate_factor(factor_name, calculation_params)
                results[factor_name] = result
            except Exception as e:
                logger.error(f"批量计算中处理因子 {factor_name} 失败: {e}")
                results[factor_name] = CalculationResult(
                    factor_name=factor_name,
                    success=False,
                    error_msg=str(e)
                )
        
        successful_count = sum(1 for r in results.values() if r.success)
        logger.info(f"批量计算完成: {successful_count}/{len(factor_names)} 个因子成功")
        
        return results
    
    # ========== 因子测试相关 ==========
    
    def test_factor(self, name: str, test_params: Dict[str, Any] = None, 
                   auto_evaluate: bool = True) -> Optional[TestResult]:
        """
        测试因子
        
        Parameters:
        -----------
        name : str
            因子名称
        test_params : Dict[str, Any], optional
            测试参数
        auto_evaluate : bool
            是否自动评估
            
        Returns:
        --------
        TestResult or None
            测试结果
        """
        try:
            logger.info(f"开始测试因子: {name}")
            
            # 检查因子状态
            factor = self.registry.get_factor(name)
            if not factor:
                raise ValueError(f"因子 {name} 不存在")
            
            if factor.status not in [FactorStatus.CALCULATED, FactorStatus.TESTED]:
                logger.warning(f"因子 {name} 状态为 {factor.status.value}，可能需要先计算")
            
            # 获取计算结果
            # 这里简化处理，实际应该从存储中加载计算结果
            calculation_result = None  # 实际实现中需要从文件或缓存中获取
            
            # 执行测试
            test_result = self.test_coordinator.test_factor(
                factor_name=name,
                calculation_result=calculation_result,
                test_params=test_params,
                auto_evaluate=auto_evaluate
            )
            
            # 更新跟踪器
            performance_metrics = {}
            if hasattr(test_result, 'performance_metrics') and test_result.performance_metrics:
                performance_metrics = test_result.performance_metrics
            
            self.results_tracker.update_factor_lifecycle(
                factor_name=name,
                event_type="testing",
                timestamp=datetime.now(),
                success=True,
                time_cost=0.0,  # 从test_result中获取实际耗时
                performance_metrics=performance_metrics
            )
            
            self._update_stats(True)
            logger.info(f"因子 {name} 测试完成")
            return test_result
            
        except Exception as e:
            logger.error(f"测试因子 {name} 失败: {e}")
            self._update_stats(False)
            return None
    
    def batch_test(self, factor_names: List[str] = None,
                  status_filter: List[FactorStatus] = None,
                  test_params: Dict[str, Any] = None) -> Dict[str, TestResult]:
        """批量测试因子"""
        return self.test_coordinator.batch_test_factors(
            factor_names, status_filter, test_params
        )
    
    # ========== 因子评估和管理 ==========
    
    def promote_factor(self, name: str, target_category: str) -> bool:
        """
        提升因子到正式repository
        
        Parameters:
        -----------
        name : str
            因子名称
        target_category : str
            目标分类
            
        Returns:
        --------
        bool
            是否成功提升
        """
        try:
            success = self.test_coordinator.promote_factor(name, target_category)
            
            if success:
                # 更新跟踪器
                self.results_tracker.update_factor_lifecycle(
                    factor_name=name,
                    event_type="decision",
                    timestamp=datetime.now(),
                    decision="promoted",
                    status="promoted",
                    reason=f"提升到{target_category}分类"
                )
            
            self._update_stats(success)
            return success
            
        except Exception as e:
            logger.error(f"提升因子 {name} 失败: {e}")
            self._update_stats(False)
            return False
    
    def archive_factor(self, name: str, reason: str = "") -> bool:
        """
        归档因子
        
        Parameters:
        -----------
        name : str
            因子名称
        reason : str
            归档原因
            
        Returns:
        --------
        bool
            是否成功归档
        """
        try:
            success = self.test_coordinator.archive_factor(name, reason)
            
            if success:
                # 更新跟踪器
                self.results_tracker.update_factor_lifecycle(
                    factor_name=name,
                    event_type="decision",
                    timestamp=datetime.now(),
                    decision="archived",
                    status="archived",
                    reason=reason
                )
            
            self._update_stats(success)
            return success
            
        except Exception as e:
            logger.error(f"归档因子 {name} 失败: {e}")
            self._update_stats(False)
            return False
    
    # ========== 完整工作流程 ==========
    
    def full_workflow(self, name: str, calculation_func: Callable,
                     description: str = "", category: str = "experimental",
                     calculation_params: Dict[str, Any] = None,
                     test_params: Dict[str, Any] = None,
                     auto_decision: bool = True) -> Dict[str, Any]:
        """
        执行完整的因子开发工作流程：注册 -> 计算 -> 测试 -> 评估
        
        Parameters:
        -----------
        name : str
            因子名称
        calculation_func : Callable
            计算函数
        description : str
            描述
        category : str
            分类
        calculation_params : Dict[str, Any], optional
            计算参数
        test_params : Dict[str, Any], optional
            测试参数
        auto_decision : bool
            是否自动决策
            
        Returns:
        --------
        Dict[str, Any]
            完整工作流程结果
        """
        workflow_result = {
            'factor_name': name,
            'workflow_start': datetime.now(),
            'stages': {},
            'final_status': 'unknown',
            'success': False
        }
        
        try:
            logger.info(f"开始执行因子 {name} 完整工作流程")
            
            # 第1阶段：注册
            logger.info(f"阶段1: 注册因子 {name}")
            register_success = self.register_factor(
                name, calculation_func, description, category
            )
            workflow_result['stages']['registration'] = {
                'success': register_success,
                'timestamp': datetime.now()
            }
            
            if not register_success:
                workflow_result['final_status'] = 'registration_failed'
                return workflow_result
            
            # 第2阶段：计算
            logger.info(f"阶段2: 计算因子 {name}")
            calc_result = self.calculate_factor(name, calculation_params)
            workflow_result['stages']['calculation'] = {
                'success': calc_result.success,
                'timestamp': calc_result.calculated_at,
                'time_cost': calc_result.calculation_time,
                'error_msg': calc_result.error_msg if not calc_result.success else None
            }
            
            if not calc_result.success:
                workflow_result['final_status'] = 'calculation_failed'
                return workflow_result
            
            # 第3阶段：测试
            logger.info(f"阶段3: 测试因子 {name}")
            test_result = self.test_factor(name, test_params, auto_evaluate=auto_decision)
            workflow_result['stages']['testing'] = {
                'success': test_result is not None,
                'timestamp': datetime.now()
            }
            
            if test_result is None:
                workflow_result['final_status'] = 'testing_failed'
                return workflow_result
            
            # 第4阶段：获取最终状态
            final_factor = self.registry.get_factor(name)
            if final_factor:
                workflow_result['final_status'] = final_factor.status.value
                workflow_result['performance_metrics'] = final_factor.performance_metrics
            
            workflow_result['success'] = True
            workflow_result['workflow_end'] = datetime.now()
            workflow_result['total_time'] = (
                workflow_result['workflow_end'] - workflow_result['workflow_start']
            ).total_seconds()
            
            logger.info(f"因子 {name} 完整工作流程完成，最终状态: {workflow_result['final_status']}")
            return workflow_result
            
        except Exception as e:
            logger.error(f"执行因子 {name} 完整工作流程失败: {e}")
            workflow_result['error_msg'] = str(e)
            workflow_result['workflow_end'] = datetime.now()
            return workflow_result
    
    # ========== 分析和报告 ==========
    
    def get_summary_report(self) -> Dict[str, Any]:
        """获取汇总报告"""
        registry_summary = self.registry.get_factor_summary()
        lifecycle_summary = self.results_tracker.generate_lifecycle_summary()
        performance_analysis = self.results_tracker.analyze_experimental_performance()
        
        # 统计各状态的因子数量
        status_counts = {}
        for factor in self.registry.list_factors():
            status = factor.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # 获取top因子
        top_factors = self.results_tracker.get_top_performing_factors(5)
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'total_factors': len(self.registry.list_factors()),
            'status_distribution': status_counts,
            'manager_stats': self.manager_stats,
            'top_performing_factors': top_factors,
            'performance_analysis': performance_analysis,
            'registry_summary_shape': registry_summary.shape if not registry_summary.empty else (0, 0),
            'lifecycle_summary_shape': lifecycle_summary.shape if not lifecycle_summary.empty else (0, 0)
        }
        
        return report
    
    def export_for_screening(self, performance_threshold: float = 50.0) -> Dict[str, Any]:
        """导出数据供筛选器使用"""
        return self.results_tracker.export_for_screening(
            performance_threshold=performance_threshold,
            include_pending=True
        )
    
    def create_factor_template(self, factor_name: str, category: str = "experimental") -> str:
        """创建因子模板代码"""
        return self.calculation_engine.create_factor_template(factor_name, category)
    
    # ========== 内部辅助方法 ==========
    
    def _calculate_data_quality_score(self, factor_data: pd.Series) -> float:
        """计算数据质量评分"""
        if factor_data is None or factor_data.empty:
            return 0.0
        
        # 简单的质量评分：基于非空比例和数值分布
        non_null_ratio = factor_data.notna().sum() / len(factor_data)
        
        # 检查是否有极值
        q1, q3 = factor_data.quantile([0.25, 0.75])
        iqr = q3 - q1
        outlier_ratio = 0.0
        if iqr > 0:
            outliers = ((factor_data < (q1 - 1.5 * iqr)) | 
                       (factor_data > (q3 + 1.5 * iqr))).sum()
            outlier_ratio = outliers / len(factor_data)
        
        # 综合评分
        quality_score = non_null_ratio * (1 - outlier_ratio * 0.5)
        return max(0.0, min(1.0, quality_score))
    
    def _update_stats(self, success: bool):
        """更新管理器统计"""
        self.manager_stats['total_operations'] += 1
        if success:
            self.manager_stats['successful_operations'] += 1
        else:
            self.manager_stats['failed_operations'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """获取管理器统计信息"""
        uptime = (datetime.now() - self.manager_stats['started_at']).total_seconds()
        
        stats = self.manager_stats.copy()
        stats.update({
            'uptime_seconds': uptime,
            'success_rate': (
                self.manager_stats['successful_operations'] / 
                max(1, self.manager_stats['total_operations'])
            ),
            'calculation_engine_stats': self.calculation_engine.get_calculation_stats(),
            'test_coordinator_stats': self.test_coordinator.get_test_stats()
        })
        
        return stats


if __name__ == "__main__":
    # 测试代码
    def test_factor_func(context=None, **kwargs):
        """测试因子计算函数"""
        financial_data = context.load_financial_data()
        tools = context.get_generators_tools()
        
        # 使用TTM工具
        ttm_data = tools['calculate_ttm'](financial_data)
        
        # 返回示例数据
        import numpy as np
        factor_data = pd.Series(
            data=np.random.randn(1000),
            index=pd.MultiIndex.from_product(
                [pd.date_range('2024-01-01', periods=100), 
                 ['000001', '000002', '000003', '300001', '600000']],
                names=['TradingDates', 'StockCodes']
            )
        )
        return factor_data
    
    # 创建管理器
    manager = ExperimentalFactorManager()
    
    # 测试注册
    success = manager.register_factor(
        name="test_unified_factor",
        calculation_func=test_factor_func,
        description="统一管理器测试因子",
        category="test",
        author="AI Assistant"
    )
    
    print(f"因子注册成功: {success}")
    
    # 获取汇总报告
    report = manager.get_summary_report()
    print(f"汇总报告生成成功，总因子数: {report['total_factors']}")
    
    # 获取统计信息
    stats = manager.get_stats()
    print(f"管理器统计: {stats['total_operations']} 次操作，成功率: {stats['success_rate']:.2%}")