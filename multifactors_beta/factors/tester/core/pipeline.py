"""
单因子测试流水线
整合数据管理、测试执行、结果保存的完整流程
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
import logging
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import os
from pathlib import Path

from core.config_manager import get_config, get_path
from .data_manager import DataManager
from .factor_tester import FactorTester
from .result_manager import ResultManager
from ..base.test_result import TestResult, BatchTestResult

logger = logging.getLogger(__name__)


class SingleFactorTestPipeline:
    """单因子测试流水线"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化测试流水线
        
        Parameters
        ----------
        config : Dict, optional
            配置字典，如果为None则从全局配置读取
        """
        # 加载配置
        self.config = config or get_config('factor_test')
        
        # 初始化组件
        self.data_manager = DataManager(self.config)
        self.tester = FactorTester(self.config)
        self.result_manager = ResultManager()
        
        logger.info("单因子测试流水线初始化完成")
        
    def run(
        self, 
        factor_name: str,
        save_result: bool = True,
        **override_config
    ) -> TestResult:
        """
        执行单因子测试
        
        Parameters
        ----------
        factor_name : str
            因子名称
        save_result : bool
            是否保存结果
        **override_config
            覆盖配置参数
            
        Returns
        -------
        TestResult
            测试结果
        """
        logger.info(f"开始测试因子: {factor_name}")
        
        # 合并配置
        test_config = self._merge_config(override_config)
        
        # 更新组件配置
        if override_config:
            self.tester.config.update(test_config)
            if 'group_nums' in test_config:
                self.tester.group_nums = test_config['group_nums']
        
        # 准备数据
        logger.info("准备测试数据...")
        test_data = self.data_manager.prepare_test_data(
            factor_name,
            use_base_factors=test_config.get('netral_base', True),
            use_industry=test_config.get('use_industry', True),
            custom_base_factors=test_config.get('custom_base_factors')
        )
        
        if not test_data or 'factor' not in test_data or test_data['factor'].empty:
            logger.error(f"因子数据准备失败: {factor_name}")
            result = TestResult(factor_name=factor_name)
            result.errors.append("因子数据为空或加载失败")
            return result
        
        # 执行测试
        logger.info("执行因子测试...")
        result = self.tester.test(test_data)
        
        # 更新结果的配置快照
        result.config_snapshot = test_config
        
        # 保存结果
        if save_result:
            logger.info("保存测试结果...")
            self.result_manager.save(result)
        
        logger.info(f"因子测试完成: {factor_name}")
        
        return result
    
    def batch_run(
        self,
        factor_names: List[str],
        save_results: bool = True,
        parallel: bool = True,
        max_workers: Optional[int] = None,
        **override_config
    ) -> BatchTestResult:
        """
        批量执行因子测试
        
        Parameters
        ----------
        factor_names : List[str]
            因子名称列表
        save_results : bool
            是否保存结果
        parallel : bool
            是否并行执行
        max_workers : int, optional
            最大工作进程数
        **override_config
            覆盖配置参数
            
        Returns
        -------
        BatchTestResult
            批量测试结果
        """
        logger.info(f"开始批量测试 {len(factor_names)} 个因子")
        
        batch_result = BatchTestResult()
        
        if parallel:
            # 并行执行
            max_workers = max_workers or min(4, os.cpu_count())
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for factor_name in factor_names:
                    future = executor.submit(
                        self.run, 
                        factor_name, 
                        save_results,
                        **override_config
                    )
                    futures.append((factor_name, future))
                
                # 收集结果
                for factor_name, future in futures:
                    try:
                        result = future.result(timeout=300)  # 5分钟超时
                        batch_result.add_result(result)
                        logger.info(f"完成测试: {factor_name}")
                    except Exception as e:
                        logger.error(f"因子测试失败 {factor_name}: {e}")
                        # 创建错误结果
                        error_result = TestResult(factor_name=factor_name)
                        error_result.errors.append(str(e))
                        batch_result.add_result(error_result)
        else:
            # 串行执行
            for factor_name in factor_names:
                try:
                    result = self.run(factor_name, save_results, **override_config)
                    batch_result.add_result(result)
                except Exception as e:
                    logger.error(f"因子测试失败 {factor_name}: {e}")
                    error_result = TestResult(factor_name=factor_name)
                    error_result.errors.append(str(e))
                    batch_result.add_result(error_result)
        
        # 生成汇总
        batch_result.generate_summary()
        
        # 保存批量结果
        if save_results:
            batch_result.save(self.result_manager.base_path)
        
        logger.info(f"批量测试完成，成功: {len([r for r in batch_result.test_results if not r.errors])}, "
                   f"失败: {len([r for r in batch_result.test_results if r.errors])}")
        
        return batch_result
    
    def test_all_factors(
        self,
        save_results: bool = True,
        skip_existing: bool = True,
        parallel: bool = True,
        **override_config
    ) -> BatchTestResult:
        """
        测试所有可用因子
        
        Parameters
        ----------
        save_results : bool
            是否保存结果
        skip_existing : bool
            是否跳过已测试的因子
        parallel : bool
            是否并行执行
        **override_config
            覆盖配置参数
            
        Returns
        -------
        BatchTestResult
            批量测试结果
        """
        # 获取所有因子文件
        raw_factors_path = get_path('raw_factors')
        all_factors = []
        
        if os.path.exists(raw_factors_path):
            for file in os.listdir(raw_factors_path):
                if file.endswith('.pkl'):
                    factor_name = file[:-4]
                    # 排除基准因子
                    if factor_name not in self.config.get('base_factors', []):
                        all_factors.append(factor_name)
        
        logger.info(f"找到 {len(all_factors)} 个待测试因子")
        
        # 检查已存在的结果
        if skip_existing:
            existing_results = set()
            test_path = self.result_manager.base_path
            
            for root, dirs, files in os.walk(test_path):
                for file in files:
                    if file.endswith('_test_result.pkl'):
                        factor_name = file.replace('_test_result.pkl', '')
                        existing_results.add(factor_name)
            
            # 过滤已测试的因子
            factors_to_test = [f for f in all_factors if f not in existing_results]
            logger.info(f"跳过 {len(existing_results)} 个已测试因子，"
                       f"将测试 {len(factors_to_test)} 个新因子")
        else:
            factors_to_test = all_factors
        
        # 执行批量测试
        return self.batch_run(
            factors_to_test,
            save_results=save_results,
            parallel=parallel,
            **override_config
        )
    
    def run_with_profiles(
        self,
        factor_name: str,
        profiles: List[str],
        save_results: bool = True
    ) -> Dict[str, TestResult]:
        """
        使用不同配置文件测试同一因子
        
        Parameters
        ----------
        factor_name : str
            因子名称
        profiles : List[str]
            配置文件名称列表
        save_results : bool
            是否保存结果
            
        Returns
        -------
        Dict[str, TestResult]
            配置名称到测试结果的映射
        """
        results = {}
        
        for profile in profiles:
            logger.info(f"使用配置 {profile} 测试因子 {factor_name}")
            
            # 加载特定配置
            profile_config = self._load_profile_config(profile)
            
            # 执行测试
            result = self.run(
                factor_name,
                save_result=save_results,
                **profile_config
            )
            
            results[profile] = result
        
        return results
    
    def parameter_sensitivity_test(
        self,
        factor_name: str,
        param_grid: Dict[str, List[Any]],
        save_results: bool = False
    ) -> pd.DataFrame:
        """
        参数敏感性测试
        
        Parameters
        ----------
        factor_name : str
            因子名称
        param_grid : Dict[str, List]
            参数网格
        save_results : bool
            是否保存结果
            
        Returns
        -------
        pd.DataFrame
            参数测试结果表
        """
        from itertools import product
        
        logger.info(f"开始参数敏感性测试: {factor_name}")
        
        # 生成参数组合
        keys = param_grid.keys()
        values = param_grid.values()
        
        results_list = []
        
        for combo in product(*values):
            # 构建配置
            test_config = dict(zip(keys, combo))
            
            logger.info(f"测试参数组合: {test_config}")
            
            # 执行测试
            result = self.run(
                factor_name,
                save_result=save_results,
                **test_config
            )
            
            # 收集结果
            result_row = {
                'factor_name': factor_name,
                **test_config,
                **result.performance_metrics
            }
            
            results_list.append(result_row)
        
        # 创建结果DataFrame
        results_df = pd.DataFrame(results_list)
        
        # 保存参数测试结果
        if save_results:
            filename = f"{factor_name}_param_sensitivity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            filepath = os.path.join(self.result_manager.base_path, filename)
            results_df.to_csv(filepath, index=False, encoding='utf-8-sig')
            logger.info(f"参数测试结果已保存: {filepath}")
        
        return results_df
    
    def _merge_config(self, override_config: Dict) -> Dict:
        """
        合并配置
        
        Parameters
        ----------
        override_config : Dict
            覆盖配置
            
        Returns
        -------
        Dict
            合并后的配置
        """
        merged = self.config.copy()
        merged.update(override_config)
        return merged
    
    def _load_profile_config(self, profile: str) -> Dict:
        """
        加载配置文件
        
        Parameters
        ----------
        profile : str
            配置文件名称
            
        Returns
        -------
        Dict
            配置字典
        """
        # 这里简化处理，实际应该从配置文件加载
        profiles = {
            'quick_test': {
                'begin_date': '2024-01-01',
                'group_nums': 5,
                'netral_base': False
            },
            'full_test': {
                'begin_date': '2018-01-01',
                'group_nums': 20,
                'netral_base': True,
                'use_industry': True
            },
            'weekly_test': {
                'backtest_type': 'weekly',
                'back_test_trading_price': 'vwap'
            }
        }
        
        return profiles.get(profile, {})
    
    def clear_cache(self):
        """清空数据缓存"""
        self.data_manager.clear_cache()
        logger.info("数据缓存已清空")