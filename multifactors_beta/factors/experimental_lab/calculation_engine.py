#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子计算引擎模块

负责实验因子的安全计算、数据管理和结果存储
严格遵循项目约束：必须使用factors.generators工具集
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import logging
from datetime import datetime
import pickle
import traceback

from factors.utils.data_loader import FactorDataLoader
from factors.generators import (
    calculate_ttm,
    calculate_yoy,
    calculate_qoq,
    calculate_single_quarter,
    calculate_zscore,
    expand_to_daily_vectorized,
    FinancialReportProcessor
)
from config import get_config

logger = logging.getLogger(__name__)


class CalculationContext:
    """计算上下文，提供标准化的数据和工具接口"""
    
    def __init__(self):
        """初始化计算上下文"""
        self.financial_data = None
        self.price_data = None
        self.market_cap_data = None
        self.trading_dates = None
        self.loaded_datasets = set()
        
    def load_financial_data(self) -> pd.DataFrame:
        """加载财务数据"""
        if self.financial_data is None:
            self.financial_data = FactorDataLoader.load_financial_data()
            self.loaded_datasets.add('financial')
            logger.debug("财务数据加载完成")
        return self.financial_data
    
    def load_price_data(self) -> pd.Series:
        """加载价格数据"""
        if self.price_data is None:
            self.price_data = FactorDataLoader.load_price_data()
            self.loaded_datasets.add('price')
            logger.debug("价格数据加载完成")
        return self.price_data
    
    def load_market_cap_data(self) -> pd.Series:
        """加载市值数据"""
        if self.market_cap_data is None:
            self.market_cap_data = FactorDataLoader.load_market_cap()
            self.loaded_datasets.add('market_cap')
            logger.debug("市值数据加载完成")
        return self.market_cap_data
    
    def load_trading_dates(self) -> pd.DatetimeIndex:
        """加载交易日期"""
        if self.trading_dates is None:
            self.trading_dates = FactorDataLoader.load_trading_dates()
            self.loaded_datasets.add('trading_dates')
            logger.debug("交易日期加载完成")
        return self.trading_dates
    
    def get_generators_tools(self) -> Dict[str, Callable]:
        """获取generators工具集"""
        return {
            'calculate_ttm': calculate_ttm,
            'calculate_yoy': calculate_yoy,
            'calculate_qoq': calculate_qoq,
            'calculate_single_quarter': calculate_single_quarter,
            'calculate_zscore': calculate_zscore,
            'expand_to_daily_vectorized': expand_to_daily_vectorized,
            'FinancialReportProcessor': FinancialReportProcessor
        }
    
    def clear_cache(self):
        """清理缓存的数据"""
        self.financial_data = None
        self.price_data = None
        self.market_cap_data = None
        self.trading_dates = None
        self.loaded_datasets.clear()
        logger.debug("计算上下文缓存已清理")


class CalculationResult:
    """计算结果封装类"""
    
    def __init__(self, factor_name: str, factor_data: pd.Series = None, 
                 success: bool = True, error_msg: str = "", 
                 calculation_time: float = 0.0, metadata: Dict[str, Any] = None):
        self.factor_name = factor_name
        self.factor_data = factor_data
        self.success = success
        self.error_msg = error_msg
        self.calculation_time = calculation_time
        self.calculated_at = datetime.now()
        self.metadata = metadata or {}
        
        # 数据质量检查
        if success and factor_data is not None:
            self._validate_result()
    
    def _validate_result(self):
        """验证计算结果的数据质量"""
        try:
            # 检查数据格式
            if not isinstance(self.factor_data, pd.Series):
                raise ValueError("因子数据必须是pandas.Series格式")
            
            # 检查MultiIndex格式
            if not isinstance(self.factor_data.index, pd.MultiIndex):
                raise ValueError("因子数据必须使用MultiIndex[TradingDates, StockCodes]格式")
            
            # 检查索引名称
            if self.factor_data.index.names != ['TradingDates', 'StockCodes']:
                logger.warning(f"索引名称不标准: {self.factor_data.index.names}，期望['TradingDates', 'StockCodes']")
            
            # 数据质量统计
            self.metadata.update({
                'total_observations': len(self.factor_data),
                'non_null_observations': self.factor_data.notna().sum(),
                'null_ratio': self.factor_data.isna().sum() / len(self.factor_data),
                'value_range': [float(self.factor_data.min()), float(self.factor_data.max())],
                'mean': float(self.factor_data.mean()),
                'std': float(self.factor_data.std())
            })
            
            logger.debug(f"因子 {self.factor_name} 数据验证通过")
            
        except Exception as e:
            self.success = False
            self.error_msg = f"数据验证失败: {e}"
            logger.error(f"因子 {self.factor_name} 数据验证失败: {e}")
    
    def save_result(self, save_path: Optional[Path] = None):
        """保存计算结果"""
        try:
            if save_path is None:
                # 使用默认路径
                base_path = Path(get_config('main.paths.factors_data', 'data/factors'))
                save_path = base_path / "experimental_lab" / "calculations"
            
            save_path.mkdir(parents=True, exist_ok=True)
            
            # 生成文件名
            timestamp = self.calculated_at.strftime("%Y%m%d_%H%M%S")
            filename = f"{self.factor_name}_{timestamp}.pkl"
            file_path = save_path / filename
            
            # 保存结果
            with open(file_path, 'wb') as f:
                pickle.dump(self, f)
            
            logger.info(f"计算结果已保存: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"保存计算结果失败: {e}")
            return None


class FactorCalculationEngine:
    """
    因子计算引擎
    
    功能：
    1. 提供标准化的计算环境
    2. 强制使用factors.generators工具集
    3. 数据质量验证和结果管理
    4. 计算性能监控和错误处理
    """
    
    def __init__(self, cache_data: bool = True, result_save_path: Optional[str] = None):
        """
        初始化计算引擎
        
        Parameters:
        -----------
        cache_data : bool
            是否缓存数据以提高性能
        result_save_path : str, optional
            结果保存路径
        """
        self.cache_data = cache_data
        self.context = CalculationContext()
        
        if result_save_path:
            self.result_save_path = Path(result_save_path)
        else:
            base_path = Path(get_config('main.paths.factors_data', 'data/factors'))
            self.result_save_path = base_path / "experimental_lab" / "calculations"
        
        self.result_save_path.mkdir(parents=True, exist_ok=True)
        
        # 计算统计
        self.calculation_stats = {
            'total_calculations': 0,
            'successful_calculations': 0,
            'failed_calculations': 0,
            'average_calculation_time': 0.0
        }
        
        logger.info(f"计算引擎初始化完成，结果保存路径: {self.result_save_path}")
    
    def calculate_factor(self, factor_name: str, calculation_func: Callable,
                        calculation_params: Dict[str, Any] = None,
                        save_result: bool = True) -> CalculationResult:
        """
        计算因子
        
        Parameters:
        -----------
        factor_name : str
            因子名称
        calculation_func : Callable
            因子计算函数
        calculation_params : Dict[str, Any], optional
            计算参数
        save_result : bool
            是否保存结果
            
        Returns:
        --------
        CalculationResult
            计算结果
        """
        start_time = datetime.now()
        logger.info(f"开始计算因子: {factor_name}")
        
        try:
            # 准备计算参数
            if calculation_params is None:
                calculation_params = {}
            
            # 准备计算上下文（注入标准数据和工具）
            calc_kwargs = {
                'context': self.context,
                **calculation_params
            }
            
            # 执行计算
            factor_data = calculation_func(**calc_kwargs)
            
            # 计算耗时
            calculation_time = (datetime.now() - start_time).total_seconds()
            
            # 创建结果对象
            result = CalculationResult(
                factor_name=factor_name,
                factor_data=factor_data,
                success=True,
                calculation_time=calculation_time,
                metadata={
                    'calculation_params': calculation_params,
                    'loaded_datasets': list(self.context.loaded_datasets)
                }
            )
            
            # 保存结果
            if save_result:
                result.save_result(self.result_save_path)
            
            # 更新统计
            self._update_stats(success=True, calculation_time=calculation_time)
            
            logger.info(f"因子 {factor_name} 计算成功，耗时: {calculation_time:.2f}秒")
            return result
            
        except Exception as e:
            calculation_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"计算失败: {str(e)}\n{traceback.format_exc()}"
            
            logger.error(f"因子 {factor_name} 计算失败: {e}")
            
            # 创建失败结果
            result = CalculationResult(
                factor_name=factor_name,
                success=False,
                error_msg=error_msg,
                calculation_time=calculation_time
            )
            
            # 更新统计
            self._update_stats(success=False, calculation_time=calculation_time)
            
            return result
    
    def batch_calculate(self, factor_definitions: Dict[str, Dict[str, Any]],
                       parallel: bool = False) -> Dict[str, CalculationResult]:
        """
        批量计算因子
        
        Parameters:
        -----------
        factor_definitions : Dict[str, Dict[str, Any]]
            因子定义字典，格式: {factor_name: {'func': callable, 'params': dict}}
        parallel : bool
            是否并行计算（暂时不实现）
            
        Returns:
        --------
        Dict[str, CalculationResult]
            计算结果字典
        """
        results = {}
        total_factors = len(factor_definitions)
        
        logger.info(f"开始批量计算 {total_factors} 个因子")
        
        for i, (factor_name, definition) in enumerate(factor_definitions.items(), 1):
            logger.info(f"计算进度: {i}/{total_factors} - {factor_name}")
            
            try:
                calculation_func = definition['func']
                calculation_params = definition.get('params', {})
                
                result = self.calculate_factor(
                    factor_name=factor_name,
                    calculation_func=calculation_func,
                    calculation_params=calculation_params
                )
                
                results[factor_name] = result
                
            except Exception as e:
                logger.error(f"批量计算中处理因子 {factor_name} 失败: {e}")
                results[factor_name] = CalculationResult(
                    factor_name=factor_name,
                    success=False,
                    error_msg=str(e)
                )
        
        # 清理缓存（批量计算后）
        if not self.cache_data:
            self.context.clear_cache()
        
        successful_count = sum(1 for r in results.values() if r.success)
        logger.info(f"批量计算完成: {successful_count}/{total_factors} 个因子成功")
        
        return results
    
    def create_factor_template(self, factor_name: str, category: str = "experimental") -> str:
        """
        创建因子计算函数模板
        
        Parameters:
        -----------
        factor_name : str
            因子名称
        category : str
            因子分类
            
        Returns:
        --------
        str
            Python代码模板
        """
        template = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
{factor_name} 因子计算函数
自动生成的模板，请根据需要修改
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

def calculate_{factor_name.lower().replace(' ', '_')}(context=None, **kwargs) -> pd.Series:
    """
    计算 {factor_name} 因子
    
    Parameters:
    -----------
    context : CalculationContext
        计算上下文，提供数据和工具
    **kwargs : dict
        其他计算参数
        
    Returns:
    --------
    pd.Series
        因子数据，MultiIndex[TradingDates, StockCodes]格式
    """
    # 1. 获取必要的数据
    financial_data = context.load_financial_data()
    # price_data = context.load_price_data()
    # market_cap_data = context.load_market_cap_data()
    # trading_dates = context.load_trading_dates()
    
    # 2. 获取generators工具集
    tools = context.get_generators_tools()
    calculate_ttm = tools['calculate_ttm']
    expand_to_daily = tools['expand_to_daily_vectorized']
    
    # 3. 实现你的计算逻辑
    # 示例：计算TTM净利润
    # ttm_data = calculate_ttm(financial_data)
    # factor_values = ttm_data['net_income']  # 根据实际需要修改
    
    # 4. 扩展到日频（如果需要）
    # daily_factor = expand_to_daily(
    #     factor_data=factor_values,
    #     release_dates=financial_calendar,
    #     trading_dates=trading_dates
    # )
    
    # TODO: 实现你的具体计算逻辑
    # 这里只是示例代码
    factor_data = pd.Series(
        data=np.random.randn(1000),
        index=pd.MultiIndex.from_product(
            [pd.date_range('2024-01-01', '2024-12-31'), ['000001', '000002']],
            names=['TradingDates', 'StockCodes']
        )
    )
    
    return factor_data


# 使用示例
if __name__ == "__main__":
    from factors.experimental_lab import calculate_experimental_factor
    
    result = calculate_experimental_factor('{factor_name}')
    print(f"因子计算完成: {{result.success}}")
    if result.success:
        print(f"数据形状: {{result.factor_data.shape}}")
'''
        return template
    
    def get_calculation_stats(self) -> Dict[str, Any]:
        """获取计算统计信息"""
        return self.calculation_stats.copy()
    
    def clear_cache(self):
        """清理数据缓存"""
        self.context.clear_cache()
        logger.info("计算引擎缓存已清理")
    
    def _update_stats(self, success: bool, calculation_time: float):
        """更新计算统计"""
        self.calculation_stats['total_calculations'] += 1
        
        if success:
            self.calculation_stats['successful_calculations'] += 1
        else:
            self.calculation_stats['failed_calculations'] += 1
        
        # 更新平均计算时间
        total = self.calculation_stats['total_calculations']
        current_avg = self.calculation_stats['average_calculation_time']
        self.calculation_stats['average_calculation_time'] = (
            (current_avg * (total - 1) + calculation_time) / total
        )


if __name__ == "__main__":
    # 测试代码
    def test_factor_calculation(context=None, **kwargs):
        """测试因子计算函数"""
        # 使用context获取数据
        financial_data = context.load_financial_data()
        tools = context.get_generators_tools()
        
        # 使用TTM工具
        ttm_data = tools['calculate_ttm'](financial_data)
        
        # 返回示例数据
        factor_data = pd.Series(
            data=np.random.randn(100),
            index=pd.MultiIndex.from_product(
                [pd.date_range('2024-01-01', periods=10), ['000001', '000002']],
                names=['TradingDates', 'StockCodes']
            )
        )
        return factor_data
    
    # 测试计算引擎
    engine = FactorCalculationEngine()
    result = engine.calculate_factor("test_factor", test_factor_calculation)
    
    print(f"计算结果: {result.success}")
    if result.success:
        print(f"数据形状: {result.factor_data.shape}")
        print(f"计算耗时: {result.calculation_time:.2f}秒")
    else:
        print(f"计算失败: {result.error_msg}")