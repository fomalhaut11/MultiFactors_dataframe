"""
可测试性混入类 - 提供单元测试友好的设计
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Callable, Protocol, Union
from unittest.mock import Mock
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class DataProvider(Protocol):
    """数据提供者协议"""
    
    def get_financial_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """获取财务数据"""
        ...
    
    def get_market_cap_data(self, start_date: str, end_date: str) -> pd.Series:
        """获取市值数据"""
        ...
    
    def get_release_dates(self) -> pd.DataFrame:
        """获取发布日期数据"""
        ...
    
    def get_trading_dates(self, start_date: str, end_date: str) -> pd.DatetimeIndex:
        """获取交易日期"""
        ...


class MockDataProvider:
    """模拟数据提供者，用于测试"""
    
    def __init__(self, 
                 financial_data: Optional[pd.DataFrame] = None,
                 market_cap_data: Optional[pd.Series] = None,
                 release_dates: Optional[pd.DataFrame] = None,
                 trading_dates: Optional[pd.DatetimeIndex] = None):
        self.financial_data = financial_data
        self.market_cap_data = market_cap_data
        self.release_dates = release_dates
        self.trading_dates = trading_dates
    
    def get_financial_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        if self.financial_data is None:
            return self.create_mock_financial_data()
        return self.financial_data
    
    def get_market_cap_data(self, start_date: str, end_date: str) -> pd.Series:
        if self.market_cap_data is None:
            return self.create_mock_market_cap_data()
        return self.market_cap_data
    
    def get_release_dates(self) -> pd.DataFrame:
        if self.release_dates is None:
            return self.create_mock_release_dates()
        return self.release_dates
    
    def get_trading_dates(self, start_date: str, end_date: str) -> pd.DatetimeIndex:
        if self.trading_dates is None:
            return self.create_mock_trading_dates()
        return self.trading_dates
    
    @staticmethod
    def create_mock_financial_data(n_stocks: int = 10, n_periods: int = 8) -> pd.DataFrame:
        """创建模拟财务数据"""
        dates = pd.date_range('2022-03-31', periods=n_periods, freq='Q')
        stocks = [f'stock_{i:03d}' for i in range(n_stocks)]
        
        index = pd.MultiIndex.from_product([dates, stocks], names=['ReportDates', 'StockCodes'])
        
        np.random.seed(42)  # 确保可重现
        data = {
            'DEDUCTEDPROFIT': np.random.normal(1000, 500, len(index)),
            'EQY_BELONGTO_PARCOMSH': np.random.normal(10000, 3000, len(index)),
            'OPREVENUE': np.random.normal(5000, 2000, len(index)),
            'd_quarter': np.tile([1, 2, 3, 4] * (n_periods // 4 + 1), n_stocks)[:len(index)],
            'd_year': np.tile([2022, 2022, 2022, 2022, 2023, 2023, 2023, 2023], n_stocks)[:len(index)]
        }
        
        return pd.DataFrame(data, index=index)
    
    @staticmethod
    def create_mock_market_cap_data(n_stocks: int = 10, n_days: int = 100, start_date: str = '2022-01-01') -> pd.Series:
        """创建模拟市值数据"""
        dates = pd.date_range(start_date, periods=n_days, freq='D')
        stocks = [f'stock_{i:03d}' for i in range(n_stocks)]
        
        index = pd.MultiIndex.from_product([dates, stocks], names=['TradingDates', 'StockCodes'])
        
        np.random.seed(42)
        values = np.random.lognormal(10, 0.5, len(index))
        
        return pd.Series(values, index=index, name='MarketCap')
    
    @staticmethod
    def create_mock_release_dates(n_stocks: int = 10, n_periods: int = 8) -> pd.DataFrame:
        """创建模拟发布日期数据"""
        report_dates = pd.date_range('2022-03-31', periods=n_periods, freq='Q')
        stocks = [f'stock_{i:03d}' for i in range(n_stocks)]
        
        index = pd.MultiIndex.from_product([report_dates, stocks], names=['ReportDates', 'StockCodes'])
        
        # 发布日期通常在报告期后1-2个月
        release_dates = []
        for report_date in report_dates:
            for _ in stocks:
                days_delay = np.random.randint(30, 60)  # 30-60天后发布
                release_date = report_date + pd.Timedelta(days=days_delay)
                release_dates.append(release_date)
        
        data = {'ReleasedDates': release_dates}
        return pd.DataFrame(data, index=index)
    
    @staticmethod
    def create_mock_trading_dates(start_date: str = '2022-01-01', 
                                 end_date: str = '2023-12-31') -> pd.DatetimeIndex:
        """创建模拟交易日期"""
        # 简化：假设除周末外都是交易日
        all_dates = pd.date_range(start_date, end_date, freq='D')
        trading_dates = all_dates[all_dates.weekday < 5]  # 周一到周五
        return trading_dates


class TestableMixin:
    """
    可测试性混入类，为因子类提供依赖注入和测试友好的接口
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._data_provider: Optional[DataProvider] = None
        self._test_mode = False
        self._mock_results = {}
        self._call_history = []
    
    def set_data_provider(self, provider: DataProvider):
        """
        设置数据提供者（依赖注入）
        
        Parameters:
        -----------
        provider : 数据提供者实例
        """
        self._data_provider = provider
    
    def enable_test_mode(self, mock_results: Optional[Dict[str, Any]] = None):
        """
        启用测试模式
        
        Parameters:
        -----------
        mock_results : 模拟结果字典
        """
        self._test_mode = True
        self._mock_results = mock_results or {}
        self._call_history = []
    
    def disable_test_mode(self):
        """禁用测试模式"""
        self._test_mode = False
        self._mock_results = {}
        self._call_history = []
    
    def get_call_history(self) -> list:
        """获取方法调用历史（用于测试验证）"""
        return self._call_history.copy()
    
    def _record_call(self, method_name: str, **kwargs):
        """记录方法调用"""
        if self._test_mode:
            self._call_history.append({
                'method': method_name,
                'kwargs': kwargs,
                'timestamp': pd.Timestamp.now()
            })
    
    def _get_mock_result(self, key: str, default: Any = None) -> Any:
        """获取模拟结果"""
        if self._test_mode and key in self._mock_results:
            return self._mock_results[key]
        return default
    
    def get_data_safely(self, 
                       data_type: str, 
                       fallback_func: Callable = None,
                       **kwargs) -> Any:
        """
        安全获取数据，支持测试模式
        
        Parameters:
        -----------
        data_type : 数据类型
        fallback_func : 回退函数
        **kwargs : 额外参数
        
        Returns:
        --------
        数据
        """
        self._record_call('get_data_safely', data_type=data_type, **kwargs)
        
        # 测试模式：返回模拟数据
        if self._test_mode:
            mock_result = self._get_mock_result(data_type)
            if mock_result is not None:
                return mock_result
        
        # 使用注入的数据提供者
        if self._data_provider:
            try:
                if data_type == 'financial':
                    return self._data_provider.get_financial_data(**kwargs)
                elif data_type == 'market_cap':
                    return self._data_provider.get_market_cap_data(**kwargs)
                elif data_type == 'release_dates':
                    return self._data_provider.get_release_dates(**kwargs)
                elif data_type == 'trading_dates':
                    return self._data_provider.get_trading_dates(**kwargs)
            except Exception as e:
                logger.warning(f"Data provider failed for {data_type}: {e}")
        
        # 回退到默认方法
        if fallback_func:
            return fallback_func(**kwargs)
        
        raise ValueError(f"No data source available for {data_type}")
    
    def create_test_data(self, 
                        n_stocks: int = 5, 
                        n_periods: int = 4) -> Dict[str, Any]:
        """
        创建测试数据集
        
        Parameters:
        -----------
        n_stocks : 股票数量
        n_periods : 时间周期数
        
        Returns:
        --------
        测试数据字典
        """
        mock_provider = MockDataProvider()
        
        return {
            'financial_data': mock_provider.create_mock_financial_data(n_stocks, n_periods),
            'market_cap': mock_provider.create_mock_market_cap_data(n_stocks, n_periods * 30),
            'release_dates': mock_provider.create_mock_release_dates(n_stocks, n_periods),
            'trading_dates': mock_provider.create_mock_trading_dates()
        }
    
    def validate_calculation_invariants(self, 
                                      result: Union[pd.Series, pd.DataFrame],
                                      input_data: Dict[str, Any]) -> bool:
        """
        验证计算不变量（用于测试）
        
        Parameters:
        -----------
        result : 计算结果
        input_data : 输入数据
        
        Returns:
        --------
        是否满足不变量
        """
        try:
            # 基本检查
            if result is None or len(result) == 0:
                logger.error("Result is empty")
                return False
            
            # 检查是否有无穷大值
            if isinstance(result, (pd.Series, pd.DataFrame)):
                inf_count = np.isinf(result).sum()
                if isinstance(inf_count, pd.Series):
                    inf_count = inf_count.sum()
                if inf_count > 0:
                    logger.warning(f"Found {inf_count} infinite values")
            
            # 检查数据类型
            if isinstance(result, pd.Series):
                if not pd.api.types.is_numeric_dtype(result):
                    logger.error("Result should be numeric")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Invariant validation failed: {e}")
            return False


class FactorTestSuite:
    """因子测试套件"""
    
    def __init__(self, factor_class):
        self.factor_class = factor_class
        self.test_results = {}
    
    def run_basic_tests(self, **factor_kwargs) -> Dict[str, bool]:
        """
        运行基础测试
        
        Parameters:
        -----------
        **factor_kwargs : 因子构造参数
        
        Returns:
        --------
        测试结果字典
        """
        results = {}
        
        try:
            # 创建因子实例
            factor = self.factor_class(**factor_kwargs)
            
            # 测试1: 实例化测试
            results['instantiation'] = True
            
            # 测试2: 基本属性测试
            results['has_name'] = hasattr(factor, 'name') and factor.name is not None
            results['has_category'] = hasattr(factor, 'category') and factor.category is not None
            
            # 测试3: 如果是可测试的因子，测试计算
            if isinstance(factor, TestableMixin):
                factor.enable_test_mode()
                test_data = factor.create_test_data()
                
                try:
                    result = factor.calculate(**test_data)
                    results['calculation'] = result is not None
                    results['invariants'] = factor.validate_calculation_invariants(result, test_data)
                except Exception as e:
                    logger.error(f"Calculation test failed: {e}")
                    results['calculation'] = False
                    results['invariants'] = False
                
                factor.disable_test_mode()
            
        except Exception as e:
            logger.error(f"Basic tests failed: {e}")
            results['instantiation'] = False
        
        self.test_results.update(results)
        return results
    
    def run_data_variation_tests(self, **factor_kwargs) -> Dict[str, bool]:
        """
        运行数据变化测试
        
        Parameters:
        -----------
        **factor_kwargs : 因子构造参数
        
        Returns:
        --------
        测试结果字典
        """
        results = {}
        
        try:
            factor = self.factor_class(**factor_kwargs)
            
            if not isinstance(factor, TestableMixin):
                results['data_variation'] = False
                return results
            
            factor.enable_test_mode()
            
            # 测试不同规模的数据
            test_cases = [
                {'n_stocks': 3, 'n_periods': 2},
                {'n_stocks': 10, 'n_periods': 8},
                {'n_stocks': 50, 'n_periods': 12}
            ]
            
            all_passed = True
            for i, case in enumerate(test_cases):
                try:
                    test_data = factor.create_test_data(**case)
                    result = factor.calculate(**test_data)
                    case_passed = result is not None and len(result) > 0
                    results[f'case_{i+1}'] = case_passed
                    all_passed = all_passed and case_passed
                except Exception as e:
                    logger.error(f"Test case {i+1} failed: {e}")
                    results[f'case_{i+1}'] = False
                    all_passed = False
            
            results['data_variation'] = all_passed
            factor.disable_test_mode()
            
        except Exception as e:
            logger.error(f"Data variation tests failed: {e}")
            results['data_variation'] = False
        
        self.test_results.update(results)
        return results
    
    def get_test_summary(self) -> Dict[str, Any]:
        """获取测试摘要"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'results': self.test_results.copy()
        }