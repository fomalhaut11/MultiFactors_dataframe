"""
数据处理管道

整合所有数据处理功能，提供向后兼容的接口
"""
import os
import gc
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
import logging

from .price_processor import PriceDataProcessor
from .return_calculator import ReturnCalculator
from .financial_processor import FinancialDataProcessor
from ..processor.base_processor import BaseDataProcessor
from core.config_manager import get_path


class DataProcessingPipeline(BaseDataProcessor):
    """数据处理管道，整合所有处理功能"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化数据处理管道
        
        Args:
            config_path: 配置文件路径
        """
        super().__init__(config_path)
        
        # 初始化各个处理器
        self.price_processor = PriceDataProcessor(config_path)
        self.return_calculator = ReturnCalculator(config_path)
        self.financial_processor = FinancialDataProcessor(config_path)
        
        # 数据保存路径
        self.data_save_path = Path(get_path('data_root'))
        
    def validate_input(self, **kwargs) -> bool:
        """验证输入参数"""
        return True  # 管道级别的验证由各子处理器完成
        
    def process(self, **kwargs):
        """管道不使用通用process方法"""
        return self.run_full_pipeline(**kwargs)
        
    def run_full_pipeline(self, save_intermediate: bool = True) -> Dict[str, Any]:
        """
        运行完整的数据处理流程
        与原始run()函数完全兼容
        
        Args:
            save_intermediate: 是否保存中间结果
            
        Returns:
            处理结果字典
        """
        self.logger.info("开始运行数据处理管道...")
        results = {}
        
        try:
            # 1. 处理价格数据
            self.logger.info("步骤1: 处理价格数据...")
            price_df, stock_3d = self.price_processor.process(save_to_file=save_intermediate)
            results['price_df'] = price_df
            results['stock_3d'] = stock_3d
            
            # 2. 生成日期序列
            self.logger.info("步骤2: 生成日期序列...")
            daily_series = self.price_processor.get_date_series(price_df, "daily")
            weekly_series = self.price_processor.get_date_series(price_df, "weekly")
            monthly_series = self.price_processor.get_date_series(price_df, "monthly")
            
            # 3. 计算各种收益率
            self.logger.info("步骤3: 计算日收益率...")
            
            # 日收益率 - o2o
            log_return_daily = self.return_calculator.calculate_log_return(
                price_df, daily_series, return_type="o2o"
            )
            if save_intermediate:
                save_path = self.data_save_path / "LogReturn_daily_o2o.pkl"
                pd.to_pickle(log_return_daily, save_path)
                self.logger.info(f"日收益率(o2o)已保存至: {save_path}")
            results['log_return_daily_o2o'] = log_return_daily
            
            # N天滚动收益率
            self.logger.info("计算20天滚动收益率...")
            log_return_20days = self.return_calculator.calculate_n_days_return(
                log_return_daily, lag=20
            )
            if save_intermediate:
                save_path = self.data_save_path / "LogReturn_20days_o2o.pkl"
                pd.to_pickle(log_return_20days, save_path)
            
            self.logger.info("计算5天滚动收益率...")
            log_return_5days = self.return_calculator.calculate_n_days_return(
                log_return_daily, lag=5
            )
            if save_intermediate:
                save_path = self.data_save_path / "LogReturn_5days_o2o.pkl"
                pd.to_pickle(log_return_5days, save_path)
            
            # 日收益率 - vwap
            self.logger.info("计算日收益率(vwap)...")
            log_return_daily_vwap = self.return_calculator.calculate_log_return(
                price_df, daily_series, return_type="vwap"
            )
            if save_intermediate:
                save_path = self.data_save_path / "LogReturn_daily_vwap.pkl"
                pd.to_pickle(log_return_daily_vwap, save_path)
            
            # 释放内存
            log_return_daily = None
            gc.collect()
            
            # 周收益率
            self.logger.info("计算周收益率...")
            log_return_weekly = self.return_calculator.calculate_log_return(
                price_df, weekly_series, return_type="o2o"
            )
            if save_intermediate:
                save_path = self.data_save_path / "LogReturn_weekly_o2o.pkl"
                pd.to_pickle(log_return_weekly, save_path)
                
            log_return_weekly_vwap = self.return_calculator.calculate_log_return(
                price_df, weekly_series, return_type="vwap"
            )
            if save_intermediate:
                save_path = self.data_save_path / "LogReturn_weekly_vwap.pkl"
                pd.to_pickle(log_return_weekly_vwap, save_path)
            
            log_return_weekly = None
            
            # 月收益率
            self.logger.info("计算月收益率...")
            log_return_monthly = self.return_calculator.calculate_log_return(
                price_df, monthly_series, return_type="o2o"
            )
            if save_intermediate:
                save_path = self.data_save_path / "LogReturn_monthly_o2o.pkl"
                pd.to_pickle(log_return_monthly, save_path)
                
            log_return_monthly_vwap = self.return_calculator.calculate_log_return(
                price_df, monthly_series, return_type="vwap"
            )
            if save_intermediate:
                save_path = self.data_save_path / "LogReturn_monthly_vwap.pkl"
                pd.to_pickle(log_return_monthly_vwap, save_path)
            
            log_return_monthly = None
            
            # 4. 处理财报数据
            self.logger.info("步骤4: 处理财报数据...")
            trading_dates0 = price_df.index.get_level_values(0).unique().tolist()
            
            # 释放价格数据内存
            price_df = None
            gc.collect()
            
            # 获取财报发布日期
            released_dates_df = self.financial_processor.get_released_dates_from_h5()
            if save_intermediate:
                save_path = self.data_save_path / "released_dates_df.pkl"
                released_dates_df.to_pickle(save_path)
            
            # 计算财报发布时间差
            trading_dates = pd.DataFrame(trading_dates0, columns=["date"])
            released_dates_count_df = self.financial_processor.calculate_released_dates_count(
                released_dates_df, trading_dates
            )
            if save_intermediate:
                save_path = self.data_save_path / "released_dates_count_df.pkl"
                released_dates_count_df.to_pickle(save_path)
            
            # 5. 计算财报发布后收益率
            self.logger.info("步骤5: 计算财报发布后收益率...")
            
            # 重新加载日收益率数据
            log_return_daily_o2o = pd.read_pickle(
                self.data_save_path / "LogReturn_daily_o2o.pkl"
            )
            
            # 20天
            lag20return, lag20alfareturn = self.return_calculator.calculate_return_after_release(
                log_return_daily_o2o, released_dates_df, lag=20
            )
            if save_intermediate:
                lag20return.to_pickle(self.data_save_path / "lag20_released_logreturn.pkl")
                lag20alfareturn.to_pickle(self.data_save_path / "lag20_released_alfa_logreturn.pkl")
            
            # 5天
            lag5return, lag5alfareturn = self.return_calculator.calculate_return_after_release(
                log_return_daily_o2o, released_dates_df, lag=5
            )
            if save_intermediate:
                lag5return.to_pickle(self.data_save_path / "lag5_released_logreturn.pkl")
                lag5alfareturn.to_pickle(self.data_save_path / "lag5_released_alfa_logreturn.pkl")
            
            # 1天
            lag1return, lag1alfareturn = self.return_calculator.calculate_return_after_release(
                log_return_daily_o2o, released_dates_df, lag=1
            )
            if save_intermediate:
                lag1return.to_pickle(self.data_save_path / "lag1_released_logreturn.pkl")
                lag1alfareturn.to_pickle(self.data_save_path / "lag1_released_alfa_logreturn.pkl")
            
            self.logger.info("数据处理管道执行完成！")
            
            # 记录处理历史
            self._record_processing(
                'full_pipeline',
                {'save_intermediate': save_intermediate},
                {'status': 'success', 'steps_completed': 5}
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"数据处理管道执行失败: {e}")
            raise


# 向后兼容的函数接口
def run(datasavepath: Optional[str] = None):
    """
    向后兼容的运行函数
    
    Args:
        datasavepath: 数据保存路径
    """
    # 如果提供了自定义路径，需要创建临时配置
    if datasavepath:
        import yaml
        config = {
            'paths': {
                'data_root': datasavepath
            }
        }
        # 创建临时配置文件
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_config_path = f.name
            
        pipeline = DataProcessingPipeline(temp_config_path)
        
        # 清理临时文件
        os.unlink(temp_config_path)
    else:
        pipeline = DataProcessingPipeline()
        
    # 运行管道
    pipeline.run_full_pipeline(save_intermediate=True)
    

# 向后兼容的独立函数
def get_price_data(savepath: Optional[str] = None):
    """向后兼容的价格数据获取函数"""
    processor = PriceDataProcessor()
    return processor.process(save_to_file=True)
    

def date_serries(price_df: pd.DataFrame, type: str = "daily"):
    """向后兼容的日期序列函数"""
    processor = PriceDataProcessor()
    return processor.get_date_series(price_df, type)
    

def logreturndf_dateserries(price_df: pd.DataFrame, 
                          datesserries: pd.DatetimeIndex,
                          ReturnType: str = "o2o",
                          inputtype: str = "unadjusted"):
    """向后兼容的收益率计算函数"""
    calculator = ReturnCalculator()
    return calculator.calculate_log_return(
        price_df, datesserries, return_type=ReturnType, input_type=inputtype
    )