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
from .sector_classification_processor import SectorClassificationProcessor
from ..processor.base_processor import BaseDataProcessor
from config import get_config


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
        self.sector_classification_processor = SectorClassificationProcessor()
        
        # 数据保存路径
        self.data_save_path = Path(get_config('main.paths.data_root'))
        
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
            
            # 2. 加载预处理的收益率数据（收益率计算已移到预处理阶段）
            self.logger.info("步骤2: 加载预处理的收益率数据...")
            auxiliary_path = self.data_save_path / "auxiliary"
            
            # 日收益率 - o2o
            log_return_daily_path = auxiliary_path / "LogReturn_daily_o2o.pkl"
            if log_return_daily_path.exists():
                log_return_daily = pd.read_pickle(log_return_daily_path)
                results['log_return_daily_o2o'] = log_return_daily
                self.logger.info(f"日收益率(o2o)已加载: {log_return_daily_path}")
            else:
                self.logger.warning(f"日收益率文件不存在: {log_return_daily_path}")
                log_return_daily = None
            
            # 其他收益率数据
            return_files = {
                'log_return_daily_vwap': 'LogReturn_daily_vwap.pkl',
                'log_return_weekly': 'LogReturn_weekly_o2o.pkl',
                'log_return_monthly': 'LogReturn_monthly_o2o.pkl',
                'log_return_5days': 'LogReturn_5days_o2o.pkl',
                'log_return_20days': 'LogReturn_20days_o2o.pkl'
            }
            
            for key, filename in return_files.items():
                filepath = auxiliary_path / filename
                if filepath.exists():
                    results[key] = pd.read_pickle(filepath)
                    self.logger.info(f"已加载: {filename}")
                else:
                    self.logger.warning(f"收益率文件不存在: {filepath}")
            
            # 3. 处理财报数据
            self.logger.info("步骤3: 处理财报数据...")
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
            
            # 4. 计算财报发布后收益率
            self.logger.info("步骤4: 计算财报发布后收益率...")
            
            # 从auxiliary目录加载日收益率数据
            log_return_daily_o2o = pd.read_pickle(
                self.data_save_path / "auxiliary" / "LogReturn_daily_o2o.pkl"
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
            
            # 5. 计算股票分类信息
            self.logger.info("步骤5: 计算股票分类信息...")
            try:
                # 获取最新的股票分类信息
                from datetime import datetime
                latest_date = datetime.now().strftime('%Y-%m-%d')
                
                # 计算完整的分类信息
                stock_classification = self.sector_classification_processor.calculate_sector_classification_at_date(latest_date)
                
                if not stock_classification.empty:
                    if save_intermediate:
                        save_path = self.data_save_path / "StockClassification_latest.pkl"
                        stock_classification.to_pickle(save_path)
                        self.logger.info(f"股票分类信息已保存至: {save_path}")
                    
                    results['stock_classification'] = stock_classification
                    
                    # 分别保存申万行业分类
                    sw_classification = stock_classification[
                        stock_classification['指数类型'] == '申万行业板块'
                    ]
                    if not sw_classification.empty and save_intermediate:
                        save_path = self.data_save_path / "SW_Industry_Classification.pkl"
                        sw_classification.to_pickle(save_path)
                        self.logger.info(f"申万行业分类信息已保存至: {save_path}")
                    
                    # 获取分类统计
                    classification_stats = self.sector_classification_processor.get_classification_stats(latest_date)
                    self.logger.info(f"分类统计: {classification_stats}")
                    results['classification_stats'] = classification_stats
                    
                else:
                    self.logger.warning("未获取到股票分类信息")
                    
            except Exception as e:
                self.logger.error(f"股票分类计算失败: {e}")
                # 分类计算失败不影响整个管道的执行
            
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
    
    def calculate_stock_classification(self, target_date: Optional[str] = None, 
                                     concept_types: Optional[list] = None,
                                     export_format: str = 'pkl') -> pd.DataFrame:
        """
        独立的股票分类计算方法
        
        Args:
            target_date: 目标日期，None表示使用当前日期
            concept_types: 概念类型列表，None表示计算所有类型
            export_format: 导出格式，'pkl'、'csv'、'excel'
            
        Returns:
            股票分类信息DataFrame
        """
        if target_date is None:
            from datetime import datetime
            target_date = datetime.now().strftime('%Y-%m-%d')
        
        self.logger.info(f"计算股票分类信息: {target_date}")
        
        # 计算分类信息
        classification_df = self.sector_classification_processor.calculate_sector_classification_at_date(
            target_date, concept_types
        )
        
        # 保存结果
        if not classification_df.empty:
            output_path = self.sector_classification_processor.export_classification_data(
                target_date, format=export_format
            )
            self.logger.info(f"分类数据已导出: {output_path}")
        
        return classification_df
    
    def get_stock_sector_history(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取指定股票的历史分类变化
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            历史分类变化DataFrame
        """
        self.logger.info(f"获取股票 {stock_code} 的历史分类: {start_date} - {end_date}")
        
        # 获取时间序列分类数据
        series_classification = self.sector_classification_processor.calculate_sector_classification_series(
            start_date, end_date, frequency='M'  # 按月计算
        )
        
        # 提取指定股票的数据
        stock_history = []
        for date_str, classification_df in series_classification.items():
            if not classification_df.empty:
                stock_data = classification_df[classification_df['code'] == stock_code]
                if not stock_data.empty:
                    stock_data = stock_data.copy()
                    stock_data['date'] = date_str
                    stock_history.append(stock_data)
        
        if stock_history:
            result = pd.concat(stock_history, ignore_index=True)
            result = result.sort_values(['date', '指数类型', 'concept_code'])
        else:
            result = pd.DataFrame()
        
        return result


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