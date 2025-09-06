"""
自定义处理器示例

展示如何添加新的数据预处理功能
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from pathlib import Path

from .base_processor import BaseDataProcessor
from config import get_config


class TechnicalIndicatorProcessor(BaseDataProcessor):
    """技术指标处理器示例"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化技术指标处理器"""
        super().__init__(config_path)
        
    def validate_input(self, **kwargs) -> bool:
        """验证输入参数"""
        if 'price_df' not in kwargs:
            self.logger.error("缺少必需参数: price_df")
            return False
        return True
        
    def process(self, price_df: pd.DataFrame, indicators: list = None) -> Dict[str, pd.DataFrame]:
        """
        处理技术指标
        
        Args:
            price_df: 价格数据
            indicators: 要计算的指标列表
            
        Returns:
            指标字典
        """
        if not self.validate_input(price_df=price_df):
            raise ValueError("输入验证失败")
            
        # 默认计算所有指标
        if indicators is None:
            indicators = ['ma', 'rsi', 'volatility', 'volume_ratio']
            
        results = {}
        
        for indicator in indicators:
            self.logger.info(f"计算{indicator}...")
            
            if indicator == 'ma':
                results['ma_5'] = self.calculate_moving_average(price_df, 5)
                results['ma_20'] = self.calculate_moving_average(price_df, 20)
                results['ma_60'] = self.calculate_moving_average(price_df, 60)
                
            elif indicator == 'rsi':
                results['rsi_14'] = self.calculate_rsi(price_df, 14)
                
            elif indicator == 'volatility':
                results['volatility_20'] = self.calculate_volatility(price_df, 20)
                
            elif indicator == 'volume_ratio':
                results['volume_ratio_5'] = self.calculate_volume_ratio(price_df, 5)
                
        return results
        
    def calculate_moving_average(self, price_df: pd.DataFrame, window: int) -> pd.DataFrame:
        """
        计算移动平均
        
        Args:
            price_df: 价格数据
            window: 窗口大小
            
        Returns:
            移动平均DataFrame
        """
        # 按股票分组计算
        ma_list = []
        
        for stock in price_df.index.get_level_values(1).unique():
            stock_data = price_df.xs(stock, level=1)
            ma = stock_data['c'].rolling(window=window, min_periods=1).mean()
            ma.name = f'ma_{window}'
            
            # 重建MultiIndex
            ma_df = pd.DataFrame(ma)
            ma_df['code'] = stock
            ma_df = ma_df.set_index('code', append=True)
            ma_list.append(ma_df)
            
        result = pd.concat(ma_list)
        result.index.names = ['TradingDates', 'StockCodes']
        
        return result
        
    def calculate_rsi(self, price_df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        计算RSI指标
        
        Args:
            price_df: 价格数据
            period: RSI周期
            
        Returns:
            RSI DataFrame
        """
        rsi_list = []
        
        for stock in price_df.index.get_level_values(1).unique():
            stock_data = price_df.xs(stock, level=1)
            
            # 计算价格变化
            delta = stock_data['c'].diff()
            
            # 分离涨跌
            gain = (delta.where(delta > 0, 0))
            loss = (-delta.where(delta < 0, 0))
            
            # 计算平均涨跌幅
            avg_gain = gain.rolling(window=period, min_periods=1).mean()
            avg_loss = loss.rolling(window=period, min_periods=1).mean()
            
            # 计算RSI
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            rsi.name = f'rsi_{period}'
            
            # 重建MultiIndex
            rsi_df = pd.DataFrame(rsi)
            rsi_df['code'] = stock
            rsi_df = rsi_df.set_index('code', append=True)
            rsi_list.append(rsi_df)
            
        result = pd.concat(rsi_list)
        result.index.names = ['TradingDates', 'StockCodes']
        
        return result
        
    def calculate_volatility(self, price_df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        计算收益率波动率
        
        Args:
            price_df: 价格数据
            window: 窗口大小
            
        Returns:
            波动率DataFrame
        """
        vol_list = []
        
        for stock in price_df.index.get_level_values(1).unique():
            stock_data = price_df.xs(stock, level=1)
            
            # 计算对数收益率
            log_returns = np.log(stock_data['c'] / stock_data['c'].shift(1))
            
            # 计算滚动标准差（年化）
            volatility = log_returns.rolling(window=window, min_periods=2).std() * np.sqrt(252)
            volatility.name = f'volatility_{window}'
            
            # 重建MultiIndex
            vol_df = pd.DataFrame(volatility)
            vol_df['code'] = stock
            vol_df = vol_df.set_index('code', append=True)
            vol_list.append(vol_df)
            
        result = pd.concat(vol_list)
        result.index.names = ['TradingDates', 'StockCodes']
        
        return result
        
    def calculate_volume_ratio(self, price_df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """
        计算成交量比率
        
        Args:
            price_df: 价格数据
            window: 窗口大小
            
        Returns:
            成交量比率DataFrame
        """
        vr_list = []
        
        for stock in price_df.index.get_level_values(1).unique():
            stock_data = price_df.xs(stock, level=1)
            
            # 计算成交量移动平均
            volume_ma = stock_data['v'].rolling(window=window, min_periods=1).mean()
            
            # 计算成交量比率
            volume_ratio = stock_data['v'] / volume_ma
            volume_ratio.name = f'volume_ratio_{window}'
            
            # 重建MultiIndex
            vr_df = pd.DataFrame(volume_ratio)
            vr_df['code'] = stock
            vr_df = vr_df.set_index('code', append=True)
            vr_list.append(vr_df)
            
        result = pd.concat(vr_list)
        result.index.names = ['TradingDates', 'StockCodes']
        
        return result


# 示例：如何集成到管道中
class ExtendedProcessingPipeline(DataProcessingPipeline):
    """扩展的处理管道示例"""
    
    def __init__(self, config_path: Optional[str] = None):
        super().__init__(config_path)
        # 添加技术指标处理器
        self.tech_processor = TechnicalIndicatorProcessor(config_path)
        
    def run_extended_pipeline(self, save_intermediate: bool = True) -> Dict[str, Any]:
        """运行扩展的处理流程"""
        
        # 1. 运行基础处理
        self.logger.info("运行基础数据处理...")
        results = self.run_full_pipeline(save_intermediate=False)
        
        # 2. 计算技术指标
        self.logger.info("计算技术指标...")
        price_df = results.get('price_df')
        
        if price_df is not None:
            tech_indicators = self.tech_processor.process(
                price_df,
                indicators=['ma', 'rsi', 'volatility', 'volume_ratio']
            )
            
            # 保存技术指标
            if save_intermediate:
                for name, indicator_df in tech_indicators.items():
                    save_path = self.data_save_path / f"{name}.pkl"
                    pd.to_pickle(indicator_df, save_path)
                    self.logger.info(f"{name} 已保存至: {save_path}")
                    
            results['technical_indicators'] = tech_indicators
            
        # 3. 可以继续添加其他处理步骤...
        
        return results


# 使用示例
if __name__ == "__main__":
    # 方式1：单独使用技术指标处理器
    processor = TechnicalIndicatorProcessor()
    
    # 加载价格数据
    price_file = Path(get_config('main.paths.data_root')) / "Price.pkl"
    price_df = pd.read_pickle(price_file)
    
    # 计算指标
    indicators = processor.process(price_df, indicators=['ma', 'rsi'])
    
    # 方式2：使用扩展管道
    pipeline = ExtendedProcessingPipeline()
    results = pipeline.run_extended_pipeline()
    
    print("技术指标计算完成！")