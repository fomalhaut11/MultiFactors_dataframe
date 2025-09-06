#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票池管理器
统一管理股票池的创建、加载、验证和缓存
支持多种数据来源：实时计算、文件导入、预定义池
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Literal
import logging
from pathlib import Path
import json
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class UniverseMetadata:
    """股票池元数据"""
    name: str
    description: str
    stock_count: int
    created_date: str
    data_source: str
    criteria: Dict
    last_updated: Optional[str] = None


class StockUniverseManager:
    """
    股票池管理器
    
    设计理念：
    1. 支持多种数据来源：计算生成、文件导入、预定义
    2. 智能缓存机制：避免重复计算
    3. 格式标准化：统一股票代码格式
    4. 元数据管理：记录股票池的创建标准和统计信息
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化股票池管理器
        
        Parameters
        ----------
        config : Dict, optional
            配置参数
        """
        self.config = config or {}
        self.cache_dir = Path(self.config.get('cache_dir', './cache/stock_universes'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 股票池缓存
        self._universe_cache = {}
        
        # 预定义股票池
        self._predefined_universes = self._load_predefined_universes()
        
        logger.info("股票池管理器初始化完成")
    
    def get_universe(self, 
                    universe_name: str,
                    refresh: bool = False,
                    format_type: str = 'multiindex',
                    **kwargs) -> Union[List[str], pd.Series]:
        """
        获取股票池
        
        Parameters
        ----------
        universe_name : str
            股票池名称，支持：
            - 'full' / None : 全市场股票
            - 'liquid_1000' : 流动性前1000只
            - 'large_cap_500' : 大盘股前500只
            - 'main_board' : 主板股票（排除ST）
            - 文件路径 : 从文件加载
            - 预定义名称 : 从预定义配置加载
        refresh : bool
            是否强制刷新缓存
        format_type : str
            返回格式：'list' 或 'multiindex_series'
        **kwargs
            额外参数
            
        Returns
        -------
        List[str] or pd.Series
            股票池，格式根据format_type确定：
            - 'list': List[str] 股票代码列表
            - 'multiindex_series': MultiIndex[TradingDates, StockCodes] Series，值为1
        """
        # 更新缓存键以包含格式类型
        cache_key = f"{universe_name}_{format_type}_{hash(str(kwargs))}"
        if not refresh and cache_key in self._universe_cache:
            logger.info(f"从缓存加载股票池: {universe_name} ({format_type})")
            return self._universe_cache[cache_key]
        
        # 处理全市场股票池（特殊情况，直接支持MultiIndex Series）
        if universe_name is None or universe_name in ['full', 'all']:
            if format_type == 'list':
                result = self._get_full_market_universe_as_list()
            else:  # multiindex_series or multiindex
                result = self._get_full_market_universe()
            
            self._universe_cache[cache_key] = result
            if format_type == 'list':
                logger.info(f"股票池 '{universe_name}' 加载完成，股票数量: {len(result)}")
            else:
                stocks_count = len(result.index.get_level_values(1).unique()) if len(result) > 0 else 0
                logger.info(f"股票池 '{universe_name}' 加载完成，股票数量: {stocks_count}")
            return result
        
        # 其他类型股票池：智能处理不同格式
        if self._is_file_path(universe_name):
            # 检查是否是pickle文件且需要MultiIndex格式
            file_path = Path(universe_name)
            if file_path.suffix.lower() == '.pkl' and format_type in ['multiindex_series', 'multiindex']:
                try:
                    # 直接加载MultiIndex Series
                    result = self._load_universe_series_from_file(universe_name)
                    self._universe_cache[cache_key] = result
                    stocks_count = len(result.index.get_level_values(1).unique()) if len(result) > 0 else 0
                    logger.info(f"股票池 '{universe_name}' 加载完成，股票数量: {stocks_count} (直接从文件加载MultiIndex Series)")
                    return result
                except Exception as e:
                    logger.warning(f"直接加载MultiIndex Series失败，降级到List格式: {e}")
            
            # 默认处理：加载为List格式
            stocks = self._load_from_file(universe_name)
        elif universe_name in self._predefined_universes:
            stocks = self._get_predefined_universe(universe_name, **kwargs)
        elif universe_name.startswith('liquid_'):
            count = self._extract_number_from_name(universe_name, default=1000)
            stocks = self._get_liquid_universe(count, **kwargs)
        elif universe_name.startswith('large_cap_'):
            count = self._extract_number_from_name(universe_name, default=500)
            stocks = self._get_large_cap_universe(count, **kwargs)
        elif universe_name == 'main_board':
            stocks = self._get_main_board_universe(**kwargs)
        else:
            logger.warning(f"未知股票池类型: {universe_name}, 使用全市场")
            if format_type == 'list':
                result = self._get_full_market_universe_as_list()
            else:
                result = self._get_full_market_universe()
            self._universe_cache[cache_key] = result
            return result
        
        # 标准化股票代码
        stocks = self._standardize_stock_codes(stocks)
        
        # 根据format_type返回相应格式
        if format_type in ['multiindex_series', 'multiindex']:
            result = self._convert_to_multiindex_series(stocks, **kwargs)
            stocks_count = len(result.index.get_level_values(1).unique()) if len(result) > 0 else 0
            logger.info(f"股票池 '{universe_name}' 加载完成，股票数量: {stocks_count} (MultiIndex Series)")
        else:  # format_type == 'list'
            result = stocks
            logger.info(f"股票池 '{universe_name}' 加载完成，股票数量: {len(result)} (List)")
        
        # 缓存结果
        self._universe_cache[cache_key] = result
        
        # 保存元数据（总是使用股票列表）
        stock_list = stocks if format_type == 'list' else stocks
        self._save_universe_metadata(universe_name, stock_list, kwargs)
        
        return result
    
    def _get_full_market_universe(self) -> pd.Series:
        """获取全市场股票池，返回MultiIndex Series格式"""
        cache_key = 'full_market_universe_series'
        if cache_key in self._universe_cache:
            logger.debug("从缓存加载全市场股票池")
            return self._universe_cache[cache_key]
            
        try:
            from factors.utils.data_loader import get_price_data
            
            price_data = get_price_data()
            
            if not isinstance(price_data.index, pd.MultiIndex):
                logger.error("价格数据必须是MultiIndex格式")
                return pd.Series(dtype=int, name='universe')
            
            # 创建全市场股票池：所有股票在所有有效交易日都包含在池中
            # 直接使用价格数据的index，去除重复并设值为1
            universe_index = price_data.index.drop_duplicates()
            universe_series = pd.Series(
                data=1,
                index=universe_index,
                name='universe'
            )
            
            self._universe_cache[cache_key] = universe_series
            stocks_count = len(universe_series.index.get_level_values(1).unique())
            dates_count = len(universe_series.index.get_level_values(0).unique())
            
            logger.info(f"全市场股票池: {stocks_count} 只股票, {dates_count} 个交易日, 总数据点: {len(universe_series)}")
            return universe_series
            
        except Exception as e:
            logger.error(f"获取全市场股票池失败: {e}")
            return pd.Series(dtype=int, name='universe')
    
    def _get_full_market_universe_as_list(self) -> List[str]:
        """获取全市场股票池，返回List[str]格式（向后兼容）"""
        try:
            universe_series = self._get_full_market_universe()
            if len(universe_series) == 0:
                return []
            stocks = universe_series.index.get_level_values(1).unique().tolist()
            logger.info(f"全市场股票数: {len(stocks)} 只股票")
            return stocks
        except Exception as e:
            logger.error(f"获取全市场股票列表失败: {e}")
            return []
    
    def _get_liquid_universe(self, count: int, lookback_days: int = 60) -> List[str]:
        """
        获取流动性股票池（按成交量排序）
        
        Parameters
        ----------
        count : int
            股票数量
        lookback_days : int
            回看天数
            
        Returns
        -------
        List[str]
            股票代码列表
        """
        try:
            from factors.utils.data_loader import get_price_data
            
            price_data = get_price_data()
            
            # 计算最近lookback_days的平均成交量
            if isinstance(price_data.index, pd.MultiIndex):
                # 获取最近的交易日
                latest_dates = price_data.index.get_level_values(0).unique()
                latest_dates = sorted(latest_dates)[-lookback_days:]
                
                # 过滤最近数据
                recent_data = price_data.loc[latest_dates]
                
                # 计算平均成交量（假设成交量列名）
                volume_cols = [col for col in recent_data.columns if 'vol' in col.lower() or 'v' == col.lower()]
                if volume_cols:
                    volume_col = volume_cols[0]
                    avg_volume = recent_data.groupby(level=1)[volume_col].mean()
                    
                    # 按成交量降序排列
                    top_liquid_stocks = avg_volume.nlargest(count).index.tolist()
                    
                    logger.info(f"流动性前{count}只股票池创建完成")
                    return top_liquid_stocks
                else:
                    logger.warning("未找到成交量数据，使用全市场")
                    return self._get_full_market_universe()[:count]
            else:
                logger.warning("价格数据格式不支持流动性计算")
                return self._get_full_market_universe()[:count]
                
        except Exception as e:
            logger.error(f"创建流动性股票池失败: {e}")
            return self._get_full_market_universe()[:count]
    
    def _get_large_cap_universe(self, count: int) -> List[str]:
        """
        获取大盘股票池（按市值排序）
        
        Parameters
        ----------
        count : int
            股票数量
            
        Returns
        -------
        List[str]
            股票代码列表
        """
        try:
            from factors.utils.data_loader import get_market_cap
            
            market_cap = get_market_cap()
            
            # 获取最新市值数据
            if isinstance(market_cap.index, pd.MultiIndex):
                latest_date = market_cap.index.get_level_values(0).max()
                latest_cap = market_cap.loc[latest_date]
                
                # 按市值降序排列
                top_cap_stocks = latest_cap.nlargest(count).index.tolist()
                
                logger.info(f"大盘股前{count}只股票池创建完成")
                return top_cap_stocks
            else:
                logger.warning("市值数据格式不支持排序")
                return self._get_full_market_universe()[:count]
                
        except Exception as e:
            logger.error(f"创建大盘股股票池失败: {e}")
            return self._get_full_market_universe()[:count]
    
    def _get_main_board_universe(self, exclude_st: bool = True) -> List[str]:
        """
        获取主板股票池
        
        Parameters
        ----------
        exclude_st : bool
            是否排除ST股票
            
        Returns
        -------
        List[str]
            股票代码列表
        """
        full_universe = self._get_full_market_universe()
        
        # 简单的主板筛选逻辑（实际项目中应该更复杂）
        main_board_stocks = []
        
        for stock in full_universe:
            stock_str = str(stock)
            
            # 排除ST股票（简单规则）
            if exclude_st and ('ST' in stock_str or 'st' in stock_str):
                continue
            
            # 其他主板筛选条件可以在这里添加
            # 例如：排除退市风险股票、停牌股票等
            
            main_board_stocks.append(stock)
        
        logger.info(f"主板股票池创建完成: {len(main_board_stocks)} 只股票")
        return main_board_stocks
    
    def _load_from_file(self, file_path: str) -> List[str]:
        """从文件加载股票池"""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"股票池文件不存在: {file_path}")
        
        try:
            if path.suffix.lower() == '.csv':
                # CSV格式：假设第一列是股票代码
                df = pd.read_csv(path)
                stocks = df.iloc[:, 0].tolist()
            elif path.suffix.lower() == '.json':
                # JSON格式
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    stocks = data
                elif isinstance(data, dict) and 'stocks' in data:
                    stocks = data['stocks']
                else:
                    raise ValueError("JSON格式不正确")
            elif path.suffix.lower() == '.txt':
                # 文本格式：每行一个股票代码
                with open(path, 'r', encoding='utf-8') as f:
                    stocks = [line.strip() for line in f if line.strip()]
            elif path.suffix.lower() == '.pkl':
                # Pickle格式：支持MultiIndex Series股票池
                data = pd.read_pickle(path)
                if isinstance(data, pd.Series) and isinstance(data.index, pd.MultiIndex):
                    # MultiIndex Series格式：提取股票代码
                    stocks = data.index.get_level_values(1).unique().tolist()
                    logger.info(f"从MultiIndex Series中提取 {len(stocks)} 只股票")
                elif isinstance(data, list):
                    # List格式
                    stocks = data
                elif isinstance(data, pd.Series):
                    # 普通Series格式
                    stocks = data.tolist()
                else:
                    raise ValueError(f"Pickle文件格式不支持: {type(data)}")
            else:
                raise ValueError(f"不支持的文件格式: {path.suffix}")
            
            logger.info(f"从文件加载股票池: {file_path}, 股票数量: {len(stocks)}")
            return stocks
            
        except Exception as e:
            logger.error(f"从文件加载股票池失败 {file_path}: {e}")
            return []
    
    def _load_universe_series_from_file(self, file_path: str) -> pd.Series:
        """
        从pickle文件加载MultiIndex Series格式的股票池
        
        Parameters
        ----------
        file_path : str
            股票池文件路径
            
        Returns
        -------
        pd.Series
            MultiIndex[TradingDates, StockCodes] Series，值为1
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"股票池文件不存在: {file_path}")
        
        if path.suffix.lower() != '.pkl':
            raise ValueError(f"此方法仅支持pickle格式文件，实际: {path.suffix}")
        
        try:
            data = pd.read_pickle(path)
            
            if isinstance(data, pd.Series) and isinstance(data.index, pd.MultiIndex):
                logger.info(f"从文件加载MultiIndex Series股票池: {file_path}")
                logger.info(f"数据形状: {data.shape}, 交易日数: {len(data.index.get_level_values(0).unique())}, 股票数: {len(data.index.get_level_values(1).unique())}")
                return data
            else:
                raise ValueError(f"文件内容不是MultiIndex Series格式: {type(data)}")
                
        except Exception as e:
            logger.error(f"从文件加载MultiIndex Series股票池失败 {file_path}: {e}")
            raise
    
    def save_universe(self, 
                     universe_name: str,
                     stocks: List[str],
                     description: str = "",
                     criteria: Dict = None) -> bool:
        """
        保存股票池到文件
        
        Parameters
        ----------
        universe_name : str
            股票池名称
        stocks : List[str]
            股票代码列表
        description : str
            描述信息
        criteria : Dict
            创建标准
            
        Returns
        -------
        bool
            是否保存成功
        """
        try:
            # 保存股票列表
            stocks_file = self.cache_dir / f"{universe_name}.json"
            
            data = {
                'name': universe_name,
                'description': description,
                'stocks': stocks,
                'stock_count': len(stocks),
                'created_date': datetime.now().isoformat(),
                'criteria': criteria or {}
            }
            
            with open(stocks_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"股票池保存成功: {stocks_file}")
            return True
            
        except Exception as e:
            logger.error(f"股票池保存失败: {e}")
            return False
    
    def _standardize_stock_codes(self, stocks: List[str]) -> List[str]:
        """标准化股票代码格式"""
        standardized = []
        
        for stock in stocks:
            stock_str = str(stock).strip()
            
            # 根据实际数据格式进行标准化
            # 当前系统中股票代码是纯数字，保持原样
            if stock_str:
                standardized.append(stock_str)
        
        return standardized
    
    def _load_predefined_universes(self) -> Dict:
        """加载预定义股票池配置"""
        from config import get_config
        
        # 获取股票池专用路径
        try:
            universe_root = Path(get_config('main.paths.stock_universe'))
        except:
            # 向后兼容：如果没有配置股票池路径，使用data_root下的stock_universe目录
            data_root = Path(get_config('main.paths.data_root'))
            universe_root = data_root / 'stock_universe'
        
        return {
            'a_share_main': {
                'description': 'A股主板股票',
                'criteria': {'board': 'main', 'exclude_st': True}
            },
            'index_300': {
                'description': '沪深300成分股',
                'criteria': {'index': 'hs300'}
            },
            'index_500': {
                'description': '中证500成分股',  
                'criteria': {'index': 'zz500'}
            },
            'high_turnover_5d_20m': {
                'description': '高成交额股票池（5日平均成交额超过2000万元）',
                'criteria': {'turnover_5d_avg': 20000000},
                'file_path': str(universe_root / 'high_turnover_5d_20m.pkl'),
                'format': 'multiindex_series'  # 标记这是MultiIndex Series格式
            }
        }
    
    def _get_predefined_universe(self, name: str, **kwargs) -> List[str]:
        """获取预定义股票池"""
        if name not in self._predefined_universes:
            logger.warning(f"预定义股票池 '{name}' 不存在，使用全市场")
            return self._get_full_market_universe_as_list()
        
        config = self._predefined_universes[name]
        
        # 处理文件型预定义股票池
        if 'file_path' in config:
            file_path = config['file_path']
            if Path(file_path).exists():
                try:
                    logger.info(f"从文件加载预定义股票池: {name}")
                    return self._load_from_file(file_path)
                except Exception as e:
                    logger.error(f"加载预定义股票池文件失败: {e}")
        
        # 其他预定义股票池（暂未实现）
        logger.warning(f"预定义股票池 '{name}' 暂未实现，使用全市场")
        return self._get_full_market_universe_as_list()
    
    def _save_universe_metadata(self, name: str, stocks: List[str], criteria: Dict):
        """保存股票池元数据"""
        metadata = UniverseMetadata(
            name=name,
            description=f"Stock universe: {name}",
            stock_count=len(stocks),
            created_date=datetime.now().isoformat(),
            data_source="computed",
            criteria=criteria
        )
        
        metadata_file = self.cache_dir / f"{name}_metadata.json"
        
        try:
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata.__dict__, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"保存元数据失败: {e}")
    
    def _is_file_path(self, name: str) -> bool:
        """判断是否为文件路径"""
        return ('/' in name or '\\' in name or 
                name.endswith('.csv') or 
                name.endswith('.json') or 
                name.endswith('.txt'))
    
    def _extract_number_from_name(self, name: str, default: int) -> int:
        """从名称中提取数字"""
        import re
        numbers = re.findall(r'\d+', name)
        return int(numbers[0]) if numbers else default
    
    def list_available_universes(self) -> Dict[str, str]:
        """列出可用的股票池"""
        universes = {
            'full': '全市场股票（默认）',
            'liquid_1000': '流动性前1000只股票', 
            'large_cap_500': '大盘股前500只',
            'main_board': '主板股票（排除ST）'
        }
        
        # 添加预定义股票池
        for name, config in self._predefined_universes.items():
            universes[name] = config.get('description', name)
        
        # 添加文件股票池
        for file_path in self.cache_dir.glob('*.json'):
            if not file_path.name.endswith('_metadata.json'):
                universes[str(file_path)] = f'文件股票池: {file_path.name}'
        
        return universes
    
    def _convert_to_multiindex_series(self, stocks: List[str], **kwargs) -> pd.Series:
        """
        将股票列表转换为MultiIndex Series格式
        
        Parameters
        ----------
        stocks : List[str]
            股票代码列表
        **kwargs
            额外参数，可包含：
            - start_date: 开始日期
            - end_date: 结束日期
            - rebalance_frequency: 重平衡频率
            
        Returns
        -------
        pd.Series
            MultiIndex[TradingDates, StockCodes] Series，值为1
        """
        try:
            from factors.utils.data_loader import FactorDataLoader
            
            # 获取交易日期
            trading_dates = FactorDataLoader.get_trading_dates()
            
            # 处理日期范围参数
            start_date = kwargs.get('start_date')
            end_date = kwargs.get('end_date')
            
            if start_date:
                start_date = pd.to_datetime(start_date)
                trading_dates = trading_dates[trading_dates >= start_date]
            if end_date:
                end_date = pd.to_datetime(end_date)
                trading_dates = trading_dates[trading_dates <= end_date]
            
            # 处理重平衡频率
            rebalance_freq = kwargs.get('rebalance_frequency', 'daily')
            if rebalance_freq == 'monthly':
                # 月末重平衡：只在月末生效
                trading_dates = trading_dates[trading_dates.is_month_end]
            elif rebalance_freq == 'quarterly':
                # 季度重平衡：只在季末生效
                trading_dates = trading_dates[trading_dates.is_quarter_end]
            elif rebalance_freq == 'yearly':
                # 年度重平衡：只在年末生效
                trading_dates = trading_dates[trading_dates.is_year_end]
            # 'daily' 不做处理，保持所有交易日
            
            # 创建MultiIndex
            index_tuples = []
            for date in trading_dates:
                for stock in stocks:
                    index_tuples.append((date, stock))
            
            multiindex = pd.MultiIndex.from_tuples(
                index_tuples, 
                names=['TradingDates', 'StockCodes']
            )
            
            # 创建Series，值全为1
            universe_series = pd.Series(
                data=1, 
                index=multiindex, 
                name='stock_universe',
                dtype='int8'  # 节省内存
            )
            
            logger.info(f"转换为MultiIndex Series完成: "
                       f"{len(trading_dates)} 个交易日 × {len(stocks)} 只股票 = {len(universe_series)} 个数据点")
            
            return universe_series
            
        except Exception as e:
            logger.error(f"转换为MultiIndex Series失败: {e}")
            # 降级处理：创建一个简单的静态股票池
            try:
                # 使用最近一年的交易日
                dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
                dates = dates[dates.weekday < 5]  # 只保留工作日
                
                index_tuples = [(date, stock) for date in dates for stock in stocks]
                multiindex = pd.MultiIndex.from_tuples(index_tuples, names=['TradingDates', 'StockCodes'])
                
                return pd.Series(data=1, index=multiindex, name='stock_universe', dtype='int8')
            except Exception as e2:
                logger.error(f"降级处理也失败: {e2}")
                raise e
    
    def create_time_varying_universe(self, 
                                   universe_data: Dict[str, List[str]], 
                                   name: str = "time_varying") -> pd.Series:
        """
        创建时变股票池
        
        Parameters
        ----------
        universe_data : Dict[str, List[str]]
            时变股票池数据，格式：{'2024-01-01': ['000001', '000002'], '2024-02-01': ['000001', '601318']}
        name : str
            股票池名称
            
        Returns
        -------
        pd.Series
            时变股票池 MultiIndex Series
        """
        index_tuples = []
        
        for date_str, stocks in universe_data.items():
            date = pd.to_datetime(date_str)
            for stock in stocks:
                index_tuples.append((date, stock))
        
        multiindex = pd.MultiIndex.from_tuples(
            index_tuples, 
            names=['TradingDates', 'StockCodes']
        )
        
        universe_series = pd.Series(
            data=1, 
            index=multiindex, 
            name=name,
            dtype='int8'
        )
        
        logger.info(f"时变股票池 '{name}' 创建完成: {len(universe_series)} 个数据点")
        return universe_series
    
    def clear_cache(self):
        """清空股票池缓存"""
        self._universe_cache.clear()
        logger.info("股票池缓存已清空")


# 全局管理器实例
_stock_universe_manager = None

def get_stock_universe_manager() -> StockUniverseManager:
    """获取股票池管理器单例"""
    global _stock_universe_manager
    if _stock_universe_manager is None:
        _stock_universe_manager = StockUniverseManager()
    return _stock_universe_manager


def get_stock_universe(universe_name: str, **kwargs) -> List[str]:
    """便捷函数：获取股票池（向后兼容，返回List格式）"""
    manager = get_stock_universe_manager()
    return manager.get_universe(universe_name, format_type='list', **kwargs)


def get_stock_universe_series(universe_name: str, **kwargs) -> pd.Series:
    """便捷函数：获取股票池（新接口，返回MultiIndex Series格式）"""
    manager = get_stock_universe_manager()
    return manager.get_universe(universe_name, format_type='multiindex', **kwargs)