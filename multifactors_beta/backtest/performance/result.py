"""
回测结果类

存储和管理回测的所有结果数据
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)

class BacktestResult:
    """
    回测结果存储和管理类
    
    存储回测过程中的所有数据，包括日度数据、绩效指标、交易记录等
    """
    
    def __init__(self):
        """初始化回测结果"""
        # 基础信息
        self.start_date: Optional[datetime] = None
        self.end_date: Optional[datetime] = None
        self.initial_capital: float = 0.0
        self.final_capital: float = 0.0
        
        # 日度数据
        self.daily_portfolio_value = pd.Series(dtype=float, name='portfolio_value')
        self.daily_returns = pd.Series(dtype=float, name='returns')
        self.daily_positions = pd.DataFrame()  # 每日持仓
        self.daily_weights = pd.DataFrame()    # 每日权重
        self.daily_trades = pd.DataFrame()     # 每日交易
        self.daily_costs = pd.Series(dtype=float, name='costs')
        
        # 绩效指标
        self.performance_metrics: Dict[str, float] = {}
        
        # 交易统计
        self.trade_statistics: Dict[str, Any] = {}
        
        # 风险指标
        self.risk_metrics: Dict[str, float] = {}
        
        # 基准比较（如果有基准）
        self.benchmark_data: Optional[pd.Series] = None
        self.relative_metrics: Dict[str, float] = {}
        
        # 其他信息
        self.metadata: Dict[str, Any] = {}
        
        logger.debug("BacktestResult 初始化完成")
    
    def record_daily_data(self, 
                         date: datetime,
                         portfolio_value: float,
                         positions: pd.Series,
                         trades: Optional[pd.DataFrame] = None,
                         costs: float = 0.0) -> None:
        """
        记录每日数据
        
        Parameters
        ----------
        date : datetime
            交易日期
        portfolio_value : float
            组合总价值
        positions : pd.Series
            当日持仓，index为股票代码，values为持仓数量
        trades : pd.DataFrame, optional
            当日交易记录
        costs : float
            当日交易成本
        """
        # 记录组合价值
        self.daily_portfolio_value.loc[date] = portfolio_value
        
        # 计算日收益率
        if len(self.daily_portfolio_value) > 1:
            prev_value = self.daily_portfolio_value.iloc[-2]
            daily_return = (portfolio_value - prev_value) / prev_value
            self.daily_returns.loc[date] = daily_return
        else:
            self.daily_returns.loc[date] = 0.0
        
        # 记录持仓
        if not positions.empty:
            self.daily_positions = pd.concat([
                self.daily_positions, 
                positions.to_frame(date).T
            ])
            
            # 计算权重
            total_value = positions.sum() if positions.sum() != 0 else portfolio_value
            weights = positions / total_value
            self.daily_weights = pd.concat([
                self.daily_weights,
                weights.to_frame(date).T
            ])
        
        # 记录交易
        if trades is not None and not trades.empty:
            trades['date'] = date
            self.daily_trades = pd.concat([self.daily_trades, trades])
        
        # 记录成本
        self.daily_costs.loc[date] = costs
        
        # 更新基础信息
        if self.start_date is None:
            self.start_date = date
        self.end_date = date
        self.final_capital = portfolio_value
    
    def calculate_performance_metrics(self) -> Dict[str, float]:
        """
        计算绩效指标
        
        Returns
        -------
        Dict[str, float]
            绩效指标字典
        """
        if len(self.daily_returns) < 2:
            logger.warning("数据不足，无法计算绩效指标")
            return {}
        
        returns = self.daily_returns.dropna()
        
        # 基础收益指标
        total_return = (self.final_capital / self.initial_capital - 1) if self.initial_capital > 0 else 0
        
        # 年化处理
        trading_days = len(returns)
        years = trading_days / 252.0  # 假设一年252个交易日
        
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        annual_volatility = returns.std() * np.sqrt(252)
        
        # 夏普比率
        risk_free_rate = self.metadata.get('risk_free_rate', 0.025)  # 默认2.5%
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
        
        # 最大回撤
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdowns.min()
        
        # Calmar比率
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # 胜率
        win_rate = (returns > 0).mean()
        
        # 盈亏比
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        profit_loss_ratio = (positive_returns.mean() / abs(negative_returns.mean()) 
                           if len(negative_returns) > 0 and negative_returns.mean() != 0 else np.inf)
        
        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio,
            'trading_days': trading_days,
            'years': years
        }
        
        self.performance_metrics.update(metrics)
        logger.info("绩效指标计算完成")
        
        return metrics
    
    def calculate_trade_statistics(self) -> Dict[str, Any]:
        """
        计算交易统计
        
        Returns
        -------
        Dict[str, Any]
            交易统计字典
        """
        if self.daily_trades.empty:
            logger.warning("无交易记录")
            return {}
        
        # 总交易次数
        total_trades = len(self.daily_trades)
        
        # 总交易金额
        total_trade_amount = self.daily_trades.get('amount', pd.Series()).sum()
        
        # 总交易成本
        total_costs = self.daily_costs.sum()
        
        # 平均每日交易次数
        trading_days_with_trades = self.daily_trades['date'].nunique()
        avg_daily_trades = total_trades / trading_days_with_trades if trading_days_with_trades > 0 else 0
        
        # 换手率
        avg_portfolio_value = self.daily_portfolio_value.mean()
        turnover_rate = (total_trade_amount / 2) / avg_portfolio_value if avg_portfolio_value > 0 else 0  # 除以2因为买卖各算一次
        annual_turnover = turnover_rate * 252 / len(self.daily_portfolio_value) if len(self.daily_portfolio_value) > 0 else 0
        
        # 成本占比
        cost_ratio = total_costs / self.final_capital if self.final_capital > 0 else 0
        
        statistics = {
            'total_trades': total_trades,
            'total_trade_amount': total_trade_amount,
            'total_costs': total_costs,
            'avg_daily_trades': avg_daily_trades,
            'turnover_rate': turnover_rate,
            'annual_turnover': annual_turnover,
            'cost_ratio': cost_ratio,
            'trading_days_with_trades': trading_days_with_trades
        }
        
        self.trade_statistics.update(statistics)
        logger.info("交易统计计算完成")
        
        return statistics
    
    def set_benchmark(self, benchmark_returns: pd.Series) -> None:
        """
        设置基准数据并计算相对指标
        
        Parameters
        ----------
        benchmark_returns : pd.Series
            基准收益率序列，index为日期
        """
        # 对齐日期
        aligned_dates = self.daily_returns.index.intersection(benchmark_returns.index)
        if len(aligned_dates) == 0:
            logger.warning("无法对齐基准数据的日期")
            return
        
        portfolio_returns = self.daily_returns.loc[aligned_dates]
        benchmark_returns = benchmark_returns.loc[aligned_dates]
        
        self.benchmark_data = benchmark_returns
        
        # 计算相对指标
        excess_returns = portfolio_returns - benchmark_returns
        
        # 信息比率
        tracking_error = excess_returns.std() * np.sqrt(252)
        excess_annual_return = excess_returns.mean() * 252
        information_ratio = excess_annual_return / tracking_error if tracking_error > 0 else 0
        
        # Beta
        if len(portfolio_returns) > 1 and portfolio_returns.var() > 0:
            beta = portfolio_returns.cov(benchmark_returns) / benchmark_returns.var()
        else:
            beta = 0
        
        # Alpha (CAPM)
        portfolio_annual_return = self.performance_metrics.get('annual_return', 0)
        benchmark_annual_return = benchmark_returns.mean() * 252
        risk_free_rate = self.metadata.get('risk_free_rate', 0.025)
        alpha = portfolio_annual_return - (risk_free_rate + beta * (benchmark_annual_return - risk_free_rate))
        
        relative_metrics = {
            'information_ratio': information_ratio,
            'tracking_error': tracking_error,
            'excess_annual_return': excess_annual_return,
            'beta': beta,
            'alpha': alpha,
            'benchmark_annual_return': benchmark_annual_return
        }
        
        self.relative_metrics.update(relative_metrics)
        logger.info("基准比较指标计算完成")
    
    def generate_summary(self) -> Dict[str, Any]:
        """
        生成回测结果摘要
        
        Returns
        -------
        Dict[str, Any]
            回测摘要
        """
        # 确保指标已计算
        if not self.performance_metrics:
            self.calculate_performance_metrics()
        if not self.trade_statistics:
            self.calculate_trade_statistics()
        
        summary = {
            'basic_info': {
                'start_date': self.start_date.strftime('%Y-%m-%d') if self.start_date else None,
                'end_date': self.end_date.strftime('%Y-%m-%d') if self.end_date else None,
                'initial_capital': self.initial_capital,
                'final_capital': self.final_capital,
                'trading_days': len(self.daily_returns)
            },
            'performance': self.performance_metrics,
            'trading': self.trade_statistics,
            'risk': self.risk_metrics
        }
        
        if self.relative_metrics:
            summary['relative_performance'] = self.relative_metrics
        
        return summary
    
    def print_summary(self) -> None:
        """打印回测结果摘要"""
        summary = self.generate_summary()
        
        print("=" * 80)
        print("回测结果摘要")
        print("=" * 80)
        
        # 基础信息
        basic = summary['basic_info']
        print(f"\n📅 回测期间: {basic['start_date']} 到 {basic['end_date']}")
        print(f"💰 初始资金: {basic['initial_capital']:,.2f}")
        print(f"💰 最终资金: {basic['final_capital']:,.2f}")
        print(f"📊 交易天数: {basic['trading_days']}")
        
        # 绩效指标
        perf = summary['performance']
        if perf:
            print(f"\n📈 绩效指标:")
            print(f"  总收益率: {perf.get('total_return', 0):.2%}")
            print(f"  年化收益率: {perf.get('annual_return', 0):.2%}")
            print(f"  年化波动率: {perf.get('annual_volatility', 0):.2%}")
            print(f"  夏普比率: {perf.get('sharpe_ratio', 0):.3f}")
            print(f"  最大回撤: {perf.get('max_drawdown', 0):.2%}")
            print(f"  Calmar比率: {perf.get('calmar_ratio', 0):.3f}")
            print(f"  胜率: {perf.get('win_rate', 0):.1%}")
        
        # 交易统计
        trading = summary['trading']
        if trading:
            print(f"\n🔄 交易统计:")
            print(f"  总交易次数: {trading.get('total_trades', 0):,}")
            print(f"  年化换手率: {trading.get('annual_turnover', 0):.2f}x")
            print(f"  总交易成本: {trading.get('total_costs', 0):.2f}")
            print(f"  成本占比: {trading.get('cost_ratio', 0):.2%}")
        
        # 相对基准（如果有）
        if 'relative_performance' in summary:
            rel = summary['relative_performance']
            print(f"\n📊 相对基准:")
            print(f"  信息比率: {rel.get('information_ratio', 0):.3f}")
            print(f"  跟踪误差: {rel.get('tracking_error', 0):.2%}")
            print(f"  Alpha: {rel.get('alpha', 0):.2%}")
            print(f"  Beta: {rel.get('beta', 0):.3f}")
        
        print("=" * 80)
    
    def save_to_file(self, filepath: str, format: str = 'pickle') -> None:
        """
        保存回测结果到文件
        
        Parameters
        ---------- 
        filepath : str
            保存路径
        format : str
            保存格式 ('pickle', 'json', 'excel')
        """
        if format == 'pickle':
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
                
        elif format == 'json':
            summary = self.generate_summary()
            # 转换datetime等不能序列化的对象
            def convert_for_json(obj):
                if isinstance(obj, (pd.Timestamp, datetime)):
                    return obj.isoformat()
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif pd.isna(obj):
                    return None
                return obj
            
            def recursive_convert(d):
                if isinstance(d, dict):
                    return {k: recursive_convert(v) for k, v in d.items()}
                elif isinstance(d, list):
                    return [recursive_convert(v) for v in d]
                else:
                    return convert_for_json(d)
            
            json_data = recursive_convert(summary)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
                
        elif format == 'excel':
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # 保存各种数据表
                if not self.daily_returns.empty:
                    self.daily_returns.to_excel(writer, sheet_name='每日收益率')
                if not self.daily_portfolio_value.empty:
                    self.daily_portfolio_value.to_excel(writer, sheet_name='组合价值')
                if not self.daily_positions.empty:
                    self.daily_positions.to_excel(writer, sheet_name='每日持仓')
                if not self.daily_trades.empty:
                    self.daily_trades.to_excel(writer, sheet_name='交易记录')
                
                # 保存摘要
                summary_df = pd.DataFrame.from_dict(
                    self.generate_summary(), orient='index'
                )
                summary_df.to_excel(writer, sheet_name='回测摘要')
        
        else:
            raise ValueError(f"不支持的保存格式: {format}")
        
        logger.info(f"回测结果已保存到: {filepath}")
    
    @classmethod
    def load_from_file(cls, filepath: str, format: str = 'pickle') -> 'BacktestResult':
        """
        从文件加载回测结果
        
        Parameters
        ----------
        filepath : str
            文件路径
        format : str
            文件格式
            
        Returns
        -------
        BacktestResult
            加载的回测结果
        """
        if format == 'pickle':
            import pickle
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"不支持从{format}格式加载")