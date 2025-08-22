#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
辅助数据准备脚本
生成因子计算所需的辅助数据文件：
1. 财报发布日期数据
2. 交易日期列表
3. 股票基本信息

注意：基于正确的字段理解
- reportday: 财报公布日期
- tradingday: 财报截止日期（名称误导）
- d_year + d_quarter: 财报期间标识
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AuxiliaryDataPreparer:
    """辅助数据准备器"""
    
    def __init__(self, raw_data_path: str, output_path: str):
        """
        初始化
        
        Parameters:
        -----------
        raw_data_path : str
            原始数据路径
        output_path : str
            输出数据路径
        """
        self.raw_data_path = Path(raw_data_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
    def _get_report_period_date(self, year: int, quarter: int) -> pd.Timestamp:
        """
        根据年份和季度获取财报截止日期
        
        Parameters:
        -----------
        year : int
            年份
        quarter : int
            季度 (1-4)
            
        Returns:
        --------
        财报截止日期
        """
        quarter_end_dates = {
            1: f"{year}-03-31",
            2: f"{year}-06-30", 
            3: f"{year}-09-30",
            4: f"{year}-12-31"
        }
        return pd.Timestamp(quarter_end_dates[int(quarter)])
        
    def prepare_release_dates(self) -> pd.DataFrame:
        """
        准备财报发布日期数据
        
        从财务数据中提取报表发布日期信息
        注意：reportday 是公布日期，作为 ReleasedDates
        """
        logger.info("准备财报发布日期数据...")
        
        try:
            financial_files = ['lrb.pkl', 'xjlb.pkl', 'fzb.pkl']
            release_dates_list = []
            
            for file_name in financial_files:
                file_path = self.raw_data_path / file_name
                if not file_path.exists():
                    logger.warning(f"财务数据文件不存在: {file_path}")
                    continue
                    
                logger.info(f"处理 {file_name}...")
                df = pd.read_pickle(file_path)
                
                # 检查必要的列
                required_cols = ['reportday', 'code', 'd_year', 'd_quarter']
                if not all(col in df.columns for col in required_cols):
                    logger.warning(f"{file_name} 缺少必要的列")
                    continue
                
                # 创建财报期间标识
                df['ReportPeriod'] = df.apply(
                    lambda row: self._get_report_period_date(row['d_year'], row['d_quarter']), 
                    axis=1
                )
                
                # 提取发布日期信息
                release_info = df[['ReportPeriod', 'code', 'reportday']].copy()
                release_info['ReleasedDates'] = pd.to_datetime(release_info['reportday'])
                release_info = release_info[['ReportPeriod', 'code', 'ReleasedDates']]
                
                # 去重 - 同一财报期间同一股票只保留一条记录
                release_info = release_info.drop_duplicates(['ReportPeriod', 'code'])
                release_dates_list.append(release_info)
                
            # 合并所有数据
            if release_dates_list:
                release_dates = pd.concat(release_dates_list, ignore_index=True)
                release_dates = release_dates.drop_duplicates(['ReportPeriod', 'code'])
                
                # 设置索引为财报期间
                release_dates = release_dates.set_index(['ReportPeriod', 'code'])
                release_dates.index.names = ['ReportDates', 'StockCodes']
                
                # 保存
                output_file = self.output_path / 'ReleaseDates.pkl'
                release_dates.to_pickle(output_file)
                logger.info(f"财报发布日期数据已保存: {output_file}")
                logger.info(f"数据形状: {release_dates.shape}")
                
                return release_dates
            else:
                logger.error("没有找到有效的财务数据")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"准备财报发布日期数据失败: {e}")
            return pd.DataFrame()
            
    def prepare_trading_dates(self) -> pd.Series:
        """
        准备交易日期列表
        
        从价格数据中提取交易日期
        """
        logger.info("准备交易日期数据...")
        
        try:
            # 加载价格数据
            price_file = self.raw_data_path / 'Price.pkl'
            if not price_file.exists():
                logger.error(f"价格数据文件不存在: {price_file}")
                return pd.Series()
                
            price_data = pd.read_pickle(price_file)
            
            # 提取交易日期
            if isinstance(price_data.index, pd.MultiIndex):
                trading_dates = price_data.index.get_level_values('TradingDates').unique().sort_values()
            else:
                # 假设第一列是日期
                if 'tradingday' in price_data.columns:
                    trading_dates = pd.to_datetime(price_data['tradingday']).unique()
                    trading_dates = pd.Series(trading_dates).sort_values()
                else:
                    logger.error("无法从价格数据中提取交易日期")
                    return pd.Series()
                    
            # 保存
            output_file = self.output_path / 'TradingDates.pkl'
            pd.Series(trading_dates).to_pickle(output_file)
            logger.info(f"交易日期数据已保存: {output_file}")
            logger.info(f"日期范围: {trading_dates[0]} 到 {trading_dates[-1]}")
            logger.info(f"交易日数: {len(trading_dates)}")
            
            return pd.Series(trading_dates)
            
        except Exception as e:
            logger.error(f"准备交易日期数据失败: {e}")
            return pd.Series()
            
    def prepare_stock_info(self) -> pd.DataFrame:
        """
        准备股票基本信息
        
        包括股票代码、名称、上市日期、退市日期等
        """
        logger.info("准备股票基本信息...")
        
        try:
            # 加载股票基本信息
            stock_info_file = self.raw_data_path / 'StockInfo.pkl'
            if stock_info_file.exists():
                stock_info = pd.read_pickle(stock_info_file)
                
                # 处理日期
                date_cols = ['list_date', 'delist_date']
                for col in date_cols:
                    if col in stock_info.columns:
                        stock_info[col] = pd.to_datetime(stock_info[col])
                        
                # 设置索引
                if 'code' in stock_info.columns:
                    stock_info = stock_info.set_index('code')
                    
                # 保存
                output_file = self.output_path / 'StockInfo.pkl'
                stock_info.to_pickle(output_file)
                logger.info(f"股票基本信息已保存: {output_file}")
                logger.info(f"股票数量: {len(stock_info)}")
                
                return stock_info
            else:
                logger.warning("股票基本信息文件不存在，将从其他数据中提取")
                
                # 从价格数据中提取股票列表
                price_file = self.raw_data_path / 'Price.pkl'
                if price_file.exists():
                    price_data = pd.read_pickle(price_file)
                    
                    if isinstance(price_data.index, pd.MultiIndex):
                        stocks = price_data.index.get_level_values('StockCodes').unique()
                    else:
                        stocks = price_data['code'].unique() if 'code' in price_data.columns else []
                        
                    # 创建基本信息DataFrame
                    stock_info = pd.DataFrame({
                        'code': stocks,
                        'name': [f'Stock_{code}' for code in stocks]  # 占位名称
                    }).set_index('code')
                    
                    # 保存
                    output_file = self.output_path / 'StockInfo.pkl'
                    stock_info.to_pickle(output_file)
                    logger.info(f"股票基本信息已保存: {output_file}")
                    logger.info(f"股票数量: {len(stock_info)}")
                    
                    return stock_info
                    
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"准备股票基本信息失败: {e}")
            return pd.DataFrame()
            
    def prepare_financial_data_unified(self) -> pd.DataFrame:
        """
        准备统一格式的财务数据
        
        将三张财务报表合并为统一的MultiIndex格式
        注意：
        - 使用财报期间（根据d_year和d_quarter计算）作为主索引
        - reportday作为发布日期保存在数据列中
        """
        logger.info("准备统一格式的财务数据...")
        
        try:
            financial_files = {
                'lrb.pkl': '利润表',
                'xjlb.pkl': '现金流量表', 
                'fzb.pkl': '资产负债表'
            }
            
            financial_data_list = []
            
            for file_name, table_name in financial_files.items():
                file_path = self.raw_data_path / file_name
                if not file_path.exists():
                    logger.warning(f"{table_name}文件不存在: {file_path}")
                    continue
                    
                logger.info(f"处理 {table_name}...")
                df = pd.read_pickle(file_path)
                
                # 检查必要列
                if 'd_year' not in df.columns or 'd_quarter' not in df.columns:
                    logger.error(f"{table_name} 缺少 d_year 或 d_quarter 列")
                    continue
                
                # 创建财报期间
                df['ReportPeriod'] = df.apply(
                    lambda row: self._get_report_period_date(row['d_year'], row['d_quarter']), 
                    axis=1
                )
                
                # 保存原始的公布日期
                df['ReleasedDates'] = pd.to_datetime(df['reportday'])
                
                # 保存 tradingday（虽然名字误导，但这是原始数据中的财报截止日期）
                if 'tradingday' in df.columns:
                    df['OriginalTradingDay'] = pd.to_datetime(df['tradingday'])
                
                # 设置索引为财报期间
                if 'code' in df.columns:
                    df = df.set_index(['ReportPeriod', 'code'])
                    df.index.names = ['ReportDates', 'StockCodes']
                    
                    financial_data_list.append(df)
                    
            # 合并财务数据
            if financial_data_list:
                # 使用外连接合并，保留所有数据
                financial_data = financial_data_list[0]
                for df in financial_data_list[1:]:
                    # 识别重复列
                    common_cols = set(financial_data.columns) & set(df.columns)
                    # 只合并新列
                    new_cols = [col for col in df.columns if col not in common_cols]
                    if new_cols:
                        financial_data = financial_data.join(df[new_cols], how='outer')
                
                # 确保关键列存在
                if 'ReleasedDates' not in financial_data.columns:
                    logger.error("合并后的数据缺少 ReleasedDates 列")
                
                # 保存
                output_file = self.output_path / 'FinancialData_unified.pkl'
                financial_data.to_pickle(output_file)
                logger.info(f"统一财务数据已保存: {output_file}")
                logger.info(f"数据形状: {financial_data.shape}")
                logger.info(f"列数: {len(financial_data.columns)}")
                
                # 检查数据样本
                logger.info("\n数据样本检查:")
                sample = financial_data.head(3)
                for idx in sample.index:
                    logger.info(f"  财报期间: {idx[0]}, 股票: {idx[1]}")
                    if 'ReleasedDates' in sample.columns:
                        logger.info(f"    发布日期: {sample.loc[idx, 'ReleasedDates']}")
                    if 'd_year' in sample.columns and 'd_quarter' in sample.columns:
                        logger.info(f"    年份: {sample.loc[idx, 'd_year']}, 季度: {sample.loc[idx, 'd_quarter']}")
                
                return financial_data
            else:
                logger.error("没有找到有效的财务数据")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"准备统一财务数据失败: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
            
    def verify_data_consistency(self):
        """验证数据一致性"""
        logger.info("\n验证数据一致性...")
        
        # 加载准备好的数据
        financial_data_file = self.output_path / 'FinancialData_unified.pkl'
        release_dates_file = self.output_path / 'ReleaseDates.pkl'
        
        if financial_data_file.exists() and release_dates_file.exists():
            financial_data = pd.read_pickle(financial_data_file)
            release_dates = pd.read_pickle(release_dates_file)
            
            # 检查索引是否一致
            logger.info("检查索引一致性...")
            
            # 获取一个样本股票
            sample_stock = financial_data.index.get_level_values('StockCodes')[0]
            stock_financial = financial_data.xs(sample_stock, level='StockCodes')
            
            logger.info(f"\n样本股票 {sample_stock} 的财报期间:")
            for report_date in stock_financial.index[:5]:
                row = stock_financial.loc[report_date]
                logger.info(f"  {report_date}:")
                if 'd_year' in row:
                    logger.info(f"    年份: {row['d_year']}, 季度: {row['d_quarter']}")
                if 'ReleasedDates' in row:
                    logger.info(f"    发布日期: {row['ReleasedDates']}")
                    
            # 检查同一天发布多份财报的情况
            logger.info("\n检查同天发布多份财报的情况...")
            if 'ReleasedDates' in financial_data.columns:
                # 按股票和发布日期分组
                grouped = financial_data.reset_index().groupby(['StockCodes', 'ReleasedDates'])
                multi_reports = grouped.size()[grouped.size() > 1]
                
                if len(multi_reports) > 0:
                    logger.info(f"发现 {len(multi_reports)} 个同天发布多份财报的情况")
                    # 显示前5个例子
                    for (stock, release_date), count in multi_reports.head().items():
                        logger.info(f"  股票 {stock} 在 {release_date} 发布了 {count} 份财报")
                        # 显示具体是哪些财报
                        mask = (financial_data.index.get_level_values('StockCodes') == stock) & \
                               (financial_data['ReleasedDates'] == release_date)
                        reports = financial_data[mask].index.get_level_values('ReportDates')
                        for report in reports:
                            logger.info(f"    - {report}")
                else:
                    logger.info("未发现同天发布多份财报的情况")
                    
    def prepare_all(self):
        """准备所有辅助数据"""
        logger.info("开始准备所有辅助数据...")
        logger.info(f"原始数据路径: {self.raw_data_path}")
        logger.info(f"输出路径: {self.output_path}")
        
        # 1. 准备财报发布日期
        release_dates = self.prepare_release_dates()
        
        # 2. 准备交易日期
        trading_dates = self.prepare_trading_dates()
        
        # 3. 准备股票基本信息
        stock_info = self.prepare_stock_info()
        
        # 4. 准备统一格式的财务数据
        financial_data = self.prepare_financial_data_unified()
        
        # 5. 验证数据一致性
        self.verify_data_consistency()
        
        # 生成数据摘要
        summary = {
            'prepared_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'release_dates_shape': release_dates.shape if not release_dates.empty else (0, 0),
            'trading_dates_count': len(trading_dates) if not trading_dates.empty else 0,
            'stock_info_count': len(stock_info) if not stock_info.empty else 0,
            'financial_data_shape': financial_data.shape if not financial_data.empty else (0, 0),
            'note': '使用财报期间作为索引，reportday作为发布日期'
        }
        
        # 保存摘要
        import json
        summary_file = self.output_path / 'data_preparation_summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            
        logger.info("\n辅助数据准备完成!")
        logger.info(f"数据摘要已保存: {summary_file}")
        
        return {
            'release_dates': release_dates,
            'trading_dates': trading_dates,
            'stock_info': stock_info,
            'financial_data': financial_data,
            'summary': summary
        }


def main():
    """主函数"""
    # 配置路径
    raw_data_path = r"E:\Documents\PythonProject\StockProject\StockData"
    output_path = r"E:\Documents\PythonProject\StockProject\MultiFactors\mulitfactors_beta\data\auxiliary"
    
    # 创建准备器
    preparer = AuxiliaryDataPreparer(raw_data_path, output_path)
    
    # 准备所有数据
    results = preparer.prepare_all()
    
    # 打印结果
    print("\n" + "="*60)
    print("数据准备结果汇总")
    print("="*60)
    
    if 'summary' in results:
        summary = results['summary']
        print(f"准备时间: {summary['prepared_date']}")
        print(f"财报发布日期数据: {summary['release_dates_shape']}")
        print(f"交易日期数量: {summary['trading_dates_count']}")
        print(f"股票数量: {summary['stock_info_count']}")
        print(f"统一财务数据: {summary['financial_data_shape']}")
        print(f"说明: {summary['note']}")
    
    print("\n[OK] 所有辅助数据准备完成!")
    print(f"数据保存路径: {output_path}")


if __name__ == "__main__":
    main()