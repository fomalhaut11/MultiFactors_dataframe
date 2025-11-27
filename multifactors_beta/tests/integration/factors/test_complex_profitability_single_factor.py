#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
复合盈利能力因子的单因子测试
使用系统的SingleFactorTestPipeline对复合因子进行全面评估
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def load_complex_factor_data():
    """加载复合因子数据"""
    try:
        logger.info("加载复合盈利能力因子数据...")
        
        # 从测试结果文件加载
        factor_file = Path('test_outputs/complex_profitability_factor_test_result.pkl')
        if factor_file.exists():
            factor_data = pd.read_pickle(factor_file)
            logger.info(f"从文件加载因子数据: {factor_data.shape}")
        else:
            # 如果文件不存在，重新计算
            logger.info("测试结果文件不存在，重新计算因子...")
            from factors.repository.mixed.complex_profitability_factor import create_complex_profitability_factor
            
            factor = create_complex_profitability_factor()
            factor_data = factor.calculate()
            
            # 保存结果
            test_outputs = Path('test_outputs')
            test_outputs.mkdir(exist_ok=True)
            factor_data.to_pickle(factor_file)
            logger.info(f"因子数据已保存到: {factor_file}")
        
        # 数据质量检查
        logger.info(f"因子数据质量检查:")
        logger.info(f"  数据形状: {factor_data.shape}")
        logger.info(f"  数据类型: {type(factor_data)}")
        logger.info(f"  索引级别: {factor_data.index.nlevels if hasattr(factor_data, 'index') else 'N/A'}")
        logger.info(f"  有效数据点: {factor_data.notna().sum():,}")
        logger.info(f"  数值范围: [{factor_data.min():.4f}, {factor_data.max():.4f}]")
        
        return factor_data
        
    except Exception as e:
        logger.error(f"加载复合因子数据失败: {e}")
        raise

def prepare_factor_for_testing(factor_data):
    """准备因子数据用于单因子测试"""
    try:
        logger.info("准备因子数据格式...")
        
        # 确保是MultiIndex Series格式
        if not isinstance(factor_data, pd.Series):
            raise ValueError("因子数据必须是pandas Series格式")
        
        if not isinstance(factor_data.index, pd.MultiIndex):
            raise ValueError("因子数据必须是MultiIndex格式 [TradingDates, StockCodes]")
        
        # 检查索引名称
        index_names = factor_data.index.names
        logger.info(f"原始索引名称: {index_names}")
        
        # 标准化索引名称（确保与测试系统兼容）
        if index_names != ['TradingDates', 'StockCodes']:
            factor_data.index.names = ['TradingDates', 'StockCodes']
            logger.info("已标准化索引名称为 [TradingDates, StockCodes]")
        
        # 排序数据
        factor_data = factor_data.sort_index()
        
        # 去除极端异常值（可选，保留数据原貌用于测试）
        logger.info("数据准备完成")
        
        return factor_data
        
    except Exception as e:
        logger.error(f"准备因子数据失败: {e}")
        raise

def save_factor_to_standard_location(factor_data):
    """将因子数据保存到标准位置供测试系统使用"""
    try:
        logger.info("将因子保存到标准存储位置...")
        
        # 检查标准因子存储目录
        from pathlib import Path
        
        # 尝试从配置获取路径，如果没有就使用默认路径
        try:
            from config import get_config
            config = get_config('main')
            data_root = config.get('paths', {}).get('data_root')
            if data_root:
                factor_dir = Path(data_root) / 'factors' / 'mixed'
            else:
                raise ValueError("配置中未找到data_root")
        except:
            # 使用默认相对路径
            factor_dir = Path('E:/Documents/PythonProject/StockProject/StockData/factors/mixed')
        
        factor_dir.mkdir(parents=True, exist_ok=True)
        factor_file = factor_dir / 'ComplexProfitability.pkl'
        
        # 保存因子数据
        factor_data.to_pickle(factor_file)
        logger.info(f"复合因子已保存到: {factor_file}")
        
        return str(factor_file)
        
    except Exception as e:
        logger.error(f"保存因子到标准位置失败: {e}")
        raise

def run_single_factor_test():
    """运行单因子测试"""
    try:
        logger.info("=" * 60)
        logger.info("开始复合盈利能力因子单因子测试")
        logger.info("=" * 60)
        
        # 1. 加载因子数据
        factor_data = load_complex_factor_data()
        
        # 2. 准备数据格式
        factor_data = prepare_factor_for_testing(factor_data)
        
        # 3. 保存因子到标准位置
        factor_file_path = save_factor_to_standard_location(factor_data)
        
        # 4. 导入测试框架
        from factors.tester import SingleFactorTestPipeline
        
        # 5. 创建测试流水线
        pipeline = SingleFactorTestPipeline()
        
        # 6. 使用因子名称进行测试
        logger.info("开始执行单因子测试...")
        
        # 测试配置
        test_config = {
            'begin_date': '2020-01-01',
            'end_date': '2024-12-31',
            'group_nums': 10,  # 分10组
            'netral_base': True,  # 使用基准中性化
            'use_industry': True,  # 使用行业分类
            'outlier_method': 'mad',  # MAD去极值
            'outlier_param': 5,
            'normalization_method': 'zscore'  # z-score标准化
        }
        
        # 执行测试
        result = pipeline.run(
            factor_name='ComplexProfitability',
            save_result=True,
            **test_config
        )
        
        # 6. 输出测试结果
        logger.info("=" * 60)
        logger.info("测试结果汇总")
        logger.info("=" * 60)
        
        logger.info(f"因子名称: {result.factor_name}")
        logger.info(f"测试ID: {result.test_id}")
        logger.info(f"测试时间范围: {result.config_snapshot.get('begin_date')} 到 {result.config_snapshot.get('end_date')}")
        
        # 数据信息
        if hasattr(result, 'data_info') and result.data_info:
            logger.info(f"样本信息:")
            for key, value in result.data_info.items():
                logger.info(f"  {key}: {value:,}" if isinstance(value, (int, float)) else f"  {key}: {value}")
        
        # 核心性能指标
        if hasattr(result, 'performance_metrics') and result.performance_metrics:
            logger.info(f"核心性能指标:")
            for key, value in result.performance_metrics.items():
                if isinstance(value, (int, float)):
                    logger.info(f"  {key}: {value:.4f}")
                else:
                    logger.info(f"  {key}: {value}")
        
        # IC分析结果
        if hasattr(result, 'ic_result') and result.ic_result:
            logger.info(f"IC分析结果:")
            logger.info(f"  IC均值: {result.ic_result.ic_mean:.4f}")
            logger.info(f"  IC标准差: {result.ic_result.ic_std:.4f}")
            logger.info(f"  ICIR: {result.ic_result.icir:.4f}")
            logger.info(f"  IC胜率: {result.ic_result.ic_win_rate:.2%}")
            logger.info(f"  Rank IC均值: {result.ic_result.rank_ic_mean:.4f}")
            logger.info(f"  Rank ICIR: {result.ic_result.rank_icir:.4f}")
        
        # 分组测试结果
        if hasattr(result, 'group_result') and result.group_result:
            logger.info(f"分组测试结果:")
            logger.info(f"  单调性得分: {result.group_result.monotonicity_score:.4f}")
            if hasattr(result.group_result, 'long_short_return') and not result.group_result.long_short_return.empty:
                ls_return = result.group_result.long_short_return
                logger.info(f"  多空组合年化收益: {ls_return.mean() * 252:.2%}")
                logger.info(f"  多空组合夏普比率: {ls_return.mean() / ls_return.std() * np.sqrt(252):.4f}")
        
        # 回归分析结果
        if hasattr(result, 'regression_result') and result.regression_result:
            logger.info(f"回归分析结果:")
            logger.info(f"  因子载荷显著性: {result.regression_result.factor_significance}")
            if hasattr(result.regression_result, 'cumulative_return') and not result.regression_result.cumulative_return.empty:
                total_return = result.regression_result.cumulative_return.iloc[-1]
                logger.info(f"  因子累计收益: {total_return:.2%}")
        
        logger.info("=" * 60)
        logger.info("单因子测试完成！")
        
        return result
        
    except Exception as e:
        logger.error(f"单因子测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_factor_report(result):
    """生成因子评估报告"""
    try:
        if result is None:
            logger.error("无测试结果，无法生成报告")
            return
        
        logger.info("=" * 60)
        logger.info("生成因子评估报告")
        logger.info("=" * 60)
        
        # 创建报告目录
        report_dir = Path('reports')
        report_dir.mkdir(exist_ok=True)
        
        # 生成文本报告
        report_file = report_dir / 'ComplexProfitability_Factor_Report.txt'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("复合盈利能力因子评估报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"因子名称: {result.factor_name}\n")
            f.write(f"测试ID: {result.test_id}\n")
            f.write(f"测试时间: {result.config_snapshot.get('begin_date')} 到 {result.config_snapshot.get('end_date')}\n\n")
            
            # 核心评价
            if result.performance_metrics:
                icir = result.performance_metrics.get('icir', 0)
                if icir > 0.03:
                    rating = "优秀"
                elif icir > 0.02:
                    rating = "良好"
                elif icir > 0.01:
                    rating = "一般"
                else:
                    rating = "较差"
                
                f.write(f"因子评级: {rating} (ICIR: {icir:.4f})\n\n")
            
            # 详细指标
            f.write("详细性能指标:\n")
            if result.performance_metrics:
                for key, value in result.performance_metrics.items():
                    if isinstance(value, (int, float)):
                        f.write(f"  {key}: {value:.4f}\n")
            
            f.write("\nIC分析:\n")
            if result.ic_result:
                f.write(f"  IC均值: {result.ic_result.ic_mean:.4f}\n")
                f.write(f"  ICIR: {result.ic_result.icir:.4f}\n")
                f.write(f"  IC胜率: {result.ic_result.ic_win_rate:.2%}\n")
            
            f.write("\n因子特征:\n")
            f.write("  因子类型: 复合因子 (财务+技术+市场)\n")
            f.write("  计算公式: {(TTM利润-TTM财务费用)-单季度存货}/短期债务 / 5日收益率截面z-score\n")
            f.write("  更新频率: 日频\n")
            f.write("  数据要求: 财务数据 + 价格数据\n")
        
        logger.info(f"因子评估报告已保存到: {report_file}")
        
        # 如果有结果管理器，也可以导出到Excel
        try:
            from factors.tester import ResultManager
            result_manager = ResultManager()
            
            excel_file = report_dir / 'ComplexProfitability_Detailed_Results.xlsx'
            result_manager.export_to_excel([result], str(excel_file))
            logger.info(f"详细测试结果已导出到: {excel_file}")
            
        except Exception as e:
            logger.warning(f"导出Excel失败: {e}")
        
    except Exception as e:
        logger.error(f"生成报告失败: {e}")

def main():
    """主函数"""
    print("复合盈利能力因子单因子测试")
    print("=" * 60)
    print("使用标准测试框架对复合因子进行全面评估")
    print("=" * 60)
    
    # 执行单因子测试
    result = run_single_factor_test()
    
    if result:
        # 生成评估报告
        generate_factor_report(result)
        
        print("\n" + "=" * 60)
        print("复合盈利能力因子测试完成！")
        print("测试结果和报告已保存到相应目录")
        print("=" * 60)
        
        return True
    else:
        print("\n" + "=" * 60)
        print("复合盈利能力因子测试失败！")
        print("请检查错误信息并修复问题")
        print("=" * 60)
        
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)