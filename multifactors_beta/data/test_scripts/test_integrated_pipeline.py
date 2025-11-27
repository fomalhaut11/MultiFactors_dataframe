"""
测试集成的数据处理管道
验证板块估值计算是否正确集成
"""

import sys
from pathlib import Path
import logging

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_integrated_pipeline():
    """测试集成管道"""
    logger.info("="*60)
    logger.info("测试集成数据处理管道")
    logger.info("="*60)

    from data.processor import IntegratedDataPipeline

    # 创建管道实例
    pipeline = IntegratedDataPipeline()

    # 测试板块估值独立更新
    logger.info("\n1. 测试板块估值独立更新...")
    try:
        sector_valuation = pipeline.update_sector_valuation(
            date_range=1,  # 只计算最新1天
            force_update=True
        )

        if not sector_valuation.empty:
            logger.info(f"✓ 板块估值更新成功，共{len(sector_valuation)}条记录")

            # 显示最新统计
            latest_date = sector_valuation['TradingDate'].max()
            latest_data = sector_valuation[sector_valuation['TradingDate'] == latest_date]

            if 'PE_TTM' in latest_data.columns:
                pe_median = latest_data['PE_TTM'].median()
                logger.info(f"  最新日期: {latest_date}")
                logger.info(f"  PE中位数: {pe_median:.2f}")
        else:
            logger.error("✗ 板块估值更新失败")
    except Exception as e:
        logger.error(f"✗ 板块估值更新出错: {e}")
        return False

    return True


def test_data_update_scheduler():
    """测试数据更新调度器"""
    logger.info("\n" + "="*60)
    logger.info("测试数据更新调度器")
    logger.info("="*60)

    from data.processor import DataUpdateScheduler, IntegratedDataPipeline

    # 创建调度器
    pipeline = IntegratedDataPipeline()
    scheduler = DataUpdateScheduler(pipeline)

    # 测试自定义更新
    logger.info("\n2. 测试自定义更新（只更新板块估值）...")
    try:
        scheduler.run_custom_update(
            update_price=False,
            update_financial=False,
            update_sector_valuation=True,
            sector_date_range=1
        )
        logger.info("✓ 自定义更新成功")
    except Exception as e:
        logger.error(f"✗ 自定义更新失败: {e}")
        return False

    return True


def test_configuration():
    """测试配置功能"""
    logger.info("\n" + "="*60)
    logger.info("测试配置功能")
    logger.info("="*60)

    from data.processor import IntegratedDataPipeline

    pipeline = IntegratedDataPipeline()

    # 测试配置更新
    logger.info("\n3. 测试配置更新...")
    new_config = {
        'enabled': True,
        'date_range': 5,
        'save_intermediate': True,
        'output_formats': ['pkl', 'csv']
    }

    pipeline.configure_sector_valuation(new_config)
    logger.info("✓ 配置更新成功")

    # 验证配置
    if pipeline.sector_valuation_config['date_range'] == 5:
        logger.info("✓ 配置验证通过")
    else:
        logger.error("✗ 配置验证失败")
        return False

    return True


def test_command_line():
    """测试命令行接口"""
    logger.info("\n" + "="*60)
    logger.info("测试命令行接口")
    logger.info("="*60)

    import subprocess

    # 测试帮助信息
    logger.info("\n4. 测试命令行帮助...")
    try:
        result = subprocess.run(
            ["python", "../update_data.py", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )

        if result.returncode == 0:
            logger.info("✓ 命令行接口正常")
            # 显示部分帮助信息
            help_lines = result.stdout.split('\n')[:5]
            for line in help_lines:
                logger.info(f"  {line}")
        else:
            logger.error("✗ 命令行接口异常")
            return False
    except Exception as e:
        logger.error(f"✗ 命令行测试失败: {e}")
        return False

    return True


def verify_output_files():
    """验证输出文件"""
    logger.info("\n" + "="*60)
    logger.info("验证输出文件")
    logger.info("="*60)

    from pathlib import Path

    # 检查输出文件
    data_root = Path("E:/Documents/PythonProject/StockProject/StockData")
    sector_data_path = data_root / "SectorData"

    expected_files = [
        "sector_valuation_from_stock_pe.pkl",
        "sector_valuation_from_stock_pe.csv",
        "sector_valuation_summary.json"
    ]

    logger.info("\n5. 检查输出文件...")
    all_exist = True
    for filename in expected_files:
        filepath = sector_data_path / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            logger.info(f"  ✓ {filename} ({size_mb:.2f} MB)")
        else:
            logger.error(f"  ✗ {filename} 不存在")
            all_exist = False

    return all_exist


def main():
    """主测试函数"""
    print("\n" + "="*60)
    print("集成数据处理管道 - 完整测试套件")
    print("="*60)

    # 运行所有测试
    tests = [
        ("集成管道测试", test_integrated_pipeline),
        ("调度器测试", test_data_update_scheduler),
        ("配置功能测试", test_configuration),
        ("命令行接口测试", test_command_line),
        ("输出文件验证", verify_output_files)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            logger.info(f"\n执行: {test_name}")
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"{test_name} 异常: {e}")
            results.append((test_name, False))

    # 显示测试结果汇总
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)

    passed = 0
    failed = 0

    for test_name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{test_name:20} {status}")
        if success:
            passed += 1
        else:
            failed += 1

    print("-"*60)
    print(f"总计: {len(results)} 个测试")
    print(f"通过: {passed} 个")
    print(f"失败: {failed} 个")

    if failed == 0:
        print("\n🎉 所有测试通过！板块估值计算已成功集成到数据更新流程。")
    else:
        print(f"\n⚠️ 有 {failed} 个测试失败，请检查日志。")

    print("="*60)


if __name__ == "__main__":
    main()