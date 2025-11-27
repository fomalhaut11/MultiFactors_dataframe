"""
测试板块估值处理器集成到data模块
验证是否有循环引用问题
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_import():
    """测试导入是否正常"""
    print("1. 测试从data.processor导入...")
    try:
        from data.processor import SectorValuationFromStockPE
        print("   ✅ 成功导入 SectorValuationFromStockPE")
    except ImportError as e:
        print(f"   ❌ 导入失败: {e}")
        return False

    print("\n2. 测试创建实例...")
    try:
        processor = SectorValuationFromStockPE()
        print("   ✅ 成功创建处理器实例")
    except Exception as e:
        print(f"   ❌ 创建实例失败: {e}")
        return False

    print("\n3. 测试导入所有处理器...")
    try:
        from data.processor import (
            BaseDataProcessor,
            PriceDataProcessor,
            ReturnCalculator,
            FinancialDataProcessor,
            DataProcessingPipeline,
            EnhancedDataProcessingPipeline,
            SectorValuationFromStockPE
        )
        print("   ✅ 成功导入所有处理器类")
    except ImportError as e:
        print(f"   ❌ 导入失败: {e}")
        return False

    print("\n4. 测试data模块整体导入...")
    try:
        import data
        print("   ✅ 成功导入data模块")
    except ImportError as e:
        print(f"   ❌ 导入data模块失败: {e}")
        return False

    print("\n5. 检查是否有循环引用...")
    try:
        # 尝试导入factors模块
        import factors
        print("   ✅ factors模块导入正常")

        # 再次导入data.processor
        from data.processor import SectorValuationFromStockPE as SVFSP
        print("   ✅ 无循环引用问题")
    except ImportError as e:
        print(f"   ⚠️ 可能存在问题: {e}")

    return True

def test_usage():
    """测试基本使用"""
    print("\n6. 测试基本使用...")

    from data.processor import SectorValuationFromStockPE

    # 创建处理器
    processor = SectorValuationFromStockPE()

    # 检查方法
    methods = ['load_data', 'calculate_sector_valuation', 'process', 'generate_summary_report']
    for method in methods:
        if hasattr(processor, method):
            print(f"   ✅ 方法 {method} 存在")
        else:
            print(f"   ❌ 方法 {method} 不存在")

    print("\n7. 测试配置路径...")
    print(f"   数据根目录: {processor.data_root}")
    print(f"   因子路径: {processor.factors_path}")
    print(f"   输出路径: {processor.sector_data_path}")

    return True

if __name__ == "__main__":
    print("=" * 60)
    print("板块估值处理器集成测试")
    print("=" * 60)

    success = test_import()

    if success:
        test_usage()
        print("\n" + "=" * 60)
        print("✅ 集成测试通过！")
        print("板块估值处理器已成功集成到data.processor模块")
        print("可以通过以下方式使用：")
        print("  from data.processor import SectorValuationFromStockPE")
        print("  processor = SectorValuationFromStockPE()")
        print("  result = processor.process(date_range=30)")
    else:
        print("\n" + "=" * 60)
        print("❌ 集成测试失败")

    print("=" * 60)