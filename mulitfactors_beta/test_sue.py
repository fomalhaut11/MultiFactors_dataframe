import pickle
import pandas as pd
from pathlib import Path

# 检查SUE文件
sue_path = Path(r'E:\Documents\PythonProject\StockProject\StockData\SingleFactorTestData\SUE.pkl')
print(f"SUE文件存在: {sue_path.exists()}")

if sue_path.exists():
    with open(sue_path, 'rb') as f:
        data = pickle.load(f)
    print(f"数据形状: {data.shape}")
    print(f"数据类型: {type(data)}")
    print(f"列名（前5个）: {list(data.columns[:5])}")
    print(f"索引（前5个）: {list(data.index[:5])}")
    print(f"数据样本:\n{data.iloc[:3, :3]}")

# 测试因子
from core import test_single_factor
result = test_single_factor('SUE')
if result and result.ic_result:
    print(f"\nSUE因子测试结果:")
    print(f"IC均值: {result.ic_result.ic_mean:.4f}")
    print(f"ICIR: {result.ic_result.icir:.4f}")
else:
    print("\n测试失败，检查错误信息")