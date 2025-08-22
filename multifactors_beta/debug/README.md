# 调试目录说明

本目录用于存放临时调试代码和验证脚本。

## 目录结构

```
debug/
├── README.md        # 本文件
├── temp/           # 临时脚本目录（自动清理）
└── results/        # 调试结果输出目录
```

## 使用说明

1. **临时脚本**：所有临时调试脚本应放在 `temp/` 目录下
2. **结果文件**：调试产生的结果文件应保存在 `results/` 目录下
3. **命名规范**：建议使用日期前缀，如 `20250801_test_bp.py`

## 注意事项

- `temp/` 目录中的文件为临时文件，可能会被定期清理
- 重要的调试结果请及时备份或移至其他目录
- 不要将敏感信息（如密码、密钥）保存在调试代码中

## 常用调试模板

### 因子验证模板
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试脚本：[功能描述]
创建日期：[日期]
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入需要的模块
# from factors.financial.fundamental_factors import BPFactor

def main():
    """主函数"""
    # 调试代码
    pass

if __name__ == "__main__":
    main()
```

## 清理策略

- 超过7天的临时文件可能被清理
- 重要结果请及时整理到项目文档中