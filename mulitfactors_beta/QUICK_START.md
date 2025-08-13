# 多因子量化系统 - 快速开始

## 🚀 3分钟快速上手

### 1. 检查项目状态
```bash
# 查看最新项目进度
cat PROJECT_PROGRESS_20250804.md

# 运行测试验证系统
pytest tests/ -v
```

### 2. 数据准备
```bash
# 生成预处理数据
python data/prepare_auxiliary_data.py

# 更新价格数据
python scheduled_data_updater.py --data-type price
```

### 3. 计算因子
```python
from factors.financial.fundamental_factors import EPFactor, BPFactor

# 加载数据（假设已准备好）
# financial_data, market_cap, release_dates, trading_dates = load_data()

# 计算EP因子
ep_factor = EPFactor(method='ttm')
ep_values = ep_factor.calculate(financial_data, market_cap, release_dates, trading_dates)

# 计算BP因子
bp_factor = BPFactor()
bp_values = bp_factor.calculate(financial_data, market_cap, release_dates, trading_dates)

print(f"EP因子计算完成，数据点数: {len(ep_values)}")
print(f"BP因子计算完成，数据点数: {len(bp_values)}")
```

## 📚 重要文档

- 📈 **[项目进度](./PROJECT_PROGRESS_20250804.md)** - 当前状态和完成情况
- 🏗️ **[项目结构](./PROJECT_STRUCTURE.md)** - 详细的系统架构说明
- ⚡ **[重构总结](./REFACTORING_SUMMARY.md)** - 性能优化和技术改进
- 🧹 **[清理总结](./PROJECT_CLEANUP_SUMMARY.md)** - 文件整理和测试框架
- 📖 **[主要说明](./README.md)** - 完整的使用文档

## 🧪 测试验证

```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/unit/ -v              # 单元测试
pytest tests/integration/ -v       # 集成测试
pytest tests/performance/ -v       # 性能测试

# 生成覆盖率报告
pytest --cov=factors tests/
```

## ⚙️ 系统配置

### 必需依赖
- Python 3.8+
- pandas >= 1.3.0
- numpy >= 1.20.0
- pytest (用于测试)

### 配置文件
- `config.yaml` - 主配置文件
- `factors/config/factor_config.py` - 因子配置

## 🎯 当前可用功能

✅ **基本面因子**: EP, BP, ROE, PEG  
✅ **数据处理**: 自动化获取和预处理  
✅ **性能优化**: 大数据集高效处理  
✅ **测试框架**: 完整的单元和集成测试  
✅ **灵活配置**: 多数据源适配  

## 📞 需要帮助？

1. **技术问题**: 查看 `docs/` 目录下的详细文档
2. **使用示例**: 运行 `examples/` 目录下的示例代码
3. **测试问题**: 参考 `tests/README.md`
4. **性能优化**: 查看 `REFACTORING_SUMMARY.md`

---

**系统状态**: 🟢 **生产就绪**  
**建议**: 可以开始正式使用！