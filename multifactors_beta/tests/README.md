# 测试目录结构说明

## 目录组织

```
tests/
├── README.md              # 本文件，测试说明
├── __init__.py            # 测试包初始化
├── conftest.py            # pytest配置和夹具
├── unit/                  # 单元测试
│   ├── test_factors/      # 因子计算单元测试
│   ├── test_data_processing/  # 数据处理单元测试
│   └── test_utils/        # 工具函数单元测试
├── integration/           # 集成测试
│   ├── test_factor_pipeline/  # 因子计算流程测试
│   ├── test_data_validation/  # 数据验证集成测试
│   └── test_end_to_end/   # 端到端测试
├── performance/           # 性能测试
│   ├── benchmark_factors.py   # 因子计算性能基准
│   ├── benchmark_data_processing.py  # 数据处理性能基准
│   └── memory_usage_tests.py  # 内存使用测试
└── data/                  # 测试数据
    ├── mock_data/         # 模拟数据
    ├── fixtures/          # 测试夹具数据
    └── samples/           # 样本数据
```

## 测试分类

### 1. 单元测试 (unit/)
- 测试单个函数或类的功能
- 使用mock数据，快速执行
- 覆盖边界条件和异常情况

### 2. 集成测试 (integration/)
- 测试多个模块的协作
- 使用真实或接近真实的数据
- 验证数据流和业务逻辑

### 3. 性能测试 (performance/)
- 测试系统性能指标
- 内存使用、执行时间等
- 不同数据规模的表现

## 运行测试

### 运行所有测试
```bash
pytest tests/
```

### 运行特定类型测试
```bash
# 单元测试
pytest tests/unit/

# 集成测试  
pytest tests/integration/

# 性能测试
pytest tests/performance/
```

### 运行特定测试文件
```bash
pytest tests/unit/test_factors/test_fundamental_factors.py
```

### 生成覆盖率报告
```bash
pytest --cov=factors tests/
```

## 测试数据管理

### Mock数据
- 使用 `factors.base.testable_mixin.MockDataProvider` 生成
- 轻量级，快速生成
- 适用于单元测试

### 测试夹具
- 预定义的测试数据集
- 标准化的测试场景
- 可复用的测试环境

### 样本数据
- 小规模的真实数据样本
- 用于验证算法正确性
- 保护隐私的脱敏数据

## 测试最佳实践

1. **命名规范**: 测试文件以 `test_` 开头，测试函数以 `test_` 开头
2. **测试隔离**: 每个测试独立，不依赖其他测试的状态
3. **Mock使用**: 对外部依赖使用mock，提高测试速度和稳定性
4. **断言清晰**: 使用描述性的断言消息
5. **测试文档**: 复杂测试添加文档说明测试目的

## 测试文件管理规范

### ⚠️ 强制规范

1. **所有测试文件必须放在tests/目录下**
   - ❌ 禁止在项目根目录创建测试文件
   - ❌ 禁止在业务代码目录（factors/、data/等）创建测试文件
   - ✅ 所有测试文件统一在tests/子目录管理

2. **测试文件分类存放**
   ```
   tests/
   ├── unit/              # 单元测试（快速、隔离、mock数据）
   ├── integration/       # 集成测试（多模块协作、真实数据）
   │   └── factors/       # 因子相关集成测试
   ├── experimental/      # 实验性测试（新框架验证、原型测试）
   └── performance/       # 性能测试（基准测试、压力测试）
   ```

3. **测试文件命名规范**
   - 单元测试: `test_<module_name>.py`
   - 集成测试: `test_<feature_name>.py`
   - 性能测试: `benchmark_<feature>.py`
   - 实验性测试: `test_<experiment_name>_experimental.py`

### 测试文件放置决策树

```
新建测试文件，应该放哪里？
│
├─ 测试单个函数/类？ → tests/unit/
│
├─ 测试多模块协作？ → tests/integration/
│
├─ 验证新框架/实验性功能？ → tests/experimental/
│
└─ 性能基准测试？ → tests/performance/
```

### 常见错误示例

❌ **错误做法**:
```bash
# 在根目录创建测试文件
./test_my_feature.py
./debug_something.py

# 在业务代码目录创建测试
factors/test_factor.py
data/processor/test_processor.py
```

✅ **正确做法**:
```bash
# 按分类放入tests/子目录
tests/integration/test_my_feature.py
tests/unit/test_processor.py
tests/experimental/test_new_framework_experimental.py
```

### 代码审查检查清单

提交代码前检查：
- [ ] 没有在根目录创建test_*.py文件
- [ ] 没有在业务代码目录创建测试文件
- [ ] 测试文件已按分类正确放置
- [ ] 测试文件命名符合规范
- [ ] 复杂测试添加了文档说明

### 清理旧测试文件

如发现测试文件位置不规范：
```bash
# 移动到正确位置
mv ./test_*.py tests/integration/
mv data/test_scripts/*.py tests/integration/

# 或使用git追踪移动
git mv ./test_*.py tests/integration/
```