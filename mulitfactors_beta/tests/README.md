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