# 多因子量化投资系统

## 项目简介

本项目是一个完整的多因子量化投资框架，提供了从数据获取、因子计算到投资组合管理的全流程解决方案。系统采用模块化设计，支持灵活扩展和自定义开发。

## 快速开始

### 1. 环境准备

```bash
# Python 3.8+ 环境
python --version

# 安装依赖（如有requirements.txt）
pip install pandas numpy scipy statsmodels
```

### 2. 数据准备

```bash
# 生成预处理的辅助数据
python data/prepare_auxiliary_data.py

# 获取历史价格数据（首次运行）
python get_historical_price_2014.py

# 更新价格数据（日常更新）
python scheduled_data_updater.py --data-type price
```

### 3. 因子计算示例

```python
from factors.financial.fundamental_factors import BPFactor, EPFactor

# 计算BP因子
bp_factor = BPFactor()
bp_values = bp_factor.calculate(
    financial_data=financial_data,
    market_cap=market_cap,
    release_dates=release_dates,
    trading_dates=trading_dates
)

# 计算EP因子（TTM方式）
ep_factor = EPFactor(method='ttm')
ep_values = ep_factor.calculate(
    financial_data=financial_data,
    market_cap=market_cap,
    release_dates=release_dates,
    trading_dates=trading_dates
)
```

详细示例请参考 `examples/factor_calculation_demo.py`

## 项目结构

```
mulitfactors_beta/
├── core/                    # 核心功能模块
│   ├── config_manager.py   # 配置管理
│   ├── database/           # 数据库连接管理
│   └── utils/              # 工具函数
├── data/                   # 数据处理模块
│   ├── auxiliary/          # 预处理数据存储
│   ├── fetcher/           # 数据获取
│   ├── processor/         # 数据处理
│   └── prepare_auxiliary_data.py  # 数据预处理主程序
├── factors/               # 因子计算模块
│   ├── base/             # 基础类和处理器
│   ├── financial/        # 基本面因子
│   ├── technical/        # 技术因子
│   └── risk/            # 风险因子
├── validation/           # 因子验证脚本
├── examples/            # 使用示例
├── tests/              # 测试用例
└── docs/               # 项目文档
```

## 主要功能

### 数据管理
- **自动化数据更新**: 支持定时任务和手动更新
- **增量数据获取**: 智能检测缺失数据，只更新必要部分
- **数据预处理**: 统一财务数据格式，处理财报发布时间
- **多数据源支持**: 支持聚宽、Wind等多个数据源

### 因子计算
- **基本面因子**: 
  - BP (Book-to-Price): 账面价值比
  - EP (Earnings-to-Price): 盈利收益率
  - ROE (Return on Equity): 净资产收益率
  - 更多因子开发中...
  
- **时间序列处理**:
  - TTM (Trailing Twelve Months): 滚动12月
  - YoY (Year-over-Year): 同比增长
  - QoQ (Quarter-over-Quarter): 环比增长
  - 单季度数据提取

### 系统特性
- **模块化设计**: 各功能模块独立，易于维护和扩展
- **错误处理**: 完善的异常处理和日志记录
- **性能优化**: 支持大数据集处理，内存使用优化
- **跨平台支持**: Windows/Linux兼容，处理编码问题

## 使用文档

- [数据预处理指南](docs/数据预处理功能完整指南.md)
- [因子计算指南](docs/因子计算模块迁移指南.md)
- [数据更新说明](docs/Price数据增量更新使用说明.md)
- [数据字段说明](docs/数据字段理解修正说明.md)

## 配置说明

主要配置文件 `config.yaml` 包含：
```yaml
# 数据库连接配置
database:
  host: your_host
  user: your_user
  password: your_password

# 数据路径配置
paths:
  data_root: E:\Documents\PythonProject\StockProject\StockData
  project_root: E:\Documents\PythonProject\StockProject\MultiFactors\mulitfactors_beta

# 系统参数
system:
  log_level: INFO
  backup_days: 3
```

## 重要说明

### 数据字段理解
- **reportday**: 财报发布日期（报表公告日）
- **tradingday**: 财报截止日期（报告期末）
- **d_year + d_quarter**: 可靠的报告期标识

### Windows用户注意
- 项目包含 `utils/io_utils.py` 处理编码问题
- 使用绝对路径避免路径问题
- 注意反斜杠转义或使用原始字符串

## 开发计划

- [x] 数据获取和更新系统
- [x] 基本面因子计算框架
- [x] 数据预处理模块
- [ ] 回测系统开发
- [ ] 组合优化模块
- [ ] 实时监控系统
- [ ] Web管理界面

## 贡献指南

欢迎提交Issue和Pull Request。开发新功能请：
1. Fork本项目
2. 创建功能分支
3. 提交变更
4. 发起Pull Request

## 版本历史

- **v2.0-beta** (2025-08-01): 
  - 重构因子计算框架
  - 修正数据字段理解
  - 优化性能和稳定性
  
- **v1.2-beta** (2025-07-30): 
  - 完成数据更新系统
  - 实现连接池管理

## 联系方式

如有问题或建议，请提交Issue或联系项目维护者。

---

**项目状态**: Beta测试中  
**License**: MIT