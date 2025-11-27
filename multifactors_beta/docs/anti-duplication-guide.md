# 防重复造轮子完整指南

## 关键约束（强制执行）

**绝对禁止重复实现项目中已存在的功能，这是最高优先级的约束。**

---

## 检查清单（强制执行，不可跳过）

### 第一步：搜索现有实现
在实现任何功能前，必须先执行以下搜索：

```bash
# 1. 搜索关键功能词
grep -r "ttm\|TTM" factors/generators/
grep -r "yoy\|YOY\|同比" factors/generators/
grep -r "qoq\|QOQ\|环比" factors/generators/

# 2. 搜索具体计算方法
grep -r "rolling\|shift\|pct_change" factors/generators/
grep -r "calculate_\|compute_" factors/generators/

# 3. 检查可用工具
cat factors/generators/__init__.py
```

### 第二步：验证工具可用性
- [ ] 确认factors/generators中没有相同功能
- [ ] 确认不会使用generator_backup目录的代码
- [ ] 确认已读取generators/__init__.py了解可用工具

---

## 常见重复造轮子情况

### 1. TTM（滚动12个月）计算
```python
# ❌ 绝对禁止：自己实现TTM
def calculate_ttm_wrong(data):
    return data.groupby('StockCodes').rolling(4).sum()

# ✅ 必须使用：现有工具
from factors.generators import calculate_ttm
ttm_data = calculate_ttm(financial_data)
```

### 2. 同比增长率计算
```python
# ❌ 绝对禁止：自己实现同比
def calculate_yoy_wrong(data):
    return data.pct_change(4) * 100

# ✅ 必须使用：现有工具  
from factors.generators import calculate_yoy
yoy_data = calculate_yoy(financial_data)
```

### 3. 环比增长率计算
```python
# ❌ 绝对禁止：自己实现环比
def calculate_qoq_wrong(data):
    return data.pct_change(1) * 100

# ✅ 必须使用：现有工具
from factors.generators import calculate_qoq
qoq_data = calculate_qoq(financial_data)
```

### 4. 单季度计算
```python
# ❌ 绝对禁止：自己实现单季度
def single_quarter_wrong(data):
    return data - data.shift(1)

# ✅ 必须使用：现有工具
from factors.generators import calculate_single_quarter
single_q_data = calculate_single_quarter(financial_data)
```

### 5. 数据日频扩展
```python
# ❌ 绝对禁止：自己实现日频扩展
def expand_to_daily_wrong(quarterly_data, dates):
    # 复杂的扩展逻辑...
    pass

# ✅ 必须使用：现有工具
from factors.generators import expand_to_daily_vectorized
daily_data = expand_to_daily_vectorized(quarterly_data, release_dates, trading_dates)
```

---

## 发现重复实现时的处理流程

### 立即停止原则
**发现已有工具时，必须立即停止当前实现，改用现有工具。**

### 处理步骤：
1. **停止编码** - 立即停止当前的实现工作
2. **删除重复代码** - 移除已写的重复实现代码  
3. **导入现有工具** - 使用factors.generators中的工具
4. **调整调用方式** - 修改代码调用现有工具
5. **验证结果** - 确认使用现有工具后结果正确

### 示例修正过程：
```python
# 原本的错误实现
def my_ttm_calculation(data):
    # 复杂的TTM计算逻辑...
    pass

# 修正后的正确实现
from factors.generators import calculate_ttm

def my_factor_calculation(data):
    # 直接使用现有工具
    ttm_data = calculate_ttm(data)
    # 继续其他特有逻辑...
    return result
```

---

## 项目架构约束

### 可用工具目录结构
```
factors/generators/          # ✅ 官方工具，必须使用
├── __init__.py             # 查看所有可用工具
├── financial/              # 财务计算工具
├── technical/              # 技术指标工具
├── mixed/                  # 混合数据工具
└── alpha191/               # Alpha191工具

factors/generator_backup/    # ❌ 备份文件，禁止使用
```

### 导入约束
```python
# ✅ 正确：使用官方工具
from factors.generators import calculate_ttm, calculate_yoy
from factors.generators import FinancialReportProcessor

# ❌ 错误：使用备份文件
from factors.generator_backup.financial import xxx
```

---

## 强制验证检查点

### 代码审查检查点
编写完代码后，必须检查：

1. **导入检查**
   - [ ] 所有导入都来自factors.generators/
   - [ ] 没有导入generator_backup中的任何内容
   - [ ] 没有导入过时的或备份的模块

2. **功能检查**
   - [ ] 没有重新实现TTM、YOY、QOQ计算
   - [ ] 没有重新实现数据扩展或对齐功能
   - [ ] 没有重新实现已有的数据处理逻辑

3. **工具使用检查**
   - [ ] 查阅了factors/generators/__init__.py
   - [ ] 使用了合适的现有工具
   - [ ] 参数传递符合现有工具接口

### 提交前最终检查
```bash
# 检查是否使用了backup目录
grep -r "generator_backup" your_file.py

# 检查是否重复实现TTM
grep -r "rolling(4)\|\.shift(4)" your_file.py

# 检查是否重复实现百分比变化
grep -r "pct_change\|/ .* - 1" your_file.py
```

---

## 记忆强化提醒

### 核心原则复述
**在项目的factors模块开发中，严禁重复实现任何已存在的功能。必须使用factors.generators工具集。**

### 违反后果
- 代码质量降低
- 维护困难增加
- 性能可能下降
- 架构设计混乱
- 团队开发效率降低

### 成功标志
- 代码中只导入factors.generators的工具
- 没有重复的计算逻辑
- 充分利用现有工具的功能
- 代码简洁、易维护

---

## 应急措施

### 如果不确定是否重复
1. **暂停开发** - 立即停止编码
2. **搜索确认** - 执行上述搜索命令
3. **查阅文档** - 查看generators/__init__.py
4. **请求确认** - 如仍不确定，请求明确指导

### 如果发现已经违反
1. **承认错误** - 明确认识到重复造轮子的问题
2. **立即修正** - 删除重复代码，使用现有工具
3. **加强检查** - 增强后续开发中的检查意识
4. **记录教训** - 避免同类错误再次发生

---

**记住：使用现有工具不是偷懒，而是遵循良好的软件工程实践。**