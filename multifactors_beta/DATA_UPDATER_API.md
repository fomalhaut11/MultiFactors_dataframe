# 数据更新器 API 文档

## 📚 API 概述

本文档详细说明了多因子量化投资系统中各个数据更新器的API接口、参数和使用方法。

**目标用户**: 开发者、数据工程师、系统集成人员  
**更新日期**: 2025-08-29

## 🏗️ 基础架构

### 基类设计

```python
class BaseDataUpdater:
    """数据更新器基类"""
    
    def get_update_info(self) -> Dict:
        """获取更新信息"""
        
    def needs_update(self) -> bool:
        """检查是否需要更新"""
        
    def update_data(self) -> bool:
        """执行数据更新"""
        
    def get_health_status(self) -> Dict:
        """获取健康状态"""
```

## 💰 价格数据更新器

### PriceDataUpdater

**类路径**: `scheduled_data_updater.PriceDataUpdater`  
**依赖**: `data.fetcher.incremental_price_updater.IncrementalPriceUpdater`

#### API 接口

```python
class PriceDataUpdater(BaseDataUpdater):
    """价格数据更新器"""
    
    def __init__(self):
        """初始化价格数据更新器"""
        
    def get_update_info(self) -> Dict:
        """
        获取价格数据更新信息
        
        Returns:
            Dict: {
                'data_type': 'price_data',
                'price_file_exists': bool,
                'file_size_mb': float,
                'local_latest_date': str,      # 'YYYY-MM-DD'
                'db_latest_date': str,         # 'YYYY-MM-DD'
                'need_update': bool,
                'timestamp': str               # ISO格式时间戳
            }
        """
        
    def needs_update(self) -> bool:
        """
        检查是否需要更新价格数据
        
        Returns:
            bool: True表示需要更新，False表示已是最新
            
        判断逻辑:
            本地最新日期 < 数据库最新日期
        """
        
    def update_data(self) -> bool:
        """
        执行价格数据更新
        
        Returns:
            bool: True表示更新成功，False表示更新失败
            
        更新流程:
            1. 检查本地和数据库最新日期
            2. 如果需要更新，创建备份
            3. 增量获取新数据
            4. 合并并去重数据
            5. 保存更新后的文件
        """
        
    def get_health_status(self) -> Dict:
        """
        获取价格数据健康状态
        
        Returns:
            Dict: {
                'data_type': 'price_data',
                'timestamp': str,
                'file_exists': bool,
                'file_size_mb': float,
                'local_latest_date': str,
                'db_latest_date': str,
                'need_update': bool,
                'data_gap_days': int,         # 数据延迟天数
                'status': str,                # 'healthy'/'warning'/'error'
                'message': str                # 状态描述
            }
        """
```

#### 使用示例

```python
from scheduled_data_updater import PriceDataUpdater

# 创建更新器
updater = PriceDataUpdater()

# 检查健康状态
health = updater.get_health_status()
print(f"价格数据状态: {health['status']} - {health['message']}")

# 检查是否需要更新
if updater.needs_update():
    print("正在更新价格数据...")
    success = updater.update_data()
    if success:
        print("价格数据更新成功")
    else:
        print("价格数据更新失败")
else:
    print("价格数据已是最新")
```

## 💼 财务数据更新器

### FinancialDataUpdater

**类路径**: `scheduled_data_updater.FinancialDataUpdater`  
**依赖**: `data.fetcher.incremental_financial_updater.IncrementalFinancialUpdater`

#### API 接口

```python
class FinancialDataUpdater(BaseDataUpdater):
    """财务数据更新器"""
    
    def get_update_info(self) -> Dict:
        """
        获取财务数据更新信息
        
        Returns:
            Dict: {
                'data_type': 'financial_data',
                'timestamp': str,
                'tables': {
                    'fzb': {
                        'description': '资产负债表',
                        'file_exists': bool,
                        'file_size_mb': float,
                        'local_latest_date': int,    # YYYYMMDD格式
                        'db_latest_date': int,       # YYYYMMDD格式
                        'need_update': bool
                    },
                    'lrb': {...},     # 利润表
                    'xjlb': {...}     # 现金流量表
                }
            }
        """
        
    def needs_update(self) -> bool:
        """
        检查是否需要更新财务数据
        
        Returns:
            bool: 任意一张表需要更新则返回True
        """
        
    def update_data(self) -> bool:
        """
        执行财务数据更新（三张表）
        
        Returns:
            bool: 所有表都更新成功才返回True
            
        更新流程:
            1. 逐表检查local_date字段
            2. 对需要更新的表执行增量更新
            3. 合并数据并去重
            4. 原子性保存所有表
        """
        
    def get_health_status(self) -> Dict:
        """
        获取财务数据健康状态
        
        Returns:
            Dict: {
                'data_type': 'financial_data',
                'timestamp': str,
                'status': str,              # 'healthy'/'warning'/'error'
                'message': str,
                'total_tables': int,        # 总表数(3)
                'existing_tables': int,     # 存在的表数
                'total_size_mb': float,     # 总文件大小
                'tables_detail': str        # 各表状态详情
            }
        """
```

#### 使用示例

```python
from scheduled_data_updater import FinancialDataUpdater

updater = FinancialDataUpdater()

# 获取详细更新信息
info = updater.get_update_info()
for table_name, table_info in info['tables'].items():
    print(f"{table_info['description']}: {'需要更新' if table_info['need_update'] else '已是最新'}")

# 执行更新
if updater.needs_update():
    success = updater.update_data()
    print(f"财务数据更新: {'成功' if success else '失败'}")
```

## 📊 涨跌停数据更新器

### StopPriceDataUpdater

**类路径**: `scheduled_data_updater.StopPriceDataUpdater`

#### API 接口

```python
class StopPriceDataUpdater(BaseDataUpdater):
    """涨跌停数据更新器"""
    
    def get_update_info(self) -> Dict:
        """
        获取涨跌停数据更新信息
        
        Returns:
            Dict: {
                'data_type': 'stop_price_data',
                'stop_price_file_exists': bool,
                'file_size_mb': float,
                'local_latest_date': str,       # 'YYYY-MM-DD'
                'db_latest_date': str,          # 'YYYY-MM-DD'
                'need_update': bool,
                'data_gap_days': int            # 数据缺口天数
            }
        """
        
    def needs_update(self) -> bool:
        """检查涨跌停数据是否需要更新"""
        
    def update_data(self) -> bool:
        """执行涨跌停数据更新"""
        
    def get_health_status(self) -> Dict:
        """
        获取涨跌停数据健康状态
        
        健康判定标准:
            - 数据缺口 > 5天: ERROR
            - 数据缺口 > 2天: WARNING  
            - 其他情况: HEALTHY
        """
```

## 📈 ST股票数据更新器

### STDataUpdater

**类路径**: `scheduled_data_updater.STDataUpdater`

#### API 接口

```python
class STDataUpdater(BaseDataUpdater):
    """ST股票数据更新器"""
    
    def get_update_info(self) -> Dict:
        """
        获取ST股票数据信息
        
        Returns:
            Dict: {
                'data_type': 'st_data',
                'st_file_exists': bool,
                'st_file_size_mb': float,
                'st_latest_date': str,          # 最新数据日期
                'need_update': bool,
                'days_since_update': int        # 距离上次更新天数
            }
        """
        
    def needs_update(self) -> bool:
        """
        检查ST数据是否需要更新
        
        更新策略:
            - 文件不存在: 需要更新
            - 距离上次更新 > 30天: 需要更新
            - 其他: 不需要更新
        """
        
    def update_data(self) -> bool:
        """
        执行ST股票数据更新
        
        更新流程:
            1. 从数据库获取全量ST数据
            2. 直接覆盖本地文件（非增量）
            3. 记录统计信息
        """
```

## 🏢 板块数据更新器

### SectorChangesDataUpdater

**类路径**: `scheduled_data_updater.SectorChangesDataUpdater`

#### API 接口

```python
class SectorChangesDataUpdater(BaseDataUpdater):
    """板块进出数据更新器"""
    
    def get_update_info(self) -> Dict:
        """
        获取板块数据更新信息
        
        Returns:
            Dict: {
                'data_type': 'sector_changes_data',
                'file_exists': bool,
                'latest_date': str,             # 最新sel_day
                'record_count': int,
                'days_since_update': int,
                'need_update': bool
            }
        """
        
    def needs_update(self, force: bool = False) -> bool:
        """
        检查板块数据是否需要更新
        
        Args:
            force: 是否强制更新
            
        Returns:
            bool: 是否需要更新
        """
        
    def update_data(self) -> bool:
        """
        执行板块数据更新
        
        更新策略:
            - 首次运行: 从20200101开始全量获取
            - 增量更新: 从最新sel_day+1开始获取
            - 数据合并: 去重并排序
        """
```

## 🔄 统一更新管理器

### ScheduledDataUpdater

**类路径**: `scheduled_data_updater.ScheduledDataUpdater`

#### API 接口

```python
class ScheduledDataUpdater:
    """定时数据更新管理器"""
    
    def __init__(self, data_types: Optional[List[str]] = None):
        """
        初始化更新管理器
        
        Args:
            data_types: 要管理的数据类型列表
                       None表示使用默认配置['price', 'stop_price']
                       
        可用数据类型:
            - 'price': 价格数据
            - 'stop_price': 涨跌停数据  
            - 'financial': 财务数据
            - 'sector_changes': 板块数据
            - 'st': ST股票数据
        """
        
    def should_update_now(self) -> bool:
        """
        判断是否应该在当前时间更新
        
        Returns:
            bool: 是否适合更新
            
        更新时间策略:
            - 工作日 16:00-23:59: 允许更新
            - 周末全天: 允许更新
            - 其他时间: 不允许更新
        """
        
    def run_single_update(self, data_type: str, force: bool = False) -> 'DataUpdateResult':
        """
        运行单个数据类型的更新
        
        Args:
            data_type: 数据类型名称
            force: 是否强制更新（忽略时间检查）
            
        Returns:
            DataUpdateResult: 更新结果对象
                .data_type: str      # 数据类型
                .success: bool       # 是否成功
                .message: str        # 结果消息
                .duration: float     # 耗时（秒）
                .details: dict       # 详细信息
                .timestamp: datetime # 时间戳
        """
        
    def run_all_updates(self, force: bool = False) -> List['DataUpdateResult']:
        """
        运行所有活跃数据类型的更新
        
        Args:
            force: 是否强制更新
            
        Returns:
            List[DataUpdateResult]: 所有更新结果
        """
        
    def run_health_check(self) -> Dict[str, Dict]:
        """
        运行所有数据类型的健康检查
        
        Returns:
            Dict[str, Dict]: 各数据类型的健康状态
        """
```

#### 使用示例

```python
from scheduled_data_updater import ScheduledDataUpdater

# 创建更新管理器
updater = ScheduledDataUpdater(['price', 'financial'])

# 健康检查
health_results = updater.run_health_check()
for data_type, health in health_results.items():
    print(f"{data_type}: {health['status']} - {health['message']}")

# 批量更新
results = updater.run_all_updates(force=True)
for result in results:
    status = "✅" if result.success else "❌"
    print(f"{status} {result.data_type}: {result.message} ({result.duration:.1f}s)")

# 单个更新
result = updater.run_single_update('price', force=True)
if result.success:
    print(f"价格数据更新成功，耗时{result.duration:.1f}秒")
    print(f"详细信息: {result.details}")
```

## 🛠️ 高级用法

### 自定义更新策略

```python
class CustomUpdateManager:
    """自定义更新管理器"""
    
    def __init__(self):
        self.updater = ScheduledDataUpdater()
    
    def smart_update(self):
        """智能更新策略"""
        
        # 1. 先进行健康检查
        health = self.updater.run_health_check()
        
        # 2. 根据健康状态决定更新顺序
        priority_updates = []
        normal_updates = []
        
        for data_type, status in health.items():
            if status['status'] == 'error':
                priority_updates.append(data_type)
            elif status['status'] == 'warning':
                normal_updates.append(data_type)
        
        # 3. 优先处理错误状态的数据
        for data_type in priority_updates:
            result = self.updater.run_single_update(data_type, force=True)
            print(f"紧急修复 {data_type}: {'成功' if result.success else '失败'}")
        
        # 4. 处理警告状态的数据
        for data_type in normal_updates:
            result = self.updater.run_single_update(data_type)
            print(f"常规更新 {data_type}: {'成功' if result.success else '失败'}")

# 使用
manager = CustomUpdateManager()
manager.smart_update()
```

### 监控和告警

```python
import smtplib
from datetime import datetime, timedelta

class UpdateMonitor:
    """更新监控器"""
    
    def __init__(self):
        self.updater = ScheduledDataUpdater()
        self.alert_threshold = {
            'price': timedelta(days=1),      # 价格数据1天未更新告警
            'financial': timedelta(days=7),   # 财务数据7天未更新告警
            'stop_price': timedelta(days=2)   # 涨跌停2天未更新告警
        }
    
    def check_and_alert(self):
        """检查并发送告警"""
        health = self.updater.run_health_check()
        alerts = []
        
        for data_type, status in health.items():
            if status['status'] in ['error', 'warning']:
                
                # 检查更新时间
                if 'local_latest_date' in status:
                    last_update = datetime.strptime(status['local_latest_date'], '%Y-%m-%d')
                    threshold = self.alert_threshold.get(data_type, timedelta(days=1))
                    
                    if datetime.now() - last_update > threshold:
                        alerts.append(f"{data_type}: {status['message']}")
        
        if alerts:
            self.send_alert("\n".join(alerts))
    
    def send_alert(self, message):
        """发送告警邮件"""
        # 实现邮件发送逻辑
        print(f"🚨 数据更新告警:\n{message}")
```

## ⚡ 性能优化

### 并行更新

```python
import concurrent.futures
from scheduled_data_updater import ScheduledDataUpdater

class ParallelUpdater:
    """并行更新器"""
    
    def __init__(self):
        self.updater = ScheduledDataUpdater()
    
    def parallel_update(self, data_types: List[str], max_workers: int = 3):
        """并行执行多个数据类型的更新"""
        
        def update_single(data_type):
            return self.updater.run_single_update(data_type, force=True)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有更新任务
            future_to_type = {
                executor.submit(update_single, dt): dt 
                for dt in data_types
            }
            
            results = {}
            for future in concurrent.futures.as_completed(future_to_type):
                data_type = future_to_type[future]
                try:
                    result = future.result()
                    results[data_type] = result
                except Exception as e:
                    print(f"更新 {data_type} 时发生错误: {e}")
        
        return results

# 使用示例
parallel_updater = ParallelUpdater()
results = parallel_updater.parallel_update(['price', 'financial', 'stop_price'])
```

---

**API版本**: v2.1  
**维护状态**: 活跃维护  
**技术支持**: 通过GitHub Issues获取支持