"""
因子文件仓库模块

每个因子作为独立的Python文件存储，支持动态加载和管理。
这是对现有集中式因子管理的补充，两种方式可以并存。
"""

# 导入各类别的因子
from .technical import *
from .profitability import *  
from .value import *
from .quality import *
from .experimental import *

__version__ = '1.0.0'