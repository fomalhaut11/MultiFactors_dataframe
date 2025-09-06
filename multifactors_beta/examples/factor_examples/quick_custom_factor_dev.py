#!/usr/bin/env python3
"""
快速自定义因子开发工具
简化因子开发流程的一站式工具
"""

import sys
import os
import argparse
from pathlib import Path
import logging

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_factor_template(factor_name: str, formula: str, description: str = ""):
    """创建因子模板代码"""
    
    class_name = ''.join(word.capitalize() for word in factor_name.split('_'))
    
    template = f'''#!/usr/bin/env python3
"""
自定义因子: {factor_name}
{description}
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any
import logging

from factors.base.factor_base import FactorBase
from factors.base.data_processing_mixin import DataProcessingMixin
from factors.base.validation import DataValidator

logger = logging.getLogger(__name__)


class {class_name}(FactorBase, DataProcessingMixin):
    """
    {factor_name} 因子
    
    计算公式：{formula}
    描述：{description}
    """
    
    def __init__(self):
        super().__init__()
        self.factor_name = "{factor_name}"
        self.factor_description = "{description}"
        
        # TODO: 设置必需的数据字段
        self.required_fields = [
            # 在这里添加需要的数据字段
        ]
    
    def validate_data_requirements(self, data: Dict[str, pd.DataFrame]) -> bool:
        """验证数据完整性"""
        try:
            # TODO: 实现数据验证逻辑
            return True
        except Exception as e:
            logger.error(f"数据验证失败: {{e}}")
            return False
    
    def calculate(self, data: Dict[str, pd.DataFrame]) -> pd.Series:
        """
        计算因子值
        
        Parameters
        ----------
        data : Dict[str, pd.DataFrame]
            输入数据字典
            
        Returns
        -------
        pd.Series
            计算得到的因子值
        """
        try:
            logger.info(f"开始计算{{self.factor_name}}因子")
            
            # 验证数据
            if not self.validate_data_requirements(data):
                return pd.Series()
            
            # TODO: 实现具体的计算逻辑
            # 示例代码：
            # result = data['field1'] / data['field2']
            
            result = pd.Series()  # 替换为实际计算
            result.name = self.factor_name
            
            logger.info(f"✅ {{self.factor_name}}因子计算完成")
            return result
            
        except Exception as e:
            logger.error(f"计算{{self.factor_name}}因子失败: {{e}}")
            return pd.Series()


def create_{factor_name.lower()}() -> {class_name}:
    """创建{factor_name}因子实例"""
    return {class_name}()


def register_factor_metadata():
    """注册因子到元数据系统"""
    try:
        from factors.meta import get_factor_registry, FactorType, NeutralizationCategory
        
        registry = get_factor_registry()
        
        registry.register_factor(
            name="{factor_name}",
            factor_type=FactorType.DERIVED,  # 根据需要调整
            description="{description}",
            formula="{formula}",
            neutralization_category=NeutralizationCategory.OPTIONAL_NEUTRALIZE,
            generator="{class_name}",
            tags=["custom"],
            priority=5
        )
        
        logger.info("✅ 因子元数据注册成功")
        
    except Exception as e:
        logger.warning(f"因子元数据注册失败: {{e}}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    register_factor_metadata()
    print("{factor_name}因子模板创建完成！")
'''
    
    return template


def generate_calculation_example(factor_name: str):
    """生成计算示例代码"""
    
    example = f'''#!/usr/bin/env python3
"""
{factor_name} 因子计算示例
"""

import sys
import os
import pandas as pd
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from factors.generator.custom.{factor_name.lower()}_factor import create_{factor_name.lower()}, register_factor_metadata
from data.fetcher.data_fetcher import DataFetcher
from config import get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """计算{factor_name}因子"""
    try:
        # 1. 注册因子元数据
        register_factor_metadata()
        
        # 2. 准备数据
        data_fetcher = DataFetcher()
        
        # TODO: 根据因子需求准备相应的数据
        data = {{
            'financial_data': data_fetcher.get_financial_data(),
            # 添加其他需要的数据
        }}
        
        # 3. 创建因子并计算
        factor = create_{factor_name.lower()}()
        result = factor.calculate(data)
        
        # 4. 保存结果
        if not result.empty:
            output_path = os.path.join(get_config('main.paths.raw_factors'), f'{factor_name}.pkl')
            result.to_pickle(output_path)
            logger.info(f"因子结果已保存: {{output_path}}")
        
        # 5. 显示统计信息
        if not result.empty:
            print(f"\\n{factor_name} 因子统计:")
            print(f"样本数: {{len(result)}}")
            print(f"有效值: {{result.notna().sum()}}")
            print(f"均值: {{result.mean():.6f}}")
            print(f"标准差: {{result.std():.6f}}")
        
    except Exception as e:
        logger.error(f"因子计算失败: {{e}}")


if __name__ == "__main__":
    main()
'''
    
    return example


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="快速自定义因子开发工具")
    parser.add_argument('action', choices=['create', 'example'], help='操作类型')
    parser.add_argument('--name', required=True, help='因子名称')
    parser.add_argument('--formula', help='计算公式')
    parser.add_argument('--description', default='', help='因子描述')
    parser.add_argument('--output-dir', default='factors/generator/custom', help='输出目录')
    
    args = parser.parse_args()
    
    try:
        # 创建输出目录
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        factor_name = args.name
        formula = args.formula or "请填写计算公式"
        description = args.description
        
        if args.action == 'create':
            # 创建因子模板
            template_code = create_factor_template(factor_name, formula, description)
            
            factor_file = output_dir / f"{factor_name.lower()}_factor.py"
            with open(factor_file, 'w', encoding='utf-8') as f:
                f.write(template_code)
            
            print(f"✅ 因子模板已创建: {factor_file}")
            
        elif args.action == 'example':
            # 创建计算示例
            example_code = generate_calculation_example(factor_name)
            
            example_file = output_dir / f"calculate_{factor_name.lower()}_example.py"
            with open(example_file, 'w', encoding='utf-8') as f:
                f.write(example_code)
            
            print(f"✅ 计算示例已创建: {example_file}")
        
        # 创建__init__.py文件
        init_file = output_dir / "__init__.py"
        if not init_file.exists():
            init_content = f'''"""
自定义因子模块
"""

from .{factor_name.lower()}_factor import create_{factor_name.lower()}, {factor_name.replace('_', '')}

__all__ = ['create_{factor_name.lower()}', '{factor_name.replace('_', '')}']
'''
            with open(init_file, 'w', encoding='utf-8') as f:
                f.write(init_content)
        
        print("\\n📖 接下来的步骤:")
        print(f"1. 编辑 {output_dir}/{factor_name.lower()}_factor.py 实现具体计算逻辑")
        print(f"2. 运行 python {output_dir}/calculate_{factor_name.lower()}_example.py 测试因子")
        print(f"3. 使用 python factor_manager.py show {factor_name} 查看注册信息")
        print(f"4. 使用测试流水线验证因子有效性")
        
    except Exception as e:
        logger.error(f"操作失败: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())