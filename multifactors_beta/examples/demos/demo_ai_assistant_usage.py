#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI量化助手使用演示 - 完整示例

本演示展示如何使用AI助手进行量化研究：
1. 智能因子生成
2. 因子筛选和分析
3. 自然语言交互
4. 工作流自动化

运行要求：
- Python 3.9+
- 完整的multifactors_beta环境
- 有效的数据库连接
"""

import sys
import os
import logging
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from factors.ai_quant_assistant_v2 import AIQuantAssistant
from config import get_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AIAssistantDemo:
    """AI助手演示类"""
    
    def __init__(self):
        """初始化演示"""
        self.assistant = AIQuantAssistant()
        self.config = get_config()
        
    def demo_basic_usage(self):
        """演示基础使用方法"""
        print("\n=== AI助手基础使用演示 ===")
        
        # 自然语言因子研究请求
        requests = [
            "帮我生成一个基于盈利能力的因子，使用ROE指标",
            "筛选出表现最好的价值类因子",
            "分析当前因子库中的财务因子表现",
            "生成一个混合因子，结合财务数据和市场数据"
        ]
        
        for i, request in enumerate(requests, 1):
            print(f"\n{i}. 用户请求: {request}")
            
            try:
                # 处理请求
                result = self.assistant.process_request(request)
                print(f"   AI助手回复: {result.get('response', '处理完成')}")
                print(f"   建议操作: {result.get('suggested_actions', '无')}")
                
                if result.get('factor_data') is not None:
                    print(f"   因子数据形状: {result['factor_data'].shape}")
                    
            except Exception as e:
                print(f"   处理失败: {e}")
                
    def demo_factor_generation(self):
        """演示智能因子生成"""
        print("\n=== 智能因子生成演示 ===")
        
        generation_tasks = [
            {
                "request": "生成ROE_ttm因子",
                "expected_type": "financial"
            },
            {
                "request": "创建一个动量因子，基于20天价格变化",
                "expected_type": "technical"
            },
            {
                "request": "生成BP因子（净资产与市值比）",
                "expected_type": "mixed"
            }
        ]
        
        for task in generation_tasks:
            print(f"\n请求: {task['request']}")
            print(f"预期类型: {task['expected_type']}")
            
            try:
                result = self.assistant.process_request(task['request'])
                
                if result.get('factor_generated'):
                    print("✅ 因子生成成功")
                    print(f"   因子名称: {result.get('factor_name', '未知')}")
                    print(f"   因子类型: {result.get('factor_type', '未知')}")
                    print(f"   数据质量: {result.get('data_quality', '未知')}")
                else:
                    print("❌ 因子生成失败")
                    
            except Exception as e:
                print(f"❌ 处理异常: {e}")
                
    def demo_factor_screening(self):
        """演示智能因子筛选"""
        print("\n=== 智能因子筛选演示 ===")
        
        screening_requests = [
            "筛选IC值大于0.05的因子",
            "找出最稳定的财务因子",
            "筛选换手率低于5%的因子",
            "找出与市场相关性最低的因子"
        ]
        
        for request in screening_requests:
            print(f"\n筛选请求: {request}")
            
            try:
                result = self.assistant.process_request(request)
                
                selected_factors = result.get('selected_factors', [])
                print(f"筛选结果: {len(selected_factors)} 个因子")
                
                for factor in selected_factors[:3]:  # 只显示前3个
                    print(f"  - {factor.get('name', '未知')}: "
                          f"IC={factor.get('ic', 'N/A'):.3f}, "
                          f"稳定性={factor.get('stability', 'N/A'):.3f}")
                          
            except Exception as e:
                print(f"筛选失败: {e}")
                
    def demo_performance_analysis(self):
        """演示智能性能分析"""
        print("\n=== 智能性能分析演示 ===")
        
        analysis_requests = [
            "分析ROE_ttm因子的历史表现",
            "比较所有价值因子的表现",
            "分析因子在不同市场环境下的表现",
            "评估因子组合的风险收益特征"
        ]
        
        for request in analysis_requests:
            print(f"\n分析请求: {request}")
            
            try:
                result = self.assistant.process_request(request)
                
                analysis_result = result.get('analysis_result', {})
                print("分析结果:")
                
                for metric, value in analysis_result.items():
                    if isinstance(value, (int, float)):
                        print(f"  {metric}: {value:.3f}")
                    else:
                        print(f"  {metric}: {value}")
                        
            except Exception as e:
                print(f"分析失败: {e}")
                
    def demo_workflow_automation(self):
        """演示工作流自动化"""
        print("\n=== 工作流自动化演示 ===")
        
        workflow_request = """
        请执行完整的因子研究工作流：
        1. 生成ROE相关的财务因子
        2. 对因子进行质量检验
        3. 与现有因子进行相关性分析
        4. 给出投资建议
        """
        
        print("复杂工作流请求:")
        print(workflow_request)
        
        try:
            result = self.assistant.process_request(workflow_request)
            
            workflow_steps = result.get('workflow_steps', [])
            print(f"\n工作流包含 {len(workflow_steps)} 个步骤:")
            
            for i, step in enumerate(workflow_steps, 1):
                status = "✅" if step.get('completed') else "⏳"
                print(f"  {i}. {status} {step.get('description', '未知步骤')}")
                
            final_recommendation = result.get('recommendation', '无建议')
            print(f"\n最终建议: {final_recommendation}")
            
        except Exception as e:
            print(f"工作流执行失败: {e}")
            
    def demo_configuration_usage(self):
        """演示配置系统使用"""
        print("\n=== 配置系统演示 ===")
        
        # 显示AI助手配置
        ai_config = self.config.get('agents', {}).get('ai_assistant', {})
        
        print("AI助手配置:")
        print(f"  版本: {ai_config.get('version', '未知')}")
        print(f"  路由策略: {ai_config.get('routing_strategy', '未知')}")
        print(f"  最大Token: {ai_config.get('max_tokens', '未知')}")
        
        capabilities = ai_config.get('capabilities', [])
        print(f"  能力清单: {', '.join(capabilities)}")
        
        # 演示配置修改
        print("\n演示动态配置修改:")
        original_max_tokens = ai_config.get('max_tokens', 3000)
        print(f"  原始max_tokens: {original_max_tokens}")
        
        # 注意：实际修改需要谨慎处理
        print("  (配置修改功能需要管理员权限)")
        
    def run_complete_demo(self):
        """运行完整演示"""
        print("🤖 AI量化助手完整使用演示")
        print("=" * 50)
        
        try:
            # 检查AI助手状态
            print("检查AI助手状态...")
            if hasattr(self.assistant, 'is_ready') and self.assistant.is_ready():
                print("✅ AI助手已就绪")
            else:
                print("⚠️  AI助手可能未完全初始化")
                
            # 运行各项演示
            self.demo_basic_usage()
            self.demo_factor_generation()
            self.demo_factor_screening() 
            self.demo_performance_analysis()
            self.demo_workflow_automation()
            self.demo_configuration_usage()
            
            print("\n" + "=" * 50)
            print("🎉 AI助手演示完成!")
            print("\n使用提示:")
            print("1. 确保数据库连接正常")
            print("2. 检查因子数据是否最新")
            print("3. 根据需要调整AI助手配置")
            print("4. 查看 factors/CLAUDE_USAGE_GUIDE.md 获取详细使用说明")
            
        except Exception as e:
            logger.error(f"演示运行失败: {e}")
            print(f"❌ 演示失败: {e}")
            print("请检查系统配置和数据连接")


def main():
    """主函数"""
    try:
        demo = AIAssistantDemo()
        demo.run_complete_demo()
        
    except ImportError as e:
        print(f"❌ 模块导入失败: {e}")
        print("请确保已正确安装所有依赖包")
        
    except Exception as e:
        print(f"❌ 演示启动失败: {e}")
        print("请检查项目配置和环境设置")


if __name__ == "__main__":
    main()