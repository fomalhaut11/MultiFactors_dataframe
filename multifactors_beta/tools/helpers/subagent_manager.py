#!/usr/bin/env python3
"""
SubAgent管理器

用于管理和调度不同专业领域的AI助手，基于配置文件动态加载agent定义。

Usage:
    from subagent_manager import SubAgentManager
    
    manager = SubAgentManager()
    
    # 启动单个agent
    result = manager.invoke_agent('factor_expert', '帮我分析ROE因子的有效性')
    
    # 启动协作场景
    results = manager.invoke_scenario('new_factor_development', 'ROE_enhanced因子开发')

Author: AI Assistant
Date: 2025-08-26
"""

import yaml
import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

class SubAgentManager:
    """SubAgent管理器"""
    
    def __init__(self, config_path: str = None):
        """
        初始化SubAgent管理器
        
        Parameters
        ----------
        config_path : str, optional
            配置文件路径，默认为同目录下的subagent_config.yaml
        """
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'subagent_config.yaml')
        
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = self._setup_logging()
        
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            raise RuntimeError(f"无法加载配置文件 {self.config_path}: {e}")
    
    def _setup_logging(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger('SubAgentManager')
        
        # 创建日志目录
        log_dir = self.config.get('settings', {}).get('conversation_log_path', 'logs/agent_conversations/')
        os.makedirs(log_dir, exist_ok=True)
        
        # 配置日志处理器
        if not logger.handlers:
            handler = logging.FileHandler(
                os.path.join(log_dir, f'subagent_manager_{datetime.now().strftime("%Y%m%d")}.log'),
                encoding='utf-8'
            )
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        
        return logger
    
    def list_agents(self) -> List[str]:
        """列出所有可用的agent"""
        return list(self.config.get('agents', {}).keys())
    
    def list_scenarios(self) -> List[str]:
        """列出所有可用的协作场景"""
        return list(self.config.get('scenarios', {}).keys())
    
    def get_agent_info(self, agent_name: str) -> Dict[str, Any]:
        """获取agent详细信息"""
        agents = self.config.get('agents', {})
        if agent_name not in agents:
            raise ValueError(f"Agent '{agent_name}' 不存在。可用agents: {self.list_agents()}")
        
        return agents[agent_name]
    
    def invoke_agent(self, agent_name: str, task_description: str, **kwargs) -> str:
        """
        调用指定的agent
        
        Parameters
        ----------
        agent_name : str
            Agent名称
        task_description : str
            任务描述
        **kwargs
            其他参数传递给Task工具
            
        Returns
        -------
        str
            Agent的响应结果
        """
        # 获取agent配置
        agent_config = self.get_agent_info(agent_name)
        
        # 记录调用
        self.logger.info(f"调用Agent: {agent_name}, 任务: {task_description}")
        
        # 构建完整prompt
        full_prompt = f"""
{agent_config['prompt']}

当前任务: {task_description}

请基于你的专业知识提供详细的分析和建议。
"""
        
        try:
            # 这里需要调用Task工具 - 在实际使用时需要导入
            # 由于这是示例代码，我们返回一个模拟结果
            result = f"[模拟] {agent_name} 的响应: 已收到任务 '{task_description}'"
            
            self.logger.info(f"Agent {agent_name} 调用成功")
            return result
            
        except Exception as e:
            error_msg = f"调用Agent {agent_name} 失败: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def invoke_scenario(self, scenario_name: str, task_description: str) -> Dict[str, str]:
        """
        调用协作场景
        
        Parameters
        ----------
        scenario_name : str
            场景名称
        task_description : str
            任务描述
            
        Returns
        -------
        Dict[str, str]
            包含主要agent和支持agents响应的字典
        """
        scenarios = self.config.get('scenarios', {})
        if scenario_name not in scenarios:
            raise ValueError(f"场景 '{scenario_name}' 不存在。可用场景: {self.list_scenarios()}")
        
        scenario_config = scenarios[scenario_name]
        
        self.logger.info(f"启动协作场景: {scenario_name}, 任务: {task_description}")
        
        results = {}
        
        # 调用主要agent
        primary_agent = scenario_config['primary_agent']
        try:
            results['primary'] = {
                'agent': primary_agent,
                'response': self.invoke_agent(primary_agent, task_description)
            }
        except Exception as e:
            self.logger.error(f"主要agent {primary_agent} 调用失败: {e}")
            results['primary'] = {
                'agent': primary_agent,
                'error': str(e)
            }
        
        # 调用支持agents
        supporting_agents = scenario_config.get('supporting_agents', [])
        results['supporting'] = []
        
        for agent in supporting_agents:
            try:
                response = self.invoke_agent(
                    agent, 
                    f"协作任务: {task_description}\n请从你的专业角度提供支持意见。"
                )
                results['supporting'].append({
                    'agent': agent,
                    'response': response
                })
            except Exception as e:
                self.logger.error(f"支持agent {agent} 调用失败: {e}")
                results['supporting'].append({
                    'agent': agent,
                    'error': str(e)
                })
        
        return results
    
    def print_agent_summary(self):
        """打印所有agent的摘要信息"""
        print("🤖 可用的SubAgent列表:")
        print("=" * 50)
        
        agents = self.config.get('agents', {})
        for name, config in agents.items():
            print(f"\n📋 {name}")
            print(f"   描述: {config['description']}")
            print(f"   类型: {config['subagent_type']}")
            
        print(f"\n🎯 可用的协作场景:")
        print("=" * 30)
        
        scenarios = self.config.get('scenarios', {})
        for name, config in scenarios.items():
            print(f"\n🔄 {name}")
            print(f"   描述: {config['description']}")
            print(f"   主要Agent: {config['primary_agent']}")
            print(f"   支持Agents: {', '.join(config.get('supporting_agents', []))}")
    
    def reload_config(self):
        """重新加载配置文件"""
        self.config = self._load_config()
        self.logger.info("配置文件已重新加载")


# 全局管理器实例
_global_manager = None

def get_manager() -> SubAgentManager:
    """获取全局SubAgent管理器实例"""
    global _global_manager
    if _global_manager is None:
        _global_manager = SubAgentManager()
    return _global_manager


def invoke_agent(agent_name: str, task_description: str, **kwargs) -> str:
    """便捷函数：调用指定agent"""
    return get_manager().invoke_agent(agent_name, task_description, **kwargs)


def invoke_scenario(scenario_name: str, task_description: str) -> Dict[str, str]:
    """便捷函数：调用协作场景"""
    return get_manager().invoke_scenario(scenario_name, task_description)


def list_available():
    """便捷函数：列出所有可用的agents和场景"""
    manager = get_manager()
    manager.print_agent_summary()


if __name__ == "__main__":
    # 演示用法
    manager = SubAgentManager()
    
    print("🎯 SubAgent管理器演示")
    print("=" * 40)
    
    # 显示所有可用的agents和场景
    manager.print_agent_summary()
    
    print("\n\n📝 使用示例:")
    print("=" * 20)
    
    print("\n# 1. 调用单个agent")
    print("manager.invoke_agent('factor_expert', '分析ROE因子的有效性')")
    
    print("\n# 2. 调用协作场景")  
    print("manager.invoke_scenario('new_factor_development', '开发ROE增强因子')")
    
    print("\n# 3. 便捷函数")
    print("from subagent_manager import invoke_agent, list_available")
    print("list_available()")
    print("result = invoke_agent('system_architect', '评估系统架构')")