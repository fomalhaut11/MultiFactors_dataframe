"""
报告基类
定义报告生成的基础框架
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ReportBase(ABC):
    """
    报告基类
    
    定义报告生成的标准接口和通用功能
    """
    
    def __init__(self, 
                 title: str = "Factor Analysis Report",
                 author: str = "MultiFactors System",
                 template: Optional[str] = None):
        """
        初始化报告
        
        Parameters
        ----------
        title : str
            报告标题
        author : str
            报告作者
        template : str, optional
            报告模板路径
        """
        self.title = title
        self.author = author
        self.template = template
        self.creation_time = datetime.now()
        self.sections = []
        self.data = {}
        self.figures = []
        
    @abstractmethod
    def generate(self, analysis_results: Dict[str, Any]) -> str:
        """
        生成报告
        
        Parameters
        ----------
        analysis_results : Dict[str, Any]
            分析结果
            
        Returns
        -------
        str
            报告内容或路径
        """
        pass
    
    @abstractmethod
    def add_section(self, title: str, content: Any):
        """
        添加报告章节
        
        Parameters
        ----------
        title : str
            章节标题
        content : Any
            章节内容
        """
        pass
    
    def add_summary(self, summary_data: Dict[str, Any]):
        """
        添加摘要
        
        Parameters
        ----------
        summary_data : Dict[str, Any]
            摘要数据
        """
        summary_content = self._format_summary(summary_data)
        self.add_section("Executive Summary", summary_content)
    
    def add_table(self, title: str, df: pd.DataFrame, description: str = ""):
        """
        添加表格
        
        Parameters
        ----------
        title : str
            表格标题
        df : pd.DataFrame
            数据表
        description : str
            表格说明
        """
        table_content = {
            'type': 'table',
            'title': title,
            'data': df,
            'description': description
        }
        self.add_section(title, table_content)
    
    def add_figure(self, title: str, figure_obj: Any, description: str = ""):
        """
        添加图表
        
        Parameters
        ----------
        title : str
            图表标题
        figure_obj : Any
            图表对象（matplotlib figure或plotly figure）
        description : str
            图表说明
        """
        figure_content = {
            'type': 'figure',
            'title': title,
            'figure': figure_obj,
            'description': description
        }
        self.figures.append(figure_content)
        self.add_section(title, figure_content)
    
    def _format_summary(self, summary_data: Dict[str, Any]) -> str:
        """
        格式化摘要
        
        Parameters
        ----------
        summary_data : Dict[str, Any]
            摘要数据
            
        Returns
        -------
        str
            格式化的摘要文本
        """
        lines = []
        for key, value in summary_data.items():
            if isinstance(value, (int, float)):
                lines.append(f"• {key}: {value:.4f}")
            else:
                lines.append(f"• {key}: {value}")
        return "\n".join(lines)
    
    def save(self, output_path: str):
        """
        保存报告
        
        Parameters
        ----------
        output_path : str
            输出路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 根据扩展名选择保存格式
        if output_path.suffix == '.html':
            self.save_html(str(output_path))
        elif output_path.suffix == '.pdf':
            self.save_pdf(str(output_path))
        elif output_path.suffix == '.md':
            self.save_markdown(str(output_path))
        else:
            # 默认保存为文本
            self.save_text(str(output_path))
        
        logger.info(f"Report saved to {output_path}")
    
    def save_text(self, path: str):
        """保存为文本格式"""
        content = self.generate_text()
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def save_markdown(self, path: str):
        """保存为Markdown格式"""
        content = self.generate_markdown()
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def save_html(self, path: str):
        """保存为HTML格式"""
        content = self.generate_html()
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def save_pdf(self, path: str):
        """保存为PDF格式（需要额外依赖）"""
        # 这里可以使用reportlab或weasyprint等库
        raise NotImplementedError("PDF generation not implemented yet")
    
    def generate_text(self) -> str:
        """生成文本格式报告"""
        lines = []
        lines.append("=" * 80)
        lines.append(f"{self.title}")
        lines.append("=" * 80)
        lines.append(f"Author: {self.author}")
        lines.append(f"Generated: {self.creation_time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("-" * 80)
        
        for section in self.sections:
            lines.append(f"\n## {section['title']}")
            lines.append("-" * 40)
            
            content = section['content']
            if isinstance(content, dict):
                if content['type'] == 'table':
                    lines.append(str(content['data']))
                elif content['type'] == 'figure':
                    lines.append("[Figure: " + content['title'] + "]")
            else:
                lines.append(str(content))
        
        return "\n".join(lines)
    
    def generate_markdown(self) -> str:
        """生成Markdown格式报告"""
        lines = []
        lines.append(f"# {self.title}")
        lines.append("")
        lines.append(f"**Author**: {self.author}  ")
        lines.append(f"**Generated**: {self.creation_time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        for section in self.sections:
            lines.append(f"## {section['title']}")
            lines.append("")
            
            content = section['content']
            if isinstance(content, dict):
                if content['type'] == 'table':
                    # 转换DataFrame为Markdown表格
                    lines.append(content['data'].to_markdown())
                elif content['type'] == 'figure':
                    lines.append(f"![{content['title']}](figure_{content['title']}.png)")
                lines.append("")
                if content.get('description'):
                    lines.append(f"*{content['description']}*")
            else:
                lines.append(str(content))
            lines.append("")
        
        return "\n".join(lines)
    
    def generate_html(self) -> str:
        """生成HTML格式报告"""
        html_parts = []
        
        # HTML头部
        html_parts.append("""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .metadata {{
            color: #666;
            font-size: 0.9em;
        }}
        .figure {{
            text-align: center;
            margin: 20px 0;
        }}
        .description {{
            font-style: italic;
            color: #666;
            margin: 10px 0;
        }}
    </style>
</head>
<body>
""".format(title=self.title))
        
        # 标题和元数据
        html_parts.append(f"<h1>{self.title}</h1>")
        html_parts.append(f'<div class="metadata">')
        html_parts.append(f"<p><strong>Author:</strong> {self.author}</p>")
        html_parts.append(f"<p><strong>Generated:</strong> {self.creation_time.strftime('%Y-%m-%d %H:%M:%S')}</p>")
        html_parts.append("</div>")
        html_parts.append("<hr>")
        
        # 内容章节
        for section in self.sections:
            html_parts.append(f"<h2>{section['title']}</h2>")
            
            content = section['content']
            if isinstance(content, dict):
                if content['type'] == 'table':
                    # 转换DataFrame为HTML表格
                    html_parts.append(content['data'].to_html(classes='dataframe'))
                elif content['type'] == 'figure':
                    html_parts.append(f'<div class="figure">')
                    html_parts.append(f'<img src="figure_{content["title"]}.png" alt="{content["title"]}">')
                    html_parts.append('</div>')
                
                if content.get('description'):
                    html_parts.append(f'<p class="description">{content["description"]}</p>')
            else:
                # 将文本转换为HTML段落
                lines = str(content).split('\n')
                for line in lines:
                    if line.strip():
                        html_parts.append(f"<p>{line}</p>")
        
        # HTML尾部
        html_parts.append("""
</body>
</html>
""")
        
        return "\n".join(html_parts)


class InteractiveReportMixin:
    """
    交互式报告混入类
    提供交互式图表和控件功能
    """
    
    def add_interactive_chart(self, title: str, data: pd.DataFrame, chart_type: str = 'line'):
        """
        添加交互式图表（使用plotly）
        
        Parameters
        ----------
        title : str
            图表标题
        data : pd.DataFrame
            数据
        chart_type : str
            图表类型（line, bar, scatter, heatmap等）
        """
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            
            if chart_type == 'line':
                fig = px.line(data, title=title)
            elif chart_type == 'bar':
                fig = px.bar(data, title=title)
            elif chart_type == 'scatter':
                fig = px.scatter(data, title=title)
            elif chart_type == 'heatmap':
                fig = px.imshow(data, title=title)
            else:
                fig = go.Figure()
            
            self.add_figure(title, fig)
            
        except ImportError:
            logger.warning("Plotly not installed, falling back to static chart")
            self.add_figure(title, data)
    
    def add_dashboard(self, metrics: Dict[str, Any]):
        """
        添加仪表板视图
        
        Parameters
        ----------
        metrics : Dict[str, Any]
            关键指标
        """
        dashboard_html = self._generate_dashboard_html(metrics)
        self.add_section("Dashboard", {'type': 'html', 'content': dashboard_html})
    
    def _generate_dashboard_html(self, metrics: Dict[str, Any]) -> str:
        """生成仪表板HTML"""
        cards = []
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                card = f"""
                <div class="metric-card">
                    <h3>{key}</h3>
                    <p class="metric-value">{value:.4f}</p>
                </div>
                """
                cards.append(card)
        
        return f"""
        <div class="dashboard">
            <style>
                .dashboard {{
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                .metric-card {{
                    background: white;
                    border-radius: 8px;
                    padding: 20px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    text-align: center;
                }}
                .metric-value {{
                    font-size: 2em;
                    font-weight: bold;
                    color: #4CAF50;
                }}
            </style>
            {''.join(cards)}
        </div>
        """