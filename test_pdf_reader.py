"""
临时测试脚本：直接使用 PyPDF2 读取 PDF 文件
用于在 MCP 配置完成前测试 PDF 读取功能
"""

import sys
from pathlib import Path

try:
    from PyPDF2 import PdfReader
except ImportError:
    print("正在安装 PyPDF2...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PyPDF2"])
    from PyPDF2 import PdfReader


def read_pdf(pdf_path: str, max_pages: int = None):
    """
    读取 PDF 文件内容
    
    Args:
        pdf_path: PDF 文件路径
        max_pages: 最大读取页数（None 表示读取全部）
    
    Returns:
        包含每页文本的列表
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")
    
    reader = PdfReader(str(pdf_path))
    pages_text = []
    
    total_pages = len(reader.pages)
    pages_to_read = min(total_pages, max_pages) if max_pages else total_pages
    
    print(f"PDF 文件: {pdf_path.name}")
    print(f"总页数: {total_pages}")
    print(f"读取页数: {pages_to_read}")
    print("-" * 60)
    
    for i in range(pages_to_read):
        page = reader.pages[i]
        text = page.extract_text()
        pages_text.append(text)
        print(f"\n=== 第 {i+1} 页 ===\n{text[:500]}...")  # 只显示前500字符
    
    return pages_text


def search_in_pdf(pdf_path: str, search_term: str):
    """
    在 PDF 中搜索关键词
    
    Args:
        pdf_path: PDF 文件路径
        search_term: 搜索关键词
    
    Returns:
        包含关键词的页面编号列表
    """
    pages_text = read_pdf(pdf_path)
    matching_pages = []
    
    for i, text in enumerate(pages_text):
        if search_term.lower() in text.lower():
            matching_pages.append(i + 1)
    
    return matching_pages


if __name__ == "__main__":
    # 测试：读取你的 PDF 文件
    pdf_file = "Book-Mathematical-Foundation-of-Reinforcement-Learning/3 - Chapter 2 State Values and Bellman Equation.pdf"
    
    # 搜索矩阵形式相关内容
    print("搜索关键词: 'matrix form', 'P_π', 'r_π', 'v_π = (I - γP_π)^(-1)'")
    print("=" * 60)
    
    search_terms = ["matrix form", "P_π", "r_π", "v_π", "I - γP", "direct solution", "linear system"]
    
    for term in search_terms:
        pages = search_in_pdf(pdf_file, term)
        if pages:
            print(f"\n找到 '{term}' 在以下页面: {pages}")
    
    # 读取前10页内容（用于查找矩阵形式部分）
    print("\n" + "=" * 60)
    print("读取前10页内容（查找矩阵形式部分）:")
    print("=" * 60)
    pages_text = read_pdf(pdf_file, max_pages=10)

