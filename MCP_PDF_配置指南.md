# PDF Tools MCP Server 配置指南

## ✅ 已完成的步骤

1. ✅ 已安装 `pdf-tools-mcp` 包
2. ✅ 已在 Cursor MCP 配置文件中添加 pdf-tools 服务器配置

## 📋 在 Cursor 中配置 MCP 服务器

### 方法1：通过 Cursor 设置界面

1. 打开 Cursor
2. 按 `Ctrl + ,` 打开设置
3. 搜索 "MCP" 或 "Model Context Protocol"
4. 找到 MCP 服务器配置部分
5. 添加新的 MCP 服务器配置

### 方法2：手动编辑配置文件

Cursor 的 MCP 配置文件通常位于：

**Windows:**
```
%APPDATA%\Cursor\User\globalStorage\mcp.json
或
%USERPROFILE%\.cursor\mcp.json
```

**配置示例：**
```json
{
  "mcpServers": {
    "pdf-tools": {
      "command": "python",
      "args": [
        "-m",
        "pdf_tools_mcp"
      ],
      "env": {}
    }
  }
}
```

### 方法3：使用完整路径（推荐）

如果上述方法不工作，使用 Python 的完整路径：

**当前环境配置（已应用）：**
```json
{
  "mcpServers": {
    "pdf-tools": {
      "command": "C:\\Users\\adim\\AppData\\Local\\Programs\\Python\\Python312\\python.exe",
      "args": [
        "-m",
        "pdf_tools_mcp",
        "--workspace_path",
        "E:\\Personal_porject\\20251116_rl_learning\\RL_learning_2025"
      ]
    }
  }
}
```

**配置文件位置：**
- `C:\Users\adim\.cursor\mcp.json`

## 🔍 验证配置

配置完成后：
1. 重启 Cursor
2. 在聊天中尝试使用 MCP 资源
3. 应该能看到 PDF 相关的资源

## 📝 注意事项

- 确保 Python 路径正确
- 如果遇到权限问题，可能需要以管理员身份运行 Cursor
- 某些 MCP 服务器可能需要额外的环境变量配置

## 🚀 临时方案：直接使用 Python 读取 PDF

如果 MCP 配置遇到问题，我已经创建了一个测试脚本 `test_pdf_reader.py`，可以直接读取 PDF 文件内容。

