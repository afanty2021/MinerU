[根目录](../../../CLAUDE.md) > [projects](../) > **mcp**

# MinerU MCP 项目

## 变更记录 (Changelog)
- 2025-11-17 16:36:36 - 初始化MCP项目文档

## 项目职责

MCP (Model Context Protocol) 项目为MinerU提供标准化的模型上下文协议支持，使MinerU能够作为AI助手的文档处理插件。该项目实现了：

- MCP协议服务器
- 文档解析服务接口
- AI助手集成支持
- 标准化的工具和资源定义

## 核心组件

### Server Implementation (`src/mineru/server.py`)
MCP协议服务器的主要实现，负责处理来自AI助手的请求。

#### 核心功能
- 协议处理和路由
- 会话管理
- 错误处理
- 日志记录

### API Interface (`src/mineru/api.py`)
MinerU功能的API封装，提供标准化的调用接口。

#### 主要接口
```python
async def parse_document(
    file_path: str,
    options: Dict[str, Any]
) -> ParseResult:
    """文档解析接口"""

async def extract_content(
    document: Document,
    content_type: str
) -> str:
    """内容提取接口"""
```

### Configuration (`src/mineru/config.py`)
MCP服务器配置管理。

#### 配置项
- 服务器设置（端口、主机）
- MinerU模型配置
- 日志配置
- 安全设置

### Language Support (`src/mineru/language.py`)
多语言支持和本地化功能。

#### 支持语言
- 中文（简体/繁体）
- 英文
- 日文
- 其他OCR支持语言

### Examples (`src/mineru/examples.py`)
使用示例和测试用例。

#### 示例场景
- 基本文档解析
- 批量处理
- 自定义配置解析
- 错误处理示例

## MCP协议支持

### 标准工具定义
```python
# 文档解析工具
{
    "name": "parse_document",
    "description": "Parse PDF document to structured content",
    "parameters": {
        "file_path": {"type": "string", "description": "Path to PDF file"},
        "output_format": {"type": "string", "enum": ["markdown", "json"]},
        "language": {"type": "string", "default": "auto"}
    }
}
```

### 资源定义
```python
# 文档资源
{
    "uri": "document://{document_id}",
    "name": "Document Content",
    "description": "Parsed document content",
    "mimeType": "application/json"
}
```

## 部署选项

### Docker部署
```bash
# 构建镜像
docker build -t mineru-mcp .

# 运行容器
docker run -p 8080:8080 mineru-mcp
```

### Docker Compose部署
```yaml
version: '3.8'
services:
  mineru-mcp:
    build: .
    ports:
      - "8080:8080"
    environment:
      - LOG_LEVEL=INFO
      - MINERU_MODEL_PATH=/models
    volumes:
      - ./models:/models
```

## 集成示例

### Claude Desktop集成
```json
{
  "mcpServers": {
    "mineru": {
      "command": "docker",
      "args": ["run", "-i", "--rm", "mineru-mcp"],
      "env": {
        "MINERU_SERVER_URL": "http://localhost:8080"
      }
    }
  }
}
```

### 自定义AI助手集成
```python
import mcp

# 连接到MCP服务器
client = mcp.Client("stdio", command="python", args=["server.py"])

# 解析文档
result = await client.call_tool("parse_document", {
    "file_path": "/path/to/document.pdf",
    "output_format": "markdown"
})
```

## 配置参数

### 服务器配置
```python
{
    "host": "0.0.0.0",
    "port": 8080,
    "workers": 4,
    "timeout": 30,
    "max_request_size": "100MB"
}
```

### MinerU配置
```python
{
    "backend": "vlm-transformers",
    "language": "auto",
    "formula_enable": True,
    "table_enable": True,
    "model_path": "/models/mineru"
}
```

### 安全配置
```python
{
    "allowed_paths": ["/data/documents"],
    "max_file_size": "50MB",
    "rate_limit": "100/hour",
    "api_key_required": False
}
```

## API接口

### 解析接口
```
POST /parse
Content-Type: application/json

{
    "file_path": "/path/to/document.pdf",
    "options": {
        "output_format": "markdown",
        "language": "auto",
        "backend": "vlm-transformers"
    }
}
```

### 健康检查
```
GET /health

Response:
{
    "status": "healthy",
    "version": "2.6.3",
    "models_loaded": true
}
```

## 错误处理

### 标准错误响应
```python
{
    "error": {
        "code": "INVALID_FILE",
        "message": "File not found or invalid format",
        "details": {
            "file_path": "/path/to/missing.pdf"
        }
    }
}
```

### 错误类型
- `INVALID_FILE` - 文件无效或不存在
- `PARSE_ERROR` - 解析过程出错
- `CONFIG_ERROR` - 配置错误
- `RATE_LIMIT` - 请求频率限制
- `INTERNAL_ERROR` - 内部服务器错误

## 性能优化

### 缓存策略
- 文档解析结果缓存
- 模型预加载
- 内存缓存
- 磁盘缓存

### 并发处理
- 异步处理架构
- 工作池管理
- 请求队列
- 负载均衡

## 监控和日志

### 日志级别
- `DEBUG` - 详细调试信息
- `INFO` - 一般信息
- `WARNING` - 警告信息
- `ERROR` - 错误信息

### 监控指标
- 请求处理时间
- 内存使用情况
- 错误率统计
- 模型推理时间

## 安全考虑

### 输入验证
- 文件路径验证
- 文件类型检查
- 大小限制
- 恶意文件检测

### 访问控制
- API密钥认证
- 路径访问限制
- 请求频率限制
- 审计日志

## 测试与质量

### 测试覆盖
- 单元测试：各组件功能测试
- 集成测试：MCP协议兼容性测试
- 性能测试：并发处理测试
- 安全测试：输入验证测试

### 质量保证
- 代码覆盖率 > 85%
- 协议兼容性验证
- 性能基准测试
- 安全漏洞扫描

## 常见问题 (FAQ)

### Q: 如何在不同AI助手中使用？
A: 根据AI助手的MCP支持文档进行配置，通常需要添加MCP服务器配置。

### Q: 支持哪些文档格式？
A: 支持MinerU的所有格式：PDF、图像文件等。

### Q: 如何自定义解析选项？
A: 通过API参数或配置文件设置后端、语言、公式/表格解析等选项。

### Q: 处理大文档时如何优化性能？
A: 使用分页处理、启用缓存、调整并发参数等。

## 相关文件清单

### 核心代码
- src/mineru/server.py - MCP服务器实现
- src/mineru/api.py - API接口封装
- src/mineru/config.py - 配置管理
- src/mineru/language.py - 语言支持
- src/mineru/examples.py - 使用示例
- src/mineru/cli.py - 命令行工具

### 配置文件
- pyproject.toml - 项目配置和依赖
- docker-compose.yml - Docker Compose配置
- Dockerfile - Docker镜像构建

### 文档
- README.md - 项目说明
- DOCKER_README.md - Docker部署指南

## 变更记录 (Changelog)
- 2025-11-17 16:36:36 - 初始化MCP项目文档