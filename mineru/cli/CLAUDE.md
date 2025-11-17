[根目录](../../CLAUDE.md) > [mineru](../) > **cli**

# MinerU CLI 模块

## 变更记录 (Changelog)
- 2025-11-17 16:36:36 - 初始化CLI模块文档

## 模块职责

CLI模块是MinerU的主要用户接口层，提供多种交互方式：
- 命令行工具（CLI）
- REST API服务
- Gradio Web界面
- 模型下载管理
- VLLM服务器支持

## 入口与启动

### 主要入口文件
- **client.py** - 主命令行工具入口，使用Click框架
- **fast_api.py** - FastAPI REST服务实现
- **gradio_app.py** - Gradio Web用户界面
- **models_download.py** - 模型下载管理工具
- **vlm_vllm_server.py** - VLLM加速服务器

### 启动命令
```bash
# 主CLI工具
mineru -p input_path -o output_path [options]

# API服务
mineru-api --host 0.0.0.0 --port 8888

# Gradio Web界面
mineru-gradio --host 0.0.0.0 --port 7860

# 模型下载
mineru-models-download --model-type all

# VLLM服务器
mineru-vllm-server --model-path /path/to/model
```

## 对外接口

### CLI参数接口
```python
@click.option('-p', '--path', 'input_path', required=True)
@click.option('-o', '--output', 'output_dir', required=True)
@click.option('-m', '--method', type=click.Choice(['auto', 'txt', 'ocr']))
@click.option('--backend', type=click.Choice(backends))
@click.option('--language', default='auto')
@click.option('--formula-enable/--disable-formula', default=True)
@click.option('--table-enable/--disable-table', default=True)
```

### API接口
```python
# 主要API端点
@app.post("/v1/pdf_extract")
async def pdf_extract(
    file: UploadFile = File(...),
    parse_method: str = Form("auto"),
    language: str = Form("auto"),
    formula_enable: bool = Form(True),
    table_enable: bool = Form(True),
    backend: str = Form("pipeline")
)
```

### Gradio界面组件
- PDF文件上传
- 后端选择（pipeline/vlm）
- 语言选择
- 公式/表格解析开关
- 页面范围控制
- 结果下载

## 关键依赖与配置

### 核心依赖
- **click** - 命令行框架
- **fastapi** - REST API框架
- **gradio** - Web界面框架
- **uvicorn** - ASGI服务器
- **loguru** - 日志记录
- **pathlib** - 路径处理

### 配置管理
- 支持环境变量配置
- 支持配置文件读取
- 设备自动检测（GPU/CPU）
- 模型路径自动管理

### 后端支持
```python
backends = ['pipeline', 'vlm-transformers', 'vlm-vllm-engine', 'vlm-http-client']
if is_mac_os_version_supported():
    backends.append("vlm-mlx-engine")
```

## 数据模型

### 输入格式
- PDF文件
- 图像文件（PNG, JPEG, JPG, TIFF等）
- 支持批量处理
- 支持目录扫描

### 输出格式
- Markdown文件
- JSON格式（middle_json, content_list.json）
- 图像文件
- 可视化结果

### 中间格式
- **middle_json** - 标准化中间表示
- **content_list.json** - 按阅读顺序排序的内容

## 测试与质量

### 测试覆盖
- CLI参数解析测试
- API端点测试
- 文件处理测试
- 错误处理测试

### 质量保证
- 输入文件验证
- 参数合法性检查
- 错误处理和日志记录
- 资源清理机制

## 常见问题 (FAQ)

### Q: 如何选择合适的后端？
A:
- **pipeline**: 速度快，无幻觉，适合一般文档
- **vlm**: 精度高，支持复杂布局，适合高质量需求

### Q: 支持哪些语言？
A: 支持109种语言的OCR识别，包括中文、英文、日文、阿拉伯文等

### Q: 如何处理大文件？
A: 支持页面范围控制，可以分批处理大文档

### Q: GPU内存不足怎么办？
A: 可以使用CPU模式，或者降低batch size，使用量化模型

## 相关文件清单

### 核心文件
- client.py - 主CLI入口
- fast_api.py - API服务
- gradio_app.py - Web界面
- common.py - 通用处理逻辑
- models_download.py - 模型下载

### 配置文件
- 无特定配置文件，使用代码内配置

### 测试文件
- 无独立测试文件，使用项目的统一测试框架

### 依赖文件
- 在项目根目录的pyproject.toml中定义

## 变更记录 (Changelog)
- 2025-11-17 16:36:36 - 初始化CLI模块文档