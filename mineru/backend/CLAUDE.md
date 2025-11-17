[根目录](../../CLAUDE.md) > [mineru](../) > **backend**

# MinerU Backend 模块

## 变更记录 (Changelog)
- 2025-11-17 16:36:36 - 初始化Backend模块文档

## 模块职责

Backend模块是MinerU的核心解析引擎，提供两种不同的文档解析方案：
- **Pipeline后端**: 传统的分步处理流水线，快速稳定
- **VLM后端**: 基于视觉语言模型的多模态理解，高精度

## 子模块概览

### Pipeline后端 (`pipeline/`)
传统流水线处理方式，通过多个独立模型串联处理文档：
1. 布局检测 → OCR识别 → 公式解析 → 表格识别 → 阅读排序

### VLM后端 (`vlm/`)
视觉语言模型处理方式，使用统一的VLM模型进行端到端理解：
1. VLM模型推理 → 结果解析 → 内容生成

## 关键接口

### 统一处理接口
```python
# Pipeline接口
def doc_analyze(pdf_bytes, **kwargs):
    """Pipeline后端文档分析"""

# VLM接口
def doc_analyze(pdf_bytes, **kwargs):
    """VLM后端文档分析"""

# 异步接口
async def aio_doc_analyze(pdf_bytes, **kwargs):
    """异步文档分析"""
```

### 输出格式
两种后端都输出标准化的`middle_json`格式：
- 布局信息
- 文本内容
- 公式内容
- 表格结构
- 阅读顺序

## Pipeline后端详解

### 核心组件
- **pipeline_analyze.py** - 主要分析流程
- **pipeline_magic_model.py** - 模型管理
- **model_init.py** - 模型初始化
- **batch_analyze.py** - 批量处理

### 处理流程
1. **预处理**: PDF页面处理和图像准备
2. **布局检测**: 使用YOLO模型检测文档布局
3. **OCR识别**: 多语言文字识别
4. **公式解析**: 数学公式识别和LaTeX转换
5. **表格识别**: 表格结构识别和HTML转换
6. **阅读排序**: 基于空间位置的内容排序

### 特点
- 速度快，适合大批量处理
- 结果稳定，无幻觉问题
- 内存占用相对较低
- 支持纯CPU模式

## VLM后端详解

### 核心组件
- **vlm_analyze.py** - VLM分析主流程
- **vlm_magic_model.py** - VLM模型管理
- **model_output_to_middle_json.py** - 结果转换
- **vlm_middle_json_mkcontent.py** - 内容生成

### 支持的推理引擎
- **transformers**: 标准transformers推理，兼容性好
- **vllm-engine**: VLLM加速推理，高性能
- **mlx-engine**: MLX推理（Apple Silicon），本地优化
- **http-client**: OpenAI兼容API接口

### 模型版本
- **MinerU2.0-2505-0.9B**: 早期版本（已废弃）
- **MinerU2.5-2509-1.2B**: 当前最新版本，SOTA性能

### 特点
- 精度高，支持复杂布局
- 统一模型，简化流程
- 支持跨页表格合并
- 更好的公式识别效果

## 性能对比

| 特性 | Pipeline | VLM |
|------|----------|-----|
| 精度 | 82+ (OmniDocBench) | 90+ (OmniDocBench) |
| 速度 | 快 | 中等 |
| 内存要求 | 较低 | 较高 |
| GPU要求 | 可选 | 推荐 |
| 幻觉风险 | 无 | 极低 |
| 复杂布局 | 有限 | 优秀 |
| 跨页支持 | 部分支持 | 完整支持 |

## 配置选项

### 共同参数
```python
{
    "parse_method": "auto",  # auto/txt/ocr
    "language": "auto",      # 语言设置
    "formula_enable": True,  # 公式解析
    "table_enable": True,    # 表格解析
    "end_page_id": None      # 页面范围
}
```

### Pipeline特定参数
```python
{
    "layout_model": "doclayout_yolo",
    "ocr_model": "ppocr_v5",
    "formula_model": "unimernet",
    "table_model": "rapid_table"
}
```

### VLM特定参数
```python
{
    "vlm_backend": "vlm-transformers",  # 推理引擎
    "model_path": "opendatalab/MinerU2.5-2509-1.2B",
    "server_url": "",  # HTTP客户端URL
    "batch_size": 1
}
```

## 测试与质量

### 测试覆盖
- 单元测试：各组件独立测试
- 集成测试：端到端流程测试
- 性能测试：速度和内存测试
- 精度测试：标准基准测试

### 质量保证
- 输入验证
- 错误处理
- 资源管理
- 日志记录

## 使用建议

### 选择Pipeline后端如果：
- 处理大量简单文档
- 对处理速度要求高
- 硬件资源有限
- 需要稳定可预测的结果

### 选择VLM后端如果：
- 处理复杂学术文档
- 对精度要求极高
- 有充足的GPU资源
- 需要处理跨页内容

## 常见问题 (FAQ)

### Q: 两种后端的输出格式是否一致？
A: 是的，都输出标准的middle_json格式，但VLM版本支持更多的布局类型。

### Q: 如何在不同后端间切换？
A: 通过`--backend`参数选择，如`--backend pipeline`或`--backend vlm-transformers`。

### Q: VLM后端是否支持混合精度？
A: 支持部分推理引擎的混合精度，可以降低内存使用。

### Q: Pipeline后端的OCR模型可以更换吗？
A: 可以，通过配置文件选择不同的OCR模型，如PP-OCRv5等。

## 相关文件清单

### Pipeline后端
- pipeline_analyze.py - 主分析流程
- pipeline_magic_model.py - 模型管理
- model_init.py - 模型初始化
- batch_analyze.py - 批量处理
- para_split.py - 段落分割
- pipeline_middle_json_mkcontent.py - 内容生成

### VLM后端
- vlm_analyze.py - VLM分析流程
- vlm_magic_model.py - VLM模型管理
- model_output_to_middle_json.py - 结果转换
- vlm_middle_json_mkcontent.py - 内容生成
- utils.py - 工具函数

### 共享文件
- utils.py - 后端通用工具
- __init__.py - 模块初始化

## 变更记录 (Changelog)
- 2025-11-17 16:36:36 - 初始化Backend模块文档