[根目录](../../CLAUDE.md) > [mineru](../) > **backend**

# MinerU Backend 模块

## 变更记录 (Changelog)
- 2025-12-31 11:45:00 - 新增Hybrid混合后端文档（v2.7.0核心特性，当前默认后端）
- 2025-11-17 16:36:36 - 初始化Backend模块文档

## 模块职责

Backend模块是MinerU的核心解析引擎，提供三种不同的文档解析方案：
- **Pipeline后端**: 传统的分步处理流水线，快速稳定
- **VLM后端**: 基于视觉语言模型的多模态理解，高精度
- **Hybrid后端**: 混合模式（v2.7.0新增），结合Pipeline和VLM优势，**当前为默认后端**

## 子模块概览

### Pipeline后端 (`pipeline/`)
传统流水线处理方式，通过多个独立模型串联处理文档：
1. 布局检测 → OCR识别 → 公式解析 → 表格识别 → 阅读排序

### VLM后端 (`vlm/`)
视觉语言模型处理方式，使用统一的VLM模型进行端到端理解：
1. VLM模型推理 → 结果解析 → 内容生成

### Hybrid后端 (`hybrid/`) - **默认后端**
混合处理方式，结合VLM和Pipeline的优势：
1. 智能文本抽取 → VLM模型理解 → 结果优化 → 内容生成
- 文本PDF直接抽取文本，减少幻觉
- 扫描PDF使用多语言OCR（支持109种语言）
- 保留VLM的高精度布局理解能力

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

| 特性 | Pipeline | Hybrid | VLM |
|------|----------|--------|-----|
| 精度 | 82+ (OmniDocBench) | 88+ (OmniDocBench) | 90+ (OmniDocBench) |
| 速度 | 快 (<3s/页) | 中等 (<4s/页) | 中等 (<5s/页) |
| 内存要求 | 较低 (8GB+) | 中等 (10GB+) | 较高 (12GB+) |
| GPU要求 | 可选 | 推荐 | 推荐 |
| 幻觉风险 | 无 | 极低 | 极低 |
| 复杂布局 | 有限 | 优秀 | 优秀 |
| 跨页支持 | 部分支持 | 完整支持 | 完整支持 |
| 多语言 | 依赖OCR | 原生支持109种 | 原生支持 |
| 默认后端 | ❌ | ✅ | ❌ |

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

### Hybrid特定参数
```python
{
    "backend": "hybrid-auto-engine",  # 自动引擎选择
    "model_path": "opendatalab/MinerU2.5-2509-1.2B",
    "server_url": "",  # HTTP客户端URL
    "formula_enable": True,  # 独立公式开关
    "batch_ratio": 1,  # 小模型batch倍率
    "force_pipeline_enable": False  # 强制使用Pipeline文本提取
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

### 选择Hybrid后端如果（推荐默认）：
- 需要平衡精度和速度
- 处理包含文本和扫描页面的混合PDF
- 需要多语言支持（109种语言）
- 希望减少解析幻觉
- **新用户推荐使用**

### 选择Pipeline后端如果：
- 处理大量简单文档
- 对处理速度要求极高
- 硬件资源有限
- 需要稳定可预测的结果

### 选择VLM后端如果：
- 处理极其复杂的学术文档
- 对精度要求达到极致
- 有充足的GPU资源
- 需要处理跨页内容

## 常见问题 (FAQ)

### Q: 三种后端的输出格式是否一致？
A: 是的，都输出标准的middle_json格式。VLM和Hybrid版本支持更多的布局类型。

### Q: 如何在不同后端间切换？
A: 通过`--backend`参数选择，如`--backend pipeline`、`--backend hybrid-auto-engine`（默认）或`--backend vlm-auto-engine`。

### Q: 为什么默认后端是Hybrid？
A: Hybrid结合了Pipeline的精确文本提取和VLM的高精度理解，在大多数场景下提供最佳的平衡体验。

### Q: 什么时候应该使用auto-engine后缀？
A: 建议始终使用auto-engine后缀（如hybrid-auto-engine），让系统根据当前环境自动选择最佳推理引擎。

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

### Hybrid后端（v2.7.0新增）
- hybrid_analyze.py - Hybrid分析流程
- hybrid_magic_model.py - Hybrid模型管理
- hybrid_model_output_to_middle_json.py - 结果转换

### 共享文件
- utils.py - 后端通用工具
- __init__.py - 模块初始化

## 变更记录 (Changelog)
- 2025-12-31 11:45:00 - 新增Hybrid混合后端文档（v2.7.0核心特性，当前默认后端）
- 2025-11-17 16:36:36 - 初始化Backend模块文档