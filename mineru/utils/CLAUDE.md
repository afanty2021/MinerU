[根目录](../../CLAUDE.md) > [mineru](../) > **utils**

# MinerU Utils 工具模块

## 变更记录 (Changelog)
- 2025-11-17 16:46:31 - 初始化Utils模块文档，完成核心工具函数分析

## 模块职责

Utils模块是MinerU的核心工具集，提供配置管理、设备检测、模型下载、图像处理、OCR预处理等基础功能。该模块为整个系统提供底层支撑，确保各个模块能够正常运行和协作。

## 核心功能模块

### 配置管理 (`config_reader.py`)
负责系统配置的读取和管理，支持多种配置来源。

#### 核心功能
- **配置文件读取**: 支持`~/mineru.json`配置文件
- **S3存储配置**: 多桶S3存储支持
- **设备自动检测**: CUDA/MPS/NPU/CPU自动选择
- **功能开关控制**: 公式识别、表格识别等开关
- **LaTeX配置**: LaTeX分隔符配置
- **LLM辅助配置**: AI助手功能配置

#### 主要接口
```python
def read_config() -> dict
def get_device() -> str
def get_s3_config(bucket_name: str) -> tuple
def get_formula_enable(formula_enable: bool) -> bool
def get_table_enable(table_enable: bool) -> bool
```

#### 环境变量支持
- `MINERU_TOOLS_CONFIG_JSON`: 配置文件路径
- `MINERU_DEVICE_MODE`: 强制指定设备类型
- `MINERU_FORMULA_ENABLE`: 公式识别开关
- `MINERU_TABLE_ENABLE`: 表格识别开关
- `MINERU_VIRTUAL_VRAM_SIZE`: 虚拟显存大小

### 模型下载管理 (`models_download_utils.py`)
提供统一的模型下载和管理接口。

#### 支持的模型源
- **Hugging Face**: 默认模型仓库
- **ModelScope**: 国产模型仓库
- **Local**: 本地模型路径

#### 仓库模式
- **pipeline**: 传统流水线模型
- **vlm**: 视觉语言模型

#### 核心接口
```python
def auto_download_and_get_model_root_path(
    relative_path: str,
    repo_mode='pipeline'
) -> str
```

### 图像处理工具 (`model_utils.py`)
提供文档图像处理和区域分析功能。

#### 核心功能
- **图像裁剪**: 支持白背景填充的智能裁剪
- **区域合并**: 高IoU表格合并算法
- **重叠处理**: 复杂重叠区域处理逻辑
- **内存管理**: GPU/CPU显存优化管理

#### 关键算法
- **IoU计算**: 精确的交并比计算
- **嵌套检测**: 嵌套表格识别和过滤
- **重叠合并**: 低置信度区块智能合并
- **内存清理**: 多设备内存优化

#### 主要接口
```python
def crop_img(input_res, input_img, crop_paste_x=0, crop_paste_y=0)
def merge_high_iou_tables(table_res_list, layout_res, table_indices, iou_threshold=0.7)
def filter_nested_tables(table_res_list, overlap_threshold=0.8, area_threshold=0.8)
def clean_memory(device='cuda')
def get_vram(device)
```

### PDF处理工具 (`pdf_reader.py`)
提供PDF到图像的转换功能。

#### 核心功能
- **页面渲染**: 高质量PDF页面转图像
- **DPI控制**: 可调节渲染精度
- **尺寸限制**: 防止内存溢出的尺寸控制
- **格式转换**: 图像转字节和Base64编码

#### 主要接口
```python
def page_to_image(page: PdfPage, dpi: int = 200, max_width_or_height: int = 3500)
def image_to_bytes(image: Image.Image, image_format: str = "JPEG")
def image_to_b64str(image: Image.Image, image_format: str = "JPEG")
```

### OCR工具 (`ocr_utils.py`)
提供OCR预处理和文本处理功能。

#### 核心功能
- **文本块合并**: 智能文本行合并
- **重叠检测**: Y轴重叠比例计算
- **置信度管理**: OCR结果置信度过滤

#### 置信度类
```python
class OcrConfidence:
    min_confidence = 0.5
    min_width = 3
```

#### 主要接口
```python
def merge_spans_to_line(spans, threshold=0.6)
def _is_overlaps_y_exceeds_threshold(bbox1, bbox2, overlap_ratio_threshold=0.8)
```

### CLI参数解析 (`cli_parser.py`)
处理命令行额外参数的解析。

#### 核心功能
- **参数类型推断**: 自动识别int/float/bool/string类型
- **布尔标志处理**: 支持--flag形式的无值参数
- **格式转换**: 连字符转下划线命名转换

#### 主要接口
```python
def arg_parse(ctx: 'click.Context') -> dict
```

## 其他工具模块

### 区块处理
- **block_pre_proc.py**: 区块预处理
- **block_sort.py**: 区块排序
- **span_block_fix.py**: Span区块修复
- **span_pre_proc.py**: Span预处理

### 图像和绘制
- **boxbase.py**: 边界框基础操作
- **cut_image.py**: 图像裁剪工具
- **draw_bbox.py**: 边界框绘制

### 文本和格式
- **format_utils.py**: 格式化工具
- **language.py**: 语言检测和处理
- **guess_suffix_or_lang.py**: 文件类型推断

### 哈希和安全
- **hash_utils.py**: 哈希计算工具
- **pdf_classify.py**: PDF分类

### 异步和AI
- **run_async.py**: 异步运行工具
- **llm_aided.py**: LLM辅助功能
- **magic_model_utils.py**: 魔法模型工具

## 系统特性

### 设备兼容性
- **CUDA**: NVIDIA GPU支持
- **MPS**: Apple Metal Performance Shaders
- **NPU**: 华为昇腾NPU支持
- **CPU**: 通用CPU回退

### 性能优化
- **内存管理**: 智能显存和内存清理
- **批处理**: 支持批量图像处理
- **缓存机制**: 模型和配置缓存
- **异步支持**: 异步IO处理

### 错误处理
- **容错机制**: 模型加载失败回退
- **日志记录**: 详细的操作日志
- **异常捕获**: 优雅的错误处理

## 配置示例

### mineru.json配置文件
```json
{
  "bucket_info": {
    "[default]": ["access_key", "secret_key", "endpoint"],
    "my-bucket": ["ak", "sk", "endpoint"]
  },
  "latex-delimiter-config": {
    "inline": ["$", "$"],
    "display": ["$$", "$$"]
  },
  "llm-aided-config": {
    "enabled": true,
    "model": "gpt-4"
  },
  "models-dir": {
    "pipeline": "/path/to/pipeline/models",
    "vlm": "/path/to/vlm/models"
  }
}
```

## 使用指南

### 设备检测
```python
from mineru.utils.config_reader import get_device

device = get_device()
# 返回: "cuda" | "mps" | "npu" | "cpu"
```

### 模型下载
```python
from mineru.utils.models_download_utils import auto_download_and_get_model_root_path

# 下载布局检测模型
layout_path = auto_download_and_get_model_root_path("models/Layout/YOLO/doc_layout_yolo_pt.onnx")

# 下载VLM模型
vlm_path = auto_download_and_get_model_root_path("/", repo_mode='vlm')
```

### 图像处理
```python
from mineru.utils.model_utils import crop_img, clean_memory

# 裁剪图像区域
cropped_img, crop_info = crop_img(region_info, input_image)

# 清理GPU内存
clean_memory('cuda')
```

## 性能特点

### 内存管理
- **智能清理**: 根据显存大小自动清理
- **多设备支持**: CUDA/MPS/NPU统一接口
- **阈值控制**: 可配置的内存阈值

### 处理效率
- **批量处理**: 支持多图像并行处理
- **算法优化**: 高效的IoU计算和合并算法
- **缓存机制**: 避免重复计算和下载

## 常见问题 (FAQ)

### Q: 如何切换到CPU模式？
A: 设置环境变量`MINERU_DEVICE_MODE=cpu`或在配置文件中指定。

### Q: 模型下载失败怎么办？
A: 可以尝试切换模型源`MINERU_MODEL_SOURCE=modelscope`或使用本地模型。

### Q: 内存不足如何处理？
A: 系统会自动清理内存，也可以设置`MINERU_VIRTUAL_VRAM_SIZE`限制使用。

### Q: 如何添加新的存储后端？
A: 在config_reader.py中扩展get_s3_config函数支持新的存储类型。

## 相关文件清单

### 核心工具
- config_reader.py - 配置管理
- model_utils.py - 图像处理和内存管理
- models_download_utils.py - 模型下载
- pdf_reader.py - PDF图像转换
- ocr_utils.py - OCR预处理

### CLI和参数
- cli_parser.py - 命令行解析

### 区块处理
- block_pre_proc.py - 区块预处理
- block_sort.py - 区块排序
- span_block_fix.py - Span修复
- span_pre_proc.py - Span预处理

### 图像处理
- boxbase.py - 边界框操作
- cut_image.py - 图像裁剪
- draw_bbox.py - 边界框绘制

### 文本处理
- format_utils.py - 格式化
- language.py - 语言处理
- guess_suffix_or_lang.py - 类型推断

### 工具函数
- hash_utils.py - 哈希工具
- pdf_classify.py - PDF分类
- run_async.py - 异步工具
- llm_aided.py - LLM辅助
- magic_model_utils.py - 模型工具
- check_mac_env.py - macOS环境检查

## 变更记录 (Changelog)
- 2025-11-17 16:46:31 - 初始化Utils模块文档，完成核心工具函数分析