[根目录](../../../CLAUDE.md) > [mineru](../../) > [backend](../CLAUDE.md) > **hybrid**

# MinerU Hybrid 后端模块

## 变更记录 (Changelog)
- 2025-12-31 11:45:00 - 初始化Hybrid模块文档

## 模块职责

Hybrid后端是MinerU 2.7.0版本引入的全新混合解析引擎，**当前为默认后端**。它结合了Pipeline和VLM两种后端的优势：

- **基于VLM高精度理解**：保留VLM对复杂布局的强大处理能力
- **融入Pipeline精确提取**：直接从文本PDF抽取文本，减少解析幻觉
- **原生多语言支持**：支持109种语言的文本识别
- **灵活配置选项**：独立行内公式识别开关

## 核心文件

| 文件 | 职责 |
|------|------|
| `hybrid_analyze.py` | Hybrid分析主流程 |
| `hybrid_magic_model.py` | Hybrid模型管理 |
| `hybrid_model_output_to_middle_json.py` | 结果转换为middle_json格式 |

## 技术特点

### 1. 智能文本提取
- **文本PDF场景**：直接抽取文本，原生支持多语言
- **扫描PDF场景**：通过指定OCR语言支持109种语言
- **减少幻觉**：精确的文本提取降低理解错误

### 2. 自动引擎选择
使用`hybrid-auto-engine`后端时，系统会根据当前环境自动选择最佳推理引擎：
- **本地GPU环境**：选择transformers或vllm引擎
- **Apple Silicon**：优先使用mlx引擎
- **远程服务器**：使用http-client模式

### 3. 独立公式开关
- 可单独关闭行内公式识别
- 提升解析结果视觉效果
- 适用于不需要公式的文档类型

## 使用方式

### CLI命令
```bash
# 默认模式（自动引擎选择）
mineru -p input.pdf -o output_dir

# 显式指定Hybrid后端
mineru -p input.pdf -o output_dir -b hybrid-auto-engine

# HTTP客户端模式（连接远程服务器）
mineru -p input.pdf -o output_dir -b hybrid-http-client -u http://server:30000
```

### 环境变量
```bash
# 控制小模型batch倍率（常用于HTTP客户端模式）
export MINERU_HYBRID_BATCH_RATIO=1

# 强制使用Pipeline进行文本提取（减少极端情况下的幻觉）
export MINERU_HYBRID_FORCE_PIPELINE_ENABLE=true
```

## 性能特征

| 指标 | 值 | 说明 |
|------|-----|------|
| 单页处理时间 | <4秒 | GPU模式 |
| GPU显存需求 | 10GB+ | 推荐配置 |
| 支持语言数 | 109种 | 扫描PDF场景 |
| 精度 | 高 | 接近VLM水平 |

## 与其他后端对比

| 特性 | Pipeline | Hybrid | VLM |
|------|----------|--------|-----|
| 精度 | 82+ | 88+ | 90+ |
| 速度 | 快 (<3s) | 中等 (<4s) | 中等 (<5s) |
| 幻觉风险 | 无 | 极低 | 极低 |
| 多语言 | 依赖OCR | 原生支持 | 原生支持 |
| 文本提取 | 直接抽取 | 直接抽取 | VLM理解 |
| 默认后端 | ❌ | ✅ | ❌ |

## 最佳实践

### 选择Hybrid后端如果：
- 需要平衡精度和速度
- 处理包含文本和扫描页面的混合PDF
- 需要多语言支持
- 希望减少解析幻觉
- **新用户推荐使用（默认后端）**

### 选择Pipeline后端如果：
- 处理大量简单文档
- 对处理速度要求极高
- 硬件资源有限
- 需要完全稳定可预测的结果

### 选择VLM后端如果：
- 处理极其复杂的学术文档
- 对精度要求达到极致
- 有充足的GPU资源
- 需要处理跨页内容

## 扩展阅读

- [Pipeline后端文档](../pipeline/CLAUDE.md)
- [VLM后端文档](../vlm/CLAUDE.md)
- [后端模块总览](../CLAUDE.md)
- [项目主文档](../../../CLAUDE.md)
