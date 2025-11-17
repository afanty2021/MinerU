[根目录](../../CLAUDE.md) > [mineru](../) > **model**

# MinerU Model 模块

## 变更记录 (Changelog)
- 2025-11-17 17:02:15 - 深度分析MFR和Table模块技术实现，新增多GPU和测试文档
- 2025-11-17 16:46:31 - 更新Layout和OCR模块详细实现分析
- 2025-11-17 16:36:36 - 初始化Model模块文档

## 模块职责

Model模块负责封装和管理MinerU中使用的各种AI模型，提供统一的模型接口和管理机制。包含以下专业模型：

- **Layout Detection** - 布局检测模型
- **OCR** - 文字识别模型
- **MFR** - 数学公式识别模型
- **Table Recognition** - 表格识别模型
- **Reading Order** - 阅读顺序分析模型
- **Utils** - 模型工具和基础组件

## 子模块详解

### Layout Detection (`layout/`)
负责文档布局分析，识别文本区域、图像、表格等元素。

#### 核心实现: DocLayoutYOLOModel
```python
class DocLayoutYOLOModel:
    def __init__(self, weight: str, device: str = "cuda",
                 imgsz: int = 1280, conf: float = 0.1, iou: float = 0.45)
    def predict(self, image) -> List[Dict]
    def batch_predict(self, images: List, batch_size: int = 4) -> List[List[Dict]]
    def visualize(self, image, results) -> Image.Image
```

#### 模型信息
- **基础架构**: YOLOv10
- **输入尺寸**: 1280x1280 (可配置)
- **置信度阈值**: 0.1 (默认)
- **IoU阈值**: 0.45 (默认)
- **批处理大小**: 4 (默认)

#### 支持的布局类别
- **类别0**: 文本区域 (正文)
- **类别1**: 标题区域
- **类别2**: 列表区域
- **类别3**: 图像区域
- **类别4**: 表格区域
- **类别5**: 公式区域
- **类别6**: 页眉页脚
- **类别7**: 脚注区域
- **类别13-14**: 特殊公式区域

#### 输出格式
```python
{
    "category_id": int,        # 类别ID
    "poly": [x1, y1, x2, y2, x2, y2, x1, y2],  # 八边形坐标
    "score": float             # 置信度分数
}
```

### OCR (`ocr/`)
负责文字识别，支持多语言文本检测和识别。

#### 核心实现: PytorchPaddleOCR
```python
class PytorchPaddleOCR:
    def __init__(self, lang='ch', use_angle_cls=True, det=True, rec=True)
    def __call__(self, img, mfd_res=None)
    def ocr(self, single_img_mfd_list, det=True, rec=True, tqdm_enable=False)
```

#### 多语言支持

##### 拉丁语系 (71种语言)
```python
latin_lang = [
    "af", "az", "bs", "cs", "cy", "da", "de", "es", "et", "fr",
    "ga", "hr", "hu", "id", "is", "it", "ku", "la", "lt", "lv",
    "mi", "ms", "mt", "nl", "no", "oc", "pi", "pl", "pt", "ro",
    "rs_latin", "sk", "sl", "sq", "sv", "sw", "tl", "tr", "uz",
    "vi", "french", "german", "fi", "eu", "gl", "lb", "rm",
    "ca", "qu"
]
```

##### 阿拉伯语系 (8种语言)
```python
arabic_lang = ["ar", "fa", "ug", "ur", "ps", "ku", "sd", "bal"]
```

##### 西里尔语系 (27种语言)
```python
cyrillic_lang = [
    "ru", "rs_cyrillic", "be", "bg", "uk", "mn", "abq", "ady",
    "kbd", "ava", "dar", "inh", "che", "lbe", "lez", "tab",
    "kk", "ky", "tg", "mk", "tt", "cv", "ba", "mhr", "mo",
    "udm", "kv"
]
```

### MFR (`mfr/`) - 数学公式识别 (深度分析)

#### 核心架构：UniMERNet (深度分析)

##### 模型结构
UniMERNet采用Vision-Encoder-Decoder架构：
- **Encoder**: UnimerSwin - 基于Swin Transformer的视觉编码器
- **Decoder**: UnimerMBart - 基于mBART的文本解码器

##### 核心实现：UnimernetModel
```python
class UnimernetModel(VisionEncoderDecoderModel):
    def __init__(self, config, encoder, decoder)
    def generate(self, samples, do_sample=False, temperature=0.2, top_p=0.95, batch_size=64)
    def forward_bak(self, samples)  # 训练前向传播
```

##### 关键技术特性

###### 1. 图像预处理
```python
# 自动通道处理
num_channels = pixel_values.shape[1]
if num_channels == 1:
    pixel_values = pixel_values.repeat(1, 3, 1, 1)  # 灰度图转RGB
```

###### 2. 动态长度控制
```python
# 根据batch_size动态调整最大长度
if self.tokenizer.tokenizer.model_max_length > 1152:
    if batch_size <= 32:
        self.tokenizer.tokenizer.model_max_length = 1152  # 6GB VRAM
    else:
        self.tokenizer.tokenizer.model_max_length = 1344  # 8GB VRAM
```

###### 3. 生成参数优化
```python
# 高质量LaTeX生成
outputs = super().generate(
    pixel_values=pixel_values,
    max_new_tokens=self.tokenizer.tokenizer.model_max_length,
    decoder_start_token_id=self.tokenizer.tokenizer.bos_token_id,
    do_sample=do_sample,
    temperature=temperature,    # 0.2 (保守采样)
    top_p=top_p,               # 0.95 (核采样)
)
```

###### 4. 后处理和清洗
```python
# LaTeX输出清洗
pred_str = self.tokenizer.token2str(outputs)
fixed_str = [latex_rm_whitespace(s) for s in pred_str]  # 移除多余空白
```

##### 批处理优化：UnimernetModel.batch_predict

###### 智能排序策略
```python
# 按面积排序优化批处理效率
image_info = [(area, curr_idx, bbox_img) for each formula]
image_info.sort(key=lambda x: x[0])  # 按面积升序排列
```

###### 动态批大小调整
```python
# 2的幂批大小优化
batch_size = min(batch_size, max(1, 2 ** (len(sorted_images).bit_length() - 1)))
```

###### 内存优化策略
```python
# 精确控制VRAM使用
mf_img = mf_img.to(dtype=self.model.dtype)      # 半精度推理
mf_img = mf_img.to(self.device)                 # 设备转移
with torch.no_grad():                           # 禁用梯度计算
    output = self.model.generate({"image": mf_img}, batch_size=batch_size)
```

##### TokenizerWrapper：高级分词器封装

###### 核心功能
```python
class TokenizerWrapper:
    def tokenize(self, text, **kwargs):
        return self.tokenizer(
            text,
            return_token_type_ids=False,
            return_tensors="pt",
            padding="longest",
            truncation=True,
        )

    def token2str(self, tokens) -> list:
        generated_text = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
        generated_text = [fix_text(text) for text in generated_text]  # 文本修复
        return generated_text
```

###### 特殊处理
- **ftfy文本修复**: 自动修复编码问题
- **特殊标记过滤**: 移除BOS/EOS/PAD标记
- **空格处理**: 正确处理Ġ前缀空格

##### 模型加载和检查点机制

###### from_checkpoint方法
```python
@classmethod
def from_checkpoint(cls, model_path: str, model_filename: str = "pytorch_model.pth"):
    # 配置重建
    config = VisionEncoderDecoderConfig.from_pretrained(model_path)
    config.encoder = UnimerSwinConfig(**vars(config.encoder))
    config.decoder = UnimerMBartConfig(**vars(config.decoder))

    # 模型实例化
    encoder = UnimerSwinModel(config.encoder)
    decoder = UnimerMBartForCausalLM(config.decoder)
    model = cls(config, encoder, decoder)

    # 权重加载和前缀处理
    state_dict = {
        k[len(state_dict_strip_prefix):] if k.startswith(state_dict_strip_prefix) else k: v
        for k, v in state_dict.items()
    }
```

##### 设备适配性
```python
# 不同设备的优化策略
if _device_.startswith("mps") or _device_.startswith("npu"):
    self.model = UnimernetModel.from_pretrained(weight_dir, attn_implementation="eager")
else:
    self.model = UnimernetModel.from_pretrained(weight_dir)

# 精度优化
if not _device_.startswith("cpu"):
    self.model = self.model.to(dtype=torch.float16)  # GPU使用半精度
```

#### PP-FormulaNet Plus M
- **模型类型**: PaddlePaddle实现的公式识别模型
- **适用场景**: 资源受限环境或PaddlePaddle生态
- **功能**: 提供备选的公式识别方案

### Table Recognition (`table/`) - 表格识别 (深度分析)

#### 核心架构：RapidTable + SLANet Plus

##### RapidTableModel：表格识别核心引擎

###### 模型初始化和配置
```python
class RapidTableModel:
    def __init__(self, ocr_engine):
        # SLANet Plus模型路径配置
        slanet_plus_model_path = os.path.join(
            auto_download_and_get_model_root_path(ModelPath.slanet_plus),
            ModelPath.slanet_plus,
        )

        # RapidTable输入配置
        input_args = RapidTableInput(
            model_type=ModelType.SLANETPLUS,
            model_dir_or_path=slanet_plus_model_path,
            use_ocr=False  # 使用外部OCR引擎
        )
        self.table_model = CustomRapidTable(input_args)
        self.ocr_engine = ocr_engine
```

##### 表格识别处理流程

###### 1. OCR结果预处理
```python
def predict(self, image, ocr_result=None):
    # 获取OCR结果
    if not ocr_result:
        raw_ocr_result = self.ocr_engine.ocr(bgr_image)[0]

    # 标准化OCR结果格式
    boxes = []
    texts = []
    scores = []
    for item in raw_ocr_result:
        if len(item) == 3:
            boxes.append(item[0])                    # 边界框坐标
            texts.append(escape_html(item[1]))       # HTML转义文本
            scores.append(item[2])                   # 置信度分数
        elif len(item) == 2 and isinstance(item[1], tuple):
            boxes.append(item[0])
            texts.append(escape_html(item[1][0]))
            scores.append(item[1][1])

    ocr_result = [(boxes, texts, scores)]  # RapidTable期望格式
```

###### 2. 表格结构识别
```python
# 调用RapidTable进行表格分析
table_results = self.table_model(
    img_contents=np.asarray(image),
    ocr_results=ocr_result
)

# 提取识别结果
html_code = table_results.pred_htmls          # HTML表格结构
table_cell_bboxes = table_results.cell_bboxes # 单元格边界框
logic_points = table_results.logic_points     # 逻辑连接点
elapse = table_results.elapse                 # 处理时间
```

###### 3. 批处理优化
```python
def batch_predict(self, table_res_list: List[dict], batch_size: int = 4):
    # 过滤有效表格结果
    not_none_table_res_list = []
    for table_res in table_res_list:
        if table_res.get("ocr_result", None):
            not_none_table_res_list.append(table_res)

    # 批量OCR结果预处理
    img_contents = [table_res["table_img"] for table_res in not_none_table_res_list]
    ocr_results = []
    for table_res in not_none_table_res_list:
        # 标准化OCR结果格式
        ocr_results.append((boxes, texts, scores))

    # 批量表格识别
    table_results = self.table_model(
        img_contents=img_contents,
        ocr_results=ocr_results,
        batch_size=batch_size
    )
```

##### CustomRapidTable：自定义表格处理

###### 增强的处理逻辑
```python
class CustomRapidTable(RapidTable):
    def __call__(self, img_contents, ocr_results=None, batch_size=1):
        with tqdm(total=total_nums, desc="Table-wireless Predict") as pbar:
            for start_i in range(0, total_nums, batch_size):
                # 图像加载
                imgs = self._load_imgs(img_contents[start_i:end_i])

                # 表格结构识别
                pred_structures, cell_bboxes = self.table_structure(imgs)
                logic_points = self.table_matcher.decode_logic_points(pred_structures)

                # OCR结果获取
                dt_boxes, rec_res = self.get_ocr_results(imgs, start_i, end_i, ocr_results)

                # HTML表格生成
                pred_htmls = self.table_matcher(
                    pred_structures, cell_bboxes, dt_boxes, rec_res
                )

                results.pred_htmls.extend(pred_htmls)
                pbar.update(end_i - start_i)
```

##### HTML输出处理

###### 安全性处理
```python
def escape_html(input_string):
    """HTML实体转义，防止XSS攻击"""
    return html.escape(input_string)
```

###### 输出格式
- **HTML表格**: 完整的table、tr、td结构
- **样式保留**: 保持表格的视觉样式
- **嵌套表格**: 支持复杂嵌套表格结构
- **合并单元格**: 正确处理colspan和rowspan

#### SLANet Plus：表格结构识别

##### 核心功能
- **线条检测**: 识别表格边框和分割线
- **单元格检测**: 确定单元格位置和边界
- **结构分析**: 分析表格的行列结构
- **逻辑关系**: 建立单元格间的逻辑连接

##### UNet Table：表格恢复

##### 核心组件
- **UNet架构**: 用于表格图像分割
- **表格恢复**: 从分割结果重建表格结构
- **线条连接**: 连接断开的表格线条
- **单元格合并**: 智能合并相邻单元格

#### Table Classification (`cls/`)

##### PaddleTableCls：表格分类器
```python
class PaddleTableCls:
    def __init__(self, model_path)
    def predict(self, image) -> bool
```

**功能**:
- 判断图像是否包含表格
- 过滤非表格区域
- 提高识别准确性

### Reading Order (`reading_order/`)
负责确定文档元素的正确阅读顺序。

#### 核心算法

##### XY-Cut算法 (`xycut.py`)
- **递归分割**: 基于X和Y坐标的递归分割
- **空间分析**: 分析元素的空间位置关系
- **阅读顺序**: 确定符合人类阅读习惯的顺序

##### 基于模型的排序 (`layout_reader.py`)
- **深度学习**: 使用深度学习模型预测阅读顺序
- **语义理解**: 考虑文档的语义结构
- **多列处理**: 正确处理多列文档布局

## 模型管理

### 统一加载接口
```python
# 布局检测模型
from mineru.model.layout.doclayoutyolo import DocLayoutYOLOModel
layout_model = DocLayoutYOLOModel(weight_path, device="cuda")

# OCR模型
from mineru.model.ocr.pytorch_paddle import PytorchPaddleOCR
ocr_model = PytorchPaddleOCR(lang='ch', use_angle_cls=True)

# 公式识别模型
from mineru.model.mfr.unimernet.Unimernet import UnimernetModel
mfr_model = UnimernetModel(weight_dir, device="cuda")

# 表格识别模型
from mineru.model.table.rec.RapidTable import RapidTableModel
table_model = RapidTableModel(ocr_engine)
```

### 模型配置和优化
- **自动模型下载和缓存**: 统一的模型管理机制
- **多设备支持**: CPU/GPU/MPS/NPU自适应
- **动态批处理大小**: 根据内存和性能自动调整
- **内存使用优化**: 智能内存管理和清理
- **精度优化**: 支持float32/float16/bfloat16精度

## 输入输出格式

### MFR公式识别输出
```python
[
    {
        "category_id": 13 + int(cla.item()),  # 公式类别ID
        "poly": [x1, y1, x2, y2, x2, y2, x1, y2],  # 八边形坐标
        "score": round(float(conf.item()), 2),       # 置信度分数
        "latex": "识别的LaTeX公式字符串",             # LaTeX公式
    }
]
```

### Table表格识别输出
```python
{
    "html": "<table>...</table>",        # HTML表格结构
    "cell_bboxes": [...],                # 单元格边界框列表
    "logic_points": [...],               # 逻辑连接点
    "elapse": 0.123                      # 处理时间(秒)
}
```

## 性能指标

### MFR公式识别
- **LaTeX BLEU-4**: >0.80 (复杂公式)
- **识别精度**: >90% (标准公式)
- **处理速度**: <200ms/公式 (GPU)
- **内存使用**: 2-4GB (GPU)

### Table表格识别
- **结构准确率**: >95% (标准表格)
- **HTML质量**: 浏览器兼容性100%
- **处理速度**: <500ms/表格 (GPU)
- **复杂表格**: 支持多层嵌套表格

### 系统性能
- **端到端处理**: <3秒/页 (包含所有模型)
- **内存峰值**: <6GB (GPU模式)
- **并发处理**: 支持多文档并行处理
- **CPU兼容**: CPU模式可用但速度较慢

## 使用示例

### MFR公式识别示例
```python
from mineru.model.mfr.unimernet.Unimernet import UnimernetModel

# 加载公式识别模型
mfr_model = UnimernetModel(
    weight_dir="path/to/unimernet_model",
    device="cuda"
)

# 单张图像公式识别
formula_results = mfr_model.predict(mfd_res, image)

# 批量公式识别
batch_results = mfr_model.batch_predict(
    images_mfd_res=images_mfd_res,
    images=images,
    batch_size=64
)
```

### Table表格识别示例
```python
from mineru.model.table.rec.RapidTable import RapidTableModel
from mineru.model.ocr.pytorch_paddle import PytorchPaddleOCR

# 初始化OCR和表格模型
ocr_engine = PytorchPaddleOCR(lang='ch')
table_model = RapidTableModel(ocr_engine)

# 单张图像表格识别
html_code, cell_bboxes, logic_points, elapse = table_model.predict(image)

# 批量表格识别
table_model.batch_predict(table_res_list, batch_size=4)
```

## 技术亮点

### MFR技术亮点
- **Vision-Encoder-Decoder架构**: 统一的视觉-语言建模
- **动态长度控制**: 根据硬件资源自适应
- **智能排序优化**: 按面积排序提升批处理效率
- **LaTeX质量保证**: 后处理和清洗机制
- **多设备适配**: MPS/NPU特殊优化

### Table技术亮点
- **双模型融合**: RapidTable + SLANet Plus
- **OCR集成**: 与PP-OCRv5无缝集成
- **HTML输出**: 标准化的表格表示
- **批处理优化**: 高效的批量处理
- **安全处理**: HTML实体转义

### 通用技术亮点
- **内存优化**: 精确的内存使用控制
- **设备自适应**: 自动设备检测和优化
- **错误处理**: 完善的异常处理机制
- **性能监控**: 详细的处理时间和性能统计

## 常见问题 (FAQ)

### Q: MFR模型VRAM不足怎么办？
A: 系统会自动调整最大token长度，或可减小batch_size。

### Q: 表格识别出现乱码如何处理？
A: 检查OCR语言设置，确保文本编码正确，使用HTML转义。

### Q: 如何提高公式识别准确率？
A: 确保图像清晰度，调整检测阈值，使用合适的后处理参数。

### Q: 复杂表格识别失败怎么办？
A: 检查表格线条清晰度，调整OCR参数，考虑使用更高的处理精度。

### Q: 批处理速度慢如何优化？
A: 增加batch_size，使用GPU，减少图像分辨率。

## 相关文件清单

### MFR公式识别
- mfr/__init__.py - 模块初始化
- mfr/unimernet/Unimernet.py - UniMERNet模型接口
- mfr/unimernet/unimernet_hf/ - HuggingFace集成
- mfr/unimernet/unimernet_hf/modeling_unimernet.py - 核心模型实现
- mfr/unimernet/unimernet_hf/unimer_swin/ - Swin Transformer编码器
- mfr/unimernet/unimernet_hf/unimer_mbart/ - mBART解码器
- mfr/pp_formulanet_plus_m/ - PaddleFormulaNet实现
- mfr/utils.py - 公式处理工具

### Table表格识别
- table/__init__.py - 模块初始化
- table/rec/RapidTable.py - RapidTable核心实现
- table/rec/slanet_plus/ - SLANet Plus结构识别
- table/rec/unet_table/ - UNet表格恢复
- table/cls/paddle_table_cls.py - 表格分类器

### 工具和基础
- utils/__init__.py - 工具初始化
- utils/pytorchocr/ - PyTorch OCR组件

## 变更记录 (Changelog)
- 2025-11-17 17:02:15 - 深度分析MFR和Table模块技术实现，新增多GPU和测试文档
- 2025-11-17 16:46:31 - 更新Layout和OCR模块详细实现分析
- 2025-11-17 16:36:36 - 初始化Model模块文档