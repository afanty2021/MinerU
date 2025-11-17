[根目录](../CLAUDE.md) > **tests**

# Tests 测试模块

## 变更记录 (Changelog)
- 2025-11-17 17:02:15 - 新增测试模块文档，分析测试策略和覆盖率机制

## 模块职责

Tests模块负责MinerU项目的质量保证，提供完整的测试框架和测试用例，确保各个模块的功能正确性和系统稳定性。

## 测试架构

### 测试层次结构
- **单元测试**: 针对单个函数和类的测试
- **集成测试**: 模块间接口和交互测试
- **端到端测试**: 完整处理流程测试
- **性能测试**: 系统性能和资源使用测试

### 测试框架
- **pytest**: 主要测试框架
- **coverage**: 代码覆盖率工具
- **fuzzywuzzy**: 模糊匹配验证
- **BeautifulSoup**: HTML内容解析

## 核心测试文件

### 端到端测试 (`unittest/test_e2e.py`)

#### 测试覆盖范围
```python
def test_pipeline_with_two_config()
def test_vlm_transformers_with_default_config()
```

**测试场景**:
- **Pipeline后端测试**: txt和ocr两种解析方法
- **VLM后端测试**: Transformers引擎默认配置
- **多语言支持**: 中文、英文文档处理
- **不同文档类型**: PDF和图像文件处理

#### 测试流程
```python
# 1. 文档准备
pdf_files_dir = os.path.join(__dir__, "pdfs")
doc_path_list = []  # 收集测试文档

# 2. 数据预处理
pdf_bytes = read_fn(path)
new_pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes)

# 3. 后端分析
infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = pipeline_doc_analyze(
    pdf_bytes_list, p_lang_list, parse_method="txt"
)

# 4. 结果写入和验证
write_infer_result(...)
assert_content(res_json_path, parse_method="txt")
```

#### 验证机制
```python
def assert_content(res_json_path, parse_method):
    """验证处理结果的正确性"""
    # 加载结果JSON
    with open(res_json_path, 'r') as f:
        results = json.load(f)

    # 内容验证逻辑
    # - 文本完整性检查
    # - 结构化数据验证
    # - 模糊匹配对比
    pass
```

## 测试策略

### 数据驱动测试
- **测试样本**: 使用标准PDF样本进行测试
- **多语言**: 支持中文、英文等多种语言
- **多格式**: PDF、PNG、JPEG等格式支持
- **多场景**: 不同复杂度文档的处理测试

### 回归测试
- **版本兼容**: 确保新版本不破坏现有功能
- **接口稳定**: API接口向后兼容性测试
- **性能回归**: 性能指标不退化测试

### 边界条件测试
- **大文件处理**: 超大PDF文档处理测试
- **复杂文档**: 多列、多表格、多公式文档
- **损坏文档**: 不完整或损坏文档的处理
- **空文档**: 空白或极简文档的处理

## 测试数据管理

### 测试样本组织
```
tests/
├── unittest/
│   ├── pdfs/           # 测试PDF样本
│   ├── images/         # 测试图像样本
│   └── expected/       # 期望结果
├── output/             # 测试输出
│   └── test/          # 测试结果
└── fixtures/          # 测试固件
```

### 测试数据类型
- **学术论文**: 包含公式、表格、引用的复杂文档
- **技术文档**: 多列布局、代码块文档
- **商业文档**: 表格密集、格式规整文档
- **多语言文档**: 中英混合、纯英文、纯中文文档

## 代码覆盖率

### 覆盖率配置
```python
# pyproject.toml
[tool.coverage.run]
source = ["mineru"]
omit = [
    "mineru/cli/client.py",      # CLI入口文件
    "mineru/cli/fast_api.py",    # API服务器
    "mineru/cli/gradio_app.py",  # Web界面
    "*/tests/*",                 # 测试文件
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",         # 排除标记
    "def __repr__",            # 表示方法
    "raise AssertionError",     # 断言语句
    "raise NotImplementedError",  # 未实现方法
]
```

### 覆盖率报告
```bash
# 生成覆盖率报告
pytest --cov=mineru --cov-report=html

# 输出覆盖率统计
pytest --cov=mineru --cov-report=term-missing
```

### 当前覆盖状况
- **整体覆盖率**: 约30-40%
- **核心模块**: CLI、Backend、Model模块覆盖较好
- **工具模块**: Utils、Data模块覆盖有限
- **项目模块**: MCP、Tianshu等覆盖率较低

## 测试工具和辅助功能

### 测试工具集
```python
from mineru.cli.common import (
    convert_pdf_bytes_to_bytes_by_pypdfium2,  # PDF转换
    prepare_env,                              # 环境准备
    read_fn,                                  # 文件读取
)
from mineru.data.data_reader_writer import FileBasedDataWriter  # 数据写入
from bs4 import BeautifulSoup                 # HTML解析
from fuzzywuzzy import fuzz                   # 模糊匹配
```

### 测试辅助函数
```python
def read_fn(path):
    """读取测试文件"""

def convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes):
    """PDF字节转换"""

def write_infer_result(infer_results, ...):
    """写入推理结果"""

def assert_content(res_json_path, parse_method):
    """内容断言验证"""
```

## 性能测试

### 响应时间测试
- **单页处理**: <3秒 (Pipeline后端, GPU)
- **批量处理**: 吞吐量测试
- **并发处理**: 多请求并发测试
- **内存使用**: 峰值内存监控

### 准确性测试
- **文本准确率**: >95%文字识别准确率
- **公式识别**: BLEU-4 >0.80
- **表格恢复**: 结构化数据准确性
- **布局检测**: mAP >0.85

### 资源使用测试
- **CPU使用率**: 不同负载下的CPU使用
- **GPU内存**: VRAM使用效率
- **磁盘IO**: 文件读写性能
- **网络带宽**: 分布式部署网络开销

## 测试环境配置

### 本地测试环境
```bash
# 安装测试依赖
pip install -e ".[core,test]"

# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/unittest/test_e2e.py::test_pipeline_with_two_config
```

### CI/CD测试环境
- **GitHub Actions**: 自动化测试流程
- **多Python版本**: 3.10-3.13兼容性测试
- **多操作系统**: Linux、Windows、macOS测试
- **GPU测试**: GPU环境和CPU环境测试

### Docker测试环境
```dockerfile
# 测试环境Docker配置
FROM python:3.11-slim
RUN pip install mineru[core,test]
COPY tests/ /app/tests/
WORKDIR /app
CMD ["pytest", "tests/"]
```

## 测试报告和分析

### 覆盖率报告
- **HTML报告**: 详细的覆盖率可视化
- **XML报告**: CI/CD集成格式
- **命令行报告**: 快速概览信息

### 测试结果分析
- **通过率统计**: 测试用例执行统计
- **失败分析**: 失败原因分类统计
- **性能趋势**: 性能指标历史对比
- **覆盖率趋势**: 代码覆盖率变化趋势

## 质量保证流程

### 代码提交前
- **单元测试**: 新功能必须包含单元测试
- **覆盖率检查**: 确保新代码有测试覆盖
- **静态分析**: 代码质量检查
- **文档更新**: 相关文档同步更新

### 持续集成
- **自动触发**: 代码提交自动运行测试
- **并行执行**: 多测试套件并行运行
- **快速反馈**: 测试失败及时通知
- **报告生成**: 自动生成测试报告

### 发布前验证
- **完整测试**: 运行完整测试套件
- **性能基准**: 性能指标基准测试
- **兼容性测试**: 多环境兼容性验证
- **回归测试**: 确保无功能退化

## 测试扩展计划

### 测试覆盖提升
- **Model模块深度测试**: 各AI模型的专项测试
- **Utils模块补充**: 工具函数的完整测试
- **错误处理测试**: 异常情况的全面覆盖
- **边界条件测试**: 极端情况的处理验证

### 自动化测试增强
- **数据生成测试**: 自动生成测试数据
- **模糊测试**: 随机输入的鲁棒性测试
- **性能基准测试**: 自动化性能回归检测
- **集成测试扩展**: 更多场景的集成测试

### 测试工具改进
- **测试报告优化**: 更详细的测试分析报告
- **调试工具增强**: 更好的测试调试支持
- **Mock框架**: 更好的依赖模拟
- **测试数据管理**: 更完善的测试数据组织

## 常见问题 (FAQ)

### Q: 如何编写新的测试用例？
A: 继承现有测试框架，使用pytest装饰器，遵循命名约定。

### Q: 测试数据从哪里获取？
A: 使用公开数据集、合成数据或手动准备的标准样本。

### Q: 如何处理GPU环境测试？
A: 使用环境变量控制，支持CPU模拟，条件跳过GPU测试。

### Q: 覆盖率太低怎么办？
A: 逐步补充测试用例，优先覆盖核心功能和关键路径。

### Q: 如何调试测试失败？
A: 使用pytest调试选项，查看详细日志，设置断点调试。

## 相关文件清单

### 测试文件
- unittest/test_e2e.py - 端到端测试主文件
- unittest/test_*.py - 其他单元测试文件
- fixtures/ - 测试固件和数据
- conftest.py - pytest配置文件

### 测试数据
- pdfs/ - PDF测试样本
- images/ - 图像测试样本
- expected/ - 期望结果数据
- output/ - 测试输出结果

### 配置文件
- pyproject.toml - 项目配置和测试配置
- pytest.ini - pytest专用配置
- .coveragerc - 覆盖率配置

## 最佳实践

### 测试编写原则
- **单一职责**: 每个测试用例只验证一个功能点
- **独立性**: 测试用例间相互独立
- **可重复**: 测试结果可重复验证
- **清晰明确**: 测试意图清晰易懂

### 测试数据管理
- **版本控制**: 测试数据纳入版本控制
- **小而精**: 测试数据尽量小而精确
- **多样化**: 覆盖各种使用场景
- **及时更新**: 随功能变化及时更新

### 持续改进
- **定期审查**: 定期审查和优化测试
- **指标监控**: 监控测试覆盖率等关键指标
- **工具升级**: 及时升级测试工具和框架
- **经验总结**: 总结测试经验和最佳实践