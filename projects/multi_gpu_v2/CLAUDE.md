[根目录](../../CLAUDE.md) > [projects](../) > **multi_gpu_v2**

# Multi-GPU V2 项目

## 变更记录 (Changelog)
- 2025-11-17 17:02:15 - 新增Multi-GPU V2项目文档，深度分析多GPU并行处理架构

## 项目职责

Multi-GPU V2项目是MinerU的多GPU并行处理解决方案，基于LitServe框架构建，提供高性能的分布式PDF处理能力。该项目解决了单GPU处理大规模文档时的性能瓶颈问题。

## 核心架构

### 设计模式
- **服务端/客户端架构**: 基于LitServe的微服务架构
- **异步处理**: 支持并发请求处理，提升吞吐量
- **设备自适应**: 智能GPU资源分配和管理
- **模型源动态切换**: 支持HuggingFace/ModelScope/本地模型

### 系统组件

#### 1. 服务器端 (`server.py`)
```python
class MinerUAPI(ls.LitAPI):
    def setup(self, device)
    def decode_request(self, request)
    def predict(self, inputs)
    def encode_response(self, response)
```

**核心功能**:
- **环境配置**: 自动设置MINERU_DEVICE_MODE和VRAM配置
- **模型源检测**: 智能选择HuggingFace或ModelScope
- **请求解析**: Base64文件解码和参数提取
- **批量处理**: 支持多文件并发处理
- **临时文件管理**: 自动清理临时文件

#### 2. 客户端 (`client.py`)
```python
async def mineru_parse_async(session, file_path, url='http://127.0.0.1:8000/predict', **options)
async def main()
```

**核心功能**:
- **异步请求**: 使用aiohttp进行异步HTTP请求
- **并发处理**: 支持多文件同时处理
- **参数配置**: 灵活的解析参数传递
- **错误处理**: 完善的异常处理和日志记录

#### 3. 配置端点 (`_config_endpoint.py`)
```python
def config_endpoint()
```

**核心功能**:
- **连接检测**: 测试HuggingFace和ModelScope连接性
- **模型源切换**: 智能回退机制
- **超时控制**: 3秒连接超时设置
- **日志记录**: 详细的连接状态日志

## 技术实现细节

### 设备配置策略
```python
# 自动设备检测和配置
if os.getenv('MINERU_DEVICE_MODE', None) == None:
    os.environ['MINERU_DEVICE_MODE'] = device if device != 'auto' else get_device()

# VRAM动态配置
if device_mode.startswith("cuda") or device_mode.startswith("npu"):
    vram = round(get_vram(device_mode))
    os.environ['MINERU_VIRTUAL_VRAM_SIZE'] = str(vram)
```

### 模型源管理
```python
# 优先级: HuggingFace -> ModelScope -> Local
os.environ.setdefault('MINERU_MODEL_SOURCE', 'huggingface')

# 连接测试和智能回退
if os.environ['MINERU_MODEL_SOURCE'] == 'huggingface':
    try:
        response = requests.head("https://huggingface.co/models", timeout=TIMEOUT)
        if response.ok:
            return True
    except:
        os.environ['MINERU_MODEL_SOURCE'] = 'modelscope'
```

### 异步处理机制
```python
# 批量异步处理
async with aiohttp.ClientSession() as session:
    basic_tasks = [mineru_parse_async(session, file_path) for file_path in existing_files[:2]]
    custom_tasks = [mineru_parse_async(session, file_path, **custom_options) for file_path in existing_files[2:]]
    all_results = await asyncio.gather(*all_tasks)
```

## 部署配置

### 服务器启动
```bash
# 启动多GPU服务器
python server.py
```

**服务器参数**:
- **端口**: 8000 (默认)
- **加速器**: auto (自动检测)
- **设备**: auto (自动选择)
- **工作进程**: 每设备1个进程
- **超时**: False (无超时限制)

### 客户端使用
```bash
# 启动客户端测试
python client.py
```

**处理选项**:
```python
options = {
    'backend': 'pipeline',      # 解析后端
    'lang': 'ch',              # 语言
    'method': 'auto',          # 解析方法
    'formula_enable': True,    # 公式识别
    'table_enable': True,      # 表格识别
    'start_page_id': 0,        # 起始页
    'end_page_id': None,       # 结束页
    'server_url': None         # 远程VLM服务器
}
```

## 性能优化

### GPU资源管理
- **自动VRAM检测**: 动态获取GPU显存容量
- **内存优化**: 使用float16精度减少内存占用
- **批处理优化**: 智能批处理大小调整
- **设备负载均衡**: 多GPU工作负载分配

### 网络优化
- **异步IO**: 非阻塞网络请求
- **连接复用**: aiohttp会话复用
- **超时控制**: 合理的请求超时设置
- **错误重试**: 自动重试机制

### 文件处理优化
- **Base64编码**: 高效文件传输
- **临时文件管理**: 自动清理机制
- **流式处理**: 避免内存溢出
- **并发限制**: 控制并发数量

## 监控和日志

### 日志系统
```python
from loguru import logger

# 详细的处理日志
logger.info(f"Setting up on device: {device}")
logger.info(f"MINERU_VIRTUAL_VRAM_SIZE: {os.environ['MINERU_VIRTUAL_VRAM_SIZE']}")
logger.info(f"✅ Processed: {file_path} -> {result.get('output_dir', 'N/A')}")
```

### 健康检查
- **连接状态**: 实时连接状态监控
- **处理统计**: 成功/失败处理统计
- **性能指标**: 处理时间和吞吐量统计
- **错误追踪**: 详细的错误日志和堆栈

## 扩展性设计

### 多GPU支持
```python
server = ls.LitServer(
    MinerUAPI(output_dir='/tmp/mineru_output'),
    accelerator='auto',        # 自动加速器检测
    devices='auto',           # 自动设备选择
    workers_per_device=1,     # 每设备工作进程数
    timeout=False             # 无超时限制
)
```

### 水平扩展
- **负载均衡**: 支持多实例部署
- **服务发现**: 可集成服务注册中心
- **容器化**: 支持Docker和Kubernetes部署
- **云原生**: 适配云环境部署

### 插件化扩展
- **自定义后端**: 支持自定义解析后端
- **模型插件**: 支持新模型类型集成
- **存储插件**: 支持多种存储后端
- **监控插件**: 支持自定义监控指标

## 安全特性

### 数据安全
- **临时文件清理**: 自动清理临时文件
- **敏感信息保护**: 日志中不记录敏感信息
- **访问控制**: 支持API访问控制
- **数据加密**: 支持传输加密

### 系统安全
- **资源限制**: 内存和CPU使用限制
- **进程隔离**: 独立进程处理请求
- **权限控制**: 最小权限原则
- **审计日志**: 完整的操作审计

## 与其他模块集成

### 与CLI模块集成
```python
from mineru.cli.common import do_parse, read_fn
from mineru.utils.config_reader import get_device
from mineru.utils.model_utils import get_vram
```

### 与Utils模块集成
- **设备检测**: 使用get_device()进行设备检测
- **VRAM管理**: 使用get_vram()获取显存信息
- **配置管理**: 统一的配置管理机制

### 与Model模块集成
- **模型加载**: 自动模型加载和初始化
- **推理调用**: 集成各种AI模型推理
- **结果处理**: 统一的结果处理格式

## 测试和验证

### 功能测试
- **端到端测试**: 完整的处理流程测试
- **并发测试**: 多并发请求测试
- **错误测试**: 异常情况处理测试
- **性能测试**: 吞吐量和延迟测试

### 集成测试
- **多GPU测试**: 多GPU环境兼容性测试
- **网络测试**: 网络异常情况测试
- **存储测试**: 不同存储后端测试
- **模型测试**: 不同模型源切换测试

## 常见问题 (FAQ)

### Q: 如何配置多GPU处理？
A: 通过设置`workers_per_device`和`devices`参数，系统会自动检测可用GPU并分配工作进程。

### Q: 模型源切换失败怎么办？
A: 检查网络连接，确认HuggingFace或ModelScope可访问，或使用本地模型源。

### Q: 内存不足如何优化？
A: 调整`MINERU_VIRTUAL_VRAM_SIZE`环境变量，减少批处理大小，或使用CPU模式。

### Q: 如何监控服务状态？
A: 查看日志输出，添加健康检查端点，或集成外部监控系统。

### Q: 如何处理大文件？
A: 使用流式处理，增加内存限制，或实现文件分片处理。

## 相关文件清单

### 核心文件
- server.py - 多GPU服务器主程序
- client.py - 异步客户端示例
- _config_endpoint.py - 配置端点和模型源管理

### 依赖模块
- mineru/cli/common.py - CLI通用功能
- mineru/utils/config_reader.py - 配置管理
- mineru/utils/model_utils.py - 模型工具

### 外部依赖
- litserve - 服务框架
- aiohttp - 异步HTTP客户端
- loguru - 日志系统
- requests - HTTP请求库

## 部署建议

### 生产环境
- 使用容器化部署
- 配置负载均衡
- 设置资源限制
- 启用监控告警

### 开发环境
- 使用虚拟环境隔离
- 配置调试日志
- 启用热重载
- 集成开发工具

### 性能调优
- GPU资源优化
- 网络参数调优
- 内存使用优化
- 并发参数调整