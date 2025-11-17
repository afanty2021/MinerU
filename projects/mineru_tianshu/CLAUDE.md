[根目录](../../CLAUDE.md) > [projects](../) > **mineru_tianshu**

# MinerU Tianshu - 企业级分布式部署方案

## 变更记录 (Changelog)
- 2025-11-17 16:46:31 - 初始化Tianshu项目文档，完成核心架构分析

## 项目概述

MinerU Tianshu（天枢）是企业级多GPU文档解析服务的分布式部署方案，提供高可用、高性能的文档解析能力。该方案基于FastAPI + LitServe架构，支持水平扩展和容错处理。

## 核心特性

### 分布式架构
- **API服务器**: RESTful接口，任务提交和查询
- **任务调度器**: 智能任务分配和监控
- **工作节点**: LitServe驱动的GPU工作节点
- **数据存储**: SQLite任务队列 + MinIO对象存储

### 企业级功能
- **多GPU支持**: 自动GPU资源分配和负载均衡
- **高可用性**: 容错机制和故障恢复
- **监控告警**: 实时状态监控和健康检查
- **存储管理**: 自动文件清理和存储优化

## 核心组件

### API服务器 (`api_server.py`)
提供RESTful API接口，负责任务管理和结果返回。

#### 主要功能
```python
# 核心API端点
POST /upload          # 上传文档文件
POST /submit          # 提交解析任务
GET  /task/{task_id}  # 查询任务状态
GET  /result/{task_id} # 获取解析结果
GET  /health          # 服务健康检查
GET  /stats           # 系统统计信息
```

#### 核心类结构
```python
app = FastAPI(
    title="MinerU Tianshu API",
    description="天枢 - 企业级多GPU文档解析服务",
    version="1.0.0"
)

# 数据库管理
db = TaskDB()  # SQLite任务队列

# MinIO对象存储
MINIO_CONFIG = {
    'endpoint': os.getenv('MINIO_ENDPOINT'),
    'access_key': os.getenv('MINIO_ACCESS_KEY'),
    'secret_key': os.getenv('MINIO_SECRET_KEY'),
    'bucket_name': os.getenv('MINIO_BUCKET')
}
```

#### 关键特性
- **文件上传处理**: 支持PDF、图片等格式
- **任务队列管理**: SQLite数据库跟踪任务状态
- **图像处理**: Markdown中图像链接处理和MinIO上传
- **CORS支持**: 跨域访问支持
- **错误处理**: 完善的异常处理和日志记录

### 任务调度器 (`task_scheduler.py`)
智能任务调度和系统监控中心。

#### 调度模式
```python
class TaskScheduler:
    def __init__(
        self,
        litserve_url='http://localhost:9000/predict',
        monitor_interval=300,        # 监控间隔: 5分钟
        health_check_interval=900,   # 健康检查: 15分钟
        stale_task_timeout=60,       # 超时重置: 60分钟
        worker_auto_mode=True        # 自动循环模式
    )
```

#### 核心职责
1. **任务监控**: 监控SQLite队列状态
2. **健康检查**: 定期检查Worker节点健康
3. **故障恢复**: 重置超时任务，清理死锁
4. **统计收集**: 系统性能和任务统计
5. **资源清理**: 自动清理过期文件和记录

#### 监控功能
```python
async def monitor_queue(self):
    """监控任务队列状态"""

async def check_worker_health(self, session):
    """检查Worker健康状态"""

async def cleanup_stale_tasks(self):
    """清理超时任务"""

async def cleanup_old_files(self):
    """清理过期文件"""
```

### 工作节点 (`litserve_worker.py`)
基于LitServe的GPU工作节点，负责实际的文档解析。

#### LitServe集成
```python
# LitServe工作节点配置
worker_config = {
    "devices": "auto",              # 自动GPU选择
    "workers": 1,                   # 每GPU工作进程数
    "timeout": 300,                 # 任务超时时间
    "batch_size": 1                 # 批处理大小
}
```

#### 核心功能
- **自动GPU检测**: 多GPU环境自动分配
- **任务拉取**: 从队列自动获取待处理任务
- **模型加载**: 按需加载和卸载模型
- **结果上传**: 解析结果自动上传到存储

### 任务数据库 (`task_db.py`)
SQLite任务队列管理系统。

#### 数据表结构
```sql
CREATE TABLE tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT UNIQUE NOT NULL,
    status TEXT DEFAULT 'pending',   # pending/processing/completed/failed
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    input_file TEXT,
    output_file TEXT,
    error_message TEXT,
    worker_id TEXT,
    processing_time INTEGER
)
```

#### 核心操作
```python
class TaskDB:
    def create_task(self, task_id, input_file)
    def get_task(self, task_id)
    def update_task_status(self, task_id, status, **kwargs)
    def get_pending_tasks(self, limit=10)
    def get_task_stats(self)
```

## 部署架构

### 系统架构图
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client App    │    │  Load Balancer  │    │   API Server    │
│                 │───▶│                 │───▶│   (FastAPI)     │
│  Web/Mobile/CLI │    │  (Nginx/HAProxy)│    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                       ┌─────────────────┐            │
                       │   MinIO Storage │◀───────────┤
                       │                 │            │
                       │  (Object Store) │            │
                       └─────────────────┘            │
                                                       │
                       ┌─────────────────┐            │
                       │  Task Database  │◀───────────┤
                       │   (SQLite)      │            │
                       └─────────────────┘            │
                                                       │
┌─────────────────┐    ┌─────────────────┐            │
│  Worker Node 1  │    │  Worker Node N  │            │
│                 │    │                 │            │
│  LitServe + GPU │    │  LitServe + GPU │◀───────────┘
│                 │    │                 │
└─────────────────┘    └─────────────────┘
```

### 部署模式

#### 单机部署
```bash
# 启动API服务器
python api_server.py

# 启动工作节点
python litserve_worker.py

# 启动调度器（可选）
python task_scheduler.py
```

#### 分布式部署
```bash
# 节点1: API服务器 + 数据库 + 存储
docker-compose up api-server

# 节点2-N: 工作节点
docker-compose up worker
```

## 配置管理

### 环境变量配置
```bash
# API服务器配置
HOST=0.0.0.0
PORT=8000
WORKERS=4

# MinIO存储配置
MINIO_ENDPOINT=minio.example.com:9000
MINIO_ACCESS_KEY=your_access_key
MINIO_SECRET_KEY=your_secret_key
MINIO_BUCKET=mineru-tianshu

# 数据库配置
DATABASE_URL=sqlite:///tasks.db

# LitServe工作节点配置
LITSERVE_PORT=9000
GPU_DEVICES=auto
WORKERS_PER_GPU=1
```

### Docker配置
```yaml
# docker-compose.yml
version: '3.8'
services:
  api-server:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MINIO_ENDPOINT=minio:9000
    depends_on:
      - minio
      - redis

  worker:
    build: .
    ports:
      - "9000"
    environment:
      - LITSERVE_PORT=9000
      - GPU_DEVICES=auto
    deploy:
      replicas: 2
    depends_on:
      - api-server

  minio:
    image: minio/minio
    ports:
      - "9000:9000"
    environment:
      - MINIO_ACCESS_KEY=minioadmin
      - MINIO_SECRET_KEY=minioadmin
    command: server /data
```

## 使用指南

### API使用示例

#### 上传文档并提交任务
```python
import requests

# 1. 上传文档文件
with open('document.pdf', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/upload', files=files)
    upload_info = response.json()

# 2. 提交解析任务
task_data = {
    'file_path': upload_info['file_path'],
    'params': {
        'backend': 'pipeline',
        'ocr_lang': 'ch',
        'formula_enable': True
    }
}
response = requests.post('http://localhost:8000/submit', json=task_data)
task_info = response.json()
task_id = task_info['task_id']
```

#### 查询任务状态
```python
# 查询任务状态
response = requests.get(f'http://localhost:8000/task/{task_id}')
status_info = response.json()

if status_info['status'] == 'completed':
    # 获取解析结果
    response = requests.get(f'http://localhost:8000/result/{task_id}')
    result = response.json()
    print(f"Markdown结果: {result['markdown_content']}")
```

### 批处理处理
```python
# 批量提交任务
documents = ['doc1.pdf', 'doc2.pdf', 'doc3.pdf']
task_ids = []

for doc in documents:
    # 上传文档
    with open(doc, 'rb') as f:
        files = {'file': f}
        response = requests.post('http://localhost:8000/upload', files=files)
        upload_info = response.json()

    # 提交任务
    task_data = {'file_path': upload_info['file_path']}
    response = requests.post('http://localhost:8000/submit', json=task_data)
    task_ids.append(response.json()['task_id'])

# 等待所有任务完成
import time
completed_tasks = []
while len(completed_tasks) < len(task_ids):
    for task_id in task_ids:
        if task_id in completed_tasks:
            continue
        response = requests.get(f'http://localhost:8000/task/{task_id}')
        if response.json()['status'] == 'completed':
            completed_tasks.append(task_id)
    time.sleep(5)
```

## 监控和管理

### 系统状态监控
```bash
# 查看系统统计
curl http://localhost:8000/stats

# 检查服务健康
curl http://localhost:8000/health

# 查看任务队列状态
curl http://localhost:8000/stats | jq '.queue_stats'
```

### 日志管理
```bash
# API服务器日志
tail -f logs/api_server.log

# 工作节点日志
tail -f logs/worker.log

# 调度器日志
tail -f logs/scheduler.log
```

## 性能优化

### GPU资源配置
```python
# 高性能配置
worker_config = {
    "devices": "0,1,2,3",          # 指定GPU
    "workers": 2,                  # 每GPU 2个进程
    "batch_size": 2,               # 批处理大小
    "precision": "fp16",           # 半精度
    "model_cache": True            # 模型缓存
}
```

### 内存优化
```python
# 内存管理配置
memory_config = {
    "max_memory_usage": "10GB",    # 最大内存使用
    "cleanup_interval": 3600,      # 清理间隔
    "cache_size": "5GB",           # 缓存大小
    "enable_swap": False           # 禁用交换
}
```

### 存储优化
```python
# 存储配置
storage_config = {
    "compression": True,           # 启用压缩
    "retention_days": 30,          # 保留天数
    "auto_cleanup": True,          # 自动清理
    "backup_enabled": False        # 禁用备份（生产环境建议开启）
}
```

## 故障处理

### 常见问题及解决方案

#### 任务超时
```python
# 增加超时时间
stale_task_timeout = 120  # 2小时

# 或调整任务配置
task_config = {
    "timeout": 600,         # 单任务超时10分钟
    "retry_count": 3,       # 重试次数
    "retry_delay": 30       # 重试延迟
}
```

#### 内存不足
```python
# 减少并发数
workers_per_gpu = 1        # 每GPU 1个worker
batch_size = 1             # 批大小为1

# 启用内存清理
memory_config = {
    "cleanup_interval": 1800,  # 30分钟清理一次
    "enable_gc": True,         # 启用垃圾回收
    "max_memory_usage": "8GB"  # 限制内存使用
}
```

#### GPU资源不足
```python
# 智能GPU分配
def allocate_gpu(task_complexity):
    if task_complexity == "high":
        return "dedicated"  # 独占GPU
    elif task_complexity == "medium":
        return "shared"    # 共享GPU
    else:
        return "cpu"       # CPU模式
```

## 扩展开发

### 添加新的API端点
```python
@app.post("/custom_endpoint")
async def custom_function(data: dict):
    # 自定义处理逻辑
    return {"result": "success"}
```

### 自定义任务类型
```python
class CustomTask(Task):
    def process(self):
        # 自定义处理逻辑
        pass

    def validate_input(self):
        # 输入验证
        pass
```

### 扩展存储后端
```python
class CustomStorage:
    def upload(self, file_path, content):
        # 自定义上传逻辑
        pass

    def download(self, file_path):
        # 自定义下载逻辑
        pass
```

## 部署清单

### 生产环境检查项
- [ ] 负载均衡器配置
- [ ] SSL证书配置
- [ ] 数据库备份策略
- [ ] 监控告警设置
- [ ] 日志收集配置
- [ ] 资源限制设置
- [ ] 安全策略配置
- [ ] 容灾恢复方案

### 性能测试指标
- **并发处理**: 100+ 并发任务
- **吞吐量**: 1000+ 页/小时
- **响应时间**: API响应 < 100ms
- **可用性**: 99.9% 服务可用性
- **扩展性**: 支持水平扩展

## 相关文件清单

### 核心服务
- api_server.py - FastAPI主服务
- task_scheduler.py - 任务调度器
- litserve_worker.py - LitServe工作节点
- task_db.py - SQLite任务数据库

### 客户端和示例
- client_example.py - 客户端使用示例
- start_all.py - 一键启动脚本

### 配置文件
- docker-compose.yml - Docker编排配置
- requirements.txt - Python依赖
- Dockerfile - 容器构建文件

## 变更记录 (Changelog)
- 2025-11-17 16:46:31 - 初始化Tianshu项目文档，完成核心架构分析