[根目录](../CLAUDE.md) > **docker**

# Docker 容器化部署

## 变更记录 (Changelog)
- 2025-11-17 17:02:15 - 新增Docker容器化部署完整分析文档

## 模块职责

Docker目录提供MinerU的容器化部署解决方案，支持生产级、开发级和企业级的多种部署场景，包括单机部署、分布式部署和云原生部署。

## 容器架构概览

### 多场景部署支持
- **VLLM服务器**: 高性能推理服务器
- **API服务器**: RESTful API服务
- **Gradio界面**: Web交互界面
- **多GPU支持**: 分布式推理集群

### 部署模式
- **单容器模式**: 适合开发和测试
- **多容器模式**: 适合生产环境
- **集群模式**: 适合企业级大规模部署

## 核心文件分析

### Docker Compose 配置 (`compose.yaml`)

#### 服务架构设计
```yaml
services:
  mineru-vllm-server:    # VLLM推理服务器
  mineru-api:           # REST API服务器
  mineru-gradio:        # Gradio Web界面
```

#### 1. VLLM服务器配置
```yaml
mineru-vllm-server:
  image: mineru-vllm:latest
  container_name: mineru-vllm-server
  restart: always
  profiles: ["vllm-server"]  # 独立profile，可选择性启动
  ports:
    - 30000:30000
  environment:
    MINERU_MODEL_SOURCE: local
  entrypoint: mineru-vllm-server
  command:
    - --host 0.0.0.0
    - --port 30000
    # - --data-parallel-size 2  # 多GPU并行
    # - --gpu-memory-utilization 0.5  # VRAM使用控制
```

**关键特性**:
- **高性能推理**: 基于VLLM的高并发处理
- **多GPU支持**: 可配置data-parallel-size
- **内存优化**: 可调整GPU内存使用率
- **健康检查**: 内置健康状态监控

#### 2. API服务器配置
```yaml
mineru-api:
  image: mineru-vllm:latest
  container_name: mineru-api
  restart: always
  profiles: ["api"]
  ports:
    - 8000:8000
  environment:
    MINERU_MODEL_SOURCE: local
  entrypoint: mineru-api
  command:
    - --host 0.0.0.0
    - --port 8000
    # - --data-parallel-size 2
    # - --gpu-memory-utilization 0.5
```

**核心功能**:
- **REST API**: 标准HTTP接口
- **异步处理**: 支持并发请求
- **灵活配置**: 可调节VLLM参数
- **自动重启**: 生产环境可靠性

#### 3. Gradio界面配置
```yaml
mineru-gradio:
  image: mineru-vllm:latest
  container_name: mineru-gradio
  restart: always
  profiles: ["gradio"]
  ports:
    - 7860:7860
  environment:
    MINERU_MODEL_SOURCE: local
  entrypoint: mineru-gradio
  command:
    - --server-name 0.0.0.0
    - --server-port 7860
    - --enable-vllm-engine true  # 启用VLLM引擎
    # - --enable-api false
    # - --max-convert-pages 20
    # - --data-parallel-size 2
    # - --gpu-memory-utilization 0.5
```

**用户界面特性**:
- **Web交互**: 直观的Web操作界面
- **文件上传**: 支持PDF文档上传
- **实时处理**: 实时显示处理进度
- **API集成**: 可同时启用API接口

#### 通用资源配置
```yaml
ulimits:
  memlock: -1          # 内存锁定不限制
  stack: 67108864      # 栈大小限制(64MB)
ipc: host              # 共享主机IPC命名空间

deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          device_ids: ["0"]
          capabilities: [gpu]
```

**资源优化**:
- **GPU访问**: 直接访问主机GPU
- **内存锁定**: 提高内存访问性能
- **共享内存**: 优化进程间通信

### Dockerfile分析

#### 全球版Dockerfile (`global/Dockerfile`)
```dockerfile
# 使用官方VLLM镜像作为基础
FROM vllm/vllm-openai:v0.10.1.1  # Ampere架构GPU (Compute Capability>=8.0)
# FROM vllm/vllm-openai:v0.10.2  # Turing架构GPU (Compute Capability<8.0)

# 安装依赖
RUN apt-get update && \
    apt-get install -y \
        fonts-noto-core \     # Noto核心字体
        fonts-noto-cjk \      # 中日韩字体
        fontconfig \          # 字体配置
        libgl1 &&             # OpenGL库
    fc-cache -fv &&           # 重建字体缓存
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 安装MinerU
RUN python3 -m pip install -U 'mineru[core]' --break-system-packages && \
    python3 -m pip cache purge

# 下载模型
RUN /bin/bash -c "mineru-models-download -s huggingface -m all"

# 启动配置
ENTRYPOINT ["/bin/bash", "-c", "export MINERU_MODEL_SOURCE=local && exec \"$@\"", "--"]
```

**关键配置**:
- **GPU适配**: 针对不同GPU架构优化
- **字体支持**: 完整的多语言字体支持
- **模型预下载**: 预装所有必需模型
- **环境变量**: 自动设置本地模型源

#### 中国版Dockerfile (`china/Dockerfile`)
```dockerfile
# 中国特化版本
FROM python:3.10-slim

# 配置中国镜像源
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

# 安装中文字体和依赖
RUN apt-get update && \
    apt-get install -y \
        fonts-wqy-zenhei \      # 文泉驿正黑
        fonts-wqy-microhei \    # 文泉驿微米黑
        libgl1-mesa-glx &&      # Mesa OpenGL库
    fc-cache -fv

# 安装MinerU
RUN pip install 'mineru[core]'

# 配置ModelScope
ENV MINERU_MODEL_SOURCE=modelscope

# 启动脚本
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
```

**中国特化特性**:
- **镜像加速**: 使用清华源加速安装
- **中文字体**: 文泉驿字体优化中文显示
- **ModelScope**: 使用ModelScope模型源
- **网络优化**: 国内网络环境优化

## 部署指南

### 环境要求

#### 硬件要求
- **GPU**: NVIDIA GPU, CUDA 11.0+, 6GB+ VRAM
- **内存**: 16GB+ RAM (推荐32GB)
- **存储**: 20GB+ 可用空间
- **网络**: 模型下载需要互联网连接

#### 软件要求
- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **NVIDIA Container Toolkit**: GPU支持
- **CUDA驱动**: 与容器CUDA版本兼容

### 快速部署

#### 1. 环境准备
```bash
# 安装NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# 测试GPU支持
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

#### 2. 构建镜像
```bash
# 克隆仓库
git clone https://github.com/opendatalab/MinerU.git
cd MinerU/docker

# 构建全球版镜像
docker build -f global/Dockerfile -t mineru-vllm:latest .

# 构建中国版镜像
docker build -f china/Dockerfile -t mineru-vllm:china .
```

#### 3. 启动服务
```bash
# 启动API服务器
docker-compose --profile api up -d

# 启动Gradio界面
docker-compose --profile gradio up -d

# 启动VLLM服务器
docker-compose --profile vllm-server up -d

# 启动所有服务
docker-compose --profile api --profile gradio up -d
```

### 高级配置

#### 1. 多GPU配置
```yaml
# compose.yaml中修改
mineru-api:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            device_ids: ["0", "1"]  # 使用多个GPU
            capabilities: [gpu]

# 命令行参数
command:
  - --host 0.0.0.0
  - --port 8000
  - --data-parallel-size 2  # 使用2个GPU并行
  - --gpu-memory-utilization 0.8  # 使用80%的GPU内存
```

#### 2. 资源限制
```yaml
# CPU和内存限制
deploy:
  resources:
    limits:
      cpus: '8.0'          # 限制CPU核心数
      memory: 16G          # 限制内存使用
    reservations:
      cpus: '4.0'          # 保留CPU核心数
      memory: 8G           # 保留内存大小
```

#### 3. 存储配置
```yaml
# 挂载本地目录
volumes:
  - ./models:/app/models:ro        # 模型文件(只读)
  - ./output:/app/output           # 输出目录
  - ./logs:/app/logs               # 日志目录
  - mineru-cache:/root/.cache      # 模型缓存

# 环境变量配置
environment:
  - MINERU_MODEL_SOURCE=local
  - MINERU_CACHE_DIR=/app/models
  - PYTHONPATH=/app
```

#### 4. 网络配置
```yaml
# 自定义网络
networks:
  mineru-net:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16

services:
  mineru-api:
    networks:
      - mineru-net
    ports:
      - "8000:8000"
    extra_hosts:
      - "host.docker.internal:host-gateway"
```

### 生产环境部署

#### 1. 安全配置
```yaml
# 安全选项
security_opt:
  - no-new-privileges:true    # 禁止提升权限
  - seccomp:unconfined        # 禁用seccomp(如需要)

# 用户配置
user: "1000:1000"             # 非root用户运行

# 只读文件系统
read_only: true
tmpfs:
  - /tmp
  - /var/tmp
```

#### 2. 日志配置
```yaml
# 日志管理
logging:
  driver: "json-file"
  options:
    max-size: "100m"          # 单个日志文件最大100MB
    max-file: "5"             # 保留5个日志文件
    compress: "true"          # 压缩旧日志

# 日志输出
environment:
  - LOG_LEVEL=INFO
  - LOG_FORMAT=json
```

#### 3. 监控和健康检查
```yaml
# 健康检查
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  interval: 30s               # 每30秒检查一次
  timeout: 10s                # 超时10秒
  retries: 3                  # 重试3次
  start_period: 40s           # 启动后40秒开始检查

# 监控指标
environment:
  - PROMETHEUS_ENABLED=true
  - PROMETHEUS_PORT=9090
```

#### 4. 负载均衡
```yaml
# 使用Nginx负载均衡
nginx:
  image: nginx:alpine
  ports:
    - "80:80"
  volumes:
    - ./nginx.conf:/etc/nginx/nginx.conf
  depends_on:
    - mineru-api-1
    - mineru-api-2

# 多个API实例
mineru-api-1:
  image: mineru-vllm:latest
  environment:
    - SERVICE_ID=1

mineru-api-2:
  image: mineru-vllm:latest
  environment:
    - SERVICE_ID=2
```

## 性能优化

### 1. GPU优化
```yaml
# GPU内存优化
environment:
  - CUDA_VISIBLE_DEVICES=0,1
  - NVIDIA_VISIBLE_DEVICES=all
  - NVIDIA_DRIVER_CAPABILITIES=compute,utility

# VLLM参数优化
command:
  - --gpu-memory-utilization=0.95   # 使用95%的GPU内存
  - --max-num-seqs=256              # 最大并发序列数
  - --max-num-batched-tokens=8192   # 最大批处理token数
```

### 2. CPU优化
```yaml
# CPU绑定和亲和性
deploy:
  resources:
    reservations:
      cpus: '8.0'
  placement:
    constraints:
      - "node.labels.mineru-gpu==true"

# 进程优化
environment:
  - OMP_NUM_THREADS=8              # OpenMP线程数
  - MKL_NUM_THREADS=8              # MKL线程数
```

### 3. 网络优化
```yaml
# 网络优化
sysctls:
  - net.core.somaxconn=65535       # 增加连接队列长度
  - net.ipv4.tcp_tw_reuse=1        # 重用TIME_WAIT连接

# 缓存优化
environment:
  - REDIS_URL=redis://redis:6379   # Redis缓存
  - ENABLE_CACHE=true              # 启用缓存
```

## 故障排除

### 常见问题

#### 1. GPU不可用
```bash
# 检查NVIDIA驱动
nvidia-smi

# 检查Docker GPU支持
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# 修复方法
sudo systemctl restart docker
sudo modprobe nvidia
```

#### 2. 内存不足
```yaml
# 调整GPU内存使用
environment:
  - GPU_MEMORY_UTILIZATION=0.5

# 调整批处理大小
command:
  - --max-num-batched-tokens=4096
```

#### 3. 模型下载失败
```bash
# 手动下载模型
docker run --rm -v $(pwd)/models:/app/models mineru-vllm:latest \
  mineru-models-download -s local -m all

# 使用ModelScope
environment:
  - MINERU_MODEL_SOURCE=modelscope
```

#### 4. 端口冲突
```yaml
# 修改端口映射
ports:
  - "8001:8000"    # 主机8001端口映射到容器8000
```

### 调试技巧

#### 1. 容器调试
```bash
# 进入容器调试
docker exec -it mineru-api bash

# 查看日志
docker logs mineru-api
docker logs -f mineru-api  # 实时日志

# 检查资源使用
docker stats mineru-api
```

#### 2. 性能监控
```bash
# GPU使用监控
nvidia-smi -l 1

# 容器资源监控
docker stats --no-stream
```

## 最佳实践

### 1. 镜像管理
```bash
# 使用多阶段构建减小镜像大小
# 定期更新基础镜像
# 使用.dockerignore排除不必要文件
```

### 2. 安全实践
```yaml
# 使用非root用户
user: "1000:1000"

# 最小权限原则
security_opt:
  - no-new-privileges:true

# 定期更新
image: mineru-vllm:v2.6.3
```

### 3. 备份和恢复
```bash
# 数据备份
docker run --rm -v mineru-data:/data -v $(pwd)/backup:/backup \
  alpine tar czf /backup/mineru-data.tar.gz -C /data .

# 数据恢复
docker run --rm -v mineru-data:/data -v $(pwd)/backup:/backup \
  alpine tar xzf /backup/mineru-data.tar.gz -C /data
```

## 云平台部署

### AWS ECS部署
```yaml
# ECS任务定义
{
  "family": "mineru-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "4096",
  "memory": "8192",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole"
}
```

### Kubernetes部署
```yaml
# Kubernetes部署文件
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mineru-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mineru-api
  template:
    spec:
      containers:
      - name: mineru-api
        image: mineru-vllm:latest
        resources:
          requests:
            nvidia.com/gpu: 1
          limits:
            nvidia.com/gpu: 1
```

## 常见问题 (FAQ)

### Q: 如何选择合适的Dockerfile？
A: 全球用户使用global/Dockerfile，中国用户使用china/Dockerfile以获得更好的网络性能。

### Q: GPU内存不足如何处理？
A: 调整gpu-memory-utilization参数，减少批处理大小，或使用CPU模式。

### Q: 如何自定义模型？
A: 挂载本地模型目录，设置MINERU_MODEL_SOURCE=local环境变量。

### Q: 容器启动失败怎么办？
A: 检查GPU驱动、Docker版本、网络连接，查看容器日志进行调试。

### Q: 如何实现高可用部署？
A: 使用多个容器实例配合负载均衡器，配置健康检查和自动重启。

## 相关文件清单

### 配置文件
- compose.yaml - Docker Compose主配置
- global/Dockerfile - 全球版Docker镜像
- china/Dockerfile - 中国版Docker镜像
- nginx.conf - Nginx负载均衡配置(可选)

### 部署脚本
- deploy.sh - 自动化部署脚本
- backup.sh - 数据备份脚本
- monitor.sh - 监控脚本

### 配置模板
- .env.example - 环境变量模板
- docker-compose.prod.yml - 生产环境配置
- docker-compose.dev.yml - 开发环境配置