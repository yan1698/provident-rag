# Provident

基于用户文档的 RAG 知识库与多轮 Agent。支持文档上传、混合检索、重排序、引用定位、回答核验、综合分析和按篇阅读。

项目使用 Python / FastAPI、PostgreSQL / pgvector、Redis、React / Vite 构建，提供测试与受控评测工具，具体测试内容由使用者自行设计。

![Provident Agent 工作台](media/workspace.png)

Agent 工作台界面。

## 主要能力

- **知识库与文档**：解析 PDF、Markdown、TXT，保留页码、章节和原文片段；按账号隔离数据。
- **检索链路**：Dense + BM25，通过 RRF 融合后使用 Cross-Encoder 重排序，支持知识库范围约束。
- **问答与溯源**：回答引用实际使用的证据，支持原文定位与逐句核验。
- **统一 Agent 工作台**：按问题选择快速问答、综合分析或按篇阅读；支持同会话追问、任务取消、SSE 进度和失败后的轮询恢复。
- **回答来源区分**：文档问答与显式请求的模型通用解释分别标注，通用解释不冒充文档证据。
- **知识图谱**：章节树、概念抽取和概念邻域；图谱检索扩展默认关闭。
- **可观测与评测**：保存执行步骤、检索阶段耗时和模型用量；提供回归闸门与独立的公开数据评测适配器。

## 工作流程

上传文档 → 解析与分块 → 向量化 → 混合检索 → 重排序 → 生成带引用的回答 → 核验证据并定位原文。

综合分析和按篇阅读任务通过 Redis Worker 执行，任务状态与步骤保存在 PostgreSQL 中。前端通过 SSE 接收进度，连接中断后回退到轮询。

## 目录

```text
backend/app/       FastAPI API、检索、Agent、生成、核验与数据模型
backend/alembic/   数据库迁移
web/src/          React 页面、工作台和文档阅读器
scripts/          开发、验收、评测与发布准备工具
benchmarks/       数据版本注册表、适配逻辑和评测配置
data/raw/         本地文档目录，内容不随源码分发
data/eval/        使用者自行提供的评测配置，内容不随源码分发
docker/           PostgreSQL 初始化脚本
```

## 本地启动

需要 Python 3.11+、Node.js 22+、Docker Engine / Desktop 和 Docker Compose v2。以下流程使用默认开发配置；首次模型下载和镜像构建需要网络与额外磁盘空间。

### 1. 配置环境

复制 `.env.example` 为 `.env`，填写 `DASHSCOPE_API_KEY`，并为本地实例设置独立的 `JWT_SECRET`。其他供应商可通过 `ANSWER_LLM_PROVIDER` / `TASK_LLM_PROVIDER` 显式选择。

```sh
cp .env.example .env
```

PowerShell 使用 `Copy-Item .env.example .env`。API Key 只配置在服务端，不要放入前端源码或 `VITE_*` 变量。

### 2. 准备本地模型

创建并激活 Python 虚拟环境后运行：

```sh
python -m pip install modelscope
python scripts/download_embed_model.py
python scripts/download_rerank_model.py
```

默认模型为 `intfloat/multilingual-e5-small` 和 `BAAI/bge-reranker-base`。模型保存在 `data/runtime/models/`，Compose 将其挂载到容器内 `/models`。自定义模型需要同步调整 Compose 中的模型路径。

### 3. 构建前端并启动服务

```sh
cd web
npm ci
npm run build
cd ..
docker compose up -d --build
```

首次构建后，日常启动可以使用 `docker compose up -d --no-build`。默认 Compose 自动加载 `docker-compose.override.yml`，将当前源码、迁移和前端目录挂入容器；仅修改业务代码时可重新创建服务，无需重建依赖镜像。

打开：

- 工作台：`http://localhost:8000/ui/agent`
- API 文档：`http://localhost:8000/docs`
- 健康检查：`http://localhost:8000/health`

注册账号后，在知识库中上传你自己的文档，完成向量化，再到工作台提问。向量化和 Agent 需要 Worker 正常运行。

前端开发可另开终端执行 `cd web`、`npm run dev`，访问 `http://localhost:5173/ui/`；API 请求代理至本机 8000 端口。

## 配置

完整示例见 [.env.example](.env.example)。

- `ANSWER_LLM_PROVIDER` / `ANSWER_LLM_MODEL`：回答生成供应商与模型。
- `TASK_LLM_PROVIDER` / `TASK_LLM_MODEL`：任务理解与抽取供应商及模型。
- `DASHSCOPE_API_KEY` 等：服务端供应商凭据，示例文件必须留空。
- `AUTH_ENABLED` / `JWT_SECRET`：账号认证与 Token 签名。
- `SEARCH_TRACE_CANDIDATES`：候选检索诊断，默认关闭。
- `SPARSE_INDEX_CACHE_ENABLED`：BM25 快照缓存原型，默认关闭。

开发测试与应用运行相互独立：正常使用项目不需要任何预置测试集。运行中的应用若需使用 `stub`，必须显式设置 `ANSWER_LLM_PROVIDER=stub`、`ALLOW_LLM_STUB=true`，且不能处于 production 环境。

## 开发与验证

后端离线检查不安装 torch / CUDA，也不会调用真实模型：

```sh
python -m pip install -r backend/requirements-ci.txt
ruff check backend/app backend/alembic scripts
ruff format --check backend/app backend/alembic scripts
python scripts/accept.py offline
```

前端：

```sh
cd web
npm ci
npm run format:check
npm run build
npm run test:unit
npx playwright install chromium
npm run test:ui
```

测试脚本与断言逻辑公开，具体题目、文档正文、参考答案和案例对话不随源码分发。使用者可以自行设计测试内容；只有开发回归测试会读取 `data/test-materials/`，缺少相应文件时明确显示 `SKIP`，不计作通过。

正常使用时，用户直接在知识库页面上传自己的文档并提问，无需创建测试文件。

浏览器回归脚本使用本地模拟 API；配置了自定义测试内容后可检查会话与历史行为。Linux 首次安装 Playwright 时可使用 `npx playwright install --with-deps chromium`。也可通过 `BROWSER_CHANNEL=msedge` 使用已安装的 Edge。

开发与测试说明见 [scripts/README.md](scripts/README.md)。

## 评测与边界

[benchmarks/README.md](benchmarks/README.md) 介绍 RGB、CRUD-RAG 和 C-MTEB 的数据版本、下载范围、评测模式及指标限制。真实模型评测需要显式执行参数；准备检查、局部样本和固定上下文结果不能当作完整端到端质量成绩。

当前运行方案主要面向本地单机部署。公网部署还需要独立完成 TLS、访问控制、备份、Secret 管理、数据库与 Redis 的网络隔离和容量配置。`APP_ENV=production` 提供部分配置校验，不代表已经具备完整生产保障。

引用定位和核验结果用于帮助检查答案；检索遗漏、模型误答与核验误判仍可能发生。项目不包含用户文档、真实运行报告或模型权重。
