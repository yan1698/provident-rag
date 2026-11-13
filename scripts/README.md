# 脚本说明

## 日常开发

- `download_embed_model.py`、`download_rerank_model.py`：显式下载模型到本地运行目录。
- `embed_worker.py`：从仓库根目录运行向量化 Worker。
- `demo.py`：打印简短演示路径。
- `rotate_secrets.py`：同步轮换本地实例的数据库和 JWT 凭据，不打印秘密。
- `pin_eval_gate.py --username <现有账号>`：为指定账号授予评测管理权限，需要本地数据库。

## 测试

`accept.py offline` 是 CI 的固定离线测试入口。测试逻辑使用 FakeAgentStore、Mock 或内存 SQLite，覆盖编排预算、租户隔离、引用身份、输入边界、阅读续接、核验、检索诊断与缓存一致性。具体业务文字由使用者自行设计并从本地文件读取，不在脚本中预置。该入口不调用真实模型。缺少自定义文字内容的脚本返回退出码 77，入口单独统计跳过；不会把跳过当成通过。

`accept.py live` 执行本地 Compose 实例的生命周期检查。其余直接访问 HTTP、PostgreSQL 或 Redis 的集成测试需要运行中的开发实例，可能创建或修改测试记录；启用认证时还需要提供测试账号或 Token。

`test_chat_history_ui.mjs` 从 `web/` 的开发依赖加载 Playwright，读取自定义测试内容后启动本地模拟 API，使用当前构建产物验证会话、历史与响应式布局。输出保存在忽略的 `tmp/ui-regression/`。通过 `web/package.json` 中的 `test:ui` 运行；不依赖私人截图或旧版本源码快照。

`test_source_location.mjs`、`test_submit_idempotency.mjs` 是不需要浏览器的前端契约检查，通过 `npm run test:unit` 运行。

`test_repository.py` 在临时 Git 仓库中验证防泄露和发布工具，不修改当前项目的 Git 索引或历史。

## 评测

- `run_eval.py`、`label_gold.py`、`report_eval.py`：应用的 Golden Set 与回归报告。
- `run_benchmark.py`、`prepare_benchmark.py`、`run_benchmark_suite.py`：公开数据的下载、转换和受控执行。
- `check_benchmark_runtime.py`：检查评测运行环境。
- `report_benchmark.py`：整理实际评测输出；需要已有的运行目录。

真实模型执行需要显式授权参数。完整数据、报告和模型不进入源码仓库，详见 [评测说明](../benchmarks/README.md)。

## 发布准备

`check_repository.py` 检查上传候选，`--staged` 检查实际 Git 暂存内容。

`prepare_public_release.py` 根据检查结果与源码目录白名单复制发布目录，保留最新工作区内容、不携带历史，也不执行 Git 写操作。目标目录已存在时拒绝覆盖。

## 自定义测试内容

普通用户直接在应用知识库中上传自己的文档即可使用问答；不需要测试文件。开发者可以根据自己的文档和问题设计测试内容，放在本地 `data/test-materials/<脚本名>.json`，或用 `PROVIDENT_TEST_MATERIALS_DIR` 指定目录。`lib/test_materials.py` 与 `lib/test_materials.mjs` 只负责读取，不携带任何题目或答案。

公开发布工具保留脚本逻辑，排除 `data/raw/`、`data/eval/`、`data/test-materials/` 和 `benchmarks/fixtures/` 的实际内容，仅保留目录占位文件。
