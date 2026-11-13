# 贡献指南

欢迎问题报告、文档修订和代码贡献。请先用最小样例说明预期行为与实际结果；避免在 Issue、截图和日志中包含用户文档或凭据。

## 开发环境

- Python 3.11+；Node.js 22+。
- 后端离线依赖：`python -m pip install -r backend/requirements-ci.txt`。
- 前端依赖：在 `web/` 执行 `npm ci`。
- 应用与模型启动方式见 [README.md](README.md)。前端由 FastAPI 从 `web/dist/` 提供；构建产物不提交。

默认 Compose 配合开发 override 挂载源码。修改业务代码后可以使用现有镜像；数据库结构变化通过 Alembic revision 管理，不在启动逻辑中追加临时迁移。

## 代码风格

Python 使用仓库 `ruff.toml`，前端使用 `.prettierrc.json`：

```sh
ruff format backend/app backend/alembic scripts
ruff check backend/app backend/alembic scripts
cd web
npm run format
```

避免为低影响的排版修改新增重复测试；行为变化应包含覆盖实际风险的回归用例。保持账号隔离、证据身份、知识范围和会话连续性，不为清理代码而改变这些边界。

## 验证

在仓库根目录运行：

```sh
python scripts/test_repository.py
python scripts/check_repository.py
ruff check backend/app backend/alembic scripts
ruff format --check backend/app backend/alembic scripts
python scripts/accept.py offline
```

前端在 `web/` 运行：

```sh
npm run format:check
npm run build
npm run test:unit
npx playwright install chromium
npm run test:ui
npm audit --audit-level=moderate
```

测试流程、断言和读取数据的逻辑公开；业务题目、文档正文、参考答案与案例对话由使用者自行设计，具体内容不随源码分发。测试所需文字从本地 `data/test-materials/<脚本名>.json` 读取，也可通过 `PROVIDENT_TEST_MATERIALS_DIR` 指定目录。文件结构为 `schema_version: 1` 和 `texts` 字符串映射，键与脚本引用对应。

缺少内容时，该脚本返回退出码 77，由入口单独统计为 `SKIP`，不计作通过。公开 CI 仍执行防泄露、格式与构建检查。正常启动应用、上传文档和问答不依赖这些测试文件。

`python scripts/accept.py live` 使用本地 Compose 实例执行生命周期验收，可能创建测试记录。其他 `test_*.py` 的环境要求见 [scripts/README.md](scripts/README.md)。不要对生产实例直接运行集成测试，也不要在 CI 中配置真实模型 Key 来执行离线测试。

## 提交与安全

提交前检查实际暂存内容：

```sh
python scripts/check_repository.py --staged
git diff --cached --stat
git diff --cached --check
```

防泄露检查扫描常见凭据模式、敏感路径、大文件及本地 `.env` 的凭据原值，命中时只输出位置和类型。它不能代替对文档、截图和编码后秘密的人工审核。

保留源码、迁移、依赖清单和测试逻辑。真实 `.env`、用户文档、数据库、模型、备份、日志、依赖目录和审计输出均应留在本地。测试题目、答案和正文也不进入源码仓库；用户另行获取的数据适用上游条款。

PR 请说明问题、修改后的行为与实际运行的验证。保持改动范围清楚；不要把一次性远端实验或本地环境修补混入产品代码。

## 独立发布目录

维护者可以运行 `python scripts/prepare_public_release.py`，将经过检查的当前文件复制到 `public-release/provident-rag/`。它包含尚未提交的有效源码修改，但排除 `.git`、内部笔记、具体测试内容和运行数据；清单与 SHA256 保存在同级 manifest 文件中。

该工具只读取 Git 文件列表，不创建仓库、提交或推送。目标已存在时会拒绝覆盖，可使用 `--destination public-release/<新目录名>` 指定新目录。后续发布与 Git 操作由维护者执行。
