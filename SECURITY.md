# 敏感信息与公开发布

真实 API Key、JWT 密钥和数据库密码只放在本地 `.env` 或部署平台的 Secret 配置中。`.env.example` 中的 API Key 必须留空；其中的数据库密码和 JWT 值仅用于本地开发，公网运行前必须更换。不要通过 `VITE_*`、前端源码或前端构建参数传递服务端凭据。

`APP_ENV=production` 会校验认证开关、默认密钥和 CORS 配置。既有数据库的密码应通过 `python scripts/rotate_secrets.py` 轮换，不能只修改 `.env`。仓库示例不等于已经完成公网部署加固。

提交前运行 `python scripts/check_repository.py`，暂存后再运行 `python scripts/check_repository.py --staged`。检查覆盖常见服务商密钥、私钥、JWT、本地 `.env` 的凭据原值、敏感路径和大体积文件。它不是完整的安全审计，无法识别所有编码后的凭据或文档中的商业秘密；公开报告与截图仍需人工检查。

## 如果凭据曾经进入 Git

先在服务商或部署端撤销并轮换，再处理 Git 历史。新增 `.gitignore`、删除文件或新增一次修复提交，都不会抹去旧提交中的内容。历史清理会影响其他克隆与分支，应在确认范围并做好本地备份后单独处理。

不要在公开 Issue、提交说明或 CI 日志中粘贴密钥、完整 Token、数据库连接字符串或用户文档。报告问题时只提供脱敏的文件位置、问题类型和复现步骤。
