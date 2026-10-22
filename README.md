# RailGuard · 小佩业务模拟合同审核

以小佩智能宠物硬件业务为背景的自建模拟项目：采购方审核软件、云服务、设备数据平台及相关技术服务合同。三Agent分别检查商务、法务和数据安全风险，验证证据引用后交人工决定。

合同、审核规则及数据为合成资料，不代表小佩内部制度、真实项目关系或法律意见。项目未接入真实业务数据、企业SSO或统一AI底座。已补本地个人账号、合同授权与持久审计。本轮迁移执行日期为2026-10-21，与历史2024年6月规划设定分开记录。

## 本轮迁移

- 独立petkit场景与规则版本，保留baus旧场景及冻结评测。
- 六类规则：付款、验收、知识产权/订阅许可、违约、运维与退出、数据安全。
- 覆盖喂食器、饮水机、猫砂盆相关App接口、用户账号、设备遥测、宠物影像与售后记录。
- 区分标准SaaS订阅和定制固件/App开发，不将未取得基础平台源代码本身当作风险。
- 沿用DOCX/PDF解析、原文定位、引用校验、人审和持久checkpoint恢复。
- 小佩场景始终等待人工确认，规则未识别到风险也不自动批准。
- 隔离入口固定Mock和本地合成证据，不读取原.env或使用宿主API密钥，不覆盖旧数据库。

Mock是有限关键词与形态规则，不是大模型审核效果。来源匹配也不等于完整法律依据验证。OEM制造、经销代理、消费者协议和跨境专项尚未验证。

## 本地运行（PowerShell）

在仓库根目录运行，沿用现有.venv。新环境先安装：`python -m pip install -e ".[dev]"`。

```powershell
# 第一个终端：默认仅监听127.0.0.1:18020
.\.venv\Scripts\python.exe -m railguard.petkit_demo serve

# 第二个终端：前端
$env:RAILGUARD_API_BASE_URL = 'http://127.0.0.1:18020'
.\.venv\Scripts\python.exe -m streamlit run src/railguard/ui/app.py --server.address 127.0.0.1 --server.port 18520

# 可选：在独立运行目录生成五份合成上传样本
.\.venv\Scripts\python.exe -m railguard.petkit_demo samples
```

样本程序不覆盖已有文件，重复生成请使用新的`--data-dir`。样本是开发回归数据，不能据此宣称独立泛化准确率。

API文档：`http://127.0.0.1:18020/docs`；业务版本：`GET /profile`。新数据默认在`data/runtime/petkit/`，支持`--data-dir`指定独立目录。

免登录serve入口只用于独立单人演示。多人流程验证使用serve-secure个人凭据模式：提交人只能访问自己合同，获授权审核人才能读取及决定，审计员仅看元数据。后端从认证身份写入审核人，已有个人账号的库拒绝降级成免登录。它仍是本地单实例验证，不是企业SSO或生产上线。

## 个人权限运行

建议为权限版指定新的独立运行目录，先发放个人凭据，再启动服务：

```powershell
.\.venv\Scripts\python.exe -m railguard.identities provision --data-dir data/runtime/petkit-access
.\.venv\Scripts\python.exe -m railguard.petkit_demo serve-secure --data-dir data/runtime/petkit-access
```

个人凭据保存在所选目录的credentials.json，不打印到终端，默认8小时到期。前端访问地址仍为18020；在页面输入个人凭据。buyer-01/02是不同提交人，legal-01为审核人，auditor-01为审计员。提交人上传、启动审核并授予legal-01合同审核权限；审核人登录后点击“读取我的审核任务”处理授权任务。不要把凭据文件提交Git。

轮换、停用、撤销，以及容器命令见[个人权限与容器运行](docs/小佩个人权限与容器运行.md)。已有旧免登录合同不会自动分配新账号权限，也不会被删除；需要保留旧演示时使用独立目录。真实RAG及底座仍未联调。

## 普通配置及历史兼容

新.env.example选用BUSINESS_PROFILE=petkit和独立语料、数据库。原私有.env未改，直接运行旧uvicorn入口仍受旧配置影响，推荐上述隔离入口。

Settings和独立模型工厂的兼容默认仍为baus，避免改变冻结评测。切换普通入口需同时核对场景、语料、数据库和checkpoint。小佩模式拒绝已知旧语料及混合版本合成规则。

HTTP RAG和真实模型接口保留，本轮未联调。旧DeepSeek分支为Responses协议，不能直接认定兼容底座Chat Completions。Compose已调整默认场景和独立存储，但容器验收与续做状态见下方个人权限运行文档。

## 测试与交接

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_petkit_migration.py -q
.\.venv\Scripts\python.exe scripts/test_petkit_offline.py
.\.venv\Scripts\python.exe -m ruff check src tests scripts
```

真实模型评测可能产生费用，本轮未运行。原评测、SFT和历史鲍斯资料保留原口径，不能作为小佩模型准确率结果。

- [迁移计划与DeepSeek交接](docs/小佩合同审核迁移计划与交接.md)
- [运行与验收记录](docs/小佩合同审核运行与验收.md)
- [旧鲍斯场景说明](docs/README_baus_legacy.md)

本轮不自动提交或推送GitHub。

续做验收入口：`python scripts/verify_access_demo.py`（临时个人权限HTTP/UI验收）、`python scripts/verify_petkit_docker.py`（独立Compose验收，完成后仅清理本次生成的项目与卷）。这两个命令固定Mock，不调用外部模型。
