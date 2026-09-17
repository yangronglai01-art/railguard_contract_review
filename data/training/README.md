# RailGuard SFT数据规范

本目录用于阶段C监督微调的数据准备。当前只提交规范和空白模板，不包含训练样本，也不启动训练。

可执行校验位于 `src/railguard/training/models.py`，JSONL读取和训练消息导出位于
`src/railguard/training/exporter.py`。导出器直接复用线上系统提示词、用户输入构造器和
结构化输出模型；只有已复核、已去标识且通过职责与证据约束的记录能够导出。

```powershell
python -m railguard.training validate --input data/training/annotations.jsonl
python -m railguard.training export --input data/training/annotations.jsonl `
  --split sft_train --output data/runtime/training/sft-train-messages.jsonl
```

导出文件属于运行时产物，不应提交到Git；终端摘要只显示计数和路径，不回显合同正文。

## 数据隔离

- `data/evaluation/contract-review-v1.json` 的21例已经用于提示词开发和阶段A/B分析，只能作为开发基准。
- SFT训练集、SFT验证集和最终held-out测试集必须使用全新的合同案例和稳定案例ID。
- 加载器会同时校验合同ID和合同正文SHA256；仅更换ID不能绕过训练集与验证集的内容防泄漏门禁。
- 分区身份以 `data/evaluation/dataset-governance-v1.json` 为准；任何案例ID不得跨分区重复。
- held-out测试集冻结后，不得用于训练、提示词修改、错误分析或checkpoint选择。

## 一条标注记录

每条记录只对应一个专业Agent，以JSONL保存。字段见 `sft-annotation-template-v1.json`：

- `example_id`：全局唯一且稳定。
- `split`：只能是 `sft_train` 或 `sft_validation`。
- `agent_name`：商务、法务或数据安全Agent之一。
- `source`：记录合成/授权来源、创建者、许可证和去标识状态。
- `contract`：合同ID、文件名、全文和稳定条款。
- `evidence_by_clause`、`contract_evidence`：与线上推理一致的证据白名单。
- `target`：严格符合 `LlmRiskAnalysis` 的目标JSON。
- `quality`：双人复核、争议状态和质检备注。

## 标注要求

1. 一个风险只能由职责范围内的Agent输出。
2. `clause_risk` 必须引用真实条款ID；`missing_clause` 不得伪造条款ID。
3. 每个 `evidence_id` 必须存在于当前风险允许的白名单中，且证据类别一致。
4. 安全负例目标必须是空 `findings`，不能为了“信息丰富”虚构风险。
5. 风险等级需要按统一标注手册判定，尤其避免把所有support风险标为high。
6. 合同、证据和目标不能包含真实个人信息、商业秘密或未获授权文本。
7. 每条样本至少经过一名标注者和一名复核者；争议样本不得进入冻结版本。

## 建议数据构成

- 35% 安全负例和否定表达，重点抑制阶段B误报。
- 25% 单一风险，隔离检验六类风险边界。
- 25% 多风险组合，覆盖跨条款关系。
- 15% 条款存在但内容不足、风险等级边界和行业变体。

优先增加 liability、support、data_security 的困难负例，并增加support低/中风险等级校准样本。

## 冻结条件

- JSONL全部通过结构和业务校验。
- 案例ID与其他分区互斥。
- 来源和授权字段完整。
- 双人复核完成且争议清零。
- 生成文件SHA256并写回治理清单。
- Git提交、数据版本、底座模型、prompt、RAG和训练参数可以共同复现实验。
