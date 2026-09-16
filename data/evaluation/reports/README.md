# 可公开评测报告

该目录保存已经检查过、不包含API密钥或模型网关地址的正式或历史评测报告。

- `rules-v1.1-report.json`：21例规则基线，使用当前prompt/validator协议重新生成。
- `deepseek-v4-pro-v1-11cases-exploratory.json`：旧版11例探索报告，仅用于说明通用模型高召回、高误报现象；不能作为当前21例正式阶段B结果。

运行时增量报告继续写入被Git忽略的 `data/runtime/evaluations/`。只有确认实验条件和敏感信息后，才将最终报告复制到本目录。
