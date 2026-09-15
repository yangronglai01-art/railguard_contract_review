"""DeepSeek真实接口的单Agent冒烟测试。

本模块只提交一份简短的虚构合同，并调用一次商务风险Agent。
它用于验证API认证、Responses协议、JSON Schema和本地结果校验，
不会启动完整的三个Agent审核流程。
"""

import asyncio
from collections.abc import Sequence

from railguard.agents.llm import LlmAnalyzerError
from railguard.agents.provider import (
    create_deepseek_risk_analyzers,
)
from railguard.config import Settings
from railguard.models.schemas import (
    ContractDocument,
    RiskFinding,
)
from railguard.parsers.clauses import split_clauses

# 冒烟测试只使用虚构的简短合同，避免提交真实业务资料。
SMOKE_CONTRACT_TEXT = (
    "设备监测平台软件采购合同\n\n"
    "第一条 付款方式\n"
    "项目完成交付并经双方书面验收合格后，"
    "甲方支付90%合同款；剩余10%作为质保金，"
    "在质保期满后支付。\n\n"
    "第二条 验收\n"
    "双方按照附件测试标准逐项测试。"
    "未通过项目由乙方整改并重新验收，"
    "以双方签署的书面验收报告为准。"
)


def create_smoke_contract() -> ContractDocument:
    """创建只供真实接口冒烟测试使用的虚构合同。"""
    # 使用项目现有条款切分器生成可追溯的原文位置。
    clauses = split_clauses(SMOKE_CONTRACT_TEXT)

    return ContractDocument(
        filename="deepseek-smoke-contract.txt",
        full_text=SMOKE_CONTRACT_TEXT,
        clauses=clauses,
    )


def require_deepseek_api_key(settings: Settings) -> str:
    """验证当前配置为DeepSeek模式并返回清理后的密钥。"""
    if settings.model_provider != "deepseek":
        raise ValueError(
            "MODEL_PROVIDER必须设置为deepseek"
        )

    api_key = settings.deepseek_api_key

    if api_key is None or not api_key.strip():
        raise ValueError(
            "DEEPSEEK_API_KEY尚未配置"
        )

    return api_key.strip()


def print_findings(
    findings: Sequence[RiskFinding],
) -> None:
    """输出精简风险结果，避免打印冗长的模型原始响应。"""
    print(f"结构化风险数量：{len(findings)}")

    if not findings:
        print("商务Agent未发现风险。")
        return

    for index, finding in enumerate(findings, start=1):
        print(
            f"{index}. 类型={finding.finding_kind}，"
            f"分类={finding.category}，"
            f"等级={finding.level}"
        )


def print_failure_details(error: Exception) -> None:
    """输出不包含配置值和API密钥的精简失败信息。"""
    # 配置校验异常可能包含输入字段，因此不输出异常正文。
    if isinstance(error, LlmAnalyzerError):
        failure_message = "模型调用或结构化响应校验失败"
    else:
        failure_message = "本地DeepSeek配置无效"

    print(
        "DeepSeek冒烟测试失败："
        f"{failure_message}（{type(error).__name__}）"
    )

    # 只读取安全的异常类型和HTTP元数据。
    cause = error.__cause__

    if cause is None:
        return

    print(f"底层异常类型：{type(cause).__name__}")

    status_code = getattr(cause, "status_code", None)

    if status_code is not None:
        print(f"HTTP状态码：{status_code}")

    request_id = getattr(cause, "request_id", None)

    if request_id:
        print(f"请求编号：{request_id}")


async def run_smoke_test() -> None:
    """调用一次DeepSeek商务风险Agent并验证完整返回链路。"""
    # Settings会从项目根目录的.env文件读取本地配置。
    settings = Settings()
    api_key = require_deepseek_api_key(settings)

    # 创建三个Agent，但本测试只调用commercial一次。
    analyzers = create_deepseek_risk_analyzers(
        model_name=settings.model_name,
        api_key=api_key,
        base_url=settings.deepseek_base_url,
        timeout_seconds=settings.model_timeout_seconds,
        # 冒烟测试禁用SDK重试，保证最多只发送一次请求。
        max_retries=0,
        max_input_chars=settings.model_max_input_chars,
    )

    contract = create_smoke_contract()

    print("准备调用DeepSeek商务风险Agent一次。")
    print(f"模型：{settings.model_name}")
    print(f"Agent：{analyzers.commercial.name}")

    # 冒烟测试不连接RAG，因此传入空证据集合。
    findings = await analyzers.commercial.analyze(
        contract=contract,
        evidence_by_clause={},
        contract_evidence=(),
    )

    print("DeepSeek真实接口调用成功。")
    print_findings(findings)


def main() -> None:
    """运行冒烟测试并将预期错误转换为明确的退出状态。"""
    try:
        asyncio.run(run_smoke_test())
    except (ValueError, LlmAnalyzerError) as error:
        print_failure_details(error)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()