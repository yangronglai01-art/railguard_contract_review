"""大模型供应商客户端和三个专业风险Agent的创建工厂。"""

from dataclasses import dataclass

from langchain_openai import ChatOpenAI

from railguard.agents.llm import (
    DEFAULT_MAX_INPUT_CHARS,
    LlmRiskAnalyzer,
)
from railguard.agents.models import LlmRiskAnalysis
from railguard.agents.prompts import (
    COMMERCIAL_PROFILE,
    LEGAL_PROFILE,
    SECURITY_PROFILE,
)


@dataclass(frozen=True, slots=True)
class RiskAnalyzerSet:
    """一次审核使用的三个专业大模型风险Agent。"""

    # 负责付款和验收风险。
    commercial: LlmRiskAnalyzer

    # 负责知识产权、责任和运维风险。
    legal: LlmRiskAnalyzer

    # 负责数据安全风险。
    security: LlmRiskAnalyzer


def create_openai_risk_analyzers(
    *,
    model_name: str,
    api_key: str,
    base_url: str | None = None,
    timeout_seconds: float = 60.0,
    max_retries: int = 2,
    max_input_chars: int = DEFAULT_MAX_INPUT_CHARS,
) -> RiskAnalyzerSet:
    """创建使用OpenAI兼容接口的三个专业风险Agent。

    model_name既可以填写基础模型名称，也可以填写后续微调产生的
    模型名称。两种模型使用相同的提示词、输出协议和工作流节点，
    因此可以直接执行A/B评测。
    """
    normalized_model_name = model_name.strip()
    normalized_api_key = api_key.strip()
    normalized_base_url = (
        base_url.strip()
        if base_url is not None and base_url.strip()
        else None
    )

    if not normalized_model_name:
        raise ValueError("model_name must not be blank")

    if not normalized_api_key:
        raise ValueError("api_key must not be blank")

    if timeout_seconds <= 0:
        raise ValueError(
            "timeout_seconds must be greater than zero"
        )

    if max_retries < 0:
        raise ValueError(
            "max_retries must be greater than or equal to zero"
        )

    if max_input_chars <= 0:
        raise ValueError(
            "max_input_chars must be greater than zero"
        )

    # ChatOpenAI只负责供应商连接和请求重试。
    # 业务提示词、分类权限和引用验证仍由RailGuard控制。
    chat_model = ChatOpenAI(
        model=normalized_model_name,
        api_key=normalized_api_key,
        base_url=normalized_base_url,
        timeout=timeout_seconds,
        max_retries=max_retries,
    )

    # strict=True要求供应商按照LlmRiskAnalysis的JSON Schema
    # 返回结果；include_raw保持默认False，直接获得Pydantic对象。
    structured_invoker = chat_model.with_structured_output(
        LlmRiskAnalysis,
        method="json_schema",
        strict=True,
    )

    # 三个Agent共享同一个无状态模型调用器，但拥有不同的职责提示词
    # 和分类白名单。LangGraph仍然可以并行执行它们。
    return RiskAnalyzerSet(
        commercial=LlmRiskAnalyzer(
            profile=COMMERCIAL_PROFILE,
            invoker=structured_invoker,
            max_input_chars=max_input_chars,
        ),
        legal=LlmRiskAnalyzer(
            profile=LEGAL_PROFILE,
            invoker=structured_invoker,
            max_input_chars=max_input_chars,
        ),
        security=LlmRiskAnalyzer(
            profile=SECURITY_PROFILE,
            invoker=structured_invoker,
            max_input_chars=max_input_chars,
        ),
    )