"""大模型供应商客户端和三个专业风险Agent的创建工厂。"""

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass

from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from railguard.agents.llm import (
    DEFAULT_MAX_INPUT_CHARS,
    LlmResponseError,
    LlmRiskAnalyzer,
    StructuredAnalysisInvoker,
)
from railguard.agents.models import LlmRiskAnalysis
from railguard.observability import record_model_retry
from railguard.workflow.planner import ConstrainedPlanner, LlmPlan

# DeepSeek官方OpenAI兼容接口的根地址。
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# 统一AI网关（ai-platform）OpenAI兼容接口的根地址。
DEFAULT_AI_PLATFORM_BASE_URL = "http://127.0.0.1:18700/v1"

# 网关上游是DeepSeek推理模型，推理Token与JSON内容共享输出预算。
# 网关Chat模型默认max_tokens仅1024，不足推理模型消耗，会导致
# 输出被截断为空的JSON（网关返回502）。这里显式申请更大预算。
AI_PLATFORM_MAX_OUTPUT_TOKENS = 8192


@dataclass(frozen=True, slots=True)
class RiskAnalyzerSet:
    """一次审核使用的三个专业大模型风险Agent。"""

    # 负责付款和验收风险。
    commercial: LlmRiskAnalyzer

    # 负责知识产权、责任和运维风险。
    legal: LlmRiskAnalyzer

    # 负责数据安全风险。
    security: LlmRiskAnalyzer


class _DeepSeekAnalysisInvoker:
    """使用DeepSeek Responses协议返回风险分析JSON的调用器。"""

    def __init__(
        self,
        chat_model: ChatOpenAI,
        *,
        response_max_retries: int,
        schema_model: type = LlmRiskAnalysis,
        schema_name: str = "LlmRiskAnalysis",
    ) -> None:
        """绑定输出Schema并配置响应协议重试次数。"""
        # Responses协议使用text.format，而不是Chat Completions的
        # response_format；Schema来自现有Pydantic业务模型。
        # 只发送DeepSeek公开文档定义的type、name和schema字段。
        self._model = chat_model.bind(
            text={
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": schema_model.model_json_schema(),
                }
            }
        )
        self._parser = StrOutputParser()
        self._response_max_retries = response_max_retries

    async def ainvoke(
        self,
        messages: Sequence[BaseMessage],
    ) -> object:
        """调用模型、解析JSON，并区分网络故障和响应格式错误。"""
        for attempt in range(self._response_max_retries + 1):
            # 服务连接和请求异常继续交给SDK自身的重试策略。
            response = await self._model.ainvoke(messages)

            try:
                # json.loads要求完整合法JSON，不自动修复截断结果。
                return json.loads(self._parser.invoke(response))
            except (json.JSONDecodeError, TypeError) as exc:
                if attempt >= self._response_max_retries:
                    # 空响应或非法JSON属于响应错误，不能误报为不可用。
                    raise LlmResponseError(
                        "DeepSeek returned an invalid JSON response"
                    ) from exc

                # 记录重试次数，用于监控端点的失败重试统计。
                record_model_retry()

                # 对协议瞬时故障做短退避，避免立即重复冲击服务。
                await asyncio.sleep(min(2**attempt, 4))

        raise AssertionError("unreachable response retry state")


class _ChatCompletionsJsonInvoker:
    """使用Chat Completions+JSON模式返回风险分析JSON的调用器。

    统一AI网关（ai-platform）只暴露OpenAI兼容的Chat Completions，
    且其上游DeepSeek不支持json_schema严格结构化输出。
    因此采用response_format=json_object，并把Schema经提示词下发，
    本地再用LlmRiskAnalysis严格校验字段。
    """

    def __init__(
        self,
        chat_model: ChatOpenAI,
        *,
        response_max_retries: int,
        schema_model: type = LlmRiskAnalysis,
    ) -> None:
        """绑定JSON模式并配置响应协议重试次数。"""
        # DeepSeek的JSON模式只支持type=json_object，且要求提示词
        # 出现"json"字样；Schema不能经response_format下发，只能
        # 写入系统提示词，由网关侧做JSON合法性校验、本地做字段校验。
        self._model = chat_model.bind(
            response_format={"type": "json_object"}
        )
        self._parser = StrOutputParser()
        self._response_max_retries = response_max_retries
        self._schema_model = schema_model

    async def ainvoke(
        self,
        messages: Sequence[BaseMessage],
    ) -> object:
        """调用模型、解析JSON，并区分网络故障和响应格式错误。"""
        augmented = _inject_json_schema_contract(
            messages,
            schema_model=self._schema_model,
        )
        for attempt in range(self._response_max_retries + 1):
            # 服务连接和请求异常继续交给SDK自身的重试策略。
            response = await self._model.ainvoke(augmented)

            try:
                # json.loads要求完整合法JSON，不自动修复截断结果。
                return json.loads(self._parser.invoke(response))
            except (json.JSONDecodeError, TypeError) as exc:
                if attempt >= self._response_max_retries:
                    # 空响应或非法JSON属于响应错误，不能误报为不可用。
                    raise LlmResponseError(
                        "Gateway returned an invalid JSON response"
                    ) from exc

                # 记录重试次数，用于监控端点的失败重试统计。
                record_model_retry()

                # 对协议瞬时故障做短退避，避免立即重复冲击服务。
                await asyncio.sleep(min(2**attempt, 4))

        raise AssertionError("unreachable response retry state")


def _inject_json_schema_contract(
    messages: Sequence[BaseMessage],
    *,
    schema_model: type = LlmRiskAnalysis,
) -> list[BaseMessage]:
    """把业务模型的JSON Schema追加到系统提示词。

    JSON模式无法在response_format中携带Schema，因此把Schema
    契约写入系统提示词，指导模型按结构输出纯JSON。
    """
    schema_text = json.dumps(
        schema_model.model_json_schema(),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    contract = (
        "输出要求：只输出一个JSON对象，不要输出任何解释、前后缀"
        "文字或Markdown代码块。JSON必须严格符合以下JSON Schema：\n"
        f"{schema_text}"
    )

    augmented: list[BaseMessage] = list(messages)
    if not augmented:
        return [SystemMessage(content=contract)]

    first = augmented[0]
    if first.type == "system":
        content = first.content
        if not isinstance(content, str):
            content = str(content)
        augmented[0] = SystemMessage(content=f"{content}\n\n{contract}")
    else:
        augmented.insert(0, SystemMessage(content=contract))

    return augmented


def _normalize_model_arguments(
    *,
    model_name: str,
    api_key: str,
    base_url: str | None,
    timeout_seconds: float,
    max_retries: int,
    max_input_chars: int,
) -> tuple[str, str, str | None]:
    """验证公共连接参数并清理名称、密钥和接口地址的空白。"""
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

    return (
        normalized_model_name,
        normalized_api_key,
        normalized_base_url,
    )


def _create_analyzer_set(
    *,
    invoker: StructuredAnalysisInvoker,
    max_input_chars: int,
    business_profile: str = "baus",
) -> RiskAnalyzerSet:
    """用相同调用器创建职责和分类权限不同的三个专业Agent。"""
    # 三个Agent共享无状态调用器，分别使用自己的职责提示词。
    # LangGraph继续并行执行，所有结果遵守同一业务输出协议。
    from railguard.business import profiles_for_business

    commercial, legal, security = profiles_for_business(business_profile)
    return RiskAnalyzerSet(
        commercial=LlmRiskAnalyzer(
            profile=commercial,
            business_profile=business_profile,
            invoker=invoker,
            max_input_chars=max_input_chars,
        ),
        legal=LlmRiskAnalyzer(
            profile=legal,
            business_profile=business_profile,
            invoker=invoker,
            max_input_chars=max_input_chars,
        ),
        security=LlmRiskAnalyzer(
            profile=security,
            business_profile=business_profile,
            invoker=invoker,
            max_input_chars=max_input_chars,
        ),
    )


def create_openai_risk_analyzers(
    *,
    model_name: str,
    api_key: str,
    base_url: str | None = None,
    timeout_seconds: float = 60.0,
    max_retries: int = 2,
    max_input_chars: int = DEFAULT_MAX_INPUT_CHARS,
    business_profile: str = "baus",
) -> RiskAnalyzerSet:
    """创建使用OpenAI严格结构化接口的三个专业风险Agent。

    model_name可以填写基础模型或供应商微调模型名称。
    提示词、输出协议和工作流节点保持一致，便于复用评测流程。
    """
    (
        normalized_name,
        normalized_key,
        normalized_url,
    ) = _normalize_model_arguments(
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        max_input_chars=max_input_chars,
    )

    # SDK负责供应商连接、超时和临时故障重试。
    chat_model = ChatOpenAI(
        model=normalized_name,
        api_key=normalized_key,
        base_url=normalized_url,
        timeout=timeout_seconds,
        max_retries=max_retries,
    )

    # OpenAI服务按照Pydantic模型的严格JSON Schema返回结果。
    structured_invoker = chat_model.with_structured_output(
        LlmRiskAnalysis,
        method="json_schema",
        strict=True,
    )

    return _create_analyzer_set(
        invoker=structured_invoker,
        max_input_chars=max_input_chars,
        business_profile=business_profile,
    )


def create_deepseek_risk_analyzers(
    *,
    model_name: str,
    api_key: str,
    base_url: str | None = DEFAULT_DEEPSEEK_BASE_URL,
    timeout_seconds: float = 60.0,
    max_retries: int = 2,
    max_input_chars: int = DEFAULT_MAX_INPUT_CHARS,
    business_profile: str = "baus",
) -> RiskAnalyzerSet:
    """创建使用DeepSeek Responses API的三个专业风险Agent。

    DeepSeek的Chat Completions接口与Responses接口支持的
    结构化参数不同，因此明确启用Responses协议并发送命名Schema。
    供应商输出约束和RailGuard本地业务校验共同保障审核结果。
    """
    (
        normalized_name,
        normalized_key,
        normalized_url,
    ) = _normalize_model_arguments(
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        max_input_chars=max_input_chars,
    )

    # 显式指定DeepSeek地址，避免空地址回退到OpenAI服务。
    # use_responses_api=True让SDK发送POST /responses请求。
    chat_model = ChatOpenAI(
        model=normalized_name,
        api_key=normalized_key,
        base_url=normalized_url or DEFAULT_DEEPSEEK_BASE_URL,
        timeout=timeout_seconds,
        max_retries=max_retries,
        use_responses_api=True,
    )

    # JSON解析完成后，LlmRiskAnalyzer仍会严格验证必填字段、
    # 风险分类、条款定位和证据引用权限。
    return _create_analyzer_set(
        invoker=_DeepSeekAnalysisInvoker(
            chat_model,
            response_max_retries=max_retries,
        ),
        max_input_chars=max_input_chars,
        business_profile=business_profile,
    )


def create_ai_platform_risk_analyzers(
    *,
    model_name: str,
    api_key: str,
    base_url: str | None = DEFAULT_AI_PLATFORM_BASE_URL,
    timeout_seconds: float = 60.0,
    max_retries: int = 2,
    max_input_chars: int = DEFAULT_MAX_INPUT_CHARS,
    business_profile: str = "baus",
) -> RiskAnalyzerSet:
    """创建接入统一AI网关的三个专业风险Agent。

    网关只暴露OpenAI兼容的Chat Completions，且上游DeepSeek
    不支持json_schema严格结构化输出，因此使用JSON模式并把
    Schema经提示词下发，由本地LlmRiskAnalyzer严格校验字段。
    """
    (
        normalized_name,
        normalized_key,
        normalized_url,
    ) = _normalize_model_arguments(
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        max_input_chars=max_input_chars,
    )

    # 显式指定网关地址，避免空地址回退到OpenAI服务。
    # 网关是OpenAI兼容的Chat Completions端点，不带Responses协议。
    # 显式声明输出Token预算，覆盖网关Chat模型1024的默认值，
    # 避免推理模型在推理阶段耗尽预算导致内容为空（网关502）。
    chat_model = ChatOpenAI(
        model=normalized_name,
        api_key=normalized_key,
        base_url=normalized_url or DEFAULT_AI_PLATFORM_BASE_URL,
        timeout=timeout_seconds,
        max_retries=max_retries,
        max_tokens=AI_PLATFORM_MAX_OUTPUT_TOKENS,
    )

    # JSON解析完成后，LlmRiskAnalyzer仍会严格验证必填字段、
    # 风险分类、条款定位和证据引用权限。
    return _create_analyzer_set(
        invoker=_ChatCompletionsJsonInvoker(
            chat_model,
            response_max_retries=max_retries,
        ),
        max_input_chars=max_input_chars,
        business_profile=business_profile,
    )


def create_openai_planner(
    *,
    model_name: str,
    api_key: str,
    base_url: str | None = None,
    timeout_seconds: float = 60.0,
    max_retries: int = 2,
    max_input_chars: int = DEFAULT_MAX_INPUT_CHARS,
    business_profile: str = "baus",
) -> ConstrainedPlanner:
    """创建使用OpenAI严格结构化接口的主Agent受约束规划器。

    规划器输出的LlmPlan契约与风险Agent的LlmRiskAnalysis契约
    相互独立；规划结果仍由确定性校验函数兜底。
    """
    (
        normalized_name,
        normalized_key,
        normalized_url,
    ) = _normalize_model_arguments(
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        max_input_chars=max_input_chars,
    )

    chat_model = ChatOpenAI(
        model=normalized_name,
        api_key=normalized_key,
        base_url=normalized_url,
        timeout=timeout_seconds,
        max_retries=max_retries,
    )

    structured_invoker = chat_model.with_structured_output(
        LlmPlan,
        method="json_schema",
        strict=True,
    )

    return ConstrainedPlanner(
        invoker=structured_invoker,
        max_input_chars=max_input_chars,
        business_profile=business_profile,
    )


def create_deepseek_planner(
    *,
    model_name: str,
    api_key: str,
    base_url: str | None = DEFAULT_DEEPSEEK_BASE_URL,
    timeout_seconds: float = 60.0,
    max_retries: int = 2,
    max_input_chars: int = DEFAULT_MAX_INPUT_CHARS,
    business_profile: str = "baus",
) -> ConstrainedPlanner:
    """创建使用DeepSeek Responses API的主Agent受约束规划器。"""
    (
        normalized_name,
        normalized_key,
        normalized_url,
    ) = _normalize_model_arguments(
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        max_input_chars=max_input_chars,
    )

    chat_model = ChatOpenAI(
        model=normalized_name,
        api_key=normalized_key,
        base_url=normalized_url or DEFAULT_DEEPSEEK_BASE_URL,
        timeout=timeout_seconds,
        max_retries=max_retries,
        use_responses_api=True,
    )

    return ConstrainedPlanner(
        invoker=_DeepSeekAnalysisInvoker(
            chat_model,
            response_max_retries=max_retries,
            schema_model=LlmPlan,
            schema_name="LlmPlan",
        ),
        max_input_chars=max_input_chars,
        business_profile=business_profile,
    )


def create_ai_platform_planner(
    *,
    model_name: str,
    api_key: str,
    base_url: str | None = DEFAULT_AI_PLATFORM_BASE_URL,
    timeout_seconds: float = 60.0,
    max_retries: int = 2,
    max_input_chars: int = DEFAULT_MAX_INPUT_CHARS,
    business_profile: str = "baus",
) -> ConstrainedPlanner:
    """创建接入统一AI网关的主Agent受约束规划器。"""
    (
        normalized_name,
        normalized_key,
        normalized_url,
    ) = _normalize_model_arguments(
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        max_input_chars=max_input_chars,
    )

    chat_model = ChatOpenAI(
        model=normalized_name,
        api_key=normalized_key,
        base_url=normalized_url or DEFAULT_AI_PLATFORM_BASE_URL,
        timeout=timeout_seconds,
        max_retries=max_retries,
        max_tokens=AI_PLATFORM_MAX_OUTPUT_TOKENS,
    )

    return ConstrainedPlanner(
        invoker=_ChatCompletionsJsonInvoker(
            chat_model,
            response_max_retries=max_retries,
            schema_model=LlmPlan,
        ),
        max_input_chars=max_input_chars,
        business_profile=business_profile,
    )
