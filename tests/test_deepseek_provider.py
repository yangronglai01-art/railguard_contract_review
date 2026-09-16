"""DeepSeek模型提供器的协议兼容和响应处理测试。"""

import json
from collections.abc import Sequence
from typing import ClassVar, Self
from unittest.mock import AsyncMock

import httpx
import pytest
from langchain_core.messages import AIMessage, BaseMessage
from langchain_openai import ChatOpenAI

import railguard.agents.provider as provider_module
from railguard.agents.llm import (
    LlmInvocationError,
    LlmResponseError,
)
from railguard.agents.models import LlmRiskAnalysis
from railguard.agents.provider import (
    create_deepseek_risk_analyzers,
)
from railguard.models.schemas import ContractDocument


class FakeChatOpenAI:
    """记录连接参数并返回固定消息的本地模型替身。"""

    # 保存创建客户端时使用的连接参数。
    created_kwargs: ClassVar[
        dict[str, object] | None
    ] = None

    # 保存绑定的Responses输出参数。
    bound_kwargs: ClassVar[
        dict[str, object] | None
    ] = None

    # 每次测试可以指定模型返回的文本或文本内容块。
    response_content: ClassVar[
        str | list[str | dict]
    ] = '{"findings": []}'

    # 可选的连续响应，用于验证协议失败后的再次调用。
    response_sequence: ClassVar[
        list[str | list[str | dict]]
    ] = []

    # 记录模型实际调用次数。
    invocation_count: ClassVar[int] = 0

    # 用于模拟模型服务调用失败。
    invocation_error: ClassVar[
        Exception | None
    ] = None

    def __init__(self, **kwargs: object) -> None:
        """保存客户端连接参数，不创建真实网络连接。"""
        type(self).created_kwargs = kwargs

    def bind(self, **kwargs: object) -> Self:
        """记录绑定的Responses参数并返回当前调用器。"""
        type(self).bound_kwargs = kwargs
        return self

    async def ainvoke(
        self,
        messages: Sequence[BaseMessage],
    ) -> AIMessage:
        """返回模拟模型消息，或抛出预设的服务调用异常。"""
        del messages

        type(self).invocation_count += 1

        if self.invocation_error is not None:
            raise self.invocation_error

        if self.response_sequence:
            return AIMessage(
                content=self.response_sequence.pop(0)
            )

        return AIMessage(content=self.response_content)


@pytest.fixture
def fake_chat_openai(
    monkeypatch: pytest.MonkeyPatch,
) -> type[FakeChatOpenAI]:
    """安装模型替身，并在每个测试前重置记录和响应。"""
    FakeChatOpenAI.created_kwargs = None
    FakeChatOpenAI.bound_kwargs = None
    FakeChatOpenAI.response_content = '{"findings": []}'
    FakeChatOpenAI.response_sequence = []
    FakeChatOpenAI.invocation_count = 0
    FakeChatOpenAI.invocation_error = None

    monkeypatch.setattr(
        provider_module,
        "ChatOpenAI",
        FakeChatOpenAI,
    )

    return FakeChatOpenAI


def create_test_contract() -> ContractDocument:
    """创建只用于验证模型连接和解析的最小合同对象。"""
    return ContractDocument(
        contract_id="deepseek-provider-test",
        filename="provider-test.docx",
        full_text="设备监测平台软件采购合同",
    )


def test_factory_builds_responses_analyzers(
    fake_chat_openai: type[FakeChatOpenAI],
) -> None:
    """验证连接参数规范化并绑定同一风险输出Schema。"""
    analyzers = create_deepseek_risk_analyzers(
        model_name="  deepseek-v4-pro  ",
        api_key="  deepseek-test-key  ",
        base_url="  https://api.deepseek.com  ",
        timeout_seconds=120.0,
        max_retries=1,
    )

    assert fake_chat_openai.created_kwargs == {
        "model": "deepseek-v4-pro",
        "api_key": "deepseek-test-key",
        "base_url": "https://api.deepseek.com",
        "timeout": 120.0,
        "max_retries": 1,
        "use_responses_api": True,
    }

    assert fake_chat_openai.bound_kwargs == {
        "text": {
            "format": {
                "type": "json_schema",
                "name": "LlmRiskAnalysis",
                "schema": LlmRiskAnalysis.model_json_schema(),
            }
        }
    }

    assert analyzers.commercial.name == "commercial_risk_agent"
    assert analyzers.legal.name == "legal_risk_agent"
    assert analyzers.security.name == "security_risk_agent"


@pytest.mark.parametrize("base_url", [None, "   "])
def test_blank_base_url_uses_deepseek_official_address(
    fake_chat_openai: type[FakeChatOpenAI],
    base_url: str | None,
) -> None:
    """验证空接口地址始终回退到DeepSeek官方服务。"""
    create_deepseek_risk_analyzers(
        model_name="deepseek-v4-pro",
        api_key="deepseek-test-key",
        base_url=base_url,
    )

    assert fake_chat_openai.created_kwargs is not None
    assert (
        fake_chat_openai.created_kwargs["base_url"]
        == "https://api.deepseek.com"
    )


async def test_text_content_blocks_are_parsed(
    fake_chat_openai: type[FakeChatOpenAI],
) -> None:
    """验证模型文本内容块可以解析并通过本地业务校验。"""
    fake_chat_openai.response_content = [
        {
            "type": "text",
            "text": '{"findings": []}',
        },
    ]

    analyzers = create_deepseek_risk_analyzers(
        model_name="deepseek-v4-pro",
        api_key="deepseek-test-key",
        max_retries=0,
    )

    findings = await analyzers.commercial.analyze(
        contract=create_test_contract(),
        evidence_by_clause={},
        contract_evidence=[],
    )

    assert findings == []


@pytest.mark.parametrize(
    "response_content",
    [
        "",
        '{"findings": []',
        '```json\n{"findings": []}\n```',
    ],
)
async def test_invalid_json_is_reported_as_response_error(
    fake_chat_openai: type[FakeChatOpenAI],
    response_content: str,
) -> None:
    """验证空响应、截断JSON和Markdown包裹不会被自动修复。"""
    fake_chat_openai.response_content = response_content

    analyzers = create_deepseek_risk_analyzers(
        model_name="deepseek-v4-pro",
        api_key="deepseek-test-key",
        max_retries=0,
    )

    with pytest.raises(
        LlmResponseError,
        match="invalid JSON response",
    ):
        await analyzers.commercial.analyze(
            contract=create_test_contract(),
            evidence_by_clause={},
            contract_evidence=[],
        )


async def test_invalid_json_is_retried_with_backoff(
    fake_chat_openai: type[FakeChatOpenAI],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证瞬时空响应会退避重试并接受后续合法JSON。"""
    fake_chat_openai.response_sequence = [
        "",
        '{"findings": []}',
    ]
    sleep = AsyncMock()
    monkeypatch.setattr(provider_module.asyncio, "sleep", sleep)
    analyzers = create_deepseek_risk_analyzers(
        model_name="deepseek-v4-pro",
        api_key="deepseek-test-key",
        max_retries=1,
    )

    findings = await analyzers.commercial.analyze(
        contract=create_test_contract(),
        evidence_by_clause={},
        contract_evidence=[],
    )

    assert findings == []
    assert fake_chat_openai.invocation_count == 2
    sleep.assert_awaited_once_with(1)


async def test_valid_json_still_requires_business_schema(
    fake_chat_openai: type[FakeChatOpenAI],
) -> None:
    """验证合法JSON也必须包含风险输出协议要求的全部字段。"""
    fake_chat_openai.response_content = (
        '{"findings": [{"category": "payment"}]}'
    )

    analyzers = create_deepseek_risk_analyzers(
        model_name="deepseek-v4-pro",
        api_key="deepseek-test-key",
    )

    with pytest.raises(
        LlmResponseError,
        match="invalid structured response",
    ):
        await analyzers.commercial.analyze(
            contract=create_test_contract(),
            evidence_by_clause={},
            contract_evidence=[],
        )


async def test_service_failure_remains_invocation_error(
    fake_chat_openai: type[FakeChatOpenAI],
) -> None:
    """验证服务调用故障与模型响应错误保持不同异常类型。"""
    fake_chat_openai.invocation_error = RuntimeError(
        "simulated service failure"
    )

    analyzers = create_deepseek_risk_analyzers(
        model_name="deepseek-v4-pro",
        api_key="deepseek-test-key",
    )

    with pytest.raises(LlmInvocationError):
        await analyzers.commercial.analyze(
            contract=create_test_contract(),
            evidence_by_clause={},
            contract_evidence=[],
        )


async def test_sdk_sends_supported_responses_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """用本地HTTP替身验证真实SDK的请求路径和Schema字段。"""
    captured_requests: list[httpx.Request] = []

    def handle_request(
        request: httpx.Request,
    ) -> httpx.Response:
        """截获HTTP请求，并返回固定的Responses协议消息。"""
        captured_requests.append(request)

        return httpx.Response(
            200,
            json={
                "id": "resp-provider-test",
                "created_at": 0,
                "model": "deepseek-v4-pro",
                "object": "response",
                "status": "completed",
                "error": None,
                "output": [
                    {
                        "id": "msg-provider-test",
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": '{"findings": []}',
                                "annotations": [],
                            }
                        ],
                    }
                ],
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                    "input_tokens_details": {
                        "cached_tokens": 0,
                    },
                    "output_tokens_details": {
                        "reasoning_tokens": 0,
                    },
                },
            },
        )

    # MockTransport在本机截获全部请求，不连接DeepSeek服务。
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handle_request),
    ) as local_http_client:

        def create_local_model(
            **kwargs: object,
        ) -> ChatOpenAI:
            """创建使用本地HTTP替身的真实模型客户端。"""
            return ChatOpenAI(
                http_async_client=local_http_client,
                **kwargs,
            )

        monkeypatch.setattr(
            provider_module,
            "ChatOpenAI",
            create_local_model,
        )

        analyzers = create_deepseek_risk_analyzers(
            model_name="deepseek-v4-pro",
            api_key="deepseek-test-key",
        )

        findings = await analyzers.commercial.analyze(
            contract=create_test_contract(),
            evidence_by_clause={},
            contract_evidence=[],
        )

    assert findings == []
    assert len(captured_requests) == 1

    request = captured_requests[0]
    payload = json.loads(request.content)

    assert request.url.host == "api.deepseek.com"
    assert request.url.path == "/responses"
    assert payload["model"] == "deepseek-v4-pro"

    # Responses协议必须使用input和text.format。
    assert "messages" not in payload
    assert "response_format" not in payload
    assert payload["input"]

    assert payload["text"]["format"] == {
        "type": "json_schema",
        "name": "LlmRiskAnalysis",
        "schema": LlmRiskAnalysis.model_json_schema(),
    }
