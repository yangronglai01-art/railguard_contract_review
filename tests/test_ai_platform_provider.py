"""统一AI网关（ai-platform）模型提供器的协议兼容和响应处理测试。"""

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
from railguard.agents.provider import (
    create_ai_platform_risk_analyzers,
)
from railguard.models.schemas import ContractDocument


class FakeChatOpenAI:
    """记录连接参数并返回固定消息的本地模型替身。"""

    # 保存创建客户端时使用的连接参数。
    created_kwargs: ClassVar[
        dict[str, object] | None
    ] = None

    # 保存绑定的Chat Completions输出参数。
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
        """记录绑定的Chat Completions参数并返回当前调用器。"""
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
        contract_id="ai-platform-provider-test",
        filename="provider-test.docx",
        full_text="设备监测平台软件采购合同",
    )


def test_factory_builds_chat_completions_analyzers(
    fake_chat_openai: type[FakeChatOpenAI],
) -> None:
    """验证连接参数规范化并绑定JSON模式输出。"""
    analyzers = create_ai_platform_risk_analyzers(
        model_name="  chat-default  ",
        api_key="  gateway-test-key  ",
        base_url="  http://127.0.0.1:18700/v1  ",
        timeout_seconds=120.0,
        max_retries=1,
    )

    assert fake_chat_openai.created_kwargs == {
        "model": "chat-default",
        "api_key": "gateway-test-key",
        "base_url": "http://127.0.0.1:18700/v1",
        "timeout": 120.0,
        "max_retries": 1,
        "max_tokens": 8192,
    }

    assert fake_chat_openai.bound_kwargs == {
        "response_format": {"type": "json_object"}
    }

    assert analyzers.commercial.name == "commercial_risk_agent"
    assert analyzers.legal.name == "legal_risk_agent"
    assert analyzers.security.name == "security_risk_agent"


@pytest.mark.parametrize("base_url", [None, "   "])
def test_blank_base_url_uses_gateway_address(
    fake_chat_openai: type[FakeChatOpenAI],
    base_url: str | None,
) -> None:
    """验证空接口地址始终回退到统一AI网关。"""
    create_ai_platform_risk_analyzers(
        model_name="chat-default",
        api_key="gateway-test-key",
        base_url=base_url,
    )

    assert fake_chat_openai.created_kwargs is not None
    assert (
        fake_chat_openai.created_kwargs["base_url"]
        == "http://127.0.0.1:18700/v1"
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

    analyzers = create_ai_platform_risk_analyzers(
        model_name="chat-default",
        api_key="gateway-test-key",
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

    analyzers = create_ai_platform_risk_analyzers(
        model_name="chat-default",
        api_key="gateway-test-key",
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
    analyzers = create_ai_platform_risk_analyzers(
        model_name="chat-default",
        api_key="gateway-test-key",
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

    analyzers = create_ai_platform_risk_analyzers(
        model_name="chat-default",
        api_key="gateway-test-key",
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

    analyzers = create_ai_platform_risk_analyzers(
        model_name="chat-default",
        api_key="gateway-test-key",
    )

    with pytest.raises(LlmInvocationError):
        await analyzers.commercial.analyze(
            contract=create_test_contract(),
            evidence_by_clause={},
            contract_evidence=[],
        )


async def test_sdk_sends_chat_completions_json_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """用本地HTTP替身验证真实SDK走Chat Completions并注入Schema。"""
    captured_requests: list[httpx.Request] = []

    def handle_request(
        request: httpx.Request,
    ) -> httpx.Response:
        """截获HTTP请求，并返回固定的Chat Completions协议消息。"""
        captured_requests.append(request)

        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-ai-platform-test",
                "object": "chat.completion",
                "model": "chat-default",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": '{"findings": []}',
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

    # MockTransport在本机截获全部请求，不连接真实网关服务。
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

        analyzers = create_ai_platform_risk_analyzers(
            model_name="chat-default",
            api_key="gateway-test-key",
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

    assert request.url.host == "127.0.0.1"
    assert request.url.path == "/v1/chat/completions"
    assert payload["model"] == "chat-default"

    # Chat Completions协议必须使用messages和response_format。
    assert payload["messages"]
    assert payload["response_format"] == {"type": "json_object"}
    assert "input" not in payload
    assert "text" not in payload

    # JSON Schema必须注入系统提示词，指导模型按结构输出。
    system_content = payload["messages"][0]["content"]
    assert "JSON Schema" in system_content
    assert "findings" in system_content
