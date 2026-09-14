"""OpenAI兼容模型供应商工厂测试。"""

from collections.abc import Sequence

import pytest
from langchain_core.messages import BaseMessage

import railguard.agents.provider as provider_module
from railguard.agents.llm import LlmInputTooLargeError
from railguard.agents.models import LlmRiskAnalysis
from railguard.agents.provider import (
    create_openai_risk_analyzers,
)
from railguard.models.schemas import ContractDocument


class FakeStructuredInvoker:
    """返回空风险列表的本地结构化调用器。"""

    async def ainvoke(
        self,
        messages: Sequence[BaseMessage],
    ) -> object:
        """记录一次调用所需的最小异步接口。"""
        return {"findings": []}


class FakeChatOpenAI:
    """记录构造参数但不访问网络的ChatOpenAI替身。"""

    created_kwargs: dict[str, object] | None = None
    structured_call: tuple[object, str, bool] | None = None

    def __init__(
        self,
        **kwargs: object,
    ) -> None:
        """保存模型客户端收到的构造参数。"""
        type(self).created_kwargs = kwargs

    def with_structured_output(
        self,
        schema: object,
        *,
        method: str,
        strict: bool,
    ) -> FakeStructuredInvoker:
        """记录结构化输出配置并返回本地调用器。"""
        type(self).structured_call = (
            schema,
            method,
            strict,
        )
        return FakeStructuredInvoker()


def install_fake_chat_openai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """把供应商模块中的ChatOpenAI替换为本地测试类。"""
    FakeChatOpenAI.created_kwargs = None
    FakeChatOpenAI.structured_call = None

    monkeypatch.setattr(
        provider_module,
        "ChatOpenAI",
        FakeChatOpenAI,
    )


def test_factory_builds_strict_structured_analyzers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证工厂配置供应商并创建三个职责独立的Agent。"""
    install_fake_chat_openai(monkeypatch)

    analyzers = create_openai_risk_analyzers(
        model_name="  contract-base-model  ",
        api_key="  test-api-key  ",
        base_url="  https://model.example/v1  ",
        timeout_seconds=45.0,
        max_retries=3,
    )

    assert FakeChatOpenAI.created_kwargs == {
        "model": "contract-base-model",
        "api_key": "test-api-key",
        "base_url": "https://model.example/v1",
        "timeout": 45.0,
        "max_retries": 3,
    }
    assert FakeChatOpenAI.structured_call == (
        LlmRiskAnalysis,
        "json_schema",
        True,
    )

    assert analyzers.commercial.name == (
        "commercial_risk_agent"
    )
    assert analyzers.legal.name == "legal_risk_agent"
    assert analyzers.security.name == (
        "security_risk_agent"
    )


def test_factory_normalizes_blank_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证空白代理地址按默认OpenAI地址处理。"""
    install_fake_chat_openai(monkeypatch)

    create_openai_risk_analyzers(
        model_name="contract-model",
        api_key="test-api-key",
        base_url="   ",
    )

    assert FakeChatOpenAI.created_kwargs is not None
    assert FakeChatOpenAI.created_kwargs["base_url"] is None


@pytest.mark.parametrize(
    ("overrides", "expected_message"),
    [
        (
            {"model_name": "   "},
            "model_name must not be blank",
        ),
        (
            {"api_key": "   "},
            "api_key must not be blank",
        ),
        (
            {"timeout_seconds": 0.0},
            "timeout_seconds must be greater than zero",
        ),
        (
            {"max_retries": -1},
            (
                "max_retries must be greater than or equal "
                "to zero"
            ),
        ),
        (
            {"max_input_chars": 0},
            "max_input_chars must be greater than zero",
        ),
    ],
)
def test_factory_rejects_invalid_configuration(
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, object],
    expected_message: str,
) -> None:
    """验证无效模型配置在创建供应商客户端前失败。"""
    install_fake_chat_openai(monkeypatch)
    arguments: dict[str, object] = {
        "model_name": "contract-model",
        "api_key": "test-api-key",
    }
    arguments.update(overrides)

    with pytest.raises(
        ValueError,
        match=expected_message,
    ):
        create_openai_risk_analyzers(**arguments)

    assert FakeChatOpenAI.created_kwargs is None


async def test_factory_forwards_input_limit_to_analyzers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证工厂把输入限制应用到每个模型Agent。"""
    install_fake_chat_openai(monkeypatch)
    analyzers = create_openai_risk_analyzers(
        model_name="contract-model",
        api_key="test-api-key",
        max_input_chars=1,
    )
    contract = ContractDocument(
        contract_id="contract-provider-test",
        filename="provider-test.docx",
        full_text="测试合同",
    )

    with pytest.raises(LlmInputTooLargeError):
        await analyzers.commercial.analyze(
            contract=contract,
            evidence_by_clause={},
            contract_evidence=[],
        )