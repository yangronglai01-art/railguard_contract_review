"""应用大模型配置的校验测试。"""

import pytest
from pydantic import ValidationError

from railguard.config import Settings


def test_model_configuration_defaults_to_mock_mode() -> None:
    """验证默认配置继续使用离线确定性规则。"""
    settings = Settings(
        _env_file=None,
        model_provider="mock",
        model_name="mock-contract-reviewer",
        openai_api_key=None,
        openai_base_url=None,
    )

    assert settings.model_provider == "mock"
    assert settings.model_name == "mock-contract-reviewer"
    assert settings.openai_base_url is None
    assert settings.model_timeout_seconds == 60.0
    assert settings.model_max_retries == 2
    assert settings.model_max_input_chars == 120_000


def test_openai_mode_requires_api_key() -> None:
    """验证真实模型模式缺少密钥时立即拒绝启动。"""
    with pytest.raises(
        ValidationError,
        match="OPENAI_API_KEY is required",
    ):
        Settings(
            _env_file=None,
            model_provider="openai",
            model_name="contract-base-model",
            openai_api_key=None,
        )


def test_openai_mode_accepts_complete_configuration() -> None:
    """验证完整的基础模型或微调模型配置可以加载。"""
    settings = Settings(
        _env_file=None,
        model_provider="openai",
        model_name="ft:contract-review-model",
        openai_api_key="test-api-key",
        openai_base_url="https://model.example/v1",
        model_timeout_seconds=45.0,
        model_max_retries=3,
        model_max_input_chars=80_000,
    )

    assert settings.model_provider == "openai"
    assert settings.model_name == "ft:contract-review-model"
    assert settings.openai_api_key == "test-api-key"
    assert settings.openai_base_url == (
        "https://model.example/v1"
    )
    assert settings.model_timeout_seconds == 45.0
    assert settings.model_max_retries == 3
    assert settings.model_max_input_chars == 80_000


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("model_provider", "unsupported"),
        ("model_name", "   "),
        ("model_timeout_seconds", 0),
        ("model_max_retries", -1),
        ("model_max_input_chars", 0),
    ],
)
def test_model_configuration_rejects_invalid_values(
    field_name: str,
    invalid_value: object,
) -> None:
    """验证不支持的模式和无效限制不能进入运行阶段。"""
    arguments: dict[str, object] = {
        "_env_file": None,
        "model_provider": "mock",
        "model_name": "mock-contract-reviewer",
        "openai_api_key": None,
    }
    arguments[field_name] = invalid_value

    with pytest.raises(ValidationError):
        Settings(**arguments)