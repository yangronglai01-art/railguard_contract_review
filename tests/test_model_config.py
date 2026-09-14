"""应用大模型配置的读取和校验测试。"""

import pytest
from pydantic import ValidationError

from railguard.config import Settings


@pytest.fixture(autouse=True)
def isolate_model_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """隔离模型环境变量，使测试结果不受机器配置影响。"""
    # _env_file=None只禁止读取.env，不会忽略进程环境变量。
    # monkeypatch仅影响测试进程，测试结束后自动还原。
    variable_names = (
        "MODEL_PROVIDER",
        "MODEL_NAME",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_BASE_URL",
        "MODEL_TIMEOUT_SECONDS",
        "MODEL_MAX_RETRIES",
        "MODEL_MAX_INPUT_CHARS",
    )

    for variable_name in variable_names:
        monkeypatch.delenv(variable_name, raising=False)


def test_model_configuration_defaults_to_mock_mode() -> None:
    """验证默认模式无需外部密钥，并保留合理的模型调用限制。"""
    settings = Settings(_env_file=None)

    assert settings.model_provider == "mock"
    assert settings.model_name == "mock-contract-reviewer"
    assert settings.openai_api_key is None
    assert settings.openai_base_url is None
    assert settings.deepseek_api_key is None
    assert settings.deepseek_base_url == "https://api.deepseek.com"
    assert settings.model_timeout_seconds == 60.0
    assert settings.model_max_retries == 2
    assert settings.model_max_input_chars == 120_000


@pytest.mark.parametrize("api_key", [None, "", "   "])
def test_openai_mode_requires_api_key(
    api_key: str | None,
) -> None:
    """验证OpenAI模式拒绝缺失、空字符串和纯空白密钥。"""
    with pytest.raises(
        ValidationError,
        match="OPENAI_API_KEY is required",
    ):
        Settings(
            _env_file=None,
            model_provider="openai",
            model_name="contract-base-model",
            openai_api_key=api_key,
        )


def test_openai_mode_accepts_complete_configuration() -> None:
    """验证完整的OpenAI基础模型或微调模型配置可以加载。"""
    settings = Settings(
        _env_file=None,
        model_provider="openai",
        model_name="ft:contract-review-model",
        openai_api_key="openai-test-key",
        openai_base_url="https://model.example/v1",
        model_timeout_seconds=45.0,
        model_max_retries=3,
        model_max_input_chars=80_000,
    )

    assert settings.model_provider == "openai"
    assert settings.model_name == "ft:contract-review-model"
    assert settings.openai_api_key == "openai-test-key"
    assert settings.openai_base_url == "https://model.example/v1"
    assert settings.model_timeout_seconds == 45.0
    assert settings.model_max_retries == 3
    assert settings.model_max_input_chars == 80_000


@pytest.mark.parametrize("api_key", [None, "", "   "])
def test_deepseek_mode_requires_api_key(
    api_key: str | None,
) -> None:
    """验证DeepSeek模式拒绝缺失、空字符串和纯空白密钥。"""
    with pytest.raises(
        ValidationError,
        match="DEEPSEEK_API_KEY is required",
    ):
        Settings(
            _env_file=None,
            model_provider="deepseek",
            model_name="deepseek-v4-pro",
            deepseek_api_key=api_key,
        )


def test_deepseek_mode_accepts_complete_configuration() -> None:
    """验证DeepSeek模式可以配置企业网关和独立调用限制。"""
    settings = Settings(
        _env_file=None,
        model_provider="deepseek",
        model_name="deepseek-v4-pro",
        deepseek_api_key="deepseek-test-key",
        deepseek_base_url="https://deepseek.example",
        model_timeout_seconds=120.0,
        model_max_retries=1,
        model_max_input_chars=90_000,
    )

    assert settings.model_provider == "deepseek"
    assert settings.model_name == "deepseek-v4-pro"
    assert settings.deepseek_api_key == "deepseek-test-key"
    assert settings.deepseek_base_url == "https://deepseek.example"
    assert settings.model_timeout_seconds == 120.0
    assert settings.model_max_retries == 1
    assert settings.model_max_input_chars == 90_000


@pytest.mark.parametrize(
    ("model_provider", "credentials", "expected_message"),
    [
        (
            "openai",
            {"deepseek_api_key": "deepseek-test-key"},
            "OPENAI_API_KEY is required",
        ),
        (
            "deepseek",
            {"openai_api_key": "openai-test-key"},
            "DEEPSEEK_API_KEY is required",
        ),
    ],
)
def test_provider_requires_its_own_api_key(
    model_provider: str,
    credentials: dict[str, str],
    expected_message: str,
) -> None:
    """验证另一家供应商的密钥不能满足当前模式的启动要求。"""
    arguments: dict[str, object] = {
        "_env_file": None,
        "model_provider": model_provider,
        "model_name": "contract-review-model",
    }
    arguments.update(credentials)

    with pytest.raises(
        ValidationError,
        match=expected_message,
    ):
        Settings(**arguments)


def test_deepseek_configuration_reads_environment_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证DeepSeek环境变量可以正确映射到应用配置字段。"""
    monkeypatch.setenv("MODEL_PROVIDER", "deepseek")
    monkeypatch.setenv("MODEL_NAME", "deepseek-v4-pro")
    monkeypatch.setenv(
        "DEEPSEEK_API_KEY",
        "environment-test-key",
    )
    monkeypatch.setenv(
        "DEEPSEEK_BASE_URL",
        "https://api.deepseek.com",
    )
    monkeypatch.setenv("MODEL_TIMEOUT_SECONDS", "120")

    settings = Settings(_env_file=None)

    assert settings.model_provider == "deepseek"
    assert settings.model_name == "deepseek-v4-pro"
    assert settings.deepseek_api_key == "environment-test-key"
    assert settings.deepseek_base_url == "https://api.deepseek.com"
    assert settings.model_timeout_seconds == 120.0
    assert settings.openai_api_key is None


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
    """验证不支持的模式和无效调用限制不能进入运行阶段。"""
    arguments: dict[str, object] = {
        "_env_file": None,
        "model_provider": "mock",
        "model_name": "mock-contract-reviewer",
    }
    arguments[field_name] = invalid_value

    with pytest.raises(ValidationError):
        Settings(**arguments)