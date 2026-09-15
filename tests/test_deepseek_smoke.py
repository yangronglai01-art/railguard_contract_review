"""DeepSeek真实接口冒烟入口的离线回归测试。"""

from collections.abc import Mapping, Sequence

import pytest

import railguard.agents.smoke as smoke_module
from railguard.agents.provider import RiskAnalyzerSet
from railguard.config import Settings
from railguard.models.schemas import (
    ContractDocument,
    Evidence,
    RiskFinding,
)


class RecordingAnalyzer:
    """记录调用参数且不访问外部模型的测试Agent。"""

    def __init__(self, name: str) -> None:
        """保存Agent名称并初始化调用记录。"""
        self.name = name
        self.call_count = 0
        self.contract: ContractDocument | None = None
        self.evidence_by_clause: dict[
            str,
            Sequence[Evidence],
        ] | None = None
        self.contract_evidence: tuple[Evidence, ...] | None = None

    async def analyze(
        self,
        *,
        contract: ContractDocument,
        evidence_by_clause: Mapping[
            str,
            Sequence[Evidence],
        ],
        contract_evidence: Sequence[Evidence],
    ) -> list[RiskFinding]:
        """记录一次分析调用并返回空风险列表。"""
        self.call_count += 1
        self.contract = contract
        self.evidence_by_clause = dict(evidence_by_clause)
        self.contract_evidence = tuple(contract_evidence)

        return []


async def test_smoke_run_calls_only_commercial_without_retry(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """验证冒烟入口只调用商务Agent且禁用SDK自动重试。"""
    secret_key = "secret-deepseek-test-key"
    settings = Settings(
        _env_file=None,
        model_provider="deepseek",
        model_name="deepseek-v4-pro",
        openai_api_key=None,
        openai_base_url=None,
        deepseek_api_key=secret_key,
        deepseek_base_url="https://deepseek.example",
        model_timeout_seconds=15.0,
        model_max_retries=9,
        model_max_input_chars=50_000,
    )

    commercial = RecordingAnalyzer(
        "commercial_risk_agent"
    )
    legal = RecordingAnalyzer("legal_risk_agent")
    security = RecordingAnalyzer(
        "security_risk_agent"
    )
    captured_factory_arguments: dict[str, object] = {}

    def fake_settings_factory() -> Settings:
        """返回完全隔离的DeepSeek测试配置。"""
        return settings

    def fake_analyzer_factory(
        *,
        model_name: str,
        api_key: str,
        base_url: str | None,
        timeout_seconds: float,
        max_retries: int,
        max_input_chars: int,
    ) -> RiskAnalyzerSet:
        """记录模型工厂参数并返回三个本地测试Agent。"""
        captured_factory_arguments.update(
            {
                "model_name": model_name,
                "api_key": api_key,
                "base_url": base_url,
                "timeout_seconds": timeout_seconds,
                "max_retries": max_retries,
                "max_input_chars": max_input_chars,
            }
        )

        return RiskAnalyzerSet(
            commercial=commercial,
            legal=legal,
            security=security,
        )

    monkeypatch.setattr(
        smoke_module,
        "Settings",
        fake_settings_factory,
    )
    monkeypatch.setattr(
        smoke_module,
        "create_deepseek_risk_analyzers",
        fake_analyzer_factory,
    )

    await smoke_module.run_smoke_test()

    output = capsys.readouterr().out

    assert captured_factory_arguments == {
        "model_name": "deepseek-v4-pro",
        "api_key": secret_key,
        "base_url": "https://deepseek.example",
        "timeout_seconds": 15.0,
        "max_retries": 0,
        "max_input_chars": 50_000,
    }
    assert commercial.call_count == 1
    assert legal.call_count == 0
    assert security.call_count == 0

    assert commercial.contract is not None
    assert commercial.contract.filename == (
        "deepseek-smoke-contract.txt"
    )
    assert len(commercial.contract.clauses) == 3
    assert commercial.evidence_by_clause == {}
    assert commercial.contract_evidence == ()

    assert "DeepSeek真实接口调用成功" in output
    assert secret_key not in output


def test_failure_output_hides_exception_message(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """验证配置异常正文中的敏感值不会写入终端。"""
    secret_key = "secret-must-not-be-printed"
    error = ValueError(
        f"invalid configuration contains {secret_key}"
    )

    smoke_module.print_failure_details(error)

    output = capsys.readouterr().out

    assert "本地DeepSeek配置无效" in output
    assert "ValueError" in output
    assert secret_key not in output