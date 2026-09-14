"""离线评测命令行目标选择和模型配置测试。"""

import argparse
from collections.abc import (
    Mapping,
    Sequence,
)
from pathlib import Path

import pytest

from railguard.agents.provider import RiskAnalyzerSet
from railguard.config import Settings
from railguard.evaluation import __main__ as cli_module
from railguard.evaluation.models import EvaluationReport
from railguard.models.schemas import (
    ContractDocument,
    Evidence,
    RiskFinding,
)


class EmptyModelAnalyzer:
    """记录调用次数并返回空结果的模型Agent替身。"""

    def __init__(
        self,
        name: str,
    ) -> None:
        """保存Agent名称并初始化调用次数。"""
        self.name = name
        self.call_count = 0

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
        """记录一次模型分析但不产生风险。"""
        del contract
        del evidence_by_clause
        del contract_evidence

        self.call_count += 1
        return []


def create_arguments(
    *,
    target: str,
    system_name: str,
    output: Path | None,
) -> argparse.Namespace:
    """创建无需经过命令行解析的固定参数对象。"""
    return argparse.Namespace(
        target=target,
        system_name=system_name,
        dataset=Path(
            "data/evaluation/"
            "contract-review-v1.json"
        ),
        rag_corpus=Path(
            "data/demo/rag-corpus.json"
        ),
        output=output,
    )


def test_parser_keeps_rules_default_and_selects_output_paths() -> None:
    """验证旧规则命令保持兼容并为模型使用独立报告。"""
    parser = cli_module.build_argument_parser()

    rules_arguments = parser.parse_args([])
    llm_arguments = parser.parse_args(
        ["--target", "llm"]
    )
    explicit_arguments = parser.parse_args(
        [
            "--target",
            "llm",
            "--output",
            "custom-report.json",
        ]
    )

    assert rules_arguments.target == "rules"
    assert rules_arguments.system_name == "base_llm"
    assert cli_module.resolve_output_path(
        rules_arguments
    ) == Path(
        "data/runtime/evaluations/"
        "rules-v1-report.json"
    )
    assert cli_module.resolve_output_path(
        llm_arguments
    ) == Path(
        "data/runtime/evaluations/"
        "base-llm-v1-report.json"
    )
    assert cli_module.resolve_output_path(
        explicit_arguments
    ) == Path("custom-report.json")


async def test_llm_target_rejects_mock_configuration() -> None:
    """验证没有显式启用真实模型时不会产生模型请求。"""
    arguments = create_arguments(
        target="llm",
        system_name="base_llm",
        output=None,
    )
    settings = Settings(
        _env_file=None,
        model_provider="mock",
        model_name="mock-contract-reviewer",
        openai_api_key=None,
        openai_base_url=None,
    )

    with pytest.raises(
        ValueError,
        match="MODEL_PROVIDER=openai",
    ):
        await cli_module.run_llm_target(
            arguments,
            settings=settings,
        )


async def test_llm_cli_uses_config_and_saves_safe_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """验证阶段C也能复用入口且报告不泄露连接信息。"""
    output_path = tmp_path / "sft-report.json"
    arguments = create_arguments(
        target="llm",
        system_name="sft_llm",
        output=output_path,
    )
    settings = Settings(
        _env_file=None,
        model_provider="openai",
        model_name="ft:contract-review-model",
        openai_api_key="secret-test-key",
        openai_base_url="https://model.example/v1",
        model_timeout_seconds=48.0,
        model_max_retries=1,
        model_max_input_chars=75_000,
    )
    captured: dict[str, object] = {}
    commercial = EmptyModelAnalyzer(
        "commercial_risk_agent"
    )
    legal = EmptyModelAnalyzer(
        "legal_risk_agent"
    )
    security = EmptyModelAnalyzer(
        "security_risk_agent"
    )

    def fake_model_factory(
        *,
        model_name: str,
        api_key: str,
        base_url: str | None,
        timeout_seconds: float,
        max_retries: int,
        max_input_chars: int,
    ) -> RiskAnalyzerSet:
        """记录CLI传给供应商工厂的全部模型配置。"""
        captured.update(
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
        cli_module,
        "create_openai_risk_analyzers",
        fake_model_factory,
    )

    await cli_module.run_from_arguments(
        arguments,
        settings=settings,
    )

    assert captured == {
        "model_name": "ft:contract-review-model",
        "api_key": "secret-test-key",
        "base_url": "https://model.example/v1",
        "timeout_seconds": 48.0,
        "max_retries": 1,
        "max_input_chars": 75_000,
    }
    assert commercial.call_count == 11
    assert legal.call_count == 11
    assert security.call_count == 11

    serialized = output_path.read_text(
        encoding="utf-8"
    )
    report = EvaluationReport.model_validate_json(
        serialized
    )

    assert report.system_name == "sft_llm"
    assert report.system_metadata == {
        "model_provider": "openai",
        "model_name": "ft:contract-review-model",
        "prompt_version": "contract-risk-v1",
    }
    assert "secret-test-key" not in serialized
    assert "https://model.example/v1" not in serialized

    terminal_output = capsys.readouterr().out
    assert "预计33次模型调用" in terminal_output
    assert str(output_path) in terminal_output