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
        "rules-v1.1-report.json"
    )
    assert cli_module.resolve_output_path(
        llm_arguments
    ) == Path(
        "data/runtime/evaluations/"
        "base-llm-v1.1-report.json"
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


@pytest.mark.parametrize(
    (
        "model_provider",
        "model_name",
        "api_key",
        "base_url",
        "system_name",
    ),
    [
        (
            "openai",
            "ft:contract-review-model",
            "secret-openai-test-key",
            "https://openai.example/v1",
            "sft_llm",
        ),
        (
            "deepseek",
            "deepseek-v4-pro",
            "secret-deepseek-test-key",
            "https://deepseek.example",
            "deepseek_base_llm",
        ),
    ],
)
async def test_llm_cli_uses_config_and_saves_safe_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    model_provider: str,
    model_name: str,
    api_key: str,
    base_url: str,
    system_name: str,
) -> None:
    """验证两家供应商都能完成评测并生成正确的实验报告。"""
    output_path = tmp_path / f"{model_provider}-report.json"

    arguments = create_arguments(
        target="llm",
        system_name=system_name,
        output=output_path,
    )

    # 显式设置两家供应商的字段，隔离机器已有的连接配置。
    model_arguments: dict[str, object] = {
        "_env_file": None,
        "model_provider": model_provider,
        "model_name": model_name,
        "openai_api_key": None,
        "openai_base_url": None,
        "deepseek_api_key": None,
        "deepseek_base_url": "https://api.deepseek.com",
        "model_timeout_seconds": 48.0,
        "model_max_retries": 1,
        "model_max_input_chars": 75_000,
    }

    # 根据当前测试参数填写对应供应商的连接字段。
    model_arguments[f"{model_provider}_api_key"] = api_key
    model_arguments[f"{model_provider}_base_url"] = base_url
    settings = Settings(**model_arguments)

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
        """记录CLI传给对应供应商工厂的全部模型配置。"""
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

    # 当前供应商使用本地模型工厂，记录调用次数而不访问网络。
    monkeypatch.setattr(
        cli_module,
        f"create_{model_provider}_risk_analyzers",
        fake_model_factory,
    )

    def fail_if_other_factory_is_called(
        **kwargs: object,
    ) -> RiskAnalyzerSet:
        """误调用另一家供应商工厂时立即使测试失败。"""
        del kwargs

        raise AssertionError(
            "Unexpected model provider factory"
        )

    other_factory_name = (
        "create_deepseek_risk_analyzers"
        if model_provider == "openai"
        else "create_openai_risk_analyzers"
    )

    monkeypatch.setattr(
        cli_module,
        other_factory_name,
        fail_if_other_factory_is_called,
    )

    # 运行完整评测入口，并将报告保存到测试临时目录。
    await cli_module.run_from_arguments(
        arguments,
        settings=settings,
    )

    assert captured == {
        "model_name": model_name,
        "api_key": api_key,
        "base_url": base_url,
        "timeout_seconds": 48.0,
        "max_retries": 1,
        "max_input_chars": 75_000,
    }

    # 数据集共21个案例，每个案例分别调用三个专业Agent。
    assert commercial.call_count == 21
    assert legal.call_count == 21
    assert security.call_count == 21

    serialized = output_path.read_text(
        encoding="utf-8"
    )
    report = EvaluationReport.model_validate_json(
        serialized
    )

    assert report.system_name == system_name
    assert report.system_metadata["model_provider"] == model_provider
    assert report.system_metadata["model_name"] == model_name
    assert report.system_metadata["prompt_version"] == (
        "contract-risk-v2"
    )
    assert report.system_metadata["validator_version"] == (
        "citation-validator-v2"
    )
    assert len(report.system_metadata["dataset_sha256"]) == 64
    assert len(report.system_metadata["rag_corpus_sha256"]) == 64
    assert report.system_metadata["git_commit"]
    assert report.system_metadata["worktree_dirty"] in {
        "true",
        "false",
    }

    # 报告元数据只记录实验条件，不写入访问凭证和网关地址。
    assert api_key not in serialized
    assert base_url not in serialized

    terminal_output = capsys.readouterr().out
    assert "预计63次模型调用" in terminal_output
    assert str(output_path) in terminal_output