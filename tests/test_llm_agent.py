"""结构化大模型风险Agent的本地协议边界测试。"""

from collections.abc import Sequence

import pytest
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

from railguard.agents.llm import (
    LlmInputTooLargeError,
    LlmInvocationError,
    LlmProtocolError,
    LlmResponseError,
    LlmRiskAnalyzer,
)
from railguard.agents.prompts import COMMERCIAL_PROFILE
from railguard.models.schemas import (
    Clause,
    ContractDocument,
    Evidence,
)
from railguard.parsers.clauses import split_clauses


class FakeStructuredInvoker:
    """返回固定结构化结果的测试模型调用器。"""

    def __init__(
        self,
        response: object | Exception,
    ) -> None:
        """保存固定响应并初始化消息记录。"""
        self._response = response
        self.messages: list[BaseMessage] = []

    async def ainvoke(
        self,
        messages: Sequence[BaseMessage],
    ) -> object:
        """记录输入消息，并返回结果或抛出指定异常。"""
        self.messages = list(messages)

        if isinstance(self._response, Exception):
            raise self._response

        return self._response


def create_contract() -> ContractDocument:
    """创建包含付款条款的固定测试合同。"""
    full_text = (
        "设备监测平台软件采购合同\n\n"
        "第一条 付款方式\n\n"
        "合同签订后五日内，甲方支付全部合同款。\n\n"
        "第二条 验收方式\n\n"
        "双方按照测试标准完成书面验收。"
    )

    return ContractDocument(
        contract_id="contract-llm-test",
        filename="llm-test.docx",
        full_text=full_text,
        clauses=split_clauses(full_text),
    )


def find_payment_clause(
    contract: ContractDocument,
) -> Clause:
    """从测试合同中返回付款条款。"""
    return next(
        clause
        for clause in contract.clauses
        if clause.title.startswith("第一条")
    )


def create_evidence(
    *,
    evidence_id: str = "evidence-payment-01",
    category: str = "payment",
) -> Evidence:
    """创建分类和ID可配置的测试证据。"""
    return Evidence(
        evidence_id=evidence_id,
        document_id=f"document-{evidence_id}",
        title="软件采购审核规则",
        content="付款应与交付和验收节点相匹配。",
        source="第三条",
        score=0.94,
        metadata={
            "category": category,
            "version": "2026-01",
        },
    )


def create_clause_candidate(
    *,
    clause_id: str,
    evidence_ids: list[str] | None = None,
    category: str = "payment",
) -> dict[str, object]:
    """创建结构正确的已有条款风险候选项。"""
    return {
        "finding_kind": "clause_risk",
        "clause_id": clause_id,
        "expected_clause": None,
        "category": category,
        "level": "high",
        "reason": (
            "  全额预付使采购方失去验收前的付款制约手段。  "
        ),
        "suggested_revision": (
            "  建议设置分期付款并保留验收后的尾款。  "
        ),
        "evidence_ids": (
            evidence_ids
            if evidence_ids is not None
            else ["evidence-payment-01"]
        ),
    }


async def test_analyzer_converts_valid_structured_response() -> None:
    """验证合法模型响应转换为应用生成的RiskFinding。"""
    contract = create_contract()
    payment_clause = find_payment_clause(contract)
    evidence = create_evidence()
    invoker = FakeStructuredInvoker(
        {
            "findings": [
                create_clause_candidate(
                    clause_id=payment_clause.clause_id,
                )
            ]
        }
    )
    analyzer = LlmRiskAnalyzer(
        profile=COMMERCIAL_PROFILE,
        invoker=invoker,
    )

    findings = await analyzer.analyze(
        contract=contract,
        evidence_by_clause={
            payment_clause.clause_id: [evidence],
        },
        contract_evidence=[],
    )

    assert analyzer.name == "commercial_risk_agent"
    assert len(findings) == 1

    finding = findings[0]
    assert len(finding.finding_id) == 32
    assert finding.finding_kind == "clause_risk"
    assert finding.clause_id == payment_clause.clause_id
    assert finding.category == "payment"
    assert finding.level == "high"
    assert finding.reason == (
        "全额预付使采购方失去验收前的付款制约手段。"
    )
    assert finding.suggested_revision == (
        "建议设置分期付款并保留验收后的尾款。"
    )
    assert finding.evidence_ids == ["evidence-payment-01"]
    assert finding.citation_status == "pending"

    # 模型调用必须包含独立的系统消息和用户消息。
    assert len(invoker.messages) == 2
    assert isinstance(invoker.messages[0], SystemMessage)
    assert isinstance(invoker.messages[1], HumanMessage)


async def test_analyzer_accepts_empty_findings() -> None:
    """验证模型确认无风险时可以返回空列表。"""
    contract = create_contract()
    invoker = FakeStructuredInvoker({"findings": []})
    analyzer = LlmRiskAnalyzer(
        profile=COMMERCIAL_PROFILE,
        invoker=invoker,
    )

    findings = await analyzer.analyze(
        contract=contract,
        evidence_by_clause={},
        contract_evidence=[],
    )

    assert findings == []


async def test_analyzer_maps_model_invocation_failure() -> None:
    """验证模型SDK异常被转换为稳定的应用异常。"""
    contract = create_contract()
    provider_error = RuntimeError("provider unavailable")
    invoker = FakeStructuredInvoker(provider_error)
    analyzer = LlmRiskAnalyzer(
        profile=COMMERCIAL_PROFILE,
        invoker=invoker,
    )

    with pytest.raises(LlmInvocationError) as error:
        await analyzer.analyze(
            contract=contract,
            evidence_by_clause={},
            contract_evidence=[],
        )

    assert error.value.__cause__ is provider_error


async def test_analyzer_rejects_invalid_structured_response() -> None:
    """验证缺少findings字段的模型响应不能进入工作流。"""
    contract = create_contract()
    invoker = FakeStructuredInvoker(
        {"unexpected": "invalid response"}
    )
    analyzer = LlmRiskAnalyzer(
        profile=COMMERCIAL_PROFILE,
        invoker=invoker,
    )

    with pytest.raises(LlmResponseError):
        await analyzer.analyze(
            contract=contract,
            evidence_by_clause={},
            contract_evidence=[],
        )


async def test_analyzer_rejects_oversized_input_before_call() -> None:
    """验证超长输入在调用外部模型之前被拒绝。"""
    contract = create_contract()
    invoker = FakeStructuredInvoker({"findings": []})
    analyzer = LlmRiskAnalyzer(
        profile=COMMERCIAL_PROFILE,
        invoker=invoker,
        max_input_chars=1,
    )

    with pytest.raises(LlmInputTooLargeError):
        await analyzer.analyze(
            contract=contract,
            evidence_by_clause={},
            contract_evidence=[],
        )

    # 本地长度检查失败后，不应产生任何模型调用。
    assert invoker.messages == []


async def test_analyzer_rejects_category_outside_profile() -> None:
    """验证商务Agent不能越权输出数据安全风险。"""
    contract = create_contract()
    payment_clause = find_payment_clause(contract)
    invoker = FakeStructuredInvoker(
        {
            "findings": [
                create_clause_candidate(
                    clause_id=payment_clause.clause_id,
                    evidence_ids=[],
                    category="data_security",
                )
            ]
        }
    )
    analyzer = LlmRiskAnalyzer(
        profile=COMMERCIAL_PROFILE,
        invoker=invoker,
    )

    with pytest.raises(
        LlmProtocolError,
        match="outside the agent scope",
    ):
        await analyzer.analyze(
            contract=contract,
            evidence_by_clause={},
            contract_evidence=[],
        )


async def test_analyzer_rejects_unknown_clause_id() -> None:
    """验证模型不能伪造合同中不存在的条款ID。"""
    contract = create_contract()
    invoker = FakeStructuredInvoker(
        {
            "findings": [
                create_clause_candidate(
                    clause_id="fabricated-clause-id",
                    evidence_ids=[],
                )
            ]
        }
    )
    analyzer = LlmRiskAnalyzer(
        profile=COMMERCIAL_PROFILE,
        invoker=invoker,
    )

    with pytest.raises(
        LlmProtocolError,
        match="unknown clause_id",
    ):
        await analyzer.analyze(
            contract=contract,
            evidence_by_clause={},
            contract_evidence=[],
        )


async def test_analyzer_accepts_contract_evidence_for_clause_risk() -> None:
    """验证已有条款风险可以引用合同级同类审查指引。"""
    contract = create_contract()
    payment_clause = find_payment_clause(contract)
    contract_evidence = create_evidence()
    invoker = FakeStructuredInvoker(
        {
            "findings": [
                create_clause_candidate(
                    clause_id=payment_clause.clause_id,
                )
            ]
        }
    )
    analyzer = LlmRiskAnalyzer(
        profile=COMMERCIAL_PROFILE,
        invoker=invoker,
    )

    findings = await analyzer.analyze(
        contract=contract,
        evidence_by_clause={},
        contract_evidence=[contract_evidence],
    )

    assert len(findings) == 1
    assert findings[0].evidence_ids == ["evidence-payment-01"]


async def test_analyzer_filters_evidence_from_another_category() -> None:
    """验证付款风险会过滤掉验收分类的证据引用。"""
    contract = create_contract()
    payment_clause = find_payment_clause(contract)
    acceptance_evidence = create_evidence(
        category="acceptance",
    )
    invoker = FakeStructuredInvoker(
        {
            "findings": [
                create_clause_candidate(
                    clause_id=payment_clause.clause_id,
                )
            ]
        }
    )
    analyzer = LlmRiskAnalyzer(
        profile=COMMERCIAL_PROFILE,
        invoker=invoker,
    )

    findings = await analyzer.analyze(
        contract=contract,
        evidence_by_clause={
            payment_clause.clause_id: [
                acceptance_evidence
            ],
        },
        contract_evidence=[],
    )

    assert len(findings) == 1
    assert findings[0].evidence_ids == []


async def test_analyzer_rejects_duplicate_findings() -> None:
    """验证同一Agent不能重复返回相同风险。"""
    contract = create_contract()
    payment_clause = find_payment_clause(contract)
    candidate = create_clause_candidate(
        clause_id=payment_clause.clause_id,
        evidence_ids=[],
    )
    invoker = FakeStructuredInvoker(
        {
            "findings": [
                candidate,
                candidate,
            ]
        }
    )
    analyzer = LlmRiskAnalyzer(
        profile=COMMERCIAL_PROFILE,
        invoker=invoker,
    )

    with pytest.raises(
        LlmProtocolError,
        match="duplicate risk findings",
    ):
        await analyzer.analyze(
            contract=contract,
            evidence_by_clause={},
            contract_evidence=[],
        )