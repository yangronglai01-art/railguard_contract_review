"""大模型Agent与精确率护栏的集成测试。"""

from collections.abc import Sequence

from langchain_core.messages import BaseMessage

from railguard.agents.llm import LlmRiskAnalyzer
from railguard.agents.prompts import COMMERCIAL_PROFILE, LEGAL_PROFILE
from railguard.models.schemas import ContractDocument
from railguard.parsers.clauses import split_clauses


class FixedInvoker:
    """返回固定结构化候选项的测试调用器。"""

    def __init__(self, response: object) -> None:
        """保存下一次调用需要返回的固定对象。"""
        self._response = response

    async def ainvoke(
        self,
        messages: Sequence[BaseMessage],
    ) -> object:
        """忽略消息内容并返回固定对象。"""
        del messages
        return self._response


def build_contract(body: str) -> ContractDocument:
    """根据给定条款正文创建测试合同。"""
    full_text = f"高端装备软件采购合同\n\n{body}"
    return ContractDocument(
        filename="precision-integration.docx",
        full_text=full_text,
        clauses=split_clauses(full_text),
    )


def first_business_clause(contract: ContractDocument):
    """返回合同前言后的第一项业务条款。"""
    return next(
        clause
        for clause in contract.clauses
        if clause.title != "合同前言"
    )


def candidate(
    *,
    clause_id: str,
    category: str,
    level: str = "medium",
) -> dict[str, object]:
    """创建不引用外部证据的固定模型候选项。"""
    return {
        "finding_kind": "clause_risk",
        "clause_id": clause_id,
        "expected_clause": None,
        "category": category,
        "level": level,
        "reason": "该条款可能使采购方承担不可执行风险。",
        "suggested_revision": "建议补充明确且可执行的保护安排。",
        "evidence_ids": [],
    }


async def test_analyzer_filters_best_practice_false_positive() -> None:
    """验证模型对合格付款条款的过度建议不会进入业务结果。"""
    contract = build_contract(
        "第一条 付款方式\n\n"
        "签订后支付10%，阶段交付支付40%，"
        "最终验收支付40%，质保期满支付10%。"
    )
    clause = first_business_clause(contract)
    analyzer = LlmRiskAnalyzer(
        profile=COMMERCIAL_PROFILE,
        invoker=FixedInvoker(
            {
                "findings": [
                    candidate(
                        clause_id=clause.clause_id,
                        category="payment",
                    )
                ]
            }
        ),
    )

    findings = await analyzer.analyze(
        contract=contract,
        evidence_by_clause={},
        contract_evidence=[],
    )

    assert findings == []


async def test_analyzer_keeps_and_calibrates_high_prepayment() -> None:
    """验证高比例预付风险被保留并校准为高风险。"""
    contract = build_contract(
        "第一条 付款方式\n\n合同签订后支付合同总价的90%。"
    )
    clause = first_business_clause(contract)
    analyzer = LlmRiskAnalyzer(
        profile=COMMERCIAL_PROFILE,
        invoker=FixedInvoker(
            {
                "findings": [
                    candidate(
                        clause_id=clause.clause_id,
                        category="payment",
                        level="low",
                    )
                ]
            }
        ),
    )

    findings = await analyzer.analyze(
        contract=contract,
        evidence_by_clause={},
        contract_evidence=[],
    )

    assert len(findings) == 1
    assert findings[0].category == "payment"
    assert findings[0].level == "high"


async def test_analyzer_calibrates_limited_support_gap() -> None:
    """验证已有质保但响应时限待约定时校准为低风险。"""
    contract = build_contract(
        "第一条 运维服务\n\n"
        "乙方提供一年质保期，故障响应和修复时间另行约定。"
    )
    clause = first_business_clause(contract)
    analyzer = LlmRiskAnalyzer(
        profile=LEGAL_PROFILE,
        invoker=FixedInvoker(
            {
                "findings": [
                    candidate(
                        clause_id=clause.clause_id,
                        category="support",
                        level="high",
                    )
                ]
            }
        ),
    )

    findings = await analyzer.analyze(
        contract=contract,
        evidence_by_clause={},
        contract_evidence=[],
    )

    assert len(findings) == 1
    assert findings[0].level == "low"
