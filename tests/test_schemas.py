"""核心业务数据模型测试。"""

import pytest
from pydantic import ValidationError

from railguard.models.schemas import Clause, ContractDocument, RiskFinding


def test_clause_can_be_located_in_contract() -> None:
    """验证有效条款可以根据字符位置还原到合同原文。"""
    text = "第一条：交付后验收。"

    # 创建一条覆盖全部合同文本的条款。
    clause = Clause(
        text=text,
        start_offset=0,
        end_offset=len(text),
    )

    # 将条款加入合同，触发合同级位置校验。
    contract = ContractDocument(
        filename="demo.docx",
        full_text=text,
        clauses=[clause],
    )

    # 根据保存的位置重新截取合同原文。
    stored = contract.clauses[0]
    original = contract.full_text[
        stored.start_offset:stored.end_offset
    ]

    assert original == stored.text


def test_contract_rejects_incorrect_source_span() -> None:
    """验证条款文本和合同原文不一致时拒绝创建合同。"""
    clause = Clause(
        text="甲方付款",
        start_offset=0,
        end_offset=4,
    )

    # 合同原文是“乙方付款”，与条款记录的“甲方付款”不同。
    with pytest.raises(ValidationError, match="source span"):
        ContractDocument(
            filename="demo.docx",
            full_text="乙方付款",
            clauses=[clause],
        )


def test_missing_clause_does_not_require_a_source_anchor() -> None:
    """验证缺失条款可以没有原文位置，但必须说明预期内容。"""
    finding = RiskFinding(
        finding_kind="missing_clause",
        expected_clause="数据导出与删除条款",
        category="data_exit",
        level="high",
        reason="合同未约定服务终止后的数据导出和删除安排。",
    )

    # 缺失条款没有真实原文，因此不能关联clause_id。
    assert finding.clause_id is None

    # 新风险尚未经过引用验证，初始状态应为pending。
    assert finding.citation_status == "pending"
    assert finding.rejected_evidence_ids == []


def test_existing_clause_risk_requires_a_source_anchor() -> None:
    """验证针对现有条款的风险必须关联真实条款ID。"""
    with pytest.raises(ValidationError, match="requires clause_id"):
        RiskFinding(
            category="payment",
            level="high",
            reason="付款先于验收。",
        )