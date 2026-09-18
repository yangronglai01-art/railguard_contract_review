"""精确率护栏的高端装备制造场景挑战测试。"""

import pytest

from railguard.agents.guardrails import risk_candidate_is_eligible
from railguard.models.schemas import ContractDocument
from railguard.parsers.clauses import split_clauses


def build_contract(body: str) -> ContractDocument:
    """使用固定标题和给定正文构造一份挑战合同。"""
    full_text = f"高端装备采购与服务合同\n\n{body}"
    return ContractDocument(
        filename="guardrail-challenge.docx",
        full_text=full_text,
        clauses=split_clauses(full_text),
    )


def first_business_clause(contract: ContractDocument):
    """返回合同前言之后的第一个业务条款。"""
    return next(
        clause
        for clause in contract.clauses
        if clause.title != "合同前言"
    )


@pytest.mark.parametrize(
    ("category", "body"),
    [
        (
            "payment",
            "第一条 付款方式\n\n合同签订后支付合同总价的90%。",
        ),
        (
            "payment",
            "第一条 付款方式\n\n付款方式由双方后续确定。",
        ),
        (
            "acceptance",
            "第一条 验收\n\n甲方逾期未反馈则验收通过。",
        ),
        (
            "intellectual_property",
            "第一条 知识产权\n\n开发成果权利归属后续另议。",
        ),
        (
            "liability",
            "第一条 违约责任\n\n违约事项依法处理。",
        ),
        (
            "support",
            "第一条 运维服务\n\n服务标准后续确定。",
        ),
        (
            "data_security",
            "第一条 数据安全\n\n数据保护措施由双方另议。",
        ),
    ],
)
def test_clear_clause_risks_pass_guardrail(
    category: str,
    body: str,
) -> None:
    """验证明确高风险和不可执行条款能够通过护栏。"""
    contract = build_contract(body)
    clause = first_business_clause(contract)

    assert risk_candidate_is_eligible(
        contract=contract,
        finding_kind="clause_risk",
        clause_id=clause.clause_id,
        category=category,
    )


@pytest.mark.parametrize(
    ("category", "body"),
    [
        (
            "payment",
            "第一条 付款方式\n\n最终验收合格后支付全部合同款。",
        ),
        (
            "acceptance",
            "第一条 验收\n\n不得以沉默视为验收合格。",
        ),
        (
            "intellectual_property",
            "第一条 知识产权\n\n定制开发成果归甲方所有。",
        ),
        (
            "liability",
            "第一条 违约责任\n\n乙方承担赔偿责任，甲方有权解除合同。",
        ),
        (
            "support",
            "第一条 运维服务\n\n提供一年质保并明确响应和修复时间。",
        ),
        (
            "data_security",
            "第一条 数据安全\n\n乙方承担保密义务并在结束后删除数据。",
        ),
    ],
)
def test_baseline_protections_do_not_pass_guardrail(
    category: str,
    body: str,
) -> None:
    """验证达到最低合格线的条款不会因缺少理想细节被误报。"""
    contract = build_contract(body)
    clause = first_business_clause(contract)

    assert not risk_candidate_is_eligible(
        contract=contract,
        finding_kind="clause_risk",
        clause_id=clause.clause_id,
        category=category,
    )


def test_physical_equipment_contract_does_not_require_data_clause() -> None:
    """验证不处理数据的纯机械设备采购不强制要求数据条款。"""
    contract = build_contract(
        "第一条 标的\n\n乙方向甲方交付十台液压泵。"
    )

    assert not risk_candidate_is_eligible(
        contract=contract,
        finding_kind="missing_clause",
        clause_id=None,
        category="data_security",
    )


def test_remote_maintenance_contract_requires_data_clause() -> None:
    """验证远程运维合同缺少数据保护时允许输出缺失风险。"""
    contract = build_contract(
        "第一条 服务\n\n乙方远程维护设备控制系统。"
    )

    assert risk_candidate_is_eligible(
        contract=contract,
        finding_kind="missing_clause",
        clause_id=None,
        category="data_security",
    )
