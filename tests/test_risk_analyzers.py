"""三个演示风险Agent的行为测试。"""

from pathlib import Path

from railguard.models.schemas import (
    ContractDocument,
    Evidence,
)
from railguard.parsers.clauses import split_clauses
from railguard.rag.mock import MockRagRetriever
from railguard.rag.models import RagSearchRequest
from railguard.workflow.analyzers import (
    DemoCommercialRiskAnalyzer,
    DemoLegalRiskAnalyzer,
    DemoSecurityRiskAnalyzer,
)


def create_demo_contract() -> ContractDocument:
    """创建包含付款和默认验收风险的演示合同。"""
    full_text = (
        "设备监测平台软件采购合同\n\n"
        "第一条 项目内容\n\n"
        "乙方为甲方建设设备监测平台。\n\n"
        "第二条 付款方式\n\n"
        "合同签订后五日内，甲方支付全部合同款。\n\n"
        "第三条 验收\n\n"
        "系统上线三日后，甲方未提出异议视为验收合格。"
    )

    return ContractDocument(
        filename="software-purchase-demo.docx",
        full_text=full_text,
        clauses=split_clauses(full_text),
    )


def create_complete_contract() -> ContractDocument:
    """创建包含全部关键条款且没有演示风险的合同。"""
    full_text = (
        "设备监测平台软件采购合同\n\n"
        "第一条 付款方式\n\n"
        "项目最终验收合格并经双方书面确认后支付尾款。\n\n"
        "第二条 验收\n\n"
        "双方按照测试标准完成书面验收确认。\n\n"
        "第三条 知识产权\n\n"
        "双方明确源代码和开发成果的知识产权归属。\n\n"
        "第四条 违约责任\n\n"
        "延期交付时乙方承担违约责任和赔偿责任。\n\n"
        "第五条 运维服务\n\n"
        "乙方提供一年质保期和运维服务。\n\n"
        "第六条 数据安全\n\n"
        "乙方承担数据安全和保密义务。"
    )

    return ContractDocument(
            filename="complete-contract.docx",
            full_text=full_text,
            clauses=split_clauses(full_text),
        )


def create_mock_retriever() -> MockRagRetriever:
    """从项目演示知识库创建Mock检索器。"""
    return MockRagRetriever.from_json_file(
        Path("data/demo/rag-corpus.json")
    )


async def retrieve_contract_evidence(
    contract: ContractDocument,
) -> tuple[
    dict[str, list[Evidence]],
    list[Evidence],
]:
    """为每个已有条款和缺失条款检查准备RAG证据。"""
    retriever = create_mock_retriever()
    evidence_by_clause: dict[str, list[Evidence]] = {}

    for clause in contract.clauses:
        evidence_by_clause[clause.clause_id] = (
            await retriever.retrieve(
                RagSearchRequest(
                    clause_id=clause.clause_id,
                    query=clause.text,
                    top_k=3,
                )
            )
        )

    # 合同级查询用于取得关键条款清单证据。
    # 该查询没有真实条款位置，因此不传clause_id。
    contract_evidence = await retriever.retrieve(
        RagSearchRequest(
            query=(
                "付款、验收、知识产权、源代码、数据安全、"
                "运维、质保、违约和解除"
            ),
            top_k=20,
        )
    )

    return evidence_by_clause, contract_evidence


async def test_commercial_agent_detects_payment_and_acceptance() -> None:
    """验证商务Agent发现全额预付和默认验收风险。"""
    contract = create_demo_contract()
    evidence_by_clause, contract_evidence = (
        await retrieve_contract_evidence(contract)
    )

    findings = await DemoCommercialRiskAnalyzer().analyze(
        contract=contract,
        evidence_by_clause=evidence_by_clause,
        contract_evidence=contract_evidence,
    )

    findings_by_category = {
        finding.category: finding
        for finding in findings
    }

    assert set(findings_by_category) == {
        "payment",
        "acceptance",
    }

    payment_clause = next(
        clause
        for clause in contract.clauses
        if clause.title == "第二条 付款方式"
    )
    acceptance_clause = next(
        clause
        for clause in contract.clauses
        if clause.title == "第三条 验收"
    )

    assert (
        findings_by_category["payment"].clause_id
        == payment_clause.clause_id
    )
    assert (
        findings_by_category["acceptance"].clause_id
        == acceptance_clause.clause_id
    )

    assert all(
        finding.finding_kind == "clause_risk"
        for finding in findings
    )
    assert all(
        finding.evidence_ids
        for finding in findings
    )


async def test_legal_agent_detects_missing_key_clauses() -> None:
    """验证法务Agent发现三类关键条款缺失。"""
    contract = create_demo_contract()
    evidence_by_clause, contract_evidence = (
        await retrieve_contract_evidence(contract)
    )

    findings = await DemoLegalRiskAnalyzer().analyze(
        contract=contract,
        evidence_by_clause=evidence_by_clause,
        contract_evidence=contract_evidence,
    )

    assert {
        finding.category
        for finding in findings
    } == {
        "intellectual_property",
        "liability",
        "support",
    }

    assert all(
        finding.finding_kind == "missing_clause"
        for finding in findings
    )
    assert all(
        finding.clause_id is None
        for finding in findings
    )
    assert all(
        finding.expected_clause
        for finding in findings
    )
    assert all(
        finding.evidence_ids
        for finding in findings
    )


async def test_security_agent_detects_missing_data_clause() -> None:
    """验证安全Agent发现数据安全条款缺失。"""
    contract = create_demo_contract()
    evidence_by_clause, contract_evidence = (
        await retrieve_contract_evidence(contract)
    )

    findings = await DemoSecurityRiskAnalyzer().analyze(
        contract=contract,
        evidence_by_clause=evidence_by_clause,
        contract_evidence=contract_evidence,
    )

    assert len(findings) == 1
    assert findings[0].category == "data_security"
    assert findings[0].finding_kind == "missing_clause"
    assert findings[0].clause_id is None
    assert findings[0].evidence_ids


async def test_complete_contract_has_no_demo_rule_findings() -> None:
    """验证内容完整的合同不会触发当前演示规则。"""
    contract = create_complete_contract()

    commercial_findings = (
        await DemoCommercialRiskAnalyzer().analyze(
            contract=contract,
            evidence_by_clause={},
            contract_evidence=[],
        )
    )
    legal_findings = await DemoLegalRiskAnalyzer().analyze(
        contract=contract,
        evidence_by_clause={},
        contract_evidence=[],
    )
    security_findings = (
        await DemoSecurityRiskAnalyzer().analyze(
            contract=contract,
            evidence_by_clause={},
            contract_evidence=[],
        )
    )

    assert commercial_findings == []
    assert legal_findings == []
    assert security_findings == []