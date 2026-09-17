"""宁波高端装备制造企业演示资产测试。"""

from pathlib import Path

from railguard.config import Settings
from railguard.models.schemas import (
    ContractDocument,
    Evidence,
)
from railguard.parsers.clauses import split_clauses
from railguard.parsers.documents import parse_document
from railguard.rag.mock import MockRagRetriever
from railguard.rag.models import RagSearchRequest, RagSearchResponse
from railguard.workflow.analyzers import (
    DemoCommercialRiskAnalyzer,
    DemoLegalRiskAnalyzer,
    DemoSecurityRiskAnalyzer,
)

DEMO_CONTRACT_PATH = Path(
    "data/demo/baus-quality-traceability-demo.docx"
)
DEMO_CORPUS_PATH = Path(
    "data/demo/rag-corpus-baus-v1.json"
)


def test_default_mock_corpus_uses_baus_demo() -> None:
    """验证应用默认指向新的高端装备制造演示知识库。"""
    default_path = Settings.model_fields[
        "rag_mock_corpus_path"
    ].default

    assert default_path == DEMO_CORPUS_PATH


def test_baus_demo_contract_is_traceable() -> None:
    """验证新演示合同可以解析并保持条款原文定位。"""
    full_text = parse_document(
        filename=DEMO_CONTRACT_PATH.name,
        content=DEMO_CONTRACT_PATH.read_bytes(),
    )
    clauses = split_clauses(full_text)

    assert "生产质量追溯与试验数据管理平台" in full_text
    assert "ERP、MES和QMS" in full_text
    assert "液压元件试验台" in full_text
    assert [clause.title for clause in clauses] == [
        "合同前言",
        "第一条 项目内容",
        "第二条 付款方式",
        "第三条 验收",
    ]

    for clause in clauses:
        assert (
            full_text[clause.start_offset:clause.end_offset]
            == clause.text
        )


def test_baus_demo_corpus_covers_six_risk_categories() -> None:
    """验证企业演示知识库覆盖合同审核的六类风险。"""
    corpus = RagSearchResponse.model_validate_json(
        DEMO_CORPUS_PATH.read_text(encoding="utf-8")
    )
    categories = {
        hit.metadata["category"]
        for hit in corpus.hits
    }

    assert categories == {
        "payment",
        "acceptance",
        "intellectual_property",
        "liability",
        "support",
        "data_security",
    }
    assert all(
        hit.metadata["document_type"]
        == "synthetic_demo_policy"
        for hit in corpus.hits
    )


async def test_baus_demo_corpus_retrieves_production_data_rule() -> None:
    """验证生产与试验数据场景能检索到数据安全证据。"""
    retriever = MockRagRetriever.from_json_file(
        DEMO_CORPUS_PATH
    )
    evidence = await retriever.retrieve(
        RagSearchRequest(
            query=(
                "供应商远程运维时接触生产数据、工艺参数和账号"
            ),
            top_k=3,
        )
    )

    assert evidence
    assert evidence[0].metadata["category"] == "data_security"
    assert "生产数据处理要求" in evidence[0].source


async def test_baus_demo_contract_produces_six_risk_categories() -> None:
    """验证新的企业合同可稳定演示六类合同风险。"""
    full_text = parse_document(
        filename=DEMO_CONTRACT_PATH.name,
        content=DEMO_CONTRACT_PATH.read_bytes(),
    )
    contract = ContractDocument(
        filename=DEMO_CONTRACT_PATH.name,
        full_text=full_text,
        clauses=split_clauses(full_text),
    )
    retriever = MockRagRetriever.from_json_file(
        DEMO_CORPUS_PATH
    )
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

    contract_evidence = await retriever.retrieve(
        RagSearchRequest(
            query=(
                "付款、验收、知识产权、源代码、数据安全、"
                "远程运维、质保、违约、赔偿和解除"
            ),
            top_k=20,
        )
    )
    analyzers = (
        DemoCommercialRiskAnalyzer(),
        DemoLegalRiskAnalyzer(),
        DemoSecurityRiskAnalyzer(),
    )
    findings = []

    for analyzer in analyzers:
        findings.extend(
            await analyzer.analyze(
                contract=contract,
                evidence_by_clause=evidence_by_clause,
                contract_evidence=contract_evidence,
            )
        )

    assert {
        finding.category
        for finding in findings
    } == {
        "payment",
        "acceptance",
        "intellectual_property",
        "liability",
        "support",
        "data_security",
    }
    assert len(findings) == 6
    assert all(finding.evidence_ids for finding in findings)
