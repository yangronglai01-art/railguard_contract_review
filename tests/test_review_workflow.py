"""完整LangGraph合同审核流程集成测试。"""

from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from railguard.models.schemas import (
    ContractDocument,
    Evidence,
    RiskFinding,
)
from railguard.parsers.clauses import split_clauses
from railguard.rag.client import RagUnavailableError
from railguard.rag.mock import MockRagRetriever
from railguard.rag.models import RagSearchRequest
from railguard.workflow.graph import build_review_graph
from railguard.workflow.nodes import AnalyzerProtocolError
from railguard.workflow.state import HumanReviewDecision


def create_contract(
    *,
    filename: str,
    full_text: str,
) -> ContractDocument:
    """根据合同文本创建带有可追溯条款的合同模型。"""
    return ContractDocument(
        filename=filename,
        full_text=full_text,
        clauses=split_clauses(full_text),
    )


def create_demo_contract() -> ContractDocument:
    """创建会触发六项演示风险的合同。"""
    full_text = (
        "设备监测平台软件采购合同\n\n"
        "第一条 项目内容\n\n"
        "乙方为甲方建设设备监测平台。\n\n"
        "第二条 付款方式\n\n"
        "合同签订后五日内，甲方支付全部合同款。\n\n"
        "第三条 验收\n\n"
        "系统上线三日后，甲方未提出异议视为验收合格。"
    )

    return create_contract(
        filename="software-purchase-demo.docx",
        full_text=full_text,
    )


def create_complete_contract() -> ContractDocument:
    """创建不会触发当前演示规则的完整合同。"""
    full_text = (
        "设备监测平台软件采购合同\n\n"
        "第一条 付款方式\n\n"
        "最终验收并经双方书面确认后支付尾款。\n\n"
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

    return create_contract(
        filename="complete-contract.docx",
        full_text=full_text,
    )


def create_retriever() -> MockRagRetriever:
    """从演示知识库创建确定性RAG检索器。"""
    return MockRagRetriever.from_json_file(
        Path("data/demo/rag-corpus.json")
    )


class FailingRetriever:
    """始终模拟外部RAG连接失败的测试检索器。"""

    async def retrieve(
        self,
        request: RagSearchRequest,
    ) -> list[Evidence]:
        """抛出服务不可用异常，验证工作流不会吞掉故障。"""
        del request

        raise RagUnavailableError(
            "mock RAG service is unavailable"
        )


class InvalidClauseAnalyzer:
    """返回不存在条款ID的测试风险Agent。"""

    name = "invalid_clause_agent"

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
        """生成一条带有伪造条款ID的风险。"""
        del contract
        del evidence_by_clause
        del contract_evidence

        return [
            RiskFinding(
                finding_kind="clause_risk",
                clause_id="unknown-clause",
                category="payment",
                level="high",
                reason="测试Agent返回了不存在的条款ID。",
            )
        ]


class PartiallyInvalidCitationAnalyzer:
    """同时返回合法和越权证据ID的测试风险Agent。"""

    name = "partially_invalid_citation_agent"

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
        """构造一条部分匹配引用，验证审计信息不会丢失。"""
        del contract_evidence

        payment_clause = next(
            clause
            for clause in contract.clauses
            if "付款" in clause.title
        )
        valid_evidence = next(
            evidence
            for evidence in evidence_by_clause[
                payment_clause.clause_id
            ]
            if evidence.metadata.get("category") == "payment"
        )

        return [
            RiskFinding(
                finding_kind="clause_risk",
                clause_id=payment_clause.clause_id,
                category="payment",
                level="high",
                reason="付款安排缺少与验收结果的约束。",
                evidence_ids=[
                    valid_evidence.evidence_id,
                    "hallucinated-evidence-id",
                ],
            )
        ]


async def test_complete_contract_finishes_without_interrupt() -> None:
    """验证没有风险时跳过人工审核并直接完成。"""
    graph = build_review_graph(
        retriever=create_retriever()
    )

    result = await graph.ainvoke(
        {
            "review_id": "review-complete",
            "contract": create_complete_contract(),
        }
    )

    assert result["status"] == "approved"
    assert result["findings"] == []
    assert "Agent共识别0项风险" in result["final_summary"]
    assert "__interrupt__" not in result

    # 三个专业Agent都必须实际执行。
    assert {
        "commercial_risk_agent",
        "legal_risk_agent",
        "security_risk_agent",
    }.issubset(set(result["completed_nodes"]))


async def test_risky_contract_interrupts_and_resumes() -> None:
    """验证风险审核可以暂停并使用同一thread_id恢复。"""
    checkpointer = InMemorySaver()
    graph = build_review_graph(
        retriever=create_retriever(),
        checkpointer=checkpointer,
    )
    config = {
        "configurable": {
            "thread_id": "review-risky",
        }
    }

    interrupted_result = await graph.ainvoke(
        {
            "review_id": "review-risky",
            "contract": create_demo_contract(),
        },
        config=config,
    )

    assert interrupted_result["status"] == "awaiting_human"
    assert len(interrupted_result["findings"]) == 6
    assert all(
        finding.citation_status == "source_matched"
        for finding in interrupted_result["findings"]
    )

    interrupts = interrupted_result["__interrupt__"]
    assert len(interrupts) == 1

    review_payload = interrupts[0].value
    assert review_payload["review_id"] == "review-risky"
    assert len(review_payload["findings"]) == 6
    assert review_payload["evidence"]

    # 模拟人工驳回第一项风险，其余风险保持Agent判断。
    dismissed_finding_id = (
        interrupted_result["findings"][0].finding_id
    )
    resumed_result = await graph.ainvoke(
        Command(
            resume={
                "action": "approve",
                "reviewer": "法务审核员",
                "finding_decisions": {
                    dismissed_finding_id: "dismiss"
                },
                "comment": "已核对并驳回第一项风险。",
            }
        ),
        config=config,
    )

    decision = HumanReviewDecision.model_validate(
        resumed_result["human_decision"]
    )

    assert resumed_result["status"] == "approved"
    assert decision.reviewer == "法务审核员"
    assert "最终保留5项" in resumed_result["final_summary"]
    assert "__interrupt__" not in resumed_result


async def test_rag_failure_is_not_converted_to_empty_evidence() -> None:
    """验证RAG技术故障会终止流程并向调用方抛出。"""
    graph = build_review_graph(
        retriever=FailingRetriever()
    )

    with pytest.raises(RagUnavailableError):
        await graph.ainvoke(
            {
                "review_id": "review-rag-failure",
                "contract": create_demo_contract(),
            }
        )


async def test_analyzer_cannot_reference_unknown_clause() -> None:
    """验证风险Agent不能伪造合同条款位置。"""
    graph = build_review_graph(
        retriever=create_retriever(),
        commercial_analyzer=InvalidClauseAnalyzer(),
    )

    with pytest.raises(
        AnalyzerProtocolError,
        match="unknown clause_id",
    ):
        await graph.ainvoke(
            {
                "review_id": "review-invalid-clause",
                "contract": create_demo_contract(),
            }
        )


async def test_partial_citation_match_preserves_rejected_ids() -> None:
    """验证合法引用可继续使用，同时记录越权引用供人工审计。"""
    graph = build_review_graph(
        retriever=create_retriever(),
        commercial_analyzer=PartiallyInvalidCitationAnalyzer(),
        checkpointer=InMemorySaver(),
    )

    result = await graph.ainvoke(
        {
            "review_id": "review-partial-citation",
            "contract": create_demo_contract(),
        },
        config={
            "configurable": {
                "thread_id": "review-partial-citation",
            }
        },
    )

    finding = next(
        finding
        for finding in result["findings"]
        if finding.reason == "付款安排缺少与验收结果的约束。"
    )

    assert finding.citation_status == "partially_matched"
    assert len(finding.evidence_ids) == 1
    assert finding.rejected_evidence_ids == [
        "hallucinated-evidence-id"
    ]
