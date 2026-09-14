"""大模型异常经过工作流服务和HTTP接口的映射测试。"""

from collections.abc import (
    Mapping,
    Sequence,
)
from pathlib import Path

import pytest
from fastapi import HTTPException
from langgraph.checkpoint.memory import InMemorySaver

from railguard.agents.llm import (
    LlmAnalyzerError,
    LlmInputTooLargeError,
    LlmInvocationError,
    LlmProtocolError,
    LlmResponseError,
)
from railguard.api.reviews import start_contract_review
from railguard.models.schemas import (
    ContractDocument,
    Evidence,
    RiskFinding,
)
from railguard.rag.models import RagSearchRequest
from railguard.storage.contracts import ContractRepository
from railguard.workflow.graph import build_review_graph
from railguard.workflow.results import ReviewSnapshot
from railguard.workflow.service import (
    ReviewExecutionError,
    ReviewService,
)


class EmptyRetriever:
    """始终返回空证据的本地RAG测试检索器。"""

    async def retrieve(
        self,
        request: RagSearchRequest,
    ) -> list[Evidence]:
        """忽略查询并返回合法的空检索结果。"""
        del request

        return []


class EmptyAnalyzer:
    """始终返回空风险列表的测试Agent。"""

    def __init__(
        self,
        name: str,
    ) -> None:
        """保存工作流需要记录的Agent名称。"""
        self.name = name

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
        """忽略审核输入并返回空风险列表。"""
        del contract
        del evidence_by_clause
        del contract_evidence

        return []


class FailingAnalyzer:
    """每次分析都抛出指定大模型异常的测试Agent。"""

    name = "commercial_risk_agent"

    def __init__(
        self,
        error: LlmAnalyzerError,
    ) -> None:
        """保存本次测试需要抛出的异常。"""
        self._error = error

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
        """模拟模型Agent在工作流节点内执行失败。"""
        del contract
        del evidence_by_clause
        del contract_evidence

        raise self._error


class FailingReviewService:
    """把固定ReviewExecutionError抛给API的测试服务。"""

    def __init__(
        self,
        error: ReviewExecutionError,
    ) -> None:
        """保存API调用时需要抛出的服务异常。"""
        self._error = error

    async def start_review(
        self,
        contract: ContractDocument,
    ) -> ReviewSnapshot:
        """模拟审核服务启动失败。"""
        del contract

        raise self._error


def create_contract() -> ContractDocument:
    """创建无需解析文件的最小测试合同。"""
    return ContractDocument(
        contract_id="contract-model-error-test",
        filename="model-error-test.docx",
        full_text="设备监测平台软件采购合同",
    )


def create_review_service(
    error: LlmAnalyzerError,
) -> ReviewService:
    """创建只有商务Agent失败的内存工作流服务。"""
    graph = build_review_graph(
        retriever=EmptyRetriever(),
        commercial_analyzer=FailingAnalyzer(error),
        legal_analyzer=EmptyAnalyzer(
            "legal_risk_agent"
        ),
        security_analyzer=EmptyAnalyzer(
            "security_risk_agent"
        ),
        checkpointer=InMemorySaver(),
    )

    return ReviewService(graph)


@pytest.mark.parametrize(
    (
        "error_type",
        "expected_code",
        "expected_retryable",
        "expected_status",
    ),
    [
        (
            LlmInputTooLargeError,
            "model_input_too_large",
            False,
            413,
        ),
        (
            LlmInvocationError,
            "model_unavailable",
            True,
            503,
        ),
        (
            LlmResponseError,
            "model_response_invalid",
            True,
            502,
        ),
        (
            LlmProtocolError,
            "model_protocol_error",
            False,
            502,
        ),
    ],
)
async def test_model_errors_have_stable_service_and_http_codes(
    tmp_path: Path,
    error_type: type[LlmAnalyzerError],
    expected_code: str,
    expected_retryable: bool,
    expected_status: int,
) -> None:
    """验证模型异常经过服务和API后仍保留稳定语义。"""
    service = create_review_service(
        error_type("simulated model failure")
    )
    contract = create_contract()

    with pytest.raises(
        ReviewExecutionError
    ) as service_error:
        await service.start_review(contract)

    mapped_error = service_error.value

    assert mapped_error.review_id
    assert mapped_error.code == expected_code
    assert mapped_error.retryable is expected_retryable

    # 使用真实合同仓储调用路由函数，验证HTTP状态和响应结构。
    repository = ContractRepository(
        tmp_path / "contracts.db"
    )
    repository.initialize()
    repository.save(contract)
    api_service = FailingReviewService(mapped_error)

    with pytest.raises(HTTPException) as http_error:
        await start_contract_review(
            contract.contract_id,
            repository,
            api_service,
        )

    assert http_error.value.status_code == expected_status
    assert http_error.value.detail == {
        "review_id": mapped_error.review_id,
        "code": expected_code,
        "message": str(mapped_error),
        "retryable": expected_retryable,
    }