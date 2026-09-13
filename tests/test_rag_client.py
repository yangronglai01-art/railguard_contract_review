"""RAG请求模型和HTTP适配器测试。

测试使用httpx.MockTransport拦截请求，不会访问真实网络。
这样既能验证HTTP协议，也能保证测试稳定、可重复。
"""

import json
from collections.abc import Callable

import httpx
import pytest
from pydantic import ValidationError

from railguard.rag.client import (
    HttpRagRetriever,
    RagError,
    RagProtocolError,
    RagRequestError,
    RagTimeoutError,
    RagUnavailableError,
)
from railguard.rag.models import RagSearchRequest


def create_transport(
    handler: Callable[[httpx.Request], httpx.Response],
) -> httpx.MockTransport:
    """根据测试处理函数创建httpx模拟传输层。"""
    return httpx.MockTransport(handler)


def successful_hits() -> list[dict[str, object]]:
    """创建两条固定的RAG检索结果。"""
    return [
        {
            "document_id": "policy-payment-01",
            "title": "采购付款管理办法",
            "content": "大额软件采购付款应与验收节点相匹配。",
            "source": "第三条",
            "score": 0.92,
            "metadata": {
                "category": "payment",
                "version": "2026-01",
            },
        },
        {
            "document_id": "policy-acceptance-02",
            "title": "信息系统项目验收规范",
            "content": "项目验收应设置明确的测试标准和整改期限。",
            "source": "第五条",
            "score": 0.81,
            "metadata": {
                "category": "acceptance",
                "version": "2026-01",
            },
        },
    ]


def test_search_request_validates_query_and_top_k() -> None:
    """验证检索内容不能为空，并限制最大返回数量。"""
    request = RagSearchRequest(
        query="检查合同是否缺少知识产权条款",
        top_k=5,
    )

    # 缺失条款检索没有真实的条款ID，因此允许为None。
    assert request.clause_id is None

    with pytest.raises(ValidationError):
        RagSearchRequest(query="   ")

    with pytest.raises(ValidationError):
        RagSearchRequest(query="付款条款", top_k=21)


async def test_http_retriever_sends_request_and_maps_evidence() -> None:
    """验证HTTP请求结构、结果顺序和Evidence字段转换。"""

    def handler(request: httpx.Request) -> httpx.Response:
        """检查客户端发出的请求并返回固定检索结果。"""
        body = json.loads(request.content.decode("utf-8"))

        assert request.method == "POST"
        assert request.url.path == "/search"
        assert body == {
            "clause_id": "clause-payment",
            "query": "合同签订后支付全部合同款。",
            "top_k": 1,
        }

        return httpx.Response(
            200,
            json={"hits": successful_hits()},
        )

    transport = create_transport(handler)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://rag.example/",
    ) as client:
        retriever = HttpRagRetriever(client)
        evidence = await retriever.retrieve(
            RagSearchRequest(
                clause_id="clause-payment",
                query="合同签订后支付全部合同款。",
                top_k=1,
            )
        )

    # 服务返回两条，但客户端应按照top_k只保留第一条。
    assert len(evidence) == 1
    assert evidence[0].document_id == "policy-payment-01"
    assert evidence[0].title == "采购付款管理办法"
    assert evidence[0].source == "第三条"
    assert evidence[0].score == 0.92
    assert evidence[0].metadata["category"] == "payment"

    # evidence_id由RailGuard生成，不依赖外部RAG服务。
    assert len(evidence[0].evidence_id) == 32


async def test_http_retriever_returns_empty_list_for_empty_hits() -> None:
    """验证合法的空检索结果返回空列表。"""

    def handler(request: httpx.Request) -> httpx.Response:
        """返回格式正确但没有命中项的响应。"""
        return httpx.Response(200, json={"hits": []})

    transport = create_transport(handler)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://rag.example/",
    ) as client:
        retriever = HttpRagRetriever(client)
        evidence = await retriever.retrieve(
            RagSearchRequest(query="没有相关资料", top_k=3)
        )

    assert evidence == []


@pytest.mark.parametrize(
    ("status_code", "expected_exception"),
    [
        (401, RagRequestError),
        (503, RagUnavailableError),
        (302, RagProtocolError),
    ],
)
async def test_http_retriever_maps_status_errors(
    status_code: int,
    expected_exception: type[RagError],
) -> None:
    """验证不同HTTP状态被转换为明确的RAG异常。"""

    def handler(request: httpx.Request) -> httpx.Response:
        """返回参数指定的异常HTTP状态。"""
        return httpx.Response(status_code, json={"detail": "error"})

    transport = create_transport(handler)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://rag.example/",
    ) as client:
        retriever = HttpRagRetriever(client)

        with pytest.raises(expected_exception):
            await retriever.retrieve(
                RagSearchRequest(query="测试异常状态")
            )


async def test_http_retriever_rejects_invalid_response() -> None:
    """验证非JSON或字段缺失的响应会触发协议异常。"""

    def handler(request: httpx.Request) -> httpx.Response:
        """返回不符合约定格式的响应内容。"""
        return httpx.Response(
            200,
            content=b"not-json",
            headers={"content-type": "application/json"},
        )

    transport = create_transport(handler)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://rag.example/",
    ) as client:
        retriever = HttpRagRetriever(client)

        with pytest.raises(RagProtocolError):
            await retriever.retrieve(
                RagSearchRequest(query="测试错误响应")
            )


async def test_http_retriever_maps_timeout() -> None:
    """验证请求超时会被转换为RagTimeoutError。"""

    def handler(request: httpx.Request) -> httpx.Response:
        """模拟外部RAG服务读取超时。"""
        raise httpx.ReadTimeout(
            "mock timeout",
            request=request,
        )

    transport = create_transport(handler)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://rag.example/",
    ) as client:
        retriever = HttpRagRetriever(client)

        with pytest.raises(RagTimeoutError):
            await retriever.retrieve(
                RagSearchRequest(query="测试请求超时")
            )


async def test_http_retriever_maps_connection_failure() -> None:
    """验证连接失败会被转换为RagUnavailableError。"""

    def handler(request: httpx.Request) -> httpx.Response:
        """模拟无法连接外部RAG服务。"""
        raise httpx.ConnectError(
            "mock connection failure",
            request=request,
        )

    transport = create_transport(handler)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://rag.example/",
    ) as client:
        retriever = HttpRagRetriever(client)

        with pytest.raises(RagUnavailableError):
            await retriever.retrieve(
                RagSearchRequest(query="测试连接失败")
            )