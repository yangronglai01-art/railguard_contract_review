"""RAG检索协议、HTTP客户端和异常定义。

后续LangGraph节点只依赖RagRetriever协议。
真实知识库、本地Mock和测试替身都可以实现同一个协议，
避免审核流程直接依赖某个具体的RAG产品。
"""

from typing import Protocol

import httpx
from pydantic import ValidationError

from railguard.models.schemas import Evidence
from railguard.rag.models import (
    RagHit,
    RagSearchRequest,
    RagSearchResponse,
)


class RagError(RuntimeError):
    """所有RAG适配器异常的公共基类。"""


class RagUnavailableError(RagError):
    """RAG服务连接失败或暂时不可用。"""


class RagTimeoutError(RagUnavailableError):
    """RAG服务未在规定时间内返回结果。"""


class RagRequestError(RagError):
    """RAG服务拒绝了RailGuard发送的请求。"""


class RagProtocolError(RagError):
    """RAG服务返回了不符合约定格式的数据。"""


class RagRetriever(Protocol):
    """LangGraph审核节点所依赖的统一RAG检索接口。"""

    async def retrieve(
        self,
        request: RagSearchRequest,
    ) -> list[Evidence]:
        """执行检索并返回证据列表。

        成功但没有检索结果时返回空列表。
        服务故障或协议错误必须抛出异常，不能伪装成空结果。
        """
        ...


class HttpRagRetriever:
    """通过HTTP调用外部RAG服务的检索器。"""

    def __init__(self, client: httpx.AsyncClient) -> None:
        """保存由应用生命周期管理的HTTP客户端。

        AsyncClient应在外部配置base_url、超时时间和认证信息，
        这样生产环境可以复用连接池，测试也可以注入模拟传输层。
        """
        self._client = client

    async def retrieve(
        self,
        request: RagSearchRequest,
    ) -> list[Evidence]:
        """调用外部RAG服务并转换为RailGuard证据模型。

        只有合法的200响应会被当作成功结果。
        返回结果超过top_k时在本地截断，保证调用方获得稳定数量。
        """
        try:
            # 使用相对路径，以便base_url可以包含服务前缀。
            response = await self._client.post(
                "search",
                json=request.model_dump(mode="json"),
            )
        except httpx.TimeoutException as exc:
            # 超时与普通连接错误分开，便于API以后返回504。
            raise RagTimeoutError("RAG service request timed out") from exc
        except httpx.RequestError as exc:
            # DNS失败、连接拒绝和连接中断都表示服务暂时不可用。
            raise RagUnavailableError(
                "RAG service is unavailable"
            ) from exc

        # 4xx通常表示请求格式、认证或接口配置不正确。
        if 400 <= response.status_code < 500:
            raise RagRequestError(
                f"RAG service rejected the request: "
                f"HTTP {response.status_code}"
            )

        # 5xx表示外部RAG服务自身发生故障。
        if response.status_code >= 500:
            raise RagUnavailableError(
                f"RAG service failed: HTTP {response.status_code}"
            )

        # 当前协议只接受200，防止重定向或其他状态被误当作结果。
        if response.status_code != 200:
            raise RagProtocolError(
                f"Unexpected RAG response status: "
                f"HTTP {response.status_code}"
            )

        try:
            # 同时完成JSON解析和字段结构校验。
            payload = RagSearchResponse.model_validate_json(
                response.content
            )
        except (ValidationError, ValueError) as exc:
            # 非JSON、字段缺失或字段类型错误都属于协议异常。
            raise RagProtocolError(
                "RAG service returned an invalid response"
            ) from exc

        # 保留RAG服务原有的结果顺序，并确保不超过top_k。
        return [
            self._to_evidence(hit)
            for hit in payload.hits[: request.top_k]
        ]

    @staticmethod
    def _to_evidence(hit: RagHit) -> Evidence:
        """把外部检索结果转换为内部Evidence模型。

        evidence_id由RailGuard在转换时生成，不信任外部服务提供
        的业务主键。document_id继续保留知识库中的稳定文档标识。
        """
        return Evidence(
            document_id=hit.document_id,
            title=hit.title,
            content=hit.content,
            source=hit.source,
            score=hit.score,
            metadata=dict(hit.metadata),
        )