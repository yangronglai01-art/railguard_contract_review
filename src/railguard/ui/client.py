"""Streamlit界面使用的RailGuard HTTP API客户端。"""

from typing import Any

import httpx


class ApiClientError(RuntimeError):
    """前端API客户端异常的公共基类。"""


class ApiConnectionError(ApiClientError):
    """无法连接RailGuard后端服务。"""


class ApiTimeoutError(ApiConnectionError):
    """后端服务未在规定时间内返回。"""


class ApiProtocolError(ApiClientError):
    """后端成功响应不符合约定的JSON格式。"""


class ApiResponseError(ApiClientError):
    """后端返回了4xx或5xx错误。"""

    def __init__(
        self,
        *,
        status_code: int,
        detail: object,
    ) -> None:
        """保存HTTP状态码和后端返回的安全错误信息。"""
        super().__init__(str(detail))
        self.status_code = status_code
        self.detail = detail


class RailGuardApiClient:
    """封装Streamlit需要调用的RailGuard接口。"""

    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """创建可复用连接池的同步HTTP客户端。

        transport参数仅用于测试注入MockTransport。
        """
        self._client = httpx.Client(
            base_url=f"{base_url.rstrip('/')}/",
            timeout=timeout_seconds,
            transport=transport,
        )

    def close(self) -> None:
        """关闭HTTP连接池。"""
        self._client.close()

    def health(self) -> dict[str, Any]:
        """读取后端健康状态。"""
        return self._request(
            method="GET",
            path="health",
        )

    def upload_contract(
        self,
        *,
        filename: str,
        content: bytes,
        content_type: str,
    ) -> dict[str, Any]:
        """上传DOCX或PDF合同并返回解析结果。"""
        return self._request(
            method="POST",
            path="contracts",
            files={
                "file": (
                    filename,
                    content,
                    content_type,
                )
            },
        )

    def start_review(
        self,
        contract_id: str,
    ) -> dict[str, Any]:
        """启动一次新的多Agent合同审核。"""
        return self._request(
            method="POST",
            path=f"contracts/{contract_id}/reviews",
        )

    def get_review(
        self,
        review_id: str,
    ) -> dict[str, Any]:
        """读取审核任务当前状态。"""
        return self._request(
            method="GET",
            path=f"reviews/{review_id}",
        )

    def submit_decision(
        self,
        *,
        review_id: str,
        action: str,
        reviewer: str,
        finding_decisions: dict[str, str],
        comment: str,
    ) -> dict[str, Any]:
        """提交人工审核决定并恢复LangGraph任务。"""
        return self._request(
            method="POST",
            path=f"reviews/{review_id}/decision",
            json={
                "action": action,
                "reviewer": reviewer,
                "finding_decisions": finding_decisions,
                "comment": comment,
            },
        )

    def _request(
        self,
        *,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """发送请求并统一处理网络、HTTP和JSON错误。"""
        try:
            response = self._client.request(
                method=method,
                url=path,
                **kwargs,
            )
        except httpx.TimeoutException as exc:
            raise ApiTimeoutError(
                "RailGuard后端请求超时。"
            ) from exc
        except httpx.RequestError as exc:
            raise ApiConnectionError(
                "无法连接RailGuard后端服务。"
            ) from exc

        if response.is_error:
            raise ApiResponseError(
                status_code=response.status_code,
                detail=self._error_detail(response),
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise ApiProtocolError(
                "RailGuard后端返回了无效JSON。"
            ) from exc

        if not isinstance(payload, dict):
            raise ApiProtocolError(
                "RailGuard后端JSON顶层必须是对象。"
            )

        return payload

    @staticmethod
    def _error_detail(
        response: httpx.Response,
    ) -> object:
        """提取后端错误详情，无法解析时使用通用说明。"""
        try:
            payload = response.json()
        except ValueError:
            return "RailGuard后端请求失败。"

        if isinstance(payload, dict):
            return payload.get(
                "detail",
                "RailGuard后端请求失败。",
            )

        return "RailGuard后端请求失败。"