"""Streamlit前端HTTP API客户端测试。"""

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from railguard.ui.client import (
    ApiConnectionError,
    ApiProtocolError,
    ApiResponseError,
    ApiTimeoutError,
    RailGuardApiClient,
)

# MockTransport请求处理函数的类型。
ResponseHandler = Callable[[httpx.Request], httpx.Response]


def create_client(handler: ResponseHandler) -> RailGuardApiClient:
    """创建使用模拟HTTP传输层的RailGuard客户端。"""
    return RailGuardApiClient(
        base_url="http://railguard.test",
        timeout_seconds=1.0,
        transport=httpx.MockTransport(handler),
    )


def test_client_calls_review_api_with_expected_requests() -> None:
    """验证前端客户端会调用正确接口并提交人工审核数据。"""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """记录请求并返回可识别的模拟响应。"""
        requests.append(request)
        return httpx.Response(
            status_code=200,
            json={
                "method": request.method,
                "path": request.url.path,
            },
        )

    client = create_client(handler)

    try:
        health = client.health()
        uploaded = client.upload_contract(
            filename="demo.docx",
            content=b"contract-bytes",
            content_type=(
                "application/vnd.openxmlformats-officedocument."
                "wordprocessingml.document"
            ),
        )
        started = client.start_review("contract-001")
        review = client.get_review("review-001")
        decided = client.submit_decision(
            review_id="review-001",
            action="approve",
            reviewer="法务审核员",
            finding_decisions={
                "finding-001": "accept",
                "finding-002": "dismiss",
            },
            comment="已核对风险和引用证据。",
        )
    finally:
        client.close()

    assert health == {
        "method": "GET",
        "path": "/health",
    }
    assert uploaded["path"] == "/contracts"
    assert started["path"] == "/contracts/contract-001/reviews"
    assert review["path"] == "/reviews/review-001"
    assert decided["path"] == "/reviews/review-001/decision"

    assert [request.method for request in requests] == [
        "GET",
        "POST",
        "POST",
        "GET",
        "POST",
    ]

    upload_request = requests[1]
    assert "multipart/form-data" in upload_request.headers["content-type"]
    assert b'demo.docx' in upload_request.content
    assert b"contract-bytes" in upload_request.content

    decision_request = requests[4]
    decision_payload: dict[str, Any] = json.loads(
        decision_request.content,
    )
    assert decision_payload == {
        "action": "approve",
        "reviewer": "法务审核员",
        "finding_decisions": {
            "finding-001": "accept",
            "finding-002": "dismiss",
        },
        "comment": "已核对风险和引用证据。",
    }


def test_client_exposes_backend_error_detail() -> None:
    """验证4xx响应会保留状态码和后端错误详情。"""

    def handler(request: httpx.Request) -> httpx.Response:
        """返回包含业务错误详情的冲突响应。"""
        return httpx.Response(
            status_code=409,
            json={"detail": "审核任务已经完成。"},
            request=request,
        )

    client = create_client(handler)

    try:
        with pytest.raises(ApiResponseError) as captured:
            client.get_review("review-001")
    finally:
        client.close()

    assert captured.value.status_code == 409
    assert captured.value.detail == "审核任务已经完成。"


def test_client_rejects_invalid_success_json() -> None:
    """验证成功响应不是有效JSON时会报告协议错误。"""

    def handler(request: httpx.Request) -> httpx.Response:
        """返回无法解析为JSON的成功响应。"""
        return httpx.Response(
            status_code=200,
            content=b"invalid-json",
            request=request,
        )

    client = create_client(handler)

    try:
        with pytest.raises(
            ApiProtocolError,
            match="无效JSON",
        ):
            client.health()
    finally:
        client.close()


def test_client_rejects_non_object_json() -> None:
    """验证JSON顶层不是对象时会报告协议错误。"""

    def handler(request: httpx.Request) -> httpx.Response:
        """返回顶层为数组的JSON响应。"""
        return httpx.Response(
            status_code=200,
            json=["unexpected"],
            request=request,
        )

    client = create_client(handler)

    try:
        with pytest.raises(
            ApiProtocolError,
            match="顶层必须是对象",
        ):
            client.health()
    finally:
        client.close()


def test_client_converts_timeout_error() -> None:
    """验证HTTP超时会转换为前端可识别的超时异常。"""

    def handler(request: httpx.Request) -> httpx.Response:
        """模拟后端读取超时。"""
        raise httpx.ReadTimeout(
            "request timed out",
            request=request,
        )

    client = create_client(handler)

    try:
        with pytest.raises(
            ApiTimeoutError,
            match="请求超时",
        ):
            client.health()
    finally:
        client.close()


def test_client_converts_connection_error() -> None:
    """验证连接失败会转换为统一的后端连接异常。"""

    def handler(request: httpx.Request) -> httpx.Response:
        """模拟无法连接后端服务。"""
        raise httpx.ConnectError(
            "connection failed",
            request=request,
        )

    client = create_client(handler)

    try:
        with pytest.raises(
            ApiConnectionError,
            match="无法连接",
        ):
            client.health()
    finally:
        client.close()