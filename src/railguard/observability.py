"""OpenTelemetry风格的结构化追踪、日志和统计工具。

本模块不依赖OpenTelemetry SDK，只输出约定格式的单行结构化JSON日志，
字段命名与OTel语义保持一致，便于后续接入真正的OTel Collector。

除此之外还提供轻量的模型重试计数和分位数计算，
用于支撑审核监控端点、节点级耗时统计和失败重试统计。
"""

import json
import logging
import secrets
from contextvars import ContextVar
from datetime import UTC, datetime

logger = logging.getLogger("railguard.tracing")

# 当前异步上下文累计的模型响应重试次数。
# 三个Agent并行执行时各自拥有独立的上下文副本，互不干扰。
_model_retry_count: ContextVar[int] = ContextVar(
    "railguard_model_retry_count",
    default=0,
)


def now_iso() -> str:
    """返回当前UTC时间的ISO8601字符串。"""
    return datetime.now(UTC).isoformat()


def new_trace_id() -> str:
    """生成OpenTelemetry风格的32位十六进制trace id。"""
    return secrets.token_hex(16)


def record_model_retry() -> None:
    """在当前异步上下文累计一次模型响应重试。"""
    _model_retry_count.set(_model_retry_count.get() + 1)


def take_model_retry_count() -> int:
    """返回并清零当前异步上下文累计的模型重试次数。"""
    count = _model_retry_count.get()
    _model_retry_count.set(0)
    return count


def _emit(payload: dict[str, object]) -> None:
    """输出一条单行结构化JSON日志。"""
    logger.info(
        json.dumps(
            payload,
            ensure_ascii=False,
            default=str,
        )
    )


def emit_span(
    *,
    name: str,
    trace_id: str,
    review_id: str,
    duration_ms: float,
    status: str = "ok",
    attributes: dict[str, object] | None = None,
) -> None:
    """输出一条节点级span日志。"""
    _emit(
        {
            "event": "span",
            "trace_id": trace_id,
            "review_id": review_id,
            "span_name": name,
            "duration_ms": round(duration_ms, 3),
            "status": status,
            "attributes": attributes or {},
        }
    )


def emit_event(
    *,
    event: str,
    trace_id: str = "",
    review_id: str = "",
    **fields: object,
) -> None:
    """输出一条带追踪ID的结构化事件日志。"""
    payload: dict[str, object] = {
        "event": event,
        "trace_id": trace_id,
        "review_id": review_id,
    }
    payload.update(fields)
    _emit(payload)


def percentile(values: list[float], p: float) -> float | None:
    """计算分位数（线性插值），空列表返回None。"""
    if not values:
        return None

    ordered = sorted(values)
    rank = (len(ordered) - 1) * p
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower

    return ordered[lower] * (1 - weight) + ordered[upper] * weight
