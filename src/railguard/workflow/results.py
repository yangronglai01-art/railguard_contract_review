"""合同审核服务的统一结果模型。"""

from typing import Literal

from pydantic import Field

from railguard.models.schemas import (
    Evidence,
    RiskFinding,
    SchemaModel,
)
from railguard.workflow.state import (
    HumanReviewDecision,
    ReviewStatus,
    WorkflowError,
)

# 技术执行失败由checkpoint任务错误推导，不修改原工作流状态。
PublicReviewStatus = ReviewStatus | Literal["failed"]


class ReviewSnapshot(SchemaModel):
    """供FastAPI和前端读取的一次审核任务快照。"""

    # 每次审核的唯一标识，同时也是LangGraph的thread_id。
    review_id: str = Field(min_length=1)

    # 被审核合同的唯一标识。
    contract_id: str = Field(min_length=1)

    # 原始合同文件名，方便审核人识别任务。
    filename: str = Field(min_length=1)

    # 当前审核状态。
    status: PublicReviewStatus

    # Agent产生并完成引用验证的原始风险结果。
    # 即使人工驳回某项风险，这里仍然保留用于审计。
    findings: list[RiskFinding] = Field(default_factory=list)

    # 人工审核后最终保留的风险。
    # 审核尚未完成时使用None；审核完成但没有保留风险时使用空列表。
    final_findings: list[RiskFinding] | None = None

    # 本次审核实际使用的RAG证据。
    evidence: list[Evidence] = Field(default_factory=list)

    # 人工审核决定；尚未提交人工决定时为None。
    human_decision: HumanReviewDecision | None = None

    # 已完成节点列表，用于展示多Agent执行轨迹。
    completed_nodes: list[str] = Field(default_factory=list)

    # 从checkpoint任务错误中整理出的安全错误信息。
    errors: list[WorkflowError] = Field(default_factory=list)

    # 审核完成后生成的业务摘要。
    final_summary: str = ""

    # 审核任务启动时间（ISO8601 UTC）；启动前为空。
    started_at: str | None = None

    # OpenTelemetry风格的追踪ID。
    trace_id: str | None = None

    # 审核进度百分比（0-100），根据已完成节点估算。
    progress_percent: float = 0.0

    # 从启动到当前快照的耗时（秒）。
    elapsed_seconds: float | None = None

    # 各节点执行耗时（秒），键为节点名。
    node_durations: dict[str, float] = Field(default_factory=dict)

    # 各Agent模型调用的重试次数，键为Agent名。
    retry_stats: dict[str, int] = Field(default_factory=dict)


class NodeLatencyStats(SchemaModel):
    """单个节点的耗时统计（毫秒）。"""

    count: int = 0
    avg_ms: float | None = None
    p50_ms: float | None = None
    p95_ms: float | None = None
    p99_ms: float | None = None
    min_ms: float | None = None
    max_ms: float | None = None


class MonitoringSummary(SchemaModel):
    """跨审核任务的监控汇总。"""

    total_reviews: int = 0
    failed_reviews: int = 0
    status_counts: dict[str, int] = Field(default_factory=dict)
    node_latency: dict[str, NodeLatencyStats] = Field(
        default_factory=dict
    )
    retry_stats: dict[str, int] = Field(default_factory=dict)