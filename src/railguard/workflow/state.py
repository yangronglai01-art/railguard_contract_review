"""LangGraph合同审核流程的共享状态。

每个Agent只读取所需字段，并返回自己负责的状态更新。
并行Agent使用独立的输出字段，避免同时写入同一个字段产生冲突。
"""

from operator import add
from typing import Annotated, Literal, Required, Self, TypedDict

from pydantic import Field, field_validator, model_validator

from railguard.models.schemas import (
    ContractDocument,
    Evidence,
    RiskFinding,
    SchemaModel,
)

# 审核流程当前所处的业务状态。
ReviewStatus = Literal[
    "created",
    "retrieving",
    "analyzing",
    "verifying",
    "awaiting_human",
    "approved",
    "rejected",
    "changes_requested",
    "failed",
]


# 人工审核对整个审核任务作出的决定。
HumanReviewAction = Literal[
    "approve",
    "reject",
    "request_changes",
]


# 人工审核对单项风险作出的处理。
FindingReviewAction = Literal[
    "accept",
    "dismiss",
]


class WorkflowError(SchemaModel):
    """审核流程中可以保存和展示的一项结构化错误。"""

    # 发生错误的节点名称。
    node: str = Field(min_length=1)

    # 稳定的错误代码，供接口和前端判断错误类型。
    code: str = Field(min_length=1)

    # 可以记录到日志并展示给用户的错误说明。
    message: str = Field(min_length=1)

    # 是否适合由用户或系统重新执行当前任务。
    retryable: bool = False

    # 附加错误信息，只保存非敏感字符串。
    details: dict[str, str] = Field(default_factory=dict)

    @field_validator("node", "code", "message")
    @classmethod
    def validate_non_blank(cls, value: str) -> str:
        """拒绝只有空白字符的错误字段。"""
        if not value.strip():
            raise ValueError("value must not contain only whitespace")

        return value


class HumanReviewDecision(SchemaModel):
    """人工审核人在流程暂停后提交的审核决定。"""

    # 对整个审核任务的处理结果。
    action: HumanReviewAction

    # 审核人姓名或企业账号。
    reviewer: str = Field(min_length=1)

    # 对单项风险的确认或驳回结果。
    # 没有列出的风险保持Agent原始判断。
    finding_decisions: dict[str, FindingReviewAction] = Field(
        default_factory=dict
    )

    # 审核意见。驳回或要求修改时必须填写。
    comment: str = ""

    @field_validator("reviewer")
    @classmethod
    def validate_reviewer(cls, value: str) -> str:
        """拒绝只有空白字符的审核人名称。"""
        if not value.strip():
            raise ValueError("reviewer must not contain only whitespace")

        return value

    @model_validator(mode="after")
    def validate_decision(self) -> Self:
        """验证人工决定包含足够的审计信息。

        驳回或要求修改时必须说明原因。
        单项风险ID也不能是空白字符串。
        """
        if (
            self.action in {"reject", "request_changes"}
            and not self.comment.strip()
        ):
            raise ValueError(
                "reject or request_changes requires a comment"
            )

        if any(
            not finding_id.strip()
            for finding_id in self.finding_decisions
        ):
            raise ValueError(
                "finding decision ID must not be blank"
            )

        return self


class ReviewState(TypedDict, total=False):
    """LangGraph所有审核节点共享的状态结构。

    review_id和contract是启动流程时必须提供的字段。
    其他字段由不同节点逐步写入。
    """

    # 一次审核任务的唯一标识，也是checkpoint使用的线程标识。
    review_id: Required[str]

    # 本次审核对应的已解析合同。
    contract: Required[ContractDocument]

    # 审核流程当前状态。
    status: ReviewStatus

    # 按合同条款ID保存RAG证据，保持原文与证据的对应关系。
    evidence_by_clause: dict[str, list[Evidence]]

    # 用于检查缺失条款的合同级RAG证据。
    contract_evidence: list[Evidence]

    # 商务风险Agent的独立输出。
    commercial_findings: list[RiskFinding]

    # 法务风险Agent的独立输出。
    legal_findings: list[RiskFinding]

    # 数据安全Agent的独立输出。
    security_findings: list[RiskFinding]

    # 汇总、去重并完成引用验证后的最终风险列表。
    findings: list[RiskFinding]

    # 人工审核恢复流程时提交的决定。
    human_decision: HumanReviewDecision | None

    # 最终给业务人员阅读的审核摘要。
    final_summary: str

    # 已完成节点的审计轨迹。
    # add表示并行节点返回的列表会被合并。
    completed_nodes: Annotated[list[str], add]

    # 流程中产生的结构化错误。
    # add允许不同节点分别追加错误记录。
    errors: Annotated[list[WorkflowError], add]