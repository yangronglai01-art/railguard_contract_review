"""主从编排（supervisor_v2）的领域契约。

本模块定义 document 25 目标架构所需的全部 Pydantic 契约，用于把
「固定三路并行」升级为「主 Agent 受约束规划 → 确定性校验 → 专业子 Agent
执行 → 任务级共享记忆 → PG 事务 Outbox 事件驱动 → 有限协同复核」的编排模式。

所有模型都继承 :class:`~railguard.models.schemas.SchemaModel`，因此默认
``extra="forbid"``，并通过 ``model_dump(mode="json")`` 参与 LangGraph
checkpoint 的序列化（白名单在 ``workflow/checkpoint.py`` 中维护）。

注意：本模块只定义契约，不包含任何执行逻辑；执行逻辑位于
``workflow/supervisor.py``。这样契约与实现解耦，方便后续微调、评审与测试。
"""

from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from railguard.models.schemas import SchemaModel, new_id

# 单个编排任务的类型。每个子 Agent 只负责其中一类任务。
TaskKind = Literal[
    "retrieval",       # RAG 证据检索
    "commercial",      # 商务风险分析
    "legal",           # 法务风险分析
    "security",        # 数据安全风险分析
    "aggregation",     # 汇总去重
    "verification",    # 引用核验
]

# 编排任务的生命周期状态。
TaskStatus = Literal[
    "pending",
    "dispatched",
    "running",
    "completed",
    "failed",
    "skipped",
]

# 编排计划的整体状态。
PlanStatus = Literal[
    "draft",
    "validated",
    "executing",
    "reconciling",
    "completed",
    "failed",
]

# 协同复核冲突的解决方式。
ConflictResolution = Literal[
    "keep_primary",     # 以主 Agent 结论为准
    "keep_challenger",  # 采纳挑战方结论
    "merge",            # 合并双方结论
    "escalate",         # 升级到人工
]

# 事件信封中的事件类型。
EventType = Literal[
    "plan_created",
    "plan_validated",
    "plan_rejected",
    "task_dispatched",
    "task_completed",
    "task_failed",
    "memory_written",
    "conflict_detected",
    "conflict_resolved",
    "review_completed",
    "review_failed",
]

# 共享记忆的可见范围。
MemoryScope = Literal[
    "review",   # 本次审核全局可见
    "task",     # 仅当前任务可见
    "cross_review",  # 跨审核共享（需显式声明，默认禁用）
]


class ReviewContext(SchemaModel):
    """主 Agent 启动规划前接收的、不可变的一次审核上下文。

    它是 :class:`ReviewPlan` 的输入，规划阶段只读，不修改。
    """

    review_id: str = Field(min_length=1)
    trace_id: str = Field(min_length=1)
    business_profile: str = Field(min_length=1)
    contract_id: str = Field(min_length=1)
    contract_title: str = Field(default="")
    clause_count: int = Field(default=0, ge=0)
    tenant_id: str | None = None
    started_at_iso: str = Field(min_length=1)


class ReviewTask(SchemaModel):
    """编排计划中派发给单个子 Agent 的一项原子任务。

    任务之间通过 ``depends_on`` 表达偏序依赖，规划阶段保证无环。
    """

    task_id: str = Field(default_factory=new_id)
    plan_id: str = Field(min_length=1)
    kind: TaskKind
    # 目标条款ID集合；空表示合同级任务。
    clause_ids: list[str] = Field(default_factory=list)
    # 该任务必须等待的前置任务ID。
    depends_on: list[str] = Field(default_factory=list)
    status: TaskStatus = "pending"
    # 指定执行该任务的子 Agent；规划阶段可显式指定。
    assignee: str | None = None
    # 确定性校验器要求的输入约束，例如「必须有合同正文」。
    required_inputs: list[str] = Field(default_factory=list)

    @field_validator("clause_ids", "depends_on", "required_inputs")
    @classmethod
    def validate_ids_non_blank(cls, value: list[str]) -> list[str]:
        if any(not item.strip() for item in value):
            raise ValueError("task id entries must not be blank")
        return value


class ReviewPlan(SchemaModel):
    """主 Agent 受约束规划后产出的可执行计划。

    规划结果必须先经过确定性校验（``validate_plan``），校验通过后
    计划状态从 ``draft`` 转为 ``validated``，随后才允许派发任务。
    """

    plan_id: str = Field(default_factory=new_id)
    review_id: str = Field(min_length=1)
    context: ReviewContext
    tasks: list[ReviewTask] = Field(default_factory=list)
    status: PlanStatus = "draft"
    # 规划阶段生成的、说明每条任务为何成立的受约束说明。
    rationale: str = Field(default="")
    version: int = Field(default=1, ge=1)
    created_at_iso: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_plan_integrity(self) -> Self:
        """计划必须自洽：任务ID唯一、依赖不产生自环、引用存在的任务。"""
        ids = [task.task_id for task in self.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("task ids must be unique within a plan")

        for task in self.tasks:
            for dep in task.depends_on:
                if dep not in ids:
                    raise ValueError(
                        f"task {task.task_id} depends on unknown task {dep}"
                    )
                if dep == task.task_id:
                    raise ValueError(
                        f"task {task.task_id} depends on itself"
                    )
        return self


class TaskResult(SchemaModel):
    """子 Agent 执行完成后的产物，回传给主 Agent 汇总。

    ``payload`` 为结构化的任务输出（如风险列表、证据列表）；
    ``memory_updates`` 为该任务希望写入共享记忆的键值。
    """

    task_id: str = Field(min_length=1)
    plan_id: str = Field(min_length=1)
    status: Literal["completed", "failed"] = "completed"
    payload: dict = Field(default_factory=dict)
    error: str | None = None
    memory_updates: dict[str, object] = Field(default_factory=dict)
    produced_at_iso: str = Field(min_length=1)


class MemoryRecord(SchemaModel):
    """任务级共享记忆中的一条记录。

    共享记忆是「主从编排」区别于固定三路并行的关键：子 Agent 在任务执行后
    写入发现，主 Agent 与后续任务在需要时读取，避免重复检索与结论冲突。
    """

    memory_id: str = Field(default_factory=new_id)
    review_id: str = Field(min_length=1)
    scope: MemoryScope = "review"
    key: str = Field(min_length=1)
    value: object = None
    written_by: str = Field(min_length=1)
    written_at_iso: str = Field(min_length=1)
    version: int = Field(default=1, ge=1)

    @field_validator("key")
    @classmethod
    def validate_key(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("memory key must not be blank")
        return value


class ConflictRecord(SchemaModel):
    """有限协同复核阶段发现的两条结论冲突。

    仅当两个专业子 Agent 对同一目标产生互相矛盾结论时生成，
    按 ``resolution`` 策略有限复核，绝不引入无限争论循环。
    """

    conflict_id: str = Field(default_factory=new_id)
    review_id: str = Field(min_length=1)
    # 冲突涉及的任务ID对。
    task_a: str = Field(min_length=1)
    task_b: str = Field(min_length=1)
    # 冲突目标（如条款ID或风险ID）。
    subject: str = Field(min_length=1)
    description: str = Field(min_length=1)
    resolution: ConflictResolution | None = None
    resolved_by: str | None = None
    created_at_iso: str = Field(min_length=1)


class ReviewEvent(SchemaModel):
    """PG 事务 Outbox 事件驱动架构中的一条领域事件。

    事件在业务流程的关键节点产生，写入业务数据库的 Outbox 表，
    与业务数据在同一事务内提交，再由后台投递器异步发布。
    """

    event_id: str = Field(default_factory=new_id)
    event_type: EventType
    aggregate_id: str = Field(min_length=1)  # 通常为 review_id
    trace_id: str = Field(min_length=1)
    occurred_at_iso: str = Field(min_length=1)
    payload: dict = Field(default_factory=dict)
    # Outbox 投递状态：待投递/已投递/投递失败。
    delivery_status: Literal["pending", "published", "failed"] = "pending"


class SupervisorRunState(SchemaModel):
    """supervisor_v2 编排图的共享状态投影。

    该模型用于把 LangGraph 的 :class:`~railguard.workflow.state.ReviewState`
    扩展为可审计、可序列化的编排快照。实际运行时仍然以 TypedDict 状态为主，
    此模型仅用于序列化边界与监控聚合。
    """

    review_id: str = Field(min_length=1)
    plan: ReviewPlan | None = None
    tasks: dict[str, ReviewTask] = Field(default_factory=dict)
    results: dict[str, TaskResult] = Field(default_factory=dict)
    memory: dict[str, MemoryRecord] = Field(default_factory=dict)
    conflicts: list[ConflictRecord] = Field(default_factory=list)
    events: list[ReviewEvent] = Field(default_factory=list)
    status: PlanStatus = "draft"
