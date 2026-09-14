"""LangGraph checkpoint业务类型白名单测试。"""

import pytest
from pydantic import BaseModel

from railguard.models.schemas import (
    Clause,
    ContractDocument,
    Evidence,
    RiskFinding,
    SchemaModel,
)
from railguard.workflow.checkpoint import (
    create_checkpoint_serializer,
)
from railguard.workflow.state import (
    HumanReviewDecision,
    WorkflowError,
)


class UnregisteredCheckpointModel(BaseModel):
    """不在RailGuard白名单中的测试类型。"""

    value: str


def create_checkpoint_models() -> list[SchemaModel]:
    """创建白名单中六种业务模型的合法测试实例。"""
    text = "甲方验收合格后支付尾款。"
    clause = Clause(
        clause_id="clause-serializer-test",
        title="付款方式",
        text=text,
        start_offset=0,
        end_offset=len(text),
    )

    return [
        clause,
        ContractDocument(
            contract_id="contract-serializer-test",
            filename="serializer-test.docx",
            full_text=text,
            clauses=[clause],
        ),
        Evidence(
            evidence_id="evidence-serializer-test",
            document_id="policy-payment",
            title="软件采购付款规则",
            content="付款应与验收节点相匹配。",
            source="第三条",
            metadata={"category": "payment"},
        ),
        RiskFinding(
            finding_id="finding-serializer-test",
            finding_kind="missing_clause",
            expected_clause="运维保障条款",
            category="support",
            level="medium",
            reason="合同没有约定故障响应和修复时限。",
        ),
        HumanReviewDecision(
            action="approve",
            reviewer="测试法务",
            finding_decisions={
                "finding-serializer-test": "accept"
            },
            comment="已核对风险。",
        ),
        WorkflowError(
            node="commercial_risk_agent",
            code="model_unavailable",
            message="模型服务暂时不可用。",
            retryable=True,
        ),
    ]


@pytest.mark.parametrize(
    "model",
    create_checkpoint_models(),
)
def test_registered_business_models_can_be_restored(
    model: SchemaModel,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """验证白名单业务模型能恢复为原来的Python类型。"""
    serializer = create_checkpoint_serializer()

    serialized = serializer.dumps_typed(model)
    restored = serializer.loads_typed(serialized)

    assert type(restored) is type(model)
    assert restored == model
    assert "Deserializing unregistered type" not in caplog.text


def test_unregistered_type_is_not_instantiated(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """验证未注册类型不能被checkpoint反序列化实例化。"""
    serializer = create_checkpoint_serializer()
    model = UnregisteredCheckpointModel(
        value="untrusted checkpoint data"
    )

    serialized = serializer.dumps_typed(model)
    restored = serializer.loads_typed(serialized)

    # 当前LangGraph版本阻止实例化后会降级为普通字典，
    # 因此测试重点是拒绝恢复未授权Python类型。
    assert not isinstance(
        restored,
        UnregisteredCheckpointModel,
    )
    assert restored == {
        "value": "untrusted checkpoint data"
    }
    assert "Blocked deserialization" in caplog.text