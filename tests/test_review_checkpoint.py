"""SQLite checkpoint持久化与恢复测试。"""

from pathlib import Path

import pytest
from langgraph.types import Command

from railguard.models.schemas import ContractDocument
from railguard.parsers.clauses import split_clauses
from railguard.rag.mock import MockRagRetriever
from railguard.workflow.checkpoint import (
    create_review_config,
    open_sqlite_checkpointer,
)
from railguard.workflow.graph import build_review_graph


def create_demo_contract() -> ContractDocument:
    """创建用于测试人工审批恢复的演示合同。"""
    full_text = (
        "设备监测平台软件采购合同\n\n"
        "第一条 项目内容\n\n"
        "乙方为甲方建设设备监测平台。\n\n"
        "第二条 付款方式\n\n"
        "合同签订后五日内，甲方支付全部合同款。\n\n"
        "第三条 验收\n\n"
        "系统上线三日后，甲方未提出异议视为验收合格。"
    )

    return ContractDocument(
        filename="software-purchase-demo.docx",
        full_text=full_text,
        clauses=split_clauses(full_text),
    )


def create_retriever() -> MockRagRetriever:
    """创建本地演示RAG检索器。"""
    return MockRagRetriever.from_json_file(
        Path("data/demo/rag-corpus.json")
    )


def test_review_config_validates_review_id() -> None:
    """验证启动与恢复使用稳定的thread_id。"""
    config = create_review_config("review-001")

    assert config == {
        "configurable": {
            "thread_id": "review-001",
        }
    }

    with pytest.raises(ValueError):
        create_review_config("   ")


async def test_sqlite_checkpoint_resumes_after_reopening(
    tmp_path: Path,
) -> None:
    """验证关闭并重新打开数据库后仍能恢复人工审批。

    第一个上下文代表服务首次运行。
    第二个上下文代表服务重启后重新创建数据库连接和工作流。
    """
    checkpoint_path = (
        tmp_path
        / "runtime"
        / "review-checkpoints.sqlite3"
    )
    config = create_review_config("review-persistent")

    # 首次运行：完成分析并暂停等待人工审核。
    async with open_sqlite_checkpointer(
        checkpoint_path
    ) as checkpointer:
        graph = build_review_graph(
            retriever=create_retriever(),
            checkpointer=checkpointer,
        )

        interrupted_result = await graph.ainvoke(
            {
                "review_id": "review-persistent",
                "contract": create_demo_contract(),
            },
            config=config,
        )

        assert (
            interrupted_result["status"]
            == "awaiting_human"
        )
        assert "__interrupt__" in interrupted_result

        original_finding_ids = {
            finding.finding_id
            for finding in interrupted_result["findings"]
        }

    # 离开上下文后，首次运行使用的数据库连接已经关闭。
    assert checkpoint_path.exists()

    # 模拟服务重启：创建全新的连接和编译图。
    async with open_sqlite_checkpointer(
        checkpoint_path
    ) as reopened_checkpointer:
        restarted_graph = build_review_graph(
            retriever=create_retriever(),
            checkpointer=reopened_checkpointer,
        )

        snapshot = await restarted_graph.aget_state(config)

        assert snapshot.values["status"] == "awaiting_human"
        assert "human_review" in snapshot.next

        resumed_result = await restarted_graph.ainvoke(
            Command(
                resume={
                    "action": "approve",
                    "reviewer": "法务审核员",
                    "finding_decisions": {},
                    "comment": "服务重启后完成审核。",
                }
            ),
            config=config,
        )

        assert resumed_result["status"] == "approved"
        assert "最终保留6项" in resumed_result["final_summary"]
        assert "__interrupt__" not in resumed_result

        resumed_finding_ids = {
            finding.finding_id
            for finding in resumed_result["findings"]
        }

        # 恢复使用原来的风险记录，不重新生成一批风险ID。
        assert resumed_finding_ids == original_finding_ids

        # human_review只在获得人工决定后记录一次完成。
        assert (
            resumed_result["completed_nodes"].count(
                "human_review"
            )
            == 1
        )