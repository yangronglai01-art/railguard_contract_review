"""RailGuard合同审核LangGraph构建函数。"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from railguard.rag.client import RagRetriever
from railguard.workflow.analyzers import (
    DemoCommercialRiskAnalyzer,
    DemoLegalRiskAnalyzer,
    DemoSecurityRiskAnalyzer,
    RiskAnalyzer,
)
from railguard.workflow.nodes import (
    ReviewNodes,
    route_after_verification,
)
from railguard.workflow.state import ReviewState


def build_review_graph(
    *,
    retriever: RagRetriever,
    commercial_analyzer: RiskAnalyzer | None = None,
    legal_analyzer: RiskAnalyzer | None = None,
    security_analyzer: RiskAnalyzer | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """构建并编译合同审核工作流。

    未传入风险Agent时使用确定性的演示实现。
    后续接入真实大模型或微调模型时，只需要注入新的Agent实现。

    checkpointer允许调用方传入SQLite持久化组件。
    图本身不负责创建或关闭数据库连接。
    """
    resolved_commercial_analyzer = (
        commercial_analyzer
        if commercial_analyzer is not None
        else DemoCommercialRiskAnalyzer()
    )
    resolved_legal_analyzer = (
        legal_analyzer
        if legal_analyzer is not None
        else DemoLegalRiskAnalyzer()
    )
    resolved_security_analyzer = (
        security_analyzer
        if security_analyzer is not None
        else DemoSecurityRiskAnalyzer()
    )

    nodes = ReviewNodes(
        retriever=retriever,
        commercial_analyzer=resolved_commercial_analyzer,
        legal_analyzer=resolved_legal_analyzer,
        security_analyzer=resolved_security_analyzer,
    )

    # ReviewState定义所有节点共享的数据结构。
    builder = StateGraph(ReviewState)

    # 注册确定性准备与RAG检索节点。
    builder.add_node(
        "initialize_review",
        nodes.initialize_review,
    )
    builder.add_node(
        "retrieve_evidence",
        nodes.retrieve_evidence,
    )

    # 注册三个职责独立的风险分析Agent。
    builder.add_node(
        "commercial_risk_agent",
        nodes.commercial_risk_agent,
    )
    builder.add_node(
        "legal_risk_agent",
        nodes.legal_risk_agent,
    )
    builder.add_node(
        "security_risk_agent",
        nodes.security_risk_agent,
    )

    # 注册风险汇总、引用验证和人工审批节点。
    builder.add_node(
        "aggregate_findings",
        nodes.aggregate_findings,
    )
    builder.add_node(
        "verify_citations",
        nodes.verify_citations,
    )
    builder.add_node(
        "human_review",
        nodes.human_review,
    )
    builder.add_node(
        "finalize_review",
        nodes.finalize_review,
    )

    # 合同先初始化，再完成全部RAG检索。
    builder.add_edge(START, "initialize_review")
    builder.add_edge(
        "initialize_review",
        "retrieve_evidence",
    )

    # RAG完成后，并行执行三个专业风险Agent。
    builder.add_edge(
        "retrieve_evidence",
        "commercial_risk_agent",
    )
    builder.add_edge(
        "retrieve_evidence",
        "legal_risk_agent",
    )
    builder.add_edge(
        "retrieve_evidence",
        "security_risk_agent",
    )

    # 使用列表形式的起点，明确等待三个Agent全部完成。
    builder.add_edge(
        [
            "commercial_risk_agent",
            "legal_risk_agent",
            "security_risk_agent",
        ],
        "aggregate_findings",
    )

    builder.add_edge(
        "aggregate_findings",
        "verify_citations",
    )

    # 有风险时暂停等待人工处理，没有风险时直接生成摘要。
    builder.add_conditional_edges(
        "verify_citations",
        route_after_verification,
        {
            "human_review": "human_review",
            "finalize_review": "finalize_review",
        },
    )

    builder.add_edge(
        "human_review",
        "finalize_review",
    )
    builder.add_edge(
        "finalize_review",
        END,
    )

    return builder.compile(
        checkpointer=checkpointer,
        name="railguard_contract_review",
    )