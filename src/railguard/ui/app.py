"""RailGuard合同审核Streamlit人工审核界面。"""

import json
import os
from typing import Any

import streamlit as st

from railguard.ui.client import (
    ApiClientError,
    ApiResponseError,
    RailGuardApiClient,
)

# 后端审核状态对应的中文名称。
STATUS_LABELS = {
    "created": "已创建",
    "retrieving": "正在检索知识库",
    "analyzing": "Agent正在分析",
    "verifying": "正在验证引用",
    "awaiting_human": "等待人工审核",
    "approved": "审核通过",
    "rejected": "已拒绝",
    "changes_requested": "要求修改",
    "failed": "执行失败",
}

# 风险等级对应的中文名称。
LEVEL_LABELS = {
    "high": "高风险",
    "medium": "中风险",
    "low": "低风险",
}

# 风险分类对应的业务名称。
CATEGORY_LABELS = {
    "payment": "付款条件",
    "acceptance": "验收条件",
    "liability": "违约责任",
    "intellectual_property": "知识产权",
    "data_security": "数据安全",
    "confidentiality": "保密义务",
    "operation_maintenance": "运维服务",
    "support": "运维服务",
}

# 引用验证状态对应的中文名称。
CITATION_LABELS = {
    "pending": "等待验证",
    "source_matched": "来源已匹配",
    "partially_matched": "部分引用被拒绝",
    "unsupported": "证据不足",
    "not_required": "无需外部引用",
}

# 人工审核动作的界面名称和接口值。
REVIEW_ACTIONS = {
    "审核通过": "approve",
    "退回修改": "request_changes",
    "拒绝合同": "reject",
}


def get_api_base_url() -> str:
    """读取RailGuard后端服务地址。"""
    return os.getenv(
        "RAILGUARD_API_BASE_URL",
        "http://127.0.0.1:8000",
    ).rstrip("/")


@st.cache_resource
def get_api_client() -> RailGuardApiClient:
    """创建供整个Streamlit会话复用的HTTP API客户端。"""
    return RailGuardApiClient(
        base_url=get_api_base_url(),
        timeout_seconds=30.0,
    )


def initialize_session_state() -> None:
    """初始化当前浏览器会话使用的合同和审核状态。"""
    st.session_state.setdefault("contract_snapshot", None)
    st.session_state.setdefault("review_snapshot", None)
    st.session_state.setdefault("notice", "")


def apply_page_style() -> None:
    """设置页面宽度和主要界面的展示样式。"""
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }

        [data-testid="stMetric"] {
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 14px;
            background: #ffffff;
        }

        [data-testid="stFileUploader"] {
            border-radius: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def format_api_error(error: ApiClientError) -> str:
    """把客户端异常转换成适合在界面展示的错误信息。"""
    if not isinstance(error, ApiResponseError):
        return str(error)

    detail = error.detail

    if isinstance(detail, dict):
        message = detail.get("message")

        if isinstance(message, str):
            return (
                f"后端返回 {error.status_code}："
                f"{message}"
            )

        detail_text = json.dumps(
            detail,
            ensure_ascii=False,
        )
    elif isinstance(detail, list):
        detail_text = json.dumps(
            detail,
            ensure_ascii=False,
        )
    else:
        detail_text = str(detail)

    return (
        f"后端返回 {error.status_code}："
        f"{detail_text}"
    )


def show_api_error(error: ApiClientError) -> None:
    """在页面中显示统一格式的后端调用错误。"""
    st.error(format_api_error(error))


def show_pending_notice() -> None:
    """显示页面刷新前保存的一次性成功提示。"""
    notice = st.session_state.get("notice", "")

    if notice:
        st.success(notice)
        st.session_state["notice"] = ""


def content_type_for(filename: str, reported_type: str | None) -> str:
    """根据上传文件名补全浏览器未提供的媒体类型。"""
    if reported_type:
        return reported_type

    if filename.lower().endswith(".pdf"):
        return "application/pdf"

    return (
        "application/vnd.openxmlformats-officedocument."
        "wordprocessingml.document"
    )


def render_sidebar(client: RailGuardApiClient) -> None:
    """展示系统说明、后端地址和连接检查入口。"""
    with st.sidebar:
        st.title("RailGuard")
        st.caption("采购方合同风险审核工作台")

        st.divider()
        st.write("当前演示范围")
        st.markdown(
            """
            - 软件采购与技术服务合同
            - 商务、法务和数据安全Agent
            - RAG证据引用与来源验证
            - 人工确认、驳回和修改决定
            """
        )

        st.divider()
        st.caption("后端服务地址")
        st.code(get_api_base_url(), language=None)

        if st.button(
            "检查后端连接",
            use_container_width=True,
        ):
            try:
                health = client.health()
            except ApiClientError as error:
                show_api_error(error)
            else:
                service = health.get("service", "RailGuard")
                environment = health.get(
                    "environment",
                    "unknown",
                )
                st.success(
                    f"{service} 连接正常，"
                    f"环境：{environment}"
                )


def render_header() -> None:
    """展示审核工作台标题和业务说明。"""
    st.title("合同智能审核工作台")
    st.write(
        "上传采购合同，由三个专业Agent并行识别风险，"
        "再由人工审核人确认最终处理结果。"
    )


def render_upload(client: RailGuardApiClient) -> None:
    """展示合同上传控件并保存后端解析结果。"""
    st.subheader("1. 上传合同")

    with st.container(border=True):
        uploaded_file = st.file_uploader(
            "选择DOCX或文本型PDF合同",
            type=["docx", "pdf"],
            help="当前演示版本支持最大10 MiB的DOCX和文本型PDF。",
        )

        upload_clicked = st.button(
            "解析并保存合同",
            type="primary",
            disabled=uploaded_file is None,
            use_container_width=True,
        )

    if not upload_clicked or uploaded_file is None:
        return

    try:
        with st.spinner("正在解析合同并提取条款……"):
            contract = client.upload_contract(
                filename=uploaded_file.name,
                content=uploaded_file.getvalue(),
                content_type=content_type_for(
                    uploaded_file.name,
                    uploaded_file.type,
                ),
            )
    except ApiClientError as error:
        show_api_error(error)
        return

    st.session_state["contract_snapshot"] = contract
    st.session_state["review_snapshot"] = None
    st.session_state["notice"] = (
        "合同解析成功，可以启动多Agent审核。"
    )
    st.rerun()


def render_contract(contract: dict[str, Any]) -> None:
    """展示当前合同的基本信息、条款和解析文本。"""
    st.subheader("2. 合同解析结果")

    clauses = contract.get("clauses", [])
    filename_column, type_column, clause_column = st.columns(3)

    filename_column.metric(
        "文件名称",
        contract.get("filename", "未知"),
    )
    type_column.metric(
        "审核立场",
        "采购方",
    )
    clause_column.metric(
        "识别条款",
        len(clauses),
    )

    with st.expander("查看条款定位结果"):
        if not clauses:
            st.info("当前合同未识别出带标题的独立条款。")

        for index, clause in enumerate(clauses, start=1):
            title = clause.get("title") or f"条款 {index}"
            st.markdown(f"**{index}. {title}**")
            st.write(clause.get("text", ""))
            st.caption(
                "原文位置："
                f"{clause.get('start_offset', 0)}–"
                f"{clause.get('end_offset', 0)}"
            )

            if index < len(clauses):
                st.divider()

    with st.expander("查看合同完整解析文本"):
        st.text(contract.get("full_text", ""))


def render_workflow_trace(review: dict[str, Any]) -> None:
    """展示LangGraph已完成节点和结构化执行错误。"""
    completed_nodes = review.get("completed_nodes", [])
    errors = review.get("errors", [])

    with st.expander("查看多Agent执行轨迹"):
        if completed_nodes:
            st.write(" → ".join(completed_nodes))
        else:
            st.caption("暂时没有已完成节点。")

        for error in errors:
            st.error(
                f"{error.get('node', 'workflow')} / "
                f"{error.get('code', 'unknown')}："
                f"{error.get('message', '执行失败')}"
            )


def render_evidence(
    evidence_ids: list[str],
    evidence_by_id: dict[str, dict[str, Any]],
) -> None:
    """展示一项风险所引用的RAG证据和来源。"""
    with st.expander(
        f"引用证据（{len(evidence_ids)}）"
    ):
        if not evidence_ids:
            st.caption("该项风险没有引用外部知识库证据。")
            return

        for evidence_id in evidence_ids:
            evidence = evidence_by_id.get(evidence_id)

            if evidence is None:
                st.warning(
                    f"未找到证据记录：{evidence_id}"
                )
                continue

            st.markdown(
                f"**{evidence.get('title', '未命名资料')}**"
            )
            st.write(evidence.get("content", ""))
            st.caption(
                f"出处：{evidence.get('source', '未知')}"
            )

            score = evidence.get("score")

            if isinstance(score, int | float):
                st.caption(f"检索相关度：{score:.3f}")

            metadata = evidence.get("metadata", {})
            retrieval_mode = metadata.get("retrieval_mode")

            if retrieval_mode:
                st.caption(
                    f"检索模式：{retrieval_mode}"
                )

            st.divider()


def render_finding(
    finding: dict[str, Any],
    *,
    index: int,
    clause_by_id: dict[str, dict[str, Any]],
    evidence_by_id: dict[str, dict[str, Any]],
) -> None:
    """展示一项风险、原文定位、修改建议和证据。"""
    level = finding.get("level", "low")
    category = finding.get("category", "other")
    level_label = LEVEL_LABELS.get(level, level)
    category_label = CATEGORY_LABELS.get(
        category,
        category,
    )
    citation_status = finding.get(
        "citation_status",
        "pending",
    )

    with st.container(border=True):
        title_column, level_column = st.columns([4, 1])

        with title_column:
            st.markdown(
                f"### 风险 {index} · {category_label}"
            )
            st.caption(
                f"风险编号：{finding.get('finding_id', '')}"
            )

        with level_column:
            if level == "high":
                st.error(level_label)
            elif level == "medium":
                st.warning(level_label)
            else:
                st.info(level_label)

        st.markdown("**风险判断**")
        st.write(finding.get("reason", ""))

        if finding.get("finding_kind") == "missing_clause":
            st.markdown("**缺失条款**")
            st.warning(
                finding.get(
                    "expected_clause",
                    "未说明缺失内容",
                )
            )
        else:
            clause_id = finding.get("clause_id")
            clause = clause_by_id.get(clause_id, {})

            st.markdown("**合同原文定位**")
            st.caption(
                clause.get("title", "未命名条款")
            )
            st.write(
                clause.get(
                    "text",
                    "未找到对应的合同条款。",
                )
            )

        suggested_revision = finding.get(
            "suggested_revision"
        )

        if suggested_revision:
            st.markdown("**建议修改文本**")
            st.success(suggested_revision)

        st.caption(
            "引用状态："
            f"{CITATION_LABELS.get(citation_status, citation_status)}"
        )

        rejected_evidence_ids = finding.get(
            "rejected_evidence_ids",
            [],
        )
        if rejected_evidence_ids:
            st.warning(
                "模型返回了未通过校验的证据ID："
                + "、".join(rejected_evidence_ids)
            )

        render_evidence(
            finding.get("evidence_ids", []),
            evidence_by_id,
        )


def submit_human_decision(
    client: RailGuardApiClient,
    review: dict[str, Any],
) -> None:
    """收集人工审核意见并提交到后端恢复LangGraph流程。"""
    findings = review.get("findings", [])
    review_id = review["review_id"]
    finding_choices: dict[str, str] = {}

    st.subheader("4. 人工审核决定")

    with st.form(f"decision-form-{review_id}"):
        reviewer = st.text_input(
            "审核人",
            placeholder="请输入姓名或企业账号",
        )

        st.markdown("**逐项确认风险**")

        if not findings:
            st.info("本次审核没有发现需要逐项确认的风险。")

        for index, finding in enumerate(
            findings,
            start=1,
        ):
            finding_id = finding["finding_id"]
            category = CATEGORY_LABELS.get(
                finding.get("category", "other"),
                finding.get("category", "其他"),
            )

            finding_choices[finding_id] = st.radio(
                f"风险 {index} · {category}",
                options=["保留风险", "驳回风险"],
                horizontal=True,
                key=f"{review_id}-{finding_id}",
            )

        action_label = st.radio(
            "整体处理结果",
            options=list(REVIEW_ACTIONS),
            horizontal=True,
        )
        comment = st.text_area(
            "审核意见",
            placeholder=(
                "审核通过时可以选填；退回修改或拒绝时必须填写。"
            ),
        )

        submitted = st.form_submit_button(
            "提交人工审核决定",
            type="primary",
            use_container_width=True,
        )

    if not submitted:
        return

    if not reviewer.strip():
        st.error("请填写审核人。")
        return

    action = REVIEW_ACTIONS[action_label]

    if (
        action in {"reject", "request_changes"}
        and not comment.strip()
    ):
        st.error("退回修改或拒绝合同时必须填写审核意见。")
        return

    finding_decisions = {
        finding_id: (
            "accept"
            if choice == "保留风险"
            else "dismiss"
        )
        for finding_id, choice in finding_choices.items()
    }

    try:
        with st.spinner("正在保存决定并生成最终审核结果……"):
            completed_review = client.submit_decision(
                review_id=review_id,
                action=action,
                reviewer=reviewer.strip(),
                finding_decisions=finding_decisions,
                comment=comment.strip(),
            )
    except ApiClientError as error:
        show_api_error(error)
        return

    st.session_state["review_snapshot"] = completed_review
    st.session_state["notice"] = "人工审核决定已保存。"
    st.rerun()


def render_review(
    client: RailGuardApiClient,
    contract: dict[str, Any],
    review: dict[str, Any],
) -> None:
    """展示审核状态、风险结果和人工审核入口。"""
    st.subheader("3. 多Agent审核结果")

    refresh_column, identity_column = st.columns([1, 3])

    with refresh_column:
        refresh_clicked = st.button(
            "刷新审核状态",
            use_container_width=True,
        )

    with identity_column:
        st.caption(
            f"审核任务：{review.get('review_id', '')}"
        )

    if refresh_clicked:
        try:
            review = client.get_review(review["review_id"])
        except ApiClientError as error:
            show_api_error(error)
        else:
            st.session_state["review_snapshot"] = review
            st.success("审核状态已刷新。")

    status = review.get("status", "created")
    findings = review.get("findings", [])
    final_findings = review.get("final_findings")

    status_column, high_column, medium_column, low_column = (
        st.columns(4)
    )
    status_column.metric(
        "当前状态",
        STATUS_LABELS.get(status, status),
    )
    high_column.metric(
        "高风险",
        sum(
            finding.get("level") == "high"
            for finding in findings
        ),
    )
    medium_column.metric(
        "中风险",
        sum(
            finding.get("level") == "medium"
            for finding in findings
        ),
    )
    low_column.metric(
        "低风险",
        sum(
            finding.get("level") == "low"
            for finding in findings
        ),
    )

    final_summary = review.get("final_summary", "")

    if final_summary:
        st.success(final_summary)

    render_workflow_trace(review)

    clause_by_id = {
        clause["clause_id"]: clause
        for clause in contract.get("clauses", [])
    }
    evidence_by_id = {
        evidence["evidence_id"]: evidence
        for evidence in review.get("evidence", [])
    }

    displayed_findings = (
        final_findings
        if final_findings is not None
        else findings
    )

    if final_findings is not None:
        st.markdown(
            f"**人工审核后保留风险："
            f"{len(final_findings)} 项**"
        )

    if not displayed_findings:
        st.info("当前结果中没有保留的合同风险。")

    for index, finding in enumerate(
        displayed_findings,
        start=1,
    ):
        render_finding(
            finding,
            index=index,
            clause_by_id=clause_by_id,
            evidence_by_id=evidence_by_id,
        )

    if status == "awaiting_human":
        submit_human_decision(
            client,
            review,
        )


def start_review(
    client: RailGuardApiClient,
    contract: dict[str, Any],
) -> None:
    """展示启动按钮并创建新的合同审核任务。"""
    st.subheader("3. 启动多Agent审核")

    st.info(
        "系统将并行运行商务、法务和数据安全Agent，"
        "随后验证每项风险引用的知识库证据。"
    )

    if not st.button(
        "启动合同审核",
        type="primary",
        use_container_width=True,
    ):
        return

    try:
        with st.spinner("三个Agent正在分析合同，请稍候……"):
            review = client.start_review(
                contract["contract_id"]
            )
    except ApiClientError as error:
        show_api_error(error)
        return

    st.session_state["review_snapshot"] = review
    st.session_state["notice"] = (
        "多Agent审核完成，任务已进入人工确认阶段。"
    )
    st.rerun()


def main() -> None:
    """配置并运行RailGuard Streamlit审核页面。"""
    st.set_page_config(
        page_title="RailGuard合同审核",
        page_icon="🛡️",
        layout="wide",
    )

    apply_page_style()
    initialize_session_state()

    client = get_api_client()

    render_sidebar(client)
    render_header()
    show_pending_notice()
    render_upload(client)

    contract = st.session_state.get("contract_snapshot")

    if contract is None:
        st.info("请先上传一份DOCX或PDF合同。")
        return

    render_contract(contract)

    review = st.session_state.get("review_snapshot")

    if review is None:
        start_review(client, contract)
        return

    render_review(
        client,
        contract,
        review,
    )


if __name__ == "__main__":
    main()