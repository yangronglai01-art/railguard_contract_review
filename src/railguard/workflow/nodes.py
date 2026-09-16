"""LangGraph合同审核流程节点。

本模块只负责审核流程编排，不直接创建HTTP客户端或数据库连接。
RAG检索器和风险Agent通过构造函数注入，便于测试和替换实现。
"""

from typing import Literal

from langgraph.types import interrupt
from pydantic import ValidationError

from railguard.models.schemas import (
    Evidence,
    RiskFinding,
)
from railguard.rag.client import RagRetriever
from railguard.rag.models import RagSearchRequest
from railguard.workflow.analyzers import RiskAnalyzer
from railguard.workflow.state import (
    HumanReviewDecision,
    ReviewState,
)

# 引用校验策略版本写入正式评测元数据。
CITATION_VALIDATOR_VERSION = "citation-validator-v2"


class WorkflowProtocolError(RuntimeError):
    """审核流程收到不符合内部协议的数据。"""


class AnalyzerProtocolError(WorkflowProtocolError):
    """风险Agent返回了无效条款或证据引用。"""


class HumanReviewProtocolError(WorkflowProtocolError):
    """人工审核恢复数据不符合约定格式。"""


class ReviewNodes:
    """保存审核流程依赖并提供全部LangGraph节点。"""

    def __init__(
        self,
        *,
        retriever: RagRetriever,
        commercial_analyzer: RiskAnalyzer,
        legal_analyzer: RiskAnalyzer,
        security_analyzer: RiskAnalyzer,
    ) -> None:
        """注入RAG检索器和三个风险分析Agent。"""
        self._retriever = retriever
        self._commercial_analyzer = commercial_analyzer
        self._legal_analyzer = legal_analyzer
        self._security_analyzer = security_analyzer

    async def initialize_review(
        self,
        state: ReviewState,
    ) -> dict[str, object]:
        """初始化一次新的合同审核任务。

        review_id必须与LangGraph配置中的thread_id保持一致，
        以便后续从SQLite checkpoint恢复。
        """
        review_id = state["review_id"]

        if not review_id.strip():
            raise WorkflowProtocolError(
                "review_id must not be blank"
            )

        return {
            "status": "retrieving",
            "evidence_by_clause": {},
            "contract_evidence": [],
            "commercial_findings": [],
            "legal_findings": [],
            "security_findings": [],
            "findings": [],
            "final_findings": [],
            "human_decision": None,
            "final_summary": "",
            "completed_nodes": ["initialize_review"],
        }

    async def retrieve_evidence(
        self,
        state: ReviewState,
    ) -> dict[str, object]:
        """为每个已有条款和缺失条款检查检索RAG证据。

        合法的空检索结果会被保存为空列表。
        RAG超时、连接失败或协议错误继续向上抛出，
        不能伪装为“没有风险依据”。
        """
        contract = state["contract"]
        evidence_by_clause: dict[str, list[Evidence]] = {}

        # 每个真实条款单独检索，保留条款与证据的对应关系。
        for clause in contract.clauses:
            evidence_by_clause[clause.clause_id] = (
                await self._retriever.retrieve(
                    RagSearchRequest(
                        clause_id=clause.clause_id,
                        query=clause.text,
                        top_k=3,
                    )
                )
            )

        # 缺失条款没有真实原文位置，因此该请求不传clause_id。
        # 查询覆盖首版三个Agent负责的企业审核类别。
        contract_evidence = await self._retriever.retrieve(
            RagSearchRequest(
                query=(
                    "付款、验收、知识产权、源代码、数据安全、"
                    "运维、质保、违约和解除"
                ),
                top_k=20,
            )
        )

        return {
            "status": "analyzing",
            "evidence_by_clause": evidence_by_clause,
            "contract_evidence": contract_evidence,
            "completed_nodes": ["retrieve_evidence"],
        }

    async def commercial_risk_agent(
        self,
        state: ReviewState,
    ) -> dict[str, object]:
        """运行商务风险Agent并保存独立输出。"""
        findings = await self._commercial_analyzer.analyze(
            contract=state["contract"],
            evidence_by_clause=state.get(
                "evidence_by_clause",
                {},
            ),
            contract_evidence=state.get(
                "contract_evidence",
                [],
            ),
        )

        return {
            "commercial_findings": findings,
            "completed_nodes": [
                self._commercial_analyzer.name
            ],
        }

    async def legal_risk_agent(
        self,
        state: ReviewState,
    ) -> dict[str, object]:
        """运行法务风险Agent并保存独立输出。"""
        findings = await self._legal_analyzer.analyze(
            contract=state["contract"],
            evidence_by_clause=state.get(
                "evidence_by_clause",
                {},
            ),
            contract_evidence=state.get(
                "contract_evidence",
                [],
            ),
        )

        return {
            "legal_findings": findings,
            "completed_nodes": [
                self._legal_analyzer.name
            ],
        }

    async def security_risk_agent(
        self,
        state: ReviewState,
    ) -> dict[str, object]:
        """运行数据安全Agent并保存独立输出。"""
        findings = await self._security_analyzer.analyze(
            contract=state["contract"],
            evidence_by_clause=state.get(
                "evidence_by_clause",
                {},
            ),
            contract_evidence=state.get(
                "contract_evidence",
                [],
            ),
        )

        return {
            "security_findings": findings,
            "completed_nodes": [
                self._security_analyzer.name
            ],
        }

    async def aggregate_findings(
        self,
        state: ReviewState,
    ) -> dict[str, object]:
        """按固定顺序汇总三个Agent的风险并验证条款ID。

        商务、法务和数据安全Agent可能并行完成，
        汇总节点负责产生稳定的最终排列顺序。
        """
        findings = [
            *state.get("commercial_findings", []),
            *state.get("legal_findings", []),
            *state.get("security_findings", []),
        ]

        valid_clause_ids = {
            clause.clause_id
            for clause in state["contract"].clauses
        }
        seen_finding_ids: set[str] = set()

        for finding in findings:
            if finding.finding_id in seen_finding_ids:
                raise AnalyzerProtocolError(
                    "Analyzer returned a duplicate finding_id"
                )

            seen_finding_ids.add(finding.finding_id)

            if (
                finding.finding_kind == "clause_risk"
                and finding.clause_id not in valid_clause_ids
            ):
                raise AnalyzerProtocolError(
                    "Analyzer referenced an unknown clause_id"
                )

        return {
            "status": "verifying",
            "findings": findings,
            "completed_nodes": ["aggregate_findings"],
        }

    async def verify_citations(
        self,
        state: ReviewState,
    ) -> dict[str, object]:
        """验证风险引用的证据真实存在且属于正确范围。

        该节点只确认引用ID、来源范围和风险类别能够匹配，
        不把这种匹配解释为法律适用结论已经正确。
        """
        verified_findings: list[RiskFinding] = []

        for finding in state.get("findings", []):
            allowed_evidence = self._allowed_evidence(
                state=state,
                finding=finding,
            )

            # 过滤无效引用但保留被拒绝ID，兼顾服务韧性和审计。
            valid_evidence_ids: list[str] = []
            rejected_evidence_ids = list(
                dict.fromkeys(finding.rejected_evidence_ids)
            )

            seen_evidence_ids = set(rejected_evidence_ids)

            for evidence_id in finding.evidence_ids:
                if evidence_id in seen_evidence_ids:
                    rejected_evidence_ids.append(evidence_id)
                    continue

                seen_evidence_ids.add(evidence_id)
                evidence = allowed_evidence.get(evidence_id)

                if (
                    evidence is None
                    or evidence.metadata.get("category")
                    != finding.category
                ):
                    rejected_evidence_ids.append(evidence_id)
                    continue

                valid_evidence_ids.append(evidence_id)

            if valid_evidence_ids and rejected_evidence_ids:
                citation_status = "partially_matched"
            elif valid_evidence_ids:
                citation_status = "source_matched"
            else:
                citation_status = "unsupported"

            verified_findings.append(
                finding.model_copy(
                    update={
                        "citation_status": citation_status,
                        "evidence_ids": valid_evidence_ids,
                        "rejected_evidence_ids": (
                            rejected_evidence_ids
                        ),
                    }
                )
            )

        next_status = (
            "awaiting_human"
            if verified_findings
            else "approved"
        )

        return {
            "status": next_status,
            "findings": verified_findings,
            "completed_nodes": ["verify_citations"],
        }

    async def human_review(
        self,
        state: ReviewState,
    ) -> dict[str, object]:
        """暂停流程并等待人工审核决定。

        interrupt之前只构造可序列化的展示数据，不写数据库、
        不发送通知，避免恢复节点时重复产生外部副作用。
        """
        evidence_by_id = self._all_evidence_by_id(state)
        referenced_evidence_ids = {
            evidence_id
            for finding in state.get("findings", [])
            for evidence_id in finding.evidence_ids
        }

        review_payload = {
            "review_id": state["review_id"],
            "contract_id": state["contract"].contract_id,
            "filename": state["contract"].filename,
            "findings": [
                finding.model_dump(mode="json")
                for finding in state.get("findings", [])
            ],
            "evidence": [
                evidence.model_dump(mode="json")
                for evidence_id, evidence in evidence_by_id.items()
                if evidence_id in referenced_evidence_ids
            ],
        }

        # 第一次运行在这里暂停。
        # 使用同一个thread_id恢复时，interrupt返回人工提交的数据。
        resume_value = interrupt(review_payload)

        try:
            decision = HumanReviewDecision.model_validate(
                resume_value
            )
        except ValidationError as exc:
            raise HumanReviewProtocolError(
                "Human review decision has an invalid format"
            ) from exc

        valid_finding_ids = {
            finding.finding_id
            for finding in state.get("findings", [])
        }
        unknown_finding_ids = (
            set(decision.finding_decisions)
            - valid_finding_ids
        )

        if unknown_finding_ids:
            raise HumanReviewProtocolError(
                "Human decision referenced an unknown finding_id"
            )

        status_by_action = {
            "approve": "approved",
            "reject": "rejected",
            "request_changes": "changes_requested",
        }

        return {
            "status": status_by_action[decision.action],
            "human_decision": decision,
            "completed_nodes": ["human_review"],
        }

    async def finalize_review(
        self,
        state: ReviewState,
    ) -> dict[str, object]:
        """生成最终审核摘要并结束流程。"""
        findings = state.get("findings", [])
        accepted_findings = self._accepted_findings(state)

        level_counts = {
            "high": 0,
            "medium": 0,
            "low": 0,
        }

        for finding in accepted_findings:
            level_counts[finding.level] += 1

        status = state.get("status", "approved")
        decision = state.get("human_decision")
        
        # 没有风险时会跳过人工审核，应显示为自动完成。
        if decision is None:
            status_text = "自动审核完成"
        else:
            status_text = {
                "approved": "人工审核通过",
                "rejected": "人工审核驳回",
                "changes_requested": "需要修改后重新提交",
            }.get(status, "审核完成")

        summary = (
            f"{status_text}。Agent共识别{len(findings)}项风险，"
            f"最终保留{len(accepted_findings)}项；"
            f"其中高风险{level_counts['high']}项、"
            f"中风险{level_counts['medium']}项、"
            f"低风险{level_counts['low']}项。"
        )

        return {
            "final_summary": summary,
            "completed_nodes": ["finalize_review"],
			"final_findings": accepted_findings,
        }

    @staticmethod
    def _allowed_evidence(
        *,
        state: ReviewState,
        finding: RiskFinding,
    ) -> dict[str, Evidence]:
        """返回一项风险可以合法引用的证据集合。

        条款级风险可以引用该条款检索到的证据，也可以引用合同级
        审查指引；缺失条款风险只能引用合同级证据。类别匹配由调用方
        单独校验。
        """
        if finding.finding_kind == "clause_risk":
            clause_evidence = state.get(
                "evidence_by_clause",
                {},
            ).get(
                finding.clause_id or "",
                [],
            )
            contract_evidence = state.get(
                "contract_evidence",
                [],
            )
            evidence_items = list(clause_evidence) + list(
                contract_evidence
            )
        else:
            evidence_items = state.get(
                "contract_evidence",
                [],
            )

        return {
            evidence.evidence_id: evidence
            for evidence in evidence_items
        }

    @staticmethod
    def _all_evidence_by_id(
        state: ReviewState,
    ) -> dict[str, Evidence]:
        """汇总当前审核中的全部证据并按ID索引。"""
        evidence_by_id: dict[str, Evidence] = {}

        for evidence_items in state.get(
            "evidence_by_clause",
            {},
        ).values():
            for evidence in evidence_items:
                evidence_by_id[evidence.evidence_id] = evidence

        for evidence in state.get(
            "contract_evidence",
            [],
        ):
            evidence_by_id[evidence.evidence_id] = evidence

        return evidence_by_id

    @staticmethod
    def _accepted_findings(
        state: ReviewState,
    ) -> list[RiskFinding]:
        """根据人工决定计算最终保留的风险列表。

        没有单独处理的风险保持Agent原始判断。
        全局驳回表示当前审核结果不被业务人员接受。
        """
        findings = state.get("findings", [])
        decision = state.get("human_decision")

        if decision is None:
            return list(findings)

        if decision.action == "reject":
            return []

        return [
            finding
            for finding in findings
            if decision.finding_decisions.get(
                finding.finding_id,
                "accept",
            )
            == "accept"
        ]


def route_after_verification(
    state: ReviewState,
) -> Literal["human_review", "finalize_review"]:
    """有风险时进入人工审核，没有风险时直接生成摘要。"""
    if state.get("findings"):
        return "human_review"

    return "finalize_review"