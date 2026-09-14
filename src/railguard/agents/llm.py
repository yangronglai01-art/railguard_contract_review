"""基于严格结构化输出的大模型合同风险Agent。"""

from collections.abc import (
    Mapping,
    Sequence,
)
from typing import Protocol

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from pydantic import ValidationError

from railguard.agents.models import (
    LlmRiskAnalysis,
    LlmRiskCandidate,
)
from railguard.agents.prompts import (
    AgentProfile,
    build_system_prompt,
    build_user_prompt,
)
from railguard.models.schemas import (
    ContractDocument,
    Evidence,
    RiskFinding,
)

# 默认限制单次提交给模型的提示词字符数。
DEFAULT_MAX_INPUT_CHARS = 120_000

# 限制单个专业Agent一次最多返回的风险数量。
DEFAULT_MAX_FINDINGS = 12

# 限制一项风险最多引用的证据数量。
DEFAULT_MAX_EVIDENCE_IDS = 5

# 限制模型生成的风险理由长度。
DEFAULT_MAX_REASON_CHARS = 2_000

# 限制模型生成的修改建议长度。
DEFAULT_MAX_REVISION_CHARS = 3_000


class StructuredAnalysisInvoker(Protocol):
    """严格结构化模型调用器需要实现的最小接口。"""

    async def ainvoke(
        self,
        messages: Sequence[BaseMessage],
    ) -> object:
        """异步调用模型并返回结构化结果。"""
        ...


class LlmAnalyzerError(RuntimeError):
    """所有大模型风险Agent异常的公共基类。"""


class LlmInputTooLargeError(LlmAnalyzerError):
    """提交给模型的合同和证据超过本地字符限制。"""


class LlmInvocationError(LlmAnalyzerError):
    """模型服务调用失败或不可用。"""


class LlmResponseError(LlmAnalyzerError):
    """模型响应无法解析为约定的结构化模型。"""


class LlmProtocolError(LlmAnalyzerError):
    """模型响应违反本地业务边界或引用规则。"""


class LlmRiskAnalyzer:
    """调用结构化模型并验证风险结果的专业Agent。"""

    def __init__(
        self,
        *,
        profile: AgentProfile,
        invoker: StructuredAnalysisInvoker,
        max_input_chars: int = DEFAULT_MAX_INPUT_CHARS,
        max_findings: int = DEFAULT_MAX_FINDINGS,
        max_evidence_ids: int = DEFAULT_MAX_EVIDENCE_IDS,
        max_reason_chars: int = DEFAULT_MAX_REASON_CHARS,
        max_revision_chars: int = DEFAULT_MAX_REVISION_CHARS,
    ) -> None:
        """保存Agent职责、模型调用器和本地输出限制。"""
        self._validate_positive_limit(
            "max_input_chars",
            max_input_chars,
        )
        self._validate_positive_limit(
            "max_findings",
            max_findings,
        )
        self._validate_positive_limit(
            "max_evidence_ids",
            max_evidence_ids,
        )
        self._validate_positive_limit(
            "max_reason_chars",
            max_reason_chars,
        )
        self._validate_positive_limit(
            "max_revision_chars",
            max_revision_chars,
        )

        # name满足现有RiskAnalyzer协议，也用于工作流审计记录。
        self.name = profile.agent_name
        self._profile = profile
        self._invoker = invoker
        self._max_input_chars = max_input_chars
        self._max_findings = max_findings
        self._max_evidence_ids = max_evidence_ids
        self._max_reason_chars = max_reason_chars
        self._max_revision_chars = max_revision_chars

    async def analyze(
        self,
        *,
        contract: ContractDocument,
        evidence_by_clause: Mapping[
            str,
            Sequence[Evidence],
        ],
        contract_evidence: Sequence[Evidence],
    ) -> list[RiskFinding]:
        """调用模型并把验证后的候选结果转换为风险发现。"""
        system_prompt = build_system_prompt(self._profile)
        user_prompt = build_user_prompt(
            contract=contract,
            evidence_by_clause=evidence_by_clause,
            contract_evidence=contract_evidence,
        )

        # 字符限制在模型调用前执行，避免静默截断合同或证据。
        total_input_chars = len(system_prompt) + len(user_prompt)

        if total_input_chars > self._max_input_chars:
            raise LlmInputTooLargeError(
                "LLM input exceeds the configured character limit"
            )

        messages: list[BaseMessage] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        try:
            response = await self._invoker.ainvoke(messages)
        except LlmAnalyzerError:
            raise
        except Exception as exc:
            # 外部SDK异常统一转换，避免上层依赖具体模型供应商。
            raise LlmInvocationError(
                "Structured model invocation failed"
            ) from exc

        analysis = self._parse_response(response)

        if len(analysis.findings) > self._max_findings:
            raise LlmProtocolError(
                "LLM returned too many risk findings"
            )

        valid_clause_ids = {
            clause.clause_id
            for clause in contract.clauses
        }
        seen_candidates: set[
            tuple[str, str, str, str]
        ] = set()
        findings: list[RiskFinding] = []

        for candidate in analysis.findings:
            candidate_key = self._candidate_key(candidate)

            if candidate_key in seen_candidates:
                raise LlmProtocolError(
                    "LLM returned duplicate risk findings"
                )

            seen_candidates.add(candidate_key)

            findings.append(
                self._validate_and_convert_candidate(
                    candidate=candidate,
                    valid_clause_ids=valid_clause_ids,
                    evidence_by_clause=evidence_by_clause,
                    contract_evidence=contract_evidence,
                )
            )

        return findings

    @staticmethod
    def _validate_positive_limit(
        name: str,
        value: int,
    ) -> None:
        """验证构造参数中的数量限制必须为正整数。"""
        if value <= 0:
            raise ValueError(f"{name} must be greater than zero")

    @staticmethod
    def _parse_response(
        response: object,
    ) -> LlmRiskAnalysis:
        """把模型返回值解析成严格的风险分析模型。"""
        if isinstance(response, LlmRiskAnalysis):
            return response

        try:
            return LlmRiskAnalysis.model_validate(response)
        except (ValidationError, TypeError) as exc:
            raise LlmResponseError(
                "LLM returned an invalid structured response"
            ) from exc

    @staticmethod
    def _candidate_key(
        candidate: LlmRiskCandidate,
    ) -> tuple[str, str, str, str]:
        """生成风险业务身份，用于阻止同类重复输出。"""
        return (
            candidate.finding_kind,
            candidate.category,
            candidate.clause_id or "",
            (
                candidate.expected_clause.strip().casefold()
                if candidate.expected_clause is not None
                else ""
            ),
        )

    def _validate_and_convert_candidate(
        self,
        *,
        candidate: LlmRiskCandidate,
        valid_clause_ids: set[str],
        evidence_by_clause: Mapping[
            str,
            Sequence[Evidence],
        ],
        contract_evidence: Sequence[Evidence],
    ) -> RiskFinding:
        """验证分类、定位、文本和证据后创建风险发现。"""
        if (
            candidate.category
            not in self._profile.allowed_categories
        ):
            raise LlmProtocolError(
                "LLM returned a category outside the agent scope"
            )

        if (
            candidate.finding_kind == "clause_risk"
            and candidate.clause_id not in valid_clause_ids
        ):
            raise LlmProtocolError(
                "LLM referenced an unknown clause_id"
            )

        if len(candidate.reason) > self._max_reason_chars:
            raise LlmProtocolError(
                "LLM returned an oversized risk reason"
            )

        if (
            len(candidate.suggested_revision)
            > self._max_revision_chars
        ):
            raise LlmProtocolError(
                "LLM returned an oversized suggested revision"
            )

        if (
            len(candidate.evidence_ids)
            > self._max_evidence_ids
        ):
            raise LlmProtocolError(
                "LLM referenced too many evidence items"
            )

        allowed_evidence = self._allowed_evidence(
            candidate=candidate,
            evidence_by_clause=evidence_by_clause,
            contract_evidence=contract_evidence,
        )

        for evidence_id in candidate.evidence_ids:
            evidence = allowed_evidence.get(evidence_id)

            if evidence is None:
                raise LlmProtocolError(
                    "LLM referenced evidence outside its scope"
                )

            if (
                evidence.metadata.get("category")
                != candidate.category
            ):
                raise LlmProtocolError(
                    "LLM referenced evidence from another category"
                )

        # finding_id和citation_status由应用生成，模型无权控制。
        return RiskFinding(
            finding_kind=candidate.finding_kind,
            clause_id=candidate.clause_id,
            expected_clause=candidate.expected_clause,
            category=candidate.category,
            level=candidate.level,
            reason=candidate.reason.strip(),
            suggested_revision=(
                candidate.suggested_revision.strip()
            ),
            evidence_ids=list(candidate.evidence_ids),
        )

    @staticmethod
    def _allowed_evidence(
        *,
        candidate: LlmRiskCandidate,
        evidence_by_clause: Mapping[
            str,
            Sequence[Evidence],
        ],
        contract_evidence: Sequence[Evidence],
    ) -> dict[str, Evidence]:
        """返回当前风险类型和位置允许引用的证据。"""
        if candidate.finding_kind == "clause_risk":
            evidence_items = evidence_by_clause.get(
                candidate.clause_id or "",
                (),
            )
        else:
            evidence_items = contract_evidence

        return {
            evidence.evidence_id: evidence
            for evidence in evidence_items
        }