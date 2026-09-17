"""合同风险分析Agent接口和本地演示实现。

三个演示Agent使用确定性规则产生风险，保证离线演示结果稳定。
后续真实大模型或微调模型只需实现RiskAnalyzer协议，
LangGraph的节点、状态和人工审核流程不需要改变。
"""

from collections.abc import Mapping, Sequence
from typing import Protocol

from railguard.models.schemas import (
    ContractDocument,
    Evidence,
    RiskFinding,
)


class RiskAnalyzer(Protocol):
    """所有风险分析Agent需要实现的统一接口。"""

    # Agent名称用于LangGraph节点和审计记录。
    name: str

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
        """分析合同并返回当前Agent负责的风险。"""
        ...


def _category_evidence_ids(
    evidence_items: Sequence[Evidence],
    category: str,
) -> list[str]:
    """提取指定风险类别的证据ID并保持原有顺序。"""
    evidence_ids: list[str] = []
    seen_ids: set[str] = set()

    for evidence in evidence_items:
        if evidence.metadata.get("category") != category:
            continue

        if evidence.evidence_id in seen_ids:
            continue

        seen_ids.add(evidence.evidence_id)
        evidence_ids.append(evidence.evidence_id)

    return evidence_ids


def _contains_any(
    text: str,
    markers: Sequence[str],
) -> bool:
    """判断文本中是否出现任意一个审核标记词。"""
    normalized_text = "".join(text.casefold().split())

    return any(
        "".join(marker.casefold().split())
        in normalized_text
        for marker in markers
    )


class DemoCommercialRiskAnalyzer:
    """检查付款和验收安排的演示商务风险Agent。"""

    name = "commercial_risk_agent"

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
        """检查已有条款中的全额预付和默认验收风险。

        contract_evidence属于统一Agent接口的一部分。
        本Agent只分析已有条款，因此主要使用evidence_by_clause。
        """
        findings: list[RiskFinding] = []

        for clause in contract.clauses:
            clause_text = "".join(clause.text.split())
            clause_evidence = evidence_by_clause.get(
                clause.clause_id,
                (),
            )

            # 合同签订后即支付全部价款，会使采购方在交付和
            # 验收前失去付款制约手段。
            if (
                "支付全部合同款" in clause_text
                and (
                    "签订后" in clause_text
                    or "生效后" in clause_text
                )
            ):
                findings.append(
                    RiskFinding(
                        finding_kind="clause_risk",
                        clause_id=clause.clause_id,
                        category="payment",
                        level="high",
                        reason=(
                            "合同约定在交付和验收前支付全部价款，"
                            "采购方缺少与交付成果挂钩的付款保障。"
                        ),
                        suggested_revision=(
                            "建议按合同签订、阶段交付和最终验收"
                            "设置分期付款，并保留验收后的尾款。"
                        ),
                        evidence_ids=_category_evidence_ids(
                            clause_evidence,
                            "payment",
                        ),
                    )
                )

            # 采购方沉默即视为验收合格，会把明确验收责任
            # 转换为被动的时间条件。
            if (
                "视为验收合格" in clause_text
                and (
                    "未提出异议" in clause_text
                    or "未书面提出异议" in clause_text
                )
            ):
                findings.append(
                    RiskFinding(
                        finding_kind="clause_risk",
                        clause_id=clause.clause_id,
                        category="acceptance",
                        level="high",
                        reason=(
                            "条款以采购方未提出异议作为验收合格"
                            "条件，缺少明确的测试标准和书面确认。"
                        ),
                        suggested_revision=(
                            "建议约定验收材料、测试标准、整改期限"
                            "和复验程序，并以双方书面验收确认为准。"
                        ),
                        evidence_ids=_category_evidence_ids(
                            clause_evidence,
                            "acceptance",
                        ),
                    )
                )

        return findings


class DemoLegalRiskAnalyzer:
    """检查关键合同条款缺失情况的演示法务Agent。"""

    name = "legal_risk_agent"

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
        """检查知识产权、违约责任和运维保障条款是否缺失。

        缺失条款没有原文位置，因此生成的RiskFinding不填写
        clause_id，只填写expected_clause。
        """
        findings: list[RiskFinding] = []

        if not _contains_any(
            contract.full_text,
            (
                "知识产权",
                "著作权",
                "源代码归属",
                "开发成果归属",
            ),
        ):
            findings.append(
                RiskFinding(
                    finding_kind="missing_clause",
                    expected_clause="知识产权和开发成果归属条款",
                    category="intellectual_property",
                    level="high",
                    reason=(
                        "合同未明确源代码、定制开发成果和接口文档"
                        "的权利归属，也未约定第三方侵权责任。"
                    ),
                    suggested_revision=(
                        "建议明确源代码、定制成果、技术文档的"
                        "所有权或使用权，并约定第三方侵权处理责任。"
                    ),
                    evidence_ids=_category_evidence_ids(
                        contract_evidence,
                        "intellectual_property",
                    ),
                )
            )

        if not _contains_any(
            contract.full_text,
            (
                "违约责任",
                "违约金",
                "赔偿责任",
            ),
        ):
            findings.append(
                RiskFinding(
                    finding_kind="missing_clause",
                    expected_clause="违约责任和合同解除条款",
                    category="liability",
                    level="high",
                    reason=(
                        "合同未约定延期交付、质量缺陷等情形的"
                        "违约责任和采购方解除合同的条件。"
                    ),
                    suggested_revision=(
                        "建议补充延期交付、质量缺陷和重大违约"
                        "的责任标准，并明确采购方的解除权。"
                    ),
                    evidence_ids=_category_evidence_ids(
                        contract_evidence,
                        "liability",
                    ),
                )
            )

        if not _contains_any(
            contract.full_text,
            (
                "质保期",
                "质量保证期",
                "运维服务",
                "维护服务",
                "售后服务",
            ),
        ):
            findings.append(
                RiskFinding(
                    finding_kind="missing_clause",
                    expected_clause="运维服务和质量保证条款",
                    category="support",
                    level="medium",
                    reason=(
                        "合同未明确质保期限、故障响应时间、"
                        "修复时间和服务期结束后的移交安排。"
                    ),
                    suggested_revision=(
                        "建议补充质保期、故障等级、响应与修复"
                        "时限，以及数据和技术文档移交要求。"
                    ),
                    evidence_ids=_category_evidence_ids(
                        contract_evidence,
                        "support",
                    ),
                )
            )

        return findings


class DemoSecurityRiskAnalyzer:
    """检查数据安全条款缺失情况的演示安全Agent。"""

    name = "security_risk_agent"

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
        """检查合同是否约定供应商的数据安全义务。

        当前项目面向高端装备制造企业的信息化与系统集成采购，
        供应商可能接触设备运行数据、工艺参数、账号和生产信息。
        """
        if _contains_any(
            contract.full_text,
            (
                "数据安全",
                "信息安全",
                "数据处理",
                "保密义务",
                "数据保密",
                "个人信息",
            ),
        ):
            return []

        return [
            RiskFinding(
                finding_kind="missing_clause",
                expected_clause="数据安全和保密义务条款",
                category="data_security",
                level="high",
                reason=(
                    "合同未约定供应商对设备运行数据、用户账号"
                    "和生产信息的访问、保存、删除与事件报告义务。"
                ),
                suggested_revision=(
                    "建议明确数据使用范围、访问权限、保存期限、"
                    "返还删除要求和安全事件报告时限。"
                ),
                evidence_ids=_category_evidence_ids(
                    contract_evidence,
                    "data_security",
                ),
            )
        ]