"""Bounded deterministic simulation; not a general legal reasoning engine."""

from railguard.agents.guardrails import calibrated_risk_level
from railguard.business import petkit_candidate_is_eligible
from railguard.models.schemas import RiskFinding

RULES = {
    "payment": ("付款保障", "交付验收前的全额或高比例付款削弱采购方保障。", "将付款与里程碑及书面验收挂钩，保留尾款。"),
    "acceptance": ("验收标准", "自动验收或核心验收事项待协商，缺少可执行确认。", "明确功能、设备连接和数据导出测试、整改复验及书面确认。"),
    "intellectual_property": ("知识产权与使用许可", "合同缺少或未落实成果权利和采购使用许可。", "区分定制固件/App成果与标准云平台，明确成果权利或订阅许可范围。"),
    "liability": ("违约赔偿与解除", "延期、质量缺陷或重大违约的责任安排未落实。", "约定整改、赔偿、退款及采购方解除条件。"),
    "support": ("运维服务与退出", "故障响应、修复或服务保障未落实。", "明确质保与响应修复时间、设备服务连续性、退出导出与移交。"),
    "data_security": ("数据安全与处理范围", "账号、设备遥测或宠物影像相关的数据处理保护未落实。", "限定目的、范围和最小权限，约定返还删除、安全事件及分包责任。"),
}


class PetkitRuleAnalyzer:
    """Reuse explicit precision gates; no implicit pass from rule silence."""

    def __init__(self, name: str, categories: tuple[str, ...]):
        self.name = name
        self.categories = categories

    async def analyze(self, *, contract, evidence_by_clause, contract_evidence):
        findings = []
        for category in self.categories:
            title, reason, revision = RULES[category]
            clauses = [clause for clause in contract.clauses if petkit_candidate_is_eligible(
                contract=contract, finding_kind="clause_risk", clause_id=clause.clause_id,
                category=category,
            )]
            missing = not clauses and petkit_candidate_is_eligible(
                contract=contract, finding_kind="missing_clause", clause_id=None, category=category,
            )
            for clause in clauses or ([None] if missing else []):
                kind = "clause_risk" if clause is not None else "missing_clause"
                evidence = evidence_by_clause.get(clause.clause_id, ()) if clause else contract_evidence
                evidence_ids = list(dict.fromkeys(
                    item.evidence_id for item in evidence if item.metadata.get("category") == category
                ))
                findings.append(RiskFinding(
                    finding_kind=kind,
                    clause_id=clause.clause_id if clause else None,
                    expected_clause=None if clause else title,
                    category=category,
                    level=calibrated_risk_level(
                        contract=contract, finding_kind=kind,
                        clause_id=clause.clause_id if clause else None, category=category,
                    ),
                    reason=reason, suggested_revision=revision, evidence_ids=evidence_ids,
                ))
        return findings


def create_petkit_rule_analyzers():
    from railguard.agents.provider import RiskAnalyzerSet

    return RiskAnalyzerSet(
        commercial=PetkitRuleAnalyzer("commercial_risk_agent", ("payment", "acceptance")),
        legal=PetkitRuleAnalyzer("legal_risk_agent", ("intellectual_property", "liability", "support")),
        security=PetkitRuleAnalyzer("security_risk_agent", ("data_security",)),
    )
