"""大模型合同风险候选项的本地精确率护栏。"""

import re
from typing import Literal

from railguard.models.schemas import ContractDocument

PRECISION_GUARDRAIL_VERSION = "precision-guardrail-v1"

RiskLevel = Literal["low", "medium", "high"]

_PERCENTAGE_PATTERN = re.compile(
    r"(?:支付|预付|首付款)[^%]{0,24}?(?P<percent>[0-9]{1,3})%"
)

_NEGOTIATION_MARKERS = (
    "另行协商",
    "协商确定",
    "另行约定",
    "后续确定",
    "后续另议",
    "另议",
)


def _compact(text: str) -> str:
    """移除空白并统一英文大小写，供确定性边界判断使用。"""
    return "".join(text.casefold().split())


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    """判断规范化文本中是否包含任意一个业务标记。"""
    return any(_compact(marker) in text for marker in markers)


def _clause_text(
    contract: ContractDocument,
    clause_id: str | None,
) -> str:
    """按条款ID返回规范化原文，未知或空ID返回空字符串。"""
    if clause_id is None:
        return ""

    return next(
        (
            _compact(clause.text)
            for clause in contract.clauses
            if clause.clause_id == clause_id
        ),
        "",
    )


def _payment_clause_is_risky(text: str) -> bool:
    """识别明确全额预付或把付款核心安排留待协商的条款。"""
    if _contains_any(
        text,
        (
            "不得支付全部合同款",
            "不应支付全部合同款",
            "禁止支付全部合同款",
        ),
    ):
        return False

    negotiated_later = (
        _contains_any(
            text,
            ("付款安排", "付款方式", "支付安排"),
        )
        and _contains_any(text, _NEGOTIATION_MARKERS)
    )
    full_prepayment = (
        _contains_any(
            text,
            (
                "支付全部合同款",
                "支付全部服务费用",
                "结清合同价款总额的100%",
                "支付合同价款总额的100%",
                "支付100%合同价款",
            ),
        )
        and _contains_any(
            text,
            (
                "签订后",
                "生效后",
                "盖章",
                "收到款项后开始交付",
                "交付前",
            ),
        )
    )
    percentage_prepayment = (
        _contains_any(
            text,
            ("签订后", "生效后", "交付前", "首付款"),
        )
        and any(
            int(match.group("percent")) >= 80
            for match in _PERCENTAGE_PATTERN.finditer(text)
        )
    )
    return (
        negotiated_later
        or full_prepayment
        or percentage_prepayment
    )


def _acceptance_clause_is_risky(text: str) -> bool:
    """识别自动验收或把验收核心事项留待协商的条款。"""
    if (
        _contains_any(text, ("不得以", "不以"))
        and _contains_any(
            text,
            ("视为验收合格", "自动通过验收"),
        )
    ):
        return False

    automatic_acceptance = _contains_any(
        text,
        (
            "未提出异议视为验收合格",
            "未回复即自动通过验收",
            "自动通过验收",
            "视为验收合格",
            "默认验收合格",
            "逾期未反馈则验收通过",
            "逾期未答复则验收通过",
        ),
    )
    negotiated_later = (
        "验收" in text
        and _contains_any(text, _NEGOTIATION_MARKERS)
    )
    return automatic_acceptance or negotiated_later


def _clause_risk_is_eligible(
    *,
    category: str,
    text: str,
) -> bool:
    """按风险分类判断已有条款是否达到高置信度输出门槛。"""
    if category == "payment":
        return _payment_clause_is_risky(text)
    if category == "acceptance":
        return _acceptance_clause_is_risky(text)
    if category == "intellectual_property":
        return (
            _contains_any(text, ("知识产权", "著作权", "源代码"))
            and _contains_any(
                text,
                _NEGOTIATION_MARKERS + ("协商不成",),
            )
        )
    if category == "liability":
        return (
            _contains_any(text, ("违约", "赔偿", "解除"))
            and _contains_any(
                text,
                _NEGOTIATION_MARKERS
                + (
                    "按照法律规定",
                    "依照法律规定",
                    "依法处理",
                ),
            )
        )
    if category == "support":
        return (
            _contains_any(
                text,
                ("质保", "运维", "维护", "售后", "故障"),
            )
            and _contains_any(text, _NEGOTIATION_MARKERS)
        )
    if category == "data_security":
        return (
            _contains_any(
                text,
                ("数据", "保密", "信息安全"),
            )
            and _contains_any(
                text,
                _NEGOTIATION_MARKERS,
            )
        )
    return False


def _data_processing_is_applicable(text: str) -> bool:
    """判断合同是否会使供应商接触系统或生产经营数据。"""
    return _contains_any(
        text,
        (
            "数据",
            "软件",
            "系统",
            "平台",
            "信息化",
            "远程",
            "账号",
            "设备运维",
            "技术服务",
            "mes",
            "erp",
            "qms",
        ),
    )


def _missing_clause_is_eligible(
    *,
    contract_text: str,
    category: str,
) -> bool:
    """仅在完整合同没有达到最低覆盖线时允许缺失条款风险。"""
    coverage_markers: dict[str, tuple[str, ...]] = {
        "payment": ("付款", "合同款", "价款", "服务费用"),
        "acceptance": ("验收", "测试", "复验"),
        "intellectual_property": (
            "知识产权",
            "著作权",
            "源代码归属",
            "开发成果",
        ),
        "liability": (
            "违约责任",
            "违约金",
            "赔偿责任",
            "有权解除合同",
        ),
        "support": (
            "运维服务",
            "维护服务",
            "售后服务",
            "故障响应",
            "故障修复",
        ),
        "data_security": (
            "数据安全",
            "信息安全",
            "保密义务",
            "数据保密",
            "返还或删除数据",
            "删除甲方数据",
        ),
    }
    markers = coverage_markers.get(category)
    if markers is None:
        return False
    if category == "data_security" and not _data_processing_is_applicable(
        contract_text
    ):
        return False
    return not _contains_any(contract_text, markers)


def risk_candidate_is_eligible(
    *,
    contract: ContractDocument,
    finding_kind: str,
    clause_id: str | None,
    category: str,
) -> bool:
    """判断模型候选风险是否满足本地高置信度业务门槛。"""
    if finding_kind == "clause_risk":
        return _clause_risk_is_eligible(
            category=category,
            text=_clause_text(contract, clause_id),
        )
    if finding_kind == "missing_clause":
        return _missing_clause_is_eligible(
            contract_text=_compact(contract.full_text),
            category=category,
        )
    return False


def calibrated_risk_level(
    *,
    contract: ContractDocument,
    finding_kind: str,
    clause_id: str | None,
    category: str,
) -> RiskLevel:
    """按照首版业务口径校准高置信度候选项的风险等级。"""
    if category == "support":
        if finding_kind == "missing_clause":
            return "medium"

        text = _clause_text(contract, clause_id)
        if "一年质保期" in text and "另行约定" in text:
            return "low"
        return "medium"

    return "high"
