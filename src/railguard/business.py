"""Versioned simulated business profiles; no real enterprise policies."""

from dataclasses import replace

from railguard.agents.prompts import COMMERCIAL_PROFILE, LEGAL_PROFILE, SECURITY_PROFILE

PETKIT_RULE_VERSION = "petkit-procurement-demo-v1"
PETKIT_CORPUS = "data/demo/rag-corpus-petkit-v1.json"
PETKIT_CONTEXT = (
    "业务背景为小佩智能宠物硬件的自建模拟采购场景，非真实企业制度。"
    "仅审核软件、云服务、设备数据平台及技术服务采购，甲方为采购方。"
    "对象包括喂食器、饮水机、猫砂盆相关App、设备遥测、用户账号、"
    "宠物影像和远程运维；影像可能连带家庭环境信息。"
    "只按给定合成规则核对合同，不引用或编造法律条文，不作最终法律判断。"
    "不扩展到OEM、经销代理、消费者协议或跨境合规结论。"
)


def profiles_for_business(name: str):
    """Keep frozen legacy prompts intact; version the Petkit variant separately."""
    profiles = (COMMERCIAL_PROFILE, LEGAL_PROFILE, SECURITY_PROFILE)
    if name == "baus":
        return profiles
    if name != "petkit":
        raise ValueError("Unsupported business profile")
    commercial = replace(
        COMMERCIAL_PROFILE,
        review_scope=PETKIT_CONTEXT + COMMERCIAL_PROFILE.review_scope,
    )
    legal = replace(
        LEGAL_PROFILE,
        review_scope=PETKIT_CONTEXT + (
            "检查定制固件、App接口和开发成果的权利，以及云服务订阅使用许可。"
            "标准SaaS不以源代码归采购方为必要条件；明确订阅范围、许可期限和退出"
            "数据导出安排即可，不把供应商保留基础平台知识产权本身视为风险。"
            "定制开发与标准订阅分开判断。检查延期或质量缺陷的责任、赔偿及解除，"
            "以及故障响应、修复、质保和退出移交。已有明确保护时不因可选细节追加风险；"
            "核心事项缺失或全部留待另行协商时才报告。"
        ),
    )
    security = replace(
        SECURITY_PROFILE,
        review_scope=PETKIT_CONTEXT + (
            "核对用户账号、设备遥测、宠物影像和售后记录的处理范围与使用目的、最小权限、"
            "返还删除和分包边界。区分宠物数据与可能包含家庭环境或人员的影像。"
            "明确数据安全或保密义务并限制处理范围或约定结束后返还删除，视为基础保护覆盖；"
            "不因缺少可选细节追加风险。核心范围、保存及删除全部留待另行协商则报告。"
        ),
    )
    return commercial, legal, security


def profile_manifest(name: str) -> dict:
    """Expose business scope without endpoint credentials or database paths."""
    if name not in {"petkit", "baus"}:
        raise ValueError("Unsupported business profile")
    petkit = name == "petkit"
    return {
        "business_profile": name,
        "display_name": "小佩业务模拟采购合同审核" if petkit else "鲍斯业务模拟采购合同审核",
        "synthetic": True,
        "party_position": "buyer",
        "rule_version": PETKIT_RULE_VERSION if petkit else "baus-demo-1",
        "prompt_version": "contract-risk-v3-petkit-1" if petkit else "contract-risk-v3",
        "supported_scope": ["软件采购", "云服务订阅", "设备数据平台", "相关技术服务"],
        "unsupported_scope": ["OEM制造", "经销代理", "消费者协议", "跨境合规专项"],
        "review_mode": "辅助识别与人工复核，非自动法律批准",
    }


def validate_petkit_corpus(path):
    """Reject foreign/mixed demo policies even if renamed to the Petkit path."""
    from railguard.rag.models import RagSearchResponse

    corpus = RagSearchResponse.model_validate_json(path.read_text(encoding="utf-8-sig"))
    categories = {"payment", "acceptance", "intellectual_property", "liability", "support", "data_security"}
    if {hit.metadata.get("category") for hit in corpus.hits} != categories or any(
        hit.metadata.get("business_profile") != "petkit"
        or hit.metadata.get("version") != PETKIT_RULE_VERSION
        or hit.metadata.get("document_type") != "synthetic_demo_policy"
        for hit in corpus.hits
    ):
        raise ValueError("Petkit demo requires its complete, versioned synthetic policy corpus")


def petkit_candidate_is_eligible(*, contract, finding_kind, clause_id, category):
    """Do not demand SaaS source ownership when an explicit license is present."""
    from railguard.agents.guardrails import risk_candidate_is_eligible

    if category == "intellectual_property" and finding_kind == "missing_clause":
        for clause in contract.clauses:
            text = "".join(clause.text.split())
            if (any(word in text for word in ("使用许可", "订阅许可", "许可使用"))
                    and "期限" in text and "范围" in text
                    and not any(word in text for word in ("另行协商", "另行约定", "后续确定", "不授予", "未授予", "无使用许可"))):
                return False
    return risk_candidate_is_eligible(
        contract=contract, finding_kind=finding_kind, clause_id=clause_id, category=category,
    )
