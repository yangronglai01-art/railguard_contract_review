"""RailGuard核心业务数据模型。

本模块定义合同、条款、RAG证据和风险发现的数据结构。
后续FastAPI接口、LangGraph状态、数据库和前端都使用这些结构，
避免不同模块使用不一致的字段名称和数据格式。
"""

from typing import Literal, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


def new_id() -> str:
    """生成不带连字符的UUID字符串。

    当前演示项目直接在应用层生成ID，便于创建合同、条款、
    证据和风险记录。以后接入数据库时也可以继续使用这些ID。
    """
    return uuid4().hex


class SchemaModel(BaseModel):
    """所有RailGuard业务模型的公共基类。

    extra="forbid"表示输入出现未声明字段时立即报错。
    这样可以尽早发现Agent字段拼写错误或接口结构漂移。
    """

    model_config = ConfigDict(extra="forbid")


class Clause(SchemaModel):
    """从合同全文中提取出来的一项条款。

    start_offset和end_offset对应规范化后合同全文的Python字符索引，
    使用左闭右开区间：[start_offset, end_offset)。

    例如：
        full_text[0:5]

    表示从索引0开始，读取到索引5之前的字符。
    合同完成解析后，不应再次修改full_text，否则条款位置会失效。
    """

    # 条款唯一标识，默认自动生成。
    clause_id: str = Field(default_factory=new_id)

    # 条款标题，例如“付款方式”或“知识产权”。
    # 某些合同没有明确标题，因此允许为空字符串。
    title: str = ""

    # 条款原文，至少包含一个字符。
    text: str = Field(min_length=1)

    # 条款在合同全文中的起始字符位置。
    start_offset: int = Field(ge=0)

    # 条款在合同全文中的结束字符位置，不包含该位置对应的字符。
    end_offset: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_offsets(self) -> Self:
        """验证条款结束位置必须位于起始位置之后。

        Pydantic先完成所有字段解析，再执行该方法。
        返回self表示模型通过校验，可以继续使用。
        """
        if self.end_offset <= self.start_offset:
            raise ValueError("end_offset must be greater than start_offset")

        return self


class ContractDocument(SchemaModel):
    """经过解析和规范化的合同文档。

    当前演示版本固定采用采购方立场，只处理软件采购与技术服务合同。
    clauses中的所有位置都必须对应本模型的full_text。
    """

    # 合同唯一标识，默认自动生成。
    contract_id: str = Field(default_factory=new_id)

    # 用户上传的原始文件名。
    filename: str = Field(min_length=1)

    # 合同类型。首版只支持软件采购合同。
    contract_type: str = "software_purchase"

    # 审核立场。当前项目固定为采购方。
    party_position: Literal["buyer"] = "buyer"

    # 从PDF或DOCX提取并规范化后的完整合同文本。
    full_text: str = Field(min_length=1)

    # 从全文中提取的合同条款列表。
    clauses: list[Clause] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_clauses(self) -> Self:
        """验证合同中的条款ID、顺序、位置和原文一致性。

        校验内容包括：

        1. 同一份合同内的条款ID不能重复；
        2. 条款结束位置不能超过合同全文长度；
        3. 条款必须按照原文顺序排列；
        4. 条款位置不能重叠；
        5. 根据位置截取的原文必须等于条款文本。
        """
        # 保存已经出现过的条款ID，用于检测重复ID。
        seen_ids: set[str] = set()

        # 记录上一个条款的结束位置，用于检查排序和重叠。
        previous_end = 0

        for clause in self.clauses:
            # 防止同一合同中出现重复的条款ID。
            if clause.clause_id in seen_ids:
                raise ValueError("clause_id must be unique within a contract")

            seen_ids.add(clause.clause_id)

            # 防止条款位置超过合同全文范围。
            if clause.end_offset > len(self.full_text):
                raise ValueError("clause offsets exceed contract text")

            # 后一个条款不能从前一个条款结束位置之前开始。
            if clause.start_offset < previous_end:
                raise ValueError("clauses must be ordered and must not overlap")

            # 按照字符位置从合同全文中截取原始内容。
            original = self.full_text[
                clause.start_offset:clause.end_offset
            ]

            # 如果截取内容与条款文本不同，说明定位信息已经失效。
            if original != clause.text:
                raise ValueError(
                    "clause text does not match its source span"
                )

            # 保存当前条款结束位置，供下一个条款进行比较。
            previous_end = clause.end_offset

        return self


class Evidence(SchemaModel):
    """RAG知识库返回的一条证据片段。

    Evidence只表示检索到的资料和出处，不直接表示资料一定适用于
    当前风险。引用验证Agent还需要进一步检查来源与内容。

    同一份知识库文档可以返回多个证据片段，所以evidence_id和
    document_id是两个不同字段。
    """

    # 当前证据片段的唯一标识。
    evidence_id: str = Field(default_factory=new_id)

    # 证据所属知识库文档的稳定标识。
    document_id: str = Field(min_length=1)

    # 文档名称，例如《企业软件采购合同审核规则》。
    title: str = Field(min_length=1)

    # RAG实际返回的原始证据片段。
    content: str = Field(min_length=1)

    # 证据出处，例如“数据安全规则第3条”。
    source: str = Field(min_length=1)

    # 检索服务返回的相关度分数。
    # 不同RAG系统的分数定义可能不同，因此当前不限制取值范围。
    score: float | None = None

    # RAG返回的补充元数据，例如知识库版本和资料类型。
    metadata: dict[str, str] = Field(default_factory=dict)


class RiskFinding(SchemaModel):
    """合同审核产生的一项风险发现。

    风险分为两类：

    clause_risk：
        针对合同中已经存在的条款，必须提供clause_id。

    missing_clause：
        合同缺少应有条款，不存在可以定位的原文，因此不能提供
        clause_id，但必须说明缺少什么条款。
    """

    # 风险发现的唯一标识。
    finding_id: str = Field(default_factory=new_id)

    # 风险发现类型，默认为针对现有条款的风险。
    finding_kind: Literal[
        "clause_risk",
        "missing_clause",
    ] = "clause_risk"

    # 风险对应的合同条款ID。
    # 缺失条款没有原文，因此该字段必须为空。
    clause_id: str | None = None

    # 缺失条款的名称或要求。
    # 只有finding_kind为missing_clause时才要求填写。
    expected_clause: str | None = None

    # 风险分类，例如payment、data_security或intellectual_property。
    category: str = Field(min_length=1)

    # 风险等级，用于排序和决定是否进入人工审核。
    level: Literal["low", "medium", "high"]

    # 风险判断理由。
    reason: str = Field(min_length=1)

    # 建议替换或补充的合同文本。
    suggested_revision: str | None = None

    # 通过引用验证、能够支撑本项风险的RAG证据片段ID。
    evidence_ids: list[str] = Field(default_factory=list)

    # 模型曾返回但未通过范围或类别校验的证据ID。
    # 保留这些ID用于审计模型幻觉，不将其作为有效证据展示。
    rejected_evidence_ids: list[str] = Field(default_factory=list)

    # 引用状态：
    #
    # pending：
    #     尚未经过引用验证Agent处理。
    #
    # source_matched：
    #     引用ID和证据原文可以匹配，但不表示法律适用必然正确。
    #
    # partially_matched：
    #     至少一项引用合法，但模型还返回了被拒绝的引用ID。
    #
    # unsupported：
    #     找不到能够支持当前结论的证据。
    #
    # not_required：
    #     当前发现属于企业内部规则判断，不要求法律引用。
    citation_status: Literal[
        "pending",
        "source_matched",
        "partially_matched",
        "unsupported",
        "not_required",
    ] = "pending"

    @model_validator(mode="after")
    def validate_location(self) -> Self:
        """验证风险类型与原文定位字段是否一致。

        已存在条款的风险必须指向真实条款。
        缺失条款不能伪造原文位置，并且必须说明预期条款。
        """
        if self.finding_kind == "clause_risk":
            # 针对现有条款的风险必须提供条款ID。
            if not self.clause_id:
                raise ValueError("clause risk requires clause_id")
        else:
            # 缺失条款没有原文锚点，不能填写clause_id。
            if self.clause_id is not None:
                raise ValueError(
                    "missing clause must not have a source clause_id"
                )

            # 缺失条款必须说明合同中应该包含什么内容。
            if not self.expected_clause or not self.expected_clause.strip():
                raise ValueError("missing clause requires expected_clause")

        return self