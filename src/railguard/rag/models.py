"""外部RAG服务的请求与响应模型。

当前接口格式是RailGuard暂定的通信协议。
以后接入真实知识库时，只调整RAG适配器中的请求和响应转换，
合同审核流程继续使用已有的Evidence业务模型。
"""

from pydantic import Field, field_validator

from railguard.models.schemas import SchemaModel


class RagSearchRequest(SchemaModel):
    """发送给外部RAG服务的一次条款检索请求。"""

    # 检索已有条款时传入条款ID。
    # 检查缺失条款时没有原文位置，因此允许不传。
    clause_id: str | None = Field(default=None, min_length=1)

    # 当前直接使用条款原文作为检索内容。
    query: str = Field(min_length=1)

    # 希望返回的证据数量，限制在1到20之间。
    # strict=True防止字符串或布尔值被自动转换为整数。
    top_k: int = Field(default=3, ge=1, le=20, strict=True)

    @field_validator("clause_id", "query")
    @classmethod
    def validate_non_blank(cls, value: str | None) -> str | None:
        """允许不传条款ID，但拒绝只有空白字符的字符串。

        query是否必填由字段定义校验；此方法不修改原始文本。
        """
        if value is not None and not value.strip():
            raise ValueError("value must not contain only whitespace")

        return value


class RagHit(SchemaModel):
    """外部RAG服务返回的一条检索结果。

    字段名称与Evidence业务模型保持一致，便于适配器转换。
    检索结果只提供资料与出处，不代表已经完成风险判断。
    """

    # 证据所属知识库文档的稳定标识。
    document_id: str = Field(min_length=1)

    # 知识库文档名称。
    title: str = Field(min_length=1)

    # 检索返回的原始资料片段。
    # 保留原文，供后续引用验证使用。
    content: str = Field(min_length=1)

    # 资料出处，例如内部制度的章节或条款编号。
    source: str = Field(min_length=1)

    # 检索相关度分数。
    # 不限定分数范围，但拒绝NaN和无穷大。
    score: float | None = Field(default=None, allow_inf_nan=False)

    # 补充信息，例如资料类型、知识库版本。
    # 当前与Evidence一致，键和值都使用字符串。
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("document_id", "title", "content", "source")
    @classmethod
    def validate_non_blank(cls, value: str) -> str:
        """拒绝只有空白字符的必填资料字段。

        不对原始内容进行裁剪，避免破坏后续引用比对。
        """
        if not value.strip():
            raise ValueError("value must not contain only whitespace")

        return value


class RagSearchResponse(SchemaModel):
    """外部RAG服务的一次完整检索响应。"""

    # 必须明确返回hits字段。
    # hits=[]表示检索成功但没有结果；
    # 缺少hits则属于接口格式错误，不能当作没有检索结果。
    hits: list[RagHit]