"""本地Mock RAG检索器。

Mock检索器读取演示知识库，根据人工配置的关键词进行确定性匹配。
它用于本地演示和自动化测试，不模拟真实的向量相似度检索。
"""

from hashlib import sha256
from pathlib import Path
from typing import Self

from pydantic import ValidationError

from railguard.models.schemas import Evidence
from railguard.rag.client import RagProtocolError
from railguard.rag.models import (
    RagHit,
    RagSearchRequest,
    RagSearchResponse,
)


class MockRagRetriever:
    """使用本地演示资料执行关键词检索。"""

    def __init__(self, hits: list[RagHit]) -> None:
        """保存演示知识库中的资料片段。

        使用深拷贝，避免外部代码修改传入模型后影响检索结果。
        """
        self._hits = tuple(
            hit.model_copy(deep=True)
            for hit in hits
        )

    @classmethod
    def from_json_file(cls, path: Path) -> Self:
        """从JSON文件读取并校验演示知识库。

        文件无法读取、JSON格式错误或字段不符合RAG协议时，
        抛出RagProtocolError，避免带着损坏的数据继续审核。
        """
        try:
            # utf-8-sig同时兼容普通UTF-8和带BOM的UTF-8文件。
            content = path.read_text(encoding="utf-8-sig")
        except OSError as exc:
            raise RagProtocolError(
                f"Unable to read mock RAG corpus: {path}"
            ) from exc

        try:
            corpus = RagSearchResponse.model_validate_json(content)
        except (ValidationError, ValueError) as exc:
            raise RagProtocolError(
                f"Mock RAG corpus has an invalid format: {path}"
            ) from exc

        return cls(corpus.hits)

    async def retrieve(
        self,
        request: RagSearchRequest,
    ) -> list[Evidence]:
        """根据查询中的关键词返回排序后的演示证据。

        没有任何关键词命中时返回空列表。
        相同输入始终得到相同顺序和分数，方便演示与测试。
        """
        scored_hits: list[tuple[float, RagHit]] = []

        for hit in self._hits:
            score = self._calculate_score(
                query=request.query,
                hit=hit,
            )

            # 得分为0表示没有配置的关键词命中。
            if score > 0:
                scored_hits.append((score, hit))

        # 先按分数降序排列。
        # 分数相同时使用文档ID和出处保证顺序稳定。
        scored_hits.sort(
            key=lambda item: (
                -item[0],
                item[1].document_id,
                item[1].source,
            )
        )

        return [
            self._to_evidence(hit=hit, score=score)
            for score, hit in scored_hits[: request.top_k]
        ]

    @staticmethod
    def _normalize_text(text: str) -> str:
        """统一英文大小写并移除空白，便于关键词匹配。"""
        return "".join(text.casefold().split())

    @classmethod
    def _keywords(cls, hit: RagHit) -> tuple[str, ...]:
        """从资料元数据中读取并规范化关键词列表。"""
        raw_keywords = hit.metadata.get("keywords", "")

        return tuple(
            normalized
            for keyword in raw_keywords.split(",")
            if (
                normalized := cls._normalize_text(keyword)
            )
        )

    @classmethod
    def _calculate_score(
        cls,
        *,
        query: str,
        hit: RagHit,
    ) -> float:
        """计算一条演示资料与查询的确定性匹配分数。

        基础分数是命中关键词占全部关键词的比例。
        如果命中的词同时出现在资料出处中，再增加少量标题相关奖励。
        最终分数限制在0到1之间。
        """
        normalized_query = cls._normalize_text(query)
        keywords = cls._keywords(hit)

        if not keywords:
            return 0.0

        matched_keywords = tuple(
            keyword
            for keyword in keywords
            if keyword in normalized_query
        )

        if not matched_keywords:
            return 0.0

        normalized_source = cls._normalize_text(hit.source)

        source_bonus = (
            0.1
            if any(
                keyword in normalized_source
                for keyword in matched_keywords
            )
            else 0.0
        )

        score = len(matched_keywords) / len(keywords)
        return round(min(1.0, score + source_bonus), 4)

    @staticmethod
    def _to_evidence(
        *,
        hit: RagHit,
        score: float,
    ) -> Evidence:
        """把Mock命中项转换为内部Evidence模型。

        keywords只用于Mock匹配，不作为正式证据元数据返回。
        retrieval_mode用于明确标识证据来自本地模拟知识库。
        """
        metadata = {
            key: value
            for key, value in hit.metadata.items()
            if key != "keywords"
        }
        metadata["retrieval_mode"] = "mock"

        identity = (
            f"{hit.document_id}\x1f{hit.source}\x1f"
            f"{hit.content}"
        )
        evidence_id = (
            "mock-"
            + sha256(identity.encode("utf-8")).hexdigest()[:24]
        )

        return Evidence(
            evidence_id=evidence_id,
            document_id=hit.document_id,
            title=hit.title,
            content=hit.content,
            source=hit.source,
            score=score,
            metadata=metadata,
        )