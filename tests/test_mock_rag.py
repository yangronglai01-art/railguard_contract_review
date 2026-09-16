"""本地Mock RAG检索器测试。"""

from pathlib import Path

import pytest

from railguard.rag.client import RagProtocolError
from railguard.rag.mock import MockRagRetriever
from railguard.rag.models import RagSearchRequest


def demo_corpus_path() -> Path:
    """返回项目内演示知识库文件路径。"""
    return Path("data/demo/rag-corpus.json")


async def test_mock_rag_returns_relevant_acceptance_evidence() -> None:
    """验证验收条款能够命中对应的企业内部规则。"""
    retriever = MockRagRetriever.from_json_file(
        demo_corpus_path()
    )

    evidence = await retriever.retrieve(
        RagSearchRequest(
            clause_id="clause-acceptance",
            query=(
                "系统上线三日后，甲方未提出异议，"
                "视为验收合格。"
            ),
            top_k=3,
        )
    )

    assert evidence
    assert evidence[0].source == "第六条 验收管理"
    assert evidence[0].metadata["category"] == "acceptance"
    assert evidence[0].metadata["retrieval_mode"] == "mock"

    # keywords只用于模拟匹配，不应出现在正式证据元数据中。
    assert "keywords" not in evidence[0].metadata

    # Mock相关度分数固定限制在0到1之间。
    assert evidence[0].score is not None
    assert 0 < evidence[0].score <= 1


async def test_mock_rag_returns_empty_list_without_keyword_match() -> None:
    """验证没有任何关键词命中时返回空列表。"""
    retriever = MockRagRetriever.from_json_file(
        demo_corpus_path()
    )

    evidence = await retriever.retrieve(
        RagSearchRequest(
            query="今天室外天气晴朗",
            top_k=3,
        )
    )

    assert evidence == []


async def test_mock_rag_respects_top_k_and_stable_order() -> None:
    """验证Mock检索结果数量受限且多次执行顺序一致。"""
    retriever = MockRagRetriever.from_json_file(
        demo_corpus_path()
    )
    request = RagSearchRequest(
        query=(
            "付款、合同款、验收、异议、知识产权、源代码、"
            "数据、安全、运维、质保、违约和解除"
        ),
        top_k=2,
    )

    first_result = await retriever.retrieve(request)
    second_result = await retriever.retrieve(request)

    assert len(first_result) == 2
    assert len(second_result) == 2

    # 同一知识片段使用稳定ID，便于跨检索范围校验引用。
    first_signature = [
        (item.evidence_id, item.document_id, item.source, item.score)
        for item in first_result
    ]
    second_signature = [
        (item.evidence_id, item.document_id, item.source, item.score)
        for item in second_result
    ]

    assert first_signature == second_signature


def test_mock_rag_rejects_invalid_corpus(
    tmp_path: Path,
) -> None:
    """验证损坏的知识库文件不会被静默接受。"""
    invalid_corpus = tmp_path / "invalid-rag-corpus.json"
    invalid_corpus.write_text(
        '{"unexpected": []}',
        encoding="utf-8",
    )

    with pytest.raises(RagProtocolError):
        MockRagRetriever.from_json_file(invalid_corpus)