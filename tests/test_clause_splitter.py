"""规则型合同条款切分测试。"""

from railguard.parsers.clauses import split_clauses


def test_split_contract_with_preamble_and_articles() -> None:
    """验证合同前言和多个标准条款能够正确切分。"""
    full_text = (
        "设备监测平台软件采购合同\n\n"
        "第一条 项目内容\n\n"
        "乙方为甲方建设设备监测平台。\n\n"
        "第二条 付款方式\n\n"
        "甲方在验收合格后支付尾款。"
    )

    clauses = split_clauses(full_text)

    assert [clause.title for clause in clauses] == [
        "合同前言",
        "第一条 项目内容",
        "第二条 付款方式",
    ]

    assert "乙方为甲方建设" in clauses[1].text
    assert "验收合格后支付" in clauses[2].text


def test_every_clause_matches_its_source_span() -> None:
    """验证每个条款都能通过字符位置还原到合同原文。"""
    full_text = (
        "软件采购合同\n\n"
        "第一条 交付\n\n"
        "乙方应在三十日内完成交付。\n\n"
        "第二条 验收\n\n"
        "甲方按照需求说明书进行验收。"
    )

    clauses = split_clauses(full_text)

    for clause in clauses:
        original = full_text[
            clause.start_offset:clause.end_offset
        ]

        assert original == clause.text


def test_contract_without_headings_uses_full_text() -> None:
    """验证没有标准标题时不会凭空推测条款边界。"""
    full_text = "双方约定，系统验收合格后支付全部合同款。"

    clauses = split_clauses(full_text)

    assert len(clauses) == 1
    assert clauses[0].title == "合同全文"
    assert clauses[0].text == full_text
    assert clauses[0].start_offset == 0
    assert clauses[0].end_offset == len(full_text)


def test_regular_use_of_word_clause_is_not_a_heading() -> None:
    """验证“第一条款”这样的正文不会被识别为第一条标题。"""
    full_text = "第一条款说明仅用于介绍付款条件。"

    clauses = split_clauses(full_text)

    assert len(clauses) == 1
    assert clauses[0].title == "合同全文"