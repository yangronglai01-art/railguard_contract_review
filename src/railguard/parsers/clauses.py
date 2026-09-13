"""合同条款切分模块。

当前版本采用可解释的规则切分合同：
识别独占一行的“第X条”标题，并保存条款在全文中的字符位置。

规则无法识别标题时，将整份合同作为一个条款，
避免生成无法追溯到原文的虚假切分结果。
"""

import re

from railguard.models.schemas import Clause

# 支持“第一条”“第十二条”“第1条”等常见合同标题。
#
# ^和$要求标题从一行开头开始，并延续到该行结尾。
# (?!款)用于排除“第一条款……”这样的普通正文。
ARTICLE_HEADING_PATTERN = re.compile(
    (
        r"^(?P<title>"
        r"第[0-9〇零一二三四五六七八九十百千万]+条"
        r"(?!款)[^\n]*"
        r")$"
    ),
    flags=re.MULTILINE,
)


def _build_clause(
    full_text: str,
    start_offset: int,
    end_offset: int,
    title: str,
) -> Clause | None:
    """根据合同全文和字符范围创建条款。

    下一条标题前通常存在一个或两个换行符。
    这些换行符只用于分隔条款，不属于条款正文，
    因此从当前条款末尾移除。

    返回None表示该范围只有空白字符。
    """
    # 只删除条款末尾的换行符。
    # 不删除其他字符，保证原文定位仍然准确。
    while (
        end_offset > start_offset
        and full_text[end_offset - 1] == "\n"
    ):
        end_offset -= 1

    # 提取当前字符范围对应的原文。
    clause_text = full_text[start_offset:end_offset]

    # 空白范围不生成条款。
    if not clause_text.strip():
        return None

    return Clause(
        title=title,
        text=clause_text,
        start_offset=start_offset,
        end_offset=end_offset,
    )


def split_clauses(full_text: str) -> list[Clause]:
    """将规范化合同全文切分为可定位的条款列表。

    处理流程：

    1. 查找全部“第X条”标题；
    2. 将第一个标题之前的内容保存为“合同前言”；
    3. 每个标题到下一个标题之间形成一个条款；
    4. 没有识别到标题时，将全文作为一个条款。

    每个Clause.text都必须满足：

        full_text[start_offset:end_offset] == clause.text
    """
    # 查找所有符合规则的条款标题。
    heading_matches = list(
        ARTICLE_HEADING_PATTERN.finditer(full_text)
    )

    # 如果没有标准标题，整份文档作为一个条款。
    if not heading_matches:
        return [
            Clause(
                title="合同全文",
                text=full_text,
                start_offset=0,
                end_offset=len(full_text),
            )
        ]

    clauses: list[Clause] = []

    # 第一个条款标题前可能包含合同名称、双方信息和签约背景。
    first_heading_start = heading_matches[0].start()

    preamble = _build_clause(
        full_text=full_text,
        start_offset=0,
        end_offset=first_heading_start,
        title="合同前言",
    )

    if preamble is not None:
        clauses.append(preamble)

    # 每个标题从自身起始位置延续到下一个标题之前。
    for index, heading_match in enumerate(heading_matches):
        start_offset = heading_match.start()

        # 最后一个条款延续到合同全文结束。
        if index + 1 == len(heading_matches):
            end_offset = len(full_text)
        else:
            end_offset = heading_matches[index + 1].start()

        clause = _build_clause(
            full_text=full_text,
            start_offset=start_offset,
            end_offset=end_offset,
            title=heading_match.group("title"),
        )

        if clause is not None:
            clauses.append(clause)

    return clauses