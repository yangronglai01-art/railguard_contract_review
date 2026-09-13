"""合同文档解析模块。

负责从DOCX和文本型PDF中提取文字，并生成统一的规范化文本。
本模块只负责文本提取，条款切分将在后续模块中实现。
"""

import re
from io import BytesIO
from pathlib import Path

import pymupdf
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph


# 演示版本允许上传的最大文件大小为10MB。
DEFAULT_MAX_FILE_SIZE = 10 * 1024 * 1024

# 当前明确支持的合同文件扩展名。
SUPPORTED_EXTENSIONS = {".docx", ".pdf"}


class DocumentParserError(Exception):
    """所有文档解析异常的公共基类。"""


class UnsupportedFileTypeError(DocumentParserError):
    """上传文件类型不在允许范围内。"""


class DocumentTooLargeError(DocumentParserError):
    """上传文件超过最大体积限制。"""


class EmptyDocumentError(DocumentParserError):
    """文档中没有可以用于审核的文本。"""


class EncryptedDocumentError(DocumentParserError):
    """PDF受到密码保护，当前无法读取。"""


class InvalidDocumentError(DocumentParserError):
    """文件内容损坏，或实际格式与扩展名不一致。"""


def normalize_text(text: str) -> str:
    """将不同来源的文本转换成统一格式。

    处理规则：

    1. 将Windows和旧Mac换行符统一为换行符；
    2. 删除每行末尾的空格和制表符；
    3. 将多个连续空行压缩为一个空行；
    4. 删除文档首尾空白。

    条款定位将基于该函数返回的规范化文本。
    生成条款后不能再次修改合同全文，否则字符偏移会失效。
    """
    # 将不同操作系统的换行符统一为\n。
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")

    # 删除每行末尾的空格和制表符，保留行首缩进。
    lines = [
        re.sub(r"[ \t]+$", "", line)
        for line in normalized.split("\n")
    ]

    # 用于保存压缩空行后的文本行。
    compacted_lines: list[str] = []

    for line in lines:
        # 使用strip判断当前行是否只包含空白字符。
        is_blank = not line.strip()

        if is_blank:
            # 文本开头不保留空行。
            if not compacted_lines:
                continue

            # 连续多个空行只保留一个。
            if compacted_lines[-1] != "":
                compacted_lines.append("")
        else:
            compacted_lines.append(line)

    # 删除文本末尾可能残留的空行。
    while compacted_lines and compacted_lines[-1] == "":
        compacted_lines.pop()

    return "\n".join(compacted_lines)


def extract_docx_text(content: bytes) -> str:
    """从DOCX字节中提取段落和表格文字。

    使用iter_inner_content按照文档中的实际顺序遍历段落和表格。
    表格中的每一行使用制表符连接单元格，便于保留字段关系。
    """
    try:
        # BytesIO将内存中的字节包装成python-docx可读取的文件对象。
        document = Document(BytesIO(content))
    except Exception as exc:
        # 将第三方库异常转换为项目自己的稳定异常类型。
        raise InvalidDocumentError(
            "DOCX文件损坏或格式不正确。"
        ) from exc

    # 保存文档中的段落和表格文本。
    blocks: list[str] = []

    for block in document.iter_inner_content():
        if isinstance(block, Paragraph):
            # 普通段落直接保留文本。
            if block.text.strip():
                blocks.append(block.text)

        elif isinstance(block, Table):
            # 表格逐行读取，每一行生成一段文本。
            for row in block.rows:
                cells = [
                    cell.text.strip()
                    for cell in row.cells
                ]
                row_text = "\t".join(cells)

                if row_text.strip():
                    blocks.append(row_text)

    return normalize_text("\n\n".join(blocks))


def extract_pdf_text(content: bytes) -> str:
    """从文本型PDF字节中提取每一页的文字。

    当前版本不执行OCR。扫描件如果没有文本层，
    最终会触发EmptyDocumentError。
    """
    try:
        # 直接从内存打开PDF，不需要先保存上传文件。
        with pymupdf.open(
            stream=content,
            filetype="pdf",
        ) as document:
            # 加密PDF必须先解密才能读取。
            if document.needs_pass:
                raise EncryptedDocumentError(
                    "PDF受到密码保护，当前无法解析。"
                )

            # 分页提取文本，并用空行分隔页面。
            pages = [
                page.get_text("text")
                for page in document
            ]
    except DocumentParserError:
        # 保留我们主动抛出的业务异常。
        raise
    except Exception as exc:
        # 屏蔽PyMuPDF的内部异常类型，保持API返回稳定。
        raise InvalidDocumentError(
            "PDF文件损坏或格式不正确。"
        ) from exc

    return normalize_text("\n\n".join(pages))


def parse_document(
    filename: str,
    content: bytes,
    max_size_bytes: int = DEFAULT_MAX_FILE_SIZE,
) -> str:
    """根据文件名和实际内容解析合同。

    参数：
        filename：用户上传的原始文件名。
        content：文件的二进制内容。
        max_size_bytes：允许的最大文件大小。

    返回：
        经过规范化的合同全文。

    异常：
        UnsupportedFileTypeError：扩展名不支持；
        DocumentTooLargeError：文件过大；
        InvalidDocumentError：内容格式不正确；
        EmptyDocumentError：没有提取到有效文本。
    """
    # 统一转换为小写扩展名，支持.DOCX和.PDF等写法。
    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"不支持的文件类型：{extension or '无扩展名'}。"
        )

    if len(content) > max_size_bytes:
        raise DocumentTooLargeError(
            f"文件超过{max_size_bytes}字节限制。"
        )

    if not content:
        raise EmptyDocumentError("上传文件为空。")

    if extension == ".pdf":
        # PDF标准文件头应以%PDF-开始。
        if not content.startswith(b"%PDF-"):
            raise InvalidDocumentError(
                "文件扩展名为PDF，但内容不是有效PDF。"
            )

        text = extract_pdf_text(content)

    else:
        # DOCX本质上是ZIP包，正常文件以PK签名开始。
        if not content.startswith(b"PK"):
            raise InvalidDocumentError(
                "文件扩展名为DOCX，但内容不是有效DOCX。"
            )

        text = extract_docx_text(content)

    # 解析完成后必须包含非空文本。
    if not text.strip():
        raise EmptyDocumentError(
            "文档中没有可用于审核的文本。"
        )

    return text