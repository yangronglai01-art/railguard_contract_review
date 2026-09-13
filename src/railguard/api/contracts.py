"""合同上传与查询接口。"""

from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    UploadFile,
    status,
)

from railguard.api.dependencies import get_contract_repository
from railguard.models.schemas import ContractDocument
from railguard.parsers.documents import (
    DEFAULT_MAX_FILE_SIZE,
    DocumentTooLargeError,
    EmptyDocumentError,
    EncryptedDocumentError,
    InvalidDocumentError,
    UnsupportedFileTypeError,
    parse_document,
)
from railguard.storage.contracts import ContractRepository

# 所有接口统一使用/contracts前缀。
router = APIRouter(
    prefix="/contracts",
    tags=["contracts"],
)


@router.post(
    "",
    response_model=ContractDocument,
    status_code=status.HTTP_201_CREATED,
)
async def upload_contract(
    file: Annotated[UploadFile, File(...)],
    repository: Annotated[
        ContractRepository,
        Depends(get_contract_repository),
    ],
) -> ContractDocument:
    """上传、解析并保存一份合同。

    文件采用限量读取：

    1. 最多读取10MB加1字节；
    2. 如果读到额外字节，解析器会判断文件超限；
    3. 无论解析是否成功，都会关闭上传文件；
    4. 成功后保存规范化全文；
    5. 条款列表暂时为空，下一阶段加入条款切分。
    """
    # UploadFile的filename允许为空，因此需要提供空字符串回退值。
    filename = file.filename or ""

    try:
        # 只读取最大允许大小加1字节，避免无限制占用内存。
        content = await file.read(DEFAULT_MAX_FILE_SIZE + 1)
    finally:
        # 无论读取或解析是否成功，都及时关闭临时上传文件。
        await file.close()

    try:
        # 根据文件扩展名、文件头和实际内容提取规范化文本。
        full_text = parse_document(
            filename=filename,
            content=content,
        )
    except DocumentTooLargeError as exc:
        # 文件过大使用HTTP 413。
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=str(exc),
        ) from exc
    except UnsupportedFileTypeError as exc:
        # 不支持的媒体类型使用HTTP 415。
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(exc),
        ) from exc
    except (
        EmptyDocumentError,
        EncryptedDocumentError,
        InvalidDocumentError,
    ) as exc:
        # 格式允许但无法处理的文档使用HTTP 422。
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc

    # 创建通过Pydantic校验的合同业务模型。
    contract = ContractDocument(
        filename=filename,
        full_text=full_text,
    )

    # 将合同完整JSON保存到SQLite。
    repository.save(contract)

    return contract


@router.get(
    "/{contract_id}",
    response_model=ContractDocument,
)
def get_contract(
    contract_id: str,
    repository: Annotated[
        ContractRepository,
        Depends(get_contract_repository),
    ],
) -> ContractDocument:
    """根据合同ID读取已经保存的合同。"""
    contract = repository.get(contract_id)

    if contract is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Contract not found.",
        )

    return contract