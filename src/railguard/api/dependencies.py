"""FastAPI依赖项。"""

from fastapi import Request

from railguard.storage.contracts import ContractRepository


def get_contract_repository(
    request: Request,
) -> ContractRepository:
    """从当前FastAPI应用中取得合同存储实例。

    存储实例在应用生命周期启动阶段创建。
    测试可以为create_app传入临时存储，避免污染真实数据库。
    """
    repository = getattr(
        request.app.state,
        "contract_repository",
        None,
    )

    if repository is None:
        raise RuntimeError(
            "Contract repository is not initialized."
        )

    return repository