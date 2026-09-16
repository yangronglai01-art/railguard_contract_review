"""离线评测报告的可公开实验追溯信息。"""

import subprocess
from hashlib import sha256
from pathlib import Path


def sha256_file(path: Path) -> str:
    """计算文件SHA-256，固定数据集和RAG语料版本。"""
    digest = sha256()

    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(64 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def git_metadata(repository: Path) -> dict[str, str]:
    """读取当前Git提交和工作区状态，不把失败变成评测阻塞。"""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {
            "git_commit": "unavailable",
            "worktree_dirty": "unknown",
        }

    return {
        "git_commit": commit,
        "worktree_dirty": str(bool(status.strip())).lower(),
    }
