"""Run the offline regression without loading private model config or old runtime DBs."""

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.dont_write_bytecode = True

import pytest

from railguard.config import Settings


def main() -> int:
    os.chdir(ROOT)
    for name in Settings.model_fields:
        os.environ.pop(name.upper(), None)
    os.environ["RAILGUARD_ISOLATED_DEMO"] = "1"
    # Only this test process is changed, never the source defaults or user's .env.
    Settings.model_config["env_file"] = None
    temporary = tempfile.mkdtemp(prefix="railguard-petkit-tests-")
    return pytest.main([
        "-q", "-o", "addopts=", "-p", "no:cacheprovider", "--basetemp", temporary,
        *sys.argv[1:],
    ])


if __name__ == "__main__":
    raise SystemExit(main())
