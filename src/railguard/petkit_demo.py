"""Isolated localhost mock demo and reproducible synthetic DOCX fixture generator."""

import argparse
import json
import os
from io import BytesIO
from pathlib import Path

from railguard.business import PETKIT_CORPUS
from railguard.config import Settings

ROOT = Path(__file__).resolve().parents[2]
if not (ROOT / "data/demo").is_dir():
    ROOT = Path.cwd()
SAMPLES = ROOT / "data/demo/petkit-contract-cases-v1.json"


def demo_settings(data_dir: Path | None = None, *, secure: bool = False) -> Settings:
    """No .env, host API keys or legacy SQLite paths enter this profile."""
    data = (data_dir or ROOT / "data/runtime/petkit").resolve()
    defaults = {name: field.default for name, field in Settings.model_fields.items()}
    defaults.update(
        business_profile="petkit", app_name="RailGuard · 小佩业务模拟",
        app_env="local-simulation", model_provider="mock", model_name="petkit-rules-v1",
        rag_mode="mock", rag_mock_corpus_path=ROOT / PETKIT_CORPUS,
        database_path=data / "contracts.db", checkpoint_path=data / "checkpoints.sqlite3",
        auth_mode="token" if secure else "local_demo",
    )
    return Settings(_env_file=None, **defaults)


def case_docx(case: dict) -> bytes:
    """Turn versioned test paragraphs into an upload fixture, in memory."""
    from docx import Document

    document = Document()
    for paragraph in case["paragraphs"]:
        document.add_paragraph(paragraph)
    output = BytesIO()
    document.save(output)
    return output.getvalue()


def load_cases() -> list[dict]:
    return json.loads(SAMPLES.read_text(encoding="utf-8"))["cases"]


def create_demo_app(data_dir: Path | None = None, *, secure: bool = False):
    # main's legacy app is not started; avoid .env loading during its import too.
    previous = os.environ.get("RAILGUARD_ISOLATED_DEMO")
    os.environ["RAILGUARD_ISOLATED_DEMO"] = "1"
    try:
        from railguard.api.main import create_app
    finally:
        if previous is None:
            os.environ.pop("RAILGUARD_ISOLATED_DEMO", None)
        else:
            os.environ["RAILGUARD_ISOLATED_DEMO"] = previous

    return create_app(settings=demo_settings(data_dir, secure=secure))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["serve", "serve-secure", "samples"])
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--port", type=int, default=18020)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    settings = demo_settings(args.data_dir)
    if args.command == "samples":
        folder = settings.database_path.parent / "samples"
        folder.mkdir(parents=True, exist_ok=True)
        for case in load_cases():
            path = folder / (case["id"] + ".docx")
            # Existing uploads/examples are never overwritten.
            with path.open("xb") as target:
                target.write(case_docx(case))
        print(f"Synthetic upload fixtures: {folder}")
        return
    import uvicorn

    print("Local simulation: fixed mock rules, synthetic evidence, human review required.")
    uvicorn.run(create_demo_app(args.data_dir, secure=args.command == "serve-secure"), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
