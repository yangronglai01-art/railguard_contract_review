"""Local HTTP/UI acceptance of personal access; isolated data, no paid APIs."""

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from railguard.petkit_demo import case_docx, load_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or Path(tempfile.mkdtemp(prefix="railguard-access-"))
    if output.exists() and any(output.iterdir()):
        parser.error("Use a new empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    interpreter = sys.executable
    if sys.platform == "win32":
        env["__PYVENV_LAUNCHER__"] = sys.executable
        interpreter = sys._base_executable
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    data = output / "data"
    def cli(command, *extra):
        result = subprocess.run([interpreter, "-B", "-m", "railguard.identities", command,
                                 "--data-dir", str(data), *extra], cwd=ROOT, env=env,
                                capture_output=True, creationflags=flags, check=False)
        if result.returncode:
            raise RuntimeError("Local identity command failed")
    cli("provision")
    credentials = json.loads((data / "credentials.json").read_text(encoding="utf-8"))
    def auth(subject):
        return {"Authorization": "Bearer " + credentials[subject]["token"]}
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    processes = []
    logs = []
    def launch():
        log = (output / f"api-{len(processes)}.log").open("w", encoding="utf-8")
        logs.append(log)
        process = subprocess.Popen([interpreter, "-B", "-m", "railguard.petkit_demo", "serve-secure",
                                    "--data-dir", str(data), "--port", str(port)],
                                   cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
        processes.append(process)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("Local API exited; inspect private acceptance logs")
            try:
                response = httpx.get(url + "/profile", timeout=1, trust_env=False)
                if response.status_code == 200 and response.json().get("auth_mode") == "token":
                    return process
            except httpx.RequestError:
                pass
            time.sleep(.1)
        raise TimeoutError("Local API not ready")
    def stop(process):
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    try:
        process = launch()
        with httpx.Client(base_url=url, timeout=30, trust_env=False) as client:
            assert client.get("/reviews").status_code == 401
            response = client.post("/contracts", headers=auth("buyer-01"), files={"file": (
                "synthetic-private.docx", case_docx(load_cases()[0]), "application/octet-stream",
            )})
            assert response.status_code == 201
            contract_id = response.json()["contract_id"]
            assert client.get(f"/contracts/{contract_id}", headers=auth("buyer-02")).status_code == 404
            assert client.put(f"/contracts/{contract_id}/reviewers/legal-01", headers=auth("buyer-01")).status_code == 200
            review = client.post(f"/contracts/{contract_id}/reviews", headers=auth("buyer-01")).json()
            assert review["status"] == "awaiting_human"
        os.environ["RAILGUARD_API_BASE_URL"] = url
        from streamlit.testing.v1 import AppTest

        def login(subject):
            page = AppTest.from_file(str(ROOT / "src/railguard/ui/app.py")).run(timeout=30)
            assert not page.exception
            next(field for field in page.text_input if field.label == "个人访问凭据").input(credentials[subject]["token"])
            next(button for button in page.button if button.label == "登录").click()
            page.run(timeout=30)
            assert not page.exception and page.session_state["identity"]["subject"] == subject
            next(button for button in page.button if button.label == "读取我的审核任务").click()
            page.run(timeout=30)
            next(button for button in page.button if button.label == "打开选中任务").click()
            page.run(timeout=30)
            assert not page.exception and page.session_state["contract_snapshot"]["contract_id"] == contract_id
            return page
        buyer = login("buyer-01")
        legal = login("legal-01")
        assert buyer.session_state["identity"]["subject"] == "buyer-01"
        assert legal.session_state["identity"]["subject"] == "legal-01"
        next(button for button in buyer.button if button.label == "退出登录").click()
        buyer.run(timeout=30)
        assert not buyer.exception and buyer.session_state["contract_snapshot"] is None
        assert legal.session_state["identity"]["subject"] == "legal-01"
        stop(process)
        process = launch()
        with httpx.Client(base_url=url, timeout=30, trust_env=False) as client:
            assert client.get(f"/reviews/{review['review_id']}", headers=auth("legal-01")).status_code == 200
            old = credentials["legal-01"]["token"]
            cli("rotate", "--subject", "legal-01")
            credentials = json.loads((data / "credentials.json").read_text(encoding="utf-8"))
            assert client.get("/security/me", headers={"Authorization": "Bearer " + old}).status_code == 401
            result = client.post(f"/reviews/{review['review_id']}/decision", headers=auth("legal-01"), json={
                "action": "request_changes", "reviewer": "forged-name", "comment": "合成验收", "finding_decisions": {},
            })
            assert result.status_code == 200 and result.json()["human_decision"]["reviewer"] == "legal-01"
            audit = client.get("/security/audit", headers=auth("auditor-01"))
            assert audit.status_code == 200
            assert all(value["token"] not in audit.text for value in credentials.values())
        report = {"passed": True, "mock_only": True, "authenticated_http": True, "cross_owner_denied": True,
                  "separate_ui_sessions": True, "logout_clears_contract": True, "restart_restores_permissions": True,
                  "rotation_revokes_old_token": True, "server_owned_reviewer": True, "audit_metadata_only": True}
    finally:
        for process in processes:
            stop(process)
        for log in logs:
            log.close()
    report["owned_processes_stopped"] = all(process.poll() is not None for process in processes)
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))
    print(f"Private acceptance artifacts: {output}")


if __name__ == "__main__":
    main()
