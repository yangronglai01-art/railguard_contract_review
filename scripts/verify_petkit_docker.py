"""Isolated Compose acceptance. Only this run's generated project/volume is removed."""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from railguard.petkit_demo import case_docx, load_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    output = args.output_dir or Path(tempfile.mkdtemp(prefix="railguard-container-"))
    if output.exists() and any(output.iterdir()):
        parser.error("Use a new empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    project = "rg-petkit-check-" + uuid.uuid4().hex[:12]
    compose = ["docker", "compose", "--env-file", str(ROOT / "configs/petkit-local.env"),
               "-p", project, "-f", str(ROOT / "compose.petkit-local.yaml")]
    def command(arguments, timeout=90):
        result = subprocess.run(arguments, cwd=ROOT, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=timeout, check=False)
        if result.returncode:
            # Do not dump arbitrary container stdout, env or credentials on failures.
            raise RuntimeError("Container command failed; inspect Docker locally")
        return result.stdout.strip()
    def execute(service, *arguments):
        return command(compose + ["exec", "-T", service, *arguments])
    def address(service, port):
        return "http://" + command(compose + ["port", service, str(port)]).splitlines()[-1]
    def ready(url, path):
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            try:
                response = httpx.get(url + path, timeout=2, trust_env=False)
                if response.status_code == 200:
                    return response
            except httpx.RequestError:
                pass
            time.sleep(.3)
        raise TimeoutError("Container endpoint not ready")
    def credentials():
        # Generated, isolated acceptance credentials remain internal to this process.
        return json.loads(execute("api", "python", "-c", "from pathlib import Path; print(Path('/app/data/runtime/petkit/credentials.json').read_text())"))
    def auth(values, subject):
        return {"Authorization": "Bearer " + values[subject]["token"]}
    report = {"passed": False, "mock_only": True, "project": project}
    try:
        if not args.skip_build:
            command(compose + ["build", "api"], timeout=900)
        command(compose + ["up", "-d", "--no-build"])
        api = address("api", 8000)
        ui = address("ui", 8501)
        profile = ready(api, "/profile").json()
        assert profile["auth_mode"] == "token" and profile["model_provider"] == "mock"
        ready(ui, "/_stcore/health")
        assert execute("api", "id", "-u") == execute("ui", "id", "-u") == "10001"
        assert "No broken requirements" in execute("api", "python", "-m", "pip", "check")
        code = ("import railguard,hashlib,json,pathlib; root=pathlib.Path(railguard.__file__).parent; "
                "print(json.dumps({p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*.py')}))")
        image_hashes = json.loads(execute("api", "python", "-c", code))
        source = ROOT / "src/railguard"
        expected = {p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob("*.py")}
        assert image_hashes == expected, "Image does not match current source; rebuild it"
        container_id = command(compose + ["ps", "-q", "api"])
        report["image_id"] = command(["docker", "inspect", container_id, "--format", "{{.Image}}"])
        report["source_tree_sha256"] = hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()
        report["lock_sha256"] = hashlib.sha256((ROOT / "requirements.lock").read_bytes()).hexdigest()
        assert httpx.get(api + "/reviews", timeout=5, trust_env=False).status_code == 401
        execute("api", "python", "-m", "railguard.identities", "provision", "--data-dir", "/app/data/runtime/petkit")
        values = credentials()
        permissions = execute("api", "python", "-c", "import os,stat; print(oct(stat.S_IMODE(os.stat('/app/data/runtime/petkit/credentials.json').st_mode)))")
        assert permissions == "0o600"
        with httpx.Client(base_url=api, timeout=40, trust_env=False) as client:
            uploaded = client.post("/contracts", headers=auth(values, "buyer-01"), files={"file": (
                "container-synthetic.docx", case_docx(load_cases()[0]), "application/octet-stream",
            )})
            assert uploaded.status_code == 201
            contract_id = uploaded.json()["contract_id"]
            assert client.get(f"/contracts/{contract_id}", headers=auth(values, "buyer-02")).status_code == 404
            assert client.put(f"/contracts/{contract_id}/reviewers/legal-01", headers=auth(values, "buyer-01")).status_code == 200
            reviewed = client.post(f"/contracts/{contract_id}/reviews", headers=auth(values, "buyer-01"))
            assert reviewed.status_code == 201
            review = reviewed.json()
            assert review["status"] == "awaiting_human" and len(review["findings"]) == 6
            assert all(finding["citation_status"] == "source_matched" for finding in review["findings"])
        command(compose + ["restart", "api"])
        api = address("api", 8000)
        ready(api, "/profile")
        with httpx.Client(base_url=api, timeout=30, trust_env=False) as client:
            restored = client.get(f"/reviews/{review['review_id']}", headers=auth(values, "legal-01"))
            assert restored.status_code == 200 and restored.json()["status"] == "awaiting_human"
            result = client.post(f"/reviews/{review['review_id']}/decision", headers=auth(values, "legal-01"), json={
                "action": "request_changes", "reviewer": "forged", "comment": "合成容器验收", "finding_decisions": {},
            })
            assert result.status_code == 200 and result.json()["human_decision"]["reviewer"] == "legal-01"
            assert client.delete(f"/contracts/{contract_id}/reviewers/legal-01", headers=auth(values, "buyer-01")).status_code == 200
            assert client.get(f"/reviews/{review['review_id']}", headers=auth(values, "legal-01")).status_code == 404
            audit = client.get("/security/audit", headers=auth(values, "auditor-01"))
            assert audit.status_code == 200
            assert all(value["token"] not in audit.text for value in values.values())
        report.update(passed=True, ui_health=True, nonroot_uid=10001, dependency_check=True,
                      source_hashes_match=True, personal_credential_mode="0600", anonymous_denied=True,
                      cross_owner_denied=True, six_risks_source_matched=True, restart_persists_permissions=True,
                      server_owned_reviewer=True, grant_revocation=True, metadata_audit=True)
    finally:
        assert re.fullmatch(r"rg-petkit-check-[a-f0-9]{12}", project)
        command(compose + ["down", "--volumes", "--remove-orphans"])
        assert not command(["docker", "ps", "-aq", "--filter", f"label=com.docker.compose.project={project}"])
        assert not command(["docker", "volume", "ls", "-q", "--filter", f"label=com.docker.compose.project={project}"])
        report["owned_project_removed"] = True
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
