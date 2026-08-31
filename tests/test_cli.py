import json
import os
import subprocess
import sys


def test_validate_cli_from_repository_root(repo_root):
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-m", "morphofeatures", "validate", "--json"],
        cwd=repo_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert all(item["passed"] for item in payload["arrays"].values())


def test_streamlit_launcher_targets_dashboard():
    from morphofeatures.ui import build_streamlit_command, dashboard_path

    command = build_streamlit_command(port=8765, address="127.0.0.1", headless=True)
    assert dashboard_path().name == "dashboard.py"
    assert command[1:4] == ["-m", "streamlit", "run"]
    assert "8765" in command
    assert "127.0.0.1" in command
    assert command[-2:] == ["--browser.gatherUsageStats", "false"]
