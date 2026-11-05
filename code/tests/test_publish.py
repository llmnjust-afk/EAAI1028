import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "publish_results.py"


def _repo(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email",
                    "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name",
                    "Test"], check=True)
    return tmp_path


def _run(repo, *args):
    return subprocess.run([sys.executable, str(SCRIPT), "--repo", str(repo), *args],
                          text=True, capture_output=True)


def test_publish_commits_only_the_requested_paths(tmp_path):
    repo = _repo(tmp_path)
    (repo / "results").mkdir()
    (repo / "results" / "a.json").write_text("{}\n")
    (repo / "PAT").write_text("ghp_example_token\n")
    (repo / ".gitignore").write_text("PAT\n")
    result = _run(repo, "--no-push", "--paths", "results",
                  "--message", "results")
    assert result.returncode == 0, result.stderr
    tracked = subprocess.run(["git", "-C", str(repo), "ls-files"],
                             text=True, capture_output=True, check=True).stdout
    assert "results/a.json" in tracked
    assert "PAT" not in tracked.splitlines()


def test_publish_refuses_a_secret_bearing_path(tmp_path):
    repo = _repo(tmp_path)
    (repo / "PAT").write_text("ghp_example_token\n")
    result = _run(repo, "--no-push", "--paths", "PAT")
    assert result.returncode != 0
    assert "secret" in result.stderr.lower() or "secret" in result.stdout.lower()


def test_dry_run_is_side_effect_free(tmp_path):
    repo = _repo(tmp_path)
    (repo / "results").mkdir()
    (repo / "results" / "a.json").write_text("{}\n")
    result = _run(repo, "--dry-run", "--paths", "results")
    assert result.returncode == 0
    staged = subprocess.run(["git", "-C", str(repo), "diff", "--cached",
                             "--name-only"], text=True, capture_output=True,
                            check=True).stdout
    assert staged.strip() == ""
