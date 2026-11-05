"""Commit and push generated experiment artefacts without exposing the token.

The token is read from ``--pat-file`` or ``GITHUB_PAT`` and passed to git through
a temporary ``GIT_ASKPASS`` helper.  It never appears in the remote URL, in the
git configuration, in argv, or in the logs.  Only the requested paths are
staged, so a stray ``PAT`` or ``.env`` file cannot be committed accidentally.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


DEFAULT_PATHS = ("code/experiment_results", "code/audit_results", "figures")
PAPER_PATHS = ("main.tex", "main.pdf", "sections", "refs.bib", "highlights.tex")
SECRET_NAMES = {"PAT", ".env", ".netrc", "credentials", "id_rsa", "id_ed25519"}


def _git(repo: Path, *args, env=None, check=True):
    result = subprocess.run(["git", "-C", str(repo), *args], env=env,
                            text=True, capture_output=True)
    if check and result.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed:\n{result.stderr.strip()}")
    return result


def _read_token(args) -> str:
    if args.pat_file:
        token = Path(args.pat_file).read_text().strip()
    else:
        token = os.environ.get("GITHUB_PAT", "").strip()
    if not token:
        raise SystemExit("no token: pass --pat-file or set GITHUB_PAT")
    if any(ch.isspace() for ch in token):
        raise SystemExit("token contains whitespace; refusing to use it")
    return token


def _askpass_script(directory: Path, token: str) -> Path:
    script = directory / "askpass.sh"
    script.write_text(
        "#!/bin/sh\n"
        "case \"$1\" in\n"
        "  *Username*) printf '%s\\n' 'x-access-token' ;;\n"
        "  *Password*) cat \"$PAPERGURU_PAT_FILE\" ;;\n"
        "  *) printf '%s\\n' '' ;;\n"
        "esac\n")
    script.chmod(0o700)
    return script


def _matches_secret(path: str) -> bool:
    return any(part in SECRET_NAMES for part in Path(path).parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path,
                        default=Path(__file__).resolve().parents[2])
    parser.add_argument("--pat-file", type=Path, default=None)
    parser.add_argument("--paths", nargs="+", default=list(DEFAULT_PATHS))
    parser.add_argument("--include-paper", action="store_true",
                        help="also stage the manuscript, figures and bibliography")
    parser.add_argument("--message", default="Update experiment artefacts")
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--branch", default=None)
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo = args.repo.resolve()
    if not (repo / ".git").exists():
        raise SystemExit(f"not a git repository: {repo}")
    paths = list(args.paths) + (list(PAPER_PATHS) if args.include_paper else [])
    paths = [path for path in dict.fromkeys(paths) if (repo / path).exists()]
    if not paths:
        raise SystemExit("none of the requested paths exist")
    if any(_matches_secret(path) for path in paths):
        raise SystemExit("refusing to stage a secret-bearing path")

    branch = (args.branch
              or _git(repo, "symbolic-ref", "--short", "HEAD", check=False).stdout.strip()
              or "main")
    if not (args.no_push or args.dry_run):
        remote = _git(repo, "remote", "get-url", args.remote, check=False)
        if remote.returncode:
            raise SystemExit(f"remote {args.remote!r} is not configured")
        if not remote.stdout.strip().startswith("https://"):
            raise SystemExit("remote is not HTTPS; configure an HTTPS remote for token auth")

    token = None if (args.dry_run or args.no_push) else _read_token(args)
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    tempdir = Path(tempfile.mkdtemp(prefix="eaai-publish-"))
    try:
        if token is not None:
            pat_file = tempdir / "pat"
            pat_file.write_text(token)
            pat_file.chmod(0o600)
            askpass = _askpass_script(tempdir, token)
            env["GIT_ASKPASS"] = str(askpass)
            env["PAPERGURU_PAT_FILE"] = str(pat_file)
            env["GIT_CONFIG_COUNT"] = "2"
            env["GIT_CONFIG_KEY_0"] = "credential.helper"
            env["GIT_CONFIG_VALUE_0"] = ""
            env["GIT_CONFIG_KEY_1"] = "core.askPass"
            env["GIT_CONFIG_VALUE_1"] = str(askpass)

        if args.dry_run:
            status = _git(repo, "status", "--porcelain", "--", *paths, env=env).stdout
            print(status.strip() or "no changes in the requested paths")
            return 0
        _git(repo, "add", "-A", "--", *paths, env=env)
        staged = _git(repo, "diff", "--cached", "--name-only", env=env).stdout.splitlines()
        if not staged:
            print("no staged changes; nothing to publish")
            return 0
        unsafe = [path for path in staged if _matches_secret(path)]
        if unsafe:
            _git(repo, "reset", "--", *unsafe, env=env, check=False)
            raise SystemExit(f"refusing to commit secret-bearing paths: {unsafe}")
        diff = _git(repo, "diff", "--cached", env=env).stdout
        if token is not None and token in diff:
            _git(repo, "reset", env=env, check=False)
            raise SystemExit("token found in staged diff; aborting")
        print("staged:\n  " + "\n  ".join(staged))
        _git(repo, "commit", "-m", args.message, env=env)
        if args.no_push:
            print("committed; push skipped")
            return 0
        push = _git(repo, "push", args.remote, branch, env=env, check=False)
        if push.returncode:
            print(push.stderr.strip(), file=sys.stderr)
            raise SystemExit("push failed")
        print(f"pushed {len(staged)} file(s) to {args.remote}/{branch}")
        return 0
    finally:
        shutil.rmtree(tempdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
