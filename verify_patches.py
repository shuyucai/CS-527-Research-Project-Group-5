#!/usr/bin/env python3
"""
Verify saved patch artifacts against issue-specific regression checks.

This script:
1. Walks every `*.patch` file under results/
2. Infers the issue id from the patch path
3. Looks up the correct SWE-bench docker image / base commit from Trajectories/
4. Applies the patch inside a fresh container
5. Runs an issue-specific verification script
6. Writes a JSON summary and prints a concise report

Usage:
    python3 verify_patches.py
    python3 verify_patches.py --model gemini-flash-latest
    python3 verify_patches.py --issue django__django-16429
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


PROJECT_DIR = Path(__file__).parent
TRAJ_DIR = PROJECT_DIR / "Trajectories"
RESULTS_DIR = PROJECT_DIR / "results"


@dataclass(frozen=True)
class IssueConfig:
    issue_id: str
    docker_image: str
    base_commit: str
    platform: str


def load_issue_configs() -> dict[str, IssueConfig]:
    configs: dict[str, IssueConfig] = {}
    for traj_path in sorted(TRAJ_DIR.glob("*.traj")):
        with open(traj_path) as f:
            data = json.load(f)
        replay = json.loads(data["replay_config"])
        issue_id = replay["problem_statement"]["id"]
        configs[issue_id] = IssueConfig(
            issue_id=issue_id,
            docker_image=replay["env"]["deployment"]["image"],
            base_commit=replay["env"]["repo"]["base_commit"],
            platform=replay["env"]["deployment"].get("platform", "linux/amd64"),
        )
    return configs


VERIFY_SCRIPTS: dict[str, str] = {
    "django__django-15104": r"""
import django
from django.conf import settings
from django.db import models
from django.db.migrations.autodetector import MigrationAutodetector
from django.db.migrations.state import ModelState, ProjectState

if not settings.configured:
    settings.configure(INSTALLED_APPS=[])
django.setup()

class CustomFKField(models.ForeignKey):
    def __init__(self, *args, **kwargs):
        kwargs['to'] = 'testapp.HardcodedModel'
        super().__init__(*args, **kwargs)

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        if "to" in kwargs:
            del kwargs["to"]
        return name, path, args, kwargs

before = ProjectState()
before.add_model(ModelState('testapp', 'HardcodedModel', []))
after = ProjectState()
after.add_model(ModelState('testapp', 'HardcodedModel', []))
after.add_model(
    ModelState(
        'testapp',
        'TestModel',
        [('custom', CustomFKField(on_delete=models.CASCADE))],
    )
)

changes = MigrationAutodetector(before, after)._detect_changes()
assert isinstance(changes, dict)
print("PASS: django__django-15104")
""",
    "django__django-16429": r"""
import datetime
import django
from django.conf import settings

if not settings.configured:
    settings.configure(USE_TZ=True, INSTALLED_APPS=[])
django.setup()

from django.utils.timesince import timesince
from django.utils.timezone import get_fixed_timezone

tz = get_fixed_timezone(0)
d = datetime.datetime(2024, 1, 1, 10, 0, 0, tzinfo=tz)
now = datetime.datetime(2024, 2, 2, 10, 0, 0, tzinfo=tz)
result = timesince(d, now=now)
assert isinstance(result, str)
assert result
print("PASS: django__django-16429", result)
""",
    "psf__requests-1766": r"""
from requests.auth import HTTPDigestAuth

auth = HTTPDigestAuth('user', 'passwd')
auth.chal = {
    'realm': 'testrealm@host.com',
    'qop': 'auth',
    'nonce': 'dcd98b7102dd2f0e8b11d0f600bfb0c093',
    'opaque': '5ccc069c403ebaf9f0171e9517f40e41',
    'algorithm': 'MD5',
}

header = auth.build_digest_header('GET', 'http://example.com/')
assert 'qop="auth"' in header, header
print("PASS: psf__requests-1766", header)
""",
    "sympy__sympy-17318": r"""
from sympy import I, sqrt
from sympy.simplify.sqrtdenest import sqrtdenest

expr = (3 - sqrt(2)*sqrt(4 + 3*I) + 3*I)/2
result = sqrtdenest(expr)
assert result is not None
print("PASS: sympy__sympy-17318", result)
""",
    "sympy__sympy-24661": r"""
from sympy import Lt
from sympy.parsing.sympy_parser import parse_expr

expr = parse_expr('1 < 2', evaluate=False)
assert isinstance(expr, Lt), (type(expr), expr)
assert str(expr) == '1 < 2', str(expr)
print("PASS: sympy__sympy-24661", expr)
""",
}


def find_patch_files(model: str | None, issue: str | None) -> list[Path]:
    roots = [RESULTS_DIR / model] if model else [p for p in RESULTS_DIR.iterdir() if p.is_dir()]
    patch_files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for patch in root.glob("**/*.patch"):
            if issue and issue not in patch.parts:
                continue
            patch_files.append(patch)
    return sorted(patch_files)


def extract_issue_id(patch_path: Path, issue_configs: dict[str, IssueConfig]) -> str | None:
    parts = set(patch_path.parts)
    for issue_id in issue_configs:
        if issue_id in parts:
            return issue_id
    return None


def verify_patch(patch_path: Path, config: IssueConfig, keep_container: bool) -> dict:
    verifier = VERIFY_SCRIPTS[config.issue_id]
    container_name = f"verify-{config.issue_id}-{patch_path.parent.parent.name}-{os.getpid()}"
    patch_mount = patch_path.resolve()

    bash_script = f"""
set -euo pipefail
cd /testbed
git reset --hard {config.base_commit} >/dev/null
git clean -fd >/dev/null
git apply /tmp/patch.diff
python3 - <<'PY'
{verifier}
PY
"""
    docker_cmd = [
        "docker", "run",
        "--name", container_name,
        "--platform", config.platform,
    ]
    if not keep_container:
        docker_cmd.append("--rm")
    docker_cmd.extend([
        "-v", f"{patch_mount}:/tmp/patch.diff:ro",
        config.docker_image,
        "/bin/bash",
        "-lc",
        bash_script,
    ])

    proc = subprocess.run(
        docker_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if keep_container:
        subprocess.run(["docker", "rm", "-f", container_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return {
        "patch_path": str(patch_path),
        "issue_id": config.issue_id,
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def summarize(results: list[dict]) -> None:
    print("=" * 90)
    print("PATCH VERIFICATION")
    print("=" * 90)
    print("%-35s %-10s %-6s %s" % ("Issue", "Run", "OK", "Patch"))
    print("-" * 90)
    for r in results:
        patch_path = Path(r["patch_path"])
        run_name = patch_path.parents[1].name
        print("%-35s %-10s %-6s %s" % (
            r["issue_id"],
            run_name,
            "YES" if r["ok"] else "NO",
            patch_path.name,
        ))
    passed = sum(1 for r in results if r["ok"])
    failed = len(results) - passed
    print("-" * 90)
    print(f"Total patches: {len(results)} | Passed: {passed} | Failed: {failed}")


def print_progress(index: int, total: int, patch_path: Path, issue_id: str) -> None:
    run_name = patch_path.parents[1].name
    print(
        f"[{index}/{total}] Verifying {issue_id} {run_name} {patch_path.name}",
        flush=True,
    )


def print_result(result: dict) -> None:
    patch_path = Path(result["patch_path"])
    run_name = patch_path.parents[1].name
    print(
        f"    -> {'PASS' if result['ok'] else 'FAIL'} {result['issue_id']} {run_name}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Restrict to one results/<model> directory")
    parser.add_argument("--issue", type=str, default=None, help="Restrict to one issue id")
    parser.add_argument(
        "--keep-container",
        action="store_true",
        help="Keep verification container name allocated during execution for debugging (still force-removed afterward).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Verify only the first N matching patches.")
    args = parser.parse_args()

    issue_configs = load_issue_configs()
    patch_files = find_patch_files(args.model, args.issue)
    if args.limit is not None:
        patch_files = patch_files[:args.limit]
    if not patch_files:
        print("No patch files found.")
        return

    results: list[dict] = []
    total = len(patch_files)
    for index, patch_path in enumerate(patch_files, start=1):
        issue_id = extract_issue_id(patch_path, issue_configs)
        if issue_id is None:
            result = {
                "patch_path": str(patch_path),
                "issue_id": "UNKNOWN",
                "ok": False,
                "returncode": -1,
                "stdout": "",
                "stderr": "Could not infer issue id from patch path.",
            }
            print_progress(index, total, patch_path, "UNKNOWN")
            print_result(result)
            results.append(result)
            continue
        if issue_id not in VERIFY_SCRIPTS:
            result = {
                "patch_path": str(patch_path),
                "issue_id": issue_id,
                "ok": False,
                "returncode": -1,
                "stdout": "",
                "stderr": f"No verifier registered for {issue_id}.",
            }
            print_progress(index, total, patch_path, issue_id)
            print_result(result)
            results.append(result)
            continue
        print_progress(index, total, patch_path, issue_id)
        result = verify_patch(patch_path, issue_configs[issue_id], args.keep_container)
        print_result(result)
        results.append(result)

    summarize(results)
    out_path = PROJECT_DIR / f"patch_verification_{args.model or 'all'}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
