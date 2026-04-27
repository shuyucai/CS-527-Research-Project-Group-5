#!/usr/bin/env python3
"""
usage:
export OPENROUTER_API_KEY="sk-or-..."
python3 run_batch.py                        # run all issues, 20 times each
python3 run_batch.py --runs 1               # test with 1 run per issue
python3 run_batch.py --issue psf__requests-1766 --runs 3
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

SWEAGENT = str(Path.home() / "SWE-agent/.venv311/bin/sweagent")
SWEAGENT_DIR = str(Path.home() / "SWE-agent")
PROJECT_DIR = Path(__file__).parent
TRAJ_DIR = PROJECT_DIR / "Trajectories"

MODEL_NAME = "openrouter/openai/gpt-5-mini"
MODEL_SLUG = "gpt5mini"
RESULTS_DIR = PROJECT_DIR / "results" / MODEL_SLUG
API_BASE = "https://openrouter.ai/api/v1"
TEMPERATURE = 1.0
N_RUNS = 20


def load_issue_config(traj_path: Path) -> dict:
    with open(traj_path) as f:
        data = json.load(f)
    rc = json.loads(data["replay_config"])
    return {
        "issue_id": rc["problem_statement"]["id"],
        "problem_text": rc["problem_statement"]["text"],
        "docker_image": rc["env"]["deployment"]["image"],
        "repo_name": rc["env"]["repo"]["repo_name"],
        "base_commit": rc["env"]["repo"]["base_commit"],
        "platform": rc["env"]["deployment"].get("platform", "linux/amd64"),
    }


def run_once(cfg: dict, run_num: int, api_key: str) -> int:
    issue_id = cfg["issue_id"]
    run_dir = RESULTS_DIR / issue_id / f"run_{run_num:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Write problem statement to file (avoids shell quoting issues)
    ps_path = run_dir / "problem_statement.txt"
    ps_path.write_text(cfg["problem_text"])

    cmd = [
        SWEAGENT, "run",
        "--config", f"{SWEAGENT_DIR}/config/default.yaml",
        f"--agent.model.name={MODEL_NAME}",
        f"--agent.model.api_base={API_BASE}",
        f"--agent.model.api_key={api_key}",
        f"--agent.model.temperature={TEMPERATURE}",
        f"--agent.model.per_instance_cost_limit=1.0",
        "--problem_statement.type=text_file",
        f"--problem_statement.path={ps_path}",
        f"--problem_statement.id={issue_id}",
        "--env.deployment.type=docker",
        f"--env.deployment.image={cfg['docker_image']}",
        "--env.deployment.pull=missing",
        f"--env.deployment.platform={cfg['platform']}",
        "--env.repo.type=preexisting",
        f"--env.repo.repo_name={cfg['repo_name']}",
        f"--env.repo.base_commit={cfg['base_commit']}",
        f"--output_dir={run_dir}",
    ]

    log_path = run_dir / "run.log"
    with open(log_path, "w") as log:
        result = subprocess.run(cmd, stdout=log, stderr=log, cwd=SWEAGENT_DIR)

    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=N_RUNS)
    parser.add_argument("--issue", type=str, default=None, help="Run only this issue ID")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: Set OPENROUTER_API_KEY environment variable first:")
        print('  export OPENROUTER_API_KEY="sk-or-..."')
        sys.exit(1)

    traj_files = sorted(TRAJ_DIR.glob("*.traj"))
    if args.issue:
        traj_files = [f for f in traj_files if args.issue in f.name]
        if not traj_files:
            print(f"ERROR: No trajectory found for issue '{args.issue}'")
            sys.exit(1)

    print(f"Model: {MODEL_NAME} (temperature={TEMPERATURE})")
    print(f"Issues: {len(traj_files)}  |  Runs each: {args.runs}  |  Total: {len(traj_files) * args.runs}")
    print()

    for traj_file in traj_files:
        cfg = load_issue_config(traj_file)
        issue_id = cfg["issue_id"]
        print(f"=== {issue_id} ===")

        success = 0
        for run_num in range(1, args.runs + 1):
            run_dir = RESULTS_DIR / issue_id / f"run_{run_num:03d}"
            traj_out = run_dir / issue_id / f"{issue_id}.traj"
            if traj_out.exists():
                print(f"  run_{run_num:03d}: SKIP (already exists)")
                success += 1
                continue

            rc = run_once(cfg, run_num, api_key)
            status = "OK" if rc == 0 else f"FAIL(rc={rc})"
            if rc == 0:
                success += 1
            print(f"  run_{run_num:03d}: {status}  [log: results/{issue_id}/run_{run_num:03d}/run.log]")

        print(f"  => {success}/{args.runs} runs completed\n")


if __name__ == "__main__":
    main()
