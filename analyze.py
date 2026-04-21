#!/usr/bin/env python3
"""
Flakiness Score: fraction of runs that failed
Patch Diversity: number of unique patches / successful runs
Time to Success: total execution_time per run (submitted runs only)
Trajectory Variance: std dev of step counts across runs
"""

import argparse
import json
from pathlib import Path


def load_runs(issue_id: str, results_dir: Path) -> list[dict]:
    issue_dir = results_dir / issue_id
    runs = []
    for run_dir in sorted(issue_dir.iterdir()):
        traj_path = run_dir / issue_id / f"{issue_id}.traj"
        if not traj_path.exists():
            continue
        with open(traj_path) as f:
            d = json.load(f)
        info = d["info"]
        traj = d["trajectory"]
        runs.append({
            "run": run_dir.name,
            "submitted": info.get("exit_status") == "submitted",
            "patch": info.get("submission", ""),
            "total_time": sum(s.get("execution_time", 0) for s in traj),
            "steps": len(traj),
            "cost": info.get("model_stats", {}).get("instance_cost", 0),
            "actions": [s.get("action", "") for s in traj],
        })
    return runs


def analyze_issue(issue_id: str, results_dir: Path) -> dict:
    runs = load_runs(issue_id, results_dir)
    n = len(runs)
    submitted = [r for r in runs if r["submitted"]]
    failed = [r for r in runs if not r["submitted"]]

    # Flakiness Score: fraction of runs that failed
    flakiness_score = len(failed) / n if n > 0 else 0

    # Patch Diversity: unique patches / total submitted
    patches = [r["patch"] for r in submitted if r["patch"]]
    unique_patches = len(set(patches))
    patch_diversity = unique_patches / len(patches) if patches else 0

    # Time to Success (submitted runs only)
    times = [r["total_time"] for r in submitted]
    avg_time = sum(times) / len(times) if times else 0
    min_time = min(times) if times else 0
    max_time = max(times) if times else 0
    time_std = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5 if times else 0

    # Trajectory Variance: std dev of step counts
    steps = [r["steps"] for r in runs]
    avg_steps = sum(steps) / len(steps) if steps else 0
    steps_std = (sum((s - avg_steps) ** 2 for s in steps) / len(steps)) ** 0.5 if steps else 0

    return {
        "issue_id": issue_id,
        "total_runs": n,
        "submitted": len(submitted),
        "failed": len(failed),
        "flakiness_score": flakiness_score,
        "unique_patches": unique_patches,
        "patch_diversity": patch_diversity,
        "avg_time": avg_time,
        "min_time": min_time,
        "max_time": max_time,
        "time_std": time_std,
        "avg_steps": avg_steps,
        "steps_std": steps_std,
        "total_cost": sum(r["cost"] for r in runs),
    }


def print_report(results: list[dict]):
    print("=" * 80)
    print("AGENT FLAKINESS ANALYSIS — CS527 Group-5")
    print("=" * 80)

    print("\n--- Flakiness Score (lower = more stable) ---")
    print("%-35s %8s %8s %10s" % ("Issue", "OK", "Failed", "Flakiness"))
    print("-" * 65)
    for r in results:
        print("%-35s %8d %8d %9.1f%%" % (
            r["issue_id"], r["submitted"], r["failed"], r["flakiness_score"] * 100))

    print("\n--- Patch Diversity (lower = more consistent) ---")
    print("%-35s %10s %10s %12s" % ("Issue", "Submitted", "Unique", "Diversity"))
    print("-" * 70)
    for r in results:
        print("%-35s %10d %10d %11.1f%%" % (
            r["issue_id"], r["submitted"], r["unique_patches"], r["patch_diversity"] * 100))

    print("\n--- Time to Success (submitted runs only, seconds) ---")
    print("%-35s %8s %8s %8s %8s" % ("Issue", "Avg", "Min", "Max", "StdDev"))
    print("-" * 72)
    for r in results:
        print("%-35s %8.1f %8.1f %8.1f %8.1f" % (
            r["issue_id"], r["avg_time"], r["min_time"], r["max_time"], r["time_std"]))

    print("\n--- Trajectory Variance (step count across runs) ---")
    print("%-35s %8s %8s" % ("Issue", "AvgSteps", "StdDev"))
    print("-" * 55)
    for r in results:
        print("%-35s %8.1f %8.1f" % (r["issue_id"], r["avg_steps"], r["steps_std"]))

    total_cost = sum(r["total_cost"] for r in results)
    print(f"\nTotal experiment cost: ${total_cost:.4f}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="minimax", help="Model slug (e.g. minimax, gpt4o)")
    args = parser.parse_args()

    results_dir = Path(__file__).parent / "results" / args.model
    if not results_dir.exists():
        print(f"No results found for model '{args.model}' at {results_dir}")
        return

    issue_ids = sorted(d.name for d in results_dir.iterdir() if d.is_dir())
    if not issue_ids:
        print("No results found. Run run_batch.py first.")
        return

    results = [analyze_issue(iid, results_dir) for iid in issue_ids]
    print_report(results)

    out_path = Path(__file__).parent / f"analysis_results_{args.model}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
