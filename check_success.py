import json
import os

TRAJ_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Trajectories")

# resolved_ids sourced from CS527JBR-Team-18-1 milestone3 reports
# (traj files do not contain resolved status — cross-referenced externally)
REPORT_PATHS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../CS527JBR-Team-18-1/milestone3/deepseek.json"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../CS527JBR-Team-18-1/milestone3/gpt.json"),
]


def load_resolved_ids() -> set:
    resolved = set()
    for path in REPORT_PATHS:
        if not os.path.isfile(path):
            print(f"[WARN] Report not found: {path}")
            continue
        with open(path) as f:
            data = json.load(f)
        resolved.update(data.get("resolved_ids", []))
    return resolved


def check_trajectories():
    resolved_ids = load_resolved_ids()

    results = []
    for fname in sorted(os.listdir(TRAJ_DIR)):
        if not fname.endswith(".traj"):
            continue
        iid = fname.replace(".traj", "")
        path = os.path.join(TRAJ_DIR, fname)
        with open(path) as f:
            data = json.load(f)
        info = data["info"]
        exit_status = info.get("exit_status", "N/A")
        has_patch = bool(info.get("submission"))
        resolved = iid in resolved_ids
        cost = info.get("model_stats", {}).get("instance_cost", 0)
        success = exit_status == "submitted" and resolved
        results.append((iid, exit_status, has_patch, resolved, cost, success))

    header = "%-35s %-12s %-10s %-10s %8s  %s" % (
        "Instance", "Exit", "Patch", "Resolved", "Cost", "OK"
    )
    print(header)
    print("-" * 90)
    for iid, exit_s, patch, resolved, cost, ok in results:
        print("%-35s %-12s %-10s %-10s %8.4f  %s" % (
            iid, exit_s, str(patch), str(resolved), cost, "YES" if ok else "FAIL"
        ))

    total = len(results)
    success_count = sum(1 for r in results if r[5])
    total_cost = sum(r[4] for r in results)
    print("-" * 90)
    print("Total: %d | Successful: %d | Failed: %d | Total cost: $%.4f" % (
        total, success_count, total - success_count, total_cost
    ))

    if success_count == total:
        print("\nAll trajectories passed the check.")
    else:
        failed = [r[0] for r in results if not r[5]]
        print(f"\nFailed trajectories: {failed}")


if __name__ == "__main__":
    check_trajectories()
