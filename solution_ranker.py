#!/usr/bin/env python3
"""
Rank successful patches using existing experiment artifacts.

This script combines:
- patch verification results
- validator semantic comparisons
- per-run trajectory stats
- reference fix files from the seed Trajectories/

It produces rankings for each issue:
- best_overall
- most_reliable
- lowest_cost
- simplest_patch

Usage:
    python3 solution_ranker.py
    python3 solution_ranker.py --output solution_rankings.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_DIR = Path(__file__).parent
RESULTS_DIR = PROJECT_DIR / "results"
TRAJ_DIR = PROJECT_DIR / "Trajectories"
DEFAULT_OUTPUT = PROJECT_DIR / "solution_rankings.json"


@dataclass
class CandidatePatch:
    result_slug: str
    issue_id: str
    run_name: str
    model_name: str
    traj_path: str
    patch_path: str
    patch_text: str
    passes_verification: bool
    total_time: float
    steps: int
    cost: float
    cost_known: bool
    files_changed: int
    diff_lines: int
    added_lines: int
    removed_lines: int
    patch_hunks: int
    patch_files: list[str]
    reference_file_overlap: float
    exact_reference_match: bool
    semantic_label: str | None
    semantic_overlap: float | None
    exact_text_match: bool | None
    semantic_agreement_score: float
    normalized: dict
    composite_scores: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def parse_patch_stats(patch_text: str) -> dict:
    files: list[str] = []
    added = 0
    removed = 0
    hunks = 0
    for line in patch_text.splitlines():
        if line.startswith("+++ b/"):
            files.append(line[6:].strip())
        elif line.startswith("@@"):
            hunks += 1
        elif line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    unique_files = sorted(set(files))
    return {
        "patch_files": unique_files,
        "files_changed": len(unique_files),
        "added_lines": added,
        "removed_lines": removed,
        "diff_lines": added + removed,
        "patch_hunks": hunks,
    }


def load_reference_files() -> dict[str, list[str]]:
    reference_files: dict[str, list[str]] = {}
    for traj_path in sorted(TRAJ_DIR.glob("*.traj")):
        payload = json.loads(traj_path.read_text())
        replay = json.loads(payload["replay_config"])
        issue_id = replay["problem_statement"]["id"]
        patch_text = payload["info"].get("submission", "") or ""
        reference_files[issue_id] = parse_patch_stats(patch_text)["patch_files"]
    return reference_files


def load_verification_index() -> dict[tuple[str, str, str], dict]:
    index: dict[tuple[str, str, str], dict] = {}
    for path in sorted(PROJECT_DIR.glob("patch_verification_*.json")):
        rows = json.loads(path.read_text())
        for row in rows:
            patch_path = Path(row["patch_path"])
            parts = patch_path.parts
            if "results" not in parts:
                continue
            ridx = parts.index("results")
            if ridx + 3 >= len(parts):
                continue
            result_slug = parts[ridx + 1]
            issue_id = row["issue_id"]
            run_name = parts[ridx + 3]
            index[(result_slug, issue_id, run_name)] = row
    return index


def load_validator_index() -> dict[tuple[str, str, str], dict]:
    index: dict[tuple[str, str, str], dict] = {}
    for path in sorted(PROJECT_DIR.glob("validator_results_*.json")):
        payload = json.loads(path.read_text())
        for issue_id, issue_data in payload.get("issues", {}).items():
            for comparison in issue_data.get("comparisons", []):
                patch_path = Path(comparison["patch_path"])
                parts = patch_path.parts
                if "results" not in parts:
                    continue
                ridx = parts.index("results")
                if ridx + 3 >= len(parts):
                    continue
                result_slug = parts[ridx + 1]
                index[(result_slug, issue_id, comparison["run"])] = comparison
    return index


def extract_model_name(config_text: str) -> str:
    match = re.search(r'"model":\{"name":"([^"]+)"', config_text)
    return match.group(1) if match else "unknown"


def semantic_label_score(label: str | None, overlap: float | None) -> float:
    if label == "equivalent":
        return overlap if overlap is not None else 1.0
    if label == "different_strategy":
        return 0.8 if overlap is None else max(0.7, float(overlap))
    if label == "partially_equivalent":
        return 0.5 if overlap is None else min(0.65, float(overlap))
    if label == "not_a_fix":
        return 0.0
    return 0.5


def safe_ratio_intersection(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not b:
        return 0.0
    return len(a & b) / len(b)


def minmax_normalize_desc(values: list[float], value: float) -> float:
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        return 1.0
    return (high - value) / (high - low)


def minmax_normalize_asc(values: list[float], value: float) -> float:
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        return 1.0
    return (value - low) / (high - low)


def load_candidates() -> list[CandidatePatch]:
    reference_files = load_reference_files()
    verification_index = load_verification_index()
    validator_index = load_validator_index()
    candidates: list[CandidatePatch] = []

    for result_slug_dir in sorted(RESULTS_DIR.iterdir()):
        if not result_slug_dir.is_dir():
            continue
        result_slug = result_slug_dir.name
        for issue_dir in sorted(result_slug_dir.iterdir()):
            if not issue_dir.is_dir():
                continue
            issue_id = issue_dir.name
            expected_files = set(reference_files.get(issue_id, []))
            for run_dir in sorted(issue_dir.iterdir()):
                traj_path = run_dir / issue_id / f"{issue_id}.traj"
                if not traj_path.exists():
                    continue
                payload = json.loads(traj_path.read_text())
                patch_text = payload["info"].get("submission", "") or ""
                if not patch_text:
                    continue

                verification = verification_index.get((result_slug, issue_id, run_dir.name))
                if not verification or not verification.get("ok", False):
                    continue

                patch_stats = parse_patch_stats(patch_text)
                candidate_files = set(patch_stats["patch_files"])
                validator = validator_index.get((result_slug, issue_id, run_dir.name))
                config_path = run_dir / issue_id / "config.yaml"
                config_text = config_path.read_text(errors="replace") if config_path.exists() else ""

                overlap = safe_ratio_intersection(candidate_files, expected_files)
                exact_match = candidate_files == expected_files if expected_files else False
                semantic_label = validator.get("validator", {}).get("label") if validator else None
                semantic_overlap = validator.get("validator", {}).get("semantic_overlap") if validator else None

                candidates.append(
                    CandidatePatch(
                        result_slug=result_slug,
                        issue_id=issue_id,
                        run_name=run_dir.name,
                        model_name=extract_model_name(config_text),
                        traj_path=str(traj_path),
                        patch_path=verification["patch_path"],
                        patch_text=patch_text,
                        passes_verification=True,
                        total_time=sum(step.get("execution_time", 0.0) for step in payload["trajectory"]),
                        steps=len(payload["trajectory"]),
                        cost=float(payload["info"].get("model_stats", {}).get("instance_cost", 0.0)),
                        cost_known=float(payload["info"].get("model_stats", {}).get("instance_cost", 0.0)) > 0.0,
                        files_changed=patch_stats["files_changed"],
                        diff_lines=patch_stats["diff_lines"],
                        added_lines=patch_stats["added_lines"],
                        removed_lines=patch_stats["removed_lines"],
                        patch_hunks=patch_stats["patch_hunks"],
                        patch_files=patch_stats["patch_files"],
                        reference_file_overlap=overlap,
                        exact_reference_match=exact_match,
                        semantic_label=semantic_label,
                        semantic_overlap=semantic_overlap,
                        exact_text_match=validator.get("exact_text_match") if validator else None,
                        semantic_agreement_score=semantic_label_score(semantic_label, semantic_overlap),
                        normalized={},
                        composite_scores={},
                    )
                )

    return candidates


def score_group(candidates: list[CandidatePatch]) -> None:
    file_counts = [c.files_changed for c in candidates]
    diff_sizes = [c.diff_lines for c in candidates]
    step_counts = [c.steps for c in candidates]
    known_costs = [c.cost for c in candidates if c.cost_known]
    times = [c.total_time for c in candidates]
    semantic_scores = [c.semantic_agreement_score for c in candidates]
    file_overlaps = [c.reference_file_overlap for c in candidates]

    for candidate in candidates:
        if known_costs:
            if candidate.cost_known:
                cost_score = minmax_normalize_desc(known_costs, candidate.cost)
            else:
                cost_score = 0.5
        else:
            cost_score = 1.0
        normalized = {
            "files_changed": minmax_normalize_desc(file_counts, candidate.files_changed),
            "diff_size": minmax_normalize_desc(diff_sizes, candidate.diff_lines),
            "trajectory_length": minmax_normalize_desc(step_counts, candidate.steps),
            "cost": cost_score,
            "time": minmax_normalize_desc(times, candidate.total_time),
            "semantic_agreement": minmax_normalize_asc(semantic_scores, candidate.semantic_agreement_score),
            "reference_overlap": minmax_normalize_asc(file_overlaps, candidate.reference_file_overlap),
        }
        candidate.normalized = normalized

        candidate.composite_scores = {
            "best_overall": (
                0.22 * normalized["semantic_agreement"]
                + 0.20 * normalized["reference_overlap"]
                + 0.16 * normalized["files_changed"]
                + 0.14 * normalized["diff_size"]
                + 0.10 * normalized["trajectory_length"]
                + 0.09 * normalized["cost"]
                + 0.09 * normalized["time"]
            ),
            "most_reliable": (
                0.40 * normalized["semantic_agreement"]
                + 0.35 * normalized["reference_overlap"]
                + 0.10 * normalized["files_changed"]
                + 0.10 * normalized["diff_size"]
                + 0.05 * normalized["trajectory_length"]
            ),
            "lowest_cost": (
                0.60 * normalized["cost"]
                + 0.20 * normalized["time"]
                + 0.10 * normalized["trajectory_length"]
                + 0.10 * normalized["semantic_agreement"]
            ),
            "simplest_patch": (
                0.40 * normalized["files_changed"]
                + 0.35 * normalized["diff_size"]
                + 0.15 * normalized["reference_overlap"]
                + 0.10 * normalized["trajectory_length"]
            ),
        }


def sort_for_category(candidates: list[CandidatePatch], category: str) -> list[CandidatePatch]:
    return sorted(
        candidates,
        key=lambda c: (
            c.composite_scores[category],
            c.semantic_agreement_score,
            c.reference_file_overlap,
            1 if c.cost_known else 0,
            -c.files_changed,
            -c.diff_lines,
            -c.steps,
            -c.cost,
            -c.total_time,
        ),
        reverse=True,
    )


def candidate_summary(candidate: CandidatePatch, category: str) -> dict:
    return {
        "result_slug": candidate.result_slug,
        "issue_id": candidate.issue_id,
        "run_name": candidate.run_name,
        "model_name": candidate.model_name,
        "patch_path": candidate.patch_path,
        "traj_path": candidate.traj_path,
        "score": round(candidate.composite_scores[category], 6),
        "passes_verification": candidate.passes_verification,
        "files_changed": candidate.files_changed,
        "diff_lines": candidate.diff_lines,
        "reference_file_overlap": round(candidate.reference_file_overlap, 4),
        "exact_reference_match": candidate.exact_reference_match,
        "steps": candidate.steps,
        "cost": round(candidate.cost, 6),
        "cost_known": candidate.cost_known,
        "time": round(candidate.total_time, 4),
        "semantic_label": candidate.semantic_label,
        "semantic_overlap": candidate.semantic_overlap,
        "semantic_agreement_score": round(candidate.semantic_agreement_score, 4),
        "normalized": {k: round(v, 4) for k, v in candidate.normalized.items()},
        "patch_files": candidate.patch_files,
    }


def rank_issue_candidates(candidates: list[CandidatePatch]) -> dict:
    score_group(candidates)
    categories = ["best_overall", "most_reliable", "lowest_cost", "simplest_patch"]
    rankings = {}
    for category in categories:
        ordered = sort_for_category(candidates, category)
        rankings[category] = {
            "winner": candidate_summary(ordered[0], category),
            "top_3": [candidate_summary(candidate, category) for candidate in ordered[:3]],
        }
    return rankings


def build_payload(candidates: list[CandidatePatch]) -> dict:
    by_issue: dict[str, list[CandidatePatch]] = defaultdict(list)
    by_issue_model: dict[str, list[CandidatePatch]] = defaultdict(list)
    for candidate in candidates:
        by_issue[candidate.issue_id].append(candidate)
        by_issue_model[f"{candidate.result_slug}::{candidate.issue_id}"].append(candidate)

    payload = {
        "metadata": {
            "total_verified_candidates": len(candidates),
            "issues_with_verified_candidates": len(by_issue),
            "issue_model_groups": len(by_issue_model),
            "result_slugs": sorted({candidate.result_slug for candidate in candidates}),
            "models": sorted({candidate.model_name for candidate in candidates}),
            "result_slugs_with_missing_or_zero_costs": sorted({
                candidate.result_slug for candidate in candidates if not candidate.cost_known
            }),
        },
        "per_issue": {
            issue_id: rank_issue_candidates(group) for issue_id, group in sorted(by_issue.items())
        },
        "per_issue_model": {
            group_id: rank_issue_candidates(group) for group_id, group in sorted(by_issue_model.items())
        },
        "candidates": [asdict(candidate) for candidate in candidates],
    }
    return payload


def render_report(payload: dict) -> str:
    lines: list[str] = []
    lines.append("=" * 88)
    lines.append("SOLUTION RANKING REPORT")
    lines.append("=" * 88)
    lines.append(
        f"Verified candidate patches: {payload['metadata']['total_verified_candidates']}"
    )
    if payload["metadata"]["result_slugs_with_missing_or_zero_costs"]:
        lines.append(
            "Cost caveat: some artifacts have missing/zero cost, treated as neutral for "
            "`lowest_cost`: "
            + ", ".join(payload["metadata"]["result_slugs_with_missing_or_zero_costs"])
        )

    for issue_id, ranking in payload["per_issue"].items():
        lines.append(f"\n{issue_id}")
        for category in ["best_overall", "most_reliable", "lowest_cost", "simplest_patch"]:
            winner = ranking[category]["winner"]
            lines.append(
                f"  {category}: {winner['result_slug']} {winner['run_name']} "
                f"(score={winner['score']:.4f}, files={winner['files_changed']}, "
                f"diff={winner['diff_lines']}, cost=${winner['cost']:.4f}, "
                f"time={winner['time']:.1f}s, semantic={winner['semantic_label']})"
            )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    candidates = load_candidates()
    if not candidates:
        raise SystemExit("No verified submitted patches found in patch verification artifacts.")
    payload = build_payload(candidates)
    payload["report"] = render_report(payload)
    args.output.write_text(json.dumps(payload, indent=2))
    print(payload["report"])
    print(f"\nSaved JSON to {args.output}")


if __name__ == "__main__":
    main()
