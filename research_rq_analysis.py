#!/usr/bin/env python3
"""
Research-question oriented analysis for agent flakiness experiments.

This script aggregates the existing run artifacts in this repository and
computes five higher-level analyses:

RQ1: How often do repeated runs diverge in outcome?
RQ2: At what point in the trajectory do successful and failed runs begin to diverge?
RQ3: Are semantically distinct correct patches common or rare?
RQ4: How do temperature/model choice affect flakiness?
RQ5: Which bug/task features correlate with instability?

It also emits a lightweight failure taxonomy to help explain unstable runs.

Usage:
    python3 research_rq_analysis.py
    python3 research_rq_analysis.py --output research_rq_analysis.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean


PROJECT_DIR = Path(__file__).parent
RESULTS_DIR = PROJECT_DIR / "results"
TRAJ_DIR = PROJECT_DIR / "Trajectories"
DEFAULT_OUTPUT = PROJECT_DIR / "research_rq_analysis.json"


@dataclass
class IssueMetadata:
    issue_id: str
    repo_family: str
    problem_statement: str
    problem_chars: int
    problem_words: int
    problem_lines: int
    traceback_lines: int
    docker_image: str
    base_commit: str
    baseline_steps: int
    baseline_time: float
    baseline_cost: float
    baseline_patch_files: int
    baseline_patch_hunks: int
    baseline_patch_added_lines: int
    baseline_patch_removed_lines: int
    baseline_patch_total_delta: int
    baseline_patch_text_chars: int
    expected_fix_files: list[str]


@dataclass
class RunRecord:
    result_slug: str
    issue_id: str
    run_name: str
    model_name: str
    temperature: float | None
    traj_path: str
    patch_path: str | None
    exit_status: str
    submitted: bool
    verified_ok: bool | None
    patch_text: str
    patch_files: list[str]
    total_time: float
    steps: int
    cost: float
    raw_actions: list[str]
    normalized_actions: list[str]
    validator_label: str | None
    semantic_overlap: float | None
    exact_text_match: bool | None
    failure_type: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def safe_mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def safe_div(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


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
    return {
        "files": sorted(set(files)),
        "num_files": len(set(files)),
        "num_hunks": hunks,
        "added_lines": added,
        "removed_lines": removed,
        "total_delta": added + removed,
    }


def load_issue_metadata() -> dict[str, IssueMetadata]:
    metadata: dict[str, IssueMetadata] = {}
    for traj_path in sorted(TRAJ_DIR.glob("*.traj")):
        data = json.loads(traj_path.read_text())
        replay = json.loads(data["replay_config"])
        info = data["info"]
        problem_statement = replay["problem_statement"]["text"]
        issue_id = replay["problem_statement"]["id"]
        repo_family = issue_id.split("__", 1)[0]
        baseline_patch = info.get("submission", "") or ""
        patch_stats = parse_patch_stats(baseline_patch)
        traj = data["trajectory"]
        metadata[issue_id] = IssueMetadata(
            issue_id=issue_id,
            repo_family=repo_family,
            problem_statement=problem_statement,
            problem_chars=len(problem_statement),
            problem_words=len(problem_statement.split()),
            problem_lines=len(problem_statement.splitlines()),
            traceback_lines=sum(1 for line in problem_statement.splitlines() if "Traceback" in line or re.match(r"\s*File ", line)),
            docker_image=replay["env"]["deployment"]["image"],
            base_commit=replay["env"]["repo"]["base_commit"],
            baseline_steps=len(traj),
            baseline_time=sum(step.get("execution_time", 0.0) for step in traj),
            baseline_cost=float(info.get("model_stats", {}).get("instance_cost", 0.0)),
            baseline_patch_files=patch_stats["num_files"],
            baseline_patch_hunks=patch_stats["num_hunks"],
            baseline_patch_added_lines=patch_stats["added_lines"],
            baseline_patch_removed_lines=patch_stats["removed_lines"],
            baseline_patch_total_delta=patch_stats["total_delta"],
            baseline_patch_text_chars=len(baseline_patch),
            expected_fix_files=patch_stats["files"],
        )
    return metadata


def extract_model_name(config_text: str) -> str:
    match = re.search(r'"model":\{"name":"([^"]+)"', config_text)
    return match.group(1) if match else "unknown"


def extract_temperature(config_text: str) -> float | None:
    match = re.search(r'"temperature":([0-9.]+)', config_text)
    if not match:
        return None
    return float(match.group(1))


def load_verification_index() -> dict[tuple[str, str, str], dict]:
    index: dict[tuple[str, str, str], dict] = {}
    for path in sorted(PROJECT_DIR.glob("patch_verification_*.json")):
        rows = json.loads(path.read_text())
        for row in rows:
            patch_path = Path(row["patch_path"])
            parts = patch_path.parts
            if "results" not in parts:
                continue
            results_idx = parts.index("results")
            if results_idx + 3 >= len(parts):
                continue
            result_slug = parts[results_idx + 1]
            issue_id = row["issue_id"]
            run_name = parts[results_idx + 3]
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
                results_idx = parts.index("results")
                if results_idx + 3 >= len(parts):
                    continue
                result_slug = parts[results_idx + 1]
                run_name = comparison["run"]
                index[(result_slug, issue_id, run_name)] = comparison
    return index


def normalize_testbed_path(path_token: str) -> str:
    path = path_token.strip().strip("'\"")
    path = path.replace("/testbed/", "")
    path = path.replace("/testbed", "")
    return path or "."


def normalize_action(action: str) -> str:
    action = action.strip()
    if not action:
        return "noop"

    if action == "submit":
        return "submit"
    if action.startswith("exit_forfeit"):
        return "exit_forfeit"

    if action.startswith("str_replace_editor "):
        parts = shlex.split(action)
        subcommand = parts[1] if len(parts) > 1 else "unknown"
        target = normalize_testbed_path(parts[2]) if len(parts) > 2 else "?"
        return f"editor:{subcommand}:{target}"

    lines = [line.strip() for line in action.splitlines() if line.strip()]
    first_line = lines[0]
    try:
        parts = shlex.split(first_line)
    except ValueError:
        parts = first_line.split()
    if not parts:
        return "shell"

    cmd = parts[0]
    if cmd in {"sed", "cat", "head", "tail"}:
        file_token = next((token for token in reversed(parts) if "/" in token or token.endswith(".py")), "")
        if file_token:
            return f"{cmd}:{normalize_testbed_path(file_token)}"
        return cmd
    if cmd in {"grep", "rg"}:
        return cmd
    if cmd in {"ls", "find", "pwd"}:
        return cmd
    if cmd == "git":
        return f"git:{parts[1] if len(parts) > 1 else 'cmd'}"
    if cmd.startswith("python"):
        return "python"
    return cmd


def repeated_loop(actions: list[str]) -> bool:
    if len(actions) < 4:
        return False
    normalized = [normalize_action(action) for action in actions if action.strip()]
    counts = Counter(normalized)
    most_common = counts.most_common(1)[0][1] if counts else 0
    return most_common >= max(3, math.ceil(len(normalized) * 0.4))


def classify_failure(
    *,
    submitted: bool,
    verified_ok: bool | None,
    patch_files: list[str],
    validator_label: str | None,
    exit_status: str,
    actions: list[str],
    expected_fix_files: set[str],
) -> str:
    if verified_ok is True:
        return "verified_fix"
    if not submitted:
        if repeated_loop(actions):
            return "no_submission_tool_loop"
        return "no_submission"

    patch_file_set = set(patch_files)
    if patch_file_set and all(Path(path).name.startswith("repro") for path in patch_file_set):
        return "reproduction_only"
    if patch_file_set and expected_fix_files and patch_file_set.isdisjoint(expected_fix_files):
        return "wrong_file_touched"
    if validator_label == "not_a_fix":
        return "submitted_not_a_fix"
    if validator_label == "partially_equivalent":
        return "submitted_partial_fix"
    if "exit_cost" in exit_status:
        return "submitted_exit_cost"
    return "submitted_unverified"


def load_runs(
    issue_metadata: dict[str, IssueMetadata],
    verification_index: dict[tuple[str, str, str], dict],
    validator_index: dict[tuple[str, str, str], dict],
) -> list[RunRecord]:
    records: list[RunRecord] = []
    for result_slug_dir in sorted(RESULTS_DIR.iterdir()):
        if not result_slug_dir.is_dir():
            continue
        result_slug = result_slug_dir.name
        for issue_dir in sorted(result_slug_dir.iterdir()):
            if not issue_dir.is_dir():
                continue
            issue_id = issue_dir.name
            if issue_id not in issue_metadata:
                continue
            expected_fix_files = set(issue_metadata[issue_id].expected_fix_files)
            for run_dir in sorted(issue_dir.iterdir()):
                traj_path = run_dir / issue_id / f"{issue_id}.traj"
                if not traj_path.exists():
                    continue
                data = json.loads(traj_path.read_text())
                info = data["info"]
                traj = data["trajectory"]
                patch_text = info.get("submission", "") or ""
                patch_stats = parse_patch_stats(patch_text)
                patch_path = run_dir / issue_id / f"{issue_id}.patch"
                config_text = (run_dir / issue_id / "config.yaml").read_text(errors="replace")
                verification = verification_index.get((result_slug, issue_id, run_dir.name))
                validator = validator_index.get((result_slug, issue_id, run_dir.name))
                submitted = bool(patch_text)
                exit_status = str(info.get("exit_status", "unknown"))
                raw_actions = [step.get("action", "") for step in traj if step.get("action", "").strip()]
                failure_type = classify_failure(
                    submitted=submitted,
                    verified_ok=verification["ok"] if verification else None,
                    patch_files=patch_stats["files"],
                    validator_label=validator.get("validator", {}).get("label") if validator else None,
                    exit_status=exit_status,
                    actions=raw_actions,
                    expected_fix_files=expected_fix_files,
                )
                records.append(
                    RunRecord(
                        result_slug=result_slug,
                        issue_id=issue_id,
                        run_name=run_dir.name,
                        model_name=extract_model_name(config_text),
                        temperature=extract_temperature(config_text),
                        traj_path=str(traj_path),
                        patch_path=str(patch_path) if patch_path.exists() else None,
                        exit_status=exit_status,
                        submitted=submitted,
                        verified_ok=verification["ok"] if verification else None,
                        patch_text=patch_text,
                        patch_files=patch_stats["files"],
                        total_time=sum(step.get("execution_time", 0.0) for step in traj),
                        steps=len(traj),
                        cost=float(info.get("model_stats", {}).get("instance_cost", 0.0)),
                        raw_actions=raw_actions,
                        normalized_actions=[normalize_action(action) for action in raw_actions],
                        validator_label=validator.get("validator", {}).get("label") if validator else None,
                        semantic_overlap=validator.get("validator", {}).get("semantic_overlap") if validator else None,
                        exact_text_match=validator.get("exact_text_match") if validator else None,
                        failure_type=failure_type,
                    )
                )
    return records


def outcome_label(run: RunRecord) -> str:
    if run.verified_ok is True:
        return "verified_fix"
    if run.submitted:
        return "submitted_but_failed_verification"
    return "no_submission"


def longest_common_prefix_length(a: list[str], b: list[str]) -> int:
    prefix = 0
    for left, right in zip(a, b):
        if left != right:
            break
        prefix += 1
    return prefix


def divergence_analysis(runs: list[RunRecord]) -> dict:
    successes = [run for run in runs if run.verified_ok is True]
    failures = [run for run in runs if run.verified_ok is not True]
    if not successes or not failures:
        return {
            "num_successes": len(successes),
            "num_failures": len(failures),
            "average_divergence_step": None,
            "median_like_divergence_step": None,
            "average_divergence_fraction": None,
            "failure_details": [],
        }

    details = []
    for failure in failures:
        best_match = None
        best_prefix = -1
        for success in successes:
            prefix = longest_common_prefix_length(failure.normalized_actions, success.normalized_actions)
            if prefix > best_prefix:
                best_prefix = prefix
                best_match = success
        failure_len = max(1, len(failure.normalized_actions))
        divergence_step = min(best_prefix + 1, failure_len)
        details.append(
            {
                "failure_run": failure.run_name,
                "matched_success_run": best_match.run_name if best_match else None,
                "shared_prefix_steps": best_prefix,
                "divergence_step": divergence_step,
                "divergence_fraction_of_failure": divergence_step / failure_len,
                "failure_type": failure.failure_type,
                "failure_outcome": outcome_label(failure),
            }
        )

    divergence_steps = [item["divergence_step"] for item in details]
    divergence_fractions = [item["divergence_fraction_of_failure"] for item in details]
    sorted_steps = sorted(divergence_steps)
    return {
        "num_successes": len(successes),
        "num_failures": len(failures),
        "average_divergence_step": safe_mean(divergence_steps),
        "median_like_divergence_step": sorted_steps[len(sorted_steps) // 2] if sorted_steps else None,
        "average_divergence_fraction": safe_mean(divergence_fractions),
        "failure_details": details,
    }


def summarize_rq1(grouped_runs: dict[tuple[str, str], list[RunRecord]]) -> dict:
    per_group = {}
    overall_counter: Counter[str] = Counter()
    overall_total = 0
    for key, runs in grouped_runs.items():
        outcome_counts = Counter(outcome_label(run) for run in runs)
        overall_counter.update(outcome_counts)
        overall_total += len(runs)
        dominant = outcome_counts.most_common(1)[0][1] if outcome_counts else 0
        per_group[f"{key[0]}::{key[1]}"] = {
            "result_slug": key[0],
            "issue_id": key[1],
            "total_runs": len(runs),
            "outcome_counts": dict(outcome_counts),
            "outcome_divergence_rate": 1.0 - (dominant / len(runs)) if runs else None,
            "verified_success_rate": safe_div(outcome_counts["verified_fix"], len(runs)),
            "submission_rate": safe_div(sum(1 for run in runs if run.submitted), len(runs)),
            "failure_taxonomy": dict(Counter(run.failure_type for run in runs)),
        }
    return {
        "overall_outcomes": dict(overall_counter),
        "overall_verified_success_rate": safe_div(overall_counter["verified_fix"], overall_total),
        "per_issue_model": per_group,
    }


def summarize_rq2(grouped_runs: dict[tuple[str, str], list[RunRecord]]) -> dict:
    per_group = {}
    aggregate_steps: list[float] = []
    aggregate_fractions: list[float] = []
    for key, runs in grouped_runs.items():
        analysis = divergence_analysis(runs)
        per_group[f"{key[0]}::{key[1]}"] = analysis
        if analysis["average_divergence_step"] is not None:
            aggregate_steps.append(analysis["average_divergence_step"])
        if analysis["average_divergence_fraction"] is not None:
            aggregate_fractions.append(analysis["average_divergence_fraction"])
    return {
        "overall_average_divergence_step": safe_mean(aggregate_steps),
        "overall_average_divergence_fraction": safe_mean(aggregate_fractions),
        "per_issue_model": per_group,
    }


def summarize_rq3(grouped_runs: dict[tuple[str, str], list[RunRecord]]) -> dict:
    per_group = {}
    overall = Counter()
    exact_text_mismatches = 0
    verified_with_validator = 0
    for key, runs in grouped_runs.items():
        verified_runs = [run for run in runs if run.verified_ok is True and run.validator_label]
        labels = Counter(run.validator_label for run in verified_runs)
        overall.update(labels)
        verified_with_validator += len(verified_runs)
        exact_text_mismatches += sum(
            1 for run in verified_runs
            if run.exact_text_match is False and run.validator_label == "equivalent"
        )
        per_group[f"{key[0]}::{key[1]}"] = {
            "verified_patches_with_semantic_labels": len(verified_runs),
            "semantic_label_counts": dict(labels),
            "different_strategy_rate": safe_div(labels["different_strategy"], len(verified_runs)),
            "partial_rate": safe_div(labels["partially_equivalent"], len(verified_runs)),
            "equivalent_rate": safe_div(labels["equivalent"], len(verified_runs)),
            "equivalent_but_textually_distinct": sum(
                1 for run in verified_runs if run.exact_text_match is False and run.validator_label == "equivalent"
            ),
        }
    distinct_count = overall["different_strategy"]
    return {
        "overall_semantic_label_counts_for_verified_patches": dict(overall),
        "overall_different_strategy_rate": safe_div(distinct_count, verified_with_validator),
        "overall_equivalent_but_textually_distinct_rate": safe_div(exact_text_mismatches, verified_with_validator),
        "per_issue_model": per_group,
        "interpretation": (
            "Semantically distinct correct patches are approximated as verified patches "
            "labeled 'different_strategy' by the validator."
        ),
    }


def summarize_rq4(records: list[RunRecord]) -> dict:
    by_model_temp: dict[tuple[str, float | None], list[RunRecord]] = defaultdict(list)
    by_temp: dict[float | None, list[RunRecord]] = defaultdict(list)
    for run in records:
        by_model_temp[(run.model_name, run.temperature)].append(run)
        by_temp[run.temperature].append(run)

    model_temperature_summary = {}
    for (model_name, temperature), runs in sorted(by_model_temp.items()):
        verified = sum(1 for run in runs if run.verified_ok is True)
        labels = Counter(run.validator_label for run in runs if run.verified_ok is True and run.validator_label)
        model_temperature_summary[f"{model_name} @ {temperature}"] = {
            "model_name": model_name,
            "temperature": temperature,
            "num_runs": len(runs),
            "verified_success_rate": safe_div(verified, len(runs)),
            "submission_rate": safe_div(sum(1 for run in runs if run.submitted), len(runs)),
            "avg_steps": safe_mean([run.steps for run in runs]),
            "avg_cost": safe_mean([run.cost for run in runs]),
            "avg_time": safe_mean([run.total_time for run in runs]),
            "different_strategy_rate_among_verified": safe_div(labels["different_strategy"], sum(labels.values())),
            "failure_taxonomy": dict(Counter(run.failure_type for run in runs)),
        }

    temperature_summary = {}
    for temperature, runs in sorted(by_temp.items(), key=lambda item: (item[0] is None, item[0])):
        verified = sum(1 for run in runs if run.verified_ok is True)
        temperature_summary[str(temperature)] = {
            "temperature": temperature,
            "num_runs": len(runs),
            "verified_success_rate": safe_div(verified, len(runs)),
            "avg_steps": safe_mean([run.steps for run in runs]),
            "avg_time": safe_mean([run.total_time for run in runs]),
        }

    unique_temps = sorted({run.temperature for run in records})
    note = None
    if len(unique_temps) <= 1:
        note = (
            "Current repository artifacts contain only one temperature setting, so "
            "temperature effects cannot be estimated empirically from this dataset yet."
        )
    return {
        "by_model_and_temperature": model_temperature_summary,
        "by_temperature": temperature_summary,
        "temperature_effect_note": note,
    }


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def rankdata(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float | None:
    return pearson(rankdata(xs), rankdata(ys))


def summarize_rq5(
    grouped_runs: dict[tuple[str, str], list[RunRecord]],
    issue_metadata: dict[str, IssueMetadata],
) -> dict:
    rows = []
    for (result_slug, issue_id), runs in grouped_runs.items():
        divergence = divergence_analysis(runs)
        verified_failures = sum(1 for run in runs if run.verified_ok is not True)
        metadata = issue_metadata[issue_id]
        rows.append(
            {
                "result_slug": result_slug,
                "issue_id": issue_id,
                "verified_failure_rate": verified_failures / len(runs),
                "outcome_divergence_rate": 1.0 - Counter(outcome_label(run) for run in runs).most_common(1)[0][1] / len(runs),
                "avg_divergence_step": divergence["average_divergence_step"] or 0.0,
                "problem_words": metadata.problem_words,
                "problem_lines": metadata.problem_lines,
                "traceback_lines": metadata.traceback_lines,
                "baseline_steps": metadata.baseline_steps,
                "baseline_time": metadata.baseline_time,
                "baseline_patch_files": metadata.baseline_patch_files,
                "baseline_patch_hunks": metadata.baseline_patch_hunks,
                "baseline_patch_total_delta": metadata.baseline_patch_total_delta,
                "baseline_patch_text_chars": metadata.baseline_patch_text_chars,
            }
        )

    feature_names = [
        "problem_words",
        "problem_lines",
        "traceback_lines",
        "baseline_steps",
        "baseline_time",
        "baseline_patch_files",
        "baseline_patch_hunks",
        "baseline_patch_total_delta",
        "baseline_patch_text_chars",
    ]
    target_names = [
        "verified_failure_rate",
        "outcome_divergence_rate",
        "avg_divergence_step",
    ]

    correlations = {}
    for target_name in target_names:
        target_values = [row[target_name] for row in rows]
        correlations[target_name] = []
        for feature_name in feature_names:
            feature_values = [row[feature_name] for row in rows]
            correlations[target_name].append(
                {
                    "feature": feature_name,
                    "pearson": pearson(feature_values, target_values),
                    "spearman": spearman(feature_values, target_values),
                }
            )
        correlations[target_name].sort(
            key=lambda row: abs(row["spearman"]) if row["spearman"] is not None else -1.0,
            reverse=True,
        )

    return {
        "num_issue_model_observations": len(rows),
        "correlations": correlations,
        "observation_rows": rows,
        "note": (
            "These correlations are exploratory and based on a small number of issue-model "
            "observations, so they should be treated as directional rather than conclusive."
        ),
    }


def format_pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def render_report(payload: dict) -> str:
    lines: list[str] = []
    metadata = payload["metadata"]
    lines.append("=" * 88)
    lines.append("AGENT FLAKINESS RESEARCH ANALYSIS")
    lines.append("=" * 88)
    lines.append(
        f"Analyzed result slugs: {', '.join(metadata['analyzed_result_slugs'])}"
    )
    if metadata["skipped_result_slugs_without_verification"]:
        lines.append(
            "Skipped result slugs without patch-verification coverage: "
            + ", ".join(metadata["skipped_result_slugs_without_verification"])
        )

    rq1 = payload["rq1_outcome_divergence"]
    lines.append("\nRQ1. Outcome Divergence")
    lines.append(f"Overall verified success rate: {format_pct(rq1['overall_verified_success_rate'])}")
    lines.append(f"Overall outcomes: {rq1['overall_outcomes']}")
    for key, row in sorted(rq1["per_issue_model"].items()):
        lines.append(
            f"  {key}: divergence={format_pct(row['outcome_divergence_rate'])}, "
            f"success={format_pct(row['verified_success_rate'])}, outcomes={row['outcome_counts']}"
        )

    rq2 = payload["rq2_trajectory_divergence"]
    lines.append("\nRQ2. Trajectory Divergence")
    lines.append(
        f"Average first divergence step across comparable issue/model groups: "
        f"{rq2['overall_average_divergence_step']:.2f}"
        if rq2["overall_average_divergence_step"] is not None
        else "Average first divergence step across comparable issue/model groups: N/A"
    )
    lines.append(
        f"Average normalized divergence fraction: {format_pct(rq2['overall_average_divergence_fraction'])}"
        if rq2["overall_average_divergence_fraction"] is not None
        else "Average normalized divergence fraction: N/A"
    )
    for key, row in sorted(rq2["per_issue_model"].items()):
        if row["average_divergence_step"] is None:
            continue
        lines.append(
            f"  {key}: avg_step={row['average_divergence_step']:.2f}, "
            f"avg_fraction={format_pct(row['average_divergence_fraction'])}, "
            f"failures={row['num_failures']}"
        )

    rq3 = payload["rq3_semantic_patch_diversity"]
    lines.append("\nRQ3. Semantic Patch Diversity")
    lines.append(
        f"Different-strategy verified patches: {format_pct(rq3['overall_different_strategy_rate'])}"
    )
    lines.append(
        f"Equivalent but textually different verified patches: "
        f"{format_pct(rq3['overall_equivalent_but_textually_distinct_rate'])}"
    )
    lines.append(
        f"Verified semantic labels: {rq3['overall_semantic_label_counts_for_verified_patches']}"
    )

    rq4 = payload["rq4_model_temperature_effects"]
    lines.append("\nRQ4. Model / Temperature Effects")
    for key, row in rq4["by_model_and_temperature"].items():
        lines.append(
            f"  {key}: success={format_pct(row['verified_success_rate'])}, "
            f"avg_steps={row['avg_steps']:.1f}, avg_time={row['avg_time']:.1f}, "
            f"avg_cost=${row['avg_cost']:.4f}"
        )
    if rq4["temperature_effect_note"]:
        lines.append(f"  Note: {rq4['temperature_effect_note']}")

    rq5 = payload["rq5_task_feature_correlations"]
    lines.append("\nRQ5. Task Feature Correlations")
    lines.append(rq5["note"])
    for target_name, rows in rq5["correlations"].items():
        top_rows = rows[:3]
        lines.append(f"  Strongest correlations for {target_name}:")
        for row in top_rows:
            pearson_value = "N/A" if row["pearson"] is None else f"{row['pearson']:.3f}"
            spearman_value = "N/A" if row["spearman"] is None else f"{row['spearman']:.3f}"
            lines.append(
                f"    {row['feature']}: spearman={spearman_value}, pearson={pearson_value}"
            )

    return "\n".join(lines)


def build_payload(records: list[RunRecord], issue_metadata: dict[str, IssueMetadata]) -> dict:
    covered_result_slugs = {
        run.result_slug for run in records if run.verified_ok is not None
    }
    analysis_records = [
        run for run in records if run.result_slug in covered_result_slugs
    ]
    grouped_runs: dict[tuple[str, str], list[RunRecord]] = defaultdict(list)
    for run in analysis_records:
        grouped_runs[(run.result_slug, run.issue_id)].append(run)

    skipped_result_slugs = sorted({
        run.result_slug for run in records if run.verified_ok is None
    })

    payload = {
        "metadata": {
            "num_runs": len(records),
            "num_runs_in_analyzed_result_slugs": len(analysis_records),
            "num_result_slugs": len({run.result_slug for run in records}),
            "num_issues": len({run.issue_id for run in records}),
            "models": sorted({run.model_name for run in records}),
            "temperatures": sorted({run.temperature for run in records}),
            "analyzed_result_slugs": sorted(covered_result_slugs),
            "skipped_result_slugs_without_verification": sorted(
                set(skipped_result_slugs) - covered_result_slugs
            ),
        },
        "issue_metadata": {issue_id: asdict(meta) for issue_id, meta in issue_metadata.items()},
        "rq1_outcome_divergence": summarize_rq1(grouped_runs),
        "rq2_trajectory_divergence": summarize_rq2(grouped_runs),
        "rq3_semantic_patch_diversity": summarize_rq3(grouped_runs),
        "rq4_model_temperature_effects": summarize_rq4(analysis_records),
        "rq5_task_feature_correlations": summarize_rq5(grouped_runs, issue_metadata),
        "runs": [asdict(run) for run in records],
    }
    payload["report"] = render_report(payload)
    return payload


def main() -> None:
    args = parse_args()
    issue_metadata = load_issue_metadata()
    verification_index = load_verification_index()
    validator_index = load_validator_index()
    records = load_runs(issue_metadata, verification_index, validator_index)
    payload = build_payload(records, issue_metadata)
    args.output.write_text(json.dumps(payload, indent=2))
    print(payload["report"])
    print(f"\nSaved JSON to {args.output}")


if __name__ == "__main__":
    main()
