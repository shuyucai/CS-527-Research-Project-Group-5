#!/usr/bin/env python3
"""
Semantic patch validator for flakiness analysis.

This script uses a Gemini Flash Lite chat model to compare patches produced
across repeated runs for the same issue. It is intended as an auxiliary
validator for the research project, not as a replacement for deterministic
verification.

Workflow:
1. Load patch verification results from patch_verification_<model>.json.
2. Group patches by issue id.
3. Pick one verified-good reference patch per issue.
4. Compare every other patch for that issue against the reference patch.
5. Write a JSON report summarizing semantic consistency across runs.

Default model settings mirror the Gemini Lite runs already present in this
repository, while still allowing overrides from the command line.

Examples:
    python3 validator_agent.py
    python3 validator_agent.py --results-model gemini-3-1-flash-lite-preview
    python3 validator_agent.py --issue psf__requests-1766
    python3 validator_agent.py --api-key-env GOOGLE_API_KEY
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).parent
TRAJ_DIR = PROJECT_DIR / "Trajectories"

DEFAULT_RESULTS_MODEL = "gemini-3-1-flash-lite-preview"
DEFAULT_MODEL_NAME = "openai/gemini-3.1-flash-lite-preview"
DEFAULT_API_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_TIMEOUT = 120
MAX_PATCH_CHARS = 20000


@dataclass(frozen=True)
class PatchRecord:
    issue_id: str
    run_name: str
    patch_path: Path
    ok: bool
    returncode: int
    stdout: str
    stderr: str


def load_problem_statements() -> dict[str, str]:
    statements: dict[str, str] = {}
    for traj_path in sorted(TRAJ_DIR.glob("*.traj")):
        with open(traj_path) as f:
            data = json.load(f)
        replay = json.loads(data["replay_config"])
        statements[replay["problem_statement"]["id"]] = replay["problem_statement"]["text"]
    return statements


def choose_api_key_env(api_base: str, explicit_env: str | None) -> str:
    if explicit_env:
        return explicit_env
    lower_base = api_base.lower()
    if "googleapis.com" in lower_base:
        for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY"):
            if os.environ.get(name):
                return name
        return "GOOGLE_API_KEY"
    for name in ("OPENROUTER_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"):
        if os.environ.get(name):
            return name
    return "OPENROUTER_API_KEY"


def load_patch_records(results_model: str, issue_filter: str | None) -> dict[str, list[PatchRecord]]:
    verification_path = PROJECT_DIR / f"patch_verification_{results_model}.json"
    if not verification_path.exists():
        raise FileNotFoundError(
            f"Missing verification artifact: {verification_path}. "
            "Run verify_patches.py first."
        )

    with open(verification_path) as f:
        data = json.load(f)

    grouped: dict[str, list[PatchRecord]] = defaultdict(list)
    for row in data:
        issue_id = row["issue_id"]
        if issue_filter and issue_id != issue_filter:
            continue
        patch_path = Path(row["patch_path"])
        try:
            run_name = patch_path.parents[1].name
        except IndexError:
            run_name = "unknown"
        grouped[issue_id].append(
            PatchRecord(
                issue_id=issue_id,
                run_name=run_name,
                patch_path=patch_path,
                ok=bool(row["ok"]),
                returncode=int(row["returncode"]),
                stdout=row.get("stdout", ""),
                stderr=row.get("stderr", ""),
            )
        )

    for issue_id in grouped:
        grouped[issue_id].sort(key=lambda r: r.run_name)
    return dict(grouped)


def patch_text(path: Path) -> str:
    return path.read_text(errors="replace")


def normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").strip()


def truncate_text(text: str, limit: int = MAX_PATCH_CHARS) -> str:
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-(limit // 2) :]
    return (
        f"{head}\n\n[... truncated {len(text) - limit} characters ...]\n\n{tail}"
    )


def touched_files_from_patch(text: str) -> list[str]:
    files: list[str] = []
    for line in text.splitlines():
        if line.startswith("+++ b/"):
            files.append(line[len("+++ b/") :].strip())
    return files


def json_from_response(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def resolve_model_name(api_base: str, model_name: str) -> str:
    normalized = model_name.strip()
    lower_base = api_base.lower()

    # SWE-agent/LiteLLM stores model names like "openai/gemini-..."" when using
    # Google's OpenAI-compatible endpoint. Raw requests to that endpoint need the
    # provider-native model id instead.
    if "googleapis.com" in lower_base and normalized.startswith("openai/"):
        return normalized.split("/", 1)[1]

    # If we ever point this script at OpenRouter, raw requests should use the
    # OpenRouter-facing model name rather than an extra routing prefix.
    if "openrouter.ai" in lower_base and normalized.startswith("openrouter/"):
        return normalized.split("/", 1)[1]

    return normalized


def call_validator_model(
    *,
    api_base: str,
    api_key: str,
    model_name: str,
    issue_id: str,
    problem_statement: str,
    reference_patch: str,
    candidate_patch: str,
    timeout: int,
    retries: int,
) -> dict[str, Any]:
    url = api_base.rstrip("/") + "/chat/completions"
    resolved_model_name = resolve_model_name(api_base, model_name)
    prompt = (
        "You are validating whether two software patches for the same bug are "
        "semantically equivalent.\n\n"
        "Return JSON only with this exact schema:\n"
        "{\n"
        '  "label": "equivalent" | "partially_equivalent" | "different_strategy" | "not_a_fix",\n'
        '  "semantic_overlap": 0.0,\n'
        '  "confidence": 0.0,\n'
        '  "reference_summary": "short summary",\n'
        '  "candidate_summary": "short summary",\n'
        '  "reasoning": "1-3 concise sentences"\n'
        "}\n\n"
        "Label definitions:\n"
        "- equivalent: candidate implements essentially the same fix intent.\n"
        "- partially_equivalent: candidate overlaps meaningfully but misses part of the fix.\n"
        "- different_strategy: candidate appears to solve the same bug with a materially different approach.\n"
        "- not_a_fix: candidate does not appear to fix the described issue.\n\n"
        f"Issue ID: {issue_id}\n\n"
        "Problem statement:\n"
        f"{problem_statement}\n\n"
        "Reference patch:\n"
        f"{truncate_text(reference_patch)}\n\n"
        "Candidate patch:\n"
        f"{truncate_text(candidate_patch)}\n"
    )

    payload = {
        "model": resolved_model_name,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a careful software patch validator. "
                    "Reply with strict JSON only."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8")
            response = json.loads(body)
            content = response["choices"][0]["message"]["content"]
            parsed = json_from_response(content)
            parsed["_raw_response"] = content
            return parsed
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(
                f"HTTP {exc.code} for {url} using model '{resolved_model_name}': {error_body}"
            )
            if attempt == retries:
                break
            time.sleep(min(2 * attempt, 10))
        except (urllib.error.URLError, json.JSONDecodeError, KeyError) as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(min(2 * attempt, 10))

    raise RuntimeError(f"Validator request failed after {retries} attempts: {last_error}")


def select_reference(records: list[PatchRecord], strategy: str) -> PatchRecord | None:
    passing = [r for r in records if r.ok]
    if not passing:
        return None
    if strategy == "shortest_pass":
        return min(passing, key=lambda r: len(patch_text(r.patch_path)))
    return passing[0]


def exact_match_result() -> dict[str, Any]:
    return {
        "label": "equivalent",
        "semantic_overlap": 1.0,
        "confidence": 1.0,
        "reference_summary": "Exact textual match.",
        "candidate_summary": "Exact textual match.",
        "reasoning": "The candidate patch text is identical to the reference patch.",
        "_raw_response": None,
    }


def compare_issue_patches(
    *,
    issue_id: str,
    records: list[PatchRecord],
    problem_statement: str,
    api_base: str,
    api_key: str,
    model_name: str,
    timeout: int,
    retries: int,
    reference_strategy: str,
) -> dict[str, Any]:
    reference = select_reference(records, reference_strategy)
    if reference is None:
        return {
            "issue_id": issue_id,
            "problem_statement": problem_statement,
            "reference_run": None,
            "reference_patch_path": None,
            "comparisons": [],
            "label_counts": {},
            "notes": "No verified-good reference patch available for this issue.",
        }

    ref_text = patch_text(reference.patch_path)
    ref_norm = normalize_text(ref_text)
    ref_files = touched_files_from_patch(ref_text)
    comparisons: list[dict[str, Any]] = []
    label_counts: Counter[str] = Counter()

    for record in records:
        cand_text = patch_text(record.patch_path)
        cand_norm = normalize_text(cand_text)
        cand_files = touched_files_from_patch(cand_text)
        exact_match = cand_norm == ref_norm

        if record.patch_path == reference.patch_path:
            validator = exact_match_result()
        elif exact_match:
            validator = exact_match_result()
        else:
            validator = call_validator_model(
                api_base=api_base,
                api_key=api_key,
                model_name=model_name,
                issue_id=issue_id,
                problem_statement=problem_statement,
                reference_patch=ref_text,
                candidate_patch=cand_text,
                timeout=timeout,
                retries=retries,
            )

        label = str(validator.get("label", "unknown"))
        label_counts[label] += 1
        comparisons.append(
            {
                "run": record.run_name,
                "patch_path": str(record.patch_path),
                "verification_ok": record.ok,
                "verification_returncode": record.returncode,
                "exact_text_match": exact_match,
                "reference_files": ref_files,
                "candidate_files": cand_files,
                "shared_files": sorted(set(ref_files) & set(cand_files)),
                "validator": {
                    "label": label,
                    "semantic_overlap": validator.get("semantic_overlap"),
                    "confidence": validator.get("confidence"),
                    "reference_summary": validator.get("reference_summary"),
                    "candidate_summary": validator.get("candidate_summary"),
                    "reasoning": validator.get("reasoning"),
                },
            }
        )

    return {
        "issue_id": issue_id,
        "problem_statement": problem_statement,
        "reference_run": reference.run_name,
        "reference_patch_path": str(reference.patch_path),
        "reference_files": ref_files,
        "comparisons": comparisons,
        "label_counts": dict(label_counts),
        "notes": None,
    }


def build_summary(issue_reports: dict[str, Any]) -> dict[str, Any]:
    overall: Counter[str] = Counter()
    per_issue: dict[str, dict[str, Any]] = {}

    for issue_id, report in issue_reports.items():
        counts = Counter(report.get("label_counts", {}))
        overall.update(counts)
        comparisons = report.get("comparisons", [])
        per_issue[issue_id] = {
            "reference_run": report.get("reference_run"),
            "total_compared_patches": len(comparisons),
            "label_counts": dict(counts),
            "verified_patches": sum(1 for c in comparisons if c["verification_ok"]),
            "unverified_patches": sum(1 for c in comparisons if not c["verification_ok"]),
        }

    return {
        "overall_label_counts": dict(overall),
        "issues": per_issue,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-model",
        default=DEFAULT_RESULTS_MODEL,
        help="Model slug used in patch_verification_<slug>.json.",
    )
    parser.add_argument(
        "--validator-model",
        default=DEFAULT_MODEL_NAME,
        help="Chat model name for the validator.",
    )
    parser.add_argument(
        "--api-base",
        default=DEFAULT_API_BASE,
        help="OpenAI-compatible API base for the validator.",
    )
    parser.add_argument(
        "--api-key-env",
        default=None,
        help="Environment variable name holding the API key. Defaults are inferred from api-base.",
    )
    parser.add_argument("--issue", default=None, help="Restrict validation to one issue id.")
    parser.add_argument(
        "--reference-strategy",
        choices=("first_pass", "shortest_pass"),
        default="first_pass",
        help="How to choose the issue reference patch.",
    )
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--output",
        default=None,
        help="Optional custom output path for the JSON report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    key_env = choose_api_key_env(args.api_base, args.api_key_env)
    api_key = os.environ.get(key_env)
    if not api_key:
        print(f"ERROR: Missing API key in environment variable {key_env}", file=sys.stderr)
        sys.exit(1)

    grouped_records = load_patch_records(args.results_model, args.issue)
    if not grouped_records:
        print("No matching patch records found.", file=sys.stderr)
        sys.exit(1)

    problem_statements = load_problem_statements()
    issue_reports: dict[str, Any] = {}

    for issue_id in sorted(grouped_records):
        print(f"Validating {issue_id} ({len(grouped_records[issue_id])} patches)...", flush=True)
        report = compare_issue_patches(
            issue_id=issue_id,
            records=grouped_records[issue_id],
            problem_statement=problem_statements.get(issue_id, ""),
            api_base=args.api_base,
            api_key=api_key,
            model_name=args.validator_model,
            timeout=args.timeout,
            retries=args.retries,
            reference_strategy=args.reference_strategy,
        )
        issue_reports[issue_id] = report

    output = {
        "results_model": args.results_model,
        "validator_model": args.validator_model,
        "api_base": args.api_base,
        "api_key_env": key_env,
        "reference_strategy": args.reference_strategy,
        "summary": build_summary(issue_reports),
        "issues": issue_reports,
    }

    if args.output:
        out_path = Path(args.output)
    else:
        suffix = args.issue if args.issue else args.results_model
        out_path = PROJECT_DIR / f"validator_results_{suffix}.json"

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved validator report to {out_path}")


if __name__ == "__main__":
    main()
