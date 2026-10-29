"""Compare generation-eval runs across prompt versions as a Markdown table (latest run per version).

Usage: python3 scripts/compare_prompts.py [--cases] [VERSION ...]
"""

import argparse
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "tests" / "evals" / "results"
SUITES = ("answerable", "off_topic", "not_in_passages")


def latest_runs(versions: list[str]) -> dict[str, dict]:
    """version -> its most recent run that covers the generation suites (result files sort by timestamp)."""
    runs = {}
    for path in sorted(RESULTS.glob("*.json")):
        run = json.loads(path.read_text())
        version = run.get("prompt_version")
        if version and (not versions or version in versions) and any(c["suite"] in SUITES for c in run["cases"]):
            runs[version] = run | {"file": path.name}
    return dict(sorted(runs.items(), key=lambda kv: versions.index(kv[0]) if versions else kv[0]))


def rate(cases: list[dict]) -> str:
    return f"{sum(c['passed'] for c in cases)}/{len(cases)}" if cases else "-"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("versions", nargs="*", help="prompt versions to compare (default: all)")
    parser.add_argument("--cases", action="store_true", help="also show pass/fail per case")
    args = parser.parse_args()

    runs = latest_runs(args.versions)
    print("| version | model | " + " | ".join(SUITES) + " | total |")
    print("|---" * (len(SUITES) + 3) + "|")
    for version, run in runs.items():
        gen = [c for c in run["cases"] if c["suite"] in SUITES]
        cells = [rate([c for c in gen if c["suite"] == s]) for s in SUITES]
        print(f"| {version} | {run['model']} | " + " | ".join(cells) + f" | {rate(gen)} |")

    if args.cases:
        ids = list(dict.fromkeys(f"{c['suite']}/{c['id']}" for run in runs.values() for c in run["cases"] if c["suite"] in SUITES))
        print("\n| case | " + " | ".join(runs) + " |")
        print("|---" * (len(runs) + 1) + "|")
        for case_id in ids:
            marks = []
            for run in runs.values():
                row = next((c for c in run["cases"] if f"{c['suite']}/{c['id']}" == case_id), None)
                marks.append("-" if row is None else "pass" if row["passed"] else "**FAIL**")
            print(f"| {case_id} | " + " | ".join(marks) + " |")


if __name__ == "__main__":
    main()
