"""
auto_select_cutoffs.py -- automatically choose a temporal cutoff per project and store it in
temporal_cutoffs.json (read by temporal_config.py and scripts 0-5). Run once, before 0-CoChange.py:

    python auto_select_cutoffs.py                # all projects
    python auto_select_cutoffs.py --dry-run      # analysis only
    python auto_select_cutoffs.py --force        # overwrite stored cutoffs

Selection rule (deterministic)
------------------------------
Candidates are month-end dates, evaluated in one pass over non-merge, Java-touching commits.
Per candidate we estimate: months/commits before and after the cutoff, eligible classes
(touched before the cutoff and changed afterwards), the Q80 of their post-cutoff change counts,
and the number of classes at/above Q80 ("unstable").
A candidate is feasible only if all minimum requirements hold; among feasible candidates, the one
whose before-cutoff commit share is closest to TARGET_BEFORE_FRACTION is chosen. If none is
feasible, no cutoff is stored and the failing requirements are reported.

Approximations: class identity = file path (renames not resolved). The final choice is
re-verified with git (anchor commit and commit counts of both windows).
"""
import os
import sys
import json
import argparse
import subprocess
from collections import defaultdict, Counter
from datetime import datetime, timezone, timedelta

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from temporal_config import AUTO_CUTOFFS_FILE, load_auto_cutoffs, resolve_cutoff_commit

DEFAULT_BASE_CODE_DIR = r"D:\proje\python\Payannameh\Source"
DEFAULT_PROJECTS = ["spring-framework", "elasticsearch", "hadoop"]

# Selection settings (overridable from the command line)
TARGET_BEFORE_FRACTION = 0.70   # desired share of Java commits before the cutoff
MIN_BEFORE_MONTHS = 24
MIN_AFTER_MONTHS = 12
MIN_BEFORE_COMMITS = 1000
MIN_ELIGIBLE_CLASSES = 1000
MIN_Q80 = 3
MIN_UNSTABLE = 100              # classes at/above Q80


def load_java_commits(repo):
    """Return a time-sorted list of (datetime_utc, [java files]) for non-merge commits."""
    cmd = ["git", "-C", repo, "log", "--no-merges", "-M", "--name-only",
           "--pretty=format:%x00%cI", "--", "*.java"]
    out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                         errors="replace", check=True).stdout
    commits = []
    for chunk in out.split("\x00"):
        if not chunk.strip():
            continue
        lines = chunk.splitlines()
        try:
            d = datetime.fromisoformat(lines[0].strip()).astimezone(timezone.utc)
        except ValueError:
            continue
        files = [l.strip() for l in lines[1:] if l.strip().endswith(".java")]
        if files:
            commits.append((d, files))
    commits.sort(key=lambda x: x[0])
    return commits


def month_ends(first, last):
    y, m = first.year, first.month
    ends = []
    while (y, m) <= (last.year, last.month):
        ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
        end = datetime(ny, nm, 1, tzinfo=timezone.utc) - timedelta(seconds=1)
        ends.append(end)
        y, m = ny, nm
    return ends


def months_between(a, b):
    return (b - a).days / 30.4375


def evaluate_candidates(commits, cfg):
    n = len(commits)
    first, last = commits[0][0], commits[-1][0]
    first_seen = {}
    for d, files in commits:
        for f in files:
            first_seen.setdefault(f, d)
    cands = [c for c in month_ends(first, last) if c < last]
    rows = []
    after_counts = Counter()
    idx = n  # commits[idx:] form the "after" window
    for T in sorted(cands, reverse=True):  # move cutoff backwards, growing the "after" window
        while idx > 0 and commits[idx - 1][0] > T:
            idx -= 1
            after_counts.update(commits[idx][1])
        n_before = idx
        elig = [c for f, c in after_counts.items() if first_seen[f] <= T]
        row = {"cutoff": T, "before_commits": n_before, "after_commits": n - n_before,
               "before_fraction": n_before / n,
               "before_months": months_between(first, T), "after_months": months_between(T, last),
               "eligible_classes": len(elig), "q80": None, "unstable_classes": 0}
        if elig:
            q80 = float(np.percentile(elig, 80))
            row["q80"] = q80
            row["unstable_classes"] = int(sum(1 for c in elig if c >= q80))
        fails = []
        if row["before_months"] < cfg["min_before_months"]: fails.append("before_months")
        if row["after_months"] < cfg["min_after_months"]: fails.append("after_months")
        if row["before_commits"] < cfg["min_before_commits"]: fails.append("before_commits")
        if row["eligible_classes"] < cfg["min_eligible_classes"]: fails.append("eligible_classes")
        if row["q80"] is None or row["q80"] < cfg["min_q80"]: fails.append("q80")
        if row["unstable_classes"] < cfg["min_unstable"]: fails.append("unstable_classes")
        row["fails"] = fails
        rows.append(row)
    return sorted(rows, key=lambda r: r["cutoff"])


def choose(rows, cfg):
    feasible = [r for r in rows if not r["fails"]]
    if not feasible:
        return None
    # Closest to the target split; ties go to the later cutoff
    return min(feasible, key=lambda r: (abs(r["before_fraction"] - cfg["target_before_fraction"]),
                                        -r["cutoff"].timestamp()))


def verify_with_git(repo, cutoff_str):
    anchor = resolve_cutoff_commit(repo, cutoff_str)
    if not anchor:
        return None, 0, 0
    def count(rng):
        return int(subprocess.run(["git", "-C", repo, "rev-list", "--no-merges", "--count", rng, "--", "*.java"],
                                  capture_output=True, text=True, check=True).stdout.strip() or 0)
    return anchor, count(anchor), count(f"{anchor}..HEAD")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-dir", default=DEFAULT_BASE_CODE_DIR)
    ap.add_argument("--projects", nargs="+", default=DEFAULT_PROJECTS)
    ap.add_argument("--before-fraction", type=float, default=TARGET_BEFORE_FRACTION)
    ap.add_argument("--min-before-months", type=float, default=MIN_BEFORE_MONTHS)
    ap.add_argument("--min-after-months", type=float, default=MIN_AFTER_MONTHS)
    ap.add_argument("--min-before-commits", type=int, default=MIN_BEFORE_COMMITS)
    ap.add_argument("--min-eligible-classes", type=int, default=MIN_ELIGIBLE_CLASSES)
    ap.add_argument("--min-q80", type=float, default=MIN_Q80)
    ap.add_argument("--min-unstable", type=int, default=MIN_UNSTABLE)
    ap.add_argument("--dry-run", action="store_true", help="analyse only, do not write the JSON file")
    ap.add_argument("--force", action="store_true", help="overwrite cutoffs that are already stored")
    a = ap.parse_args()
    cfg = {"target_before_fraction": a.before_fraction, "min_before_months": a.min_before_months,
           "min_after_months": a.min_after_months, "min_before_commits": a.min_before_commits,
           "min_eligible_classes": a.min_eligible_classes, "min_q80": a.min_q80,
           "min_unstable": a.min_unstable}

    stored = load_auto_cutoffs()
    result = dict(stored)
    failed = []
    for p in a.projects:
        repo = os.path.join(a.base_dir, p)
        print("=" * 78 + f"\n{p}  ({repo})")
        if p in stored and not a.force:
            print(f"  Already has a stored cutoff ({stored[p]['cutoff']}); keeping it (use --force to "
                  f"re-select -- scripts 0-3 must then be re-run).")
            continue
        if not os.path.isdir(os.path.join(repo, ".git")):
            print("  ERROR: not a git repository."); failed.append(p); continue
        commits = load_java_commits(repo)
        if len(commits) < 10:
            print("  ERROR: too few Java commits."); failed.append(p); continue
        print(f"  {len(commits)} non-merge Java commits, {commits[0][0].date()} .. {commits[-1][0].date()}")
        rows = evaluate_candidates(commits, cfg)
        best = choose(rows, cfg)
        if best is None:
            cnt = Counter(f for r in rows for f in r["fails"])
            always = [k for k, v in cnt.items() if v == len(rows)]
            print(f"  NO FEASIBLE CUTOFF among {len(rows)} month-end candidates. "
                  f"Violations per requirement: {dict(cnt)}")
            if always:
                print(f"  Requirement(s) failed by EVERY candidate (the blocking ones): {always}")
            print("  Relax the relevant --min-* option explicitly (and justify it) or "
                  f"set PROJECT_CUTOFFS manually.")
            failed.append(p); continue
        cutoff_str = best["cutoff"].strftime("%Y-%m-%d")
        anchor, nb, na = verify_with_git(repo, cutoff_str)
        if not anchor or nb == 0 or na == 0:
            print(f"  ERROR: git verification failed for {cutoff_str} (anchor={anchor}, before={nb}, after={na})")
            failed.append(p); continue
        print(f"  SELECTED cutoff {cutoff_str}  (anchor {anchor[:10]})")
        print(f"    commits before/after (git-verified): {nb} / {na}  "
              f"(estimated before-share {best['before_fraction']:.2f}, target {cfg['target_before_fraction']:.2f})")
        print(f"    history before/after: {best['before_months']:.0f} / {best['after_months']:.0f} months")
        print(f"    eligible classes ~{best['eligible_classes']}, Q80 ~{best['q80']:.0f} changes, "
              f"unstable classes ~{best['unstable_classes']}")
        result[p] = {
            "cutoff": cutoff_str, "anchor_commit": anchor,
            "commits_before": nb, "commits_after": na,
            "before_months": round(best["before_months"], 1), "after_months": round(best["after_months"], 1),
            "est_eligible_classes": best["eligible_classes"], "est_q80": best["q80"],
            "est_unstable_classes": best["unstable_classes"],
            "selection_rule": cfg,
            "selected_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    if a.dry_run:
        print("\n--dry-run: nothing written.")
    elif result != stored:
        with open(AUTO_CUTOFFS_FILE, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nWrote {AUTO_CUTOFFS_FILE}")
    if failed:
        print(f"\nNo cutoff stored for: {failed}")
        sys.exit(1)
    print("\nNext: run 0-CoChange.py, 1-DependencyGraph.py, 2-CalculateCodeMetrics.py, "
          "3-ExtractHistoricalChanges.py, 4-MLPipeline.py, 5-GenerateProjectReport.py")


if __name__ == "__main__":
    main()
