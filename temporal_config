"""
Shared temporal-cutoff configuration (single source of truth).

Every data-producing script builds its artifacts from the same point in time:
  - 1-CoChange.py                   pre-cutoff co-change -> edge weights / propagation
  - 2-DependencyGraph.py            dependency graph from the cutoff snapshot
  - 3-CalculateCodeMetrics.py       static metrics from the cutoff snapshot
  - 4-ExtractHistoricalChanges.py   before/after windows -> prediction target
5-MLPipeline.py checks consistency via verify_cutoff_consistency().

Window semantics (disjoint by construction)
-------------------------------------------
The cutoff datetime is resolved once per project to an anchor commit
(the last commit at or before the cutoff):
    snapshot (graph, metrics) = working tree at the anchor
    "before" window           = commits reachable from the anchor
    "after" window (target)   = commits in anchor..HEAD
Merge commits are excluded everywhere (--no-merges).

Each script writes a marker file with the cutoff and anchor commit it used. 5-MLPipeline.py refuses
to run if a marker is missing or markers disagree. Whole-history mode (target leakage) is refused
unless ALLOW_WHOLE_HISTORY is True.
"""

import os
import re
import json
import subprocess

# Cutoff source, first match wins:
#   1. PROJECT_CUTOFFS[project]   manual per-project override
#   2. temporal_cutoffs.json      written by auto_select_cutoffs.py (recommended)
#   3. TEMPORAL_CUTOFF_DATE       manual default for all projects
#   4. none -> error (unless ALLOW_WHOLE_HISTORY)
# Format: "YYYY-MM-DD" (end of that day) or "YYYY-MM-DD HH:MM:SS".
# Re-running auto_select_cutoffs.py changes the cutoffs, so scripts 1-4 must then be re-run.
TEMPORAL_CUTOFF_DATE = None
PROJECT_CUTOFFS = {
    # "spring-framework": "2020-01-01",
}
AUTO_CUTOFFS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temporal_cutoffs.json")
ALLOW_WHOLE_HISTORY = False  # True = legacy leaky mode (never for the paper)

MARKER_KINDS = ("cochange", "graph", "metrics", "historical")
_MARKER_FILENAMES = {k: f"temporal_cutoff_marker_{k}.json" for k in MARKER_KINDS}


class CutoffNotConfigured(RuntimeError):
    """No cutoff set and ALLOW_WHOLE_HISTORY is False."""


class TemporalCutoffMismatch(RuntimeError):
    """Cutoff markers are missing or disagree."""


def normalize_cutoff(value):
    """Return 'YYYY-MM-DD HH:MM:SS' (date-only means end of day), or None."""
    if value is None:
        return None
    v = str(value).strip().replace("T", " ")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):
        return v + " 23:59:59"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", v):
        return v + ":00"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", v):
        return v
    raise ValueError(f"Invalid cutoff {value!r}; use 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'.")


def load_auto_cutoffs():
    """Contents of temporal_cutoffs.json ({project: {'cutoff': ..., ...}}), or {}."""
    if not os.path.exists(AUTO_CUTOFFS_FILE):
        return {}
    with open(AUTO_CUTOFFS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def get_cutoff(project_name):
    """Normalized cutoff for a project; raises if none is set and whole-history is not allowed."""
    raw = PROJECT_CUTOFFS.get(project_name)
    if raw is None:
        raw = (load_auto_cutoffs().get(project_name) or {}).get("cutoff")
    if raw is None:
        raw = TEMPORAL_CUTOFF_DATE
    if raw is None:
        if ALLOW_WHOLE_HISTORY:
            return None
        raise CutoffNotConfigured(
            f"[{project_name}] No temporal cutoff configured. Run auto_select_cutoffs.py (writes "
            f"temporal_cutoffs.json) or set PROJECT_CUTOFFS[{project_name!r}] / TEMPORAL_CUTOFF_DATE in "
            f"temporal_config.py. Whole-history mode leaks "
            f"future information into the features and is disabled "
            f"(ALLOW_WHOLE_HISTORY=False).")
    return normalize_cutoff(raw)


def resolve_cutoff_commit(code_dir, cutoff, ref="HEAD"):
    """Anchor commit: last commit at or before `cutoff` on `ref` (empty string if none)."""
    cutoff = normalize_cutoff(cutoff)
    out = subprocess.run(["git", "-C", code_dir, "rev-list", "-1", f"--before={cutoff}", ref],
                         capture_output=True, text=True, check=True).stdout.strip()
    return out


def write_cutoff_marker(project_dir, which, cutoff_value, commit=None):
    """Record the cutoff (and anchor commit) a script actually used."""
    if which not in _MARKER_FILENAMES:
        raise ValueError(f"Unknown marker kind: {which!r}")
    os.makedirs(project_dir, exist_ok=True)
    path = os.path.join(project_dir, _MARKER_FILENAMES[which])
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"script": which, "temporal_cutoff_date": cutoff_value,
                   "anchor_commit": commit}, f)
    return path


def read_cutoff_marker(project_dir, which):
    """Return the marker dict, or None if missing."""
    path = os.path.join(project_dir, _MARKER_FILENAMES[which])
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def verify_cutoff_consistency(project_dir, project_name="<project>"):
    """Fail unless all four markers exist and agree on the cutoff and anchor commit; return the cutoff.

    A missing cutoff (None) is rejected unless ALLOW_WHOLE_HISTORY.
    """
    markers = {k: read_cutoff_marker(project_dir, k) for k in MARKER_KINDS}
    missing = [k for k, m in markers.items() if m is None]
    if missing:
        raise TemporalCutoffMismatch(
            f"[{project_name}] Missing cutoff marker(s) for script(s): {missing} in '{project_dir}'. "
            f"Re-run scripts 1, 2, 3 and 4 (in that order) with the same cutoff, then 5.")
    cutoffs = {k: m.get("temporal_cutoff_date") for k, m in markers.items()}
    if len(set(cutoffs.values())) != 1:
        raise TemporalCutoffMismatch(
            f"[{project_name}] Cutoff mismatch between scripts: {cutoffs}. Re-run scripts 1-4 after any cutoff change "
            f"(cutoffs come from temporal_cutoffs.json / temporal_config.py).")
    cutoff = next(iter(cutoffs.values()))
    if cutoff is None and not ALLOW_WHOLE_HISTORY:
        raise TemporalCutoffMismatch(
            f"[{project_name}] Markers record NO cutoff (whole-history, leaky). Set a cutoff and re-run 1-4.")
    commits = {k: m.get("anchor_commit") for k, m in markers.items() if m.get("anchor_commit")}
    if len(set(commits.values())) > 1:
        raise TemporalCutoffMismatch(
            f"[{project_name}] Scripts resolved the cutoff to different anchor commits: {commits}. "
            f"Make sure every script ran on the same branch/HEAD of the repository.")
    return cutoff


def verified_anchor_commit(project_dir):
    m = read_cutoff_marker(project_dir, "historical")
    return m.get("anchor_commit") if m else None
