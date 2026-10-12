"""
Standalone co-change extraction from Git.

For each non-merge commit, finds the Java classes changed together and counts how often each
class pair co-occurs. Depends only on the project's Git repo.

Output (per project): co_change_counts.pkl with
    cochange_counts:   {(class_a, class_b): count}  (pair keys are sorted tuples)
    class_solo_counts: {class: number of commits}
"""

import os
import pickle
import logging
from collections import defaultdict
from itertools import combinations

import subprocess

from temporal_config import get_cutoff, resolve_cutoff_commit, write_cutoff_marker

# ====================== Settings ======================
BASE_CODE_DIR = r"D:\proje\python\Payannameh\Source"
PROJECTS = ["spring-framework", "elasticsearch", "hadoop"]
SINCE_DATE = None   # e.g. "2018-01-01", or None for the entire history

# The temporal cutoff lives in temporal_config.py (shared with script 3).
# Co-change is limited to commits up to the cutoff to avoid target leakage.

# Commits touching more Java files than this are ignored (usually bulk/format changes)
MAX_FILES_PER_COMMIT = 30

TOP_N = 30   # top pairs shown per project in the aggregate report
# ======================================================

EXPORT_DIR = "ExportFiles"
os.makedirs(EXPORT_DIR, exist_ok=True)
AGGREGATE_TXT_OUTPUT = os.path.join(EXPORT_DIR, "co_change_summary_all_projects.txt")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def ensure_safe_directory(code_dir):
    """Work around Git's 'dubious ownership' error."""
    # Environment variables apply directly to child git processes
    os.environ["GIT_CONFIG_COUNT"] = "1"
    os.environ["GIT_CONFIG_KEY_0"] = "safe.directory"
    os.environ["GIT_CONFIG_VALUE_0"] = "*"

    # Fallback: also write to the global config
    try:
        result = subprocess.run(
            ["git", "config", "--global", "--add", "safe.directory", code_dir],
            check=False, capture_output=True, text=True
        )
        if result.returncode != 0:
            logger.warning(f"Writing to the global config failed (ignored): {result.stderr.strip()}")
    except FileNotFoundError:
        logger.warning("The git command was not found in PATH; only the environment variable will be used.")
    except Exception as e:
        logger.warning(f"Attempt to write global config failed (ignored): {e}")


def git_relpath_to_fqn(git_relpath):
    """Convert a repo-relative path from 'git log' to a fully-qualified class name.

    The simple name is not used: it would merge unrelated classes from different packages.
    """
    if not git_relpath.endswith(".java"):
        return None
    return git_relpath.replace("/", ".")[:-5]


def _build_rename_resolution_map(rename_pairs):
    """Return resolve(path), mapping an old path to its final path by following rename chains."""
    direct_map = dict(rename_pairs)  # old -> new (last seen wins)

    def resolve(path, _seen=None):
        _seen = _seen or set()
        steps = 0
        while path in direct_map and path not in _seen and steps < 100:
            _seen.add(path)
            path = direct_map[path]
            steps += 1
        return path

    return resolve


def extract_cochange(code_dir, since=SINCE_DATE, until=None,
                     max_files_per_commit=MAX_FILES_PER_COMMIT):
    """Extract co-change counts with a single 'git log' call.

    - Merge commits are excluded (same policy as the historical-changes script).
    - Renames are detected (--name-status -M) and resolved to the final path.
    - If `until` is set, only commits reachable from its anchor commit are used, i.e. the
      exact complement of script 3's "anchor..HEAD" target window.
    """
    logger.info("Opening the Git repository...")
    ensure_safe_directory(code_dir)

    if not os.path.isdir(os.path.join(code_dir, ".git")):
        logger.error(f"The .git folder was not found in '{code_dir}'.")
        return {}, {}

    marker = "\x01"  # commit separator that cannot clash with file names
    cmd = [
        "git", "-C", code_dir, "log", "--no-merges", "-M", "--name-status",
        f"--pretty=format:{marker}",
    ]
    if since:
        cmd.append(f"--since={since}")
    if until:
        anchor = resolve_cutoff_commit(code_dir, until)
        if not anchor:
            raise RuntimeError(f"No commit found at or before cutoff {until} in '{code_dir}'.")
        cmd.append(anchor)
        logger.info(f"Cutoff {until} -> anchor commit {anchor}: co-change limited to commits reachable "
                    f"from it (propagation_frequency cannot see the target window).")
    cmd += ["--", "*.java"]

    logger.info("Running 'git log' once for the entire history (may take a few minutes)...")
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            env=os.environ.copy(),
            bufsize=1,
        )
    except FileNotFoundError:
        logger.error("The git command was not found. Make sure git is installed and in PATH.")
        return {}, {}

    # --- Pass 1: buffer per-commit status lines and collect renames ---
    raw_commits = []
    rename_pairs = []
    current_lines = []
    unparseable_status_lines = 0

    def flush_raw(lines):
        raw_commits.append(list(lines))

    for line in iter(proc.stdout.readline, ""):
        line = line.rstrip("\n")
        if line == marker:
            flush_raw(current_lines)
            current_lines = []
        elif line.strip():
            current_lines.append(line)
    flush_raw(current_lines)

    proc.stdout.close()
    proc.wait()
    if proc.returncode != 0:
        logger.error(f"git log returned an error (code {proc.returncode})")
        return {}, {}

    for lines in raw_commits:
        for entry in lines:
            parts = entry.split("\t")
            status = parts[0]
            if status.startswith("R") or status.startswith("C"):
                if len(parts) == 3:
                    old_path, new_path = parts[1], parts[2]
                    if status.startswith("R"):
                        rename_pairs.append((old_path, new_path))
                    # Copies ('C') are not renames: the old path stays a separate class
                else:
                    unparseable_status_lines += 1

    resolve_fqn_path = _build_rename_resolution_map(rename_pairs)
    if rename_pairs:
        logger.info(f"Detected {len(rename_pairs)} rename events; historical identity will be "
                    f"resolved to each file's final/current path (Stage-3 rename-tracking fix).")

    # --- Pass 2: count co-changes using rename-resolved identities ---
    cochange_counts = defaultdict(int)
    class_solo_counts = defaultdict(int)
    skipped_wide_commits = 0
    processed_commits = 0
    unresolved_class_name_paths = 0

    def flush_commit(lines):
        nonlocal skipped_wide_commits, unresolved_class_name_paths
        classes = set()
        for entry in lines:
            parts = entry.split("\t")
            status = parts[0]
            if status.startswith("R") or status.startswith("C"):
                path = parts[2] if len(parts) == 3 else None
            else:
                path = parts[1] if len(parts) == 2 else None
            if not path or not path.endswith(".java"):
                continue
            resolved_path = resolve_fqn_path(path)
            cname = git_relpath_to_fqn(resolved_path)
            if cname:
                classes.add(cname)
            else:
                unresolved_class_name_paths += 1
        if not classes:
            return
        if len(classes) > max_files_per_commit:
            skipped_wide_commits += 1
            return
        for cname in classes:
            class_solo_counts[cname] += 1
        for a, b in combinations(sorted(classes), 2):
            cochange_counts[(a, b)] += 1

    for lines in raw_commits:
        flush_commit(lines)
        processed_commits += 1
        if processed_commits % 5000 == 0:
            logger.info(f"...{processed_commits} commits processed")

    logger.info(f"Total commits processed (excluding merges): {processed_commits}")
    logger.info(f"Large commits (more than {max_files_per_commit} files) ignored: {skipped_wide_commits}")
    logger.info(f"Number of class pairs with co-change: {len(cochange_counts)}")

    # Save parsing statistics for per-project reporting
    parsing_stats = {
        'total_commits_examined': processed_commits,
        'rename_events_detected': len(rename_pairs),
        'skipped_wide_commits': skipped_wide_commits,
        'max_files_per_commit_threshold': max_files_per_commit,
        'unresolved_status_lines': unparseable_status_lines,
        'paths_that_failed_fqn_conversion': unresolved_class_name_paths,
        'unique_classes_with_cochange_data': len(class_solo_counts),
        'unique_cochanging_class_pairs': len(cochange_counts),
    }
    try:
        project_dir_for_stats = os.path.join(EXPORT_DIR, os.path.basename(os.path.normpath(code_dir)))
        os.makedirs(project_dir_for_stats, exist_ok=True)
        with open(os.path.join(project_dir_for_stats, "parsing_stats_cochange.pkl"), "wb") as f:
            pickle.dump(parsing_stats, f)
    except Exception as e:
        logger.warning(f"Could not save co-change parsing stats: {e}")

    return dict(cochange_counts), dict(class_solo_counts)


def main():
    # Start with a fresh aggregate file
    if os.path.exists(AGGREGATE_TXT_OUTPUT):
        os.remove(AGGREGATE_TXT_OUTPUT)

    summary_lines = []

    for project_name in PROJECTS:
        code_dir = os.path.join(BASE_CODE_DIR, project_name)
        project_dir = os.path.join(EXPORT_DIR, project_name)
        os.makedirs(project_dir, exist_ok=True)
        pkl_output = os.path.join(project_dir, "co_change_counts.pkl")

        logger.info("=" * 70)
        logger.info(f"Starting co-change extraction for project: {project_name}")
        logger.info(f"Repo path: {code_dir}")

        cutoff = get_cutoff(project_name)  # raises if no cutoff is configured
        anchor_commit = resolve_cutoff_commit(code_dir, cutoff) if cutoff else None
        cochange_counts, class_solo_counts = extract_cochange(code_dir, SINCE_DATE, cutoff)

        with open(AGGREGATE_TXT_OUTPUT, "a", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write(f"Project: {project_name}\n")
            f.write(f"Repo path: {code_dir}\n")
            f.write("=" * 70 + "\n")

            if not cochange_counts:
                msg = "No co-change found! This project was skipped."
                logger.error(msg)
                f.write(msg + "\n\n")
                summary_lines.append(f"- {project_name}: failed (no co-change)")
                continue

            with open(pkl_output, "wb") as pf:
                pickle.dump({
                    "cochange_counts": cochange_counts,
                    "class_solo_counts": class_solo_counts,
                }, pf)
            logger.info(f"pkl results saved to '{pkl_output}'.")

            marker_path = write_cutoff_marker(project_dir, "cochange", cutoff, anchor_commit)
            logger.info(f"Cutoff marker written to '{marker_path}' "
                        f"(temporal_cutoff_date={cutoff!r}, anchor={anchor_commit}); "
                        f"4-MLPipeline.py will check this against script 3's marker.")

            total_pairs = len(cochange_counts)
            total_classes = len(class_solo_counts)
            top_pairs = sorted(cochange_counts.items(), key=lambda x: x[1], reverse=True)[:TOP_N]

            f.write(f"Total classes seen: {total_classes}\n")
            f.write(f"Total class pairs with co-change: {total_pairs}\n")
            f.write(f"pkl file: {pkl_output}\n")
            f.write(f"\nTop {TOP_N} most frequent pairs:\n")
            f.write(f"{'class_a':<40}{'class_b':<40}{'count'}\n")
            for (a, b), count in top_pairs:
                f.write(f"{a:<40}{b:<40}{count}\n")
            f.write("\n\n")

            logger.info(f"Summary of project '{project_name}' written to the aggregate file.")
            best_pair, best_count = top_pairs[0]
            summary_lines.append(
                f"- {project_name}: {total_classes} classes, {total_pairs} co-change pairs, "
                f"most frequent: {best_pair[0]} <-> {best_pair[1]} ({best_count} times)"
            )

    with open(AGGREGATE_TXT_OUTPUT, "a", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("Overall summary of all projects\n")
        f.write("=" * 70 + "\n")
        for line in summary_lines:
            f.write(line + "\n")

    logger.info("=" * 70)
    logger.info(f"Processing of all projects finished. Aggregate report: '{AGGREGATE_TXT_OUTPUT}'")


if __name__ == "__main__":
    main()
