"""Explicit confirmation for one or several finished run folders."""

import hashlib
import sqlite3
from pathlib import Path

import streamlit as st

from morphofeatures.run_deletion import (
    FINISHED,
    delete_runs,
    forget_deleted_run,
    prepare_runs_deletion,
)


def _folder_review(plan, selected_roots, selected_ids):
    st.write(
        f"All {len(plan['jobs'])} registered job(s) in this run will be removed from Runs and Results."
    )
    st.dataframe(plan["jobs"], hide_index=True)
    st.code(plan["cleanup_directory"])
    inventory = plan["inventory"]
    st.caption(
        f"{inventory['files']:,} files · {inventory['bytes'] / 1024**2:,.1f} MiB logical file size · {inventory['symlinks']:,} symlinks. Symlink targets are not followed. Hard-linked copies elsewhere remain; the disk space released may be smaller."
    )
    if inventory["top_level"]:
        st.write("Folders/files below will be removed with all their contents:")
        st.dataframe({"Delete": inventory["top_level"]}, hide_index=True)
    dependencies = [r for r in plan["dependent_runs"] if r["run_id"] not in selected_ids]
    if dependencies:
        st.warning(
            "Other finished runs refer to these files. Their own folders will remain, but links to these deleted inputs and the ability to recompute from them will break."
        )
        st.dataframe(dependencies, hide_index=True)
    preserved = [
        p
        for p in plan["preserved_paths"]
        if not any(Path(p).is_relative_to(root) for root in selected_roots)
    ]
    if preserved:
        with st.expander("Referenced files outside the selected run folders (kept)"):
            st.dataframe({"Keep": preserved}, hide_index=True)


def deletion_controls(config, registry, selected):
    # Several table rows can belong to one run; delete that folder only once.
    if not isinstance(selected, (list, tuple)):
        selected = [selected]
    runs = {record.run_id: record for record in selected}
    run_ids = sorted(runs)
    signature = (str(registry.path), tuple(run_ids))
    if st.session_state.get("runs:deletion_selection") != signature:
        for old_key in list(st.session_state):
            if str(old_key).startswith("run-deletion:"):
                del st.session_state[old_key]
        st.session_state["runs:deletion_selection"] = signature
    key = "run-deletion:" + hashlib.sha256(repr(signature).encode()).hexdigest()
    plural = len(runs) > 1
    title = (
        f"Delete {len(runs)} selected runs and results"
        if plural
        else "Delete finished run and results"
    )
    with st.expander(title):
        st.caption(
            "Delete entire runs, including every job/stage and all files in their experiment folders. Source data, weights and shared caches outside those folders are kept. Anything stored inside is removed. This cannot be undone."
        )
        if plural:
            st.dataframe(
                [
                    {"Run": r.run_id, "Name": r.display_name, "State": r.application_state}
                    for r in runs.values()
                ],
                hide_index=True,
            )
        blocked = [
            r
            for r in selected
            if r.application_state not in FINISHED | {"deleting", "deletion_failed"}
        ]
        if blocked:
            st.info(
                "Only finished runs can be deleted. Wait for these jobs to finish, or deselect their rows: "
                + ", ".join(sorted({r.run_id for r in blocked}))
            )
        label = "Review deletion of selected runs" if plural else "Review deletion"
        if st.button(label, key=key + ":review", disabled=bool(blocked)):
            st.session_state.pop(key, None)
            try:
                with st.spinner(
                    "Checking selected run folders, job states and registered dependencies…"
                ):
                    st.session_state[key] = prepare_runs_deletion(
                        registry, config.paths.output_root, run_ids
                    )
            except (ValueError, OSError, sqlite3.Error) as error:
                st.error(str(error))
        review = st.session_state.get(key)
        if not review:
            return
        token_key = key + ":" + review["token"]
        name = (
            f"these {len(runs)} runs"
            if plural
            else f"{selected[0].display_name} (run {run_ids[0]})"
        )
        st.warning(f"Are you sure you want to permanently delete {name} and all their results?")
        files = sum(p["inventory"]["files"] for p in review["runs"])
        size = sum(p["inventory"]["bytes"] for p in review["runs"])
        st.write(
            f"Total: {len(runs)} run folder(s), {files:,} files, {size / 1024**2:,.1f} MiB logical file size."
        )
        st.caption(
            "All selected runs are rechecked before any files are removed. Dependency checks use saved records in this workspace. References in notebooks, other workspaces or external applications may also break."
        )
        roots = [Path(p["run_directory"]) for p in review["runs"]]
        for plan in review["runs"]:
            if plural:
                with st.expander(
                    f"Files and dependencies · {runs[plan['run_id']].display_name} ({plan['run_id']})"
                ):
                    _folder_review(plan, roots, run_ids)
            else:
                _folder_review(plan, roots, run_ids)
        for warning in sorted({w for p in review["runs"] for w in p["warnings"]}):
            st.warning(warning)
        confirmed = st.checkbox(
            "I understand that this permanently removes the listed run files and may break links from dependent runs.",
            key=token_key + ":confirm",
        )
        if st.button("Keep runs" if plural else "Keep run", key=token_key + ":cancel"):
            st.session_state.pop(key, None)
            st.session_state.pop(token_key + ":confirm", None)
            st.rerun()
        if st.button(
            f"Permanently delete {len(runs)} runs" if plural else "Permanently delete run",
            key=token_key + ":delete",
            disabled=not confirmed,
            type="primary",
        ):
            try:
                with st.spinner("Rechecking the confirmation and deleting run files…"):
                    outcome = delete_runs(
                        registry,
                        config.paths.output_root,
                        run_ids,
                        confirmation_token=review["token"],
                    )
                for plan in outcome["deleted"]:
                    forget_deleted_run(
                        st.session_state,
                        plan["run_id"],
                        {j["id"] for j in plan["jobs"]},
                        plan["run_directory"],
                    )
                st.session_state.pop(key, None)
                if outcome["deleted"]:
                    message = (
                        f"Deleted {len(outcome['deleted'])} run(s) and their run folders."
                        if plural
                        else f"Deleted {selected[0].display_name} ({run_ids[0]}) and its run folder."
                    )
                    st.session_state["run_notice"] = (
                        message + " Shared caches and external inputs were kept."
                    )
                st.session_state["run_deletion_errors"] = [
                    f"Run {entry['run_id']}: {entry['error']}" for entry in outcome["failed"]
                ]
                st.rerun()
            except (ValueError, OSError, sqlite3.Error) as error:
                st.session_state.pop(key, None)
                st.error(str(error))
