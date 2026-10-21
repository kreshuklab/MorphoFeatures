"""Visible stop controls for the single job currently open in the Runs inspector."""

import sqlite3
from pathlib import Path

import psutil
import streamlit as st

from morphofeatures.job_control import ACTIVE, REMOVED, can_stop_slurm, local_target, stop_local_job


def stop_controls(registry, record):
    if record.application_state in REMOVED:
        return
    if record.provenance.get("local_stop", {}).get("confirmed_at"):
        return
    local_file = Path(record.working_directory) / "process.json"
    if record.slurm_job_id:
        if not can_stop_slurm(record):
            return
        backend = "SLURM"
    elif record.application_state in ACTIVE or local_file.exists():
        backend = "local"
    else:
        return
    local_error = None
    if backend == "local":
        try:
            data, members = local_target(record)
        except (ValueError, OSError, psutil.Error) as error:
            if record.application_state not in ACTIVE:
                return
            local_error = str(error)
        else:
            if not members and record.application_state not in ACTIVE:
                return
    with st.container():
        st.markdown("**Stop job**")
        st.caption(
            f"Applies to {record.display_name} · {record.run_id} · job {record.id[:10]}. Already written outputs are kept; a file being written may remain incomplete. Other selected rows are not stopped."
        )
        key = "stop-job:" + str(registry.path) + ":" + record.id
        if backend == "SLURM":
            st.write(
                f"Cancel SLURM allocation {record.slurm_job_id}, including its worker processes."
            )
            if record.raw_scheduler_state == "CANCEL_REQUESTED":
                st.info(
                    "Cancellation was requested. Use Refresh SLURM to confirm it has stopped; you can resend the request if needed."
                )
            confirmed = st.checkbox("Confirm cancellation of this SLURM job", key=key + ":confirm")
            if st.button("Cancel SLURM job", key=key + ":cancel", disabled=not confirmed):
                from morphofeatures.experiments import cancel_job
                from morphofeatures.slurm import SlurmScheduler

                try:
                    cancel_job(record, registry, SlurmScheduler())
                    st.session_state["run_notice"] = (
                        f"Cancellation requested for SLURM job {record.slurm_job_id}. Refresh SLURM to confirm it has stopped."
                    )
                    st.rerun()
                except (ValueError, RuntimeError, OSError, sqlite3.Error) as error:
                    st.error(str(error))
        else:
            if local_error:
                st.warning(local_error)
                return
            if not members:
                st.info("The local worker process group has stopped.")
                return
            st.write(
                f"Local worker PID {data['pid']} and {len(members) - 1} other process(es) in its worker group."
            )
            if record.provenance.get("local_stop"):
                st.info(
                    "Stop requested; processes are still exiting. Refresh local job status, or force kill if they remain stuck."
                )
            force = st.checkbox(
                "Force kill local worker (SIGKILL)",
                key=key + ":force",
                help="Normally Stop sends SIGTERM to the worker and its data-loading processes. Use force kill if they do not exit. SIGKILL stops them immediately without cleanup; an output being written may remain incomplete.",
            )
            confirmed = st.checkbox(
                "Confirm stopping this local job", key=key + ":confirm:" + str(force)
            )
            if st.button(
                "Force kill local job" if force else "Stop local job",
                key=key + ":stop:" + str(force),
                disabled=not confirmed,
            ):
                try:
                    stopped = stop_local_job(record, registry, force=force)
                    st.session_state["run_notice"] = (
                        f"Local job {record.display_name} stopped. Existing outputs were kept."
                        if stopped.application_state == "cancelled"
                        else f"Stop requested for local job {record.display_name}. Refresh local job status to confirm all worker processes have exited."
                    )
                    st.rerun()
                except (ValueError, OSError, psutil.Error, sqlite3.Error) as error:
                    st.error(str(error))
            st.button("Refresh local job status", key=key + ":refresh")
