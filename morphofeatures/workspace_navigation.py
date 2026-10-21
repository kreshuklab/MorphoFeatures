"""Session-only screen history; Back never rolls back scientific settings or jobs."""

from copy import deepcopy

RESULT_PREFIXES = (
    "result-view:",
    "analysis-section:",
    "display-representations:",
    "distillation-section:",
    "analysis-combination:",
    "transfer-artifact:",
    "experiment-analysis:",
    "joint-dataset:",
    "experiment-transfer:",
    "experiment-vector:",
)


def selector_key(key):
    """Register only selectors actually rendered, excluding stale widget state."""
    import streamlit as st

    st.session_state.setdefault("navigation:selectors", []).append(key)
    return key


def begin_screen(state):
    previous = state.get("navigation:current")
    if previous and previous["page"] != state.get("workspace_navigation", "Workflow"):
        history = state.setdefault("navigation:history", [])
        history.append(previous)
        del history[:-50]
        state.pop("navigation:current", None)
    state["navigation:selectors"] = []


def screen(state):
    page = state.get("workspace_navigation", "Workflow")
    keys = {
        "Workflow": ("workflow_step",),
        "Runs": ("runs:selected", "runs:selected_ids", "runs:filter", "runs:states"),
        "Results": ("results:registered", "value:results:path"),
        "Tools": ("tools:selection",),
        "Help": ("help:selection",),
    }.get(page, ())
    values = {key: deepcopy(state[key]) for key in keys if key in state}
    if page == "Workflow":
        values["workflow_step"] = state.get("workflow_step", "Start")
    if page == "Results":
        values.update(
            {
                key: deepcopy(state[key])
                for key in state.get("navigation:selectors", [])
                if key in state
            }
        )
    return {"page": page, "values": values}


def remember_screen(state):
    current = screen(state)
    previous = state.get("navigation:current")
    if previous and previous != current:
        history = state.setdefault("navigation:history", [])
        history.append(previous)
        del history[:-50]
    state["navigation:current"] = current


def go_back(state):
    history = state.get("navigation:history", [])
    if not history:
        return
    target = history.pop()
    # Clear only navigation selectors; keep edits, drafts, jobs and computed results.
    if target["page"] == "Results":
        for key in list(state):
            if str(key).startswith(RESULT_PREFIXES):
                del state[key]
    reset_keys = {
        "Runs": ("runs:selected", "runs:selected_ids", "runs:filter", "runs:states"),
        "Results": ("results:registered", "value:results:path", "widget:results:path"),
    }.get(target["page"], ())
    for key in reset_keys:
        state.pop(key, None)
    state.update(deepcopy(target["values"]))
    if "value:results:path" in target["values"]:
        state["widget:results:path"] = target["values"]["value:results:path"]
    if "workflow_step" in target["values"]:
        state["step_widget"] = target["values"]["workflow_step"]
    state["workspace_navigation"] = target["page"]
    state["navigation:current"] = deepcopy(target)
    # Rebuild the run table's selection state so Back cannot reuse a stale row event.
    state["runs:table_epoch"] = state.get("runs:table_epoch", 0) + 1
