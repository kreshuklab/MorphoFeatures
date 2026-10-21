"""Intent routing for the workspace, independent of widgets and execution.

Presentation choices compile to existing stage documents. Imports and the
scientific defaults of direct presets are deliberately unchanged.
"""

from copy import deepcopy

FAMILIES = {
    "create": "Create representations",
    "explore": "Explore and evaluate",
    "learn": "Learn from labels",
}
FAMILY_HELP = {
    "create": "Prepare objects, extract embeddings, or construct new features from saved vectors.",
    "explore": "Explore a dataset, compare methods on the same objects, or compare independent datasets.",
    "learn": "Learn a compact representation from known cell types, or predict labels in another dataset.",
}
INPUTS = {
    "images": "Images and instance segmentation",
    "prepared": "Prepared crops or patches",
    "checkpoint": "A trained checkpoint and matching data",
    "vectors": "Saved embedding vectors",
    "result": "An existing analysis result",
    "published": "Published reference features",
}
SOURCES = {"raw": "Local containers", "platybrowser": "PlatyBrowser", "ome_zarr": "OME-Zarr"}
RECIPES = {
    "preprocess": "Prepare data only",
    "dino": "Extract with frozen DINOv2/v3",
    "checkpoint": "Extract with an existing MAE",
    "crops": "Train a new MAE, then extract",
    "dino_experiment": "Compare physical DINO fine/coarse recipes",
}
TASKS = {
    "create": {"fusion": "Combine feature blocks for the same objects", "context": "Add physical neighborhood context"},
    "explore": {
        "embeddings": "Explore one representation",
        "compare": "Compare representations of the same objects",
        "joint_embeddings": "Fit a shared map across independent datasets",
        "bilateral": "Test bilateral-partner retrieval",
    },
    "learn": {"distill": "Learn a compact representation using cell types", "transfer": "Predict labels in another dataset"},
}
PRESET_FAMILIES = {
    "raw": "create", "platybrowser": "create", "ome_zarr": "create",
    "preprocess": "create", "crops": "create", "dino": "create",
    "checkpoint": "create", "dino_experiment": "create", "fusion": "create",
    "context": "create", "embeddings": "explore", "joint_embeddings": "explore",
    "bilateral": "explore", "published": "explore", "distill": "learn",
    "transfer": "learn", "demo": "demo",
}
PAGE_ACTIONS = {
    "Prepare data": {"preprocess", "dino_experiment"},
    "Train": {"train"},
    "Extract": {"extract"},
    "Build features": {"fuse", "context"},
    "Learn": {"distill", "transfer"},
    "Analyze": {"analyze", "compare", "joint_analyze", "bilateral", "dino_experiment", "analysis_variant"},
    "Export": {"export_labels", "export_mobie", "export_bilateral"},
}


def workflow_steps(stages):
    pages = []
    for stage in stages:
        for page, actions in PAGE_ACTIONS.items():
            if stage.get("action") in actions and page not in pages:
                pages.append(page)
    return ("Start", *pages, "Review & run")


def entry_page(stages):
    return workflow_steps(stages)[1]


def _question_stages(task, name, *, embedding=None):
    """Connect one known representation to a task, leaving extra inputs explicit."""
    from morphofeatures.workspace_state import new_draft

    stages = new_draft("embeddings" if task == "compare" else task)["document"]["stages"]
    stage = stages[0]
    single = {"embedding": embedding} if embedding else {"from_embedding": name}
    if task in {"embeddings", "distill", "context"}:
        stage.pop("embedding", None)
        stage.update(single)
    elif task in {"compare", "bilateral", "fusion"}:
        stage.pop("embedding", None)
        stage["action"] = {"compare": "compare", "bilateral": "bilateral", "fusion": "fuse"}[task]
        stage["embeddings"] = {name: embedding} if embedding else {}
        if not embedding:
            stage["from_embeddings"] = [name]
        if task != "bilateral":
            stage["embeddings"]["Another representation (required)"] = ""
    elif task == "joint_embeddings":
        stage["datasets"][0] = {"name": name, **single}
    elif task == "transfer":
        stage["reference"] = {"name": name, "annotations": "", **single}
    return stages


def guided_draft(*, family, inputs, task=None, source="raw", recipe="dino", model_defaults=None):
    """Build a plan without reading inputs, reserving a run, or submitting work."""
    from morphofeatures.workspace_state import new_draft, stages_after_preparation

    if inputs == "result":
        raise ValueError("Open the existing result, then choose its next experiment.")
    if inputs == "published":
        draft = new_draft("published", model_defaults=model_defaults)
        if task and task != "embeddings":
            stage = draft["document"]["stages"][0]
            draft["document"]["stages"] = _question_stages(task, stage["published_reference"], embedding=stage["embedding"])
        draft["source_document"] = deepcopy(draft["document"])
        draft["intent"] = {"family": family, "inputs": inputs, "task": task}
        return draft
    if inputs in {"images", "prepared", "checkpoint"}:
        if inputs == "checkpoint":
            recipe = "checkpoint"
        if recipe == "dino_experiment":
            if inputs != "images":
                raise ValueError("Physical fine/coarse experiments need the original calibrated volumes.")
            draft = new_draft(recipe, model_defaults=model_defaults)
        elif inputs == "images":
            draft = new_draft(source if source != "raw" else "preprocess", model_defaults=model_defaults)
            prep = draft["document"]["stages"][0]
            draft["document"]["stages"] = stages_after_preparation(
                prep, mae={"crops": "train", "checkpoint": "checkpoint"}.get(recipe, "none"),
                dino=recipe == "dino", model_defaults=model_defaults,
            )
        else:
            draft = new_draft(recipe, model_defaults=model_defaults)
        if family != "create" and recipe not in {"preprocess", "dino_experiment"}:
            stages = [stage for stage in draft["document"]["stages"] if stage["action"] != "analyze"]
            extraction = next(stage for stage in reversed(stages) if stage["action"] == "extract")
            extraction.setdefault("name", "Extracted representation")
            task = task or next(iter(TASKS[family]))
            draft["document"]["stages"] = stages + _question_stages(task, extraction["name"])
    else:
        task = task or next(iter(TASKS[family]))
        draft = new_draft("embeddings" if task == "compare" else task, model_defaults=model_defaults)
        if task == "compare":
            stage = draft["document"]["stages"][0]
            stage.pop("embedding", None)
            stage.update(action="compare", embeddings={"Representation A": "", "Representation B": ""})
            draft["name"] = "Compare representations"
    draft["source_document"] = deepcopy(draft["document"])
    draft["intent"] = {"family": family, "inputs": inputs, "task": task}
    return draft


def retain_draft(state, root, draft, step="Start"):
    """Keep live documents, not snapshots of old edits, when switching drafts."""
    key = "workflow:" + str(root)
    bank = state.setdefault("drafts:" + str(root), {})
    previous = state.get(key)
    if previous:
        bank[previous["id"]] = previous
    draft = deepcopy(draft)
    bank[draft["id"]] = draft
    state[key] = draft
    state["workflow_active_id"] = draft["id"]
    state["workflow_step"] = step
    state.pop("workspace_plan", None)


def restore_draft(state, root, identifier):
    bank = state.get("drafts:" + str(root), {})
    current = state.get("workflow:" + str(root))
    if current:
        bank[current["id"]] = current
    if identifier in bank:
        state["workflow:" + str(root)] = bank[identifier]
        state["workflow_active_id"] = identifier
        state.pop("workspace_plan", None)
        return True
    return False
