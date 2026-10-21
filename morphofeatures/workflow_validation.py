"""Location-aware workflow checks shared by workers and the guided editor."""

import re
from contextlib import contextmanager
from dataclasses import asdict, dataclass

PAGES = {"preprocess": "Prepare data", "train": "Train"}
STAGES = {
    "distill": "Distill embeddings",
    "export_bilateral": "Export bilateral pairs in 3D",
    "bilateral": "Bilateral-partner retrieval",
    "context": "Build spatial context features",
    "dino_experiment": "DINO fine/coarse experiment",
    "fuse": "Combine embeddings",
    "transfer": "Transfer labels",
    "preprocess": "Prepare data",
    "train": "Train",
    "extract": "Extract embeddings",
    "analyze": "Analyze",
    "compare": "Compare representations",
    "joint_analyze": "Shared space across datasets",
    "export_labels": "Export labels to segmentation",
    "export_mobie": "Export to MoBIE",
}


def stage_page(action):
    return PAGES.get(action, "Analyze")


@dataclass(frozen=True)
class WorkflowIssue:
    message: str
    page: str
    stage_index: object = None
    stage_name: str = ""
    fields: tuple = ()
    severity: str = "error"

    def to_dict(self):
        return asdict(self)


class FieldError(ValueError):
    """A backend failure with known field paths relative to its configuration."""

    def __init__(self, message, *fields):
        super().__init__(message)
        self.fields = tuple(tuple(f.split(".")) if isinstance(f, str) else tuple(f) for f in fields)


class WorkflowValidationError(ValueError):
    def __init__(self, issue):
        self.issue = issue
        where = issue.page
        if issue.stage_index is not None:
            where += f" → Stage {issue.stage_index + 1}: {issue.stage_name}"
        super().__init__(where + ": " + issue.message)


@contextmanager
def at_fields(*fields):
    """Attribute lower-level file/reader failures to the field being checked."""
    try:
        yield
    except FieldError:
        raise
    except (ValueError, TypeError, KeyError, OSError, ImportError) as error:
        raise FieldError(str(error), *fields) from error


@contextmanager
def prefix_fields(*prefix):
    """Locate a nested validator's fields under its enclosing settings block."""
    try:
        yield
    except FieldError as error:
        raise FieldError(
            str(error), *(tuple(prefix) + path for path in (error.fields or ((),)))
        ) from error
    except (ValueError, TypeError, KeyError, OSError, ImportError) as error:
        raise FieldError(str(error), tuple(prefix)) from error


def _paths(value, prefix=()):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _paths(child, prefix + (str(key),))
    else:
        yield prefix


def _mentioned_fields(error, stage):
    """Legacy validators usually name their keys. Match whole keys, not substrings.

    Explicit FieldError locations take precedence. Unknown failures remain at
    stage level; we never invent a field attribution for an arbitrary exception.
    """
    if isinstance(error, FieldError):
        return error.fields
    message = str(error)
    candidates = list(_paths(stage))
    # Include missing required inputs even if the editor has not created them yet.
    candidates += [
        (k,)
        for k in (
            "checkpoint",
            "embedding",
            "annotations",
            "config",
            "model_repository",
            "variant",
            "raw",
            "segmentation",
            "roi",
        )
    ]
    matched = []
    for path in sorted(set(candidates), key=lambda p: -len(p)):
        names = [".".join(path)]
        if path and path[0] == "config":
            names.append(".".join(path[1:]))
        # Bare keys are accepted only when they identify exactly one setting.
        if path and sum(p[-1:] == path[-1:] for p in set(candidates)) == 1:
            names.append(path[-1])
        if any(
            name and re.search(r"(?<![\w.])" + re.escape(name) + r"(?![\w.])", message, re.I)
            for name in names
        ):
            if not any(existing[: len(path)] == path for existing in matched):
                matched.append(path)
    return tuple(matched)


def stage_issue(error, index, stage):
    if isinstance(error, WorkflowValidationError):
        return error.issue
    action = stage.get("action") if isinstance(stage, dict) else None
    name = STAGES.get(action, str(action or "Pipeline stage"))
    if isinstance(stage, dict) and stage.get("name"):
        name += " · " + str(stage["name"])
    fields = _mentioned_fields(error, stage) if isinstance(stage, dict) else ()
    page = stage_page(action)
    if action == "dino_experiment" and any(
        "source" in path
        or path[:1] in (("tile_sizes_um",), ("max_objects",))
        or (path[:1] == ("datasets",) and (len(path) < 3 or path[2] == "name"))
        or (
            path[:1] == ("views",)
            and len(path) > 1
            and path[1]
            in {
                "fine_fractions",
                "coarse_fractions",
                "axes",
                "size",
                "mask",
                "percentiles",
                "tile_size_um",
                "max_tiles",
                "max_candidate_tiles",
                "min_foreground_fraction",
                "min_slice_foreground_fraction",
                "tile_anchor",
                "exhaustive",
            }
        )
        for path in fields
    ):
        page = "Prepare data"
    return WorkflowIssue(
        str(error),
        page,
        index,
        name,
        tuple(("stages", index) + path for path in fields),
    )


def execution_issue(error, *, field=None, resources=None):
    if isinstance(error, WorkflowValidationError):
        return error.issue
    fields = ((field,),) if field else _mentioned_fields(error, {"slurm": resources or {}})
    if isinstance(error, FileExistsError):
        fields = (("run_id",),)
    return WorkflowIssue(str(error), "Review & run", fields=fields)


def check_workflow(document, base, *, indices=None, allow_synthetic=False):
    """Collect the first backend failure per stage, without allocating a run.

    Page checks skip other stages' I/O but retain their dependency declarations.
    Final review checks every stage, using the same resolver as CLI/submission.
    """
    from morphofeatures.workspace_jobs import resolve_job

    issues = []
    resolve_job(
        document, base, issues=issues, stage_indices=indices, allow_synthetic=allow_synthetic
    )
    return issues
