"""Plan DINO host RAM from job limits, then refine against indexed object boxes.

All amounts are MiB. The 128-byte/voxel estimate is shared with the admission
scheduler. Budgets are estimates, not OS reservations. Scientific source limits,
sampling, precision and object cohorts are never changed by this module.
"""

from __future__ import annotations

import math
import os
import re
from copy import deepcopy
from pathlib import Path

from morphofeatures.artifacts import write_json_atomic
from morphofeatures.workflow_validation import FieldError

MIB = 1024**2
OBJECT_BYTES_PER_VOXEL = 128
WORKER_BASE_MB = 512


def memory_mib(value):
    """SLURM --mem quantities: a bare number is MiB; suffixes use powers of 1024."""
    match = re.fullmatch(r"([0-9]+)(?:([KMGTP])i?B?)?", str(value).strip(), re.I)
    if not match:
        raise ValueError("Enter job memory such as 100G or 102400M.")
    scale = {"": 1, "K": 1 / 1024, "M": 1, "G": 1024, "T": 1024**2, "P": 1024**3}
    return math.floor(int(match[1]) * scale[(match[2] or "").upper()])


def resident_mib():
    try:
        return math.ceil(
            int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE") / MIB
        )
    except (OSError, ValueError, IndexError):
        return 0


def system_limits(*, environ=None, proc=Path("/proc"), cgroup=Path("/sys/fs/cgroup")):
    """Read allocation and container limits without querying a scheduler or volume.

    Available RAM adds back this process's resident set: the plan includes it in
    its model/main-process reserve. Cgroup ancestors can be stricter than the leaf.
    """
    env = os.environ if environ is None else environ
    limits = {}
    for name in ("SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU"):
        try:
            value = memory_mib(env.get(name, "0"))
            if name.endswith("CPU"):
                value *= int(env.get("SLURM_CPUS_PER_TASK", "1"))
            if value > 0:
                limits[name] = value
        except (ValueError, TypeError):
            pass
    rss = resident_mib()
    try:
        for line in (proc / "meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                limits["available host RAM"] = int(line.split()[1]) // 1024 + rss
    except (OSError, ValueError, IndexError):
        pass
    roots = [(cgroup, "memory.max", "memory.current")]
    try:
        for line in (proc / "self/cgroup").read_text().splitlines():
            _, controllers, relative = line.split(":", 2)
            if not controllers:
                roots.append((cgroup / relative.lstrip("/"), "memory.max", "memory.current"))
            elif "memory" in controllers.split(","):
                roots.append(
                    (
                        cgroup / "memory" / relative.lstrip("/"),
                        "memory.limit_in_bytes",
                        "memory.usage_in_bytes",
                    )
                )
    except (OSError, ValueError):
        pass
    seen = set()
    for leaf, maximum, current in roots:
        for directory in (leaf, *leaf.parents):
            if not directory.is_relative_to(cgroup) or (directory, maximum) in seen:
                continue
            seen.add((directory, maximum))
            try:
                ceiling = int((directory / maximum).read_text().strip())
                if ceiling >= 2**60:  # v1's unlimited sentinel
                    continue
                used = int((directory / current).read_text().strip())
                # Large volume reads can leave reclaimable file pages charged to
                # the job. Counting these as irreducible use would manufacture a
                # smaller budget on every scale/checkpoint of the same run.
                inactive = 0
                try:
                    stats = dict(
                        line.split()
                        for line in (directory / "memory.stat").read_text().splitlines()
                    )
                    inactive = int(stats.get("total_inactive_file", stats.get("inactive_file", 0)))
                except (OSError, ValueError):
                    pass
                working = used - min(used, max(0, inactive))
                limits[f"cgroup {directory}"] = max(0, (ceiling - working) // MIB) + rss
            except (OSError, ValueError):
                continue
    return limits


def object_estimates(source, objects):
    from morphofeatures.data.object_eligibility import object_geometry

    estimates = {}
    for row in objects.to_dict("records"):
        geometry, *_ = object_geometry(source, row)
        estimates[int(row["label_id"])] = (
            math.ceil(int(geometry["required_voxels"]) * OBJECT_BYTES_PER_VOXEL / MIB)
            if geometry["decision"] in {"eligible", "partial"}
            else 0
        )
    return estimates


def plan_memory(settings, *, source=None, objects=None, detect=False, limits=None, rss_mb=None):
    """Pure planning apart from explicitly requested worker-local limit detection.

    Manual + no correction fails pointedly; all other modes repair contradictory
    execution budgets. An individual object that cannot fit is never skipped.
    """
    from morphofeatures.dino_runtime import runtime_options

    requested = runtime_options(settings)
    runtime = deepcopy(requested)
    auto = runtime["memory_mode"] == "auto"
    correct = auto or runtime["auto_adjust_memory"]
    constraints = dict(system_limits() if detect and limits is None else limits or {})
    for key, title in (
        ("allocation_memory_mb", "submitted SLURM memory"),
        ("job_memory_mb", "user job RAM budget"),
    ):
        if runtime.get(key):
            constraints[title] = int(runtime[key])
    # Local auto is intentionally conservative, rather than claiming all RAM on a shared host.
    if (
        not runtime.get("allocation_memory_mb")
        and not runtime.get("job_memory_mb")
        and not any(k.startswith("SLURM_MEM") for k in constraints)
    ):
        constraints["local default (set Job RAM budget to change)"] = 8192
    total = min(constraints.values()) if constraints else 8192
    rss = (resident_mib() if detect else 0) if rss_mb is None else rss_mb
    reserve = max(2048, math.ceil(total * 0.25), rss + 512)
    usable = total - reserve
    if usable < 128:
        raise FieldError(
            f"DINO job RAM budget is {total:,} MiB, leaving less than 128 MiB after the {reserve:,} MiB model/main-process reserve. Increase Review & run → Memory (SLURM), or Analyze → Job RAM budget for a local run; check the worker's container limit too.",
            "runtime.job_memory_mb",
        )
    changes = []

    def change(key, value, why):
        previous = runtime[key]
        if previous == value:
            return
        if not correct:
            raise FieldError(
                f"DINO RAM budget: {key}={previous} needs {value}. {why} Enable automatic memory correction, change this field, or increase the job RAM allocation.",
                "runtime." + key,
                "runtime.auto_adjust_memory",
                "runtime.job_memory_mb",
            )
        runtime[key] = value
        changes.append(
            {"field": "runtime." + key, "requested": previous, "effective": value, "reason": why}
        )

    cpu_limit = runtime.get("allocation_cpus")
    if detect:
        try:
            available = len(os.sched_getaffinity(0))
        except (OSError, AttributeError):
            available = os.cpu_count() or 1
        cpu_limit = min(
            cpu_limit or available, available, int(os.environ.get("SLURM_CPUS_PER_TASK", available))
        )
    if cpu_limit:
        change(
            "worker_threads",
            min(runtime["worker_threads"], cpu_limit),
            "Keep preparation threads within allocated CPUs.",
        )
        change(
            "prepare_workers",
            min(runtime["prepare_workers"], max(0, (cpu_limit - 1) // runtime["worker_threads"])),
            "Reserve one allocated CPU for the model/main process.",
        )
    if objects is not None:
        # Using fewer workers for a short cohort is scheduling, not a user error.
        before = runtime["prepare_workers"]
        runtime["prepare_workers"] = min(runtime["prepare_workers"], len(objects))
        if before != runtime["prepare_workers"]:
            changes.append(
                {
                    "field": "runtime.prepare_workers",
                    "requested": before,
                    "effective": runtime["prepare_workers"],
                    "reason": "Use no more reader processes than indexed objects.",
                }
            )
    estimates = object_estimates(source, objects) if objects is not None else {}
    largest_id = max(estimates, key=estimates.get) if estimates else None
    largest = estimates.get(largest_id, 0)
    size = int(settings.get("views", {}).get("size", 224))
    batch = int(settings.get("views", {}).get("batch_size", 16))
    image_bytes = 2 * (size * size * 5 + 4096)
    minimum_queue = max(1, math.ceil(image_bytes / MIB))
    queue_target = max(minimum_queue, min(256, max(8, usable // 50)))
    if auto or runtime["prefetch_mb"] < minimum_queue or runtime["prefetch_mb"] > usable // 4:
        change(
            "prefetch_mb",
            queue_target,
            "Bound queued images while leaving RAM for complete object preparation.",
        )
    # Normalized RGB tensors, masks, pinned copies and transient host buffers.
    per_image_host = size * size * 64 / MIB
    max_batch = max(1, math.floor(usable * 0.10 / per_image_host))
    if batch > max_batch:
        if not correct:
            raise FieldError(
                f"Images per model batch needs about {math.ceil(batch * per_image_host):,} MiB of host buffers; reduce it to at most {max_batch} for this RAM budget.",
                "views.batch_size",
            )
        changes.append(
            {
                "field": "views.batch_size",
                "requested": batch,
                "effective": max_batch,
                "reason": "Limit host image/tensor buffers to 10% of usable job RAM.",
            }
        )
        batch = max_batch
    host_batch = math.ceil(batch * per_image_host)
    maximum_prep = usable - runtime["prefetch_mb"] - host_batch
    if maximum_prep < 1:
        raise FieldError(
            "The selected DINO image cannot fit this job RAM budget, even with a single-image batch. Increase the allocation; image dimensions are not changed automatically.",
            "runtime.job_memory_mb",
            "views.size",
        )
    if auto or runtime["prepare_memory_mb"] > maximum_prep:
        change(
            "prepare_memory_mb",
            maximum_prep,
            "Use job RAM remaining after the main-process reserve, queued images and host batch buffers.",
        )
    if auto:
        change(
            "decoded_cache_mb",
            min(128, maximum_prep // (16 * max(1, runtime["prepare_workers"]))),
            "Use small per-volume caches; raw and segmentation each have one per reader.",
        )

    def fixed_mb():
        return runtime["prepare_workers"] * WORKER_BASE_MB

    def cache_mb():
        return max(1, runtime["prepare_workers"]) * 2 * runtime["decoded_cache_mb"]

    required = largest + fixed_mb() + cache_mb() + 1
    if required > runtime["prepare_memory_mb"]:
        change(
            "prepare_memory_mb",
            min(maximum_prep, required),
            "Make room for the largest eligible object, worker overhead and both decoded caches.",
        )
    room = runtime["prepare_memory_mb"] - largest - fixed_mb() - 1
    if cache_mb() > room:
        change(
            "decoded_cache_mb",
            max(0, room // (2 * max(1, runtime["prepare_workers"]))),
            "Reduce caches so a complete object can fit; source pixels and selected views are unchanged.",
        )
    while (
        largest + fixed_mb() + cache_mb() + 1 > runtime["prepare_memory_mb"]
        and runtime["prepare_workers"]
    ):
        change(
            "prepare_workers",
            runtime["prepare_workers"] - 1,
            "Reduce concurrent reader processes and their overhead to fit the job allocation.",
        )
    if largest + fixed_mb() + cache_mb() + 1 > runtime["prepare_memory_mb"]:
        raise FieldError(
            f"Object {largest_id} needs approximately {largest:,} MiB preparation scratch, but at most {maximum_prep:,} MiB remains in the {total:,} MiB job budget. Even serial preparation with caches disabled cannot fit. Increase Review & run → Memory / Job RAM budget, or explicitly choose different data/resolution. No object has been skipped or clipped.",
            "runtime.job_memory_mb",
            "runtime.prepare_memory_mb",
        )
    runtime["worker_overhead_mb"] = WORKER_BASE_MB
    notes = [
        "Host RAM estimate, not a hard reservation or a GPU-memory allowance. Sampling, precision and object limits are unchanged."
    ]
    if objects is None:
        notes.append(
            "Object sizes are not yet known: workers recheck indexed bounds before preparing image data."
        )
    return {
        "schema": "morphofeatures.dino_memory.v1",
        "budget_mb": total,
        "constraints_mb": constraints,
        "main_reserve_mb": reserve,
        "host_batch_mb": host_batch,
        "queue_mb": runtime["prefetch_mb"],
        "preparation_mb": runtime["prepare_memory_mb"],
        "decoded_caches_mb": cache_mb(),
        "worker_overhead_mb": fixed_mb(),
        "largest_object_mb": largest,
        "largest_object_id": largest_id,
        "indexed_objects": len(estimates),
        "estimated_total_mb": reserve
        + runtime["prefetch_mb"]
        + host_batch
        + runtime["prepare_memory_mb"],
        "requested": requested,
        "runtime": runtime,
        "batch_size": batch,
        "adjustments": changes,
        "notes": notes,
    }


def announce_plan(plan, destination, writer=None, *, phase="before preparation"):
    plan = {**plan, "phase": phase}
    write_json_atomic(Path(destination) / "memory_plan.json", plan)
    limiting = ", ".join(k for k, v in plan["constraints_mb"].items() if v == plan["budget_mb"])
    message = f"DINO RAM plan ({phase}): {plan['budget_mb']:,} MiB job budget, limited by {limiting}; {plan['main_reserve_mb']:,} MiB main/model reserve; {plan['preparation_mb']:,} MiB preparation (includes workers and caches); {plan['queue_mb']:,} MiB queue; {plan['host_batch_mb']:,} MiB host batch."
    print(message, flush=True)
    if writer:
        writer.write("dino_memory_plan", message=message, **plan)
    for change in plan["adjustments"]:
        message = f"DINO memory adjustment: {change['field']}: {change['requested']} → {change['effective']}. {change['reason']}"
        print(message, flush=True)
        if writer:
            writer.write("dino_memory_adjustment", message=message, **change)


def physical_stages(document):
    for index, stage in enumerate(document.get("stages", [])):
        if stage.get("action") == "dino_experiment" or (
            stage.get("action") == "extract"
            and stage.get("model") in {"dinov2", "dinov3"}
            and stage.get("views", {}).get("mode", "stored") != "stored"
        ):
            yield index, stage


def attach_allocation(document, execution, resources):
    """Freeze the submitted allocation; worker detection can only tighten it."""
    for _, stage in physical_stages(document):
        runtime = stage.setdefault("runtime", {})
        runtime.pop("allocation_memory_mb", None)
        runtime["allocation_cpus"] = int(resources.get("cpus", 4))
        if execution in {"slurm", "dry-run"}:
            runtime["allocation_memory_mb"] = memory_mib(resources.get("memory", "8G"))


def review_plans(document):
    """No volume scan or model load on the app host; exact bounds are checked later."""
    from morphofeatures.workflow_validation import WorkflowValidationError, stage_issue

    plans = []
    for index, stage in physical_stages(document):
        try:
            plans.append({"stage_index": index, **plan_memory(stage)})
        except FieldError as error:
            raise WorkflowValidationError(stage_issue(error, index, stage)) from error
    return plans
