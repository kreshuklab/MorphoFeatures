"""Shared controls for physical DINO execution; scientific view controls stay separate."""


def runtime_fields(draft, root):
    import streamlit as st

    from morphofeatures.dino_runtime import DEFAULTS
    from morphofeatures.workflow_ui import field, validation_expander

    labels = {
        "resume": (
            "Resume completed objects",
            "Keep this enabled to reuse each fully encoded object after an interruption. Only unfinished objects are repeated. Turning it off recomputes this extraction; it does not delete the original images.",
        ),
        "prepare_workers": (
            "CPU preparation workers",
            "0 prepares images in the main process. Try 2–4 to prepare objects while the GPU encodes earlier images. Each worker opens its own raw/segmentation readers; more workers can overload a remote server or shared disk.",
        ),
        "worker_threads": (
            "CPU threads per preparation worker",
            "Limit numerical-library threads inside each worker. Start with 1 to avoid several workers competing for the same reserved CPU cores.",
        ),
        "prepare_memory_mb": (
            "Preparation RAM budget (MiB)",
            "Manual target for complete-object preparation, including reader-process overhead and two decoded caches per reader. Automatic mode calculates this from the job RAM allocation. Objects are estimated at 128 bytes per bounding-box voxel in both serial and parallel preparation. This is separate from the source's maximum object-box voxel limit.",
        ),
        "prefetch_mb": (
            "Queued image budget (MiB)",
            "Bounds prepared image messages waiting for inference, including an allowance for serialization. Worker/current model batches need additional RAM. Lower this if the CPU processes consume too much memory.",
        ),
        "decoded_cache_mb": (
            "Decoded chunk cache per volume / worker (MiB)",
            "Keep recently read raw/segmentation chunks in RAM so overlapping views do not decode them repeatedly. 0 disables it. There can be two such caches per worker, so account for all workers in your memory request.",
        ),
        "scratch_directory": (
            "Node-local scratch directory (optional)",
            "Optional writable directory on the compute node for downloaded chunks and prepared images. Use a real path available to the worker. Embeddings and recovery checkpoints still go to the shared embedding cache; losing scratch does not lose completed vectors.",
        ),
        "prepared_cache_mb": (
            "Prepared image disk cache (MiB; 0 disables)",
            "Optionally reuse the actual normalized images across compatible DINO checkpoints. This uses scratch when configured, otherwise the embedding cache directory. Entries are byte-bounded; objects larger than the budget are streamed without caching.",
        ),
        "spatial_order": (
            "Read nearby objects together",
            "Process objects in bounding-box spatial order to improve chunk reuse. Per-object sampling and final vector order remain unchanged. The order of progress events and preview examples may differ.",
        ),
        "precision": (
            "Inference precision",
            "float32 preserves the baseline arithmetic. On supported CUDA devices, bfloat16/float16 can reduce model memory and improve speed; compare feature drift before adopting them. Unsupported devices fall back to float32 and record why. Precision is recorded in cache provenance.",
        ),
        "pin_memory": (
            "Pin image batches for GPU transfer",
            "Use page-locked host memory for CUDA image batches. This can improve CPU-to-GPU transfers but consumes host RAM. CPU inference does not pin batches.",
        ),
        "async_transfer": (
            "Allow asynchronous GPU transfers",
            "Copy pinned image batches without blocking the CPU. GPU operations remain correctly ordered. CPU preparation can continue in its workers; this does not imply a guaranteed transfer speedup.",
        ),
        "batch_across_objects": (
            "Fill model batches across object boundaries",
            "Let fine/coarse images from different objects share a model batch. Every image retains its own object ID and pooling group. Disable to benchmark batches that finish at each object boundary.",
        ),
        "group_normalization_reads": (
            "Group intensity sample reads by chunk",
            "Read the same deterministic normalization samples grouped by storage chunk, preserving their coordinates and interpolation. Disable for a baseline performance comparison.",
        ),
        "reuse_object_mask": (
            "Reuse the loaded object mask",
            "Sample each view's foreground from the already read object mask instead of repeatedly fetching the segmentation. Disable for a baseline performance comparison.",
        ),
    }
    with validation_expander(draft, "DINO execution, caching and recovery"):
        st.caption(
            "Automatic memory planning uses the SLURM RAM request on Review & run, or the local job budget below. It reserves RAM for the model, then balances object preparation, caches and queued images. Patch selection, resolution and precision are unchanged."
        )
        field(
            draft,
            root + ["runtime", "memory_mode"],
            "DINO memory planning",
            "auto",
            options=["auto", "manual"],
            format_func=lambda value: {
                "auto": "Automatic from job RAM (recommended)",
                "manual": "Manual preparation, queue and cache budgets",
            }.get(value, value),
            help="Automatic calculates the three RAM sub-budgets from the whole job allocation. Manual uses your entries below and can optionally repair conflicts. Review shows the effective plan; the worker refines it using the indexed object sizes.",
        )
        field(
            draft,
            root + ["runtime", "job_memory_mb"],
            "Job RAM budget (MiB; 0 uses allocation)",
            0,
            help="For SLURM, 0 uses Review & run → Memory. Locally, 0 starts with an 8192 MiB budget, capped by available host/container RAM. Enter a local budget or a smaller ceiling here; it never increases a SLURM/container allocation. For example 102400 MiB corresponds to 100 GiB. GPU memory is separate.",
        )
        field(
            draft,
            root + ["runtime", "auto_adjust_memory"],
            "Repair manual RAM conflicts and retry smaller GPU batches",
            True,
            help="For manual budgets, permit logged changes to workers, caches, queue/preparation limits and host batch buffers to fit the job. Also retry GPU out-of-memory with smaller batches of the same images. Disable for strict manual control. Automatic memory mode always balances host budgets. A single object/image that cannot fit still needs a larger allocation; objects are never silently dropped.",
        )
        st.caption(
            "All controls below remain editable. In Automatic mode, Preparation RAM, Queued image budget and Decoded chunk cache are calculated targets; choose Manual to use your values. CPU worker count is retained where the allocation permits. Prepared image disk cache is disk space, not RAM."
        )
        for name, (label, help_text) in labels.items():
            kwargs = {"optional": True} if name == "scratch_directory" else {}
            if name == "precision":
                kwargs["options"] = ["float32", "bfloat16", "float16"]
            field(draft, root + ["runtime", name], label, DEFAULTS[name], help=help_text, **kwargs)


def memory_review(document, execution):
    """Show a cheap, provisional plan before submission; never scan image data."""
    from copy import deepcopy

    import pandas as pd
    import streamlit as st

    from morphofeatures.dino_memory import attach_allocation, review_plans

    planned = deepcopy(document)
    try:
        attach_allocation(planned, execution, document.get("slurm", {}))
        plans = review_plans(planned)
    except (ValueError, TypeError) as error:
        st.error(f"Review & run → DINO memory plan: {error}")
        return
    if not plans:
        return
    with st.expander("DINO job memory plan", expanded=True):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Stage": p["stage_index"] + 1,
                        "Job RAM (MiB)": p["budget_mb"],
                        "Main/model reserve (MiB)": p["main_reserve_mb"],
                        "Preparation (MiB)": p["preparation_mb"],
                        "Caches within preparation (MiB)": p["decoded_caches_mb"],
                        "Queue (MiB)": p["queue_mb"],
                        "Workers": p["runtime"]["prepare_workers"],
                        "Images per batch": p["batch_size"],
                    }
                    for p in plans
                ]
            ),
            hide_index=True,
        )
        st.caption(
            "This plan uses the configured allocation, not the app host's free RAM. Workers recheck actual SLURM/container limits and indexed object sizes before image preparation. Main/model reserve is at least 25% or 2 GiB. Total preparation includes caches and reader-process overhead; these are not extra allocations. GPU RAM and later PCA/UMAP/classifier memory are separate."
        )
        for p in plans:
            if p["adjustments"]:
                st.caption(
                    f"Stage {p['stage_index'] + 1}: calculated changes (requested values stay saved)"
                )
                st.dataframe(pd.DataFrame(p["adjustments"]), hide_index=True)
        st.caption(
            "No image scan occurs here. A known object too large for the worker budget produces a specific allocation request before encoding. memory_plan.json, performance.json and the run logs record effective settings and adjustments."
        )
