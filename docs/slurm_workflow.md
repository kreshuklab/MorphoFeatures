# Streamlit and SLURM workflow

## Lifecycle and safe defaults

Open the workspace on a login node (often through an SSH tunnel):

```bash
python -m pip install -e ".[ui]"
python -m morphofeatures ui --headless --address 127.0.0.1
```

For connected preprocessing, MAE training, extraction and analysis, use
**Workflow → Review & run**. **Review run** builds an exact preview without
reserving a run ID. **Save dry-run bundle** persists it without execution;
**Submit to Slurm** submits the reviewed settings and opens the run record.
The [workspace guide](workspace_workflows.md) describes this shared workflow.
The details below also cover the retained specialist training/sweep adapters.

The intended lifecycle is:

```text
validate inputs → configure experiment → preview → submit training → monitor
→ select checkpoint → submit encoding → inspect embedding → analyze/export
```

The **Tools → Published shape / texture workflows and MAE sweeps** tool supports the existing shape, texture, and MAE training/encoding
schemas. It does not flatten them into a universal scientific config. Shape/MAE YAML is copied
to an immutable run snapshot; recognized texture YAML files are copied together. Relative input
paths are made absolute before the snapshot is saved.

The optional input override is workflow-specific: a shape manifest/point-cloud root, a texture
data root, or an MAE crop `.npy`. It updates only that schema's existing input field. Other
scientific settings remain in the selected template rather than being silently discarded.

Dry run is the safe mode: **Save dry-run preview** persists the snapshot, script, provenance,
and registry record but calls no scheduler command. A real job is submitted only by the explicit
**Submit to SLURM** button. A unique content key prevents double-click/rerun duplicates for the
same run. Use a new run ID for an intentional changed or repeated experiment.

## Cluster profiles

`configs/slurm_profiles.example.yaml` contains placeholders, not institutional values. Obtain
partition, account, QoS, walltime, memory, CPU, GPU, and notification policy from cluster
documentation. Keep credentials and personal addresses out of profiles committed to source
control.

Point a deployment pipeline config at the real profile file with `slurm.profiles`, or set
`MORPHOFEATURES_SLURM_PROFILES`. Relative values resolve against the repository; an absolute
external path is appropriate for a shared cluster deployment.

Setup is represented as argument lists and is limited to `module load|purge|use` or
`source /path/to/activate`. The UI never accepts arbitrary shell text. Scientific commands are
built internally as argument lists for known `morphofeatures` subcommands and safely quoted only
when rendering the script. Scheduler subprocesses use no shell.

### Mapping a working cluster script to a profile

The renderer supports the operational pattern used by the supplied cluster example while keeping
site values separate from the repository. A private deployment profile can use this structure:

```yaml
profiles:
  gpu-production:
    partition: gpu-partition
    account: project-account
    qos: null
    nodes: 1
    ntasks: 1
    ntasks_per_node: 1
    cpus: 16
    memory: 50G
    time: "24:00:00"
    gpus: 1
    gpu_directive: gres
    mail_types: [END, FAIL]
    mail_user: researcher@example.org
    python_executable: /path/to/MorphoFeatures-environment/bin/python
    cpu_thread_env: true
    local_runtime_directories: [wandb, matplotlib]
    log_job_context: true
    setup:
      - [module, load, CUDA]
```

This produces `--gres=gpu:1`, explicit node/task directives, separate run logs, and the following
runtime behavior:

- `OMP_NUM_THREADS` and `MKL_NUM_THREADS` follow `SLURM_CPUS_PER_TASK`.
- W&B and Matplotlib caches default to directories beneath the immutable run directory.
- The job ID, host, working directory, and safely quoted command are printed at startup.
- The selected interpreter launches `python -m morphofeatures ...`; it does not permit choosing an
  arbitrary program or shell fragment.

Use `gpu_directive: gpus` only if local documentation calls for `--gpus=N`. The explicit Python
path is particularly useful when `sbatch` does not inherit an interactive Conda environment.
Alternatively, source a reviewed activation script through `setup`. Module versions and setup
paths are site-specific and belong in the private profile.

There is no generated `mkdir logs` step: the experiment materialization code creates the run
directory before calling `sbatch`, and stdout/stderr are absolute paths inside it. Likewise, the
positional `CONFIG`/`SCRIPT`/`shift` wrapper pattern from a hand-authored script is unnecessary:
the exact supported CLI command and immutable config path are generated from the validated
workflow request.

## Experiment directory and registry

Runtime state is under the configured output root:

```text
outputs/
├── .morphofeatures/registry.sqlite3
└── experiments/<run-id>/<workflow>/
    ├── config.yaml or experiment/*.yml
    ├── run.json
    ├── job.slurm
    ├── metrics.jsonl
    ├── stdout.log
    ├── stderr.log
    └── checkpoints / embedding artifacts
```

SQLite uses transactions, a busy timeout, WAL journaling, and a unique submission key. Records
include internal/run/workflow IDs, raw and normalized states, timestamps, exit status, commands,
dependency, paths, artifacts, git commit/dirty flag, Python, and platform. Browser refreshes and
Streamlit restarts do not lose records. Do not edit a saved snapshot; create a new run ID.

## Submission and dependencies

The backend calls `sbatch --parsable` and accepts only a numeric machine-readable job ID.
Encoding can select a parent job; submission adds `--dependency=afterok:<job-id>` so it begins
only after successful training. Stdout and stderr are separate. A failed `sbatch` attempt remains
in the registry with its error and proposed snapshot.

To resume texture training, select its explicit resume option and a new run ID/snapshot. For
other failed jobs, diagnose the logs/config, correct the source config, and create a new run.
Do not mutate a completed registry row to make it look resubmitted.

Cancellation requires selecting an active job and checking the confirmation box. The registry
retains the cancellation record.

### Bounded MAE soft grids

For an MAE training workflow, the same page can prepare or reopen a soft-grid manifest. The
manifest contains one immutable resolved config per variant. The default `one_at_a_time` mode is
an interpretable ablation; `cartesian` is available but both the YAML `max_runs` and a hard
64-run ceiling prevent accidental job explosions. Only these dotted configuration fields are
accepted: learning rate, scheduler, weight decay, patch encoder, embedding dimension,
reconstruction shape, mask ratio, and input normalization.

The page previews every generated script. **Prepare immutable sweep configurations** performs no
scheduler call. **Save all sweep jobs as dry runs** records only previews. **Submit soft grid to
SLURM** is the sole sweep submission action. When dependent encoding is selected, each successful
training submission creates a separate registered encoder with
`--dependency=afterok:<training-job-id>`. Double clicks and reruns are rejected by the same unique
registry keys as single jobs. Use a new sweep name for a corrected or repeated experiment.

Post-training comparison reads each explicit run directory rather than searching recursively. It
can tabulate last/best loss, plot training/validation curves, and render matched held-out target,
reconstruction, and error slices. A lower pretext loss is not sufficient evidence of a better
embedding; compare reconstruction structure, stability, and the same ID-aligned downstream probe.

## Monitoring and states

The **Runs** page filters jobs/runs and refreshes active numeric IDs in a single manual
operation. `squeue` supplies active states; IDs absent there are queried through `sacct`.

- `queued`: pending/configuring/requeued.
- `running`: running/completing/suspended/staging out.
- `completed`: scheduler completed successfully (also inspect exit status/artifacts).
- `failed`: failed, timeout, out of memory, node/boot failure, deadline, or preemption.
- `cancelled`: scheduler cancellation or confirmed cancellation request.
- `unknown`: new/unrecognized state or no current accounting record.
- `dry_run`: saved preview; never submitted.

Raw SLURM state is always retained. Accounting can be delayed or purged; an unknown record does
not by itself mean the job failed. Refresh later and consult cluster support if both commands fail.

Log views read at most 64 KiB and 200 lines. Metrics are append-only JSON Lines; a partially
written final line is ignored while training continues. Shape, texture, and MAE trainers preserve
console/WandB behavior while emitting epoch, loss, validation loss when available, learning rate,
checkpoint, completion, and controlled failure events. The real N5 MAE additionally emits named
preprocessing stages and bounded batch-level throughput/ETA events. The Runs page displays
the latest event, current phase progress, and the latest 100 structured events without reading an
unbounded log.

## Artifacts and analysis

`run.json` and registry paths define artifact discovery—there is no recursive filesystem guess.
The UI can inspect checkpoint metadata and embedding dimensions, finite values, label range, and
uniqueness. A completed embedding can become the default input on `Analyze`; existing pages then
run classification, projection/clustering, combination, context aggregation, and NumPy/TSV/CSV
or MoBIE-compatible export. All scientific joins remain by `label_id`.

## Troubleshooting

- **`sbatch` unavailable:** use dry run, or launch Streamlit on a SLURM login node.
- **Submission rejected:** inspect the retained error; verify account/partition/QoS/resources and
  profile setup with cluster documentation.
- **`squeue` unavailable:** active refresh may still recover completed state from `sacct`.
- **`sacct` delayed/purged:** the job remains `unknown`; logs and explicit artifacts remain usable.
- **Invalid job ID:** only IDs parsed from `sbatch --parsable` can be refreshed/cancelled.
- **Missing logs:** queued jobs may not have opened files yet; confirm workdir permissions.
- **Missing checkpoint/embedding:** check exit status, stderr, config snapshot, and metrics events.
- **CUDA/optional library failure:** run `morphofeatures doctor`, use CPU when appropriate, and
  install a Torch/compiled dependency set matching the cluster.

No live-cluster submission is required by or performed in automated tests. Fake-scheduler and
dry-run tests validate the lifecycle without a network, GPU, private data, or SLURM installation.
