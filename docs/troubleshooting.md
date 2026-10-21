# Troubleshooting

## A failed run or failed dependency blocks deletion as “still running”

Older workers recorded their final outcome but could leave the last SLURM
observation at `RUNNING` or `PENDING`. Refresh previously skipped finished jobs,
so that observation could persist indefinitely and also block deleting upstream
inputs.

Open **Runs** again. The UI and deletion backend now reconcile the saved final
worker status, completion time and exit status, including dependent jobs. The
old observation is retained in provenance and displayed as outdated. No results
are removed by this repair. **Refresh SLURM** also updates finished jobs awaiting
a final scheduler status, while preserving the original failure message and
partial outputs. See [deleting runs](workspace_workflows.md#delete-finished-runs-and-their-results).

If the worker was killed before saving its outcome (for example a timeout or
OOM kill), use **Refresh SLURM** to obtain the final scheduler state. A later
confirmed live allocation or pending cancellation still blocks deletion. Missing
accounting alone, a run's age, or an error line in a log does not authorize removal.

## Fusion succeeds but comparison asks to confirm the same specimen

The message **“Confirm these representations describe the same specimen and
corresponding cells before matching IDs”** comes from the optional
**bilateral-partner retrieval** probe. It does not mean that raw and normalized
vectors are incompatible. Fusion's identity confirmation and the bilateral
probe's confirmation are separate settings.

In **Analyze → Compare representations → Bilateral-partner retrieval (optional)**,
select **These representations describe the same specimen and cells** if all
inputs represent the same specimen and corresponding cells. Raw and balanced
combinations built from the same source vectors meet this condition. Check the
cell/nucleus ID setting and use a candidate table for that specimen. Alternatively,
turn off **Evaluate bilateral-partner retrieval** to run fusion, projections and
classification without this optional probe. YAML uses:

```yaml
bilateral:
  enabled: true
  same_specimen: true  # Only after verifying specimen and cell correspondence.
  # Retain your candidate table, ID mapping and retrieval variants.
```

Older versions checked this only after computing the comparison figures. A failed
run can therefore contain valid fused NPZ files and partial plots without a
completed `comparison.json`. Those files do not imply that the whole analysis
finished. The corrected validator catches the issue during review, highlights the
nested bilateral field, and also checks direct analysis calls before computing
projections. Existing failed runs remain unchanged; reload their settings, correct
the probe setting, and submit a new run. Saved fused embeddings can also be used
directly as inputs to a new comparison, without rerunning DINO.

## DINO fine/coarse extraction takes days: measured case and runtime controls

This investigation was performed on 29 September 2026 for
`outputs/experiments/run-87f54f1f/workspace` (SLURM job `62163805`).
The investigation itself did not change extraction behavior. The approved
single-worker/single-GPU improvements are now implemented for physical
`whole_label`, `fixed_tiles` and `both` DINO views. SLURM-array sharding remains
deferred. Existing scientific options and stored-crop/grouped-patch extraction
remain available; the new preparation controls do not change their behavior.

### What happened in this run

The registry reports `TIMEOUT`, and `stderr.log` confirms termination at the
72-hour limit. The worker's `status.json` still says `running`: forced scheduler
termination prevented its final update. Runs now explains this discrepancy when
the registry records a failed/cancelled job.

| Evidence | Observed value |
| --- | --- |
| Backbone / allocation | DINOv2 ViT-B/14 with four register tokens; one GPU, 20 CPUs, 100 GB RAM |
| Selected objects | 11,497 Platynereis nuclei, followed by 46 Parhyale objects |
| Cohort and tile limits | `max_objects: 0` (all); `exhaustive: true` (ignores `max_tiles: 200`) |
| Fine views | 1 µm tiles; 50% minimum 3D foreground; centroid grid; 3 axes × 3 positions = up to 9 images per eligible tile |
| Coarse views | 3 axes × 5 positions = up to 15 images per object |
| Inference | 224 × 224 images, batches of 16, no explicit mixed-precision context in the extraction loop |
| PlatyBrowser raw grid actually resolved | Z,Y,X spacing 0.025, 0.010, 0.010 µm, finest level `s0` |
| PlatyBrowser segmentation grid | Z,Y,X spacing 0.100, 0.080, 0.080 µm |
| Progress before termination | 4,274 object-attempt events; the last completed event was ID 4274; preparation of ID 4275 had started |
| Generated images in the partial manifest | 2,385,198 fine + 64,048 coarse; these include the unfinished final object |
| Selected fine tiles in that prefix | Median 61, 95th percentile 84, maximum 330 per object |
| Observed time between object events | Mean 60.5 seconds, median 53.5 seconds, 95th percentile 87.1 seconds |
| Shared remote cache | 1 GiB compressed-download budget, on the shared filesystem |

Sources are `job.yaml`, `metrics.jsonl`, `stderr.log`, the registry record, and
`00-dino_experiment/dataset-{0,1}/input/{source.json,objects.tsv,eligibility.json}`.
The matching incomplete cache is
`outputs/embedding_cache/views-8441a254a12051acae35a72d7289331a284b91933d58e244cf6d93e905c7cd76`;
its `views.jsonl` is approximately 1.40 GB and ends during object 4275. Excluded
views are separate records; generated images are not proof that the last model
batch completed. The progress event's `processed` field counts coverage rows
(fine/coarse modes), so it must not be read as a nucleus count.

Approximately 37% of the selected Platy cohort was attempted. A constant-rate
extrapolation is roughly eight days for Platy alone; later objects may have
different costs, so this is a scale estimate, not a runtime guarantee. Parhyale
extraction, feature combination and analysis had not been reached. Three Platy
objects fail the bounds-only memory screening; those exclusions do not explain
the multi-day runtime.

A 1 µm fine view fitted to 224 pixels has an effective canvas pixel size of
4.46 nm. This enlarges the native 10 × 10 × 25 nm raw grid and adds no new
physical detail. Coarse views have larger fields of view but still sample the
finest raw level. The resolved source metadata is authoritative, rather than
the placeholder spacings visible in the unconnected source form.

### Costs established at the time of the investigation

The job was launched from a dirty checkout; these code-level findings refer to
the checkout examined during the investigation, before the improvements below.
Counts and timing above come from the run's saved artifacts.

- `dino_physical.extract_physical` processes objects serially: prepare an object,
  generate its images on the CPU, fill a batch, run DINO, transfer feature outputs
  back to the CPU, then continue preparing images. There is no producer/consumer
  prefetch pool. Reserving 20 CPUs does not create 20 object-preparation workers.
- `PhysicalObjects.prepare` samples up to 65,536 foreground intensities for
  normalization. `sample_volume` splits spatial bounding-box reads recursively.
  This bounds memory but can revisit overlapping chunks, especially for scattered
  normalization samples and repeated orthogonal sections.
- `HttpN5Array` fetches intersecting chunks one at a time and decompresses each
  cached compressed chunk again on each access. `HttpCache` updates filesystem
  timestamps on hits and scans/stats/sorts the cache entries on each newly cached
  download. A shared lock includes network/cache work. These are avoidable I/O,
  decompression and filesystem-metadata costs; their share of runtime has not
  yet been measured.
- GPU input transfer, forward pass and feature readback are synchronous in the
  extraction loop. Batches finish at each object/mode boundary. Inference already
  uses `torch.inference_mode`, but does not opt into autocast.
- All object vectors are accumulated in memory and exported only after the
  complete extraction. `extraction.json` marks a completed cache. An incomplete
  cache is not resumable: rerunning recomputes earlier objects and rewrites the
  view manifest. The remaining manifest and preview images cannot recover the
  lost feature vectors.

Useful optimizations already present must be retained: CLS and foreground-token
features share each physical-view forward pass; a backbone is reused across
datasets/scales; coarse outputs are reused across fine-scale variants.

The xFormers warnings are **not sufficient evidence of the bottleneck**. The
current local DINOv2 checkout falls back to PyTorch's scaled-dot-product attention.
The logs do not record GPU utilization, model-only time, network time, decoded
cache hits, or kernel selection. Installing xFormers or adding GPUs cannot yet
be assigned a defensible speedup estimate.

### Implementation scope and deferred work

The following list records the approved design. Controls and recovery details
for the implementation are described immediately below.

1. **Measure progress and make work resumable.** Record separate timings for
   indexing, raw/label reads, normalization, tile preparation, transfer to GPU,
   inference, pooling and export, plus view/object throughput, cache counters,
   peak memory and an estimated finish time. Save each completed object's
   vectors and coverage in atomic shards, including dataset/native ID and the
   complete extraction signature. Commit matching manifest offsets; discard only
   unfinished objects after interruption. Lock shared cache writes and refuse
   incompatible resume settings. This prevents another timeout from discarding
   completed inference, even before speed improves.
2. **Reduce repeated data reads and decoding.** Add a byte-bounded decoded-chunk
   cache; reuse the already loaded object mask for label sampling; group
   normalization sample reads by chunk with the required interpolation halo.
   Replace per-download full-directory eviction scans with an indexed or
   amortized policy, and narrow locks to individual cache entries. Offer
   node-local scratch for hot caches while committing reusable outputs to shared
   storage. Preserve interpolation, filtering, physical coordinates and exact
   sampled views. Spatially ordering object reads can improve reuse without
   changing per-ID sampling or final ID order.
3. **Overlap CPU preparation and GPU inference.** Use a bounded worker pool
   (initial benchmark: 2–4 preparation processes, not blindly all reserved CPUs)
   with independently opened volume handles and per-worker thread limits. A
   single GPU owner consumes pinned-memory image batches and assembles batches
   across object/mode boundaries. Maintain explicit ownership for every image,
   deterministic per-ID sampling and correct per-object aggregation. Bound queues
   by memory as well as item count: the configured 250-million-voxel object-box
   allowance can be very expensive when expanded into coordinates and masks.
4. **Tune inference after profiling.** Benchmark larger batches and asynchronous
   transfers; add opt-in BF16/FP16 autocast where supported, retaining safe
   accumulation precision and FP32 fallback. Record precision/backend in cache
   provenance and measure feature drift before adopting it. Verify the existing
   attention path first; do not change upstream DINO code or install a new
   attention package as an assumed fix. Prepared images could also be reused
   across compatible checkpoints, under an explicit disk/memory budget.
5. **Deferred: scale out only after the single-GPU path is measured.** Split a fixed object
   cohort into deterministic SLURM-array shards, each owning whole objects and
   one GPU. Merge completed shards by dataset/native ID before the single fusion
   and comparison stage. Cap concurrent streaming readers to avoid overwhelming
   the source service/shared filesystem. Reuse the same cohorts and validation
   folds across representations; never split one object's aggregation without
   explicit ownership.

### Using the new controls

In the fine/coarse workflow's **Analyze** page, expand **DINO execution, caching
and recovery**. The same section is available for a physical-view DINO extraction
stage. All existing tile, foreground-content, resolution, slice, boundary and
error-policy controls remain available. The reusable experiment YAML includes
an annotated `runtime:` block next to `views:`.

Start with **Resume completed objects** enabled, **Inference precision** set to
`float32`, and **CPU preparation workers** set to 2. Compare with 0 (serial) and
4 on a small, fixed cohort before selecting a production setting. The backend
default remains 0 workers for configurations that omit the option. Increase
**Images per model batch** separately if the GPU has room. Images can share a
batch across fine/coarse and object boundaries; each image retains its native ID
and aggregation group. CLS and foreground-token outputs still share one forward
pass. Different scheduling can cause ordinary floating-point roundoff, not a
change in the sampled images or cohort.

Use **DINO memory planning → Automatic from job RAM** to derive the budgets from
**Review & run → Memory**. Local runs use **Job RAM budget (MiB)**, with a
conservative 8192 MiB default when zero is entered. A `100G` SLURM request therefore
no longer leaves the preparation allowance at an unrelated 4096 MiB. Review
shows the provisional split; the worker refines it against actual allocation
limits and object bounds. See the [RAM planning walkthrough](workspace_workflows.md#let-the-jobs-ram-allocation-set-the-dino-budgets).

For manual tuning, select **Manual preparation, queue and cache budgets**. These
existing controls remain available:

- **Preparation RAM budget** admits complete object boxes using a conservative
  estimate of 128 bytes per box voxel, raw/segmentation caches per reader and
  512 MiB base overhead per extra worker. The same check applies serially. The
  planner separately reserves RAM for the main process/model and image buffers.
  An object that cannot fit is identified rather than silently excluded.
- **Queued image budget** bounds image messages waiting for inference, separately
  from each worker's current object. **CPU threads per preparation worker** avoids
  numerical libraries oversubscribing the reserved CPUs.
- **Decoded chunk cache per volume / worker** trades RAM for fewer reads and
  decompressions. There can be two caches per worker. Scattered intensity samples
  are grouped by chunk with interpolation halos, and the loaded object mask is
  reused for foreground sampling. Baseline switches allow these optimizations
  to be disabled without changing sampling settings.
- **Node-local scratch directory** places hot downloads and optional prepared
  images on worker-local storage. It must exist or be creatable on the worker;
  choose a real path, not a shell expression. Recovery vectors still go to the
  shared **Reusable embedding/view cache**. Scratch loss only requires re-reading
  or preparing data, not re-encoding committed objects.
- **Prepared image disk cache** is optional (0 disables it). Complete prepared
  objects can be reused across compatible checkpoints. Entries are bounded by
  bytes and pinned while in use; an object that cannot fit is streamed normally.
  Incomplete prepared entries are regenerated. This cache does not change the
  final feature recipe.

**Why did “caches exceed preparation RAM” appear despite a 100G job?** A cache
setting of 10,000 MiB is **per volume per worker**. Two workers opening raw and
segmentation volumes reserve 40,000 MiB of caches before accounting for any object.
Setting preparation to 20,000 or 40,000 MiB cannot accommodate them. Automatic
mode uses smaller caches and gives preparation the RAM available from the whole
job. Manual mode can repair this when **Repair manual RAM conflicts and retry
smaller GPU batches** is enabled; disable that checkbox for strict rejection.

`memory_plan.json` and `performance.json` preserve requested/effective budgets;
the stdout log and `metrics.jsonl` report each adjustment. If inference exhausts
GPU memory, enabled recovery retries the same images in smaller batches, retaining
their order and precision. A single image that cannot fit still stops with a
pointed error. Neither automatic host planning nor GPU recovery changes object
cohorts or view geometry. Host planning cannot rescue a process already killed
by the OS; use completed-object checkpoints to resume with more RAM.

`bfloat16` and `float16` are opt-in CUDA experiments. Unsupported devices fall
back to `float32` and report the reason. Token pooling uses float32 and object
means use float64 accumulation before exporting float32 vectors. Effective
precision, runtime version, Torch version, device, matrix-multiplication
precision and installed xFormers version form the numerical cache contract.
Changing numerical settings produces a different extraction; changing
preparation workers, queue/cache budgets or batch size can resume the same
extraction. Check feature drift before adopting a lower precision. The upstream
attention implementation is unchanged; the report records attention module
classes and their xFormers availability, not a claim about which GPU kernel
actually executed.

### Recovery and reports

Each extraction cache contains `object-checkpoints/<native-ID>/vectors.npz` and
`object.json`. A commit records the dataset namespace, extraction signature,
coverage/exclusions and checksummed byte offsets into `views.jsonl`. Files are
flushed before the object directory is atomically committed. A shared extraction
lock prevents two workers from writing the same cache simultaneously. OS locks
are released if a worker is killed; a leftover lock filename is not an active
lock.

After an interruption, submit the same scientific configuration with the same
shared embedding cache. It verifies committed objects, discards unfinished
object work, truncates uncommitted manifest bytes and continues. Corrupt commits
or incompatible signatures are reported, not silently reused. Turning off
**Resume completed objects** explicitly recomputes that extraction. The old
72-hour run predates per-object feature checkpoints: its saved view manifest
cannot recover vectors that were only held in RAM.

- `progress.json` reports completed and resumed object counts, current-session
  images/second and objects/hour, and an estimated remaining time. ETA is based
  on completed objects and is approximate when later nuclei are larger.
- `performance.json` records indexing, reads, normalization, tile/plane
  preparation, host batches, device transfer, model inference, pooling,
  readback, checkpoint and final-export timings. Read/cache counters include
  completed objects from previous sessions; model timings refer to this session.
  Main-process and maximum worker RSS are sampled during extraction and reported
  separately, not as a measured simultaneous total. Sampling can miss brief
  memory spikes. GPU allocation is reported on CUDA. Timings overlap: for example,
  normalization includes its reads, and CPU readback can wait for queued GPU
  work. Do not add every timing field to estimate total wall time. Explicit
  decompression timings cover HTTP N5; other readers include decoding in their
  read time.
- Existing `coverage.tsv`, `excluded.tsv`, previews, embedding NPZs and metadata
  remain available. Worker outputs copy the final artifacts; internal recovery
  directories stay in the shared cache instead of being duplicated into every
  result folder.

### Benchmark before a long rerun

The maintained CLI command accepts an extraction YAML or a workflow containing
a fine/coarse experiment:

```bash
python -m morphofeatures dino-benchmark \
  --config /path/to/workflow.yaml \
  --output /path/to/new/benchmark-directory \
  --stage 0 --dataset 0 --backbone 0 --scale 0 \
  --objects 12 --workers 0 2 4 --batch-sizes 16 32 64 \
  --precisions float32 bfloat16 --device cuda
```

The indices select a stage, dataset, backbone and physical scale from the YAML.
The cohort is chosen once across the range of indexed object-box sizes, including
large objects. Supply an existing object index to avoid another volume scan.
The original scientific sampling settings are retained, so even a small cohort
can be expensive with exhaustive fine tiles. Use a new output directory for each
benchmark. No job is submitted by this command; run it inside the intended local
or SLURM allocation.

Each case runs with application-cold and application-warm caches. OS, server and
storage caches are not purged. Final feature caches are separate so both trials
actually run inference. `benchmark.tsv` summarizes timing and memory;
`benchmark.json` also records per-representation feature differences, cosine
drift, nearest-neighbor agreement and overlap and, when annotations permit, deterministic
held-out logistic-classifier checks. Small cohorts often cannot support a
classification estimate; the report explains why. GPU out-of-memory or invalid
cases are recorded separately. Compare throughput **and** numerical/biological
agreement before a full rerun; a CPU pilot cannot establish a GPU speedup.

### Local validation pilot (30 September 2026)

A small real-data pilot used the local DINOv2 ViT-B/14 register checkpoint and
three Platynereis nuclei from the local N5 source. All four fine/coarse ×
CLS/foreground-pooled outputs were **bit-identical** across the tested float32
settings. Each trial encoded 27 images at 224 × 224 pixels. Actual model-input
previews were also inspected.

| CPU-only setting | Application-cold wall time | Application-warm wall time |
| --- | ---: | ---: |
| Serial baseline, batch 4 | 31.0 s | 15.1 s |
| Optimized serial, batch 4 | 12.5 s | 12.5 s |
| Optimized serial, batch 16 | 12.3 s | 12.3 s |
| Two preparation workers, batch 4 | 15.6 s | 14.8 s |
| Two preparation workers, batch 16 | 14.8 s | 15.4 s |

This tiny CPU cohort does not amortize process startup. It supports the
correctness of the changes, but does not establish optimal worker/batch settings
or a production speedup. The pilot used at most two fine tiles per object, one
slice per axis and a 1% foreground cutoff; it is **not** equivalent to the timed-out
run's exhaustive tiles and 50% cutoff. No GPU was available, so CUDA throughput,
BF16/FP16 drift and GPU memory use remain to be measured in the intended
allocation. The local upstream attention path uses PyTorch scaled-dot-product
attention when xFormers is unavailable; no attention dependencies were installed.

A follow-up run after the memory-reporting changes again produced identical
vectors: 15.0–15.3 seconds with two workers, versus 13.6–15.7 seconds for the
serial baseline. Sampled RSS was approximately 1.3 GiB in the main process and
0.7 GiB for the largest worker; these include shared pages and are not an additive
measurement of the job's physical RAM use.

The configuration, exact cohort, timings, vectors and input previews are saved
under `outputs/benchmarks/dino-runtime-20260930/`; these are local validation
artifacts, not required repository files. Automated checks additionally cover
interruption/resume, native-ID preservation, bounded preparation, cache reuse,
DINOv2/v3 token pooling, and a controlled local HTTP N5 source.

### Validation before a full rerun

Benchmark a fixed representative subset, including large objects, with identical
views, foreground cutoffs, physical scales, model checkpoint and object IDs.
Measure cold and warm caches separately. Compare views/second, objects/hour, I/O
bytes, GPU busy time and peak CPU/GPU memory. Test a deliberate interruption and
resume; final embeddings must have exactly one row per valid ID and the same
coverage/exclusions. Parallel preparation should reproduce baseline inputs and
features within stated numerical tolerances. Mixed precision additionally needs
feature-distance, neighbor and held-out classification comparisons. Include both
local containers and a controlled HTTP source, and test DINOv2/DINOv3 pooling.
Do not promise a speedup factor until these measurements exist.

A smaller scientific experiment is a separate choice: turn off exhaustive tiles,
start with 100 objects, at most 64 tiles and one fine slice per axis (up to
192 fine + 15 coarse images/object). That reduces the experiment, rather than
optimizing identical computations. Retain the original recipe as a baseline and
compare biological performance before changing production sampling. Selecting
a coarser raw pyramid level or a different DINO input size likewise changes
image detail and must be explicit, physically validated and recorded.

**Separate Parhyale geometry issue to verify before resubmission:** the saved
source metadata and current H5 shapes have raw dimensions `(444,1716,4236)` and
segmentation dimensions `(444,3432,8472)`, yet both were assigned XY spacing
0.02 µm. Their reported physical XY extents therefore differ by a factor of two.
This may be intentional partial coverage, or the segmentation XY spacing may
not have been updated correctly. Check the original acquisition/resampling
metadata and overlay before changing it; dimensions alone cannot establish the
correct spacing. This was not the cause of the Platy timeout, since that run had
not started Parhyale extraction.
