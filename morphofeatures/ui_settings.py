"""User-facing explanations and choices shared by the configuration forms."""

HELP = {
    "patch_label_ids": "One exact integer parent-object ID per saved fine patch. Several rows can belong to the same object; load data_config.yaml to keep this array and grouping offsets together.",
    "patch_offsets": "Half-open row boundaries grouping the saved patches by unique object ID. Generated during preparation; do not reorder independently of the crop and ID arrays.",
    "positions_um": "Physical Z,Y,X center of every saved fine patch, in micrometers. Native integer object IDs are stored separately to preserve their precision.",
    "classifier_space": "Choose the values used to predict cell type: all embedding features, the first PCA components, or UMAP coordinates. PCA/UMAP are fitted only on training objects and then transform held-out or target objects. They are separate from the exploratory map fitted to all data. Logistic regression is linear in whichever input space you choose.",
    "classifier_dimensions": "Number of PCA components or UMAP coordinates given to the classifier. 2 tests a two-dimensional representation; larger values retain more information. PCA cannot exceed the feature count or number of training objects in a fold. This does not change the displayed plot dimensions.",
    "classifier_standardize": "After selecting full features or computing PCA/UMAP, center each selected dimension and scale it to unit standard deviation using training objects only. The same fitted scaling is applied to held-out and target objects. This is separate from scaling original features before projection and can change relative fine/coarse weights.",
    "classifier_umap_neighbors": "Neighborhood size for the classifier's own UMAP. Smaller values emphasize local structure; larger values include broader relationships. It is reduced when necessary, but every training fold must contain at least four objects. This does not change the exploratory UMAP plot.",
    "classifier_umap_min_dist": "How tightly points can pack in the classifier's UMAP, from 0 to 1. Lower values form tighter groups. Keep this fixed when comparing classifiers; tighter visual groups do not guarantee better held-out predictions.",
    "classifier_umap_epochs": "Optimization iterations for the classifier's UMAP. The default is 200. Each validation fold fits a separate training-only map, so this option takes longer than classification on full features or PCA. At least 3 iterations are required.",
    "classifier_models": "Use the same logistic or MLP estimators as Tools, or K-nearest neighbors. Scores and confusion matrices come from held-out folds.",
    "classify": "Evaluate predictive performance when enough labels exist. Turn off to overlay known labels without fitting a classifier.",
    "minimum_class_count": "Exclude classes with fewer matched labeled objects from classifier training and evaluation; their known labels remain on the plots.",
    "fold_policy": "reduce uses as many stratified folds as the smallest eligible class permits, matching Tools. strict rejects insufficient class counts. Specimen groups must also support the requested folds.",
    "class_weight": "balanced weights each class inversely to its training-fold count, matching the classification Tool default. null gives equal weight to individual objects. Applies to logistic regression.",
    "hidden_dimensions": "MLP hidden-layer widths, such as [64] for the same one-hidden-layer network used in Tools. ReLU activation and L2 penalty 0.0001 are shared.",
    "predict_unlabeled": "Fit on eligible labeled objects to predict types for other objects. Evaluated objects retain held-out predictions; fitted predictions are not validation evidence.",
    "prediction_model": "Classifier whose predicted labels are shown beside known types and used for categorical volume exports.",
    "unlabeled_opacity": "Opacity from 0 (invisible) to 1 (opaque) for objects without a known cell type. Applies to the same objects in adjacent cluster/class panels.",
    "input_config": "Optional mae_config.yaml, distributed-patch data_config.yaml or grouped N5 data YAML for point inspection. Extraction provenance supplies this automatically when available.",
    "subset": "Zero projects all embeddings. A positive count selects an ID-sorted, reproducible subset before normalization, matching Tools. Classification still uses all eligible labeled embeddings.",
    "umap_epochs": "Optimization epochs for UMAP. 50 matches Tools; null lets UMAP choose automatically. Match this, subset, minimum distance and seed when comparing plots.",
    "cluster_pca_components": "Use 0 (or null in YAML) to cluster the normalized embedding vectors. A value such as 50 first reduces those vectors to 50 PCA components, then fits K-means or the Leiden neighborhood graph. The count is capped to the available features and objects minus one. This is independent of PCA before UMAP and may change cluster assignments and silhouette scores. It does not refit the classifier or change predicted cell types.",
    "analysis_variants": "Additional named combinations of normalization, clustering and projection settings. Each uses the main result's exact plotted objects and saved feature vectors; extraction, adaptation and classification are not repeated. Each combination is saved separately, with its parameters and any error in Results and provenance.",
    "umap_pca_components": "Use 0 (or null in YAML) to send all embedding features directly to UMAP. Try 50 to first summarize them with 50 PCA components; this can reduce noise and computation, but can also discard useful variation. PCA is fitted after feature normalization, using the objects in this map (all selected datasets together for a shared map). The count is reduced automatically if there are too few objects or features, and Results shows the count and variance retained. This changes the UMAP plot only, not saved embedding vectors, cluster assignments or classifier inputs. It is separate from the 2D PCA plot. For DeepCORAL's already compact adapted vectors, leaving it off is a reasonable starting point.",
    "resolution": "Controls how finely Leiden divides the neighborhood graph built from embedding features (or the PCA components selected for clustering). Higher values generally produce more, smaller clusters; lower values produce fewer, larger clusters. Used only for Leiden, not K-means or UMAP point spacing.",
    "labels": "Saved object_labels.tsv with exact label_id values and known_label, predicted_label and/or cluster columns.",
    "layers": "Categorical volumes to write on the original segmentation grid. Each layer has its own numeric category-to-color lookup table; zero is background or unassigned.",
    "id_mapping": "Optional CSV/TSV with label_id and segmentation_id (or the chosen mapping column). Required when embedding IDs refer to different objects than the segmentation. Many-to-one mappings are rejected as ambiguous.",
    "mapping_column": "Column in the mapping table containing exact integer IDs from the source segmentation.",
    "name": "A descriptive name for this editable workflow. The Run ID separately names the output directory.",
    "run_id": "Unique name for this execution and its output directory. Use letters, digits, dots, underscores, or dashes; completed or dry-run directories cannot be reused.",
    "execution": "local starts a worker on the app host; slurm submits the reviewed job to the cluster.",
    "allow_synthetic": "Permit generated demonstration crops when no real data is configured. Leave disabled for biological workflows so missing data produces an error.",
    "model": "Choose a trained MAE or a frozen pretrained DINOv2/DINOv3 backbone. Each requires matching weights and the corresponding input configuration.",
    "repo_root": "Location of the MorphoFeatures checkout. Repository-relative data paths are resolved from this directory; it does not select a dataset or submit a job.",
    "data_root": "Base directory for dataset files referenced by the configuration.",
    "output_root": "Directory for generated files. Workflow runs use the workspace output root shown on Review & run.",
    "analysis_data": "Directory containing the bundled reference embeddings and annotation tables.",
    "mobie_data": "Directory containing the published MoBIE tables used as reference data.",
    "run_dir": "Base output directory in the model configuration. Workflow submission places training outputs in this run's own stage directory.",
    "checkpoint": "Saved model weights. Extraction reads this file; training output checkpoints are placed in the current run directory.",
    "best_checkpoint": "Destination for the model with the best validation loss. Workflow submission scopes this to the current run.",
    "resolved_config": "Saved copy of the exact model and data settings used for training, retained with the checkpoint.",
    "metrics": "Training metric log, including losses and progress. Workflow submission writes it in this run's stage directory.",
    "metrics_path": "Training metric log used by the worker and Runs page; scoped to the current run on submission.",
    "split_manifest": "Table recording which object IDs belong to training, validation, and test sets.",
    "run_metadata": "Record of dataset, model, and execution provenance saved with training outputs.",
    "embedding": "Saved feature vectors with their object IDs. These are the input to projection and classification.",
    "from_embedding": "Name of an earlier extraction or combination output to analyze. The workflow passes its full feature vectors and object IDs to this stage automatically; a saved embedding path is not needed.",
    "from_embeddings": "Names of earlier embedding outputs to compare, for example [Fine, Coarse, Combined]. Each must describe the same dataset; objects are matched by ID so every representation is evaluated on the shared objects.",
    "crops": "Object intensity crops in NumPy, HDF5 or N5, shaped (objects, Z, Y, X) or (objects, channels, Z, Y, X). Containers also need a dataset key. Load the generated mae_config.yaml to fill in paths and keys.",
    "label_ids": "One integer object ID per crop. These IDs connect embeddings to segmentation objects and annotation tables.",
    "loss_masks": "Optional foreground masks aligned with the crops, in NumPy, HDF5 or N5. Training measures reconstruction inside the mask; DINO uses it to select object pixels. Containers also need a mask dataset key.",
    "crops_key": "Dataset holding intensity crops inside the HDF5/N5 container; preprocessing writes crops. This whole-object layout is separate from grouped N5 patches and their positions index.",
    "label_ids_key": "Dataset containing one exact integer ID per crop; preprocessing writes label_ids. Use the same row order as the crops and masks.",
    "loss_masks_key": "Dataset holding aligned foreground masks inside the HDF5/N5 container; preprocessing writes masks, with zero outside and one inside each object.",
    "output_format": "npy writes three NumPy arrays. h5 writes one HDF5 file; n5 writes an N5 directory. Both containers use compressed datasets for crops, masks and object IDs. All formats generate mae_config.yaml for subsequent MAE, DINO and preview use. HDF5 requires h5py; N5 requires z5py.",
    "raw": "Raw intensity volume, not a label image. Supported containers are HDF5, N5, Zarr, and .npy. It must share a spatial grid with the segmentation.",
    "segmentation": "Integer instance-label volume aligned with the raw image. Each object has a distinct positive ID; zero is background.",
    "raw_key": "Dataset inside the raw container, for example exported_data or volumes/raw/s1. Leave blank for a .npy file.",
    "segmentation_key": "Dataset inside the segmentation container. Select instance IDs at the same resolution as the raw image. Leave blank for .npy.",
    "raw_axes": "Order of dimensions as stored, for example zyx or zyxc. The preview and ROI controls convert spatial dimensions into Z, Y, X order.",
    "segmentation_axes": "Order of stored segmentation dimensions, for example zyx or zyxc. It can differ from raw storage order if both describe the same spatial grid.",
    "raw_channel": "Zero-based intensity channel to read when the raw volume has a channel dimension.",
    "segmentation_channel": "Zero-based channel containing instance IDs when the segmentation has a channel dimension.",
    "roi": "Region to process in Z, Y, X voxel coordinates: start is included and stop is excluded. Smaller regions reduce scanning time; null selects the full volume.",
    "crop_shape": "Fixed output size in Z, Y, X voxels for each object. Choose a size that contains the objects and is divisible by the MAE patch size.",
    "spacing_zyx": "Physical distance between voxels along Z, Y, X, expressed in the selected coordinate unit. Use 1, 1, 1 with voxel units when physical spacing is unknown.",
    "origin_zyx": "Coordinate of voxel zero in Z, Y, X order, in the selected coordinate unit. Raw and segmentation must use the same origin.",
    "unit": "Unit used for voxel spacing and exported object coordinates. ROI bounds always use voxel indices.",
    "max_objects": "Maximum number of eligible objects to export. Zero includes all eligible objects. The selected ROI is still scanned to establish object bounds.",
    "block_shape": "Size of each segmentation scan tile in Z, Y, X. Smaller tiles reduce working memory but may increase I/O overhead.",
    "min_voxels": "Discard objects with fewer than this many labeled voxels inside the selected ROI.",
    "object_ids": "Optional list of specific instance IDs to export. Leave empty to consider every object in the ROI.",
    "center": "Center crops on the bounding-box midpoint (bbox) or the object's voxel centroid (centroid).",
    "oversized": "skip excludes objects that do not fit the crop; clip keeps a truncated crop. Truncation is recorded in the object manifest.",
    "boundary": "pad fills parts of a crop extending outside the ROI with background; skip excludes such crops.",
    "roi_boundary": "skip excludes objects touching the ROI edge because their full extent is unknown; allow keeps them and records truncation.",
    "background": "Intensity assigned outside each object's mask after normalization, usually zero.",
    "seed": "Seed for repeatable splits, initialization, and randomized analysis. Identical seeds do not guarantee identical results across hardware or library versions.",
    "device": "auto uses CUDA when available, otherwise CPU. cuda requires a visible GPU; request GPUs separately when submitting to Slurm.",
    "input_shape": "Spatial input dimensions in Z, Y, X. Must match prepared crops or stored N5 patches. Linked preprocessing supplies this automatically.",
    "input_channels": "Number of image channels per input crop. The grouped nucleus workflow uses one grayscale channel.",
    "patch_size": "Z, Y, X size of each token within a crop. Each input dimension must be divisible by its patch dimension. Smaller patches create more tokens and use more memory.",
    "embedding_dim": "Number of encoder features per token and output object embedding. Must be divisible by the encoder's attention-head count.",
    "encoder_depth": "Number of transformer encoder layers. More layers increase model capacity, runtime, and memory use.",
    "encoder_heads": "Number of parallel encoder attention heads. Must divide the embedding dimension exactly.",
    "decoder_dim": "Width of the reconstruction decoder. For grouped MAE it must be divisible by decoder heads; the crop MAE uses an MLP decoder without attention heads.",
    "decoder_depth": "Number of transformer reconstruction-decoder layers in grouped MAE. This affects training cost; embeddings come from the encoder.",
    "decoder_heads": "Number of attention heads in the grouped MAE decoder. Must divide the decoder dimension exactly.",
    "mask_ratio": "Fraction of tokens hidden during reconstruction training, between 0 and 1. Grouped MAE hides complete stored patches; crop MAE hides internal voxel patches.",
    "reconstruction_shape": "Z, Y, X reconstruction target per stored patch in grouped MAE. Smaller targets lower decoder cost and discard fine detail.",
    "norm_pix_loss": "Normalize each grouped patch's reconstruction target before computing loss, reducing sensitivity to intensity scale.",
    "patch_encoder": "linear projects each stored patch directly into a token; resnet3d first learns local 3D convolutional features. Changing this requires a matching new checkpoint.",
    "resnet_channels": "Channel widths for successive 3D ResNet stages. Used only with the resnet3d patch encoder.",
    "resnet_blocks": "Residual-block counts for each 3D ResNet stage. Provide one count for each channel width.",
    "epochs": "Maximum number of passes through the training set. Grouped early stopping may finish before this limit.",
    "batch_size": "Number of samples processed together. Larger batches need more memory; a grouped sample contains multiple stored patches from one object.",
    "learning_rate": "AdamW optimizer step size. Reduce it if training is unstable; compare validation loss when tuning it.",
    "weight_decay": "AdamW regularization strength. Zero disables weight decay; larger values penalize large weights more strongly.",
    "workers": "Number of background data-loading processes. Zero reads in the training process; more workers can improve throughput on fast storage.",
    "pin_memory": "Use page-locked host memory for data batches, which can accelerate transfers to CUDA devices.",
    "scheduler": "Learning-rate schedule: constant keeps the rate fixed, cosine gradually decays it, and step reduces it at fixed epoch intervals.",
    "warmup_epochs": "Initial epochs used to ramp up the learning rate. Must be shorter than the full training run.",
    "min_learning_rate": "Lowest learning rate reached by cosine decay.",
    "scheduler_step_size": "Number of epochs between learning-rate reductions when using the step schedule.",
    "scheduler_gamma": "Multiplier applied at each step-schedule reduction, for example 0.5 halves the rate.",
    "mixed_precision": "Use mixed-precision computation on CUDA to reduce memory use and often speed training. CPU training remains full precision.",
    "checkpoint_interval": "Save an intermediate checkpoint every this many epochs. Zero disables periodic checkpoints; final/best checkpoint handling remains separate.",
    "progress_interval_batches": "Write progress and throughput every this many batches. Zero keeps epoch-level metrics only.",
    "resume_from": "Grouped MAE checkpoint to resume, including optimizer and scheduler state. Leave blank to start a new model.",
    "enabled": "Stop grouped training when validation loss no longer improves, using patience and minimum improvement below.",
    "patience": "Number of validation epochs without sufficient improvement before early stopping.",
    "min_delta": "Minimum reduction in validation loss counted as an improvement for early stopping.",
    "restore_best": "Restore the model and optimizer state from the best validation epoch for the final checkpoint.",
    "train_fraction": "Fraction of object IDs assigned to training. Grouped train, validation, and test fractions must sum to one.",
    "validation_fraction": "Fraction of object IDs held out from optimization to measure validation loss.",
    "test_fraction": "Fraction of object IDs reserved for final evaluation. Keep these objects out of training and model selection.",
    "max_labels": "Limit the grouped workflow to this many object IDs for a bounded trial. null uses all eligible IDs.",
    "group_size": "Number of complete stored patches sampled together from one nucleus. More patches capture more context and increase attention memory.",
    "patches_container": "N5 container holding stored intensity patches. Pair it with the positions index from the same dataset release.",
    "source": "Input layout used by the reader. n5_masked_patches selects indexed stored patches grouped by their biological parent ID.",
    "qc_raw_container": "Optional unmasked raw N5 volume for checking stored masked patches against their source imagery.",
    "qc_raw_key": "Dataset key for the raw pyramid level used in patch quality-control views.",
    "position_radius_zyx": "Half-size of a patch in the position-index grid, in Z, Y, X. Used to recover the corresponding raw crop for quality control.",
    "position_to_raw_scale_zyx": "Scale from the patch-center coordinate grid to the selected raw pyramid, in Z, Y, X. Use audited values for your dataset.",
    "modality": "Imaging modality recorded with this dataset, such as EM. This describes provenance; it does not convert the images.",
    "biological_unit": "Biological object represented by each output ID, such as nucleus or cell. Keep this consistent with annotations and comparisons.",
    "positions_container": "N5 container mapping patch rows to parent object IDs and spatial positions.",
    "patches_key": "Dataset key containing the stored patches inside the N5 container.",
    "positions_key": "Dataset key containing patch-to-object IDs and spatial coordinates.",
    "ids_key": "Dataset key containing the unique biological parent IDs in the positions container.",
    "position_stride_zyx": "Spacing between candidate patch centers in index coordinates, Z, Y, X. Must agree with the audited patch sampling grid.",
    "resolution_zyx_um": "Physical voxel spacing of stored image patches in micrometers, Z, Y, X.",
    "position_resolution_zyx_um": "Physical spacing of the coordinate grid used by patch-center positions, in micrometers, Z, Y, X.",
    "patch_shape_zyx": "Stored patch dimensions in Z, Y, X; these must match the N5 array and model input shape.",
    "min_foreground_fraction": "Exclude patches whose foreground mask covers less than this fraction of voxels.",
    "mask_mode": "nonzero treats nonzero intensities as foreground; all treats the entire stored patch as foreground.",
    "random_flip": "Grouped augmentation is currently disabled because image patches and spatial coordinates must transform together. Keep false.",
    "random_rot90": "Grouped augmentation is currently disabled pending matching spatial-coordinate transforms. Keep false.",
    "annotations": "CSV or TSV with unique label_id values and known classes. Only labeled objects enter classification; every object remains in the projection.",
    "cell_table": "Optional cell annotation table used to connect learned features to biological labels and metadata.",
    "cell_to_nucleus": "Optional mapping between cell IDs and nucleus IDs. Use an authoritative mapping from the same segmentation release.",
    "symmetric_cells": "Optional table of corresponding bilateral cells, used for biological interpretation checks.",
    "label_column": "Column containing known biological classes. auto detects cell_type, label or cell_label. Use external annotations rather than computed clusters as classification targets.",
    "group_column": "Optional specimen/acquisition column. All objects from one group stay in the same evaluation fold to reduce related-object leakage.",
    "folds": "Requested held-out folds. The reduce policy lowers this to the smallest eligible class count, matching Tools; strict requires every class to support this count. Grouped evaluation also needs enough independent groups.",
    "knn_k": "Number of training-set neighbors used for k-nearest-neighbor classification and retrieval checks.",
    "linear_c": "Inverse regularization strength of the linear classifier. Smaller values regularize more strongly.",
    "max_iter": "Maximum optimizer iterations for the linear classifier; increase if it fails to converge.",
    "evaluation_pca": "Optional feature dimensions retained before classification. PCA is fitted separately within each training fold; null disables it.",
    "umap": "Also compute a nonlinear UMAP projection. PCA is always saved. UMAP requires the optional analysis dependency and at least four objects.",
    "neighbors": "Number of nearby objects considered by UMAP and the Leiden graph. Try 15 for local detail or 30 to include broader relationships. Must be at least 2; it is reduced automatically for small datasets. This can change Leiden clusters, but does not change K-means assignments.",
    "min_dist": "How tightly points can pack in the displayed UMAP, from 0 to 1. Try 0 or 0.1 for compact groups, or 0.5 for a more spread-out map. This changes visual spacing, not K-means or Leiden cluster assignments or predicted cell types. Tighter-looking groups do not by themselves establish biological classes.",
    "clusters": "How many groups K-means should form, for example 8 or 12. Must be at least 2 and is capped to the number of plotted objects. Used only for K-means; select Leiden to control groups with its resolution instead. These are exploratory groups, not a count of validated biological cell types.",
    "cluster_method": "K-means uses a chosen cluster count; Leiden uses a neighborhood graph and a resolution parameter.",
    "model_repository": "Local checkout of the official DINOv2 or DINOv3 repository containing hubconf.py. Code and pretrained weights are supplied separately.",
    "variant": "Backbone architecture matching the supplied weights. DINOv2 and DINOv3 weights are not interchangeable.",
    "axes": "Axes used for 2D views: 0 is Z, 1 is Y, and 2 is X. Each selected axis contributes slices through each 3D object crop.",
    "fractions": "Slice positions along each selected axis, from 0 to 1. [0.5] uses central slices; [0.25, 0.5, 0.75] uses three slices per axis.",
    "size": "Square image size passed to DINO. Must be divisible by 14 for DINOv2 or 16 for DINOv3; 224 works for both.",
    "mask": "Set pixels outside the object mask to zero before producing DINO views, reducing contributions from adjacent objects.",
    "percentiles": "Lower and upper foreground intensity percentiles mapped to 0 and 1 before DINO encoding.",
    "resize": "letterbox preserves slice aspect ratio with padding; stretch resizes each slice to a square.",
    "feature": "cls uses the global class token; patch_mean averages the image's patch tokens. Both produce one vector per view.",
    "aggregation": "Combine all view vectors for each object by their mean or elementwise maximum.",
    "cache": "Directory for reusable embedding results. Checkpoint, data, and view settings determine the cache identity.",
    "data_version": "Change this identifier after modifying source volumes in place so extraction does not reuse an older cached result.",
    "sequential_ids": "Generate row-number IDs if an ID file is absent. Prefer original segmentation IDs when joining annotations or inspecting objects.",
    "partition": "Cluster queue to request. Choose a partition that provides the CPU/GPU resources required by your workflow.",
    "time": "Maximum wall-clock time, in HH:MM:SS or D-HH:MM:SS. Slurm can stop the job when this limit is reached.",
    "memory": "Memory reservation, for example 16G or 128G. All stages share this reservation.",
    "cpus": "CPU threads available to the worker; on Slurm this requests CPUs per task.",
    "gpus": "Number of GPUs to reserve. Set at least one for CUDA training or DINO extraction on Slurm.",
    "account": "Cluster allocation or billing account. Leave blank if the cluster assigns your default account.",
    "qos": "Optional cluster quality-of-service policy controlling limits and priority.",
    "nodes": "Number of nodes requested. This workspace runs a single worker and does not distribute training across nodes.",
    "ntasks": "Number of Slurm tasks reserved. The workspace launches one worker; extra tasks do not enable distributed training.",
    "ntasks_per_node": "Tasks reserved per node. Use one for the current single-worker pipeline.",
    "gpu_directive": "Use gres for clusters expecting --gres=gpu:N; use gpus for --gpus=N.",
    "mail_types": "Events for which Slurm should email you. Select END and FAIL for completion/failure notices and supply a mail address.",
    "mail_user": "Email recipient for selected Slurm events. Both an address and at least one notification event are required.",
    "python_executable": "Python interpreter from the environment containing MorphoFeatures and its dependencies. It must be accessible on the execution host.",
    "setup": "Environment setup before the worker starts. Enter argument lists, for example [[module, load, CUDA], [source, /path/to/activate.sh]].",
    "cpu_thread_env": "Set OMP and MKL thread counts to CPUs per task in the Slurm script to avoid oversubscribing CPU cores.",
    "local_runtime_directories": "Store selected optional W&B and Matplotlib caches inside the run directory.",
    "log_job_context": "Write job ID, host, working directory, and command into the job log for troubleshooting.",
}


# These options reuse names such as `enabled`, `source` and `name` from other
# workflows. Resolve their meaning within a stage before consulting shared help.
PHYSICAL_SOURCE_HELP = {
    "source": "Choose where to read the original 3D images: local files, the published PlatyBrowser collection, or an OME-Zarr web link. Fine/coarse sampling needs both raw intensities and an instance segmentation.",
    "raw": "Path to the original 3D intensity image (.npy, HDF5, N5 or Zarr). Use the image in its original spatial layout so views can cover a whole nucleus; a stack of previously cut-out crops cannot provide missing tissue.",
    "segmentation": "Path to the instance-label image: each nucleus must have its own positive integer ID, with 0 for background. A binary foreground mask cannot distinguish nuclei. It must describe the same physical locations as the raw image.",
    "raw_key": "Name of the intensity array inside the container, for example volumes/raw/s1. Leave blank for a standalone .npy array.",
    "segmentation_key": "Name of the instance-label array inside the container, for example volumes/nuclei/s0. Leave blank for a standalone .npy array.",
    "raw_axes": "Order of dimensions in the stored intensity array, for example zyx, czyx or tzyx. c denotes channel and t denotes time. The form displays channel/time selection when those axes are present; spatial coordinates are entered in Z, Y, X order.",
    "segmentation_axes": "Order of dimensions in the stored label array, for example zyx or zyxc. c denotes channel and t denotes time. This can differ from raw storage order and voxel spacing, provided the physical metadata describes aligned images.",
    "unit": "Unit of the spacing and origin values: um means micrometers and nm means nanometers. Enter measured image metadata; physical tile widths are always specified in micrometers.",
    "roi": "Optional region to search for nuclei, written as [[z_start, y_start, x_start], [z_stop, y_stop, x_stop]] in segmentation voxels. For example [[0, 0, 0], [128, 128, 128]]. Stops are excluded. Remote sources need this region unless an object-bounds table is available; objects cut by its edge are skipped by default.",
    "object_index": "Optional TSV listing each object's bounding box. It avoids scanning the segmentation to find objects. Supply label_id and bbox_min_z/y/x and bbox_max_z/y/x columns in the selected segmentation's voxel grid; maximum indices are excluded. Bounds must enclose complete objects.",
    "id_mapping": "Optional TSV linking your embedding/annotation label_id to the segmentation_id (or nucleus_id) in the image. Each ID must map to one distinct object. Leave blank when both already use the same IDs.",
    "object_ids": "Optional list of objects to include, for example [1354, 1043, 13539]. Use the IDs in your annotation/embedding tables; supply an ID mapping if these differ from the segmentation. Leave null to consider all eligible objects.",
    "max_object_voxels": "Maximum voxels in an object's full segmentation bounding box plus a one-voxel halo, including background. Default 16,777,216 (256³). This is not the foreground count or fine-patch size. The eligibility check shows the required size per object. Raise it only with enough worker RAM; choose a coarser segmentation when appropriate. Raw texture resolution can stay fine. The worker never clips an object to fit this limit.",
    "boundary_policy": "Skip is the default because an edge-touching nucleus may be incomplete. Stop turns any such object into a run error. Use the available part explicitly samples only the portion inside the source volume and ROI, marking both fine and coarse features partial. For an ROI edge you can expand the ROI; missing tissue at a source-volume edge cannot be restored.",
    "oversized_policy": "Choose whether an object whose complete read box exceeds Maximum object-box voxels is skipped or stops the run. Both report its required voxel count. Neither crops the object or overrides the budget. Changing tile width, patch count or GPU batch size does not reduce this object-mask allocation.",
    "on_object_error": "For other per-object failures, such as missing foreground, stale bounds, invalid contrast, raw coverage or model errors reported as data errors, choose skip-and-report or stop on the first error. Boundary and size checks use their own policies. Configuration errors and out-of-memory failures still stop the run; an entirely excluded representation always fails.",
    "data_version": "A name or number for this version of the source images, for example 1. Change it after replacing image or segmentation values at the same path, so saved results from the old data are not reused.",
    "project_url": "GitHub URL of the MoBIE/PlatyBrowser project containing the image catalog. Use the project URL here; select its raw and segmentation image names below.",
    "revision": "Version of the published project to use: a commit, tag or branch such as main. A fixed commit makes it possible to return to the same image catalog in later experiments.",
    "dataset": "Dataset folder within the published project, for example 1.0.1. Raw and segmentation source names below must belong to this dataset.",
    "raw_source": "Name of the published intensity image, for example raw. Choose the image containing the texture you want DINO to describe.",
    "segmentation_source": "Name of the published instance segmentation, for example nuclei. Choose a source with a separate integer ID per nucleus or cell, rather than a tissue-region or binary mask.",
    "raw_level": "Published raw resolution to read. Larger level numbers are usually coarser and faster, but may remove fine texture. Choose a level that retains the structures you want to compare.",
    "segmentation_level": "Published segmentation resolution to read. Coarser levels reduce memory use but may lose small objects or boundary detail. Its physical coordinates must match the raw image; voxel sizes may differ in this workflow.",
    "cache_directory": "Writable folder for downloaded image chunks. Reusing this folder avoids fetching the same image regions again. It must be accessible from the machine running the workflow.",
    "url": "Paste the OME-Zarr image-group or layer URL containing the multiscale metadata, not a link to an individual chunk. Supply raw intensities for the raw layer and integer object IDs for the segmentation layer.",
    "level": "Resolution to read, numbered from 0 in the image's metadata. Click Find resolutions to see voxel sizes and array dimensions. Coarser images load faster but contain less detail; this number is not a magnification factor.",
    "channel": "Image channel to read, counting from 0. Use 0 for a single-channel image; select the intensity or instance-label channel appropriate to this layer.",
    "timepoint": "Time frame to read, counting from 0. Use 0 for a single time point. Raw and segmentation must refer to the same biological sample and time.",
}

PHYSICAL_VIEW_HELP = {
    "max_tiles": "Maximum number of small 3D regions sampled per nucleus at each fine scale. Up to 64 are spread across the object by default, beginning near its center. Increase to sample more locations at greater runtime; Process all eligible tiles overrides this cap.",
    "exhaustive": "Read every tile that passes the foreground threshold instead of stopping at the tile cap. This can be much slower for large nuclei. It still encodes the chosen 2D slices in each tile, not every voxel.",
    "axes": "Choose the directions to slice through each nucleus: 0 gives XY images, 1 gives XZ images, and 2 gives YZ images. [0, 1, 2] samples all three orientations; [0] uses XY only. More axes produce more DINO images.",
    "fine_fractions": "Slice positions through the nucleus foreground inside each fine tile. [0.5] uses its median foreground position; [0.25, 0.5, 0.75] samples three foreground positions along each selected axis. More positions increase runtime.",
    "coarse_fractions": "Slice positions from one end of the complete nucleus to the other, from 0 to 1. [0.1, 0.3, 0.5, 0.7, 0.9] gives five evenly spaced views per axis. Each slice is fitted into the DINO image while preserving physical proportions.",
    "min_foreground_fraction": "Keep a 3D fine tile only if the selected nucleus occupies at least this fraction of its full volume. Enter 0.25 for one quarter, 0.50 for half, or 0.75 for three quarters. The default 0.01 retains thin peripheral regions. Neighboring nuclei do not count. This is a volume cutoff; use the separate 2D image cutoff to control how full the actual DINO images are.",
    "min_slice_foreground_fraction": "After a 3D tile passes its cutoff, also require this fraction of each 2D fine image to contain the selected nucleus. 0.50 means at least half the image. The default 0 keeps every nonempty slice. Padding counts as background. This filter affects fine images only; it does not remove coarse whole-nucleus slices.",
    "tile_anchor": "Choose where the non-overlapping fine tile grid is placed. bbox_min starts at the lowest corner of the nucleus bounding box (the original behavior). bbox_center places a central tile at the box midpoint. centroid places a central tile at the average position of all foreground voxels, accounting for an irregular shape. Tiles extend outward to cover the entire nucleus. With a tile cap, centered modes start selection near that center, then spread samples across the object; the centroid can lie outside a hollow or concave nucleus.",
    "size": "Width and height of the final 2D image sent to DINO, in pixels. 224 works for both families; other sizes must be divisible by 14 for DINOv2 or 16 for DINOv3. This changes model input size, not the physical area sampled. Larger inputs use more memory.",
    "mask": "Replace image pixels outside the selected nucleus with black. Keep enabled for nucleus-only comparisons. Disable to test surrounding tissue as an additional experiment; neighboring cells may then influence the features.",
    "percentiles": "Contrast limits estimated once from each nucleus and shared by its fine and coarse views. [1, 99] maps the 1st percentile to black and the 99th to white, clipping more extreme values. Keep the same limits across datasets for comparison.",
    "batch_size": "Number of 2D images sent through DINO together. This is an image count, not a nucleus count. Lower it if the GPU runs out of memory; it does not change which views are selected.",
    "max_candidate_tiles": "Upper limit on the number of possible tiles considered for one nucleus before selecting the final views. Tiny tile widths can create very large candidate lists. If the grid exceeds this limit, the object is excluded or the run stops according to Other object-reading or encoding errors. Increase the tile width or deliberately raise this limit; changing the final tile cap does not reduce the candidate grid.",
    "aggregation": "How to turn a nucleus's valid view vectors into one vector. mean gives each view equal influence and is the default; max keeps the largest value in each feature dimension and emphasizes strong responses from individual views.",
}

TRANSFER_HELP = {
    "enabled": "Learn the relationship between embeddings and known cell types in one reference dataset, then predict labels for the other datasets. Requires reference annotations and at least one target dataset. Target annotations are optional and are used only to evaluate predictions.",
    "classifier": "Method that learns cell types from the labeled reference. logistic fits a linear decision rule; mlp learns nonlinear relationships using the same neural-network classifier as the Classification Tool; knn predicts from nearby labeled reference objects. These classifiers do not train or change DINO.",
    "classifier_normalization": "All transfer inputs first receive whole-vector L2 normalization. none keeps relative fine/coarse block weights intact. standardize additionally scales individual features using reference training objects only, as in the Classification Tool; this can help MLP but changes the influence of individual features and blocks. The fitted scaling is also applied to targets.",
    "hidden_dimensions": "Widths of the MLP's hidden layers, for example [64] for one layer, matching the Classification Tool, or [64, 32] for two. Larger networks need more labeled examples and can overfit. This changes the label classifier, not DINO.",
    "max_iter": "Maximum training iterations for the reference classifier. The default 2000 matches the Classification Tool. Increase if training reports that it did not converge; more iterations are not guaranteed to improve held-out performance.",
    "conditions": "Compare uncorrected predictions, classical CORAL in either direction, and a small neural adapter with or without a CORAL loss. All conditions reuse the same objects and reference folds. Classical CORAL and Deep CORAL use unlabeled target features; no condition trains on target cell-type labels. Select all five for the complete experiment.",
    "min_confidence": "Mark a prediction as unknown when its highest class score is below this value, between 0 and 1. The default is 0.6; raising it rejects more uncertain assignments. Distance can also trigger rejection. This score is a screening heuristic, not a guarantee of correctness.",
    "distance_quantile": "How far a target object may lie from reference objects of its predicted class. 0.99 uses the 99th percentile of held-out reference distances as the cutoff. Lower values reject more distant objects as unknown, even when their class score is high.",
    "label_mapping": "Optional aliases from target class names to equivalent reference names, for example {sensory_neuron: neuron}. Enter only biologically justified correspondences. This affects evaluation labels, not model fitting or predictions; unmatched new classes remain visible in evaluation.",
    "reference_model": "Optional path to the original reference/reference.json to reuse its classifier settings and labeled cohort. Keep parameters.npz, annotations.tsv, splits.tsv and original_embeddings.npz beside it. The original external embedding file is not needed. New adaptation conditions reuse the saved folds and fit their own models; target embeddings must use the same representation recipe.",
    "reference": "Settings for the labeled dataset used to train the classifier. Example: {name: Platynereis, embedding: outputs/platy.npz, annotations: labels.tsv, label_column: cell_type}. The table must link label_id to known cell types. Target labels never belong here.",
    "targets": "Datasets to predict, as a YAML list. Example: [{name: Parhyale, embedding: outputs/parhyale.npz}]. Use the same checkpoint, image preparation and feature combination as the reference. Add annotations only to evaluate predictions; they are not required for prediction.",
    "assume_same_model": "Use only for saved embeddings whose model metadata is missing and whose checkpoint, view settings and feature combination you have independently checked. Equal vector lengths alone are insufficient. Known incompatible settings still stop the run.",
}

FUSION_HELP = {
    "name": "Name for the combined vectors, such as Fine + coarse. Later analysis stages can select this output by name; it does not rename the original feature files.",
    "embeddings": "Named feature files from the same objects, for example {fine: outputs/fine.npz, coarse: outputs/coarse.npz}. Keep their metadata files beside them. Objects are matched by exact ID, not row order; only objects present in every component are combined.",
    "from_embeddings": "Select named outputs from earlier extraction or combination stages instead of recomputing them. You can include these alongside saved files. All components must describe the same dataset and object IDs.",
    "component_order": "Order of feature blocks in the combined vector, for example [fine, coarse]. Weights below follow this order. Leave [] to use the order of the entered components; that order is recorded when the workflow is saved.",
    "weights": "Relative influence of the components in the order above. [0.5, 0.5] balances two blocks; [0.75, 0.25] gives the first more influence on distances. Use one nonnegative value per block, with at least one above zero; values are normalized to sum to one. Raw combination ignores these weights.",
    "combination": "balanced puts each component on an equal vector-length scale before applying its weight and joining all feature dimensions. This is the default for fine/coarse comparisons. raw simply joins original values, ignoring weights, so large-magnitude components may dominate distances.",
    "assume_same_dataset": "Enable only if metadata is missing and you have checked that every file's object IDs refer to the same biological objects. The number 42 in two independent animals does not identify the same object. Known dataset mismatches still stop the run.",
}

DINO_EXPERIMENT_HELP = {
    "name": "A unique name for this dataset, such as Platynereis or Parhyale. It labels plots and reports and keeps identical object numbers from different datasets separate.",
    "model": "Choose the frozen pretrained DINO family to evaluate. Each family needs its own matching weights and local code checkout. Adding another backbone compares it on the same selected objects; it does not train a new model.",
    "checkpoint": "Path to a downloaded DINO weights file matching the selected family and variant. The worker must be able to read it. This workflow loads those weights as supplied; it neither downloads nor trains them.",
    "tile_sizes_um": "Side length of the small 3D regions used to measure fine texture, in micrometers. Enter [1.0] to start with 1 µm tiles, or [0.5, 1.0, 2.0] to compare three scales. Smaller tiles focus on finer structures; larger tiles include more context. Use Prepare data's preview to check the actual pixels available at each scale. Resizing cannot recover detail absent from the source image.",
    "fusion_weights": "Used only for balanced block L2; raw concatenation ignores weights. Fine texture is first, coarse second: [0.5, 0.5] gives equal influence on squared distances, while [0.75, 0.25] emphasizes fine detail. All-four combinations split each scale's share equally between CLS and foreground pooling. Within-scale pairs use equal weights. Every feature dimension is retained.",
    "extra_pairings": "Advanced named pairs, in addition to the combinations selected above. Example: [{name: Custom pair, fine: cls, coarse: foreground_mean, weights: [0.75, 0.25]}]. Without a combination field they use the selected raw/balanced methods; combination: raw or balanced overrides that choice for this pair. Leave [] when the regular combination selectors cover your experiment.",
    "max_objects": "Number of objects to try per dataset. Start with the default 100 for a manageable pilot; enter 0 for the full selection. The same reproducible selection is used across methods, without looking at cell-type labels. Excluded objects are reported, so final counts can be lower.",
    "export_tsv": "Also save vectors as tab-separated tables for spreadsheet or scripting use. NPZ files with exact object IDs and all feature values are always saved. TSV files can be much larger; large files remain available in the worker output folder.",
    "seed": "Number controlling repeatable object selection, validation splits and projections. Keep 42, or another fixed number, when comparing settings. Every representation uses the same selected objects and held-out splits within a run.",
    "folds": "Number of groups used in turn for testing the classifier: with 5 folds, roughly one fifth of labeled objects is tested at a time. Training never uses that fold's labels. Small classes can reduce the usable fold count; specimen grouping also needs enough independent specimens.",
    "classify": "Test whether embeddings predict the known cell types within each annotated dataset, using held-out objects. Requires at least two usable classes. Leave off for projection and label overlays only; prediction between datasets is configured separately below.",
    "annotations": "Optional CSV/TSV with a unique label_id per object and a cell-type column. IDs must match this dataset's embedding IDs. Leave null for unlabeled data; known labels are needed for within-dataset classification and for the reference in label transfer.",
    "group_column": "Optional column naming independent specimens or acquisitions, for example specimen. All objects from one group stay together during validation. Leave null if unavailable; results will then describe object-level validation, not performance on a new specimen.",
    "baselines": "Optional saved embeddings to compare with the new DINO outputs, for example [{name: MAE, embedding: outputs/mae.npz}]. They must describe this dataset's same object IDs; keep metadata files beside them. Only shared objects are compared. Reuse baseline names across datasets to also obtain a joint map.",
    "cache": "Writable folder for reusable DINO results. Repeating a run can reuse matching source, checkpoint and view settings instead of encoding again. Change the source data version after editing images in place.",
}


def workflow_setting_help(path, action):
    """Resolve stage-specific help, including nested dataset and transfer fields."""
    parts = [str(part) for part in path if not isinstance(part, int)]
    key = parts[-1]
    # Discard the draft/document prefix so a workflow name is not a dataset name.
    if "stages" not in parts:
        return None
    parts = parts[parts.index("stages") + 1 :]
    if action == "dino_experiment":
        if "source" in parts[:-1]:
            if key == "object_ids" and "platybrowser" in parts:
                return "Choose segmentation when your tables use the image's nucleus IDs. Choose cell when they use published cell IDs; PlatyBrowser's cell-to-nucleus table will connect them. This selects the ID convention, not a list of objects."
            if key.endswith("_spacing_zyx"):
                return "Measured voxel size along Z, Y and X, in the physical unit below. For 25 × 20 × 20 nm voxels with um selected, enter [0.025, 0.020, 0.020]. Raw and segmentation voxel sizes may differ if their physical coordinates align."
            if key.endswith("_origin_zyx"):
                return "Physical coordinate of the center of voxel [0, 0, 0], in Z, Y, X order and the unit below. Use the image metadata, especially for cropped volumes. Raw and segmentation can have different origins, but must describe the same physical frame; this workflow does not estimate registration."
            if key.endswith("_timepoint"):
                return PHYSICAL_SOURCE_HELP["timepoint"]
            return PHYSICAL_SOURCE_HELP.get(key)
        if "views" in parts:
            return PHYSICAL_VIEW_HELP.get(key)
        if "transfer" in parts:
            if key == "reference":
                return "Dataset with known cell types used to teach the classifier. Its class-label table above must be supplied. Every other dataset is a prediction target; its labels, if provided, are used only for evaluation."
            return TRANSFER_HELP.get(key)
        return DINO_EXPERIMENT_HELP.get(key)
    if action == "fuse":
        return FUSION_HELP.get(key)
    if action == "transfer":
        return TRANSFER_HELP.get(key)
    if action == "joint_analyze":
        return {
            "name": "Unique dataset name used in plots and reports. Objects keep their original IDs within this dataset, even when another dataset uses the same numbers.",
            "from_embedding": "Named output from an earlier extraction or combination stage for this dataset. All datasets must use the same checkpoint and feature settings to share one meaningful projection.",
        }.get(key)
    return None


WORKFLOW_REQUIREMENTS = {
    "distill": (
        "**Embeddings:** saved MAE/DINO/MorphoFeatures vectors, or an earlier named embedding output; no new image extraction is needed.",
        "**Known cell types:** TSV/CSV with unique label_id and a label column in exactly the same specimen and cell/nucleus ID namespace as the vectors.",
        "**Independent validation:** at least two classes with repeated examples. A specimen/group column is recommended; without it, validation is at object level. A reserved fold is never used to fit either reducer.",
        "**Optional bilateral probe:** independently defined candidate partners, plus a cell-to-nucleus mapping if your vectors use nucleus IDs. All methods must search the same cells.",
    ),
    "published": (
        "**Bundled references:** original MorphoFeatures, MorphoContextFeatures and manually defined vectors are included. No image preprocessing or model checkpoint is needed.",
        "**Historical display:** keep the saved published map, or explicitly recompute a new one. The 200-dimensional context baseline is available, but its original fitted reducer and contact graph are not bundled.",
        "**Optional evaluation / inspection:** a cell-ID annotation table for classification; matching raw/segmentation sources and ID mapping to inspect cells or export to MoBIE.",
    ),
    "bilateral": (
        "**Features:** one or more ID-preserving embeddings from the same specimen. Published Platy baselines are preloaded; add your MAE/DINO files to compare on shared cells.",
        "**Candidates:** the bundled Platy bilateral candidate pairs, or your own independently derived cell-ID partner table. Do not construct features from these evaluation pairs.",
        "**Identity:** know whether each file uses cell or nucleus IDs; nucleus IDs require an explicit cell-to-nucleus mapping. Opposite-side search additionally needs the body-side table.",
        "**Optional previous runs:** keep each bilateral.json with its saved vectors and ID bridge. New and saved methods are recomputed on shared cells with the current probe settings.",
    ),
    "export_bilateral": (
        "**Probe result:** a completed bilateral.json and its representation folders, or a linked earlier bilateral stage.",
        "**3D coordinates:** a physical cell-position table, or matching segmentation/catalogue for a MoBIE export. Use an explicit cell-to-nucleus mapping for nucleus volumes.",
        "**Bounded selection:** choose pairs to inspect. Optional HDF5 pair masks need a calibrated, bounded volume ROI; published MoBIE views stream their source data.",
    ),
    "context": (
        "**Features:** full saved MAE/DINO/other embedding vectors with exact IDs, or a named earlier extraction output.",
        "**Physical neighborhood:** a cell-contact edge table or instance segmentation (local, PlatyBrowser catalogue, or OME-Zarr URL); alternatively, physical centroids for radius/KNN neighborhoods. Bilateral candidates are not a spatial graph.",
        "**Coverage and identity:** provide embeddings for neighboring objects, choose how to handle missing/boundary cells, and supply a mapping wherever nucleus IDs must become cell IDs. Nuclei rarely touch; centroid neighborhoods are usually more useful for nucleus-only segmentations.",
    ),
    "dino_experiment": (
        "**Images:** original raw volumes and instance segmentations, with a distinct integer ID for each nucleus or cell; local, PlatyBrowser and OME-Zarr sources are supported.",
        "**Physical scale:** measured voxel sizes and aligned physical coordinates for each raw/segmentation pair. Crops alone cannot supply a whole nucleus if part of it is missing.",
        "**DINO:** downloaded weights and the matching official local code checkout, accessible to the worker. Start with the 100-object pilot and check the input preview.",
        "**Optional labels:** a CSV/TSV linking label_id to known cell types. Label transfer additionally needs a labeled reference and at least one other dataset; target labels are optional.",
    ),
    "fusion": (
        "**Features:** at least two embedding files, or named earlier outputs, describing the same objects in the same dataset. Use full vectors, not PCA/UMAP coordinates.",
        "**Object IDs:** retain the exact IDs and accompanying metadata files. Only objects present in every component are combined.",
        "**Combination:** choose a block order and relative weights; equal weights are a useful starting point. Optional class labels let you test the combined features.",
    ),
    "transfer": (
        "**Reference:** embeddings with known cell types in a label_id-linked CSV/TSV, or a saved reference.json with its parameters and annotation files.",
        "**Targets:** embeddings from other datasets made with the same checkpoint, view settings and feature combination as the reference; retain metadata files.",
        "**Optional evaluation:** target cell-type labels and specimen IDs. Target labels are not needed to predict; without them, cross-dataset accuracy cannot be measured.",
        "**Adaptation comparison:** saved full embedding vectors are sufficient; images are not reread. Neural conditions need PyTorch and train on a CPU or allocated GPU. Keep target labels reserved for evaluation and retain splits.tsv when reusing a reference.",
    ),
    "raw": (
        "**Images:** a raw intensity volume and an aligned instance segmentation, with one positive integer ID per object and 0 for background.",
        "**Image layout:** container dataset keys, stored axis order and matching spatial grids. Use the preview to check the region and crop size.",
        "**Compute:** a writable output folder and resources for preparation and MAE training. Known cell-type labels are optional for later analysis.",
    ),
    "distributed_patches": (
        "**Images and IDs:** raw data plus integer instance labels in a common physical frame; zero is background.",
        "**Physical geometry:** measured voxel spacing and origins in nm or µm and complete objects inside the ROI. Raw and segmentation may have different resolutions. Existing object bounds are optional and avoid a background scan for previews.",
        "**Sampling and storage:** choose a physical width with a resampled grid, or native raw voxel counts without spatial resampling. Set the foreground fraction and patch cap. The generated data_config.yaml groups patches by native object ID and currently feeds DINO, not crop MAE.",
        "**DINO (optional):** local weights and the matching official checkout. Preparation alone needs no model or class labels.",
    ),
    "preprocess": (
        "**Images:** a raw intensity volume and matching instance labels, one distinct positive integer per object.",
        "**Selection:** dataset keys, axis order, the region to process and a crop size that contains your objects. Check these in the preview.",
        "**Output:** a writable folder for crops, masks, IDs and the generated mae_config.yaml. No trained model or class-label table is required.",
    ),
    "platybrowser": (
        "**Published sources:** a PlatyBrowser project/dataset and the names of its raw image and instance segmentation, such as raw and nuclei.",
        "**Resolution and selection:** compatible raw/segmentation grids, a crop size and the objects to process. Published bounds help locate objects without downloading the entire volume.",
        "**Access and models:** network access from the worker and a writable cache. You can train a new MAE, or provide matching weights/configuration for existing MAE or DINO extraction.",
    ),
    "ome_zarr": (
        "**Image links:** OME-Zarr raw and instance-segmentation layer URLs. Either layer can be previewed alone; running object preparation needs both.",
        "**Layout and region:** select resolutions, channels and time points with matching spatial grids, then specify the region to process in segmentation voxels.",
        "**Access and models:** worker network access and a writable cache. Choose new MAE training or supply matching weights/configuration for an existing MAE or DINO model.",
    ),
    "crops": (
        "**Prepared data:** object crops, their exact IDs and optional aligned masks, or grouped N5 patches with the corresponding positions/ID index.",
        "**Configuration:** the generated mae_config.yaml or a grouped-patch training YAML, with model input dimensions matching the data.",
        "**Training:** a writable output folder and suitable CPU/GPU resources. No pretrained checkpoint is required to train a new MAE.",
    ),
    "dino": (
        "**Prepared data:** single-object crops (mae_config.yaml), distributed fine patches (data_config.yaml), or grouped N5 patches with their positions/ID index. Load the matching YAML to retain native IDs and masks.",
        "**DINO:** downloaded DINOv2/v3 weights and a matching official local code checkout. No MAE training is required.",
        "**Optional labels:** a CSV/TSV linking object IDs to known cell types for classification checks. The workflow can produce embeddings and plots without labels.",
    ),
    "checkpoint": (
        "**Model:** a saved MAE checkpoint and its matching model configuration, such as resolved_config.yaml.",
        "**Data:** prepared crops or grouped patches, their IDs and a data configuration compatible with that model.",
        "**Optional labels:** known cell types linked by label_id for evaluation; an inspection configuration connects points to their source images.",
    ),
    "embeddings": (
        "**Features:** saved embedding vectors with exact object IDs. Keep extraction metadata beside the file so sources and model settings remain available.",
        "**Optional labels:** a CSV/TSV with label_id and known cell types; specimen IDs support validation across independent samples.",
        "**Optional inspection:** the original raw/segmentation sources or an inspection YAML to view selected objects and export labels back to the volume.",
    ),
    "joint_embeddings": (
        "**Datasets:** at least two independent datasets encoded with the same checkpoint and feature settings, using saved vectors or earlier extraction outputs.",
        "**Identity:** a unique name for each dataset and embeddings retaining its original object IDs and metadata. The datasets need not share ID numbers.",
        "**Optional interpretation:** each dataset's own labels and inspection YAML. All selected vectors are projected together into one PCA/UMAP space; existing 2D maps are not merged.",
    ),
    "demo": (
        "**No biological inputs needed:** the workflow generates eight synthetic crops for a short CPU demonstration.",
        "**Output:** a writable workspace folder. The resulting embeddings illustrate the workflow and are not biological evidence.",
    ),
    "export_labels": (
        "**Results:** an object_labels.tsv with native label_id values and known, predicted or cluster labels.",
        "**Volume:** the original instance segmentation and its spatial metadata; supply an ID mapping if its IDs differ from the table.",
        "**Output:** a writable HDF5 or Zarr destination with room for the requested full-volume label layers.",
    ),
    "export_mobie": (
        "**Results:** saved analysis or comparison results with native object IDs and the labels/features you want to share.",
        "**Spatial sources:** original raw and instance-segmentation volumes or published PlatyBrowser sources, plus their metadata or an inspection YAML. Isolated crop stacks cannot place cells back in the animal.",
        "**Output:** a writable project folder; portable image copies need enough space for the selected region. Supply an ID mapping if table and segmentation IDs differ.",
    ),
}


def workflow_requirements(stages, *, allow_synthetic=False):
    """Describe the actual draft, including imported and edited pipelines."""
    if not stages:
        return ()
    first = stages[0]
    action = first.get("action")
    route = {
        "fuse": "fusion", "analyze": "embeddings", "compare": "embeddings",
        "joint_analyze": "joint_embeddings", "train": "crops",
    }.get(action, action)
    if action == "preprocess":
        route = first.get("source", "local")
        if route == "local":
            route = "preprocess" if len(stages) == 1 else "raw"
    elif action == "extract":
        route = "dino" if str(first.get("model") or "").startswith("dino") else "checkpoint"
    elif action == "train" and allow_synthetic:
        route = "demo"
    if action == "preprocess" and first.get("layout") == "distributed_patches":
        route = "distributed_patches"
    if first.get("published_reference"):
        route = "published"
    items = list(WORKFLOW_REQUIREMENTS.get(route, WORKFLOW_REQUIREMENTS["embeddings"]))
    for stage in stages[1:]:
        extra = {
            "distill": "**Embedding distillation:** provide known cell types in the input vectors' ID namespace and reserve a validation fold; optional specimen groups prevent groups crossing the split.",
            "context": "**Spatial context:** supply a physical adjacency graph, instance segmentation, or measured centroids, with explicit cell/nucleus identity and neighbor-coverage policy.",
            "bilateral": "**Bilateral evaluation:** supply independently defined candidate partners and compare representations of the same specimen on shared cell IDs.",
            "transfer": "**Label transfer:** a labeled reference and target embeddings with the same representation settings. Target annotations are optional evaluation data.",
            "joint_analyze": "**Shared projection:** independent datasets must use the same checkpoint and feature settings; give each dataset its own name and native IDs.",
            "export_labels": "**Label-volume export:** retain the original instance segmentation, spatial metadata and any mapping from table IDs to segmentation IDs.",
            "export_mobie": "**MoBIE export:** retain the original spatial raw/segmentation sources, their metadata and any required ID mapping; isolated crop stacks are insufficient.",
            "export_bilateral": "**Bilateral 3D export:** supply physical positions or a matching spatial segmentation/catalogue, and an explicit ID mapping when showing cell-ID results on nuclei.",
        }.get(stage.get("action"))
        if extra and extra not in items:
            items.append(extra)
    return tuple(items)


def setting_help(path, *, action=None):
    contextual = workflow_setting_help(path, action)
    if contextual:
        return contextual
    parts = [str(part) for part in path if not isinstance(part, int)]
    key = parts[-1]
    if key.endswith("_spacing_zyx"):
        return "Voxel spacing for this volume, in Z, Y, X and the chosen unit. It must match the common grid; align or resample mismatched volumes first."
    if key.endswith("_origin_zyx"):
        return "Coordinate of this volume's voxel zero, in Z, Y, X. It must match the common origin for aligned preprocessing."
    if key == "normalization":
        if "views" in parts:
            return "DINO intensity scaling: foreground_percentile uses object intensities; dtype requires integer crops; unit expects values already in [0, 1]."
        if "data" in parts:
            return "Intensity scaling before encoding or crop export. dtype uses the integer data range; foreground_percentile uses object intensities; none preserves values."
        return "Scale embedding features before analysis: standardize centers each feature, l2 normalizes each object vector, and none preserves values. Classification fits learned scaling on training folds only."
    if key == "axes" and "views" not in parts:
        return "Spatial dimension order of stored patches; the grouped N5 workflow expects zyx."
    if key == "split" and "inference" in parts:
        return "Which object IDs to encode: training, validation, test, or all. Use test for held-out model checks."
    return HELP.get(
        key,
        "Additional option preserved from the loaded configuration. Its meaning is defined by the selected data or model adapter; keep the supplied value unless you intend to override it.",
    )
