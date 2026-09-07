import subprocess

import pytest

from morphofeatures.slurm import (
    ClusterProfile,
    SlurmScheduler,
    load_cluster_profiles,
    normalize_slurm_state,
    parse_sacct_output,
    parse_sbatch_job_id,
    parse_squeue_output,
    render_slurm_script,
)
from morphofeatures.workflows import (
    WorkflowRequest,
    build_workflow_command,
    format_command,
    validate_command,
    validate_workflow_request,
)


@pytest.fixture
def workflow_inputs(tmp_path):
    config = tmp_path / "config with spaces.yaml"
    config.write_text("seed: 7\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    texture = tmp_path / "texture"
    texture.mkdir()
    for name in ("train_config.yml", "data_config.yml", "test_config.yml", "test_config_patches.yml"):
        content = "{}\n" if name == "train_config.yml" else "data_config:\n  data_root: ..\n"
        (texture / name).write_text(content, encoding="utf-8")
    return config, checkpoint, texture


@pytest.mark.parametrize(
    ("workflow", "is_texture"),
    (
        ("shape_train", False),
        ("shape_encode", False),
        ("texture_train", True),
        ("texture_encode", True),
        ("mae_train", False),
        ("mae_encode", False),
    ),
)
def test_commands_cover_every_supported_workflow(workflow_inputs, tmp_path, workflow, is_texture):
    config, checkpoint, texture = workflow_inputs
    request = WorkflowRequest(
        workflow,
        texture if is_texture else config,
        input_path=tmp_path if workflow.startswith("shape_") else None,
        device="cpu",
        checkpoint=(
            checkpoint
            if workflow.endswith("encode")
            else tmp_path / "new-checkpoint.pt" if workflow == "mae_train" else None
        ),
        output=tmp_path / "embedding.npy" if workflow.endswith("encode") else None,
        save_patches=False,
    )
    report = validate_workflow_request(request)
    assert report.valid, report.errors
    command = build_workflow_command(request)
    assert command[:3] == ["python", "-m", "morphofeatures"]
    assert "morphofeatures" in format_command(command)


def test_texture_resume_uses_explicit_checkpoint(workflow_inputs):
    _, checkpoint, texture = workflow_inputs
    request = WorkflowRequest(
        "texture_train", texture, device="cpu", checkpoint=checkpoint, resume=True
    )
    assert validate_workflow_request(request).valid
    command = build_workflow_command(request)
    assert command[-3:] == ["--from-checkpoint", "--checkpoint", str(checkpoint)]


def test_safe_quoting_and_invalid_command_rejection(workflow_inputs, tmp_path):
    config, _, _ = workflow_inputs
    command = build_workflow_command(WorkflowRequest("shape_train", config))
    profile = ClusterProfile("test", "compute", time="00:10:00", memory="4G", cpus=2)
    script = render_slurm_script(
        command,
        profile,
        job_name="mf-test",
        working_directory=tmp_path / "work dir",
        stdout_path=tmp_path / "stdout log.txt",
        stderr_path=tmp_path / "stderr log.txt",
    )
    assert "'{}'".format(config) in script
    assert "cd -- '{}'".format(tmp_path / "work dir") in script
    with pytest.raises(ValueError):
        validate_command(["bash", "-c", "echo unsafe"])
    with pytest.raises(ValueError):
        validate_command(["python", "-m", "morphofeatures", "mae-train", "bad\nargument"])
    with pytest.raises(ValueError):
        ClusterProfile("bad profile", "compute")
    with pytest.raises(ValueError):
        ClusterProfile("bad", "compute", setup=(("bash", "-c", "anything"),))


def test_cluster_example_style_directives_and_runtime_setup(workflow_inputs, tmp_path):
    config, _, _ = workflow_inputs
    interpreter = "/path/to/MorphoFeats_dev/bin/python"
    profile = ClusterProfile(
        "gpu-cluster",
        "gpu-partition",
        account="project-account",
        time="24:00:00",
        memory="50G",
        cpus=16,
        gpus=1,
        setup=(("module", "load", "CUDA/12.8"),),
        nodes=1,
        ntasks=1,
        ntasks_per_node=1,
        gpu_directive="gres",
        mail_types=("END", "FAIL"),
        mail_user="researcher@example.org",
        python_executable=interpreter,
        cpu_thread_env=True,
        local_runtime_directories=("wandb", "matplotlib"),
        log_job_context=True,
    )
    command = build_workflow_command(WorkflowRequest("shape_train", config))
    script = render_slurm_script(
        command,
        profile,
        job_name="mf-test",
        working_directory=tmp_path,
        stdout_path=tmp_path / "stdout.log",
        stderr_path=tmp_path / "stderr.log",
    )

    expected_lines = {
        "#SBATCH --account=project-account",
        "#SBATCH --partition=gpu-partition",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        "#SBATCH --ntasks-per-node=1",
        "#SBATCH --cpus-per-task=16",
        "#SBATCH --mem=50G",
        "#SBATCH --time=24:00:00",
        "#SBATCH --gres=gpu:1",
        "#SBATCH --mail-type=END,FAIL",
        "#SBATCH --mail-user=researcher@example.org",
        'export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"',
        'export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"',
        'export WANDB_DIR="${WANDB_DIR:-${PWD}/wandb}"',
        'export MPLCONFIGDIR="${MPLCONFIGDIR:-${PWD}/.matplotlib-cache}"',
        "module load CUDA/12.8",
        'echo "Job ID: ${SLURM_JOB_ID:-local}"',
        'echo "Host: $(hostname)"',
    }
    assert expected_lines.issubset(set(script.splitlines()))
    assert script.splitlines()[-1].startswith(interpreter + " -m morphofeatures shape-train")
    syntax_check = subprocess.run(
        ("bash", "-n"), input=script, capture_output=True, text=True, check=False
    )
    assert syntax_check.returncode == 0, syntax_check.stderr


def test_cluster_profile_mapping_and_invalid_advanced_fields(tmp_path):
    legacy_profile = ClusterProfile("legacy", "gpu", gpus=1)
    assert legacy_profile.gpu_directive == "gpus"
    assert not legacy_profile.cpu_thread_env
    assert not legacy_profile.log_job_context

    profile_path = tmp_path / "profiles.yaml"
    profile_path.write_text(
        """
profiles:
  gpu:
    partition: gpu
    gpus: 1
    gpu_directive: gpus
    mail_types: END,FAIL
    mail_user: researcher@example.org
    python_executable: /opt/morphofeatures/bin/python3.11
    local_runtime_directories: [matplotlib]
""",
        encoding="utf-8",
    )
    profile = load_cluster_profiles(profile_path)["gpu"]
    assert profile.mail_types == ("END", "FAIL")
    assert profile.gpu_directive == "gpus"
    assert profile.local_runtime_directories == ("matplotlib",)

    with pytest.raises(ValueError, match="gpu_directive"):
        ClusterProfile("bad", "gpu", gpu_directive="arbitrary")
    with pytest.raises(ValueError, match="mail_user"):
        ClusterProfile("bad", "gpu", mail_types=("FAIL",))
    with pytest.raises(ValueError, match="mail_user"):
        ClusterProfile("bad", "gpu", mail_user="bad address@example.org")
    with pytest.raises(ValueError, match="Python interpreter"):
        ClusterProfile("bad", "gpu", python_executable="/bin/bash")
    with pytest.raises(ValueError, match="runtime directory"):
        ClusterProfile("bad", "gpu", local_runtime_directories=("arbitrary",))
    with pytest.raises(ValueError, match="true or false"):
        ClusterProfile.from_mapping("bad", {"partition": "gpu", "cpu_thread_env": "yes"})


def test_sbatch_and_scheduler_output_parsing():
    assert parse_sbatch_job_id("12345\n") == "12345"
    assert parse_sbatch_job_id("12345;cluster-a\n") == "12345"
    with pytest.raises(ValueError):
        parse_sbatch_job_id("Submitted batch job 12345")
    queue = parse_squeue_output("123|RUNNING|2026-01-02T03:04:05\n124|PENDING|N/A\n")
    assert queue["123"].state == "running"
    assert queue["124"].state == "queued"
    accounting = parse_sacct_output(
        "123|COMPLETED|2026-01-02T03:04:05|2026-01-02T03:05:05|0:0\n"
        "123.batch|COMPLETED|x|y|0:0\n125|OUT_OF_MEMORY|x|y|1:0\n"
    )
    assert accounting["123"].state == "completed"
    assert accounting["125"].state == "failed"
    assert normalize_slurm_state("CANCELLED by 42") == "cancelled"
    assert normalize_slurm_state("something-new") == "unknown"


def test_slurm_backend_uses_argument_lists_and_afterok(tmp_path):
    calls = []

    def runner(arguments):
        calls.append(list(arguments))
        return subprocess.CompletedProcess(arguments, 0, stdout="9821;cluster\n", stderr="")

    scheduler = SlurmScheduler(runner=runner)
    job_id = scheduler.submit(tmp_path / "job.slurm", dependency_job_id="123")
    assert job_id == "9821"
    assert calls == [[
        "sbatch",
        "--parsable",
        "--dependency=afterok:123",
        str(tmp_path / "job.slurm"),
    ]]
