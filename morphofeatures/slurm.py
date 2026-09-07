"""SLURM script rendering, parsing, and scheduler backends without shell execution."""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, Optional, Protocol, Sequence, Tuple

import yaml

from morphofeatures.workflows import validate_command

_SLURM_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
_MEMORY = re.compile(r"^[1-9][0-9]*(?:[KMGTP]i?B?|[KMGTP])?$", re.IGNORECASE)
_TIME = re.compile(r"^(?:[0-9]+-)?[0-9]{1,2}:[0-9]{2}(?::[0-9]{2})?$")
_ENVIRONMENT_KEY = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_MAIL_USER = re.compile(r"^[A-Za-z0-9_.+@%-]+$")
_MAIL_TYPES = {
    "ALL",
    "ARRAY_TASKS",
    "BEGIN",
    "END",
    "FAIL",
    "INVALID_DEPEND",
    "NONE",
    "REQUEUE",
    "STAGE_OUT",
    "TIME_LIMIT",
    "TIME_LIMIT_50",
    "TIME_LIMIT_80",
    "TIME_LIMIT_90",
}
_GPU_DIRECTIVES = {"gpus", "gres"}
_LOCAL_RUNTIME_DIRECTORIES = {"matplotlib", "wandb"}


def _name(value: Optional[str], field_name: str) -> Optional[str]:
    if value is None or value == "":
        return None
    if not _SLURM_NAME.fullmatch(value):
        raise ValueError("{} contains unsupported characters".format(field_name))
    return value


def _mail_types(value: object) -> Tuple[str, ...]:
    if value is None or value == "":
        return ()
    raw = value.split(",") if isinstance(value, str) else value
    try:
        parsed = tuple(str(item).strip().upper() for item in raw)
    except TypeError as error:
        raise ValueError("mail_types must be a list or comma-separated string") from error
    if not parsed:
        return ()
    invalid = [item for item in parsed if item not in _MAIL_TYPES]
    if invalid:
        raise ValueError("Unsupported SLURM mail type(s): {}".format(", ".join(invalid)))
    if "NONE" in parsed and len(parsed) > 1:
        raise ValueError("SLURM mail type NONE cannot be combined with other values")
    return tuple(dict.fromkeys(parsed))


def _runtime_directories(value: object) -> Tuple[str, ...]:
    if value is None or value == "":
        return ()
    raw = (value,) if isinstance(value, str) else value
    try:
        parsed = tuple(str(item).strip().lower() for item in raw)
    except TypeError as error:
        raise ValueError("local_runtime_directories must be a list") from error
    invalid = [item for item in parsed if item not in _LOCAL_RUNTIME_DIRECTORIES]
    if invalid:
        raise ValueError(
            "Unsupported local runtime directory type(s): {}".format(", ".join(invalid))
        )
    return tuple(dict.fromkeys(parsed))


def _boolean(value: object, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError("{} must be true or false".format(field_name))


@dataclass(frozen=True)
class ClusterProfile:
    name: str
    partition: str
    account: Optional[str] = None
    qos: Optional[str] = None
    time: str = "01:00:00"
    memory: str = "8G"
    cpus: int = 4
    gpus: int = 0
    setup: Tuple[Tuple[str, ...], ...] = ()
    nodes: int = 1
    ntasks: int = 1
    ntasks_per_node: int = 1
    gpu_directive: str = "gpus"
    mail_types: Tuple[str, ...] = ()
    mail_user: Optional[str] = None
    python_executable: Optional[str] = None
    cpu_thread_env: bool = False
    local_runtime_directories: Tuple[str, ...] = ()
    log_job_context: bool = False

    def __post_init__(self) -> None:
        _name(self.name, "profile name")
        _name(self.partition, "partition")
        _name(self.account, "account")
        _name(self.qos, "QoS")
        if not _TIME.fullmatch(self.time):
            raise ValueError("time must use HH:MM, HH:MM:SS, or D-HH:MM:SS")
        if not _MEMORY.fullmatch(self.memory):
            raise ValueError("memory must be a positive SLURM quantity such as 16G")
        if int(self.cpus) < 1 or int(self.gpus) < 0:
            raise ValueError("cpus must be positive and gpus cannot be negative")
        if int(self.nodes) < 1 or int(self.ntasks) < 1 or int(self.ntasks_per_node) < 1:
            raise ValueError("nodes, ntasks, and ntasks_per_node must be positive")
        normalized_gpu_directive = str(self.gpu_directive).lower()
        if normalized_gpu_directive not in _GPU_DIRECTIVES:
            raise ValueError("gpu_directive must be one of: gpus, gres")
        object.__setattr__(self, "gpu_directive", normalized_gpu_directive)
        normalized_mail_types = _mail_types(self.mail_types)
        object.__setattr__(self, "mail_types", normalized_mail_types)
        if self.mail_user and not _MAIL_USER.fullmatch(self.mail_user):
            raise ValueError("mail_user contains unsupported characters")
        if normalized_mail_types and normalized_mail_types != ("NONE",) and not self.mail_user:
            raise ValueError("mail_user is required when mail notifications are enabled")
        if self.mail_user and (not normalized_mail_types or normalized_mail_types == ("NONE",)):
            raise ValueError("mail_types must enable a notification when mail_user is set")
        if self.python_executable:
            if any(char in self.python_executable for char in "\x00\r\n"):
                raise ValueError("python_executable cannot contain control characters")
            executable_name = Path(self.python_executable).name
            if executable_name not in {"python", "python3"} and not executable_name.startswith(
                "python3."
            ):
                raise ValueError("python_executable must name a Python interpreter")
        object.__setattr__(
            self,
            "local_runtime_directories",
            _runtime_directories(self.local_runtime_directories),
        )
        object.__setattr__(self, "cpu_thread_env", _boolean(self.cpu_thread_env, "cpu_thread_env"))
        object.__setattr__(
            self, "log_job_context", _boolean(self.log_job_context, "log_job_context")
        )
        for command in self.setup:
            _validate_setup(command)

    @classmethod
    def from_mapping(cls, name: str, values: Mapping[str, object]) -> "ClusterProfile":
        setup = tuple(tuple(str(part) for part in command) for command in values.get("setup", ()))
        return cls(
            name=name,
            partition=str(values.get("partition", "compute")),
            account=str(values["account"]) if values.get("account") else None,
            qos=str(values["qos"]) if values.get("qos") else None,
            time=str(values.get("time", "01:00:00")),
            memory=str(values.get("memory", "8G")),
            cpus=int(values.get("cpus", 4)),
            gpus=int(values.get("gpus", 0)),
            setup=setup,
            nodes=int(values.get("nodes", 1)),
            ntasks=int(values.get("ntasks", 1)),
            ntasks_per_node=int(values.get("ntasks_per_node", 1)),
            gpu_directive=str(values.get("gpu_directive", "gpus")),
            mail_types=_mail_types(values.get("mail_types", ())),
            mail_user=str(values["mail_user"]) if values.get("mail_user") else None,
            python_executable=(
                str(values["python_executable"]) if values.get("python_executable") else None
            ),
            cpu_thread_env=_boolean(values.get("cpu_thread_env", False), "cpu_thread_env"),
            local_runtime_directories=_runtime_directories(
                values.get("local_runtime_directories", ())
            ),
            log_job_context=_boolean(values.get("log_job_context", False), "log_job_context"),
        )


def load_cluster_profiles(path: Path) -> Dict[str, ClusterProfile]:
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    values = raw.get("profiles", raw)
    if not isinstance(values, dict) or not values:
        raise ValueError("Cluster profile YAML must contain a non-empty profiles mapping")
    return {
        name: ClusterProfile.from_mapping(name, config or {})
        for name, config in values.items()
    }


def _validate_setup(command: Sequence[str]) -> None:
    if not command or any(not str(value) or any(char in str(value) for char in "\x00\r\n") for value in command):
        raise ValueError("Profile setup entries must be non-empty argument lists")
    if command[0] == "module" and len(command) >= 2 and command[1] in {"load", "purge", "use"}:
        return
    if command[0] == "source" and len(command) == 2:
        return
    raise ValueError("Profile setup supports only module load/purge/use or source <path>")


def _setup_line(command: Sequence[str]) -> str:
    _validate_setup(command)
    return " ".join(shlex.quote(str(value)) for value in command)


def apply_profile_command(command: Sequence[str], profile: ClusterProfile) -> list[str]:
    """Apply a profile's explicit interpreter without broadening the command allow-list."""
    validate_command(command)
    resolved = list(command)
    if profile.python_executable:
        resolved[0] = profile.python_executable
    validate_command(resolved)
    return resolved


def render_slurm_script(
    command: Sequence[str],
    profile: ClusterProfile,
    *,
    job_name: str,
    working_directory: Path,
    stdout_path: Path,
    stderr_path: Path,
    environment: Optional[Mapping[str, str]] = None,
) -> str:
    command = apply_profile_command(command, profile)
    _name(job_name, "job name")
    lines = [
        "#!/bin/bash",
        "#SBATCH --job-name={}".format(job_name),
    ]
    if profile.account:
        lines.append("#SBATCH --account={}".format(profile.account))
    lines.extend(
        (
            "#SBATCH --partition={}".format(profile.partition),
            "#SBATCH --nodes={}".format(profile.nodes),
            "#SBATCH --ntasks={}".format(profile.ntasks),
            "#SBATCH --ntasks-per-node={}".format(profile.ntasks_per_node),
            "#SBATCH --cpus-per-task={}".format(profile.cpus),
            "#SBATCH --mem={}".format(profile.memory),
            "#SBATCH --time={}".format(profile.time),
        )
    )
    if profile.qos:
        lines.append("#SBATCH --qos={}".format(profile.qos))
    if profile.gpus:
        if profile.gpu_directive == "gres":
            lines.append("#SBATCH --gres=gpu:{}".format(profile.gpus))
        else:
            lines.append("#SBATCH --gpus={}".format(profile.gpus))
    lines.extend(
        (
            "#SBATCH --output={}".format(shlex.quote(str(Path(stdout_path)))),
            "#SBATCH --error={}".format(shlex.quote(str(Path(stderr_path)))),
        )
    )
    if profile.mail_types:
        lines.append("#SBATCH --mail-type={}".format(",".join(profile.mail_types)))
    if profile.mail_user:
        lines.append("#SBATCH --mail-user={}".format(profile.mail_user))
    lines.extend(("", "set -euo pipefail", "cd -- {}".format(shlex.quote(str(working_directory)))))
    if profile.cpu_thread_env:
        lines.extend(
            (
                'export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"',
                'export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"',
            )
        )
    if "wandb" in profile.local_runtime_directories:
        lines.extend(
            (
                'export WANDB_DIR="${WANDB_DIR:-${PWD}/wandb}"',
                'mkdir -p -- "${WANDB_DIR}"',
            )
        )
    if "matplotlib" in profile.local_runtime_directories:
        lines.extend(
            (
                'export MPLCONFIGDIR="${MPLCONFIGDIR:-${PWD}/.matplotlib-cache}"',
                'mkdir -p -- "${MPLCONFIGDIR}"',
            )
        )
    for setup_command in profile.setup:
        lines.append(_setup_line(setup_command))
    for key, value in (environment or {}).items():
        if not _ENVIRONMENT_KEY.fullmatch(key):
            raise ValueError("Invalid environment variable name: {}".format(key))
        if "\x00" in value or "\n" in value or "\r" in value:
            raise ValueError("Environment values cannot contain control characters")
        lines.append("export {}={}".format(key, shlex.quote(value)))
    if profile.log_job_context:
        lines.extend(
            (
                'echo "Job ID: ${SLURM_JOB_ID:-local}"',
                'echo "Host: $(hostname)"',
                'echo "Working directory: ${PWD}"',
                "printf 'Command:'",
                "printf ' %q' {}".format(shlex.join(list(command))),
                "printf '\\n'",
            )
        )
    lines.extend((shlex.join(list(command)), ""))
    return "\n".join(lines)


def parse_sbatch_job_id(output: str) -> str:
    value = output.strip().splitlines()
    if len(value) != 1:
        raise ValueError("Unexpected sbatch output: {!r}".format(output))
    job_id = value[0].split(";", 1)[0]
    if not job_id.isdigit():
        raise ValueError("Could not parse a numeric SLURM job ID from {!r}".format(output))
    return job_id


def normalize_slurm_state(raw_state: str) -> str:
    state = raw_state.strip().upper().split("+", 1)[0].split(" ", 1)[0]
    if state in {"PENDING", "CONFIGURING", "REQUEUED", "REQUEUE_FED", "RESV_DEL_HOLD"}:
        return "queued"
    if state in {"RUNNING", "COMPLETING", "SUSPENDED", "STAGE_OUT"}:
        return "running"
    if state == "COMPLETED":
        return "completed"
    if state.startswith("CANCELLED"):
        return "cancelled"
    if state in {
        "FAILED",
        "TIMEOUT",
        "OUT_OF_MEMORY",
        "NODE_FAIL",
        "BOOT_FAIL",
        "DEADLINE",
        "PREEMPTED",
        "REVOKED",
    }:
        return "failed"
    return "unknown"


@dataclass(frozen=True)
class SchedulerStatus:
    job_id: str
    raw_state: str
    state: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    exit_status: Optional[str] = None


def parse_squeue_output(output: str) -> Dict[str, SchedulerStatus]:
    statuses = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = line.strip().split("|")
        if len(fields) < 2 or not fields[0].isdigit():
            continue
        state = normalize_slurm_state(fields[1])
        started = (
            fields[2]
            if state == "running" and len(fields) > 2 and fields[2] not in {"", "N/A", "Unknown"}
            else None
        )
        statuses[fields[0]] = SchedulerStatus(
            fields[0], fields[1], state, started_at=started
        )
    return statuses


def parse_sacct_output(output: str) -> Dict[str, SchedulerStatus]:
    statuses = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = line.strip().split("|")
        if len(fields) < 2:
            continue
        job_id = fields[0]
        if not job_id.isdigit():
            continue
        started = fields[2] if len(fields) > 2 and fields[2] not in {"", "Unknown"} else None
        completed = fields[3] if len(fields) > 3 and fields[3] not in {"", "Unknown"} else None
        exit_status = fields[4] if len(fields) > 4 and fields[4] else None
        statuses[job_id] = SchedulerStatus(
            job_id,
            fields[1],
            normalize_slurm_state(fields[1]),
            started_at=started,
            completed_at=completed,
            exit_status=exit_status,
        )
    return statuses


@dataclass(frozen=True)
class SchedulerQuery:
    statuses: Mapping[str, SchedulerStatus]
    warnings: Tuple[str, ...] = ()


class SchedulerBackend(Protocol):
    def submit(self, script: Path, dependency_job_id: Optional[str] = None) -> Optional[str]: ...

    def query(self, job_ids: Iterable[str]) -> SchedulerQuery: ...

    def cancel(self, job_id: str) -> None: ...


Runner = Callable[[Sequence[str]], subprocess.CompletedProcess]


def _default_runner(arguments: Sequence[str]) -> subprocess.CompletedProcess:
    return subprocess.run(list(arguments), check=False, capture_output=True, text=True)


class SlurmScheduler:
    def __init__(self, runner: Runner = _default_runner):
        self.runner = runner

    @staticmethod
    def available() -> bool:
        return shutil.which("sbatch") is not None

    @staticmethod
    def accounting_available() -> bool:
        return shutil.which("squeue") is not None or shutil.which("sacct") is not None

    @staticmethod
    def submission_command(script: Path, dependency_job_id: Optional[str] = None) -> list[str]:
        command = ["sbatch", "--parsable"]
        if dependency_job_id:
            if not dependency_job_id.isdigit():
                raise ValueError("Dependency job ID must be numeric")
            command.append("--dependency=afterok:{}".format(dependency_job_id))
        command.append(str(Path(script)))
        return command

    def submit(self, script: Path, dependency_job_id: Optional[str] = None) -> str:
        result = self.runner(self.submission_command(script, dependency_job_id))
        if result.returncode != 0:
            raise RuntimeError("sbatch failed: {}".format((result.stderr or result.stdout).strip()))
        return parse_sbatch_job_id(result.stdout)

    def query(self, job_ids: Iterable[str]) -> SchedulerQuery:
        requested = tuple(dict.fromkeys(str(value) for value in job_ids if str(value).isdigit()))
        if not requested:
            return SchedulerQuery({})
        warnings = []
        statuses: Dict[str, SchedulerStatus] = {}
        joined = ",".join(requested)
        try:
            result = self.runner(("squeue", "--noheader", "--jobs", joined, "--format=%i|%T|%S"))
            if result.returncode == 0:
                statuses.update(parse_squeue_output(result.stdout))
            else:
                warnings.append("squeue failed: {}".format((result.stderr or result.stdout).strip()))
        except FileNotFoundError:
            warnings.append("squeue is unavailable")
        missing = [job_id for job_id in requested if job_id not in statuses]
        if missing:
            try:
                result = self.runner(
                    (
                        "sacct",
                        "--noheader",
                        "--parsable2",
                        "--jobs",
                        ",".join(missing),
                        "--format=JobIDRaw,State,Start,End,ExitCode",
                    )
                )
                if result.returncode == 0:
                    statuses.update(parse_sacct_output(result.stdout))
                else:
                    warnings.append("sacct failed: {}".format((result.stderr or result.stdout).strip()))
            except FileNotFoundError:
                warnings.append("sacct is unavailable")
        return SchedulerQuery(statuses, tuple(warnings))

    def cancel(self, job_id: str) -> None:
        if not job_id.isdigit():
            raise ValueError("Job ID must be numeric")
        result = self.runner(("scancel", job_id))
        if result.returncode != 0:
            raise RuntimeError("scancel failed: {}".format((result.stderr or result.stdout).strip()))


class DryRunScheduler:
    """Backend that persists a preview without invoking scheduler commands."""

    def submit(self, script: Path, dependency_job_id: Optional[str] = None) -> None:
        return None

    def query(self, job_ids: Iterable[str]) -> SchedulerQuery:
        return SchedulerQuery({})

    def cancel(self, job_id: str) -> None:
        raise RuntimeError("Dry-run jobs were not submitted and cannot be cancelled")


@dataclass
class FakeScheduler:
    """Deterministic scheduler backend for tests and interface demonstrations."""

    next_job_id: int = 1000
    states: Dict[str, SchedulerStatus] = field(default_factory=dict)
    submissions: list[Tuple[Path, Optional[str]]] = field(default_factory=list)
    fail_submission: bool = False

    def submit(self, script: Path, dependency_job_id: Optional[str] = None) -> str:
        if self.fail_submission:
            raise RuntimeError("fake submission failure")
        job_id = str(self.next_job_id)
        self.next_job_id += 1
        self.submissions.append((Path(script), dependency_job_id))
        self.states[job_id] = SchedulerStatus(job_id, "PENDING", "queued")
        return job_id

    def set_state(self, job_id: str, raw_state: str, exit_status: Optional[str] = None) -> None:
        self.states[job_id] = SchedulerStatus(
            job_id, raw_state, normalize_slurm_state(raw_state), exit_status=exit_status
        )

    def query(self, job_ids: Iterable[str]) -> SchedulerQuery:
        return SchedulerQuery(
            {job_id: self.states[job_id] for job_id in job_ids if job_id in self.states}
        )

    def cancel(self, job_id: str) -> None:
        if job_id not in self.states:
            raise RuntimeError("unknown fake job")
        self.set_state(job_id, "CANCELLED", "0:0")
