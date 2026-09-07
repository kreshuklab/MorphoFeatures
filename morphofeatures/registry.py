"""Concurrency-safe persistent experiment and job registry backed by SQLite."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
import uuid

from morphofeatures.metrics import utc_now


JSON_FIELDS = {"command", "submission_command", "artifacts", "provenance"}
UPDATABLE_FIELDS = {
    "slurm_job_id",
    "submitted_at",
    "started_at",
    "completed_at",
    "raw_scheduler_state",
    "application_state",
    "exit_status",
    "artifacts",
    "provenance",
    "error_message",
}


class DuplicateSubmissionError(RuntimeError):
    pass


@dataclass(frozen=True)
class JobRecord:
    run_id: str
    workflow: str
    stage: str
    command: Tuple[str, ...]
    submission_key: str
    working_directory: str
    config_snapshot: str
    stdout_path: str
    stderr_path: str
    metrics_path: str
    checkpoint_path: Optional[str] = None
    embedding_path: Optional[str] = None
    parent_job_id: Optional[str] = None
    dependency_job_id: Optional[str] = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    slurm_job_id: Optional[str] = None
    submission_command: Tuple[str, ...] = ()
    created_at: str = field(default_factory=utc_now)
    submitted_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    raw_scheduler_state: Optional[str] = None
    application_state: str = "created"
    exit_status: Optional[str] = None
    artifacts: Tuple[str, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


class JobRegistry:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @classmethod
    def under_output_root(cls, output_root: Path) -> "JobRegistry":
        return cls(Path(output_root) / ".morphofeatures" / "registry.sqlite3")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 10000")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    workflow TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    slurm_job_id TEXT,
                    command TEXT NOT NULL,
                    submission_command TEXT NOT NULL,
                    submission_key TEXT NOT NULL UNIQUE,
                    config_snapshot TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    submitted_at TEXT,
                    started_at TEXT,
                    completed_at TEXT,
                    raw_scheduler_state TEXT,
                    application_state TEXT NOT NULL,
                    exit_status TEXT,
                    working_directory TEXT NOT NULL,
                    checkpoint_path TEXT,
                    embedding_path TEXT,
                    stdout_path TEXT NOT NULL,
                    stderr_path TEXT NOT NULL,
                    metrics_path TEXT NOT NULL,
                    parent_job_id TEXT REFERENCES jobs(id),
                    dependency_job_id TEXT,
                    artifacts TEXT NOT NULL,
                    provenance TEXT NOT NULL,
                    error_message TEXT
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS jobs_run_id ON jobs(run_id)")
            connection.execute("CREATE INDEX IF NOT EXISTS jobs_state ON jobs(application_state)")
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS jobs_run_workflow ON jobs(run_id, workflow)"
            )

    @staticmethod
    def _values(record: JobRecord) -> Dict[str, Any]:
        values = dict(record.__dict__)
        for name in JSON_FIELDS:
            values[name] = json.dumps(values[name], sort_keys=True)
        return values

    def create(self, record: JobRecord) -> JobRecord:
        values = self._values(record)
        columns = tuple(values)
        sql = "INSERT INTO jobs ({}) VALUES ({})".format(
            ", ".join(columns), ", ".join("?" for _ in columns)
        )
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(sql, tuple(values[column] for column in columns))
        except sqlite3.IntegrityError as error:
            if "submission_key" in str(error) or "jobs.run_id, jobs.workflow" in str(error):
                raise DuplicateSubmissionError(
                    "This workflow already has a saved or submitted job for the run; use a new run ID"
                ) from error
            raise
        return record

    def update(self, job_id: str, **changes: Any) -> JobRecord:
        unsupported = set(changes).difference(UPDATABLE_FIELDS)
        if unsupported:
            raise ValueError("Unsupported registry fields: {}".format(", ".join(sorted(unsupported))))
        if not changes:
            return self.get(job_id)
        encoded = {
            key: json.dumps(value, sort_keys=True) if key in JSON_FIELDS else value
            for key, value in changes.items()
        }
        assignments = ", ".join("{} = ?".format(key) for key in encoded)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                "UPDATE jobs SET {} WHERE id = ?".format(assignments),
                tuple(encoded.values()) + (job_id,),
            )
            if cursor.rowcount != 1:
                raise KeyError("Unknown job record: {}".format(job_id))
        return self.get(job_id)

    @staticmethod
    def _record(row: sqlite3.Row) -> JobRecord:
        values = dict(row)
        values["command"] = tuple(json.loads(values["command"]))
        values["submission_command"] = tuple(json.loads(values["submission_command"]))
        values["artifacts"] = tuple(json.loads(values["artifacts"]))
        values["provenance"] = json.loads(values["provenance"])
        return JobRecord(**values)

    def get(self, job_id: str) -> JobRecord:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError("Unknown job record: {}".format(job_id))
        return self._record(row)

    def find_by_submission_key(self, submission_key: str) -> Optional[JobRecord]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE submission_key = ?", (submission_key,)
            ).fetchone()
        return self._record(row) if row is not None else None

    def list(
        self,
        *,
        run_id: Optional[str] = None,
        workflow: Optional[str] = None,
        states: Iterable[str] = (),
        limit: int = 500,
    ) -> List[JobRecord]:
        clauses = []
        parameters: List[Any] = []
        if run_id:
            clauses.append("run_id = ?")
            parameters.append(run_id)
        if workflow:
            clauses.append("workflow = ?")
            parameters.append(workflow)
        normalized_states = tuple(states)
        if normalized_states:
            clauses.append("application_state IN ({})".format(",".join("?" for _ in normalized_states)))
            parameters.extend(normalized_states)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        parameters.append(max(1, int(limit)))
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM jobs{} ORDER BY created_at DESC LIMIT ?".format(where), parameters
            ).fetchall()
        return [self._record(row) for row in rows]
