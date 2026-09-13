"""Core data structures for the stylised cluster simulation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


def _require_finite(name: str, value: float, *, minimum: float) -> None:
    if not isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")


@dataclass(frozen=True, slots=True)
class Job:
    """A single-node batch job with an estimated (not measured) runtime."""

    job_id: str
    cpus: int
    gpus: int
    memory_gb: float
    submit_time: float
    estimated_runtime: float
    deadline: float | None = None
    priority: int = 0

    def __post_init__(self) -> None:
        if not self.job_id:
            raise ValueError("job_id must not be empty")
        if self.cpus <= 0:
            raise ValueError("cpus must be positive")
        if self.gpus < 0:
            raise ValueError("gpus must be non-negative")
        _require_finite("memory_gb", self.memory_gb, minimum=0.0)
        _require_finite("submit_time", self.submit_time, minimum=0.0)
        _require_finite("estimated_runtime", self.estimated_runtime, minimum=0.0)
        if self.estimated_runtime == 0:
            raise ValueError("estimated_runtime must be positive")
        if self.deadline is not None:
            _require_finite("deadline", self.deadline, minimum=self.submit_time)


@dataclass(frozen=True, slots=True)
class Node:
    """A heterogeneous compute node and its schedulable capacities."""

    node_id: str
    cpus: int
    gpus: int
    memory_gb: float

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("node_id must not be empty")
        if self.cpus <= 0:
            raise ValueError("cpus must be positive")
        if self.gpus < 0:
            raise ValueError("gpus must be non-negative")
        _require_finite("memory_gb", self.memory_gb, minimum=0.0)

    def can_host(self, job: Job) -> bool:
        """Return whether the job can fit on this node when it is idle."""

        return (
            job.cpus <= self.cpus
            and job.gpus <= self.gpus
            and job.memory_gb <= self.memory_gb
        )


@dataclass(frozen=True, slots=True)
class Allocation:
    """The simulated placement and timing of one job."""

    job: Job
    node: Node
    start_time: float
    end_time: float

    def __post_init__(self) -> None:
        _require_finite("start_time", self.start_time, minimum=self.job.submit_time)
        _require_finite("end_time", self.end_time, minimum=self.start_time)
        if not self.node.can_host(self.job):
            raise ValueError(
                f"job {self.job.job_id!r} is incompatible with node "
                f"{self.node.node_id!r}"
            )

    @property
    def waiting_time(self) -> float:
        return self.start_time - self.job.submit_time

    @property
    def runtime(self) -> float:
        return self.end_time - self.start_time

    @property
    def missed_deadline(self) -> bool:
        return self.job.deadline is not None and self.end_time > self.job.deadline


@dataclass(frozen=True, slots=True)
class Schedule:
    """Completed allocations plus the cluster on which they were replayed."""

    allocations: tuple[Allocation, ...]
    nodes: tuple[Node, ...]

    def __post_init__(self) -> None:
        job_ids = [allocation.job.job_id for allocation in self.allocations]
        if len(job_ids) != len(set(job_ids)):
            raise ValueError("a job may only appear once in a schedule")

        node_ids = [node.node_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("node IDs must be unique")

    def allocation_for(self, job_id: str) -> Allocation:
        """Look up a job allocation, raising KeyError if it is absent."""

        for allocation in self.allocations:
            if allocation.job.job_id == job_id:
                return allocation
        raise KeyError(job_id)
