"""Schedule-level performance metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .models import Schedule


@dataclass(frozen=True, slots=True)
class ScheduleMetrics:
    """Metrics for comparing scheduling approaches.

    An energy field can be added here next without changing schedule replay.
    """

    average_waiting_time: float
    makespan: float
    deadline_misses: int
    cpu_utilisation: float
    gpu_utilisation: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def calculate_metrics(schedule: Schedule) -> ScheduleMetrics:
    """Calculate time and cluster-wide capacity utilisation metrics."""

    if not schedule.allocations:
        return ScheduleMetrics(
            average_waiting_time=0.0,
            makespan=0.0,
            deadline_misses=0,
            cpu_utilisation=0.0,
            gpu_utilisation=0.0,
        )

    start_of_window = min(
        allocation.job.submit_time for allocation in schedule.allocations
    )
    end_of_window = max(allocation.end_time for allocation in schedule.allocations)
    makespan = end_of_window - start_of_window

    cpu_time = sum(
        allocation.job.cpus * allocation.runtime
        for allocation in schedule.allocations
    )
    gpu_time = sum(
        allocation.job.gpus * allocation.runtime
        for allocation in schedule.allocations
    )
    total_cpus = sum(node.cpus for node in schedule.nodes)
    total_gpus = sum(node.gpus for node in schedule.nodes)

    return ScheduleMetrics(
        average_waiting_time=sum(
            allocation.waiting_time for allocation in schedule.allocations
        )
        / len(schedule.allocations),
        makespan=makespan,
        deadline_misses=sum(
            allocation.missed_deadline for allocation in schedule.allocations
        ),
        cpu_utilisation=(cpu_time / (total_cpus * makespan))
        if total_cpus and makespan
        else 0.0,
        gpu_utilisation=(gpu_time / (total_gpus * makespan))
        if total_gpus and makespan
        else 0.0,
    )
