from __future__ import annotations

import pytest

from hpc_optimiser.fcfs import schedule_fcfs
from hpc_optimiser.metrics import calculate_metrics
from hpc_optimiser.milp_scheduler import MILPConfig, MILPScheduler
from hpc_optimiser.models import Job, Node, Schedule


def test_small_milp_schedules_each_job_once_after_release() -> None:
    jobs = [
        Job("cpu-a", 4, 0, 8, 0.2, 1.2, deadline=4),
        Job("gpu-a", 4, 1, 16, 0.0, 2.0, deadline=3),
        Job("cpu-b", 2, 0, 4, 0.5, 1.0),
    ]
    nodes = [
        Node("cpu", 4, 0, 16),
        Node("gpu", 8, 2, 32),
    ]

    scheduler = MILPScheduler(MILPConfig(slot_size=0.5))
    schedule = scheduler.schedule(jobs, nodes)

    assert len(schedule.allocations) == len(jobs)
    assert {item.job.job_id for item in schedule.allocations} == {
        job.job_id for job in jobs
    }
    assert all(item.start_time >= item.job.submit_time for item in schedule.allocations)
    assert scheduler.last_solve_info is not None
    assert scheduler.last_solve_info.status == "Optimal"


def test_milp_respects_compatibility_and_all_resource_capacities() -> None:
    nodes = [Node("cpu", 4, 0, 16), Node("gpu", 6, 2, 24)]
    jobs = [
        Job("gpu-1", 3, 1, 12, 0, 2),
        Job("gpu-2", 3, 1, 12, 0, 2),
        Job("cpu-1", 4, 0, 16, 0, 3),
        Job("cpu-2", 2, 0, 8, 1, 2),
    ]

    schedule = MILPScheduler().schedule(jobs, nodes)

    assert all(item.node.can_host(item.job) for item in schedule.allocations)
    assert schedule.allocation_for("gpu-1").node.node_id == "gpu"
    assert schedule.allocation_for("gpu-2").node.node_id == "gpu"
    _assert_capacities(schedule)


def test_milp_rejects_an_incompatible_workload_clearly() -> None:
    job = Job("needs-gpu", 2, 1, 4, 0, 1)

    with pytest.raises(ValueError, match="cannot fit on any node"):
        MILPScheduler().schedule([job], [Node("cpu", 8, 0, 32)])


def test_milp_improves_makespan_over_first_fit_fcfs() -> None:
    """MILP preserves the GPU node for its uniquely compatible job."""

    nodes = [
        Node("gpu", cpus=4, gpus=1, memory_gb=16),
        Node("cpu", cpus=4, gpus=0, memory_gb=16),
    ]
    jobs = [
        Job("cpu-job", 4, 0, 16, 0, 10),
        Job("gpu-job", 4, 1, 16, 0, 10),
    ]

    fcfs_metrics = calculate_metrics(schedule_fcfs(jobs, nodes))
    milp_schedule = MILPScheduler().schedule(jobs, nodes)
    milp_metrics = calculate_metrics(milp_schedule)

    assert milp_schedule.allocation_for("cpu-job").node.node_id == "cpu"
    assert milp_schedule.allocation_for("gpu-job").node.node_id == "gpu"
    assert milp_metrics.makespan < fcfs_metrics.makespan
    assert milp_metrics.average_waiting_time < fcfs_metrics.average_waiting_time


def _assert_capacities(schedule: Schedule) -> None:
    event_times = {
        time
        for allocation in schedule.allocations
        for time in (allocation.start_time, allocation.end_time)
    }
    for time in event_times:
        for node in schedule.nodes:
            active = [
                item
                for item in schedule.allocations
                if item.node == node and item.start_time <= time < item.end_time
            ]
            assert sum(item.job.cpus for item in active) <= node.cpus
            assert sum(item.job.gpus for item in active) <= node.gpus
            assert sum(item.job.memory_gb for item in active) <= node.memory_gb
