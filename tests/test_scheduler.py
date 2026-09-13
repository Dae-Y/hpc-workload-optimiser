from __future__ import annotations

import pytest

from hpc_optimiser.fcfs import schedule_fcfs
from hpc_optimiser.metrics import calculate_metrics
from hpc_optimiser.models import Job, Node
from hpc_optimiser.workload import generate_workload


def test_hand_written_fcfs_schedule_and_metrics() -> None:
    """A small exact example checks queueing, timing, and metric arithmetic."""

    node = Node(node_id="gpu-0", cpus=4, gpus=2, memory_gb=32)
    jobs = [
        Job(
            job_id="first",
            cpus=4,
            gpus=2,
            memory_gb=16,
            submit_time=0,
            estimated_runtime=2,
            deadline=1,
        ),
        Job(
            job_id="second",
            cpus=4,
            gpus=0,
            memory_gb=8,
            submit_time=1,
            estimated_runtime=1,
            deadline=4,
        ),
    ]

    schedule = schedule_fcfs(jobs, [node])

    assert schedule.allocation_for("first").start_time == 0
    assert schedule.allocation_for("first").end_time == 2
    assert schedule.allocation_for("second").start_time == 2
    assert schedule.allocation_for("second").end_time == 3

    metrics = calculate_metrics(schedule)
    assert metrics.average_waiting_time == pytest.approx(0.5)
    assert metrics.makespan == pytest.approx(3.0)
    assert metrics.deadline_misses == 1
    assert metrics.cpu_utilisation == pytest.approx(1.0)
    assert metrics.gpu_utilisation == pytest.approx(2 / 3)


def test_every_job_is_scheduled_at_or_after_submission() -> None:
    nodes = _mixed_nodes()
    jobs = generate_workload(num_jobs=50, seed=17)

    schedule = schedule_fcfs(jobs, nodes)

    assert {allocation.job.job_id for allocation in schedule.allocations} == {
        job.job_id for job in jobs
    }
    assert all(
        allocation.start_time >= allocation.job.submit_time
        for allocation in schedule.allocations
    )


def test_allocations_are_compatible_and_never_exceed_capacity() -> None:
    nodes = _mixed_nodes()
    schedule = schedule_fcfs(generate_workload(num_jobs=40, seed=8), nodes)

    for allocation in schedule.allocations:
        assert allocation.node.can_host(allocation.job)

    event_times = sorted(
        {
            time
            for allocation in schedule.allocations
            for time in (allocation.start_time, allocation.end_time)
        }
    )
    for time in event_times:
        for node in nodes:
            active = [
                allocation
                for allocation in schedule.allocations
                if allocation.node == node
                and allocation.start_time <= time < allocation.end_time
            ]
            assert sum(item.job.cpus for item in active) <= node.cpus
            assert sum(item.job.gpus for item in active) <= node.gpus
            assert sum(item.job.memory_gb for item in active) <= node.memory_gb


def test_workload_generation_is_reproducible_and_varied() -> None:
    jobs = generate_workload(num_jobs=100, seed=42)

    assert jobs == generate_workload(num_jobs=100, seed=42)
    assert jobs != generate_workload(num_jobs=100, seed=43)
    assert any(job.gpus == 0 for job in jobs)
    assert any(0 < job.gpus <= 2 for job in jobs)
    assert any(job.gpus >= 4 for job in jobs)


def test_incompatible_job_is_rejected() -> None:
    impossible = Job("too-large", 8, 1, 16, 0, 1)

    with pytest.raises(ValueError, match="cannot fit"):
        schedule_fcfs(
            [impossible], [Node("cpu-only", cpus=4, gpus=0, memory_gb=32)]
        )


def _mixed_nodes() -> list[Node]:
    return [
        Node("cpu-0", cpus=64, gpus=0, memory_gb=256),
        Node("gpu-0", cpus=64, gpus=8, memory_gb=512),
        Node("gpu-1", cpus=32, gpus=4, memory_gb=256),
    ]
