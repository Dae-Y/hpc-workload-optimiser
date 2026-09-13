"""Discrete-event replay for scheduling policies."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from .models import Allocation, Job, Node, Schedule


class SchedulingPolicy(Protocol):
    """Small interface shared by the simulator and future schedulers."""

    def order_jobs(self, jobs: Sequence[Job]) -> list[Job]: ...

    def select_node(
        self,
        job: Job,
        nodes: Sequence[Node],
        running: Sequence[Allocation],
    ) -> Node | None: ...


def simulate(
    jobs: Sequence[Job],
    nodes: Sequence[Node],
    scheduler: SchedulingPolicy,
) -> Schedule:
    """Replay jobs by jumping virtual time between submissions/completions."""

    jobs = list(jobs)
    nodes = list(nodes)
    _validate_inputs(jobs, nodes)

    if not jobs:
        return Schedule(allocations=(), nodes=tuple(nodes))

    pending = scheduler.order_jobs(jobs)
    queue: list[Job] = []
    running: list[Allocation] = []
    allocations: list[Allocation] = []
    now = pending[0].submit_time

    while len(allocations) < len(jobs):
        running = [allocation for allocation in running if allocation.end_time > now]

        while pending and pending[0].submit_time <= now:
            queue.append(pending.pop(0))

        # Strict FCFS: repeatedly consider only the head of the waiting queue.
        while queue:
            job = queue[0]
            node = scheduler.select_node(job, nodes, running)
            if node is None:
                break

            allocation = Allocation(
                job=job,
                node=node,
                start_time=now,
                end_time=now + job.estimated_runtime,
            )
            allocations.append(allocation)
            running.append(allocation)
            queue.pop(0)

        if len(allocations) == len(jobs):
            break

        future_events = [allocation.end_time for allocation in running]
        if pending:
            future_events.append(pending[0].submit_time)
        future_events = [event for event in future_events if event > now]
        if not future_events:
            raise RuntimeError("simulation cannot advance; scheduler made no progress")
        now = min(future_events)

    return Schedule(
        allocations=tuple(
            sorted(allocations, key=lambda item: (item.start_time, item.job.job_id))
        ),
        nodes=tuple(nodes),
    )


def _validate_inputs(jobs: Sequence[Job], nodes: Sequence[Node]) -> None:
    if jobs and not nodes:
        raise ValueError("at least one node is required to schedule jobs")
    if len({job.job_id for job in jobs}) != len(jobs):
        raise ValueError("job IDs must be unique")
    if len({node.node_id for node in nodes}) != len(nodes):
        raise ValueError("node IDs must be unique")

    for job in jobs:
        if not any(node.can_host(job) for node in nodes):
            raise ValueError(f"job {job.job_id!r} cannot fit on any node")
