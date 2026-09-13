"""An intentionally simple, interpretable FCFS scheduling policy."""

from __future__ import annotations

from collections.abc import Sequence

from .models import Allocation, Job, Node, Schedule


class FCFSScheduler:
    """Strict FCFS policy using deterministic first-fit node placement.

    If the oldest queued job cannot start, younger jobs do not backfill around
    it.  Priority is retained in the data model for later policies but is
    deliberately ignored by this baseline.
    """

    def order_jobs(self, jobs: Sequence[Job]) -> list[Job]:
        # Python's stable sort preserves input order for simultaneous submits.
        return sorted(jobs, key=lambda job: job.submit_time)

    def select_node(
        self,
        job: Job,
        nodes: Sequence[Node],
        running: Sequence[Allocation],
    ) -> Node | None:
        """Return the first node with enough currently free resources."""

        for node in nodes:
            on_node = (allocation for allocation in running if allocation.node == node)
            used_cpus = used_gpus = 0
            used_memory = 0.0
            for allocation in on_node:
                used_cpus += allocation.job.cpus
                used_gpus += allocation.job.gpus
                used_memory += allocation.job.memory_gb

            if (
                used_cpus + job.cpus <= node.cpus
                and used_gpus + job.gpus <= node.gpus
                and used_memory + job.memory_gb <= node.memory_gb
            ):
                return node
        return None


def schedule_fcfs(jobs: Sequence[Job], nodes: Sequence[Node]) -> Schedule:
    """Replay jobs with the FCFS policy and return their schedule."""

    # Local import keeps the policy independent of the simulation engine.
    from .simulator import simulate

    return simulate(jobs, nodes, scheduler=FCFSScheduler())
