"""Time-indexed MILP scheduling for small heterogeneous clusters.

This is an educational formulation for toy/research workloads.  It is not
intended to reproduce a production batch scheduler or scale to large traces.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil, isfinite

import pulp

from .models import Allocation, Job, Node, Schedule


@dataclass(frozen=True, slots=True)
class MILPConfig:
    """Discretisation, objective weights, and CBC runtime settings.

    The objective weights express a trade-off rather than claiming that wait,
    tardiness, and makespan have interchangeable real-world costs.
    """

    slot_size: float = 1.0
    wait_weight: float = 1.0
    tardiness_weight: float = 5.0
    makespan_weight: float = 1.0
    solver_time_limit: float | None = None
    solver_msg: bool = False

    def __post_init__(self) -> None:
        if not isfinite(self.slot_size) or self.slot_size <= 0:
            raise ValueError("slot_size must be finite and positive")
        for name in ("wait_weight", "tardiness_weight", "makespan_weight"):
            value = getattr(self, name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not any(
            (self.wait_weight, self.tardiness_weight, self.makespan_weight)
        ):
            raise ValueError("at least one objective weight must be positive")
        if self.solver_time_limit is not None and (
            not isfinite(self.solver_time_limit) or self.solver_time_limit <= 0
        ):
            raise ValueError("solver_time_limit must be finite and positive")


@dataclass(frozen=True, slots=True)
class MILPSolveInfo:
    """Diagnostics from the most recent solve."""

    status: str
    solution_status: str
    objective_value: float | None
    binary_variables: int
    constraints: int
    horizon_slots: int
    slot_size: float


class MILPSolverError(RuntimeError):
    """Raised when CBC does not produce a proven optimal schedule."""


class MILPScheduler:
    """Build and solve a time-indexed mixed-integer scheduling model."""

    def __init__(self, config: MILPConfig | None = None) -> None:
        self.config = config or MILPConfig()
        self.last_solve_info: MILPSolveInfo | None = None

    def schedule(self, jobs: Sequence[Job], nodes: Sequence[Node]) -> Schedule:
        """Optimise node placement and discrete start time for every job."""

        jobs = list(jobs)
        nodes = list(nodes)
        _validate_inputs(jobs, nodes)

        if not jobs:
            self.last_solve_info = MILPSolveInfo(
                status="Optimal",
                solution_status="Optimal Solution Found",
                objective_value=0.0,
                binary_variables=0,
                constraints=0,
                horizon_slots=0,
                slot_size=self.config.slot_size,
            )
            return Schedule(allocations=(), nodes=tuple(nodes))

        release_slots = [
            ceil(job.submit_time / self.config.slot_size) for job in jobs
        ]
        duration_slots = [
            ceil(job.estimated_runtime / self.config.slot_size) for job in jobs
        ]
        first_slot, horizon_slot = _serial_horizon(
            jobs, release_slots, duration_slots
        )

        model = pulp.LpProblem("hpc_time_indexed_scheduling", pulp.LpMinimize)
        starts: dict[tuple[int, int, int], pulp.LpVariable] = {}
        job_variables: dict[int, list[pulp.LpVariable]] = defaultdict(list)
        # Pre-index variables by every slot they occupy for compact constraints.
        active: dict[tuple[int, int], list[tuple[int, pulp.LpVariable]]] = (
            defaultdict(list)
        )

        for job_index, job in enumerate(jobs):
            latest_start = horizon_slot - duration_slots[job_index]
            for node_index, node in enumerate(nodes):
                if not node.can_host(job):
                    continue
                for start_slot in range(release_slots[job_index], latest_start + 1):
                    variable = model.add_variable(
                        f"x_j{job_index}_n{node_index}_t{start_slot}",
                        cat=pulp.LpBinary,
                    )
                    starts[job_index, node_index, start_slot] = variable
                    job_variables[job_index].append(variable)
                    for slot in range(
                        start_slot, start_slot + duration_slots[job_index]
                    ):
                        active[node_index, slot].append((job_index, variable))

        for job_index in range(len(jobs)):
            model += (
                pulp.lpSum(job_variables[job_index]) == 1,
                f"schedule_once_j{job_index}",
            )

        for node_index, node in enumerate(nodes):
            for slot in range(first_slot, horizon_slot):
                active_variables = active.get((node_index, slot), [])
                if not active_variables:
                    continue
                model += (
                    pulp.lpSum(
                        jobs[job_index].cpus * variable
                        for job_index, variable in active_variables
                    )
                    <= node.cpus,
                    f"cpu_n{node_index}_t{slot}",
                )
                model += (
                    pulp.lpSum(
                        jobs[job_index].gpus * variable
                        for job_index, variable in active_variables
                    )
                    <= node.gpus,
                    f"gpu_n{node_index}_t{slot}",
                )
                model += (
                    pulp.lpSum(
                        jobs[job_index].memory_gb * variable
                        for job_index, variable in active_variables
                    )
                    <= node.memory_gb,
                    f"memory_n{node_index}_t{slot}",
                )

        start_expressions: list[pulp.LpAffineExpression] = []
        completion_expressions: list[pulp.LpAffineExpression] = []
        for job_index, job in enumerate(jobs):
            start_expression = pulp.lpSum(
                start_slot * self.config.slot_size * variable
                for (candidate_job, _node, start_slot), variable in starts.items()
                if candidate_job == job_index
            )
            start_expressions.append(start_expression)
            completion_expressions.append(start_expression + job.estimated_runtime)

        makespan = model.add_variable("makespan", lowBound=0)
        for job_index, completion in enumerate(completion_expressions):
            model += makespan >= completion, f"makespan_after_j{job_index}"

        tardiness_variables: list[pulp.LpVariable] = []
        for job_index, job in enumerate(jobs):
            if job.deadline is None:
                continue
            tardiness = model.add_variable(f"tardiness_j{job_index}", lowBound=0)
            model += (
                tardiness >= completion_expressions[job_index] - job.deadline,
                f"tardiness_after_j{job_index}",
            )
            tardiness_variables.append(tardiness)

        total_wait = pulp.lpSum(
            start_expressions[index] - job.submit_time
            for index, job in enumerate(jobs)
        )
        model += (
            self.config.wait_weight * total_wait
            + self.config.tardiness_weight * pulp.lpSum(tardiness_variables)
            + self.config.makespan_weight * makespan
        )

        solver = pulp.PULP_CBC_CMD(
            msg=self.config.solver_msg,
            timeLimit=self.config.solver_time_limit,
            threads=1,
        )
        status_code = model.solve(solver)
        status = pulp.LpStatus.get(status_code, f"Unknown ({status_code})")
        solution_status = pulp.LpSolution.get(
            model.sol_status, f"Unknown ({model.sol_status})"
        )
        objective = pulp.value(model.objective)
        self.last_solve_info = MILPSolveInfo(
            status=status,
            solution_status=solution_status,
            objective_value=float(objective) if objective is not None else None,
            binary_variables=len(starts),
            constraints=model.numConstraints(),
            horizon_slots=horizon_slot - first_slot,
            slot_size=self.config.slot_size,
        )

        if status_code != pulp.LpStatusOptimal or (
            model.sol_status != pulp.LpSolutionOptimal
        ):
            raise MILPSolverError(
                "CBC did not prove an optimal schedule "
                f"(model status: {status}; solution status: {solution_status}). "
                "Increase solver_time_limit, coarsen slot_size, or inspect workload "
                "feasibility."
            )

        allocations: list[Allocation] = []
        for job_index, job in enumerate(jobs):
            selected = [
                (node_index, start_slot)
                for (candidate_job, node_index, start_slot), variable in starts.items()
                if candidate_job == job_index and (pulp.value(variable) or 0.0) > 0.5
            ]
            if len(selected) != 1:
                raise MILPSolverError(
                    f"optimal model returned {len(selected)} starts for job "
                    f"{job.job_id!r}"
                )
            node_index, start_slot = selected[0]
            start_time = start_slot * self.config.slot_size
            allocations.append(
                Allocation(
                    job=job,
                    node=nodes[node_index],
                    start_time=start_time,
                    end_time=start_time + job.estimated_runtime,
                )
            )

        return Schedule(
            allocations=tuple(
                sorted(allocations, key=lambda item: (item.start_time, item.job.job_id))
            ),
            nodes=tuple(nodes),
        )


def schedule_milp(
    jobs: Sequence[Job],
    nodes: Sequence[Node],
    config: MILPConfig | None = None,
) -> Schedule:
    """Convenience wrapper returning a metrics-compatible optimised schedule."""

    return MILPScheduler(config).schedule(jobs, nodes)


def _serial_horizon(
    jobs: Sequence[Job], release_slots: Sequence[int], duration_slots: Sequence[int]
) -> tuple[int, int]:
    """Return a finite horizon from a guaranteed-feasible serial schedule.

    Running all jobs one at a time is conservative but always feasible once
    basic node compatibility has been checked.  It avoids using FCFS placement
    decisions as part of the optimisation.
    """

    first_slot = min(release_slots)
    cursor = first_slot
    ordered_indices = sorted(range(len(jobs)), key=lambda index: release_slots[index])
    for index in ordered_indices:
        cursor = max(cursor, release_slots[index]) + duration_slots[index]
    return first_slot, cursor


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
