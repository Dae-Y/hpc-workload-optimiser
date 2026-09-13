"""Post-scheduling nonlinear optimisation of stylised GPU power levels.

The model deliberately describes no specific GPU or HPC system.  By default,
one simulation time unit is treated as one minute solely for converting the
energy estimate to kWh.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite

import numpy as np
from scipy.optimize import minimize

from .models import Allocation, Schedule


@dataclass(frozen=True, slots=True)
class EnergyOptimisationConfig:
    """Parameters for the power/performance model and SLSQP solver."""

    milp_slot_size: float = 1.0
    minimum_power_fraction: float = 0.5
    performance_exponent: float = 0.7
    nominal_gpu_power_kw: float = 0.3
    minutes_per_time_unit: float = 1.0
    solver_tolerance: float = 1e-9
    feasibility_tolerance: float = 1e-7
    max_iterations: int = 500

    def __post_init__(self) -> None:
        _positive("milp_slot_size", self.milp_slot_size)
        _positive("nominal_gpu_power_kw", self.nominal_gpu_power_kw)
        _positive("minutes_per_time_unit", self.minutes_per_time_unit)
        _positive("solver_tolerance", self.solver_tolerance)
        _positive("feasibility_tolerance", self.feasibility_tolerance)
        if (
            not isfinite(self.minimum_power_fraction)
            or not 0 < self.minimum_power_fraction <= 1
        ):
            raise ValueError("minimum_power_fraction must be in (0, 1]")
        if (
            not isfinite(self.performance_exponent)
            or not 0 < self.performance_exponent < 1
        ):
            raise ValueError("performance_exponent must be in (0, 1)")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")


@dataclass(frozen=True, slots=True)
class GPUPowerDecision:
    """Optimised power, runtime, and energy for one GPU job."""

    job_id: str
    gpu_count: int
    power_fraction: float
    base_runtime: float
    adjusted_runtime: float
    reserved_runtime: float
    runtime_increase: float
    baseline_energy_kwh: float
    optimised_energy_kwh: float


@dataclass(frozen=True, slots=True)
class EnergyOptimisationResult:
    """Adjusted schedule and diagnostics from nonlinear optimisation."""

    adjusted_schedule: Schedule
    decisions: tuple[GPUPowerDecision, ...]
    baseline_gpu_energy_kwh: float
    optimised_gpu_energy_kwh: float
    energy_reduction_percent: float
    average_power_fraction: float | None
    total_runtime_increase: float
    optimiser_success: bool
    optimiser_status: int
    optimiser_message: str
    objective_value: float


class EnergyOptimisationError(RuntimeError):
    """Raised when SLSQP fails or returns an infeasible power allocation."""


def relative_performance(power_fraction: float, exponent: float = 0.7) -> float:
    """Return stylised relative GPU performance ``p**exponent``.

    With ``0 < exponent < 1`` this is increasing and concave, is normalised to
    one at full power, and increases runtime whenever power is reduced.
    """

    if not isfinite(power_fraction) or not 0 < power_fraction <= 1:
        raise ValueError("power_fraction must be in (0, 1]")
    if not isfinite(exponent) or not 0 < exponent < 1:
        raise ValueError("exponent must be in (0, 1)")
    return power_fraction**exponent


def optimise_gpu_energy(
    schedule: Schedule,
    config: EnergyOptimisationConfig | None = None,
) -> EnergyOptimisationResult:
    """Minimise GPU energy without exceeding MILP-reserved time windows."""

    config = config or EnergyOptimisationConfig()
    gpu_allocations = [
        allocation for allocation in schedule.allocations if allocation.job.gpus > 0
    ]

    if not gpu_allocations:
        return EnergyOptimisationResult(
            adjusted_schedule=schedule,
            decisions=(),
            baseline_gpu_energy_kwh=0.0,
            optimised_gpu_energy_kwh=0.0,
            energy_reduction_percent=0.0,
            average_power_fraction=None,
            total_runtime_increase=0.0,
            optimiser_success=True,
            optimiser_status=0,
            optimiser_message="No GPU jobs to optimise",
            objective_value=0.0,
        )

    base_runtimes, reserved_runtimes, allowed_runtimes = _runtime_windows(
        gpu_allocations, config
    )

    def objective(power_fractions: np.ndarray) -> float:
        adjusted = base_runtimes / np.power(
            power_fractions, config.performance_exponent
        )
        return float(
            np.sum(
                _gpu_counts(gpu_allocations)
                * config.nominal_gpu_power_kw
                * power_fractions
                * adjusted
                * config.minutes_per_time_unit
                / 60.0
            )
        )

    def runtime_slack(power_fractions: np.ndarray) -> np.ndarray:
        adjusted = base_runtimes / np.power(
            power_fractions, config.performance_exponent
        )
        return allowed_runtimes - adjusted

    initial_power = np.ones(len(gpu_allocations), dtype=float)
    optimisation = minimize(
        objective,
        initial_power,
        method="SLSQP",
        bounds=[(config.minimum_power_fraction, 1.0)] * len(gpu_allocations),
        constraints=[{"type": "ineq", "fun": runtime_slack}],
        options={
            "ftol": config.solver_tolerance,
            "maxiter": config.max_iterations,
            "disp": False,
        },
    )

    if not optimisation.success:
        raise EnergyOptimisationError(
            "SLSQP failed to find a feasible energy allocation "
            f"(status {optimisation.status}: {optimisation.message})"
        )

    power_fractions = np.clip(
        np.asarray(optimisation.x, dtype=float),
        config.minimum_power_fraction,
        1.0,
    )
    # Repair tiny solver-tolerance violations by moving power upward.
    required_power = np.power(
        base_runtimes / allowed_runtimes, 1.0 / config.performance_exponent
    )
    power_fractions = np.maximum(power_fractions, required_power)
    power_fractions = np.minimum(power_fractions, 1.0)
    adjusted_runtimes = base_runtimes / np.power(
        power_fractions, config.performance_exponent
    )

    if np.any(
        adjusted_runtimes
        > allowed_runtimes + config.feasibility_tolerance
    ):
        raise EnergyOptimisationError(
            "SLSQP returned runtimes outside the reserved scheduling windows"
        )

    gpu_counts = _gpu_counts(gpu_allocations)
    hours_per_time_unit = config.minutes_per_time_unit / 60.0
    baseline_energy = (
        gpu_counts
        * config.nominal_gpu_power_kw
        * base_runtimes
        * hours_per_time_unit
    )
    optimised_energy = (
        gpu_counts
        * config.nominal_gpu_power_kw
        * power_fractions
        * adjusted_runtimes
        * hours_per_time_unit
    )
    baseline_total = float(np.sum(baseline_energy))
    optimised_total = float(np.sum(optimised_energy))
    if optimised_total > baseline_total + config.feasibility_tolerance:
        raise EnergyOptimisationError(
            "optimised energy exceeds the feasible full-power baseline"
        )

    decisions = tuple(
        GPUPowerDecision(
            job_id=allocation.job.job_id,
            gpu_count=allocation.job.gpus,
            power_fraction=float(power_fractions[index]),
            base_runtime=float(base_runtimes[index]),
            adjusted_runtime=float(adjusted_runtimes[index]),
            reserved_runtime=float(reserved_runtimes[index]),
            runtime_increase=float(adjusted_runtimes[index] - base_runtimes[index]),
            baseline_energy_kwh=float(baseline_energy[index]),
            optimised_energy_kwh=float(optimised_energy[index]),
        )
        for index, allocation in enumerate(gpu_allocations)
    )

    runtimes_by_job = {
        decision.job_id: decision.adjusted_runtime for decision in decisions
    }
    adjusted_allocations = tuple(
        Allocation(
            job=allocation.job,
            node=allocation.node,
            start_time=allocation.start_time,
            end_time=allocation.start_time
            + runtimes_by_job.get(allocation.job.job_id, allocation.runtime),
        )
        for allocation in schedule.allocations
    )
    reduction = (
        100.0 * (baseline_total - optimised_total) / baseline_total
        if baseline_total
        else 0.0
    )

    return EnergyOptimisationResult(
        adjusted_schedule=Schedule(
            allocations=adjusted_allocations,
            nodes=schedule.nodes,
        ),
        decisions=decisions,
        baseline_gpu_energy_kwh=baseline_total,
        optimised_gpu_energy_kwh=optimised_total,
        energy_reduction_percent=reduction,
        average_power_fraction=float(np.mean(power_fractions)),
        total_runtime_increase=float(np.sum(adjusted_runtimes - base_runtimes)),
        optimiser_success=True,
        optimiser_status=int(optimisation.status),
        optimiser_message=str(optimisation.message),
        objective_value=optimised_total,
    )


def _runtime_windows(
    allocations: list[Allocation], config: EnergyOptimisationConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base_runtimes: list[float] = []
    reserved_runtimes: list[float] = []
    allowed_runtimes: list[float] = []

    for allocation in allocations:
        base_runtime = allocation.job.estimated_runtime
        if abs(allocation.runtime - base_runtime) > config.feasibility_tolerance:
            raise ValueError(
                f"allocation for {allocation.job.job_id!r} does not contain its "
                "unmodified estimated runtime"
            )
        slot_position = allocation.start_time / config.milp_slot_size
        aligned_start = round(slot_position) * config.milp_slot_size
        if (
            abs(allocation.start_time - aligned_start)
            > config.feasibility_tolerance
        ):
            raise ValueError(
                f"allocation for {allocation.job.job_id!r} is not aligned to the "
                "configured MILP slot size"
            )

        reserved_runtime = (
            ceil(base_runtime / config.milp_slot_size) * config.milp_slot_size
        )
        allowed_runtime = reserved_runtime
        deadline = allocation.job.deadline
        originally_on_time = (
            deadline is not None
            and allocation.end_time <= deadline + config.feasibility_tolerance
        )
        if originally_on_time:
            allowed_runtime = min(
                allowed_runtime, deadline - allocation.start_time
            )
        if allowed_runtime < base_runtime - config.feasibility_tolerance:
            raise ValueError(
                f"allocation for {allocation.job.job_id!r} has no feasible "
                "full-power runtime window"
            )
        # Remove harmless floating-point undershoot so p=1 is exactly feasible.
        allowed_runtime = max(allowed_runtime, base_runtime)

        base_runtimes.append(base_runtime)
        reserved_runtimes.append(reserved_runtime)
        allowed_runtimes.append(allowed_runtime)

    return (
        np.asarray(base_runtimes, dtype=float),
        np.asarray(reserved_runtimes, dtype=float),
        np.asarray(allowed_runtimes, dtype=float),
    )


def _gpu_counts(allocations: list[Allocation]) -> np.ndarray:
    return np.asarray([allocation.job.gpus for allocation in allocations], dtype=float)


def _positive(name: str, value: float) -> None:
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
