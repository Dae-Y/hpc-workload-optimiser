from __future__ import annotations

import pytest

from hpc_optimiser.models import Allocation, Job, Node, Schedule
from hpc_optimiser.nonlinear_energy import (
    EnergyOptimisationConfig,
    optimise_gpu_energy,
    relative_performance,
)


def test_relative_performance_is_monotone_concave_and_normalised() -> None:
    low = relative_performance(0.4)
    middle = relative_performance(0.7)
    full = relative_performance(1.0)

    assert 0 < low < middle < full
    assert full == pytest.approx(1.0)
    assert middle - low > full - middle


def test_energy_optimisation_is_feasible_and_reduces_energy() -> None:
    schedule = _small_schedule()
    config = EnergyOptimisationConfig(
        milp_slot_size=2.0,
        minimum_power_fraction=0.4,
    )

    result = optimise_gpu_energy(schedule, config)

    assert result.optimiser_success
    assert result.optimised_gpu_energy_kwh < result.baseline_gpu_energy_kwh
    assert result.energy_reduction_percent > 0
    assert result.objective_value == pytest.approx(result.optimised_gpu_energy_kwh)
    assert result.average_power_fraction is not None

    assert {decision.job_id for decision in result.decisions} == {"gpu-job"}
    decision = result.decisions[0]
    assert config.minimum_power_fraction <= decision.power_fraction <= 1.0
    assert decision.adjusted_runtime <= (
        decision.reserved_runtime + config.feasibility_tolerance
    )
    assert decision.adjusted_runtime > decision.base_runtime


def test_cpu_jobs_are_unchanged_and_on_time_job_does_not_become_late() -> None:
    schedule = _small_schedule()

    result = optimise_gpu_energy(
        schedule,
        EnergyOptimisationConfig(
            milp_slot_size=2.0,
            minimum_power_fraction=0.4,
        ),
    )

    original_cpu = schedule.allocation_for("cpu-job")
    adjusted_cpu = result.adjusted_schedule.allocation_for("cpu-job")
    adjusted_gpu = result.adjusted_schedule.allocation_for("gpu-job")

    assert all(decision.job_id != "cpu-job" for decision in result.decisions)
    assert adjusted_cpu == original_cpu
    assert adjusted_gpu.end_time <= adjusted_gpu.job.deadline
    assert not adjusted_gpu.missed_deadline


def test_schedule_without_gpu_jobs_is_a_successful_no_op() -> None:
    node = Node("cpu", 4, 0, 16)
    job = Job("cpu-job", 2, 0, 4, 0, 1)
    schedule = Schedule((Allocation(job, node, 0, 1),), (node,))

    result = optimise_gpu_energy(schedule)

    assert result.adjusted_schedule is schedule
    assert result.decisions == ()
    assert result.optimised_gpu_energy_kwh == 0
    assert result.average_power_fraction is None


@pytest.mark.parametrize(
    "keyword, value",
    [
        ("milp_slot_size", 0.0),
        ("minimum_power_fraction", 0.0),
        ("minimum_power_fraction", 1.1),
        ("performance_exponent", 0.0),
        ("performance_exponent", 1.0),
        ("nominal_gpu_power_kw", -1.0),
        ("max_iterations", 0),
    ],
)
def test_invalid_configuration_is_rejected(keyword: str, value: float) -> None:
    with pytest.raises(ValueError):
        EnergyOptimisationConfig(**{keyword: value})


def _small_schedule() -> Schedule:
    node = Node("gpu-node", cpus=4, gpus=1, memory_gb=16)
    gpu_job = Job(
        "gpu-job",
        cpus=2,
        gpus=1,
        memory_gb=8,
        submit_time=0,
        estimated_runtime=3,
        deadline=3.5,
    )
    cpu_job = Job(
        "cpu-job",
        cpus=2,
        gpus=0,
        memory_gb=8,
        submit_time=0,
        estimated_runtime=1,
        deadline=2,
    )
    return Schedule(
        allocations=(
            Allocation(gpu_job, node, start_time=0, end_time=3),
            Allocation(cpu_job, node, start_time=0, end_time=1),
        ),
        nodes=(node,),
    )
