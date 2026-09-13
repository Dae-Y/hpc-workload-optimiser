from __future__ import annotations

import pandas as pd
import pytest

from experiments.benchmark import aggregate_results


def test_aggregate_results_keeps_raw_scenarios_separate() -> None:
    rows = []
    for seed, wait in ((1, 2.0), (2, 4.0)):
        row = {
            "number_jobs": 10,
            "seed": seed,
            "method": "FCFS",
            "average_waiting_time": wait,
        }
        for metric in (
            "makespan",
            "deadline_misses",
            "cpu_utilisation",
            "gpu_utilisation",
            "gpu_energy_kwh",
            "baseline_gpu_energy_kwh",
            "optimised_gpu_energy_kwh",
            "gpu_energy_reduction_percent",
            "average_gpu_power_fraction",
            "milp_solve_time_seconds",
            "nlp_solve_time_seconds",
            "milp_binary_variables",
            "milp_constraints",
        ):
            row[metric] = float("nan")
        rows.append(row)

    summary = aggregate_results(pd.DataFrame(rows))

    assert len(summary) == 1
    assert summary.loc[0, "scenario_count"] == 2
    assert summary.loc[0, "successful_scenarios"] == 2
    assert summary.loc[0, "average_waiting_time_mean"] == pytest.approx(3.0)
    assert summary.loc[0, "average_waiting_time_std"] == pytest.approx(2**0.5)
