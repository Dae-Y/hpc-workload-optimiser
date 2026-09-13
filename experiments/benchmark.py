"""Reproducible benchmark pipeline for the three scheduling strategies."""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

# Permit `python experiments/benchmark.py` from a fresh, uninstalled checkout.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from hpc_optimiser.fcfs import schedule_fcfs
from hpc_optimiser.metrics import ScheduleMetrics, calculate_metrics
from hpc_optimiser.milp_scheduler import (
    MILPConfig,
    MILPScheduler,
    MILPSolveInfo,
    MILPSolverError,
)
from hpc_optimiser.models import Node
from hpc_optimiser.nonlinear_energy import (
    EnergyOptimisationConfig,
    EnergyOptimisationError,
    estimate_full_power_gpu_energy,
    optimise_gpu_energy,
)
from hpc_optimiser.workload import generate_workload


DEFAULT_JOB_COUNTS = (10, 20, 50)
DEFAULT_SEEDS = (2026, 7, 42, 123, 999)
METHODS = ("FCFS", "MILP", "MILP + nonlinear")
METRIC_COLUMNS = (
    "average_waiting_time",
    "makespan",
    "deadline_misses",
    "cpu_utilisation",
    "gpu_utilisation",
)
SUMMARY_METRICS = (
    *METRIC_COLUMNS,
    "gpu_energy_kwh",
    "baseline_gpu_energy_kwh",
    "optimised_gpu_energy_kwh",
    "gpu_energy_reduction_percent",
    "average_gpu_power_fraction",
    "milp_solve_time_seconds",
    "nlp_solve_time_seconds",
    "milp_binary_variables",
    "milp_constraints",
)


def benchmark_cluster() -> list[Node]:
    """Return the fixed heterogeneous cluster shared by every scenario."""

    return [
        Node("cpu-0", cpus=64, gpus=0, memory_gb=256),
        Node("gpu-0", cpus=64, gpus=8, memory_gb=512),
        Node("gpu-1", cpus=32, gpus=4, memory_gb=256),
    ]


def run_scenario(
    number_jobs: int,
    seed: int,
    *,
    milp_config: MILPConfig,
    energy_config: EnergyOptimisationConfig,
) -> list[dict[str, Any]]:
    """Run all methods on one shared deterministic workload and cluster."""

    jobs = generate_workload(number_jobs, seed=seed)
    nodes = benchmark_cluster()

    fcfs_schedule = schedule_fcfs(jobs, nodes)
    fcfs_metrics = calculate_metrics(fcfs_schedule)
    full_power_energy = estimate_full_power_gpu_energy(
        fcfs_schedule, energy_config
    )
    rows = [
        _result_row(
            number_jobs,
            seed,
            "FCFS",
            metrics=fcfs_metrics,
            gpu_energy_kwh=full_power_energy,
            baseline_gpu_energy_kwh=full_power_energy,
        )
    ]

    scheduler = MILPScheduler(milp_config)
    milp_started = perf_counter()
    try:
        milp_schedule = scheduler.schedule(jobs, nodes)
    except MILPSolverError as error:
        milp_seconds = perf_counter() - milp_started
        info = scheduler.last_solve_info
        failure_fields = _milp_fields(info, milp_seconds)
        failure_fields["error_message"] = str(error)
        rows.append(
            _result_row(
                number_jobs,
                seed,
                "MILP",
                gpu_energy_kwh=full_power_energy,
                baseline_gpu_energy_kwh=full_power_energy,
                **failure_fields,
            )
        )
        rows.append(
            _result_row(
                number_jobs,
                seed,
                "MILP + nonlinear",
                baseline_gpu_energy_kwh=full_power_energy,
                **failure_fields,
            )
        )
        return rows

    milp_seconds = perf_counter() - milp_started
    info = scheduler.last_solve_info
    milp_fields = _milp_fields(info, milp_seconds)
    milp_metrics = calculate_metrics(milp_schedule)
    rows.append(
        _result_row(
            number_jobs,
            seed,
            "MILP",
            metrics=milp_metrics,
            gpu_energy_kwh=full_power_energy,
            baseline_gpu_energy_kwh=full_power_energy,
            **milp_fields,
        )
    )

    nlp_started = perf_counter()
    try:
        energy_result = optimise_gpu_energy(milp_schedule, energy_config)
    except EnergyOptimisationError as error:
        rows.append(
            _result_row(
                number_jobs,
                seed,
                "MILP + nonlinear",
                nlp_solve_time_seconds=perf_counter() - nlp_started,
                nlp_success=False,
                error_message=str(error),
                **milp_fields,
            )
        )
        return rows

    nlp_seconds = perf_counter() - nlp_started
    rows.append(
        _result_row(
            number_jobs,
            seed,
            "MILP + nonlinear",
            metrics=calculate_metrics(energy_result.adjusted_schedule),
            gpu_energy_kwh=energy_result.optimised_gpu_energy_kwh,
            baseline_gpu_energy_kwh=energy_result.baseline_gpu_energy_kwh,
            optimised_gpu_energy_kwh=energy_result.optimised_gpu_energy_kwh,
            gpu_energy_reduction_percent=energy_result.energy_reduction_percent,
            average_gpu_power_fraction=energy_result.average_power_fraction,
            nlp_solve_time_seconds=nlp_seconds,
            nlp_success=energy_result.optimiser_success,
            nlp_status=energy_result.optimiser_status,
            nlp_message=energy_result.optimiser_message,
            total_runtime_increase=energy_result.total_runtime_increase,
            **milp_fields,
        )
    )
    return rows


def aggregate_results(results: pd.DataFrame) -> pd.DataFrame:
    """Compute per-size/method mean and sample standard deviation by seed."""

    grouped = results.groupby(["number_jobs", "method"], sort=False)
    aggregated = grouped[list(SUMMARY_METRICS)].agg(["mean", "std"])
    aggregated.columns = [
        f"{metric}_{statistic}" for metric, statistic in aggregated.columns
    ]
    aggregated = aggregated.reset_index()
    counts = grouped["average_waiting_time"].count().reset_index(
        name="successful_scenarios"
    )
    totals = grouped.size().reset_index(name="scenario_count")
    return (
        totals.merge(counts, on=["number_jobs", "method"])
        .merge(aggregated, on=["number_jobs", "method"])
    )


def create_plots(summary: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Write five simple mean-and-standard-deviation benchmark figures."""

    specifications = (
        (
            "average_waiting_time",
            "Average waiting time (simulation units)",
            "average_waiting_time.png",
            METHODS,
        ),
        ("makespan", "Makespan (simulation units)", "makespan.png", METHODS),
        ("deadline_misses", "Deadline misses", "deadline_misses.png", METHODS),
        (
            "gpu_energy_kwh",
            "Estimated dynamic GPU energy (kWh)",
            "gpu_energy.png",
            METHODS,
        ),
        (
            "milp_solve_time_seconds",
            "MILP solve time (seconds)",
            "milp_solve_time.png",
            ("MILP",),
        ),
    )
    paths: list[Path] = []
    for metric, ylabel, filename, methods in specifications:
        path = output_dir / filename
        _plot_metric(summary, metric, ylabel, path, methods)
        paths.append(path)
    return paths


def _plot_metric(
    summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    path: Path,
    methods: Sequence[str],
) -> None:
    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    omitted_failures = False
    for method in methods:
        method_rows = summary[summary["method"] == method].sort_values("number_jobs")
        if metric != "milp_solve_time_seconds":
            omitted_failures = omitted_failures or bool(
                (
                    method_rows["successful_scenarios"]
                    < method_rows["scenario_count"]
                ).any()
            )
        valid = method_rows[f"{metric}_mean"].notna()
        method_rows = method_rows[valid]
        if method_rows.empty:
            continue
        errors = method_rows[f"{metric}_std"].fillna(0.0)
        axis.errorbar(
            method_rows["number_jobs"],
            method_rows[f"{metric}_mean"],
            yerr=errors,
            marker="o",
            capsize=4,
            linewidth=1.5,
            label=method,
        )

    axis.set_xlabel("Number of jobs")
    axis.set_ylabel(ylabel)
    axis.set_xticks(sorted(summary["number_jobs"].unique()))
    axis.grid(axis="y", alpha=0.3)
    axis.legend(frameon=False)
    if omitted_failures:
        axis.text(
            0.99,
            0.02,
            "Non-optimal schedules omitted; see CSV sample counts.",
            transform=axis.transAxes,
            horizontalalignment="right",
            fontsize=8,
        )
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _result_row(
    number_jobs: int,
    seed: int,
    method: str,
    *,
    metrics: ScheduleMetrics | None = None,
    **values: Any,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "number_jobs": number_jobs,
        "seed": seed,
        "method": method,
        **{metric: math.nan for metric in METRIC_COLUMNS},
        "gpu_energy_kwh": math.nan,
        "baseline_gpu_energy_kwh": math.nan,
        "optimised_gpu_energy_kwh": math.nan,
        "gpu_energy_reduction_percent": math.nan,
        "average_gpu_power_fraction": math.nan,
        "total_runtime_increase": math.nan,
        "milp_solve_time_seconds": math.nan,
        "nlp_solve_time_seconds": math.nan,
        "milp_binary_variables": math.nan,
        "milp_constraints": math.nan,
        "milp_status": None,
        "milp_model_status": None,
        "milp_solution_status": None,
        "nlp_success": None,
        "nlp_status": None,
        "nlp_message": None,
        "error_message": None,
    }
    if metrics is not None:
        row.update(metrics.as_dict())
    row.update(values)
    return row


def _milp_fields(
    info: MILPSolveInfo | None, solve_seconds: float
) -> dict[str, Any]:
    if info is None:
        return {
            "milp_solve_time_seconds": solve_seconds,
            "milp_status": "Unavailable",
        }
    proven_optimal = (
        info.status == "Optimal"
        and info.solution_status == "Optimal Solution Found"
    )
    reported_status = (
        "Optimal"
        if proven_optimal
        else f"Not proven optimal ({info.solution_status})"
    )
    return {
        "milp_solve_time_seconds": solve_seconds,
        "milp_binary_variables": info.binary_variables,
        "milp_constraints": info.constraints,
        "milp_status": reported_status,
        "milp_model_status": info.status,
        "milp_solution_status": info.solution_status,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-counts", nargs="+", type=int, default=DEFAULT_JOB_COUNTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--slot-size", type=float, default=2.0)
    parser.add_argument("--milp-time-limit", type=float, default=60.0)
    parser.add_argument("--minimum-power-fraction", type=float, default=0.5)
    parser.add_argument("--nominal-gpu-power-kw", type=float, default=0.3)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--solver-msg", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if any(count <= 0 for count in args.job_counts):
        raise ValueError("job counts must be positive")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    milp_config = MILPConfig(
        slot_size=args.slot_size,
        solver_time_limit=args.milp_time_limit,
        solver_msg=args.solver_msg,
    )
    energy_config = EnergyOptimisationConfig(
        milp_slot_size=args.slot_size,
        minimum_power_fraction=args.minimum_power_fraction,
        nominal_gpu_power_kw=args.nominal_gpu_power_kw,
    )

    rows: list[dict[str, Any]] = []
    scenario_total = len(args.job_counts) * len(args.seeds)
    scenario_number = 0
    for number_jobs in args.job_counts:
        for seed in args.seeds:
            scenario_number += 1
            print(
                f"[{scenario_number}/{scenario_total}] jobs={number_jobs}, seed={seed}",
                flush=True,
            )
            scenario_rows = run_scenario(
                number_jobs,
                seed,
                milp_config=milp_config,
                energy_config=energy_config,
            )
            rows.extend(scenario_rows)
            milp_row = scenario_rows[1]
            print(
                f"  MILP status={milp_row['milp_status']}, "
                f"time={milp_row['milp_solve_time_seconds']:.3f}s",
                flush=True,
            )

    results = pd.DataFrame(rows)
    summary = aggregate_results(results)
    raw_path = output_dir / "benchmark_results.csv"
    summary_path = output_dir / "benchmark_summary.csv"
    results.to_csv(raw_path, index=False, na_rep="")
    summary.to_csv(summary_path, index=False, na_rep="")
    plot_paths = create_plots(summary, output_dir)

    print(f"\nRaw results: {raw_path}")
    print(f"Summary:     {summary_path}")
    for path in plot_paths:
        print(f"Plot:        {path}")

    display_columns = [
        "number_jobs",
        "method",
        "successful_scenarios",
        "average_waiting_time_mean",
        "makespan_mean",
        "deadline_misses_mean",
        "gpu_energy_kwh_mean",
        "milp_solve_time_seconds_mean",
    ]
    print("\nAggregated means (standard deviations are in the summary CSV):")
    print(summary[display_columns].to_string(index=False, float_format="%.4f"))

    failed = results[
        results["method"].eq("MILP") & results["average_waiting_time"].isna()
    ]
    if not failed.empty:
        print("\nNon-optimal MILP scenarios:")
        print(
            failed[["number_jobs", "seed", "milp_status", "error_message"]]
            .to_string(index=False)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
