"""Reproducible synthetic workloads with SLURM-like resource requests."""

from __future__ import annotations

import random

from .models import Job


def generate_workload(num_jobs: int, seed: int = 0) -> list[Job]:
    """Generate a deterministic mix of CPU-only and GPU batch jobs.

    Times are abstract simulation units.  The generator aims for plausible
    variety for experiments, not a statistical model of a real HPC system.
    """

    if num_jobs < 0:
        raise ValueError("num_jobs must be non-negative")

    rng = random.Random(seed)
    jobs: list[Job] = []
    submit_time = 0.0

    for index in range(num_jobs):
        if index:
            # Bursty enough to create contention without requiring wall-clock time.
            submit_time += rng.expovariate(1 / 2.5)

        job_class = rng.choices(
            ("cpu", "small_gpu", "large_gpu"), weights=(0.65, 0.25, 0.10), k=1
        )[0]

        if job_class == "cpu":
            cpus = rng.choice((1, 2, 4, 8, 16, 32))
            gpus = 0
            memory_per_cpu = rng.choice((2, 4, 6))
            runtime = rng.uniform(1.0, 12.0)
        elif job_class == "small_gpu":
            cpus = rng.choice((4, 8, 16))
            gpus = rng.choice((1, 2))
            memory_per_cpu = rng.choice((4, 6, 8))
            runtime = rng.uniform(2.0, 16.0)
        else:
            cpus = rng.choice((16, 32, 64))
            gpus = rng.choice((4, 8))
            memory_per_cpu = rng.choice((4, 6, 8))
            runtime = rng.uniform(4.0, 24.0)

        rounded_submit = round(submit_time, 3)
        rounded_runtime = round(runtime, 3)
        # Deadlines are absolute and deliberately range from tight to relaxed.
        deadline = rounded_submit + round(
            rounded_runtime * rng.uniform(1.3, 3.5), 3
        )

        jobs.append(
            Job(
                job_id=f"job-{index:04d}",
                cpus=cpus,
                gpus=gpus,
                memory_gb=float(cpus * memory_per_cpu),
                submit_time=rounded_submit,
                estimated_runtime=rounded_runtime,
                deadline=round(deadline, 3),
                priority=rng.choices((0, 1, 2), weights=(0.70, 0.25, 0.05), k=1)[0],
            )
        )

    return jobs
