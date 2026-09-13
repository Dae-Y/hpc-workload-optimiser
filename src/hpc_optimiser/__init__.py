"""Tools for stylised HPC scheduling and optimisation experiments."""

from .fcfs import FCFSScheduler, schedule_fcfs
from .metrics import ScheduleMetrics, calculate_metrics
from .milp_scheduler import (
    MILPConfig,
    MILPScheduler,
    MILPSolveInfo,
    MILPSolverError,
    schedule_milp,
)
from .models import Allocation, Job, Node, Schedule
from .nonlinear_energy import (
    EnergyOptimisationConfig,
    EnergyOptimisationError,
    EnergyOptimisationResult,
    GPUPowerDecision,
    optimise_gpu_energy,
    relative_performance,
)
from .workload import generate_workload

__all__ = [
    "Allocation",
    "FCFSScheduler",
    "EnergyOptimisationConfig",
    "EnergyOptimisationError",
    "EnergyOptimisationResult",
    "GPUPowerDecision",
    "Job",
    "MILPConfig",
    "MILPScheduler",
    "MILPSolveInfo",
    "MILPSolverError",
    "Node",
    "Schedule",
    "ScheduleMetrics",
    "calculate_metrics",
    "generate_workload",
    "optimise_gpu_energy",
    "relative_performance",
    "schedule_fcfs",
    "schedule_milp",
]
