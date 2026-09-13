# HPC Workload Optimiser

A small simulation-based project exploring mathematical optimisation for
heterogeneous CPU/GPU cluster scheduling.

## Motivation

During my research internship at the Pawsey Supercomputing Research Centre,
I used SLURM to submit and monitor computational workloads on HPC systems.

This made me interested in what happens underneath the user-facing scheduler:
how jobs are assigned to limited CPU/GPU resources, and how scheduling
decisions affect waiting time, utilisation, deadlines, and energy usage.

This project explores that problem using:

- a simple HPC cluster simulator
- FCFS scheduling as a baseline
- mixed-integer optimisation for workload placement
- constrained nonlinear optimisation for energy/resource allocation

## Goals

Compare scheduling strategies using:

- average job waiting time
- total makespan
- CPU/GPU utilisation
- deadline misses
- estimated energy consumption

## Status

Work in progress.
