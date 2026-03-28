# LiLAW PoC Implementation Plan — Index

This plan has been split into separate files to reduce token load:

## Overview & Setup
- **[PLAN-OVERVIEW.md](PLAN-OVERVIEW.md)** — Project goals, paper verification checkpoints, file structure, and task summary

## Individual Tasks
- **[PLAN-TASK-01.md](PLAN-TASK-01.md)** — Project scaffolding (uv, ruff, ty, pytest)
- **[PLAN-TASK-02.md](PLAN-TASK-02.md)** — Asymmetric noise injection
- **[PLAN-TASK-03.md](PLAN-TASK-03.md)** — LiLAW weight functions & meta-parameters
- **[PLAN-TASK-04.md](PLAN-TASK-04.md)** — MLP model (MLMD mock)
- **[PLAN-TASK-05.md](PLAN-TASK-05.md)** — Dataset loaders
- **[PLAN-TASK-06.md](PLAN-TASK-06.md)** — Evaluation metrics
- **[PLAN-TASK-07.md](PLAN-TASK-07.md)** — Training loop — baseline BCE
- **[PLAN-TASK-08.md](PLAN-TASK-08.md)** — Training loop — LiLAW weighted
- **[PLAN-TASK-09.md](PLAN-TASK-09.md)** — Experiment runner
- **[PLAN-TASK-10.md](PLAN-TASK-10.md)** — End-to-end smoke test

## Quick Start
1. Start with **PLAN-OVERVIEW.md** to understand the project
2. Implement tasks in order: Task 1 → Task 10
3. Each task file is self-contained with tests, implementation, and commit instructions
4. Use `ruff check` and `ty check` as specified in each task
5. Run `pytest` after each task to verify

**Total: ~42 tests across 10 tasks.**
