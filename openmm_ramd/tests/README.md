# Running the Openmm_ramd test suite

This directory contains the parametrized pytest suite for openmm_ramd.
It is assumed to be running in the parent openmm_ramd/ folder.

## Basic usage

Activate the project environment first if needed:

Run the default test suite:

```bash
pytest openmm_ramd/tests
```

Some of the tests require CUDA. To skip CUDA-mandatory tests, run:

```bash
pytest -m "not needs_cuda"
```

## Useful verbosity and debugging options

Show full test IDs and parametrized scenario names:

```bash
pytest openmm_ramd/tests -vv --full
```

> Note: the `--durations` summary is printed **at the end** of the pytest run.

## Running a single test or scenario

Run one specific test function:

```bash
pytest openmm_ramd/tests/test_run.py::test_run_produces_model -vv --full
```

## Code coverage

Requires `pytest-cov` (already listed in `devtools/conda-envs/test_env.yaml`).
Install if needed:

```bash
pip install pytest-cov
```

Generate a visual HTML report (recommended — shows line-by-line highlighting):

```bash
pytest openmm_ramd/tests --cov=openmm_ramd --cov-branch --cov-report=html:htmlcov
xdg-open htmlcov/index.html
```

The HTML report highlights each source file with green (covered), red (missed),
and yellow (partial branch) lines. Click any file in the summary table to see
the detail view.

Run tests with a terminal summary showing which lines are not covered:

```bash
pytest openmm_ramd/tests --full --cov=openmm_ramd --cov-report=term-missing
```
