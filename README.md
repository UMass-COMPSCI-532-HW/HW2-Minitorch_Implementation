# COMPSCI 532 HW2 - MiniTorch Implementation

This repository contains my implementation work for **COMPSCI 532 (Systems for Data Structures), Spring 2026**, based on the MiniTorch Module 0 assignment scaffold.

## Overview

This project implements core pieces of a minimal deep learning framework, including:

- Elementary mathematical operators
- Higher-order list operators (`map`, `zipWith`, `reduce`)
- A `Module` / `Parameter` system for nested model definitions
- Property-based tests using `pytest` + `hypothesis`

## Repository Structure

- `minitorch/`: core implementation (`operators.py`, `module.py`, datasets/testing helpers)
- `tests/`: unit and property-based tests for operators and modules
- `project/`: optional Streamlit training/demo interfaces
- `Screenshots/`: test/task screenshots
- `README_.md`: original course scaffold readme

## Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Optional extras for visualization/demo apps:

```bash
pip install -r requirements.extra.txt
```

## Running Tests

Use the virtual environment pytest binary:

```bash
.venv/bin/pytest -q
```

Run specific homework task groups:

```bash
.venv/bin/pytest -q -m task0_1
.venv/bin/pytest -q -m task0_2
.venv/bin/pytest -q -m task0_3
.venv/bin/pytest -q -m task0_4
```

## Running the Demo App (Optional)

The Streamlit interface is under `project/`:

```bash
cd project
../.venv/bin/streamlit run app.py -- 0
```

`0` selects Module 0 pages.

## Notes

- Tested in this repository with Python `3.14.3` and `.venv/bin/pytest`.
- Current local test status: `43 passed, 1 xfailed`.

## Academic Integrity

This assignment is course work. Do not redistribute:

- The original course scaffold
- Solution code
- Report content

Refer to course policy/syllabus for integrity rules.

## Credits

Original MiniTorch materials:

- Docs: https://minitorch.github.io/
- Module 0 overview: https://minitorch.github.io/module0/module0/
