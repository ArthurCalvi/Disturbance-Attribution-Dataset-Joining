# AGENTS.md ─ Codex guidance for Disturbance‑Attribution‑Dataset‑Joining

## 1  Purpose

This file tells the Codex agent how to prepare the runtime, run the test
suite, and adhere to our coding conventions for this repository.

---

## 2  Environment setup

### 2.1 Required versions

* **Python ≥ 3.9**, tested on **3.11**.
* **pip ≥ 23.3**.

### 2.2 System packages (Debian/Ubuntu)

```bash
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
    gdal-bin libgdal-dev libproj-dev libspatialindex-dev libgeos-dev \
    build-essential ca-certificates
```

These libraries are required for GeoPandas, PyProj, Shapely and
rtree.

### 2.3 Python dependencies

```bash
python -m pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
```

Run the commands above **once** at the start of every container
(`postCreateCommand` if using Codespaces).
Codex: **do not** attempt to install dependencies individually; call the
full script so package versions remain consistent.

---

## 3  Test & quality checks

After *every* code change Codex must ensure all of the following exit
with status 0:

```bash
pytest -q            # runs unit tests
ruff check .         # linting
ruff format --check .  # formatting (should yield no diff)
```

The unit tests are located in the `tests/` directory and can also be
invoked with `python -m unittest discover tests`.

---

## 4  Coding conventions

* Follow **PEP 8** (max line length = 100 chars).
* Use **type hints** everywhere.
* Prefer **f‑strings** over `%` or `format`.
* Write Google‑style docstrings for public functions & classes.
* Keep external I/O (disk, network) out of `src/join_datasets` pure
  logic.

---

## 5  Files to ignore

Codex must never commit, create, or modify the following paths:

```
data/
results/
*.parquet
*.png
.ipynb_checkpoints/
```

---

## 6  Pull‑request message template

````
### Summary
<one‑line description>

### Details
- Why
- How
- Notes / trade‑offs

### Validation
```bash
ruff check . && pytest -q
````

```

---

## 7  Programmatic validation

This AGENTS.md includes programmatic checks (section 3).  Codex **must
run them** and obtain all‑green output before considering a task
complete.  

---

(End of file)

```
