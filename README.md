# GIC 2026 Phase 3 — Mitsubishi Chemical Group / AIST

**Team:** Quantum Pattern Recognition  
**Project:** A Sector-Scalable Generative Quantum Eigensolver with Adaptive
Topology and Exact Structured Circuit Synthesis  
**Track:** Quantum Materials Discovery Challenge: Scaling Generative Quantum
Eigensolver (GQE) Using NVIDIA CUDA-Q  
**Submission phase:** Phase 3 — Prototype Development

### Repository:

https://github.com/PR2aid/Mitsubishi.git

The best way is to download this package locally, then  package to qBrain then run the bellow commants:

### To run locally

```bash

unzip Mitsubishi-main.zip

cd ~/Mitsubishi-main/Source_Code

export QBRAID_GQE_ENV=~/Mitsubishi-main/.qbraid_gqe_env_clean

PYTHON_BIN=python bash setup.sh

"$QBRAID_GQE_ENV/bin/python" \
  -I -B certify_release.py --full

```

then the result can be seen both on the screen and is saved in

thanks dear so the results will be in ~/Mitsubishi-main/Source_Code/results/judge_reproduction/latest_certificate.json 
and ~/Mitsubishi-main/Source_Code/results/judge_reproduction/latest_run.json


This archive is the authoritative submission package. It contains the
official cover page, five technical pages, one references page, all source
code, checksum-bound frozen scientific inputs, a complete exact
126-distribution lock with a deterministic installer, tests, and a one-click
qBraid reproduction notebook.

The reproduction is CPU-only and credential-free. It does not import a
quantum provider, submit a QPU job, consume quantum credits, or require a GPU.

[<img src="https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_white.png" width="150" alt="Launch on qBraid">](https://account.qbraid.com?gitHubUrl=https://github.com/PR2aid/Mitsubishi.git)

## Judge quick start

1. Upload and extract
   `QuantumPatternRecognition_MitsubishiAIST_Phase3.zip` in qBraid Lab.
2. Open `Source_Code/QBRAID_RUNME.ipynb`.
3. Select a **Python 3.12** kernel.
4. Choose **Run → Run All Cells** and leave the Lab session open.
5. Accept the run only when the final cell displays:

   ```text
   ALL REQUESTED PHASE 3 RESULTS REPRODUCED WITHIN DECLARED TOLERANCES
   ```

During successful certification, Cell 5 prints these markers in order:

```text
VALIDATION PASS: <number> passed, 0 failed
ALL REQUESTED PHASE 3 RESULTS REPRODUCED WITHIN DECLARED TOLERANCES
<path to reproduction_summary.json>
PASS: invocation-bound certificate written to .../certificate.json
```

The final notebook cell then reads the current certificate and run pointer,
asserts their PASS status, prints their paths, and repeats the success banner.
Its certificate assertion is:

```json
{
  "status": "PASS",
  "mode": "full"
}
```

If any scientific comparison, environment check, test, or artifact binding
fails, the command exits nonzero and the notebook stops. An older PASS is not
accepted as the result of a failed current invocation.

## What is saved, and where

The result is both displayed on screen and saved to machine-readable files.
Every admitted certification invocation gets its own certificate directory,
and every scientific child run gets a new timestamped directory. No previous
scientific run is overwritten.

The two stable entry points are:

```text
Source_Code/results/judge_reproduction/latest_certificate.json
Source_Code/results/judge_reproduction/latest_run.json
```

`latest_certificate.json` is authoritative for the most recently admitted
certificate attempt. It records the
invocation ID, `PASS`/`FAILED`/`INTERRUPTED` status, mode, start and completion
times, wall time, source identity before and after execution, environment,
child-process result, and QPU/provider flags. On PASS it also contains the
SHA-256 manifest of every generated artifact; on an early failure or interrupt,
`generated_artifacts` can be null.

After a successful reproduction, `latest_run.json` points to the exact
timestamped scientific run:

```text
Source_Code/results/judge_reproduction/full_<UTC>_<invocation-prefix>/
```

It is updated only after scientific PASS. If a later attempt fails,
`latest_run.json` may still point to an older PASS; do not treat it as the
failed attempt's run. On PASS, require its `invocation_id` to equal the one in
`latest_certificate.json`.

The most useful files inside that run are:

| File | Meaning |
|---|---|
| `reproduction_summary.json` | Judge-facing scientific verdict; every check includes its name, status, actual value, and expected value or tolerance. |
| `execution_record.json` | Full-run status, timing, platform, Python version, lock hash, and `qpu_contacted=false`. |
| `environment.json` | Verified Python/platform/distribution contract. |
| `frozen_reference_audit.json` | Independent ten-problem RHF/CASCI reconstruction and checksum audit. |
| `qpd_result.json` | Exact 10,000-branch six-qubit QPD reconstruction check. |
| `advanced_method/enhanced_release_summary.json` | Complete baseline, topology, warm-start, residual-GQE, QSCI, and structured-resource results. |
| `advanced_method/canonical_table3_candidate/` | Fresh ten-file Table 3 evidence: one manifest, three parameter NPZ files, and six QASM files. |
| `advanced_method/tables/*.csv` | Human-readable regenerated result tables. |
| `objectives/*.json` | Full exact-energy versus QSCI objective records. |
| `finite_shot_beh2_6q.json` | Fixed-seed finite-shot Aer experiment. |
| `*.log` | Complete stage logs for diagnosis and review. |

The invocation-specific wrapper evidence is saved separately:

```text
Source_Code/results/judge_reproduction/certificates/<invocation-id>/
  certificate.json
  reproduce_console.log
```

## What the workflow compares

Judges do not need to compare numbers manually. The validator compares fresh
outputs against:

- `Source_Code/expected_metrics.json` for exact values and declared numerical
  tolerances;
- `Source_Code/frozen_inputs/MANIFEST.json` for molecular identity, active
  spaces, electron sectors, tensor checksums, orbital ordering, RHF/CASCI
  references, and scientific fingerprints;
- `Source_Code/requirements.lock` for the exact 126-distribution environment;
- the six exact QASM SHA-256 values preserved in
  `expected_metrics.json`.

For Table 3, full mode freshly optimizes the three matched cases and writes a
noncanonical candidate. The validator then:

1. requires the exact ten-file candidate boundary;
2. binds every declared NPZ and QASM path to its actual SHA-256;
3. reloads the QASM and independently rederives its operation set, CX count,
   size, and depth;
4. exactly replays the serialized parameter arrays and independently
   recomputes cutting-accounting quantities;
5. checks dense state, sector leakage, and energy equivalence at 6 and 12
   qubits;
6. enforces the declared compositional-exactness boundary at 40 qubits; and
7. compares all six QASM hashes and corrected resources to the immutable
   expected contract.

No historical promoted Table 3 directory is needed for this full reproduction.
The fresh full-run result is self-contained and output-bound. The legacy
`--quick` replay remains intentionally fail-closed unless a separately audited
promoted reference is installed; it is not the competition judge path. Use
`--full`.

## Meaning of a PASS

A full PASS means that, in the exact pinned environment:

- the supplied molecular inputs and all ten RHF/CASCI references were
  independently reconstructed and verified;
- the complete frozen simulator matrix and controls executed successfully;
- all tests invoked by the full judge workflow passed;
- every scientific value or tolerance asserted by
  `validate_submission_results.py` passed;
- the fresh circuit artifacts produced the declared hashes and resources;
- the source tree did not change during certification; and
- the run imported no provider and contacted no QPU.

A PASS is evidence of scientific and artifact reproducibility in the declared
pinned CPU/simulator workflow. It is not a claim of computational
quantum advantage, device-native performance, hardware fidelity, or a new
40-qubit physical-QPU execution.

## Package contents

The ZIP root intentionally contains exactly three items:

```text
Write-Up.pdf       official cover + five technical pages + references (7 pages)
README.md          this reproduction and interpretation guide
Source_Code/       code, frozen inputs, tests, notices, and qBraid notebook
```

No credentials, API tokens, generated result directories, virtual
environments, caches, or private external datasets are included.

The upload-and-extract path above is self-contained and requires no repository
checkout. Do not substitute another source tree unless it is explicitly
verified against the final archive SHA-256 published with this submission.

Release integrity values before any generated `results/` are:

```text
Source_Code identity algorithm:
  sha256-path-hash-size-notebook-source-v3
Source_Code file count:
  82
Source_Code identity SHA-256:
  bdb34ded0371cc08d30d6f10808a27a5aacf4aef9c5262264165b3244ec30080
Write-Up.pdf SHA-256:
  59c42c5222d12f6f33225e6e1f7171be79e56a65c85006d7d10053efdac7ea13
```

## Requirements

- qBraid Lab on Linux x86-64
- **Python 3.12**
- at least **4 vCPU**
- **8 GB RAM recommended** for the complete run
- at least **13 GiB free disk** for a fresh environment
- at least **2 GiB free disk** when reusing a verified environment
- Internet access only for the first dependency installation

No GPU, QPU, provider account, API key, or quantum credits are required.

The first setup commonly takes 10–30 minutes. The complete full replay
typically takes 25–60 minutes depending on qBraid load. Wall time is recorded
but is not itself a scientific pass/fail value except for broad sanity
boundaries explicitly declared in the validator.

The validated direct versions include:

- Python 3.12 on Linux x86-64
- CUDA-QX Solvers 0.6.0
- CUDA-Q 0.14.2
- NumPy 2.5.1
- SciPy 1.18.0
- PySCF 2.14.0
- PyTorch 2.13.0
- Transformers 5.14.1
- Qiskit 2.5.0
- Qiskit Aer 0.17.2

`requirements.lock` freezes all 126 installed distributions. Its SHA-256 is:

```text
ddcd17b8cc5f1a6bb35f371768d1551c9527b7c9afbde054a088d4f3f9065caa
```

The installer removes ambient pip/Python configuration, installs exact pins
in small no-cache batches, runs `pip check`, and verifies the complete lock.
Do not substitute Python 3.11/3.13, CUDA-Q 0.15.x, or a different platform
without rerunning the entire validation contract.

## Equivalent terminal workflow

Run from the extracted `Source_Code/` directory:

```bash
cd Source_Code
export QBRAID_GQE_ENV="$(cd .. && pwd)/.qbraid_gqe_env"
PYTHON_BIN="$(command -v python3.12)" bash setup.sh
"$QBRAID_GQE_ENV/bin/python" -I -B certify_release.py --full
```

If `python3.12` is not found, launch a qBraid Python 3.12 environment and
restart. Do not continue under another Python minor version.

The notebook performs the same setup and certification commands. It creates
`.qbraid_gqe_env` beside `Source_Code/`, outside the immutable source tree.

## Full stage sequence

The judge command performs these stages without source modification:

1. Verify Python, platform, at least four logical CPUs, disk boundaries, all
   126 locked distributions, and dependency consistency. The 8 GB RAM
   recommendation is operational guidance rather than a fail-closed probe.
2. Run environment, installer, interruption, runtime, artifact-binding, and
   semantic hardening regressions.
3. Independently reconstruct and audit all ten frozen RHF/CASCI problems,
   including full determinant p-space checks.
4. Require exactly `48 passed, 0 failed` from the core scientific suite, then
   run the advanced, structured-export, cutting-budget, portability, restart,
   and exact-QPD tests.
5. Regenerate the 6-, 12-, 16-, 40-, 44-, and 48-qubit frozen-input matrix.
6. Run matched topology and warm/cold controls for seeds 17, 42, and 3047.
7. Run the fixed 126-evaluation exact-energy and QSCI residual-GQE arms.
8. Freshly generate, serialize, reload, and validate the six Table 3 QASM
   artifacts and three parameter artifacts.
9. Execute CUDA-Q `qpp-cpu`, finite-shot Aer, and exact 10,000-branch QPD
   checks.
10. Reproduce the four-qubit H₂ CUDA-QX Transformer-GQE benchmark and validate
    the immutable historical Forte-1 record without importing a provider.
11. Write the scientific summary, execution record, complete logs, and
    invocation-bound certificate.

## Principal expected scientific outputs

### Corrected Table 3 circuit evidence

All circuits use Qiskit optimization level 3, seed 3047, basis
`rz,sx,x,cx`, and diagnostic logical all-to-all connectivity.

| Case | Energy (Ha) | Phi | Generic CX / depth | Structured CX / depth | CX / depth reduction |
|---|---:|---:|---:|---:|---:|
| BeH₂-6 | −15.768337100483603 | 2.066483025125732 | 374 / 1,278 | 65 / 192 | 82.62% / 84.98% |
| BeH₂-12 | −15.770628301540164 | 3.208931011980439 | 1,325 / 2,316 | 235 / 343 | 82.26% / 85.19% |
| LiH-40 | −7.978231627630391 | 2.485077523577975 | 17,190 / 9,483 | 3,091 / 1,537 | 82.02% / 83.79% |

The corresponding exact QASM SHA-256 values are:

```text
BeH2-6  generic     7926c8c2868ae358d5a7479307c027b101a963cb3fb18e1886748a0b1cbddf78
BeH2-6  structured  7f9d5f6f1f33f81feec2c0033f0723202ed805f2990ec0a67d510f685e1c3d2d
BeH2-12 generic     27bb6c68bc93f2f073dd6d80ce3e7f5179c46ba3dbcd14f1eee30d2d2113a18d
BeH2-12 structured  73ff85a69941b10b93df011a74c364ce31c8292c52a9e078eced65540cf3b188
LiH-40  generic     201915c8dc6f24f6d3e5db5171c71b1dd3f87cf76acca6db85573a32db5d449b
LiH-40  structured  9ba5b2af3f520e5b538c4ded56b99cb59fb21510fb9c231f8fc8ef9063f72dd2
```

This contract was independently rederived from the exact emitted QASM in two
isolated replay processes. The current full workflow regenerates and checks it
again.

### Fixed-sector dimensions and optimized backbone

For `m` active spatial orbitals, the workflow evaluates the fixed sector:

```text
D = C(m, N_alpha) × C(m, N_beta).
```

LiH-40 has `m=20` and `(N_alpha,N_beta)=(1,1)`, so `D=400`. It is therefore a
40-logical-qubit, low-filling fixed-sector calculation, not allocation of a
dense `2^40` state.

| Configuration | Qubits / D | Seeds | RHF error (mHa) | Optimized Givens error (mHa) |
|---|---:|---:|---:|---:|
| BeH₂, partitioned | 6 / 9 | 3 | 1.021487 | 0.000101 |
| BeH₂, partitioned | 12 / 225 | 3 | 3.972715 | 0.649167 |
| BeH₂, all-pair | 16 / 784 | 1 | 8.351624 | 0.186663 |
| H₂O, held-out partition | 12 / 225 | 3 | 6.531403 | 3.128911 |
| N₂, adaptive depth | 16 / 3,136 | 1 | 80.252335 | 2.807957 |
| LiH, partitioned | 40 / 400 | 3 | 16.386118 | 0.000831 |
| LiH, all-pair | 40 / 400 | 1 | 16.386118 | 6.768×10⁻⁸ |
| LiH, all-pair | 44 / 484 | 1 | 18.939233 | 6.806×10⁻⁸ |
| LiH, all-pair | 48 / 576 | 1 | 19.197582 | 1.580×10⁻⁵ |

Error means `E_method − E_CASCI`; chemical accuracy is at most 1.6 mHa. H₂O
and N₂ are deliberate controls showing that sector compression does not
guarantee ansatz accuracy.

### Residual GQE and QSCI decision

The Transformer arm uses identity plus 25 iterations × 5 samples, for 126
shot-free analytic candidate evaluations at each matched width.

| Case | Frozen backbone (mHa) | Sampled nonidentity (mHa) | Guarded result (mHa) | Selection |
|---|---:|---:|---:|---|
| BeH₂-6 | 3.448×10⁻⁸ | 0.001049 | 3.448×10⁻⁸ | identity |
| BeH₂-12 | 0.651671 | 0.653761 | 0.651671 | identity |
| LiH-40 | 0.000685 | 0.001263 | 0.000685 | identity |

The identity guard prevented degradation. At LiH-40 the QSCI arm changed the
raw policy but did not pass the predeclared held-out promotion rule; exact
energy remains the selected objective.

### Other fixed checks

- BeH₂-6 finite-shot Aer: 9 commuting groups × 20,000 shots = 180,000
  simulator shots, seed 7.
- BeH₂-6 exact QPD: 10,000 algebraic branches, 4+2 fragment widths, absolute
  energy difference at most `1×10⁻¹⁰ Ha`.
- CUDA-Q `qpp-cpu`: 6- and 12-qubit state-energy convention checks agree
  within `1×10⁻¹⁰ Ha`.
- Warm-start embedded-state fidelity is at least `0.9999999999`, but complete
  cascades require `2.172×` the direct final-rung energy evaluations; no total
  acceleration is claimed.

## Historical physical-QPU record

The package preserves and validates a previous four-qubit H₂ Pauli-pool
Transformer-GQE circuit executed on IonQ Forte-1 through qBraid:

| Item | Recorded value |
|---|---|
| Logical qubits | 4 |
| Requested / returned shots | 350 / 350 |
| Observable | `X₀ ⊗ X₁ ⊗ Y₂ ⊗ Y₃` |
| Ideal / measured expectation | −0.210899 / −0.142857 |
| Exact 95% interval | [−0.247808, −0.035469] |
| Source-level CX / submitted logical depth | 58 / 108 |
| qBraid credits / provider duration | 2,830 / 277.264 s |

The full judge run reconstructs the circuit and validates the preserved count
record locally. It does not contact Forte-1. This single observable is not an
energy, fidelity, entanglement certificate, chemical-accuracy result, or
quantum-advantage demonstration.

`Source_Code/forte/FORTE_HARDWARE_RUN.ipynb` is optional. Its default
`CONTACT_QBRAID=False`; Run All does not import a provider or create a job.
Judges do not need hardware access.

## Failure and retry guidance

Every admitted certification attempt writes an invocation-specific certificate
and console log. On failure:

1. read the final error in the notebook or terminal;
2. inspect
   `certificates/<invocation-id>/reproduce_console.log`;
3. if the child progressed far enough to create a scientific run, inspect the
   named stage `*.log` in that timestamped run directory;
4. fix the reported environment or resource condition; and
5. rerun the complete full command.

Accept a successful current invocation only when:

```text
latest_certificate.json.status == "PASS"
latest_run.json.status == "PASS"
latest_certificate.json.invocation_id == latest_run.json.invocation_id
<pointed-current-run>/reproduction_summary.json.status == "PASS"
<pointed-current-run>/execution_record.json.status == "PASS"
```

After a FAILED or INTERRUPTED certificate, ignore any older PASS still named by
`latest_run.json` and use the current certificate plus its console log. If
setup fails before a run directory is created, use the final setup error.
If the installed environment is incomplete or mismatched, choose a new
dedicated `QBRAID_GQE_ENV` path and rerun `setup.sh`; do not merge packages
into an unrelated environment.

## Scientific scope and limitations

- The 40–48-qubit values are low-filling classical fixed-sector evaluations
  over 400–576 amplitudes, not dense-state simulations or hardware runs.
- The residual Transformer did not outperform the optimized Givens backbone
  at 6, 12, or 40 qubits; the identity guard is part of the result.
- The topology search is not universal TTN optimization and was not beneficial
  for every held-out case.
- Exact QPD was executed for the six-qubit 4+2 control. The LiH 20+20 value is
  a resource projection, not a sampled 40-qubit cutting experiment.
- Generic Aer noise is not target calibrated.
- Logical all-to-all CX/depth values are not device-native resource reports or
  hardware-fidelity predictions.
- CASCI remains practical in every tested fixed sector, so no computational
  quantum advantage is claimed.

## AI-support disclosure

Generative AI tools assisted language refinement and software review. The
authors remained responsible for, reviewed, and validated the technical
contributions, formulations, experimental design, code changes, and reported
results.

Third-party licensing information is in
`Source_Code/THIRD_PARTY_NOTICES.md`.
