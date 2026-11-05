# Apple Silicon Germinal Platform-Validation Closure Report

Date: 2026-11-05  
Decision: **CLOSED-PASS**  
Workstream: Native Apple Silicon Germinal platform validation  
Authoritative repeatability evidence: Phase 2C Run 1 and Run 2  
Pull request: <https://github.com/russellchang54/germinal/pull/1>

## 1. Closure decision

Phase 2C is formally accepted as **PASS**, and the Apple Silicon
platform-validation workstream is closed. The accepted result establishes that
the frozen hybrid runtime faithfully and repeatably executes the literal
Germinal production workflow. It does **not** establish antibody-design success.

Phase 2C produced zero accepted candidates and five scientifically rejected
candidates. That is a valid platform-validation result because execution,
filtering, candidate accounting, and repeatability all passed without changing
any threshold or scientific rule.

## 2. Frozen validated configuration

| Component | Frozen runtime | Placement |
|---|---|---|
| AF2 / ColabDesign | JAX/JAXlib `0.10.2`, `jax-mps==0.10.9` | Apple MPS |
| AF3 | pinned revision `8f8ac383afba995f5953182d39a560ffb8bbbfd3`, `jax-mps==0.10.9` | Apple MPS |
| AbMPNN | JAX/JAXlib `0.10.2`, `jax-mps==0.10.11` | Apple MPS |
| PyRosetta, DSSP, DAlphaBall | existing validated native environment | ARM64 CPU |

Runtime process isolation is mandatory. AF2, AF3, and AbMPNN use separate
interpreter prefixes. No live JAX array, model state, or MPS runtime object
crosses a component boundary. AbMPNN rejects non-finite logits or scores and
retries the same seeded computation in a fresh interpreter, with a maximum of
five trials and fail-closed behavior.

The accepted source snapshot has these exact production hashes:

| Path | SHA-256 |
|---|---|
| `run_germinal.py` | `83ac56a1b77da7452ce2f4a7d7c431efe323d7c2bdc12eba6cb1abbac235d0ab` |
| `germinal/design/design.py` | `e2edb4dd4a612a5bda79090cf925f7b7b2b5fd222c7c5a273679d3cd605fe8b5` |
| `germinal/filters/af3.py` | `8f415c690323564c316ef42ea31b681f20938e88424a8fdb272a800bfdbef129` |
| `germinal/filters/filter_utils.py` | `4077ee71a577b29641f633231de73d5dd934665523ac29e21c3063bf6e325ea1` |
| `germinal/filters/pyrosetta_utils.py` | `8e727b7877d30c0cd061a30ab7b444acbe01f4079dcf0a3b8103bb2cb00cdb3d` |
| `germinal/filters/redesign.py` | `e817fccf56b0f693248bb1ff39003c980ca906133ee3fdfabbad8439bd1fa399` |
| `germinal/utils/utils.py` | `1122611395c03971f8355fa30fdda7689e89607a488739bb305974e2ec8a6019` |
| `scripts/af2_mps_compat.py` | `815a47c79ed70be7dd7af217790ec98dc06fa567a06159592ac4f409922d3780` |
| `scripts/run_af3_mps.py` | `d4d76013b634a05da1073d377833a82b9bebae7fd4a203f2e4f108b9fa21e0bc` |
| `scripts/run_abmpnn_worker.py` | `5c8ff9df6892a8651c13c4cfc83f6fa304ba1af4b3d6991911a3ed83805d52a5` |

The accepted execution began from Git commit
`1e1c1a5b79884ae45abae030c9df90d9423a990a` with the validated working-tree
snapshot identified by the hashes above. That accepted production source was
later preserved in validation branch commit
`a68e9b121eeeea9530417029a7505b89309d52b7`. Closure documentation is a
non-runtime addition. All production source hashes were rechecked at closure and
remain identical to the Phase 2C preflight snapshot.

## 3. Scientific baseline

- Trajectory seed: `1791095845`
- AF2 schedule: 60 logits, 35 softmax, 10 PSSM steps
- Models: five AF2 multimer v3 checkpoints, sampled during design
- Design optimizer: SGD, learning rate `0.1`, PCGrad, normalized gradients,
  linear annealing to minimum scale `0.25`
- AbMPNN: temperature `0.2`, batch size `40`, maximum five selected sequences,
  interface residues fixed
- AF3: five diffusion samples, ten recycles, worst-sample selection, full
  production MSA/template pipeline
- Relaxation: PyRosetta FastRelax with propagated trajectory seed
- Initial and final VHH/IL3 thresholds: unchanged
- Inputs and checkpoints: hashes recorded in
  `phase2c_platform_baseline_manifest.json`

No optimization schedule, loss equation, weight, threshold, filter definition,
checkpoint, input structure, seed, sampling distribution, or candidate-selection
rule was changed to obtain the PASS result.

## 4. Authoritative production repeatability evidence

The two Phase 2C executions are the only authoritative full-production
repeatability evidence for this closure.

| Run | Result | Wall time | AF2 | AbMPNN | AF3 | Final filters | PDB integrity |
|---|---|---:|---:|---:|---:|---|---|
| Run 1 | PASS | 6h 38m 38s | 105 records | 5 finite, diverse | 6 production invocations | 0 accepted, 5 rejected | 18 valid |
| Run 2 | PASS | 6h 49m 0s | 105 records | 5 finite, diverse | 6 production invocations | 0 accepted, 5 rejected | 18 valid |

Run 1 and Run 2 exactly matched:

- all 105 AF2 iteration records, final sequence, and AF2 boundary PDB;
- pLDDT `0.8867384791374207`, iPTM `0.742160439491272`, and iPAE
  `0.21023310720920563`;
- the AbMPNN input PDB, normalized settings, sequence set, scores, and retry
  outcome;
- AF3 inputs, production-search-derived data, target MSA, initial prediction,
  initial relaxed structure, and all invocation inputs;
- PyRosetta metrics, every final filter decision, candidate count, and rejection
  reasons;
- all 18 production PDB hashes, finite coordinates, and expected chain
  composition A=112 and B=131.

The comparison used exact equality for scientific values. Only documented
paths, timing metadata, design-time fields, and non-scientific CIF timestamps
were normalized. The frozen comparison produced 24 passing checks and no failed
checks.

## 5. Production database reference

Phase 2C executed against the qualified production database at
`/Volumes/T9/database`. That path and its integrity evidence remain the
authoritative historical execution reference.

After Phase 2C, the database content was copied to
`/Volumes/990Pro/database` and the original external volume was retired. The
active APFS copy contains 195,899 real files, including 195,859 mmCIF files,
with zero AppleDouble sidecars. A full content-checksum comparison completed
with exit code 0 and no differences. Required database files have identical
sizes, and the four copied readiness/integrity manifests retain their original
SHA-256 values.

Therefore:

- historical Phase 2C database evidence: `/Volumes/T9/database`;
- active frozen content reference: `/Volumes/990Pro/database`;
- migration status: content-equivalent, but not independently validated by a
  new full production run.

The current macOS configuration records the active path. This is an external
database-location update only; it did not alter database content or any
scientific setting.

## 6. Evidence preservation

All Phase 2B and Phase 2C reports, summaries, execution manifests, run JSON
files, comparison results, failed-run records, checksum manifests, raw logs,
boundary snapshots, intermediate artifacts, and final PDBs remain preserved
under `docs/porting/af3_mps_probe/`. The consolidated progression is in
`phase2b3_phase2c_evidence_index.md`.

The frozen machine-readable baseline is
`phase2c_platform_baseline_manifest.json`. Its closure checksums are recorded in
`phase2c_closure_checksums.sha256`. Existing phase evidence was not edited.

## 7. Historical failures and corrective gates

The workstream intentionally preserves failures because they define the final
platform boundary:

1. Phase 2B.3 found and avoided a defective MPS sort VJP with equivalent
   `jax.lax.top_k`.
2. Phase 2B.4 passed the full AF2 design function but not the literal entry
   point because PyRosetta was unavailable.
3. Phase 2B.5 passed with miniature AF3 databases but could not qualify
   production repeatability.
4. Phase 2B.6 found independent AbMPNN and PyRosetta seed-propagation defects.
5. Phase 2B.7 exposed non-finite poly-alanine AbMPNN output during full
   production execution.
6. Phase 2B.8 isolated the boundary-state cause and added fail-closed retry.
7. Phase 2B.9 proved the retry alone was insufficient when Run 2 rejected five
   non-finite trials.
8. Phase 2B.10 localized the hazard to the JAX-MPS/MLX fused-control-flow class
   and introduced `0.10.11` only for AbMPNN.
9. Phase 2B.11 proved global `0.10.11` would alter the accepted AF2 trajectory,
   so AF2 and AF3 remained on `0.10.9`.
10. Phase 2B.12 validated component isolation before authorizing Phase 2C.
11. Phase 2C completed the final two-run production repeatability gate.

## 8. Unresolved limitations

- The exact Metal operator, kernel, or resource-lifetime defect remains
  unidentified. The platform uses a validated isolation and fail-closed
  mitigation, not an upstream root-cause fix.
- `jax-mps==0.10.11` is approved only for AbMPNN. Global AF2 use is rejected
  because it reproducibly reaches iPTM `0.649` on the frozen baseline.
- The full-content database migration to `/Volumes/990Pro/database` is verified
  by checksum comparison but not by a new production execution.
- Phase 2C validated one frozen VHH/IL3 trajectory and runtime. It does not
  generalize automatically to other targets, inputs, modes, seeds, or platform
  versions.
- Zero accepted antibody candidates is a valid unchanged-filter outcome and
  must not be represented as therapeutic or design success.

## 9. Closure restrictions

After this closure:

1. Do not modify the accepted source snapshot, scientific settings, seed,
   checkpoints, original inputs, schedules, sampling settings, thresholds, or
   filter definitions without explicit authorization.
2. Do not launch another full production run under this closed workstream.
3. Do not pursue a new seed or replacement trajectory under this closure.
4. Do not globally replace either accepted JAX-MPS version.
5. Do not reinterpret Phase 2C as antibody-design success.
6. Any new scientific or runtime work requires a fresh authorization, preflight,
   source/config manifest, and explicit success criteria.
