# Apple Silicon Germinal Platform-Validation Evidence Index

Closure date: 2026-11-05  
Closure state: **CLOSED-PASS**  
Accepted trajectory seed: `1791095845`  
Authoritative full-production evidence: Phase 2C Run 1 and Run 2

This index covers Phase 2B.3 through Phase 2C only. It preserves both passing
gates and historical failures. A PARTIAL or FAIL record is evidence of a real
boundary failure, not a reason to discard the record.

## Gate progression

| Phase | Decision | Result and corrective meaning | Primary evidence |
|---|---:|---|---|
| 2B.3 | PASS | The MPS sort VJP was defective. Replacing only the beta-strand reduction with `jax.lax.top_k` preserved gradients; four fresh MPS processes reproduced optimization state, gradients, PCGrad output, and optimizer updates byte-for-byte. | `phase2b3_report.md`, `phase2b3_summary.json` |
| 2B.4 | PARTIAL | `germinal_design()` completed the full 60/35/10 AF2 schedule with bitwise-reproducible MPS traces. The literal entry point remained blocked by the absent PyRosetta dependency. | `phase2b4_report.md`, `phase2b4_summary.json`, `phase2b4_assets.json` |
| 2B.5 | PARTIAL | Native PyRosetta was restored and the literal entry point completed with miniature AF3 databases. Production databases were absent and Run 1 versus Run 2 was not same-code repeatability evidence. | `phase2b5_report.md`, `phase2b5_summary.json`, `phase2b5_full_pipeline_run1.json`, `phase2b5_full_pipeline_run2.json` |
| 2B.6 | PASS | Two same-code miniature-database runs matched 105 AF2 records, AF2/AF3/AbMPNN artifacts, 85 comparable PyRosetta/filter fields, and all filter decisions. Two independent seed-propagation defects were corrected: AbMPNN received `seed=None`, and PyRosetta FastRelax was unseeded. | `phase2b6_report.md`, `phase2b6_summary.json`, `phase2b6_run1.json`, `phase2b6_run2.json`, `phase2b6_repeatability.json` |
| 2B.6 | PASS | Existing AF3 production database was qualified with integrity checks, a full data-pipeline smoke test, and native MPS inference. The authoritative evidence used `/Volumes/T9/database`, later copied and content-verified to `/Volumes/990Pro/database`. | `phase2b6_database_manifest.json`, `phase2b6_database_integrity.json`, `phase2b6_database_readiness.json`, `phase2b6_database_smoke.json` |
| 2B.7 | PARTIAL | Both full-production runs completed, used the production database, and otherwise repeated, but AbMPNN emitted one poly-alanine sequence with NaN score instead of five valid candidates. The unchanged filters rejected it. | `phase2b7_report.md`, `phase2b7_summary.json`, `phase2b7_run1.json`, `phase2b7_run2.json`, `phase2b7_repeatability.json` |
| 2B.8 | PASS | Root cause was intermittent non-finite JAX-MPS raw logits in the AbMPNN worker after AF2 state existed. The correction used a dedicated interpreter, rejected non-finite logits/scores, and retried the unchanged seeded computation in a fresh process. Patch SHA-256: `d164b8e9e11a1743cde560e7267e09b457f6a1530e3a4b33809563b57666ce45`. | `phase2b8_report.md`, `phase2b8_summary.json`, `phase2b8_validation.json`, `phase2b8_exact_patch.diff` |
| 2B.9 | FAIL | Run 1 completed and produced five finite candidates, but Run 2 rejected five consecutive non-finite AbMPNN trials and exited nonzero. This proved that the mitigation reduced but did not eliminate the boundary hazard. | `phase2b9_report.md`, `phase2b9_summary.json`, `phase2b9_run1.json`, `phase2b9_run2.json`, `phase2b9_repeatability.json` |
| 2B.10 | PARTIAL | The post-AF3 failure class was localized to the upstream JAX-MPS/MLX fused-control-flow hazard. AbMPNN's isolated environment moved to `jax-mps==0.10.11`; JAX/JAXlib and scientific settings remained unchanged. Global replacement was not approved. | `phase2b10_report.md`, `phase2b10_summary.json`, `phase2b10_validation.json`, `phase2b10_exact_patch.diff` |
| 2B.11 | PASS | JAX-MPS `0.10.11` was bitwise reproducible across fresh AF2 processes, but its reproducible AF2 trajectory diverged from the accepted 0.10.9 baseline and ended at iPTM `0.649`, below the unchanged `0.68` threshold. This rejected global 0.10.11 use for AF2 without claiming a specific Metal operator defect. | `phase2b11_report.md`, `phase2b11_summary.json`, `phase2b11_comparison.json` |
| 2B.12 | PASS | Component-isolated runtimes passed all gates: AF2 and AF3 remained on `jax-mps==0.10.9`; AbMPNN used `0.10.11`; PyRosetta, DSSP, and DAlphaBall remained native ARM64 CPU. No full pipeline was run. | `phase2b12_report.md`, `phase2b12_summary.json`, `phase2b12_release_decision.json`, `phase2b12_baseline_integrity.json`, `phase2b12_backend_provenance.json` |
| 2C | PASS | Two sequential literal production executions completed every stage against `/Volumes/T9/database`. All 24 repeatability checks passed. Zero accepted and five rejected candidates was the valid unchanged-filter outcome. | `phase2c_report.md`, `phase2c_summary.json`, `phase2c_run1.json`, `phase2c_run2.json`, `phase2c_repeatability_comparison.json`, `phase2c_execution_manifest.json`, `phase2c_checksums.sha256` |

## Authoritative Phase 2C record

- Run 1: exit 0, 6h 38m 38s, 105 AF2 records, five finite AbMPNN candidates,
  six production AF3 invocations, 18 valid PDBs, zero accepted, five rejected.
- Run 2: exit 0, 6h 49m 0s, 105 AF2 records, five finite AbMPNN candidates,
  six production AF3 invocations, 18 valid PDBs, zero accepted, five rejected.
- All 24 repeatability checks passed with exact scientific comparison.
- The one failed pre-production Run 2 startup attempt remains preserved and is
  explicitly excluded because no scientific stage began.
- Primary result: `phase2c_summary.json`.
- Primary comparison: `phase2c_repeatability_comparison.json`.
- Execution evidence: `results/phase2c/run1/`, `results/phase2c/run2/`, and
  `results/phase2c/run1.log`, `results/phase2c/run2.log`.

## Preserved corrective changes

1. Replace defective MPS sort reduction with equivalent `jax.lax.top_k`.
2. Restore native ARM64 PyRosetta and downstream execution.
3. Correct AbMPNN seed propagation.
4. Correct PyRosetta FastRelax seed propagation.
5. Qualify the full production AF3 database.
6. Add fail-closed AbMPNN logits/score validation with a dedicated interpreter
   and bounded same-seed fresh-process retry.
7. Approve stage-scoped JAX-MPS versions rather than globally migrating AF2.
8. Preserve the literal production entry point, schedules, thresholds, filters,
   checkpoints, input PDBs, and seed.

## Closure limitations

- The exact Metal operator or resource-lifetime defect remains unidentified.
- JAX-MPS `0.10.11` is approved only for AbMPNN, not globally.
- Phase 2C validated one frozen VHH/IL3 platform trajectory, not all targets or
  antibody-design modes.
- Zero accepted antibody candidates is a legitimate scientific result and does
  not indicate an execution failure.
- `/Volumes/T9/database` was later retired. `/Volumes/990Pro/database` is the
  active content-equivalent database location; no new production run was used
  to validate the migrated path.
- Reuse of the frozen baseline for new targets, seeds, or scientific changes
  requires explicit authorization and a new preflight gate.
