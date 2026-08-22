# P2D bottleneck analysis notes

Audience: technical.

Required report roles: title; technical summary; key findings with visual evidence; scope/data/metric definitions; methodology; limitations/uncertainty/robustness; recommended next steps; further questions. All roles are present in `artifact.json`.

## Reproducibility

- Scope: directories matching `DroneVehicle_OBB_FusionTransfer/P2D-*`.
- Included test runs: P2D-001, 002, 003, 004, 006, 007, 008, 009, 010.
- Excluded: P2D-005 (91 training rows, no `test_result/test.txt`); P2D-011 to 013 (2-3 training rows, no test result).
- Metric extraction: parsed the `all`, `car`, `truck`, `bus`, `van`, and `freight_car` rows from each test file.
- Validation context: maximum `metrics/mAP50(B)` and `metrics/mAP50-95(B)` from each included `results.csv`.
- Historical context: scanned every non-P2D `*/test_result/test.txt` in the same experiment root and ranked the `all` row.
- No confidence intervals were computed because per-image predictions and repeated seeds were not available.

## Chart map

1. Overall plateau: ordered multi-series line; fields `run`, `metric`, `value`; supports the narrow 0.005 range and lack of cumulative improvement; blue/gold two-root palette; portable HTML report.
2. Class responsiveness: single-series bar; fields `class`, `range_map50`; supports the distinction between saturated car/bus and still-responsive van/freight_car; single-root palette; portable HTML report.

The ordered line uses experiment number as a discrete design sequence, not as continuous time. The class range chart measures observed sensitivity across configurations and is not presented as guaranteed future headroom.
