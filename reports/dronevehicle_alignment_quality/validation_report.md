# DroneVehicle RGB/IR alignment audit validation

## Decision

**Share with caveats.** The evidence supports the engineering conclusion that the prepared DroneVehicle pairs are generally well registered but not pixel-perfect. It does not support a claim that every vehicle has an independently measured RGB/IR center error.

## Data quality checks

- 28,439/28,439 RGB, IR, and label stems form exact triplets.
- All paired images are readable and have matching 640×512 dimensions.
- All 500,515 non-empty polygon rows parse successfully. Boundary-clipped targets legitimately contain 3, 5, or 6 vertices; they are not malformed OBB labels.
- The same label file is used for both modalities. This confirms the supervision convention, not independent geometric truth.

## Calculation checks

- The registration sample is deterministic and evenly spaced: train 200, val 50, test 150.
- Translation estimates use normalized gradient magnitudes, phase correlation initialization, and translation-only ECC.
- Trusted estimates require ECC correlation ≥0.50 and shift ≤10 px; 293/400 pass.
- The median shift (1.741 px), P90 (6.857 px), and within-threshold shares were recomputed from `alignment_samples.csv` by the executed notebook.
- Object scale was recomputed from each label polygon's minimum-area rotated rectangle. Median short side is 25 px; median registration shift is 6.97% of that size.
- A separate affine audit on every tenth sampled pair yields 30 trusted estimates, with median center shift 1.709 px, median rotation 0.094°, scale near 1, and P90 maximum corner shift 10.451 px.

## Visual validation

- `low_shift_montage.jpg` shows close overlap of stable scene edges and vehicle outlines.
- `high_shift_montage.jpg` shows directionally consistent red/green fringes across roads, poles, and vehicles, supporting that the 8–10 px tail is not solely thermal appearance variation.
- Visual overlays are diagnostic rather than a substitute for independent per-object dual-modality annotation.

## Known limitations

- 107/400 cross-modal pairs fail the trust filter. They are excluded from displacement quantiles and should be treated as unknown, not automatically misregistered.
- Scene-level ECC may be affected by thermal blooming, shadows, vegetation, moving vehicles, and capture timing.
- One global translation or affine transform cannot fully characterize local parallax from physically separated RGB and IR sensors.
- The audit does not contain independently annotated RGB and IR vehicle centers, so it cannot produce a ground-truth per-vehicle center-error distribution.

## Artifact QA

- Notebook executed successfully: 3/3 code cells completed, with no error outputs.
- Portable report validation and packaging passed.
- Verification was structural only because no Chromium executable is installed; interactive source dialogs and viewport rendering were not browser-verified.
