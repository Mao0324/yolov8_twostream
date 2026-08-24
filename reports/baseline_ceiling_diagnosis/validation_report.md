# Validation Report

## Overall Assessment: Share with caveats

The committed evidence supports the descriptive claim that the best observed pretrained variant is 0.8 mAP50 percentage points above the pretrained baseline. It does not yet support a statistically stable superiority claim.

## Methodology Review

- The requested population was enforced: `mAP50 > 0.83`, with checkpoint-direct initialization and `pretrained: true` verified from `args.yaml`.
- All retained runs use the same DroneVehicle test population: 8,980 images, 159,614 instances, split `test`, image size 640.
- Fourteen decision-relevant training fields reconcile to one signature across all 23 retained runs.
- Git blob identities were used to remove inherited cross-branch duplicates.

## Issues Found

1. **Severity: High — no repeated seeds.** Every comparable run uses seed 0, so run-to-run variance is unknown.
2. **Severity: High — test-set selection.** Thirty-six unique committed test artifacts show that the test split has been used for repeated architecture comparison; selecting the maximum is therefore optimistic.
3. **Severity: Medium — no per-image predictions.** Aggregate AP files cannot support paired bootstrap confidence intervals or image-level error attribution.
4. **Severity: Low — two experiment branches have no additional committed test output.** `exp/cfgpnet-lite` and `exp/protohgfnet-lite` add no new `test_result/test.txt` beyond inherited DarkACT artifacts.

## Calculation Spot-Checks

- Baseline mAP50: **verified**, 0.833 from `main:runs_baseline/train2/test_result/test.txt`.
- Best observed mAP50: **verified**, 0.841; delta = `(0.841 - 0.833) × 100 = 0.8pp`.
- Best observed mAP50-95: **verified**, 0.708; delta = 0.7pp.
- Variant median mAP50: **verified**, 0.837 across 22 non-baseline comparable runs.
- Baseline validation best epoch: **verified**, epoch 61 using repository fitness `0.1×mAP50 + 0.9×mAP50-95`.
- Class contribution: **verified**, the highest-mAP50 model gains 1.1pp on truck and 2.2pp on freight_car, with no AP50 gain on car or bus.

## Visualization Review

- The model chart uses signed percentage-point deltas from zero, avoiding a misleading truncated absolute mAP axis.
- The class chart starts from zero and uses grouped bars for baseline versus best observed.
- Both charts state the cohort and single-seed limitation next to the evidence.

## Suggested Improvements

1. Repeat baseline and the two strongest candidate architectures with at least seeds 0, 1, and 2.
2. Save per-image predictions and run paired bootstrap confidence intervals.
3. Use validation only for architecture and hyperparameter choices; reserve a hidden holdout for one final evaluation.
4. If tuning learning rate, give baseline and module models equal search budgets and select on validation performance.

## Required Caveats for Stakeholders

- The reported 0.8pp is the maximum observed single-seed difference, not an estimated mean effect.
- Repeated use of the test split makes the selected maximum potentially optimistic.
- The aggregate improvement is concentrated in truck and freight_car rather than being uniform across classes.
