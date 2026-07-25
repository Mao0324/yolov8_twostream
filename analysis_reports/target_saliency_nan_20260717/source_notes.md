# TargetSaliency NaN diagnostic evidence

Generated: 2026-07-17 20:43 Asia/Shanghai

## Verified result

- Displayed epochs 1-6 had finite train and validation box/cls/dfl/saliency losses.
- From displayed epoch 7 onward, all eight loss fields became NaN together.
- Precision, recall, mAP50 and mAP50-95 repeated the displayed-epoch-6 values after divergence. These are cached validator values, not fresh valid metrics.
- `best.pt` stored checkpoint epoch 3 (displayed epoch 4) and all 24,565,192 floating EMA values were finite. Its 24,525,304 floating optimizer-state values were also finite.
- `last.pt` stored checkpoint epoch 16 (displayed epoch 17). Of 24,565,192 floating EMA values, 24,565,176 were NaN. All 24,525,304 floating SGD momentum values were NaN.
- The migrated start checkpoint was finite and the run `args.yaml` used the migrated `.pt` directly, so this event is unrelated to the earlier DDP checkpoint-loading issue.
- Existing DarkAct V2, V3 and NoStaticMAA comparison runs completed 100 epochs with no non-finite loss values.

## Strongest root-cause inference

`_StaticTargetSaliencyBranch.forward` performs `query.flatten(2) @ saliency.flatten(2).T` inside global AMP autocast. At P3 the reduction length is 80x80=6400, but the result is divided by `sqrt(active_channels)=sqrt(32)` and is not promoted to FP32. The safe-checkpoint probe already measured P3 scaled attention magnitudes of 88.19 (RGB) and 56.16 (IR), with Softmax maxima of 1 and minima of 0. This means severe saturation was present by displayed epoch 4.

The inherited PaperLAF attention explicitly disables autocast and converts Q/K/V to FP32 because its code comments identify exactly the same P3 `N=6400` FP16 overflow risk. The new target-saliency attention omitted that guard. This is the highest-confidence explanation for the abrupt epoch-7 transition, but the exact first failing batch cannot be proven because no per-batch finite-value trace or displayed-epoch-6 checkpoint was saved.

The auxiliary BCE+Dice formula is less likely to be the first source: logits are converted to FP32; empty targets return finite zero masks; foreground has a `1e-6` lower bound; positive weighting is capped at 20; and the Dice denominator includes `+1`.

## Validator cache explanation

`ultralytics/models/yolo/detect/val.py:get_stats()` only calls `self.metrics.process(**stats)` when `stats["tp"].any()` is true. After the NaN model produced no true positives, the method returned the previous `self.metrics.results_dict`, which explains the repeated epoch-6 P/R/mAP values.

## Source availability caveat

The original TargetSaliency run directory was still updating at displayed epoch 18 around 20:36. It disappeared from the shared workspace around 20:39 after the evidence above had been collected. The report therefore preserves a bounded evidence snapshot, but the original CSV and checkpoints can no longer be reopened at handoff time.

## Chart map

- Section: epoch-level failure pattern
- Question: when did the run stop producing finite losses?
- Family/type: trend / single-series line
- Fields: displayed epoch, count of finite train+validation loss fields out of eight
- Claim: the run changes abruptly from 8/8 finite fields through epoch 6 to 0/8 from epoch 7
- Palette: single blue root; no categorical color legend
- Delivery: `report.html`, with semantic table fallback from the same artifact dataset

