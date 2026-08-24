# P2D-013N trade-off evidence notes

## Comparison contract

- Baseline test metrics: `runs_baseline_n/YOLOv8n_TwoStream_Baseline_v1/test_result/test.txt`.
- Module test metrics: `DroneVehicle_OBB_FusionTransfer/P2D-013N_IRPrompt-AsymIdentityGDER-P4-NoStaticMAA_PostC2f_v13_scalen/test_result/test.txt`.
- Both tests use the same `data/dronevehicle.yaml`, test split, 8,980 images, 159,614 instances, image size 640, and batch size 16.
- Training configuration parity was checked by diffing both `args.yaml` files. Only model path, project/name, and save directory differ.
- Both training histories contain exactly 100 epochs with no missing metric rows. Baseline best validation mAP50-95 is 0.70896 at epoch 87; module best is 0.71713 at epoch 99.

## Confusion matrices

Counts in `confusion_focus.csv` were transcribed from the raw test confusion-matrix PNGs. Rows are predicted labels and columns are true labels; the rate denominator is the true-class instance count. The focused entries reconcile to the class-column totals shown in `test.txt`.

## IRPrompt and GDER implementation

- Architecture and stage placement: `yaml/yolov8s-P2Det-IRPrompt-AsymIdentityGDER-P4-NoStaticMAA-PostC2f-v13_scalen.yaml`.
- Prompt head, additive embedding, asymmetric modality expert, attention expert, gate, and residual feedback: `ultralytics/nn/modules/p2det_prompt_gder.py`.
- Class-agnostic rotated soft-centerness teacher: `ultralytics/models/yolo/obb/prompt_utils.py`.
- Prompt loss and training telemetry: `ultralytics/models/yolo/obb/p2det_train.py`.
- The IR prompt is supervised only at P3/P4 with stage weights 1.0/0.5, loss gain 0.025, and 10-epoch warmup. Its target map uses box geometry only and does not read class labels.
- GDER exists only at P4. It adds a raw-RGB/IR-prior modality expert and a prompt-free fused-feature CBAM expert through a channel-wise two-way softmax gate.

## Learned-behavior probe

`probe_module_effects.py` loads the tested best checkpoint and runs eight fixed paired test images (`00001`–`00008`) through the model. `module_probe.json` records prompt distributions, additive embedding RMS relative to the incoming IR feature, GDER gate statistics, and weighted expert-output RMS relative to the fused P4 feature. This is a deterministic mechanism probe, not a population estimate.

## Scale-s ablations

`scale_s_ablation.csv` uses the same test split and four scale-s runs: P2D-011 prompt-only, P2D-013 full P4 GDER, P2D-014 modality expert only, and P2D-015 attention expert only. These isolate GDER branches at scale s, not at scale n, so they are supporting evidence only.

## Object geometry

`test_box_geometry.csv` was calculated from all 8,980 test label files. Normalized polygon vertices were scaled to 640×512 and converted to minimum-area rectangles. P4 cell counts divide pixel dimensions by stride 16.

## Limitations

- Each scale-n result is one seed. No paired per-image AP bootstrap or repeated-seed variance is available.
- The two scale-n architectures differ in more than IRPrompt/GDER: P2D-013N replaces the baseline ADD fusion carrier with post-C2f RIFusion/LAF feedback blocks. Therefore the baseline-to-module delta cannot causally identify either named module.
- Test logs are rounded to three decimals. Small deltas should be treated as directional.
- Runtime speed is from the saved test logs and is not a controlled hardware benchmark.
