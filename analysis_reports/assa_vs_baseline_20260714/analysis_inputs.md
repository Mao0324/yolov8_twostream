# ASSAFusion vs baseline analysis inputs

Generated: 2026-07-14 (Asia/Shanghai)

## Controlling artifacts

- Baseline training arguments: `runs_baseline/train/args.yaml`
- Baseline validation curve: `runs_baseline/train/results.csv`
- Baseline test report: `runs_baseline/train/test_result/test.txt`
- ASSAFusion training arguments: `DroneVehicle_OBB_FusionTransfer/ASSANet_ASSAFusion_P345_H2-4-8_DynK3-FFN2_v1/args.yaml`
- ASSAFusion validation curve: `DroneVehicle_OBB_FusionTransfer/ASSANet_ASSAFusion_P345_H2-4-8_DynK3-FFN2_v1/results.csv`
- ASSAFusion test report: `DroneVehicle_OBB_FusionTransfer/ASSANet_ASSAFusion_P345_H2-4-8_DynK3-FFN2_v1/test_result/test.txt`
- Model definitions: `yaml/baseline.yaml`, `yaml/yolov8s-ASSAFusion.yaml`, `ultralytics/nn/modules/assa_fusion.py`
- Checkpoints inspected: `runs_baseline/train/weights/best.pt`, `DroneVehicle_OBB_FusionTransfer/ASSANet_ASSAFusion_P345_H2-4-8_DynK3-FFN2_v1/weights/best.pt`

Both runs used the same dataset, 100 epochs, batch 64, image size 640, SGD, seed 0, deterministic mode, and the same main augmentation/loss settings. The architecture and initialization checkpoint path were the meaningful differences.

## Recomputed aggregate results

| Metric | Baseline | ASSAFusion | Delta |
| --- | ---: | ---: | ---: |
| Best validation mAP50 | 0.85150 (epoch 94) | 0.85970 (epoch 88) | +0.00820 |
| Best validation mAP50-95 | 0.70117 (epoch 96) | 0.71129 (epoch 90) | +0.01012 |
| Test mAP50 | 0.812 | 0.820 | +0.008 |
| Test mAP50-95 | 0.670 | 0.679 | +0.009 |
| Test precision | 0.789 | 0.785 | -0.004 |
| Test recall | 0.782 | 0.798 | +0.016 |
| Inference time, batch 16 | 3.9 ms/image | 7.8 ms/image | +100% |
| Parameters from current checkpoints | 14,734,722 | 21,289,898 | +6,555,176 (+44.5%) |
| GFLOPs at 640 | 37.0351 | 49.4951 | +12.4600 (+33.6%) |

Parameter counts and GFLOPs were recomputed from the current checkpoints with the repo-local Ultralytics code. The baseline test console contains a slightly smaller fused summary; the current-checkpoint counts above control the structural comparison because both were computed by the same method in this analysis.

## Test class changes

| Class | Baseline mAP50 | ASSAFusion mAP50 | Delta, pp | Baseline mAP50-95 | ASSAFusion mAP50-95 | Delta, pp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| car | 0.985 | 0.985 | 0.0 | 0.836 | 0.838 | +0.2 |
| truck | 0.817 | 0.832 | +1.5 | 0.646 | 0.662 | +1.6 |
| bus | 0.959 | 0.962 | +0.3 | 0.815 | 0.818 | +0.3 |
| van | 0.621 | 0.633 | +1.2 | 0.509 | 0.522 | +1.3 |
| freight_car | 0.678 | 0.689 | +1.1 | 0.542 | 0.554 | +1.2 |

The all-class mAP metrics are macro averages across the five classes. Training gradients nevertheless see a highly imbalanced instance distribution: car has 137,146 test instances, versus 4,282-8,657 for each other class.

## ASSAFusion parameter and learned-scale inspection

| Scale | Channels | Full module params | FFN params | Attention-side params after removing FFN |
| --- | ---: | ---: | ---: | ---: |
| P3 | 128 | 346,632 | 205,824 | 140,294 |
| P4 | 256 | 1,283,084 | 804,864 | 477,194 |
| P5 | 512 | 4,925,460 | 3,182,592 | 1,740,818 |
| Total | - | 6,555,176 | 4,193,280 | 2,361,896 |

The FFNs consume 64.0% of all ASSAFusion parameters. In the trained checkpoint, attention residual scales grew from the 0.1 initialization to approximately 1.11-1.57, while FFN residual scales fell to approximately 0.037-0.042. This supports testing removal or strong bottlenecking of the FFN first; it does not by itself prove the FFN has zero accuracy contribution.

The dynamic K3 implementation runs nine depth-wise weight-generator convolutions and nine shifted multiply-accumulate paths per modality and fusion scale. This produces many small operations and Python-loop dispatches, explaining why latency grows more than the static GFLOP ratio.

## Parameter-only projections for ablations

These are arithmetic projections from the current module decomposition, not trained results.

| Candidate | Projected params | Delta vs baseline |
| --- | ---: | ---: |
| P3/P4/P5, dynamic K3, no FFN | 17,096,618 | +16.0% |
| P4 only, full current block | 16,017,806 | +8.7% |
| P4 only, dynamic K3, no FFN | 15,211,916 | +3.2% |
| P3+P4, dynamic K3, no FFN | 15,352,210 | +4.2% |
| P4 only, static DW K3, no FFN | about 15,138,188 | +2.7% |
| P3+P4, static DW K3, no FFN | about 15,241,618 | +3.4% |

Static-DW projections replace each nine-generator dynamic depth-wise operator with one static depth-wise 3x3 operator while retaining 1x1 projections and attention. Runtime must be benchmarked; parameter arithmetic alone cannot establish latency.
