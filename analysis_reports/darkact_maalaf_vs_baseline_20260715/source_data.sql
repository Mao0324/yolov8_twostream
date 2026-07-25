-- DuckDB-compatible literal queries that reproduce the bounded datasets embedded in artifact.json.

-- headline
SELECT
    0.673 AS variant_map,
    0.669562 AS baseline_map,
    0.3438 AS map_delta_pp,
    5.5 AS variant_latency,
    3.8523 AS baseline_latency,
    42.77 AS latency_delta_pct,
    15.702371 AS variant_params_m,
    14.734722 AS baseline_params_m,
    6.57 AS params_delta_pct;

-- class_delta
SELECT * FROM (VALUES
    ('truck', 8657, 0.64594, 0.654, 0.806),
    ('bus', 4467, 0.81501, 0.818, 0.299),
    ('van', 4282, 0.50901, 0.511, 0.199),
    ('car', 137146, 0.83585, 0.837, 0.115),
    ('freight_car', 5062, 0.542, 0.543, 0.100)
) AS t(class, instances, baseline, variant, delta_pp);

-- stage_diagnostics
SELECT * FROM (VALUES
    ('P3', 0.03128, -0.55957, 1.0132, 0.5521, 0.6845, 0.9202, -0.004696, 1),
    ('P4', -0.00872, -0.06317, 0.9955, 0.9625, 0.7701, 0.8640, -0.003702, 2),
    ('P5', -0.000249, 0.04816, 0.9999, 1.0215, 0.7915, 0.7940, 0.004135, 3)
) AS t(stage, beta_rgb, beta_ir, maa_gain_rgb, maa_gain_ir, laf_rgb_weight, brightness_corr, cross_scale, stage_order);

-- implementation_gap
SELECT * FROM (VALUES
    ('MAA saliency', 'Adjacent-frame temporal difference + normalization + sine mapping + MLP', 'Single-frame |F-AvgPool3(F)| local contrast mask', 'Motion semantics are entirely absent; static edges/noise can become salient', 1),
    ('MAA attention', 'Spatial-tolerant query from max/avg pooling, MLP and learnable map; channel attention', 'One-channel spatial sigmoid mask with scalar beta', 'Misalignment tolerance and channel selection are not equivalent', 2),
    ('LAF query', 'Explicit phi([Y_r*F_r; Y_t*F_t])', 'Global feature pooling plus local contrast logits', 'MAA output does not explicitly condition the fusion query', 3),
    ('LAF key/value', 'Dilated-conv modality keys, bidirectional cross values, hierarchical pooling and final attention', 'Reliability softmax plus C/4 partial-channel cross residual', 'A useful efficient adaptation, but not Eq.4-Eq.8 reproduction', 4),
    ('Fusion placement', 'After every Transformer stage; residual enters the next stage', 'P3/P4/P5 streams merge only at the neck entrance', 'No stage-to-stage cross-modal refinement', 5),
    ('Task and data', 'Paired RGB-thermal videos for human action classification', 'Paired static aerial images for vehicle OBB detection', 'Paper ablation effect sizes are not transferable', 6)
) AS t(component, paper, current, impact, "order");
