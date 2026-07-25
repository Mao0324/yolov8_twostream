-- DuckDB-compatible literal datasets embedded in artifact.json.

-- experiment_comparison
SELECT * FROM (VALUES
  ('Baseline ADD', 'baseline', 0.70117, 0.669562, 0.000, 3.161, 14.734722, 37.0351, 259.59, 0),
  ('V1 MAA2D + LAFMerge', 'v1', 0.70309, 0.673000, 0.344, 3.009, 15.702371, 38.5563, 181.60, 1),
  ('V2 StaticMAA + LAF feedback', 'v2', 0.70874, 0.683000, 1.344, 2.574, 16.698253, 40.3729, 200.16, 2),
  ('V3 full-C paper LAF feedback', 'v3', 0.71097, 0.675000, 0.544, 3.597, 24.693320, 55.3006, 180.79, 3)
) AS t(experiment, run_id, best_val_map5095, test_map5095, test_delta_pp, val_test_gap_pp, params_m, gflops, fps, ordering);

-- v2_class_comparison
SELECT * FROM (VALUES
  ('car', 137146, 0.83585, 0.839, 0.315, 1),
  ('truck', 8657, 0.64594, 0.662, 1.606, 2),
  ('bus', 4467, 0.81501, 0.820, 0.499, 3),
  ('van', 4282, 0.50901, 0.531, 2.199, 4),
  ('freight_car', 5062, 0.54200, 0.564, 2.200, 5)
) AS t(class, instances, baseline_map5095, v2_map5095, delta_pp, ordering);

-- v2_stage_probe
SELECT * FROM (VALUES
  ('P3', 1.0800, 1.0800, 0.000009, 0.3176, 1.6824, 0.5976, 0.000010, 0.9159, 1),
  ('P4', 1.0800, 1.0800, 0.000012, 1.5339, 0.4661, 0.5432, 0.000001, -0.3485, 2),
  ('P5', 1.0799, 1.0800, 0.000007, 1.3305, 0.6695, 0.3480, 0.000002, -0.4089, 3)
) AS t(stage, maa_gain_rgb, maa_gain_ir, maa_gate_spatial_std_max, laf_rgb_weight, laf_ir_weight, correction_rms_ratio, cross_rms_ratio, brightness_rgb_weight_corr, ordering);

-- v3_stage_probe
SELECT * FROM (VALUES
  ('P3', 1.0805, 1.0820, 0.000013, 0.2087, 0.1196, 1),
  ('P4', 1.0800, 1.0825, 0.000010, 0.1374, 0.1004, 2),
  ('P5', 1.0813, 1.0813, 0.000004, 0.0336, 0.0258, 3)
) AS t(stage, maa_gain_rgb, maa_gain_ir, maa_gate_spatial_std_max, correction_rms_ratio, output_bn_gamma_rms, ordering);

