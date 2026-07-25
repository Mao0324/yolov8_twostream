DROP TABLE IF EXISTS summary;
CREATE TABLE summary (
  model_key TEXT, model TEXT, params INTEGER, best_epoch INTEGER,
  val_best_map50 REAL, val_best_map5095 REAL,
  test_precision REAL, test_recall REAL, test_map50 REAL, test_map5095 REAL,
  test_delta_pp REAL
);
INSERT INTO summary VALUES
  ('V2', 'V2 pre-C2f', 16698253, 92, 0.85330, 0.70874, 0.797, 0.790, 0.824, 0.683, 0.0),
  ('PostC2f', 'Post-C2f', 16698253, 95, 0.85266, 0.70668, 0.790, 0.798, 0.822, 0.683, 0.0),
  ('Refine', 'Post-C2f + Refine', 18423059, 100, 0.85057, 0.70601, 0.795, 0.793, 0.824, 0.685, 0.2);

DROP TABLE IF EXISTS validation_anchor;
CREATE TABLE validation_anchor (model TEXT, epoch INTEGER, map50 REAL, map5095 REAL);
INSERT INTO validation_anchor VALUES
  ('V2 pre-C2f',1,0.17872,0.08712),('V2 pre-C2f',5,0.56667,0.40599),
  ('V2 pre-C2f',10,0.68535,0.52792),('V2 pre-C2f',20,0.77501,0.61723),
  ('V2 pre-C2f',40,0.83831,0.68582),('V2 pre-C2f',60,0.84714,0.70007),
  ('V2 pre-C2f',80,0.85164,0.70530),('V2 pre-C2f',100,0.85286,0.70860),
  ('Post-C2f',1,0.19830,0.10787),('Post-C2f',5,0.55725,0.40847),
  ('Post-C2f',10,0.65758,0.50126),('Post-C2f',20,0.78875,0.62900),
  ('Post-C2f',40,0.83301,0.68202),('Post-C2f',60,0.85175,0.70217),
  ('Post-C2f',80,0.85286,0.70548),('Post-C2f',100,0.85215,0.70602),
  ('Post-C2f + Refine',1,0.20573,0.10153),('Post-C2f + Refine',5,0.56220,0.40719),
  ('Post-C2f + Refine',10,0.65274,0.49456),('Post-C2f + Refine',20,0.78148,0.62380),
  ('Post-C2f + Refine',40,0.82540,0.67855),('Post-C2f + Refine',60,0.84318,0.69593),
  ('Post-C2f + Refine',80,0.84859,0.70169),('Post-C2f + Refine',100,0.85057,0.70601);

DROP TABLE IF EXISTS test_class;
CREATE TABLE test_class (
  class TEXT, model TEXT, precision REAL, recall REAL, map50 REAL, map5095 REAL
);
INSERT INTO test_class VALUES
  ('car','V2 pre-C2f',0.948,0.969,0.986,0.839),
  ('car','Post-C2f',0.948,0.970,0.985,0.840),
  ('car','Post-C2f + Refine',0.951,0.967,0.985,0.840),
  ('truck','V2 pre-C2f',0.768,0.791,0.825,0.662),
  ('truck','Post-C2f',0.763,0.808,0.827,0.664),
  ('truck','Post-C2f + Refine',0.769,0.800,0.827,0.666),
  ('bus','V2 pre-C2f',0.920,0.946,0.962,0.820),
  ('bus','Post-C2f',0.917,0.947,0.963,0.824),
  ('bus','Post-C2f + Refine',0.923,0.943,0.965,0.824),
  ('van','V2 pre-C2f',0.697,0.559,0.646,0.531),
  ('van','Post-C2f',0.681,0.558,0.640,0.527),
  ('van','Post-C2f + Refine',0.685,0.557,0.645,0.533),
  ('freight_car','V2 pre-C2f',0.651,0.687,0.700,0.564),
  ('freight_car','Post-C2f',0.641,0.705,0.693,0.559),
  ('freight_car','Post-C2f + Refine',0.646,0.695,0.699,0.564);

DROP TABLE IF EXISTS learned_components;
CREATE TABLE learned_components (
  sort_key INTEGER, model TEXT, stage TEXT, beta_rgb REAL, beta_ir REAL,
  rgb_spatial_std REAL, ir_spatial_std REAL, laf_cross_scale REAL
);
INSERT INTO learned_components VALUES
  (1,'V2 pre-C2f','P3',0.159935,0.159935,0.000005674,0.000008184,-0.005398),
  (2,'V2 pre-C2f','P4',0.159935,0.160080,0.000006040,0.000004719,0.000524),
  (3,'V2 pre-C2f','P5',0.159791,0.159935,0.000003809,0.000004285,0.001359),
  (4,'Post-C2f','P3',0.170644,0.113659,0.000006931,0.000006982,-0.031525),
  (5,'Post-C2f','P4',0.150940,0.183802,0.000006395,0.000004546,-0.000487),
  (6,'Post-C2f','P5',0.154816,0.153144,0.000001539,0.000001641,-0.003040),
  (7,'Post-C2f + Refine','P3',0.152314,0.121678,0.000006958,0.000006352,-0.006588),
  (8,'Post-C2f + Refine','P4',0.160803,0.171412,0.000004427,0.000004724,0.010155),
  (9,'Post-C2f + Refine','P5',0.158641,0.158069,0.000001670,0.000001283,-0.000241);

DROP TABLE IF EXISTS refine_gamma;
CREATE TABLE refine_gamma (
  sort_key INTEGER, stage TEXT, rgb_gamma REAL, ir_gamma REAL, interpretation TEXT
);
INSERT INTO refine_gamma VALUES
  (1,'P3',-0.431641,0.576660,'RGB negative / IR positive'),
  (2,'P4',-0.422363,0.450684,'RGB negative / IR positive'),
  (3,'P5',-0.668457,0.322266,'RGB negative / IR positive');
