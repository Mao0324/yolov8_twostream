-- Reproducible diagnostic query for the actually active PT-R013 run.
-- The CSV was normalized into active_training_metrics with one row per completed epoch.
SELECT
  epoch,
  precision,
  recall,
  map50,
  map50_95,
  train_box_loss,
  train_cls_loss,
  train_dfl_loss,
  val_box_loss,
  val_cls_loss,
  val_dfl_loss,
  lr_pg0
FROM active_training_metrics
WHERE epoch BETWEEN 44 AND 83
ORDER BY epoch;

-- Resume-window linear trends (epoch 51 was the first completed resumed epoch).
SELECT
  regr_slope(map50, epoch) AS map50_per_epoch,
  regr_slope(map50_95, epoch) AS map50_95_per_epoch,
  regr_slope(val_dfl_loss, epoch) AS val_dfl_per_epoch
FROM active_training_metrics
WHERE epoch BETWEEN 51 AND 83;
