-- Executed against an in-memory SQLite table populated from the run's results.csv.
-- Column names were normalized during CSV ingestion; values were not transformed.
SELECT
    epoch,
    map50,
    map50_95,
    val_dfl_loss,
    val_box_loss,
    val_cls_loss,
    train_dfl_loss,
    lr_pg0
FROM training_metrics
WHERE epoch BETWEEN 24 AND 43
ORDER BY epoch;
