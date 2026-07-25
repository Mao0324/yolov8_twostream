-- Frozen epoch-health snapshot captured before the original run directory disappeared.
WITH epoch_health(epoch, finite_loss_fields, train_saliency_loss, map50, metric_fresh) AS (
  VALUES
    (1, 8, 0.11768, 0.43544, 'fresh'),
    (2, 8, 0.09716, 0.61324, 'fresh'),
    (3, 8, 0.09590, 0.66313, 'fresh'),
    (4, 8, 0.08740, 0.73373, 'fresh'),
    (5, 8, 0.07390, 0.71991, 'fresh'),
    (6, 8, 0.06504, 0.70473, 'fresh'),
    (7, 0, NULL, 0.70473, 'cached epoch 6'),
    (8, 0, NULL, 0.70473, 'cached epoch 6'),
    (9, 0, NULL, 0.70473, 'cached epoch 6'),
    (10, 0, NULL, 0.70473, 'cached epoch 6'),
    (11, 0, NULL, 0.70473, 'cached epoch 6'),
    (12, 0, NULL, 0.70473, 'cached epoch 6'),
    (13, 0, NULL, 0.70473, 'cached epoch 6'),
    (14, 0, NULL, 0.70473, 'cached epoch 6'),
    (15, 0, NULL, 0.70473, 'cached epoch 6'),
    (16, 0, NULL, 0.70473, 'cached epoch 6'),
    (17, 0, NULL, 0.70473, 'cached epoch 6'),
    (18, 0, NULL, 0.70473, 'cached epoch 6')
)
SELECT * FROM epoch_health ORDER BY epoch;

-- Frozen checkpoint finite-value scan.
WITH checkpoint_health(sort_order, checkpoint, displayed_epoch, ema_nonfinite, optimizer_nonfinite, verdict) AS (
  VALUES
    (1, 'migration checkpoint', '0', 0, 'n/a', 'finite; restart source'),
    (2, 'best.pt', '4', 0, '0 / 24525304', 'finite; diagnosis only'),
    (3, 'last.pt', '17', 24565176, '24525304 / 24525304', 'corrupt; never resume')
)
SELECT * FROM checkpoint_health ORDER BY sort_order;

-- One-image attention probe from the finite best.pt checkpoint.
WITH attention_probe(sort_order, stage, modality, reduction_n, scaled_absmax, softmax_max, softmax_min) AS (
  VALUES
    (1, 'P3', 'RGB', 6400, 88.1914, 1.0, 0.0),
    (2, 'P3', 'IR', 6400, 56.1619, 1.0, 0.0),
    (3, 'P4', 'RGB', 1600, 14.1984, 0.996524, 1.2176e-10),
    (4, 'P4', 'IR', 1600, 44.1497, 1.0, 3.04562e-31),
    (5, 'P5', 'RGB', 400, 0.438496, 0.0117367, 0.00491977),
    (6, 'P5', 'IR', 400, 0.267642, 0.0101338, 0.00608136)
)
SELECT * FROM attention_probe ORDER BY sort_order;

