#!/usr/bin/env python3
"""Create compact RGB/IR/overlay visual-audit montages."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np


OUT = Path(__file__).resolve().parent
ROOT = Path("/media/biiteam/新加卷1/biiteam/MCONG/datasets/DroneVehicle_twostream_3")


def load_rows() -> list[dict[str, str]]:
    with (OUT / "alignment_samples.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        row
        for row in rows
        if row["ecc_ok"] == "True"
        and float(row["ecc_correlation"]) >= 0.5
        and float(row["ecc_shift_px"]) <= 10
    ]


def render(rows: list[dict[str, str]], target: Path) -> None:
    tiles = []
    for row in rows:
        split, stem = row["split"], row["stem"]
        rgb = cv2.imread(str(ROOT / "images" / split / f"{stem}.jpg"))
        ir_gray = cv2.imread(str(ROOT / "image" / split / f"{stem}.jpg"), cv2.IMREAD_GRAYSCALE)
        if rgb is None or ir_gray is None:
            continue
        ir = cv2.cvtColor(ir_gray, cv2.COLOR_GRAY2BGR)
        rgb_gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        overlay = np.zeros_like(rgb)
        overlay[:, :, 2] = rgb_gray
        overlay[:, :, 1] = ir_gray
        label = (
            f"{split}/{stem}  shift={float(row['ecc_shift_px']):.2f}px  "
            f"dx={float(row['ecc_dx_px']):+.1f} dy={float(row['ecc_dy_px']):+.1f}"
        )
        triptych = np.hstack([rgb, ir, overlay])
        cv2.rectangle(triptych, (0, 0), (triptych.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(triptych, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(triptych)
    if tiles:
        cv2.imwrite(str(target), np.vstack(tiles))


def main() -> None:
    rows = load_rows()
    rows_by_shift = sorted(rows, key=lambda row: float(row["ecc_shift_px"]))
    render(rows_by_shift[:6], OUT / "low_shift_montage.jpg")
    render(rows_by_shift[-6:][::-1], OUT / "high_shift_montage.jpg")


if __name__ == "__main__":
    main()
