#!/usr/bin/env python3
"""Train the frozen RGB-only OBB teacher used by P2Det V16."""

from tools.train_single_modality_teacher_m2dlif import train_single_modality_teacher


if __name__ == "__main__":
    train_single_modality_teacher("rgb")
