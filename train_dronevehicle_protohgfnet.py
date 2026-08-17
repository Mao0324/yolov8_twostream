#!/usr/bin/env python3
"""Launch a registered ProtoHGFNet-family experiment with monitor integration."""

import sys

from tools.train_experiment import main


if __name__ == "__main__":
    if len(sys.argv) == 1 or sys.argv[1].startswith("-"):
        sys.argv.insert(1, "PHG-001")
    raise SystemExit(main())
