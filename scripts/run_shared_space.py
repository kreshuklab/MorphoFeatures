#!/usr/bin/env python
"""Run a configured paired morphology/expression experiment on CPU."""

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, help="New output directory; existing directories are refused"
    )
    args = parser.parse_args()
    # Bound compilation/math pools before importing numerical packages.
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/morpho_shared_space_matplotlib")
    os.environ.setdefault("NUMBA_NUM_THREADS", "2")
    from morphofeatures.analysis.shared_space.experiment import read_config, run

    run(read_config(args.config), output=args.output)


if __name__ == "__main__":
    main()
