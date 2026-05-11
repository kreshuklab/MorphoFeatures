"""Analysis command-line interface."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    """Dispatch analysis subcommands."""

    args_list = list(sys.argv[1:] if argv is None else argv)
    commands = ["cluster", "classify", "bilateral", "genes", "recluster"]
    parser = argparse.ArgumentParser(description="MorphoFeatures analysis commands.")
    parser.add_argument("command", choices=commands)
    if not args_list or args_list[0] in {"-h", "--help"}:
        parser.parse_args(args_list)
        return
    command = args_list[0]
    remaining = args_list[1:]

    if command == "cluster":
        from morphofeatures.analysis.clustering import main as command_main
    elif command == "classify":
        from morphofeatures.analysis.classification import main as command_main
    elif command == "bilateral":
        from morphofeatures.analysis.bilateral import main as command_main
    elif command == "genes":
        from morphofeatures.analysis.genes import main as command_main
    elif command == "recluster":
        from morphofeatures.analysis.reclustering import main as command_main
    else:  # pragma: no cover
        raise ValueError(f"Unknown analysis command: {command}")

    command_main(remaining)


if __name__ == "__main__":
    main()
