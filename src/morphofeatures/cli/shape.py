"""Shape pipeline command-line interface."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    """Dispatch shape subcommands."""

    args_list = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="MorphoFeatures shape commands.")
    parser.add_argument("command", choices=["train", "embed"])
    if not args_list or args_list[0] in {"-h", "--help"}:
        parser.parse_args(args_list)
        return
    command = args_list[0]
    remaining = args_list[1:]

    if command == "train":
        from morphofeatures.shape.trainer import main as train_main

        train_main(remaining)
    elif command == "embed":
        from morphofeatures.shape.inference import main as embed_main

        embed_main(remaining)


if __name__ == "__main__":
    main()
