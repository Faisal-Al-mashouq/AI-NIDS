#!/usr/bin/env python3
"""Merge, clean, label, and split raw dataset."""

from argparse import ArgumentParser

from src.preprocessing import run

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--target",
        choices=["binary", "multiclass"],
        default="binary",
        help="Target label mode.",
    )
    args = parser.parse_args()
    run(args.target)
