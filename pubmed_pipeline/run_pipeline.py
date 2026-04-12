#!/usr/bin/env python3
"""CLI entry point for the PubMed pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from pipeline import (
    aggregate_by_year,
    add_trend_features,
    fetch_pubmed_abstracts,
    generate_all_plots,
    process_records,
    save_aggregated,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_QUERY = (
    "biomarkers OR machine learning OR deep learning OR artificial intelligence "
    "AND (clinical OR diagnosis OR prognosis)"
)
DEFAULT_START = 2000
DEFAULT_END = 2024


def cmd_fetch(args: argparse.Namespace) -> None:
    """Run the fetch subcommand."""
    fetch_pubmed_abstracts(
        query=args.query,
        start_year=args.start_year,
        end_year=args.end_year,
        max_per_year=args.max_per_year,
        email=args.email,
        output_path=args.output,
    )


def cmd_extract(args: argparse.Namespace) -> None:
    """Run the extract subcommand."""
    process_records(
        input_path=args.input,
        biomarker_yaml=args.biomarker_yaml,
        ml_yaml=args.ml_yaml,
        output_path=args.output,
    )


def cmd_aggregate(args: argparse.Namespace) -> None:
    """Run the aggregate subcommand."""
    df = aggregate_by_year(args.input)
    df = add_trend_features(df)
    save_aggregated(df, args.output)


def cmd_visualize(args: argparse.Namespace) -> None:
    """Run the visualize subcommand."""
    import pandas as pd

    csv_path = Path(args.input).with_suffix(".csv")
    df = pd.read_csv(csv_path)
    generate_all_plots(df, output_dir=args.output_dir)


def cmd_all(args: argparse.Namespace) -> None:
    """Run the full pipeline end-to-end."""
    logger.info("=== STEP 1: Fetch ===")
    raw_path = "data/raw/abstracts.jsonl"
    fetch_pubmed_abstracts(
        query=args.query,
        start_year=args.start_year,
        end_year=args.end_year,
        max_per_year=args.max_per_year,
        email=args.email,
        output_path=raw_path,
    )

    logger.info("=== STEP 2: Extract ===")
    enriched_path = "data/processed/enriched.jsonl"
    process_records(
        input_path=raw_path,
        biomarker_yaml="config/biomarkers.yaml",
        ml_yaml="config/ml_methods.yaml",
        output_path=enriched_path,
    )

    logger.info("=== STEP 3: Aggregate ===")
    agg_path = "data/processed/aggregated"
    df = aggregate_by_year(enriched_path)
    df = add_trend_features(df)
    save_aggregated(df, agg_path)

    logger.info("=== STEP 4: Visualize ===")
    generate_all_plots(df, output_dir="data/figures/")

    logger.info("=== Pipeline complete ===")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PubMed AI/ML vs Biomarker trend pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # fetch
    p_fetch = sub.add_parser("fetch", help="Fetch PubMed abstracts")
    p_fetch.add_argument("--query", default=DEFAULT_QUERY)
    p_fetch.add_argument("--start-year", type=int, default=DEFAULT_START)
    p_fetch.add_argument("--end-year", type=int, default=DEFAULT_END)
    p_fetch.add_argument("--max-per-year", type=int, default=500)
    p_fetch.add_argument("--email", default="researcher@example.com")
    p_fetch.add_argument("--output", default="data/raw/abstracts.jsonl")
    p_fetch.set_defaults(func=cmd_fetch)

    # extract
    p_extract = sub.add_parser("extract", help="Extract entities from abstracts")
    p_extract.add_argument("--input", default="data/raw/abstracts.jsonl")
    p_extract.add_argument(
        "--biomarker-yaml", default="config/biomarkers.yaml",
    )
    p_extract.add_argument("--ml-yaml", default="config/ml_methods.yaml")
    p_extract.add_argument("--output", default="data/processed/enriched.jsonl")
    p_extract.set_defaults(func=cmd_extract)

    # aggregate
    p_agg = sub.add_parser("aggregate", help="Aggregate records by year")
    p_agg.add_argument("--input", default="data/processed/enriched.jsonl")
    p_agg.add_argument("--output", default="data/processed/aggregated")
    p_agg.set_defaults(func=cmd_aggregate)

    # visualize
    p_vis = sub.add_parser("visualize", help="Generate trend plots")
    p_vis.add_argument("--input", default="data/processed/aggregated.csv")
    p_vis.add_argument("--output-dir", default="data/figures/")
    p_vis.set_defaults(func=cmd_visualize)

    # all
    p_all = sub.add_parser("all", help="Run full pipeline")
    p_all.add_argument("--query", default=DEFAULT_QUERY)
    p_all.add_argument("--start-year", type=int, default=DEFAULT_START)
    p_all.add_argument("--end-year", type=int, default=DEFAULT_END)
    p_all.add_argument("--max-per-year", type=int, default=500)
    p_all.add_argument("--email", default="researcher@example.com")
    p_all.set_defaults(func=cmd_all)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
