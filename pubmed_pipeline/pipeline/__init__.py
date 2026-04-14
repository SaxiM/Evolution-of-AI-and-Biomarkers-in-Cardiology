"""PubMed pipeline: fetch, extract, aggregate, and visualize trends."""

from .aggregate import add_trend_features, aggregate_by_year, save_aggregated
from .extract import (
    apply_temporal_constraints,
    ENTITY_YEAR_CONSTRAINTS,
    extract_entities,
    generate_integrity_report,
    load_entity_dict,
    normalize_text,
    process_records,
)
from .fetch import fetch_pubmed_abstracts
from .visualize import generate_all_plots, plot_single_biomarker_vs_ml

__all__ = [
    "fetch_pubmed_abstracts",
    "load_entity_dict",
    "normalize_text",
    "extract_entities",
    "process_records",
    "aggregate_by_year",
    "add_trend_features",
    "save_aggregated",
    "generate_all_plots",
    "plot_single_biomarker_vs_ml",
    "apply_temporal_constraints",
    "ENTITY_YEAR_CONSTRAINTS",
    "generate_integrity_report",
]
