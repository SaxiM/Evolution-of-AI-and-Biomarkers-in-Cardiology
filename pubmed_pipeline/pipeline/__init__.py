"""PubMed pipeline: fetch, extract, aggregate, and visualize trends."""

from .aggregate import add_trend_features, aggregate_by_year, save_aggregated
from .extract import (
    apply_temporal_constraints,
    coerce_publication_year,
    ENTITY_YEAR_CONSTRAINTS,
    extract_entities,
    filter_ml_methods_by_year,
    generate_integrity_report,
    load_entity_dict,
    normalize_text,
    process_records,
)
from .fetch import fetch_pubmed_abstracts
from .validation_sample import export_validation_sample
from .visualize import generate_all_plots, generate_interactive_plots, plot_single_biomarker_vs_ml

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
    "generate_interactive_plots",
    "plot_single_biomarker_vs_ml",
    "apply_temporal_constraints",
    "coerce_publication_year",
    "ENTITY_YEAR_CONSTRAINTS",
    "filter_ml_methods_by_year",
    "generate_integrity_report",
    "export_validation_sample",
]
