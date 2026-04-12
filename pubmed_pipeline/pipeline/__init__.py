"""PubMed pipeline: fetch, extract, aggregate, and visualize trends."""

from .aggregate import add_trend_features, aggregate_by_year, save_aggregated
from .extract import extract_entities, load_entity_dict, normalize_text, process_records
from .fetch import fetch_pubmed_abstracts
from .visualize import generate_all_plots

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
]
