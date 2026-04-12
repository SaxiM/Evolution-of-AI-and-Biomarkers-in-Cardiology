"""NLP entity extraction from PubMed abstracts."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TextIO

import yaml

logger = logging.getLogger(__name__)


def load_entity_dict(yaml_path: str) -> dict[str, list[str]]:
    """Load a YAML synonym dictionary.

    Args:
        yaml_path: Path to the YAML file with canonical names as keys
                   and lists of aliases as values.

    Returns:
        Dictionary mapping canonical name -> list of alias strings.
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected dict at {yaml_path}, got {type(data).__name__}")

    result: dict[str, list[str]] = {}
    for canonical, aliases in data.items():
        if not isinstance(aliases, list):
            raise ValueError(
                f"Expected list of aliases for '{canonical}', got {type(aliases).__name__}"
            )
        result[canonical] = [a.lower().strip() for a in aliases if a]

    return result


def normalize_text(text: str) -> str:
    """Normalize text for entity matching.

    - Lowercase
    - Replace punctuation (except hyphens) with spaces
    - Collapse multiple spaces
    - Strip

    Args:
        text: Raw text to normalize.

    Returns:
        Normalized text string.
    """
    # Replace punctuation with spaces, but keep hyphens and apostrophes
    text = re.sub(r"[^\w\s'-]", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def extract_entities(abstract: str, entity_dict: dict[str, list[str]]) -> dict[str, int]:
    """Extract canonical entities from an abstract.

    Normalizes the abstract first, then searches for each alias using
    regex word boundaries, matching longest aliases first to avoid
    double-counting via shorter sub-aliases.

    Args:
        abstract: Raw abstract text.
        entity_dict: Dictionary mapping canonical names to alias lists.

    Returns:
        Dictionary mapping canonical name -> count of mentions.
    """
    normalized = normalize_text(abstract)
    results: dict[str, int] = {}

    for canonical, aliases in entity_dict.items():
        # Sort by length descending so longer (more specific) aliases match first
        sorted_aliases = sorted(aliases, key=len, reverse=True)
        found = False

        for alias in sorted_aliases:
            # Escape regex special characters in the alias
            escaped = re.escape(alias)
            pattern = r"\b" + escaped + r"\b"
            if re.search(pattern, normalized):
                results[canonical] = results.get(canonical, 0) + 1
                found = True
                break  # Avoid double-counting via different aliases

        if not found:
            results[canonical] = 0

    return results


def process_records(
    input_path: str,
    biomarker_yaml: str,
    ml_yaml: str,
    output_path: str,
) -> None:
    """Enrich JSONL records with extracted entities.

    Reads records from input_path (one JSON object per line), extracts
    biomarker and ML method entities, adds derived fields, and writes
    enriched records to output_path.

    Args:
        input_path: Path to input .jsonl file.
        biomarker_yaml: Path to biomarkers YAML config.
        ml_yaml: Path to ML methods YAML config.
        output_path: Path for output .jsonl file.
    """
    biomarker_dict = load_entity_dict(biomarker_yaml)
    ml_dict = load_entity_dict(ml_yaml)

    input_p = Path(input_path)
    output_p = Path(output_path)
    output_p.parent.mkdir(parents=True, exist_ok=True)

    processed = 0

    with open(input_p, "r", encoding="utf-8") as infile, \
         open(output_p, "w", encoding="utf-8") as outfile:

        for line in infile:
            line = line.strip()
            if not line:
                continue

            import json
            record = json.loads(line)

            abstract = record.get("abstract", "")
            if not abstract:
                record["biomarkers_found"] = {}
                record["ml_methods_found"] = {}
                record["has_biomarker"] = False
                record["has_ml"] = False
                record["category"] = "neither"
                outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue

            biomarkers = extract_entities(abstract, biomarker_dict)
            ml_methods = extract_entities(abstract, ml_dict)

            has_biomarker = any(v > 0 for v in biomarkers.values())
            has_ml = any(v > 0 for v in ml_methods.values())

            if has_biomarker and has_ml:
                category = "both"
            elif has_biomarker:
                category = "biomarker_only"
            elif has_ml:
                category = "ml_only"
            else:
                category = "neither"

            record["biomarkers_found"] = biomarkers
            record["ml_methods_found"] = ml_methods
            record["has_biomarker"] = has_biomarker
            record["has_ml"] = has_ml
            record["category"] = category

            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
            processed += 1

    logger.info("Processed %d records -> %s", processed, output_p)
