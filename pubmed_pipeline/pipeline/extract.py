"""NLP entity extraction from PubMed abstracts."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TextIO

import yaml

logger = logging.getLogger(__name__)

ENTITY_YEAR_CONSTRAINTS = {
    "XGBoost":             2014,
    "transformer":         2017,
    "deep_learning":       2006,
    "random_forest":       2001,
    "SVM":                 1995,
    "naive_bayes":         None,
    "clustering":          None,
    "logistic_regression": None,
    "neural_network":      None,
    "decision_tree":       None,
}


def apply_temporal_constraints(entities_found, year, constraints):
    """
    Usuwa encje, które nie mogły istnieć w danym roku.
    Loguje ostrzeżenie przy każdym usunięciu.
    """
    filtered = {}
    for entity, count in entities_found.items():
        min_year = constraints.get(entity)
        if min_year is not None and year < min_year:
            logging.warning(
                f"Usunięto '{entity}' dla roku {year} "
                f"(metoda niedostępna przed {min_year})"
            )
        else:
            filtered[entity] = count
    return filtered


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
        Dictionary mapping canonical name -> 1 if found (binary counting).
    """
    text = normalize_text(abstract)
    found = {}

    for canonical, aliases in entity_dict.items():
        for alias in sorted(aliases, key=len, reverse=True):
            pattern = r'\b' + re.escape(alias) + r'\b'
            if re.search(pattern, text):
                found[canonical] = 1
                break

    return found


def generate_integrity_report(
    stats: dict,
    output_path: str,
) -> None:
    """Generate and save data integrity report.

    Args:
        stats: Dictionary with processing statistics.
        output_path: Path for the report file.
    """
    lines = [
        "=== RAPORT INTEGRALNOŚCI DANYCH ===",
        f"Łączna liczba przetworzonych abstraktów: {stats['total_processed']}",
        f"Pominięte (brak tekstu abstraktu): {stats['skipped_no_abstract']}",
        "",
        "Encje usunięte przez ograniczenia czasowe:",
    ]

    temporal_removals = stats.get("temporal_removals", {})
    for entity, count in sorted(temporal_removals.items()):
        lines.append(f"  - {entity}: {count} przypadków")

    lines.append("")
    lines.append("Liczba abstraktów per rok:")
    for year, count in sorted(stats.get("abstracts_per_year", {}).items()):
        lines.append(f"  {year}: {count}")

    low_reliability_years = {
        year: count
        for year, count in stats.get("abstracts_per_year", {}).items()
        if count < 100
    }
    if low_reliability_years:
        lines.append("")
        lines.append("OSTRZEŻENIE — lata z < 100 abstraktami (niska wiarygodność):")
        for year, count in sorted(low_reliability_years.items()):
            lines.append(f"  {year}: {count}  <- rozważ wykluczenie z analizy trendów")

    report_content = "\n".join(lines) + "\n"
    Path(output_path).write_text(report_content, encoding="utf-8")
    logger.info("Zapisano raport integralności: %s", output_path)


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

    # Statistics for integrity report
    stats = {
        "total_processed": 0,
        "skipped_no_abstract": 0,
        "temporal_removals": {},
        "abstracts_per_year": {},
    }

    with open(input_p, "r", encoding="utf-8") as infile, \
         open(output_p, "w", encoding="utf-8") as outfile:

        for line in infile:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            year = record.get("year")
            abstract = record.get("abstract", "")

            if year is not None:
                stats["abstracts_per_year"][year] = stats["abstracts_per_year"].get(year, 0) + 1

            if not abstract:
                record["biomarkers_found"] = {}
                record["ml_methods_found"] = {}
                record["has_biomarker"] = False
                record["has_ml"] = False
                record["category"] = "neither"
                outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats["skipped_no_abstract"] += 1
                continue

            biomarkers = extract_entities(abstract, biomarker_dict)
            ml_methods = extract_entities(abstract, ml_dict)

            # Apply temporal constraints to ML methods
            if year is not None:
                ml_methods_filtered = apply_temporal_constraints(
                    ml_methods, year, ENTITY_YEAR_CONSTRAINTS
                )
                # Track removals for integrity report
                for entity in ml_methods:
                    if entity in ml_methods_filtered:
                        pass  # kept
                    else:
                        key = f"{entity} przed {ENTITY_YEAR_CONSTRAINTS.get(entity)}"
                        stats["temporal_removals"][key] = stats["temporal_removals"].get(key, 0) + 1
                ml_methods = ml_methods_filtered

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
            stats["total_processed"] += 1

    logger.info("Processed %d records -> %s", stats["total_processed"], output_p)

    # Generate integrity report
    report_dir = Path(output_p).parent
    generate_integrity_report(stats, str(report_dir / "integrity_report.txt"))
