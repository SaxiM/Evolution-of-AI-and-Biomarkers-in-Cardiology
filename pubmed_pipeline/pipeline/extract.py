"""NLP entity extraction from PubMed abstracts."""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

import yaml

from .llm_ner import call_ollama_ner, ollama_server_reachable

logger = logging.getLogger(__name__)

ENTITY_YEAR_CONSTRAINTS = {
    "XGBoost":             2014,
    "lightgbm":            2016,
    "catboost":            2017,
    "transformer":         2017,
    "deep_learning":       2006,
    "random_forest":       2001,
    "SVM":                 1995,
    "naive_bayes":         None,
    "clustering":          None,
    "logistic_regression": None,
    "linear_regression":   None,
    "cox_model":           None,
    "neural_network":      None,
    "decision_tree":       None,
    "gradient_boosting":   None,
}

NerBackend = Literal["lexicon", "ollama", "hybrid"]


def _merge_entity_dicts(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    keys = set(a) | set(b)
    return {k: 1 for k in keys if a.get(k, 0) or b.get(k, 0)}


def coerce_publication_year(year) -> int | None:
    """Zwraca rok jako int albo None, jeśli nie da się go ustalić."""
    if year is None:
        return None
    if isinstance(year, int):
        return year
    if isinstance(year, float) and year.is_integer():
        return int(year)
    if isinstance(year, str):
        s = year.strip()
        if s.isdigit() and len(s) == 4:
            return int(s)
        m = re.search(r"(19|20)\d{2}", s)
        if m:
            return int(m.group(0))
    return None


def apply_temporal_constraints(entities_found, year, constraints):
    """
    Usuwa encje, które nie mogły istnieć w danym roku.
    Loguje ostrzeżenie przy każdym usunięciu.
    """
    filtered = {}
    for entity, count in entities_found.items():
        min_year = constraints.get(entity)
        if min_year is not None and year < min_year:
            logger.warning(
                "Usunięto '%s' dla roku %s (metoda niedostępna przed %s)",
                entity,
                year,
                min_year,
            )
        else:
            filtered[entity] = count
    return filtered


def filter_ml_methods_by_year(
    ml_methods: dict[str, int],
    year: int | None,
    constraints: dict[str, int | None],
) -> tuple[dict[str, int], list[str]]:
    """Stosuje reguły chronologiczne do wykrytych metod ML.

    Gdy rok publikacji jest nieznany, usuwa wszystkie encje z ustalonym min_year
    (konserwatywnie — unikamy fałszywych pozytywów typu „XGBoost” bez daty).

    Returns:
        (przefiltrowany słownik, lista nazw encji usuniętych)
    """
    stripped: list[str] = []
    y = coerce_publication_year(year)
    if y is None:
        filtered: dict[str, int] = {}
        for entity, count in ml_methods.items():
            if constraints.get(entity) is None:
                filtered[entity] = count
            else:
                stripped.append(entity)
        return filtered, stripped
    before = set(ml_methods.keys())
    out = apply_temporal_constraints(ml_methods, y, constraints)
    stripped = sorted(before - set(out.keys()))
    return out, stripped


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


# „logistic” i „regression” często rozdzielone wyrazami („logistic and cox regression” itd.).
_LOGISTIC_NEAR_REGRESSION = re.compile(
    r"\blogistic\b(?:\s+\w+){0,10}\s+\bregression\b"
)


def augment_logistic_regression_detection(abstract: str, ml_methods: dict[str, int]) -> None:
    """Zwiększa czułość na regresję logistyczną bez zmiany pozostałych encji."""
    if ml_methods.get("logistic_regression"):
        return
    norm = normalize_text(abstract)
    if _LOGISTIC_NEAR_REGRESSION.search(norm):
        ml_methods["logistic_regression"] = 1


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


def _enrich_one_record(
    record: dict[str, Any],
    *,
    biomarker_dict: dict[str, list[str]],
    ml_dict: dict[str, list[str]],
    bio_keys: list[str],
    ml_keys: list[str],
    ner_backend: NerBackend,
    use_ollama: bool,
    effective_backend: str,
    ollama_model: str,
    ollama_base_url: str,
    ollama_timeout: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Zwraca (rekord_wyjściowy, fragment_statystyk do scalenia)."""
    frag: dict[str, Any] = {
        "year_inc": None,
        "skipped": False,
        "processed": 0,
        "temporal": {},
    }

    year_raw = record.get("year")
    y_norm = coerce_publication_year(year_raw)
    if y_norm is not None:
        record["year"] = y_norm
        frag["year_inc"] = y_norm

    abstract = record.get("abstract", "")

    if not abstract:
        record["biomarkers_found"] = {}
        record["ml_methods_found"] = {}
        record["has_biomarker"] = False
        record["has_ml"] = False
        record["category"] = "neither"
        record["ml_temporal_stripped"] = []
        record["ner_backend_effective"] = effective_backend
        frag["skipped"] = True
        return record, frag

    bio_lex = extract_entities(abstract, biomarker_dict)
    ml_lex = extract_entities(abstract, ml_dict)
    augment_logistic_regression_detection(abstract, ml_lex)

    bio_llm: dict[str, int] = {}
    ml_llm: dict[str, int] = {}
    if use_ollama:
        bio_llm, ml_llm = call_ollama_ner(
            abstract,
            bio_keys,
            ml_keys,
            model=ollama_model,
            base_url=ollama_base_url,
            timeout=ollama_timeout,
        )
        augment_logistic_regression_detection(abstract, ml_llm)

    if ner_backend == "lexicon" or not use_ollama:
        biomarkers = bio_lex
        ml_methods = ml_lex
    elif ner_backend == "ollama":
        biomarkers = bio_llm
        ml_methods = ml_llm
    else:
        biomarkers = _merge_entity_dicts(bio_lex, bio_llm)
        ml_methods = _merge_entity_dicts(ml_lex, ml_llm)

    ml_methods, temporal_stripped = filter_ml_methods_by_year(
        ml_methods, year_raw, ENTITY_YEAR_CONSTRAINTS
    )
    record["ml_temporal_stripped"] = temporal_stripped
    for entity in temporal_stripped:
        min_y = ENTITY_YEAR_CONSTRAINTS.get(entity)
        if coerce_publication_year(year_raw) is None:
            key = f"{entity} (brak/niepoprawny rok — encja z ograniczeniem czasowym odrzucona)"
        else:
            key = f"{entity} przed {min_y}"
        frag["temporal"][key] = frag["temporal"].get(key, 0) + 1

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
    record["ner_backend_effective"] = effective_backend
    record["ner_backend_requested"] = ner_backend
    if use_ollama:
        record["ollama_model"] = ollama_model

    frag["processed"] = 1
    return record, frag


def _merge_stat_fragments(stats: dict[str, Any], frag: dict[str, Any]) -> None:
    y = frag.get("year_inc")
    if y is not None:
        stats["abstracts_per_year"][y] = stats["abstracts_per_year"].get(y, 0) + 1
    if frag.get("skipped"):
        stats["skipped_no_abstract"] += 1
    elif frag.get("processed"):
        stats["total_processed"] += 1
    for k, v in frag.get("temporal", {}).items():
        stats["temporal_removals"][k] = stats["temporal_removals"].get(k, 0) + v


def process_records(
    input_path: str,
    biomarker_yaml: str,
    ml_yaml: str,
    output_path: str,
    *,
    ner_backend: NerBackend = "hybrid",
    ollama_model: str = "llama3.2",
    ollama_base_url: str = "http://127.0.0.1:11434",
    ollama_timeout: int = 180,
    ollama_workers: int = 1,
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
        ner_backend: ``lexicon`` (tylko RegEx), ``ollama`` (tylko lokalny LLM),
            ``hybrid`` (LLM + RegEx, suma logiczna).
        ollama_model: Nazwa modelu w Ollama (np. ``llama3.2``, ``mistral``).
        ollama_base_url: Bazowy URL API Ollama.
        ollama_timeout: Timeout HTTP na jeden abstrakt (sekundy).
        ollama_workers: Równoległe żądania do Ollama (>=2 przyspiesza extract; większe
            obciążenie GPU/RAM — zacznij od 2–4).
    """
    biomarker_dict = load_entity_dict(biomarker_yaml)
    ml_dict = load_entity_dict(ml_yaml)
    bio_keys = list(biomarker_dict.keys())
    ml_keys = list(ml_dict.keys())

    use_ollama = ner_backend in ("ollama", "hybrid")
    if use_ollama and not ollama_server_reachable(ollama_base_url):
        if ner_backend == "ollama":
            raise RuntimeError(
                f"Nie można połączyć się z Ollama pod {ollama_base_url}. "
                "Uruchom `ollama serve`, pobierz model (`ollama pull llama3.2`) "
                "lub ustaw --ner-backend lexicon / hybrid (hybrid użyje tylko RegEx, gdy Ollama nie działa)."
            )
        logger.warning(
            "Ollama niedostępna pod %s — tryb hybrid ogranicza się do leksykonu RegEx.",
            ollama_base_url,
        )
        use_ollama = False

    effective_backend: str = ner_backend if use_ollama or ner_backend == "lexicon" else "lexicon"

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

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore

    ollama_workers = max(1, int(ollama_workers))
    parallel_ollama = use_ollama and ollama_workers > 1

    def _worker(rec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        return _enrich_one_record(
            rec,
            biomarker_dict=biomarker_dict,
            ml_dict=ml_dict,
            bio_keys=bio_keys,
            ml_keys=ml_keys,
            ner_backend=ner_backend,
            use_ollama=use_ollama,
            effective_backend=effective_backend,
            ollama_model=ollama_model,
            ollama_base_url=ollama_base_url,
            ollama_timeout=ollama_timeout,
        )

    if parallel_ollama:
        records_in: list[dict[str, Any]] = []
        with open(input_p, "r", encoding="utf-8") as infile:
            for line in infile:
                line = line.strip()
                if not line:
                    continue
                records_in.append(json.loads(line))

        logger.info(
            "Extract równoległy: %d abstraktów, %d workerów Ollama",
            len(records_in),
            ollama_workers,
        )
        with ThreadPoolExecutor(max_workers=ollama_workers) as pool:
            mapped = pool.map(_worker, records_in, chunksize=1)
            if tqdm:
                mapped = tqdm(mapped, total=len(records_in), desc="extract", unit="abs")
            with open(output_p, "w", encoding="utf-8") as outfile:
                for out_rec, frag in mapped:
                    _merge_stat_fragments(stats, frag)
                    outfile.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
    else:
        with open(input_p, "r", encoding="utf-8") as infile, open(
            output_p, "w", encoding="utf-8"
        ) as outfile:
            source = tqdm(infile, desc="extract", unit="abs") if (tqdm and use_ollama) else infile
            for line in source:
                line = line.strip()
                if not line:
                    continue

                record = json.loads(line)
                out_rec, frag = _worker(record)
                _merge_stat_fragments(stats, frag)
                outfile.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

    logger.info("Processed %d records -> %s", stats["total_processed"], output_p)

    # Generate integrity report
    report_dir = Path(output_p).parent
    generate_integrity_report(stats, str(report_dir / "integrity_report.txt"))
