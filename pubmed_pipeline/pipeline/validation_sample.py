"""Eksport małej próbki rekordów do ręcznej walidacji ekstrakcji encji."""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Kategorie ze schematu po extract; „raw” = rekord bez pól predykcji.
_STRATIFY_ORDER = ("both", "biomarker_only", "ml_only", "neither", "unknown", "raw")


def _record_category(rec: dict[str, Any]) -> str:
    if "category" in rec and rec["category"]:
        return str(rec["category"])
    if "biomarkers_found" in rec or "ml_methods_found" in rec:
        return "unknown"
    return "raw"


def stratified_sample(records: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    """Losuje n rekordów z dążeniem do równowagi kategorii (both / biomarker_only / ml_only / neither)."""
    rng = random.Random(seed)
    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_cat[_record_category(r)].append(r)

    for pool in by_cat.values():
        rng.shuffle(pool)

    per = max(1, n // 4)
    picked: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    def _key(r: dict[str, Any]) -> str:
        pmid = str(r.get("pmid", "")).strip()
        return f"pmid:{pmid}" if pmid else f"id:{id(r)}"

    for cat in _STRATIFY_ORDER:
        if len(picked) >= n:
            break
        pool = by_cat.get(cat, [])
        take = min(per, len(pool), n - len(picked))
        added = 0
        for r in pool:
            if added >= take or len(picked) >= n:
                break
            k = _key(r)
            if k in seen_keys:
                continue
            seen_keys.add(k)
            picked.append(r)
            added += 1

    if len(picked) < n:
        rng.shuffle(records)
        for r in records:
            if len(picked) >= n:
                break
            k = _key(r)
            if k in seen_keys:
                continue
            seen_keys.add(k)
            picked.append(r)

    return picked[:n]


def export_validation_sample(
    input_path: str,
    output_path: str,
    n: int = 20,
    seed: int = 42,
    stratify: bool = True,
) -> None:
    """Zapisuje JSONL z polami predykcji i pustymi polami „gold_*” do uzupełnienia ręcznego.

    Po adnotacji (uzupełnieniu gold_biomarkers / gold_ml_methods tym samym formatem co pred_*)
    możesz policzyć precision/recall po encji albo po poziomie „czy jakikolwiek biomarker / ML”.
    """
    path = Path(input_path)
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if not records:
        raise ValueError(f"Brak rekordów w {input_path}")

    n = min(n, len(records))
    if stratify:
        chosen = stratified_sample(records, n, seed)
    else:
        rng = random.Random(seed)
        chosen = rng.sample(records, n)

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    with open(out_p, "w", encoding="utf-8") as out:
        for rec in chosen:
            bio = rec.get("biomarkers_found")
            ml = rec.get("ml_methods_found")
            row = {
                "pmid": rec.get("pmid"),
                "year": rec.get("year"),
                "abstract": rec.get("abstract", ""),
                "category": _record_category(rec),
                "pred_biomarkers": bio if isinstance(bio, dict) else {},
                "pred_ml_methods": ml if isinstance(ml, dict) else {},
                "gold_biomarkers": None,
                "gold_ml_methods": None,
                "annotator_notes": None,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info(
        "Zapisano %d rekordów walidacyjnych -> %s (stratyfikacja=%s)",
        len(chosen),
        out_p,
        stratify,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(
        description="Eksport próbki JSONL do ręcznej walidacji biomarkerów i metod ML.",
    )
    p.add_argument("--input", default="data/processed/enriched.jsonl")
    p.add_argument("--output", default="data/validation/sample_for_annotation.jsonl")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-stratify", action="store_true", help="Czysty los zamiast stratyfikacji")
    args = p.parse_args()
    export_validation_sample(
        args.input,
        args.output,
        n=args.n,
        seed=args.seed,
        stratify=not args.no_stratify,
    )


if __name__ == "__main__":
    main()
