"""PubMed data fetching via Biopython Entrez."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

from Bio import Entrez

logger = logging.getLogger(__name__)

# NCBI mandates a valid email
Entrez.email = "researcher@example.com"


def fetch_pubmed_abstracts(
    query: str,
    start_year: int,
    end_year: int,
    max_per_year: int = 500,
    email: str = "researcher@example.com",
    output_path: str = "data/raw/abstracts.jsonl",
) -> None:
    """Fetch PubMed abstracts for a date-range filtered query.

    Saves results as JSON Lines (one JSON object per line) containing
    PMID, year, and abstract text.

    Args:
        query: PubMed search query string.
        start_year: Start of date range (inclusive).
        end_year: End of date range (inclusive).
        max_per_year: Maximum abstracts to fetch per year.
        email: Email address for NCBI Entrez (required).
        output_path: Output path for the .jsonl file.

    Raises:
        RuntimeError: If no records are found or network errors persist.
    """
    Entrez.email = email
    output_p = Path(output_path)
    output_p.parent.mkdir(parents=True, exist_ok=True)

    records_written = 0

    with open(output_p, "w", encoding="utf-8") as outfile:
        for year in range(start_year, end_year + 1):
            year_records = 0
            retries = 0
            max_retries = 3

            while retries < max_retries:
                try:
                    # Build date-filtered query: append year constraint
                    year_query = f"{query} AND {year}[dp]"
                    logger.info("Searching: %s (max %d)", year_query, max_per_year)

                    handle = Entrez.esearch(
                        db="pubmed",
                        term=year_query,
                        retmax=max_per_year,
                        sort="date",
                    )
                    search_result = Entrez.read(handle)
                    handle.close()

                    id_list = search_result.get("IdList", [])
                    if not id_list:
                        logger.info("  No results for year %d", year)
                        break

                    total = int(search_result.get("Count", len(id_list)))
                    logger.info("  Found %d total, fetching up to %d", total, len(id_list))

                    # Fetch abstracts in batches to avoid giant requests
                    batch_size = 100
                    for i in range(0, len(id_list), batch_size):
                        batch_ids = id_list[i : i + batch_size]
                        fetch_handle = Entrez.efetch(
                            db="pubmed",
                            id=",".join(batch_ids),
                            rettype="abstract",
                            retmode="xml",
                        )
                        records = Entrez.read(fetch_handle)
                        fetch_handle.close()

                        pubmed_abstracts = records.get("PubmedArticle", [])
                        for article in pubmed_abstracts:
                            medline = article.get("MedlineCitation", {})
                            article_data = medline.get("Article", {})
                            abstract_texts = article_data.get("Abstract", {})
                            abstract_list = abstract_texts.get("AbstractText", [])

                            # Handle both single string and list of sections
                            if isinstance(abstract_list, list):
                                abstract_joined = " ".join(
                                    str(s) for s in abstract_list if s
                                )
                            else:
                                abstract_joined = str(abstract_list) if abstract_list else ""

                            if not abstract_joined:
                                continue

                            pmid = str(medline.get("PMID", ""))
                            pub_year = year  # Use the search year

                            record_out = {
                                "pmid": pmid,
                                "year": pub_year,
                                "abstract": abstract_joined,
                            }
                            outfile.write(json.dumps(record_out, ensure_ascii=False) + "\n")
                            year_records += 1
                            records_written += 1

                        # Respect NCBI rate limits: max 3 requests/second
                        time.sleep(0.34)

                    logger.info("  Year %d: %d abstracts written", year, year_records)
                    break

                except Exception as exc:
                    retries += 1
                    logger.warning(
                        "  Network error for year %d (attempt %d/%d): %s",
                        year, retries, max_retries, exc,
                    )
                    if retries >= max_retries:
                        logger.error("  Max retries exceeded for year %d, skipping", year)
                    else:
                        time.sleep(2)

    logger.info("Done. Total records written: %d -> %s", records_written, output_p)

    if records_written == 0:
        raise RuntimeError("No records were fetched. Check your query and date range.")
