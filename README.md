# Od markerów klinicznych do uczenia głębokiego — trendy w kardiologii (PubMed)

Zautomatyzowany potok: **PubMed (Entrez)** → **NER (lokalna Ollama + leksykon RegEx)** → **agregacja roczna** → **wykresy statyczne (PNG) i interaktywne (HTML, Plotly)**.

## Cel (zgodnie z założeniami projektu)

1. **Ewolucja biomarkerów** — śledzenie wzmianek o wybranych markerach (m.in. cholesterol, **non-HDL**, CRP/hs-CRP, troponiny, BNP, HbA1c) w podzbiorze literatury kardiologicznej z PubMed.
2. **Ewolucja metod analitycznych** — od modeli klasycznych (**regresja liniowa**, **Cox**, regresja logistyczna) po metody ML (random forest, SVM, sieci głębokie, XGBoost, transformery itd.), z **regułami chronologicznymi** dla metod powstałych w konkretnych latach.

Wyniki opisują **częstość wzmianek w pobranym korpusie** (nie pełna populacja wszystkich artykułów PubMed).

## Architektura (tech stack)

| Etap | Technologia |
|------|-------------|
| Pobieranie | Biopython Entrez, zapytanie domyślne + filtr roku `[dp]` |
| NER | **Domyślnie `hybrid`:** lokalny **Ollama** (`/api/chat`, `format: json`) + **słowniki YAML** i RegEx (`config/biomarkers.yaml`, `config/ml_methods.yaml`) |
| Agregacja | pandas (CSV + Parquet) |
| Wizualizacja | matplotlib/seaborn (PNG) oraz **Plotly** (HTML, interaktywne) |

### Tryby NER (`--ner-backend`)

- **`hybrid`** (domyślny): suma logiczna wykryć z LLM i z leksykonu; jeśli Ollama nie odpowiada, używany jest wyłącznie leksykon (z ostrzeżeniem w logu).
- **`ollama`**: tylko LLM — **wymaga** działającego serwera Ollama (`ollama serve`) i pobranego modelu (np. `ollama pull llama3.2`).
- **`lexicon`**: tylko RegEx (szybkie, bez GPU; przydatne do testów lub gdy brak Ollama).

### Prywatność i koszty

- Abstrakty PubMed są **publiczne**; lokalna Ollama **nie wysyła tekstu do płatnych API**.
- Dla danych wrażliwych (np. notatki kliniczne) ten sam wzorzec (lokalny LLM) nadal ma sens z perspektywy **Privacy-by-Design** — tutaj jednak źródłem są wyłącznie publiczne streszczenia.

## Szybki start

```bash
cd pubmed_pipeline
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# (Opcjonalnie) Ollama — zalecane dla zgodności z metodologią LLM
# ollama serve
# ollama pull llama3.2

python run_pipeline.py all --email twoj@email.pl
```

Pełna skala (`max_per_year` × lata × wywołania LLM) jest **czasochłonna** (np. ~4 s/abstrakt na CPU → ~15 h przy 12 tys. rekordów przy 1 workerze). Przyśpieszenie: **`--ollama-workers 4`** (lub 2–8 — sprawdź obciążenie GPU/RAM i stabilność Ollama). Szybkie testy: `--max-per-year 20`, `--end-year 2005`, lub `--ner-backend lexicon`.

## Wyniki

- `data/raw/abstracts.jsonl` — surowe abstrakty.
- `data/processed/enriched.jsonl` — encje + pola `ner_backend_effective`, opcjonalnie `ollama_model`.
- `data/processed/aggregated.csv` — szeregi czasowe.
- `data/figures/*.png` — wykresy statyczne.
- `data/figures/interactive_*.html` — **wykresy interaktywne** (otwórz w przeglądarce).

## Walidacja ręczna (precision / recall)

```bash
python run_pipeline.py validation-sample --n 20
```

Uzupełnij w pliku wyjściowym pola `gold_biomarkers` / `gold_ml_methods` i porównaj z predykcjami.

## Dostosowanie dziedziny (np. onkologia)

Zmień zapytanie w `run_pipeline.py` (`DEFAULT_QUERY`) i listy encji w YAML — reszta potoku bez zmian.

## Ograniczenia metodologiczne

- Korpus zależy od **zapytania PubMed** i limitu **`max_per_year`**.
- Metryki to **wystąpienia słów kluczowych / etykiet NER**, nie jakość kliniczna modeli ani liczba cytowań.
- LLM może pominąć lub nadmiernie oznaczyć encje — tryb `hybrid` i walidacja próbki to łagodzą.
