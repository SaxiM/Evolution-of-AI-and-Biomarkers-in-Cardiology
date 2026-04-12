#Evolution of AI and Biomarkers in Cardiology

## What is this project about?

Every year, scientists publish millions of medical research papers. We wanted to answer one simple question:

> **"How has the use of artificial intelligence (AI) in medical research changed over time compared to traditional lab tests (called biomarkers)?"**

To answer this, we built a program that:
1. Downloads thousands of medical paper summaries (called abstracts) from PubMed — a big online library of medical research
2. Counts how often AI-related words appear (like "machine learning", "deep learning", "neural network")
3. Counts how often traditional biomarker words appear (like "cholesterol", "glucose", "CRP")
4. Shows the results as easy-to-read charts

---

## Step-by-step explanation

### Step 1: Fetch — Download the papers

We use a tool called **Biopython Entrez** to connect to PubMed and download paper summaries. We search for papers that mention either:
- Biomarkers (like cholesterol, glucose, CRP)
- AI/ML terms (like machine learning, deep learning, neural network)
- Clinical research (diagnosis, prognosis)

For each year between 2000 and 2025, we download up to 500 papers. The result is a file with thousands of paper summaries — each with a title, year, and abstract text.

**Output:** `data/raw/abstracts.jsonl` — a raw list of downloaded papers.

---

### Step 2: Extract — Count what appears in each paper

For each paper, we search the abstract text for specific keywords. We look for two types of things:

#### Biomarkers (traditional lab tests)
These are substances in your body that doctors measure to check your health:

| Biomarker | What it means |
|-----------|---------------|
| **CRP** | C-reactive protein — a sign of inflammation in your body |
| **Troponin** | A protein that rises when heart muscle is damaged |
| **Cholesterol** | Fat in blood; HDL is "good", LDL is "bad" |
| **HbA1c** | Hemoglobin A1c — shows average blood sugar over time (for diabetes) |
| **IL-6** | Interleukin-6 — an inflammatory signal |
| **Glucose** | Blood sugar — the main fuel in your blood |
| **BNP** | Brain natriuretic peptide — a hormone released when the heart is strained |
| **Ferritin** | Iron storage protein — shows how much iron you have stored |
| **Albumin** | A protein made by the liver |
| **Creatinine** | A waste product from muscles, used to check kidney function |

#### AI/ML Methods (artificial intelligence techniques)
These are computer methods used to analyze medical data:

| ML Method | In simple words |
|-----------|----------------|
| **Logistic Regression** | Like drawing a straight line through data points to predict a yes/no outcome (e.g., "will this patient get better?") |
| **Random Forest** | Combines many decision trees (like asking many experts and voting) to make predictions |
| **SVM** (Support Vector Machine) | Finds the best way to separate two groups of data points with a line or curve |
| **Neural Network** | A computer program inspired by how the brain works — connections between simple "neurons" |
| **Deep Learning** | A neural network with many layers — can learn very complex patterns |
| **Transformer** | The technology behind ChatGPT and BERT — uses "attention" to understand context |
| **XGBoost** | A powerful type of gradient boosting — great for structured data |
| **Naive Bayes** | A simple probability-based classifier |
| **Decision Tree** | Like a flowchart: ask questions一步步 to reach a prediction |
| **Clustering** | Groups similar data points together without being told the groups in advance |

**Output:** `data/processed/enriched.jsonl` — each paper now has a list of what was found in it.

---

### Step 3: Aggregate — Combine the data by year

We group all papers by their publication year and count:
- Total papers per year
- Papers that mention at least one biomarker
- Papers that mention at least one ML method
- How many times each individual biomarker and ML method appears

We also calculate two important rates (per 1,000 papers):
- **Biomarker rate** — how many papers mention biomarkers
- **ML rate** — how many papers mention ML methods

And we compute:
- **AI Penetration Index** = ML rate ÷ Biomarker rate
  - If it's above 1.0 → AI methods are mentioned more often than traditional biomarkers
  - If it's below 1.0 → biomarkers are still mentioned more often

**Output:** `data/processed/aggregated.csv` — one row per year with all the numbers.

---

### Step 4: Visualize — Make charts

We automatically generate 5 charts that show the trends over time.

---

## The 5 charts explained

### Chart 1: AI/ML vs Biomarkers Over Time (`01_ai_vs_biomarkers.png`)

**What it shows:** Two lines — one for AI/ML methods and one for biomarkers. The lines are "smoothed" (3-year average) to make trends easier to see. The orange area under the ML line shows how much AI has grown.

**How to read it:**
- If the ML line goes up → AI methods are becoming more popular in research
- If the biomarker line is flat or slowly changing → traditional lab tests remain steady
- The gap between the lines tells us which type of mention is more common

**Example:** In 2000, the biomarker line was higher than ML. In 2025, the ML line is much higher — showing AI has "caught up and passed" traditional biomarkers in research interest.

---

### Chart 2: AI Penetration Index (`02_ai_penetration_index.png`)

**What it shows:** One line that divides ML rate by biomarker rate each year. A horizontal dashed line at y=1.0 marks "parity" — where both are equally mentioned.

**How to read it:**
- **Above the dashed line** → AI methods are mentioned MORE than biomarkers this year
- **Below the dashed line** → Biomarkers are still dominant
- **Crossing the line** → The moment AI became more discussed than traditional lab tests

**Key insight:** Before ~2018, the line was mostly below 1.0. After 2020, it shoots up above 1.0 — showing a dramatic shift toward AI in medical research.

---

### Chart 3: Year-over-Year ML Growth (`03_ml_yoy_growth.png`)

**What it shows:** Green and red bars showing how much the ML mention rate changed from one year to the next.

**How to read it:**
- **Green bar** → ML mentions grew compared to last year (good!)
- **Red bar** → ML mentions decreased compared to last year
- **Taller bar** → bigger change

**Key insight:** 2020 had enormous green bars — during COVID-19, AI research exploded.

---

### Chart 4: ML Method Breakdown (`04_ml_method_breakdown.png`)

**What it shows:** A stacked area chart where each color represents a different AI method. The height shows how many papers mentioned that method each year.

**How to read it:**
- Each color band = one ML method
- Wider band in a year = more papers mentioned that method that year
- Watching which band grows tells us which methods are gaining popularity

**Key trends:**
- Before 2015: Logistic regression and SVM dominated
- 2015–2020: Neural networks and deep learning started growing
- After 2020: Transformers (BERT, GPT) became very popular

---

### Chart 5: Biomarker Breakdown (`05_biomarker_breakdown.png`)

**What it shows:** Same type of chart as #4, but for traditional biomarkers.

**How to read it:**
- Each color band = one biomarker
- CRP and glucose are consistently the most mentioned
- Newer biomarkers like HbA1c have grown as diabetes research increased

---

## How to run the project

### Prerequisites

You need Python installed. Open your terminal/command prompt and go to the project folder:

```bash
cd pubmed_pipeline
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run everything at once (recommended)

```bash
python run_pipeline.py all --email your@email.com --start-year 2000 --end-year 2025
```

This will: fetch papers → extract entities → aggregate → generate charts.

### Run step by step

```bash
# 1. Download papers from PubMed
python run_pipeline.py fetch --email your@email.com

# 2. Count biomarkers and ML methods in each paper
python run_pipeline.py extract

# 3. Combine everything by year
python run_pipeline.py aggregate

# 4. Make the charts
python run_pipeline.py visualize
```

---

## Files in this project

| File | What it is |
|------|-----------|
| `data/raw/abstracts.jsonl` | Raw downloaded papers (12,000+ papers) |
| `data/processed/enriched.jsonl` | Papers with counted biomarkers and ML methods |
| `data/processed/aggregated.csv` | Year-by-year totals and rates |
| `data/figures/*.png` | The 5 charts |

---

## What the numbers mean

- **12,210 abstracts** were downloaded for years 2000–2025
- **~35%** of papers mention at least one biomarker
- **~16%** of papers mention at least one AI/ML method (growing fast)
- **Neural networks** and **deep learning** are the fastest-growing AI methods
- **CRP** and **glucose** remain the most mentioned traditional biomarkers

---

## Adding new terms

The config files let you add new keywords without touching the code:

- `config/biomarkers.yaml` — add new biomarkers (e.g., "vitamin_d")
- `config/ml_methods.yaml` — add new ML methods (e.g., "k-nearest neighbors")

Just follow the same format: the canonical name as the key, and a list of lowercase aliases as the value.


