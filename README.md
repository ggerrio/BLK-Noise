# BLK NOISE

**Type it messy. Get a flat.**

BLK NOISE turns noisy, human property searches into clean, validated filters, then finds and explains matching flats. Built on LLM function calling, with the boring-but-important stuff included: audit logs, PII masking, lead automation, and a dashboard.

`black` `grey` `white` `no fluff`

**Live demo:** [add link] | **Demo video:** [add link] | **Soundcheck (eval report):** [`docs/evaluation.md`](docs/evaluation.md)

---

## Why this exists

Nobody thinks in dropdowns. People think like this:

> "4 room in Tampines, under 600k, high floor, lease still long"

Property portals make you translate that into five filters and a slider. BLK NOISE does the translation for you, shows you exactly how it understood the message, and gets out of the way.

The name is the idea: **blk** is how HDB blocks are written (Blk 123), and **noise** is the raw, messy input that goes in. Signal comes out.

```
"4 room in Tampines under 600k, high floor, lease still long"
        │
        ▼  Gemini function calling
{ town: "TAMPINES", flat_type: "4 ROOM", max_price: 600000,
  storey_min: 10, min_remaining_lease_years: 70 }
        │
        ▼  SQL filter + semantic ranking
Top flats + a short explanation that only uses real data
```

## What's inside

Each part has a plain name and a nickname. The nickname is just for fun; the code uses the plain name.

| Plain name | Nickname | What it does |
|---|---|---|
| Natural-language to filter | **Tuning** | Gemini turns a messy sentence into a schema-validated filter |
| Hybrid retrieval | **The Pit** | SQL filters first, then semantic ranking sorts what's left |
| Evaluation harness | **Soundcheck** | Labeled test queries with real accuracy numbers, run before anything ships |
| Hallucination check | **VAR** | Generated text is checked against the data; any claim not in the data gets flagged |
| Audit log | **The Setlist** | Every request logged: what came in (masked), what was parsed, what came out |
| PII masking | **Mask Up** | Phone numbers and emails are masked before logging or any LLM call |
| Lead automation | **Backstage** | Interested user leaves contact, n8n scores and notifies, Google Sheets stores |
| Dashboard | **Scoreboard** | Looker Studio: usage, accuracy, latency, leads |

Also in the repo: `docs/business/` with requirements, process map, a savings estimate (labeled as an estimate), and governance notes. Because AI that nobody can audit is just vibes.

## Tech stack

Python, FastAPI, Streamlit, SQLite, Gemini API (function calling), sentence-transformers, n8n, Google Sheets, Looker Studio, Hugging Face Spaces. All free tier.

Details in [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Run it

```bash
git clone https://github.com/ggerrio/blk-noise.git
cd blk-noise
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # add your GEMINI_API_KEY
python scripts/load_data.py     # load the dataset into SQLite
python scripts/build_embeddings.py
uvicorn app.main:app --reload   # API on :8000
streamlit run ui/app.py         # UI on :8501
```

Run the Soundcheck:

```bash
python eval/run_eval.py         # results go to eval/results/ and docs/evaluation.md
```

## Soundcheck results

> Fill this with real numbers from a real run. Delete any row you did not measure. No placeholder numbers in the published version.

| Metric | Value |
|---|---|
| Test queries | [N] |
| Exact filter match | [X]% |
| Field-level accuracy | [X]% |
| Valid schema output | [X]% |
| Hallucination rate (descriptions) | [X]% |
| Median latency | [X] s |

Run date and model: [date] / [model]. Failures are listed and analyzed in `docs/evaluation.md`, not hidden.

## Data

Singapore HDB resale flat prices, open data from [data.gov.sg](https://data.gov.sg). No scraping, no private data. See [`docs/data.md`](docs/data.md).

## Docs

| File | What's in it |
|---|---|
| [`PRD.md`](PRD.md) | Features, user flows, scope |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Stack, schema, LLM integration |
| [`RULES.md`](RULES.md) | Coding standards, AI guardrails, tone |
| [`docs/business/`](docs/business/) | Requirements, process map, savings estimate, governance |

## Known limits

- The dataset has no photos, descriptions, or distance to MRT, so "near MRT" is not supported in the MVP. The app says so instead of guessing.
- Savings numbers in `docs/business/` are assumptions, not measurements.
- Free-tier hosting can sleep or hit rate limits. The demo video is the fallback.

## About the builder

Gerrio Irfan Pratama. Informatics grad, builds AI systems and automation, plays too much football, skates badly but consistently, and codes to loud music. [GitHub](https://github.com/ggerrio) | pratamagerrio@gmail.com

## License

MIT
