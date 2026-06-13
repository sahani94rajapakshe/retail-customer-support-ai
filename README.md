# AI-Powered Omnichannel Customer Support & Analytics Platform

![Status](https://img.shields.io/badge/status-in%20development-yellow)
![Phase](https://img.shields.io/badge/phase-3%20in%20progress-blue)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

A production-grade AI system for automating retail customer support across **Email, Live Chat, WhatsApp, and Social Media** while providing business intelligence and customer insights.

> **Status:** In development — Phase 3 (FastAPI AI microservice) in progress. .NET backend not yet deployed.  
> **Full project plan:** [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md) · **Live tracker:** [docs/PROGRESS.md](docs/PROGRESS.md)
## Vision

Automate **60–80%** of repetitive support tickets, assist agents with AI copilot features, and provide management analytics — reducing response times from minutes to seconds.

## Current Progress

> **Live tracker:** [docs/PROGRESS.md](docs/PROGRESS.md)

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Data cleaning & weak-supervision labeling | **Complete** |
| 2 | Fine-tune DistilBERT, RoBERTa, T5 + RAG | **In progress** |
| 3 | FastAPI AI microservice | **In progress** |
| 4 | ASP.NET Core backend (CQRS, RabbitMQ, PostgreSQL) | Planned |
| 5 | Omnichannel gateway (Email, Chat, WhatsApp, Social) | Planned |
| 6 | Agent copilot & analytics dashboard | Planned |
| 7 | Azure cloud deployment | Planned |

## What's implemented now

**Phase 1**
1. Clean raw support datasets (Enron emails + Twitter CS)
2. Auto-label with 8 intent classes + sentiment (weak supervision)
3. Unified dataset with train/val/test splits
4. Customer → agent response pair extraction for T5

**Phase 2** (scripts ready — run locally with GPU recommended)
5. Fine-tune DistilBERT (intent), RoBERTa (sentiment), T5-small (responses)
6. Model evaluation report
7. RAG knowledge index (Sentence Transformers)

**Phase 3** (FastAPI AI service — run locally)
8. REST API for intent, sentiment, RAG, response generation, and escalation

## Project structure

```
RetailSupport.AI/
├── data/
│   └── knowledge/           # Sample FAQ for RAG (tracked in git)
├── docs/
│   ├── PROJECT_PLAN.md      # Full architecture & module plan
│   └── PROGRESS.md          # Living progress tracker
├── services/
│   └── ai-service/          # FastAPI microservice (Phase 3)
│       ├── app/
│       │   ├── main.py
│       │   └── services/
│       ├── Dockerfile
│       └── requirements.txt
├── outputs/                 # Labeled data, splits, reports (gitignored)
├── models/                  # Fine-tuned checkpoints (gitignored)
├── scripts/
│   ├── config.py
│   ├── stage2_clean_data.py
│   ├── stage3A_auto_label.py
│   ├── stage3B_prepare_dataset.py
│   ├── stage3C_extract_response_pairs.py
│   ├── stage4_finetune_intent.py
│   ├── stage5_finetune_sentiment.py
│   ├── stage6_evaluate_models.py
│   ├── stage7_finetune_t5.py
│   └── stage8_build_rag_index.py
└── requirements.txt
```
## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/sahani94rajapakshe/retail-customer-support-ai.git
cd retail-customer-support-ai
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 2. Add datasets

Place raw datasets in the `data/` folder:

| File | Source |
|------|--------|
| `emails.csv` | [Enron Email Dataset](https://www.kaggle.com/datasets/wcukierski/enron-email-dataset) |
| `twcs.csv` | [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) |

> Datasets are excluded from this repository due to size and licensing. Download them from Kaggle and place them in `data/` before running the pipeline.

## Usage

Run from the project root:

```bash
# Phase 1 — Data pipeline
python scripts/stage2_clean_data.py
python scripts/stage3A_auto_label.py --max-rows 5000
python scripts/stage3B_prepare_dataset.py
python scripts/stage3C_extract_response_pairs.py --max-rows 5000

# Phase 2 — Model training
python scripts/stage4_finetune_intent.py --epochs 3
python scripts/stage5_finetune_sentiment.py --epochs 3
python scripts/stage6_evaluate_models.py
python scripts/stage7_finetune_t5.py --epochs 3
python scripts/stage8_build_rag_index.py --query "return policy"

# Quick dev run (small sample)
python scripts/stage4_finetune_intent.py --max-samples 200 --epochs 1

# Phase 3 — Start AI microservice
cd services/ai-service
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Or with Docker (from project root)
docker compose up --build ai-service
```

API docs: http://localhost:8000/docs

**Example requests:**

```bash
curl -X POST http://localhost:8000/analyze -H "Content-Type: application/json" -d "{\"text\": \"My order has not arrived. Terrible service!\"}"

curl -X POST http://localhost:8000/rag/search -H "Content-Type: application/json" -d "{\"query\": \"return policy\"}"

curl -X POST http://localhost:8000/generate/response -H "Content-Type: application/json" -d "{\"text\": \"Where is my order?\"}"
```

See [docs/PROGRESS.md](docs/PROGRESS.md) for current status and next steps.

## License
Use responsibly. Raw datasets are subject to their original licenses and terms of use.
