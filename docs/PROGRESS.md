# AI-Powered Omnichannel Customer Support & Analytics Platform

**Living progress tracker** — update this document as each task is completed.

| Field | Value |
|-------|-------|
| **Last updated** | 2025-06-13 |
| **Current phase** | Phase 3 — AI Microservice |
| **Overall progress** | ~40% |

> Master plan: [PROJECT_PLAN.md](PROJECT_PLAN.md)

---

## Quick status

| Phase | Name | Status | Progress |
|-------|------|--------|----------|
| 1 | Data Foundation | **Complete** | 100% |
| 2 | Model Training | **In progress** | 40% |
| 3 | AI Microservice (FastAPI) | **In progress** | 80% |
| 4 | Backend Platform (.NET) | Not started | 0% |
| 5 | Omnichannel Gateway | Not started | 0% |
| 6 | Agent Copilot & Dashboard | Not started | 0% |
| 7 | Cloud Deployment (Azure) | Not started | 0% |

---

## Phase 1 — Data Foundation ✅

| Task | Script | Status | Notes |
|------|--------|--------|-------|
| Collect Enron + TWCS datasets | — | Done | Local copies in `data/` |
| Data cleaning pipeline | `scripts/stage2_clean_data.py` | Done | Outputs `data/*_cleaned.csv` |
| Auto-labeling (weak supervision) | `scripts/stage3A_auto_label.py` | Done | 8 intent classes, normalized sentiment |
| Unified dataset + splits | `scripts/stage3B_prepare_dataset.py` | Done | 100 rows (re-run after full labeling) |
| Response pairs for T5 | `scripts/stage3C_extract_response_pairs.py` | Done | Requires `data/twcs.csv` |

**Phase 1 deliverable:** Labeled, split training data ready for fine-tuning.

---

## Phase 2 — Model Training 🔄

| Task | Script | Status | Notes |
|------|--------|--------|-------|
| Fine-tune DistilBERT (intent, 8 classes) | `scripts/stage4_finetune_intent.py` | **Ready to run** | Run locally; GPU recommended |
| Fine-tune RoBERTa (sentiment) | `scripts/stage5_finetune_sentiment.py` | **Ready to run** | Run locally; GPU recommended |
| Evaluate models vs baseline | `scripts/stage6_evaluate_models.py` | Done (baseline) | Re-run after fine-tuning |
| Fine-tune T5-small (response gen) | `scripts/stage7_finetune_t5.py` | **Ready to run** | Requires `stage3C` output |
| Build RAG index | `scripts/stage8_build_rag_index.py` | **Done** | Index at `models/rag-index/` |

**Phase 2 deliverable:** Fine-tuned models + evaluation report (F1, accuracy, BLEU/ROUGE).

---

## Phase 3 — AI Microservice 🔄

| Task | Location | Status | Notes |
|------|----------|--------|-------|
| FastAPI service | `services/ai-service/app/main.py` | **Done** | OpenAPI at `/docs` |
| `POST /classify/intent` | main.py | Done | Fine-tuned or zero-shot fallback |
| `POST /classify/sentiment` | main.py | Done | Fine-tuned or pretrained fallback |
| `POST /analyze` | main.py | Done | Intent + sentiment + escalation |
| `POST /generate/response` | main.py | Done | T5 or template fallback |
| `POST /rag/search` | main.py | Done | Vector search over knowledge base |
| `POST /rag/answer` | main.py | Done | Grounded answer from top docs |
| `GET /health` | main.py | Done | Model load status |
| Docker containerization | `Dockerfile`, `docker-compose.yml` | Done | `docker compose up ai-service` |

### Run the API

```bash
cd services/ai-service
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

API docs: http://localhost:8000/docs

**Phase 3 deliverable:** Deployable Python AI service with OpenAPI docs.

---

## Phase 4 — Backend Platform ⬜

| Task | Status |
|------|--------|
| ASP.NET Core API (Clean Architecture + CQRS) | Not started |
| PostgreSQL schema + migrations | Not started |
| Ticket & message management | Not started |
| RabbitMQ + Redis | Not started |
| AI microservice integration | Not started |

---

## Phase 5 — Omnichannel Gateway ⬜

| Task | Status |
|------|--------|
| Email (SendGrid / SMTP) | Not started |
| Web chat widget | Not started |
| WhatsApp (Twilio / Meta) | Not started |
| Facebook + Instagram DMs | Not started |

---

## Phase 6 — Agent Copilot & Dashboard ⬜

| Task | Status |
|------|--------|
| Agent dashboard UI | Not started |
| Suggested reply + ticket summary | Not started |
| Escalation workflow | Not started |
| Management analytics dashboard | Not started |

---

## Phase 7 — Cloud Deployment ⬜

| Task | Status |
|------|--------|
| Azure App Service (.NET) | Not started |
| Azure Container Apps (Python AI) | Not started |
| Azure PostgreSQL + pgvector | Not started |
| GitHub Actions CI/CD | Not started |

---

## Metrics (fill after Phase 2 evaluation)

| Metric | Baseline (zero-shot) | Fine-tuned | Target |
|--------|---------------------|------------|--------|
| Intent accuracy | 0.50 (n=10) | — | > 85% |
| Intent F1 (macro) | 0.26 (n=10) | — | > 0.85 |
| Sentiment accuracy | 1.00 (n=10) | — | > 90% |
| Sentiment F1 (macro) | 1.00 (n=10) | — | > 0.90 |
| Response BLEU | — | — | > 0.30 |
| RAG index | 8 docs indexed | Done | Expand knowledge base |

---

## What's next

1. **Run fine-tuning:** `stage4` + `stage5` (GPU recommended) — API auto-loads fine-tuned models when available
2. **Test API:** `uvicorn app.main:app --reload` → http://localhost:8000/docs
3. **Start Phase 4** — ASP.NET Core backend calling this AI service

---

## Changelog

| Date | Update |
|------|--------|
| 2025-06-13 | Phase 3: FastAPI microservice with intent, sentiment, RAG, response, escalation endpoints + Docker |
| 2025-06-13 | RAG index built; baseline evaluation run; dataset splits created (100 rows) |
| 2025-06-13 | Phase 2 started: config, fine-tuning, eval, RAG scripts added |
| 2025-06-13 | Phase 1 complete; auto-labeling expanded to 8 intent classes |
| 2025-06-13 | Initial repo pushed to GitHub |
