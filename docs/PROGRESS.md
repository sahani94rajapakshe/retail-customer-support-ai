# AI-Powered Omnichannel Customer Support & Analytics Platform

**Living progress tracker** — update this document as each task is completed.

| Field | Value |
|-------|-------|
| **Last updated** | 2025-06-13 |
| **Current phase** | Phase 2 — Model Training |
| **Overall progress** | ~30% |

> Master plan: [PROJECT_PLAN.md](PROJECT_PLAN.md)

---

## Quick status

| Phase | Name | Status | Progress |
|-------|------|--------|----------|
| 1 | Data Foundation | **Complete** | 100% |
| 2 | Model Training | **In progress** | 40% |
| 3 | AI Microservice (FastAPI) | Not started | 0% |
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

### Run Phase 2 pipeline

```bash
# From project root (after Phase 1 scripts)
python scripts/stage3B_prepare_dataset.py
python scripts/stage3C_extract_response_pairs.py --max-rows 5000

python scripts/stage4_finetune_intent.py --epochs 3
python scripts/stage5_finetune_sentiment.py --epochs 3
python scripts/stage6_evaluate_models.py

python scripts/stage7_finetune_t5.py --epochs 3 --max-samples 2000
python scripts/stage8_build_rag_index.py
```

Use `--max-rows` / `--max-samples` for faster dev runs; omit for full training.

---

## Phase 3 — AI Microservice ⬜

| Task | Status |
|------|--------|
| FastAPI service (`/classify`, `/sentiment`, `/generate`, `/rag`) | Not started |
| Model loading + inference optimization | Not started |
| Docker containerization | Not started |

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

> Baseline metrics from `stage6_evaluate_models.py` on 100-row sample. Re-run labeling at scale, fine-tune, then re-evaluate.

---

## What's next

1. **Re-label at scale:** `python scripts/stage3A_auto_label.py --max-rows 5000` (or `--full`)
2. **Re-prepare splits:** `python scripts/stage3B_prepare_dataset.py`
3. **Fine-tune models:** run `stage4` and `stage5` (GPU recommended)
4. **Re-evaluate:** `python scripts/stage6_evaluate_models.py` and update metrics above
5. **Start Phase 3** — FastAPI microservice wrapping fine-tuned models

---

## Changelog

| Date | Update |
|------|--------|
| 2025-06-13 | RAG index built; baseline evaluation run; dataset splits created (100 rows) |
| 2025-06-13 | Phase 2 started: config, fine-tuning, eval, RAG scripts added |
| 2025-06-13 | Phase 1 complete; auto-labeling expanded to 8 intent classes |
| 2025-06-13 | Initial repo pushed to GitHub |
