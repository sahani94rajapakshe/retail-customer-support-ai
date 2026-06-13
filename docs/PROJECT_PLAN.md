# AI-Powered Omnichannel Customer Support & Analytics Platform

**Project Plan Document**

| Field | Detail |
|-------|--------|
| **Project** | AI-Powered Omnichannel Customer Support & Analytics Platform |
| **Timeline** | Sep 2024 – Present |
| **Repository** | [retail-customer-support-ai](https://github.com/sahani94rajapakshe/retail-customer-support-ai) |
| **Status** | Phase 1 in progress |

---

## Executive Summary

A production-grade AI system for automating customer support across **Email, Live Chat, WhatsApp, and Social Media** while providing **business intelligence** and **customer insights**.

The platform combines fine-tuned transformer models, retrieval-augmented generation (RAG), and a hybrid **Python + ASP.NET Core** microservices architecture deployed on **Azure**. The end result is a mini enterprise support platform — comparable in concept to Zendesk, Freshworks, or Salesforce Service Cloud — enhanced with modern AI, RAG, and intelligent automation.

---

## 1. Business Problem

A medium-to-large retail company receives thousands of customer inquiries daily through multiple channels:

- Email
- Website Live Chat
- WhatsApp
- Facebook Messenger
- Instagram DMs

### Pain Points

| Problem | Impact |
|---------|--------|
| Repetitive questions consume agent time | High labor cost, low productivity |
| Customers wait hours for responses | Poor satisfaction, churn risk |
| Management lacks visibility into issues | Reactive, not data-driven decisions |
| Support costs scale linearly with growth | Unsustainable unit economics |
| Important complaints are missed | Reputation damage, escalations |

### Example: Manual vs AI-Assisted Flow

**Customer message:**
> "My order #12345 hasn't arrived yet. Where is it?"

**Manual process (today):**
1. Agent reads ticket
2. Checks order system
3. Finds shipment status
4. Composes and sends reply

**Time:** 5–10 minutes per ticket

**AI-assisted process (target):**
1. AI identifies intent (`OrderTracking`)
2. System fetches order status via API
3. LLM generates personalized response
4. Auto-send or agent review

**Time:** ~5 seconds (automated) or ~1 minute (agent-assisted)

| Metric | Before | Target |
|--------|--------|--------|
| Average response time | 5–10 min | **5 sec – 1 min** |
| Automation rate | 0% | **60–80%** of repetitive tickets |

---

## 2. Project Outcomes

### 2.1 Automation (60–80% of repetitive tickets)

Automatically handle high-volume, low-complexity intents:

- Order tracking (WISMO)
- Return requests
- Refund inquiries
- Product questions
- Shipping status
- FAQ-style policy questions

### 2.2 Agent Assistance (Copilot)

For tickets requiring human judgment:

- **Suggested replies** — AI drafts first response
- **Conversation summaries** — Long threads condensed to key points
- **Next-best-action recommendations** — e.g., offer discount, escalate, refund

### 2.3 Analytics & Business Intelligence

Management dashboard visibility into:

- Top complaints and trending issues
- Customer sentiment over time
- Agent performance metrics
- Ticket volume by channel and intent
- Product-level issue detection
- AI automation rate and accuracy

### 2.4 Final Deliverable

At project completion, the platform will support this end-to-end flow:

```
Customer sends message (Email / Chat / WhatsApp / Social)
        → AI understands intent
        → AI analyzes sentiment
        → AI searches company knowledge (RAG)
        → AI generates response
        → Auto-reply OR escalate to agent
        → Agent receives summary + suggestions (if escalated)
        → Management views analytics and trends
```

**Business result:** Reduced support costs, faster resolution, improved customer satisfaction.

---

## 3. System Architecture

```
Customer Channels
        │
        ▼
Omnichannel Gateway
        │
        ▼
Message Processing Service
        │
        ▼
AI Engine
 ├── Intent Detection        (DistilBERT)
 ├── Sentiment Analysis      (RoBERTa)
 ├── RAG Search              (Sentence Transformers + Vector DB)
 ├── Response Generation     (GPT-4o / Llama 3 / Mistral)
 └── Escalation Logic        (Business rules)
        │
        ▼
Support Dashboard             (Agent Copilot UI)
        │
        ▼
Analytics & Reporting         (Management Dashboard)
```

### Technology Stack

| Layer | Technologies |
|-------|-------------|
| **AI / NLP** | Python, Hugging Face, DistilBERT, RoBERTa, Sentence Transformers, OpenAI / Llama / Mistral |
| **AI Service** | FastAPI, Docker |
| **Backend** | ASP.NET Core, Clean Architecture, CQRS |
| **Messaging** | RabbitMQ |
| **Cache** | Redis |
| **Database** | PostgreSQL |
| **Vector Store** | pgvector / Azure AI Search |
| **Frontend** | React or Blazor (Agent Dashboard + Analytics) |
| **Cloud** | Azure (App Service, Container Apps, Key Vault) |
| **CI/CD** | GitHub Actions |

---

## 4. User Journeys

### Scenario 1: Order Tracking (Fully Automated)

**Customer:**
> "Where is my order #12345?"

| Step | Action | Output |
|------|--------|--------|
| 1 | Message arrives (Email / Chat / WhatsApp) | Ticket created in database |
| 2 | Intent Detection | `{ "intent": "OrderTracking", "confidence": 0.98 }` |
| 3 | Order API lookup | `{ "status": "Out for Delivery" }` |
| 4 | LLM response generation | Personalized reply with order status |
| 5 | Auto-send | Ticket resolved |

**Sample response:**
> Hi John, your order #12345 is currently out for delivery and should arrive today. Thank you!

---

### Scenario 2: Angry Customer (Escalation)

**Customer:**
> "I ordered 2 weeks ago and still haven't received anything. Terrible service!"

| Step | Action | Output |
|------|--------|--------|
| 1 | Sentiment Analysis | `{ "sentiment": "Negative", "score": 0.93 }` |
| 2 | Escalation rule | Negative score > 0.85 → **Escalate to human** |
| 3 | Agent Dashboard | Flagged as **HIGH PRIORITY — Angry Customer** |
| 4 | Agent Copilot | AI suggests empathetic response draft |
| 5 | Agent review | Human sends approved reply |

---

### Scenario 3: Product Question (RAG)

**Customer:**
> "Does this laptop support 32GB RAM?"

| Step | Action | Output |
|------|--------|--------|
| 1 | Intent Detection | `ProductInquiry` |
| 2 | RAG pipeline | Embed question → vector search knowledge base |
| 3 | Retrieved context | Product catalog: "Supports up to 64GB RAM" |
| 4 | LLM grounded response | Accurate answer without hallucination |

**Sample response:**
> Yes, this model supports up to 64GB RAM.

---

## 5. Main Modules

### Module 1: Omnichannel Gateway

**Purpose:** Collect and normalize messages from all customer channels.

**Channels:** Email, Live Chat, WhatsApp, Facebook Messenger, Instagram DMs

**Outcome:** Every inbound message becomes a **unified ticket** regardless of source channel.

---

### Module 2: Ticket Management

**Purpose:** Central system of record for all customer conversations.

**Core entities:**

```
Ticket
├── Id
├── CustomerId
├── Channel
├── Subject
├── Status          (Open | InProgress | Resolved | Escalated)
├── Priority        (Low | Medium | High | Critical)
└── CreatedDate

Message
├── Id
├── TicketId
├── Sender          (Customer | Agent | AI)
├── Content
└── Timestamp
```

**Outcome:** Single, complete customer conversation history across all channels.

---

### Module 3: Intent Classification

**Model:** DistilBERT (fine-tuned)

**Intent classes:**

| Intent | Example |
|--------|---------|
| OrderTracking | "Where is my order?" |
| Refund | "I want my money back" |
| ReturnRequest | "I want to return my shoes" |
| Complaint | "Your service is horrible" |
| ProductInquiry | "Does this support 32GB RAM?" |
| PaymentIssue | "My card was charged twice" |
| TechnicalIssue | "The app won't load" |
| Other | Unclassified |

**Example output:**
```json
{
  "intent": "ReturnRequest",
  "confidence": 0.96
}
```

**Outcome:** Automatic ticket routing and workflow selection.

---

### Module 4: Sentiment Analysis

**Model:** RoBERTa (fine-tuned)

**Classes:** Positive, Neutral, Negative

**Example output:**
```json
{
  "sentiment": "Negative",
  "score": 0.95
}
```

**Outcome:** Immediate identification of unhappy customers; triggers escalation rules.

---

### Module 5: AI Response Generator

**Models:** GPT-4o, Llama 3, or Mistral (configurable)

**Input:**
```json
{
  "intent": "OrderTracking",
  "orderStatus": "Delivered",
  "customerName": "John"
}
```

**Output:**
> Your package was delivered yesterday. Please check your doorstep or mailbox.

**Outcome:** Automatic or suggested replies for high-confidence, low-risk intents.

---

### Module 6: RAG Knowledge System

**Knowledge sources:**

- FAQ documents
- Return & refund policies
- Product catalog
- Product manuals
- Shipping guides

**Pipeline:**
```
Question → Embedding → Vector Search → Relevant Documents → LLM → Grounded Answer
```

**Outcome:** Accurate, factual answers grounded in company knowledge — reduced hallucination.

---

### Module 7: Agent Copilot

**Features:**

| Feature | Description |
|---------|-------------|
| Suggested Reply | AI writes first draft for agent review |
| Ticket Summary | Condenses long threads (e.g., 100 messages → 5 lines) |
| Next Best Action | Recommends actions (offer discount, escalate, refund) |

**Example summary:**
> Customer requested refund. Order delayed by 10 days. Customer frustrated.

**Outcome:** Agents resolve tickets faster with AI assistance.

---

### Module 8: Analytics Dashboard

**Management metrics:**

| Category | Metrics |
|----------|---------|
| **Ticket** | Total, Resolved, Open, Escalated |
| **AI** | Automation rate, Intent accuracy, Response accuracy |
| **Customer** | Sentiment trends, CSAT, Top complaints |
| **Agent** | Avg handling time, Resolution time, Tickets per agent |
| **Product** | Issue trends by SKU/category |

**Outcome:** Data-driven support operations and proactive issue detection.

---

## 6. Database Design

```
Customer
├── Id
├── Name
├── Email
└── Phone

Ticket
├── Id
├── CustomerId        → Customer
├── Status
├── Priority
├── Channel
├── Subject
└── CreatedDate

Message
├── Id
├── TicketId          → Ticket
├── Sender
├── Content
└── Timestamp

AIAnalysis
├── Id
├── TicketId          → Ticket
├── Intent
├── Sentiment
├── Confidence
└── CreatedDate

KnowledgeDocument
├── Id
├── Title
├── Content
├── Embedding         (vector)
└── SourceType        (FAQ | Policy | Catalog | Manual)
```

**Database:** PostgreSQL with pgvector extension for embedding storage and similarity search.

---

## 7. AI Models

| Task | Model | Purpose |
|------|-------|---------|
| Intent Classification | **DistilBERT** | Understand customer request type |
| Sentiment Analysis | **RoBERTa** | Measure customer mood |
| Embeddings | **Sentence Transformers** | Vector search for RAG |
| Response Generation | **GPT-4o / Llama 3 / Mistral** | Generate natural language replies |
| Summarization | **T5-small / GPT** | Condense long conversations |

---

## 8. Advanced Features (Differentiators)

| Feature | Description | Business Value |
|---------|-------------|----------------|
| **Conversation Summarization** | 100-message thread → 5-line summary | Faster agent onboarding to ticket |
| **Churn Prediction** | Predict if customer is likely to leave | Proactive retention |
| **Complaint Trend Detection** | e.g., "Shipping issues up 40% this week" | Early operational alerts |
| **Voice Support** | Speech-to-text for call center | Omnichannel completeness |
| **AI Quality Score** | Rate responses: Helpful, Polite, Accurate | Continuous AI improvement |

---

## 9. Implementation Phases

### Phase 1 — Data Foundation ✅

| Task | Status |
|------|--------|
| Collect Enron + Twitter CS datasets | Done |
| Data cleaning pipeline (`stage2_clean_data.py`) | Done |
| Baseline auto-labeling (`stage3A_auto_label.py`) | Done |
| Unified labeled dataset + train/val/test splits (`stage3B`) | Done |
| Response pairs for T5 (`stage3C`) | Done |

**Deliverable:** Clean, labeled training data ready for fine-tuning.

---

### Phase 2 — Model Training 🔄

| Task | Status |
|------|--------|
| Fine-tune DistilBERT for intent (8 classes) (`stage4`) | Scripts ready |
| Fine-tune RoBERTa for sentiment (`stage5`) | Scripts ready |
| Evaluate vs zero-shot baselines (`stage6`) | Scripts ready |
| Fine-tune T5-small for response generation (`stage7`) | Scripts ready |
| Build RAG pipeline (`stage8`) | Scripts ready |

> See [PROGRESS.md](PROGRESS.md) for live status updates.

---

### Phase 3 — AI Microservice 🔄

| Task | Status |
|------|--------|
| FastAPI service (`/classify`, `/sentiment`, `/generate`, `/rag`) | Done |
| Model loading and inference optimization | Done (fine-tuned + fallback) |
| Docker containerization | Done |

> See `services/ai-service/` and [PROGRESS.md](PROGRESS.md).

**Deliverable:** Deployable Python AI service with OpenAPI docs.

---

### Phase 4 — Backend Platform

| Task | Status |
|------|--------|
| ASP.NET Core API (Clean Architecture + CQRS) | Pending |
| PostgreSQL schema + migrations | Pending |
| Ticket & message management | Pending |
| RabbitMQ event bus | Pending |
| Redis caching | Pending |
| Integration with AI microservice | Pending |

**Deliverable:** Production-style .NET backend with ticket lifecycle management.

---

### Phase 5 — Omnichannel Gateway

| Task | Status |
|------|--------|
| Email integration (SendGrid / SMTP) | Pending |
| Web chat widget | Pending |
| WhatsApp (Twilio / Meta API) | Pending |
| Facebook Messenger + Instagram DMs | Pending |

**Deliverable:** Unified ticket creation from all channels.

---

### Phase 6 — Agent Copilot & Dashboard

| Task | Status |
|------|--------|
| Agent dashboard UI | Pending |
| Suggested reply + summary features | Pending |
| Escalation workflow | Pending |
| Management analytics dashboard | Pending |

**Deliverable:** Agent and management UIs with live ticket and analytics views.

---

### Phase 7 — Cloud Deployment

| Task | Status |
|------|--------|
| Azure App Service (.NET API) | Pending |
| Azure Container Apps (Python AI) | Pending |
| Azure PostgreSQL + pgvector | Pending |
| Azure Key Vault (secrets) | Pending |
| GitHub Actions CI/CD | Pending |

**Deliverable:** Live demo environment on Azure.

---

## 10. Success Metrics

| Metric | Target |
|--------|--------|
| Ticket automation rate | **60–80%** |
| Intent classification F1 | **> 0.85** |
| Sentiment classification F1 | **> 0.90** |
| Automated response time | **< 5 seconds** |
| Agent-assisted response time | **< 1 minute** |
| RAG answer accuracy (human eval) | **> 90%** |
| Customer satisfaction (CSAT) | Measurable improvement vs baseline |

---

## 11. Resume & Portfolio Impact

This project demonstrates proficiency in:

- **AI Engineering** — End-to-end ML pipeline from data to deployment
- **NLP** — Intent classification, sentiment analysis, summarization
- **LLMs** — Response generation with GPT / Llama / Mistral
- **RAG** — Vector search + grounded generation
- **Vector Databases** — pgvector / Azure AI Search
- **ASP.NET Core** — Clean Architecture, CQRS
- **Microservices** — Python AI service + .NET backend
- **Messaging** — RabbitMQ event-driven architecture
- **Caching** — Redis
- **PostgreSQL** — Relational + vector storage
- **Production AI Systems** — Escalation logic, quality scoring, monitoring

---

## 12. Target Repository Structure

```
RetailSupport.AI/
├── data/                          # Raw + cleaned datasets (gitignored)
├── outputs/                       # Labeled data, eval reports (gitignored)
├── models/                        # Fine-tuned checkpoints (gitignored)
├── scripts/
│   ├── stage2_clean_data.py       # Phase 1 — data cleaning
│   ├── stage3A_auto_label.py      # Phase 1 — weak supervision labeling
│   ├── stage4_finetune_intent.py  # Phase 2 — DistilBERT intent
│   ├── stage5_finetune_sentiment.py
│   ├── stage6_finetune_t5.py      # Phase 2 — response generation
│   └── stage7_build_rag_index.py  # Phase 2 — RAG embeddings
├── services/
│   └── ai-service/                # Phase 3 — FastAPI
├── backend/
│   └── RetailSupport.Api/         # Phase 4 — ASP.NET Core
├── gateway/                       # Phase 5 — channel adapters
├── dashboard/                     # Phase 6 — Agent + Analytics UI
├── docs/
│   ├── PROJECT_PLAN.md            # This document
│   ├── ARCHITECTURE.md
│   └── EVALUATION.md
├── docker-compose.yml
├── README.md
└── requirements.txt
```

---

## 13. Comparison to Enterprise Platforms

| Capability | Zendesk / Freshworks | This Platform |
|------------|---------------------|---------------|
| Multi-channel tickets | Yes | Yes |
| Intent routing | Rule-based / add-on | **Custom fine-tuned DistilBERT** |
| AI responses | Generic LLM add-on | **RAG-grounded + domain fine-tuned** |
| Sentiment detection | Add-on | **Built-in RoBERTa pipeline** |
| Agent copilot | Limited | **Custom summaries + next-best-action** |
| Analytics | Yes | Yes |
| Open source / ownable | No | **Yes — full codebase** |

This project is a **research and portfolio implementation** of enterprise-grade support automation — demonstrating the same architectural patterns used in commercial platforms, with modern AI at the core.

---

*Document version: 1.0 | Last updated: June 2025*
