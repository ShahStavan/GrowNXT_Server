<div align="center">

# 🚀 GrowNXT Financial AI Engine
### *Autonomous Financial Intelligence & Equity Research Platform*

[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/LangGraph-Self--RAG-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Vector Index](https://img.shields.io/badge/Index-HNSW%20Dense%20Vector-green.svg)](https://github.com/nmslib/hnswlib)
[![Model Support](https://img.shields.io/badge/LLM-Ollama%20%7C%20Groq%20%7C%20Gemini-purple.svg)](https://ollama.ai/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

*Automated, zero-hallucination fundamental financial analyst reports built from raw company filings using Parent-Child Hybrid Self-RAG.*

---

</div>

## 📌 Executive Summary & Platform Overview

**GrowNXT** turns raw financial statements (annual reports, balance sheets, quarterly results) into **clear, institutional-grade financial analyst reports**.

Designed to run smoothly even on standard laptop hardware (Intel i5 CPU, 8GB RAM), GrowNXT uses lightweight open-source AI models (`qwen2.5:1.5b`) without running into context length limits or math errors.

---

## 🏛 System Architecture & Processing Workflow

Here is how GrowNXT processes fundamental filings into verified financial reports:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ 1. INPUT DATA: Raw Financial Filings (Annual PDFs + Structured JSON Filings)  │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 2. HYBRID SEARCH ENGINE (Metadata Filtered Vector + BM25 Search)             │
│    ├── Parent-Child Chunker (Search ~300ch child -> Return ~1,024ch parent)   │
│    ├── Dense HNSW Vector Search (Semantic similarity)                        │
│    ├── Sparse BM25 Keyword Search (Exact term & code matching)               │
│    └── Reciprocal Rank Fusion (RRF) Reranking                                │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 3. AI WORKFLOW: LangGraph Stateful Self-RAG Machine                          │
│    ├── Step 1: Executive Summary & Corporate Profile                         │
│    ├── Step 2: Core Business Segments & Revenue Engine                       │
│    ├── Step 3: Strategic Expansion & Capital Allocation Pipeline             │
│    ├── Step 4: Competitive Moat, Concessions & Market Footprint              │
│    ├── Step 5: Financial Performance & Growth Metrics                        │
│    ├── Step 6: DuPont Return Decomposition (ROE & ROCE Analysis)             │
│    ├── Step 7: Capital Structure & Solvency Analysis                         │
│    ├── Step 8: Investment Thesis & Strategic Risk Audit                      │
│    └── Corrective Self-RAG Loop (Query Rewriting on low confidence)           │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 4. OUTPUT REPORT: Final Verified Financial Report (report.md & REST API)      │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔥 Core Architectural Pillars & Features

### 1. 🔀 Hybrid Search & Reciprocal Rank Fusion (RRF)
Combines **Dense HNSW Vector Search** (for semantic concepts) with **Sparse BM25 Keyword Search** (for exact financial codes & product terms). Results are fused using **Reciprocal Rank Fusion (RRF)**:

$$\text{RRF Score}(d) = \frac{1}{60 + \text{Rank}_{\text{Vector}}(d)} + \frac{1}{60 + \text{Rank}_{\text{BM25}}(d)}$$

### 2. 🧩 Parent-Child Hierarchical Document Chunking
- **Child Chunks (~300 chars)**: Used for high-precision HNSW vector and BM25 search matching.
- **Parent Chunks (~1,024 chars)**: Retained context blocks returned to the LLM for maximum output quality.

### 3. 🏷 Metadata Section Filtering
Filters vector and keyword search strictly by section tags (`company_overview`, `expansion_plans`, `balance_sheet`, `dupont_analysis`), eliminating cross-section noise contamination.

### 4. 🔄 Corrective Self-RAG Reflection Loop (CRAG)
Evaluates retrieval confidence scores. If confidence falls below threshold ($top\_sim < 0.30$), the system triggers an automatic **Query Rewriter** to expand the query with financial domain synonyms before regenerating.

### 5. 📐 Ground-Truth Math Engine (DuPont ROE & ROCE Analysis)
AI models often fail at basic math. GrowNXT calculates **DuPont Return on Equity (ROE)** and **Return on Capital Employed (ROCE)** directly using exact code formulas:

$$\text{ROE} = \text{Net Profit Margin} \times \text{Asset Turnover} \times \text{Financial Leverage}$$

$$\text{ROCE} = \frac{\text{Operating Profit (EBIT)}}{\text{Total Equity} + \text{Total Debt}}$$

### 6. 💡 Plain-English Investor Summaries
Translates technical financial terms into plain English for everyday investors:
- **Operations**: *"Uses cash from Airports to fund new Green Hydrogen projects."*
- **Capex**: *"Spending heavily on new projects; watch for project completion dates."*
- **Moat**: *"30 to 50 year government contracts protect against local competition."*

---

## 🛠 Enterprise Directory Structure

```
GrowNXT_Server/
├── api/                        # REST API Layer (Flask App & Search)
├── core/                       # Prompts, Config & LLM Provider Setup
│   ├── config.py               # Settings validator
│   ├── llm_config.py           # Local / Cloud LLM selector
│   └── prompt_registry.py      # Prompts for each report section
├── services/                   # Business Logic & AI Engines
│   ├── rag_engine.py           # Parent-child chunker, hybrid vector+BM25 search & DuPont math
│   ├── graph_pipeline.py       # LangGraph Self-RAG state machine with CRAG loop
│   └── analysis_service.py     # Main report orchestrator
├── scripts/                    # Web scrapers & batch utilities
├── .env                        # Environment settings
├── requirements.txt            # Python dependencies
└── run.py                      # Server entry point
```

---

## ⚡ Quick Start & Environment Guide

### 1. Installation
```powershell
# Create & activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 2. Configure Environment (`.env`)
Create a `.env` file in the root folder:
```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5:1.5b
OLLAMA_BASE_URL=http://localhost:11434
```

### 3. Start Local Ollama Server
```powershell
ollama pull qwen2.5:1.5b
ollama serve
```

### 4. Run Flask API Server
```powershell
.\venv\Scripts\python.exe run.py
```
Server runs at `http://127.0.0.1:5000`.

---

## 🌐 REST API Endpoints & Specification

- **Generate Report**: `GET /api/stocks/<symbol>/analysis`
- **Search Stock**: `GET /api/search?q=<query>`

---

## 🧪 Standalone CLI Verification

Run a quick test report generation directly from the command line:

```powershell
.\venv\Scripts\python.exe -c "from pathlib import Path; from services.graph_pipeline import SelfRAGReportGraph; graph = SelfRAGReportGraph(Path('D:/Stock_Fundamental/data/adanient')); report = graph.execute_pipeline(); print(report)"
```

---

## 📜 Software Licensing & Distribution

Distributed under the **MIT License**.
