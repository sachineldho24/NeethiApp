# Neethi App 🏛️

> AI-Powered Indian Legal Assistant for Common Citizens

## Overview

Neethi App is a multi-agent AI system that provides accessible legal guidance to Indian citizens. Built using CrewAI for orchestration, InLegalBERT for legal text understanding, and Qdrant for semantic search.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        OFFLINE PIPELINE                         │
│  (Run once on GPU: Data Cleaning → Chunking → Embedding → Index)│
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      QDRANT VECTOR DB                           │
│                    (150K+ legal chunks)                         │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                        ONLINE PIPELINE                          │
│  Query → Librarian (Retrieve) → Lawyer (Reason) → Response      │
└─────────────────────────────────────────────────────────────────┘
```

## Features

- **Legal Advice**: Get guidance on IPC, BNS, and relevant case precedents
- **Document Drafting**: Generate FIRs, RTI applications, and legal notices
- **Location Services**: Find nearby police stations, courts, and legal aid centers
- **Legal News**: Stay updated with latest judgments and law changes

## Project Structure

```
neethi-app/
├── data/                      # Datasets (excluded from git)
│   ├── raw/                   # Original downloads
│   └── processed/             # Cleaned, chunked data
├── models/                    # Fine-tuned models (excluded from git)
├── agents/                    # CrewAI agent definitions
├── crews/                     # Multi-agent orchestration
├── pipelines/                 # Data processing scripts
│   ├── offline/               # One-time GPU tasks
│   └── online/                # Live query processing
├── api/                       # FastAPI gateway
├── configs/                   # YAML configurations
└── tests/                     # Unit tests
```

## Quick Start

### 1. Environment Setup

```bash
# Create conda environment (Python 3.10 recommended)
conda create -n neethi python=3.10
conda activate neethi

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### 2. Configure Qdrant

Create `configs/qdrant_config.yaml`:
```yaml
qdrant:
  url: "https://your-cluster.qdrant.io"
  api_key: "your-api-key"
  collection: "neethi-legal-kb"
```

### 3. Run Data Pipeline

```bash
# Step 1: Clean IPC/BNS data
python pipelines/offline/01_data_cleaning.py

# Step 2: Create chunks
python pipelines/offline/02_chunking.py

# Step 3: Populate vector database
python pipelines/offline/04_populate_qdrant.py
```

### 4. Start API Server

```bash
uvicorn api.main:app --reload --port 8000
```

## Datasets

| Dataset | Source | Size | Purpose |
|---------|--------|------|---------|
| IPC Sections | Kaggle | 500 sections | Statute lookup |
| BNS Sections | Kaggle | 358 sections | New penal code |
| SC Judgments | Kaggle | 26,000 PDFs | Case precedents |
| IndianBailJudgments | HuggingFace | 1,200 cases | Bail prediction |
| IndicLegalQA | Mendeley | 10,000 Q&A | Fine-tuning |

## Tech Stack

- **LLM**: Llama-3-8B (4-bit quantized)
- **Embeddings**: InLegalBERT (fine-tuned)
- **Vector DB**: Qdrant Cloud
- **Orchestration**: CrewAI
- **API**: FastAPI
- **Deployment**: Lightning AI (T4 GPU)

## License

MIT License - Educational/Research Use

## Team

Developed as part of [Your University] project.
