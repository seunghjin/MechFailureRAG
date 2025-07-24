# Mechanical NLP System

This project implements an advanced NLP system for analyzing mechanical failure data and generating solution text using fine-tuned large language models, with retrieval-augmented generation (RAG).

## Folder Structure

- `data/` - Raw and processed data files (Excel, CSV, splits)
- `notebooks/` - Jupyter notebooks for exploration and reporting
- `scripts/` - Python scripts for data processing, EDA, and pipeline steps
- `models/` - Model checkpoints, configs, and logs
- `rag/` - Retrieval-Augmented Generation (RAG) components and vector DB
- `api/` - FastAPI service for model inference
- `frontend/` - Streamlit web app for user interface
- `tests/` - Unit and integration tests

## Getting Started

1. Place your raw Excel data in the `data/` folder.
2. Run scripts in `scripts/` for data preparation and analysis.
3. Use notebooks in `notebooks/` for interactive exploration and reporting.
4. Follow the pipeline for model training, RAG integration, and deployment. 