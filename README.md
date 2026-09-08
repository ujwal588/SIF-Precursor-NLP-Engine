# SIF Precursor NLP Engine

An AI/NLP engine that classifies industrial safety reports as SIF-potential (Serious Injury/Fatality) vs non-SIF, tags IOGP Life-Saving Rules, and visualizes SIF density via an interactive dashboard.

## Problem Statement
Oil India Limited collects thousands of Unsafe Act/Unsafe Condition (UA/UC) and near-miss reports. Currently, these are triaged manually on a monthly/quarterly basis, which delays critical safety interventions. This project automates the triage process to identify SIF-potential reports in real-time.

## Solution Overview
- **Classification**: Fine-tuned DistilBERT model to classify SIF vs non-SIF.
- **NER Tagging**: spaCy NER pipeline to tag IOGP Life-Saving Rules (Energy Isolation, Hot Work, Confined Space, etc.).
- **Explainability**: SHAP/LIME to explain why a report was flagged as SIF.
- **Dashboard**: Streamlit dashboard ranking sites by SIF-precursor density.

## Tech Stack
- Python 3.9+
- PyTorch / Transformers (DistilBERT)
- spaCy (NER)
- SHAP / LIME (Explainability)
- FastAPI (Backend API)
- Streamlit (Dashboard)
- Docker (Deployment)

## Setup Instructions
*(Coming soon)*
