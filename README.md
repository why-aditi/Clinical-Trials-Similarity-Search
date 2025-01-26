# Clinical-Trials-Similarity-Search

## Overview
This project uses **FAISS** (Facebook AI Similarity Search) for creating embeddings from high-dimensional data and **SHAP** (SHapley Additive exPlanations) for interpreting feature contributions in model predictions. The goal is to identify impactful features, improve transparency, and extract insights from the dataset.

---

## Features
1. **Efficient Embedding Creation**:
   - Leverages FAISS for handling high-dimensional data efficiently.
   - Enables similarity search and compact representation of data points.

2. **SHAP for Feature Interpretability**:
   - Explains individual feature contributions to the model’s predictions.
   - Provides visual insights using SHAP summary plots.

---

## Steps to Reproduce

### 1. Prerequisites
Ensure you have the following dependencies installed:

```bash
pip install numpy pandas scikit-learn faiss-cpu shap matplotlib
