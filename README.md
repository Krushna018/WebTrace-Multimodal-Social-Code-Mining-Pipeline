# WebTrace — Multimodal Social & Code Mining Pipeline

<p align="center">
  <b>A reproducible research pipeline for public-post corpus construction, code-sharing analysis, multimodal extraction, and harmful-language classification.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/NLP-Text%20Classification-orange" alt="NLP">
  <img src="https://img.shields.io/badge/Multimodal-OCR-green" alt="Multimodal">
  <img src="https://img.shields.io/badge/Models-3-purple" alt="Models">
  <img src="https://img.shields.io/badge/Research-Reproducible-red" alt="Research">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
</p>

---

## Overview

**WebTrace** is a research-oriented Python framework for building and analyzing corpora of public online posts, with particular emphasis on **how software code is shared and represented across text, repositories, and images**.

The pipeline combines:

* public-post collection and dataset import
* text normalization and preprocessing
* repository-link extraction
* inline code detection
* OCR-based image-code extraction
* code-sharing taxonomy assignment
* harmful-language annotation
* classical machine-learning classification
* neural text classification
* transformer fine-tuning
* structured error analysis

The project is designed to support **reproducible corpus construction and comparative NLP experimentation** rather than relying on a single model or data source.

---

## Research Contributions

WebTrace is organized around three primary research components.

### 1. Corpus Construction

The pipeline transforms heterogeneous public-post data into a normalized research corpus.

```text
Public Posts / Research Datasets
              │
              ▼
        Data Import
              │
              ▼
      Text Normalization
              │
       ┌──────┼─────────┐
       ▼      ▼         ▼
 Repository  Inline    Image
   Links      Code      OCR
       │      │         │
       └──────┼─────────┘
              ▼
      Code-Sharing Taxonomy
              │
              ▼
        Processed Corpus
```

The resulting corpus preserves relevant provenance and derived signals while separating collection, preprocessing, annotation, and modelling stages.

### 2. Multimodal & Code-Aware Analysis

WebTrace treats code sharing as more than plain text.

It supports three primary signals:

| Signal                     | Description                                                                               |
| -------------------------- | ----------------------------------------------------------------------------------------- |
| **Repository-linked code** | Identifies links to code-hosting repositories and optionally enriches repository metadata |
| **Inline code**            | Detects code fragments embedded directly within post text                                 |
| **OCR-derived code**       | Extracts text/code from images using OCR                                                  |

These signals can then be mapped to a structured **code-sharing taxonomy**.

### 3. NLP Model Evaluation

The project provides three executable modelling paths:

| Model                            | Purpose                                                                |
| -------------------------------- | ---------------------------------------------------------------------- |
| **TF-IDF + Logistic Regression** | Transparent classical NLP baseline                                     |
| **PyTorch TextCNN**              | Neural text-classification model trained from scratch                  |
| **Transformer**                  | Context-aware classification using a pretrained transformer checkpoint |

This enables comparison between a lightweight statistical baseline, a locally trained neural model, and transformer-based fine-tuning.

---

## Research Pipeline

The complete WebTrace workflow can be summarized as:

```text
             ┌──────────────────────────┐
             │ Public / Imported Data    │
             └────────────┬─────────────┘
                          │
                          ▼
             ┌──────────────────────────┐
             │ Collection & Import      │
             └────────────┬─────────────┘
                          │
                          ▼
             ┌──────────────────────────┐
             │ Preprocessing & Cleaning │
             └────────────┬─────────────┘
                          │
             ┌────────────┼────────────┐
             ▼            ▼            ▼
       Repository      Inline Code     OCR
          Links          Signals      Signals
             │            │            │
             └────────────┼────────────┘
                          ▼
             ┌──────────────────────────┐
             │ Code-Sharing Taxonomy    │
             └────────────┬─────────────┘
                          │
                          ▼
             ┌──────────────────────────┐
             │ Annotation / Validation  │
             └────────────┬─────────────┘
                          │
                          ▼
             ┌──────────────────────────┐
             │ Model Training           │
             │                          │
             │ TF-IDF + LR              │
             │ TextCNN                  │
             │ Transformer              │
             └────────────┬─────────────┘
                          │
                          ▼
             ┌──────────────────────────┐
             │ Evaluation & Error       │
             │ Analysis                 │
             └──────────────────────────┘
```

---

## Project Structure

```text
WebTrace/
│
├── src/
│   └── webtrace/
│       │
│       ├── collectors/
│       │   ├── mastodon.py
│       │   ├── github.py
│       │   └── csv_import.py
│       │       └── Data collection and repository enrichment
│       │
│       ├── models/
│       │   ├── baseline.py
│       │   │   └── TF-IDF + Logistic Regression
│       │   ├── textcnn.py
│       │   │   └── PyTorch TextCNN
│       │   └── transformer.py
│       │       └── Hugging Face transformer fine-tuning
│       │
│       ├── evaluation.py
│       │   └── Metrics and structured model evaluation
│       │
│       ├── ocr.py
│       │   └── OCR-based image text/code extraction
│       │
│       ├── preprocess.py
│       │   └── Text normalization and corpus preprocessing
│       │
│       ├── schema.py
│       │   └── Corpus and annotation schema
│       │
│       ├── synthetic.py
│       │   └── Synthetic dataset generation
│       │
│       └── taxonomy.py
│           └── Code-sharing category assignment
│
├── scripts/
│   ├── collect_mastodon.py
│   ├── generate_demo.py
│   ├── prepare_corpus.py
│   ├── sample_for_annotation.py
│   ├── train_baseline.py
│   ├── train_textcnn.py
│   ├── predict_textcnn.py
│   └── train_transformer.py
│
├── docs/
│   └── research_design.md
│
├── tests/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── results/
│
├── requirements.txt
└── README.md
```

---

# Installation

## Requirements

* Python **3.10+**
* pip
* virtual environment
* Internet connection for remote collection and first-time transformer checkpoint download

### Optional

OCR functionality requires the system-level **Tesseract** executable.

---

## Setup

### 1. Create a virtual environment

**macOS / Linux**

```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

---

# Quick Local Demo

WebTrace includes a synthetic-data workflow so the core pipeline can be tested **without accessing the internet or collecting real user-generated content**.

Generate a demonstration corpus:

```bash
python scripts/generate_demo.py --n 5000
```

Prepare the corpus:

```bash
python scripts/prepare_corpus.py data/raw/demo_posts.csv
```

Train the baseline:

```bash
python scripts/train_baseline.py
```

This provides a fast way to validate the preprocessing, corpus-generation, and modelling pipeline before working with external data.

---

# Public Mastodon Collection

WebTrace supports collection from permitted public Mastodon sources.

Example:

```bash
python scripts/collect_mastodon.py \
  --instance mastodon.social \
  --pages 10
```

Then prepare the collected corpus:

```bash
python scripts/prepare_corpus.py data/raw/mastodon_posts.csv
```

Collection volume depends on the instance's API behavior, availability, and rate limits.

For larger studies, multiple permitted sources or existing research datasets can be combined while preserving **source provenance**.

---

# Annotation & Validation

WebTrace provides a workflow for creating a manually reviewed validation sample.

For example:

```bash
python scripts/sample_for_annotation.py \
  --data data/processed/corpus.csv \
  --n 3000
```

The generated annotation sheet can contain fields such as:

```text
manual_label_harmful
review_notes
```

For research use, annotation should be supported by a clearly defined annotation guide.

A stronger study design can include:

* multiple independent annotators
* blinded annotation where appropriate
* inter-annotator agreement
* adjudication of disagreements
* documented label definitions
* analysis of ambiguous cases

---

# Code-Sharing Taxonomy

WebTrace separates different ways in which software code can appear in online content.

The pipeline can identify signals including:

```text
                 Code Sharing
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
      Repository   Inline Code    Image
        Link                       │
                                   ▼
                                  OCR
```

This allows code-sharing behavior to be analyzed independently from the harmful-language classification task.

The taxonomy is implemented in:

```text
src/webtrace/taxonomy.py
```

---

# NLP Models

WebTrace contains **three executable modelling paths**.

## 1. TF-IDF + Logistic Regression

The classical baseline represents documents using TF-IDF features and performs classification with logistic regression.

Implementation:

```text
src/webtrace/models/baseline.py
```

Train:

```bash
python scripts/train_baseline.py \
  --data data/processed/corpus.csv
```

Typical outputs include:

```text
results/baseline_metrics.json
results/baseline_errors.csv
```

along with the serialized model.

### Why include a classical baseline?

A simple baseline provides an interpretable reference point for determining whether more complex models provide meaningful improvements.

---

## 2. PyTorch TextCNN

WebTrace also includes a **TextCNN implemented in PyTorch**.

Unlike the transformer pipeline, this model is trained from scratch and does not require downloading pretrained model weights.

Implementation:

```text
src/webtrace/models/textcnn.py
```

Train:

```bash
python scripts/train_textcnn.py \
  --data data/processed/corpus.csv \
  --output results/textcnn
```

Run inference:

```bash
python scripts/predict_textcnn.py \
  --model-dir results/textcnn \
  "example post text"
```

The TextCNN provides a middle ground between the lightweight classical baseline and a pretrained transformer.

---

## 3. Transformer Fine-Tuning

WebTrace supports transformer-based text classification through Hugging Face.

The default configuration uses:

```text
distilbert-base-uncased
```

Train:

```bash
python scripts/train_transformer.py \
  --data data/processed/corpus.csv \
  --output results/transformer
```

The first run may download the selected pretrained checkpoint.

Once cached locally, subsequent runs can reuse the checkpoint without downloading it again.

---

# Model Comparison

The three modelling paths provide complementary perspectives:

```text
                 Classification Models
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
     TF-IDF            TextCNN        Transformer
       +                 │                │
 Logistic Regression     │        Pretrained Context
        │                │                │
        └────────────────┼────────────────┘
                         ▼
                 Comparative Evaluation
```

| Model                        | Training     | Main role                    |
| ---------------------------- | ------------ | ---------------------------- |
| TF-IDF + Logistic Regression | Classical ML | Transparent baseline         |
| TextCNN                      | From scratch | Local neural representation  |
| Transformer                  | Fine-tuning  | Context-aware representation |

This design makes it possible to study whether increasingly expressive representations produce different classification behavior.

---

# Evaluation

Model evaluation is handled through:

```text
src/webtrace/evaluation.py
```

The evaluation pipeline is intended to capture more than a single headline metric.

Depending on the configured experiment, evaluation can include:

* accuracy
* precision
* recall
* F1-score
* confusion matrix
* prediction errors
* class-level performance
* structured error analysis

The project also stores model errors so that incorrect predictions can be inspected qualitatively.

---

# Error Analysis

Quantitative metrics alone do not explain **why** a classifier succeeds or fails.

WebTrace therefore supports structured error analysis using saved prediction results.

Example categories for analysis include:

```text
False Positive
      │
      ├── Context ambiguity
      ├── Quoted language
      └── Benign technical terminology

False Negative
      │
      ├── Implicit harmful language
      ├── Context dependence
      ├── Slang / informal language
      └── Code-related context
```

The exact categories should be defined according to the research question and annotation protocol rather than assumed universally.

---

# Reproducibility

Reproducibility is a central design principle of WebTrace.

The project separates the pipeline into explicit stages:

```text
Collection
    ↓
Preprocessing
    ↓
Feature / Signal Extraction
    ↓
Annotation
    ↓
Model Training
    ↓
Evaluation
    ↓
Error Analysis
```

For reproducible experiments, record:

* source dataset/version
* collection date
* API/source configuration
* preprocessing configuration
* random seeds
* model configuration
* training parameters
* checkpoint/model version
* evaluation split
* software dependencies

This is particularly important when working with continuously changing online platforms.

---

# Research Ethics & Responsible Data Use

WebTrace is intended for research involving **publicly accessible online content**, but public availability does not automatically remove ethical or legal considerations.

When collecting real data:

* Use only sources whose API terms and applicable policies permit the intended research.
* Do **not** bypass authentication, access controls, rate limits, or platform restrictions.
* Minimize collection and storage of personal information.
* Store only information necessary for the research question.
* Consider whether raw posts need to be retained at all.
* Follow applicable institutional ethics/IRB requirements.
* Document data provenance and collection procedures.
* Respect deletion requests and platform policies where applicable.
* Take additional care when analyzing potentially harmful or sensitive language.
* Avoid publishing unnecessary personally identifying information from collected content.

The included synthetic-data workflow allows development and pipeline testing without requiring real user-generated content.

---

# Data Provenance

For research datasets, provenance should be maintained throughout the pipeline.

A recommended flow is:

```text
Source
  ↓
Raw Dataset
  ↓
Preprocessing
  ↓
Derived Dataset
  ↓
Annotation
  ↓
Train / Validation / Test Split
  ↓
Model Results
```

Keeping these stages separate makes it easier to reproduce experiments and identify how preprocessing or annotation decisions affect model outcomes.

---

# Testing

Run the complete test suite with:

```bash
pytest -q
```

Tests are intended to validate core components such as:

* preprocessing
* schema handling
* taxonomy assignment
* synthetic data generation
* model components
* evaluation utilities

---

# End-to-End Example

A typical local workflow is:

```bash
# 1. Generate synthetic data
python scripts/generate_demo.py --n 5000

# 2. Prepare the corpus
python scripts/prepare_corpus.py data/raw/demo_posts.csv

# 3. Train classical baseline
python scripts/train_baseline.py \
  --data data/processed/corpus.csv

# 4. Train TextCNN
python scripts/train_textcnn.py \
  --data data/processed/corpus.csv \
  --output results/textcnn

# 5. Run TextCNN inference
python scripts/predict_textcnn.py \
  --model-dir results/textcnn \
  "example post text"

# 6. Fine-tune transformer
python scripts/train_transformer.py \
  --data data/processed/corpus.csv \
  --output results/transformer

# 7. Run tests
pytest -q
```

---

