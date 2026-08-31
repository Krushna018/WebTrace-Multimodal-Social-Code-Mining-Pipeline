# WebTrace: Multimodal Social & Code Mining Pipeline

A research-oriented Python project for constructing reproducible corpora of public online posts, analyzing how source code is shared, and benchmarking harmful-language classifiers.

The project is designed around three research contributions:

1. **Corpus construction:** collect/import public posts, normalize text, extract repository links and code snippets, and assign a code-sharing taxonomy.
2. **Multimodal/code-aware analysis:** support repository-linked, inline-code, and OCR-derived image-code signals.
3. **NLP evaluation:** compare a TF-IDF + logistic-regression baseline with a transformer classifier and conduct structured error analysis.

## Important research-ethics note

Only collect data from sources whose API terms and applicable policies allow the intended use. Do not bypass authentication, access controls, or platform restrictions. Minimize stored personal information and follow institutional ethics requirements for any real research involving user-generated content.

## Project structure

```text
WebTrace/
├── src/webtrace/
│   ├── collectors/       # Mastodon, GitHub enrichment, CSV import
│   ├── models/           # TF-IDF baseline + transformer fine-tuning
│   ├── evaluation.py
│   ├── ocr.py
│   ├── preprocess.py
│   ├── schema.py
│   ├── synthetic.py
│   └── taxonomy.py
├── scripts/
│   ├── collect_mastodon.py
│   ├── generate_demo.py
│   ├── prepare_corpus.py
│   ├── sample_for_annotation.py
│   ├── train_baseline.py
│   └── train_transformer.py
├── docs/
│   ├── research_design.md
│   └── resume_alignment.md
├── tests/
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
pip install -e .
```

If you want OCR support, also install the system `tesseract` executable.

## Quick local demo (no internet needed)

Generate a synthetic corpus to validate the pipeline:

```bash
python scripts/generate_demo.py --n 5000
python scripts/prepare_corpus.py data/raw/demo_posts.csv
python scripts/train_baseline.py
```

The demo data is explicitly synthetic and must not be described as public social-media data on a resume.

## Collect public Mastodon data

```bash
python scripts/collect_mastodon.py --instance mastodon.social --pages 10
python scripts/prepare_corpus.py data/raw/mastodon_posts.csv
```

Collection volume depends on API availability/rate limits. For a larger corpus, combine multiple permitted sources or research datasets and preserve source provenance.

## Build a 3,000-item validation sheet

```bash
python scripts/sample_for_annotation.py --data data/processed/corpus.csv --n 3000
```

Reviewers can fill `manual_label_harmful` and `review_notes`. For real research, define an annotation guide and ideally use multiple annotators with agreement analysis.

## Baseline NLP model

```bash
python scripts/train_baseline.py --data data/processed/corpus.csv
```

Outputs include:

- `results/baseline_metrics.json`
- `results/baseline_errors.csv`
- serialized TF-IDF/logistic-regression model

## Transformer model

```bash
python scripts/train_transformer.py \
  --data data/processed/corpus.csv \
  --model distilbert-base-uncased \
  --epochs 2
```

The first run downloads the model from Hugging Face.

## Tests

```bash
pytest -q
```

## Resume-use guidance

The codebase is built to support a research-style resume project around:

- automated public-data collection and preprocessing;
- code/repository/image-code signals;
- taxonomy construction;
- NLP baselines and transformer classification;
- rigorous evaluation and error analysis.

Do **not** claim a specific corpus size, manually validated sample size, or macro-F1 until the corresponding real experiment has been run. The architecture supports the target study, but measured results must come from the completed dataset and experiment.

## Models included

WebTrace contains three executable modelling paths rather than model placeholders:

1. **TF-IDF + Logistic Regression baseline** (`src/webtrace/models/baseline.py`) for a transparent classical NLP comparison.
2. **PyTorch TextCNN** (`src/webtrace/models/textcnn.py`) that is trained fully from scratch and saved locally; it does not require downloading pretrained model weights.
3. **Transformer fine-tuning** (`src/webtrace/models/transformer.py`) using Hugging Face `AutoModelForSequenceClassification` (DistilBERT by default) for the transformer-based experiment described in the research design.

Train the local neural model:

```bash
python scripts/train_textcnn.py --data data/processed/corpus.csv --output results/textcnn
```

Run inference after training:

```bash
python scripts/predict_textcnn.py --model-dir results/textcnn "example post text"
```

Fine-tune the transformer (internet is required the first time to obtain the selected pretrained checkpoint unless it is already cached):

```bash
python scripts/train_transformer.py --data data/processed/corpus.csv --output results/transformer
```

Model weights generated from real experiments are intentionally not pre-labelled as research results in this repository. Evaluation metrics should be reported only after running the corresponding experiment on the final corpus.
