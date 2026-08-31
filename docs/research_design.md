# WebTrace research design

## Research questions

1. How can heterogeneous public posts be converted into a reproducible corpus that preserves text, repository links, code snippets, image-derived code text, and platform metadata?
2. Which observable mechanisms are used to share source code online (repository links, inline code, screenshots, code discussion)?
3. How do classical TF-IDF models compare with transformer classifiers for harmful-language detection on the same held-out split?
4. Which error modes are associated with code-linked, text-only, and image-derived posts, and how stable are findings across platforms?

## Corpus protocol

- Use only data sources whose public APIs/terms permit collection for the intended research use.
- Preserve source URL, platform, timestamp, and acquisition date when permitted.
- Avoid storing unnecessary personal information; pseudonymize or hash user identifiers in analysis exports where appropriate.
- Record collector version, query terms, sampling window, and random seeds.
- De-duplicate exact/near-exact content before model training.

## Code-sharing taxonomy

- `repository_link`: direct repository URL.
- `inline_code`: fenced or inline source code.
- `image_code`: OCR output contains source-code signals.
- `code_discussion`: programming/code discussion without direct source sharing.
- `none`: no clear code-sharing evidence.

The taxonomy is intentionally rule-driven for initial corpus construction. A manual validation sample should be used to estimate taxonomy precision and identify ambiguous cases.

## Harmful-language experiment

Baseline:
- TF-IDF uni/bi-grams
- class-weighted logistic regression

Learned model:
- DistilBERT (default) fine-tuned for binary harmful-language classification

Evaluation:
- stratified train/validation/test splits
- macro precision, recall, F1
- confusion matrix
- per-platform and per-code-sharing-category error analysis

A 3,000-item manually reviewed sample is recommended for the full study. The final resume metric must reflect the actual measured macro-F1 from the completed experiment; no target score is hard-coded into the codebase.

## Reproducibility

All sampling and model-splitting scripts use deterministic random seeds. Raw and processed data should remain separate. Generated metrics and error-analysis tables are written to `results/`.
