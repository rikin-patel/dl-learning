# Assignment 3 Workflow Guide

## Objective
Build semantically meaningful sentence-level embeddings for parallel Sanskrit-English text, while balancing:
1. High semantic alignment between aligned pairs.
2. Low embedding dimensionality.

The notebook implementation is in [Assignment3_g25ait1136.ipynb](Assignment3_g25ait1136.ipynb).

## Problem Summary
Sanskrit is low-resource and morphologically rich, so robust sentence representation is challenging. The goal is to map Sanskrit and English parallel sentences into a shared embedding space where aligned pairs are close under cosine similarity.

## Dataset Requirements
Expected files:
1. train_sa.csv
2. dev_sa.csv
3. test_sa.csv
4. train_en.csv
5. dev_en.csv
6. test_en.csv

Expected columns:
1. Sanskrit files: Source_id, Sentence_sa
2. English files: Source_id, Sentence_en

Alignment rule:
- Source_id aligns Sanskrit and English sentence pairs.

Recommended placement:
1. Same folder as the notebook, or
2. data folder under workspace root.

## What The Notebook Does
The notebook follows this pipeline:
1. Installs required dependencies.
2. Loads and validates split files.
3. Aligns Sanskrit-English pairs by Source_id.
4. Generates base multilingual sentence embeddings using LaBSE.
5. Evaluates candidate reduced dimensions on dev data.
6. Selects smallest dimension within 99% of best dev cosine.
7. Computes final average cosine similarity on test data.
8. Creates a t-SNE plot for 100 aligned test pairs.

## Environment and Dependencies
The notebook installs these packages in the first code cell:
1. pandas
2. numpy
3. scikit-learn
4. matplotlib
5. sentence-transformers

If you use a virtual environment, ensure the notebook kernel points to that environment.

## Step-by-Step Execution
Run cells top to bottom in [Assignment3_g25ait1136.ipynb](Assignment3_g25ait1136.ipynb):
1. Install dependencies cell.
2. Imports and configuration cell.
3. Data loading/alignment cell.
4. Embedding generation cell.
5. Dimension selection cell.
6. Final evaluation cell.
7. t-SNE visualization cell.

Important:
- Do not skip cells.
- Re-run all cells after changing model name, dimensions, or random seed.

## Model and Dimension Strategy
Model used:
- sentence-transformers/LaBSE

Why this choice:
1. It is multilingual and suitable for cross-lingual alignment.
2. It avoids external API usage.
3. It performs well in low-resource scenarios.

Dimension optimization approach:
1. Start with base embedding dimension from the model.
2. Apply PCA to candidate dimensions.
3. Measure dev average cosine similarity per dimension.
4. Choose smallest dimension with score >= 99% of best dev score.

This gives a practical trade-off between compactness and semantic quality.

## Evaluation Metrics
### 1. Dimensionality
Lower is preferred, provided semantic quality remains strong.

### 2. Semantic Similarity (Primary)
Average cosine similarity over aligned Sanskrit-English pairs:

$$
\text{AvgCosine} = \frac{1}{N} \sum_{i=1}^{N} \cos(\mathbf{s}_i, \mathbf{e}_i)
$$

Where:
1. $\mathbf{s}_i$ is Sanskrit embedding for pair $i$.
2. $\mathbf{e}_i$ is English embedding for pair $i$.

## t-SNE Visualization Guidelines
Notebook behavior:
1. Randomly samples up to 100 aligned test pairs.
2. Projects Sanskrit and English embeddings to 2D via t-SNE.
3. Plots Sanskrit and English points with different markers.
4. Draws faint lines between aligned pairs.

Interpretation tips:
1. Shorter pair lines generally suggest stronger cross-lingual closeness.
2. Overlap/mixing of languages can indicate shared semantic geometry.
3. t-SNE is qualitative and should not replace cosine metric reporting.

## Submission Checklist
Before final submission, confirm:
1. Notebook runs end-to-end from a clean restart.
2. Dependency installation steps are present.
3. Chosen embedding dimension is printed in output.
4. Final average cosine similarity is printed in output.
5. t-SNE plot for 100 pairs appears inline.
6. Code is reproducible and uses no external APIs.
7. Work is original and individually prepared.

## Reproducibility Notes
Use fixed randomness where possible:
1. numpy seed
2. PCA random_state
3. t-SNE random_state

This helps keep dimension search and visualization reasonably stable.

## Common Issues and Fixes
### Missing CSV Files
Symptom:
- FileNotFoundError for expected dataset files.

Fix:
1. Place all six CSV files in workspace root or data folder.
2. Check exact file names and extensions.

### Column Name Errors
Symptom:
- KeyError for Source_id or sentence columns.

Fix:
1. Verify columns are exactly Source_id/Sentence_sa and Source_id/Sentence_en.
2. Remove accidental spaces in headers.

### Out-of-Memory or Slow Runtime
Fix:
1. Reduce batch size in embedding generation.
2. Run on CPU with smaller parallel load.
3. Close other heavy applications.

### t-SNE Warnings
Fix:
1. Ensure sample size is at least moderate (the notebook already adapts perplexity).
2. Keep random_state fixed for repeatability.

## Academic Compliance Notes
1. Use only provided dataset.
2. Pre-trained models are allowed.
3. Avoid external APIs entirely.
4. Maintain originality of code and report.

## Suggested Short Report Template
You can write a concise report section with:
1. Dataset summary (train/dev/test sizes).
2. Model used (LaBSE).
3. Candidate dimensions tested.
4. Selected final dimension.
5. Final test average cosine similarity.
6. Observations from t-SNE.
7. Brief note on trade-off between compactness and semantics.

## File Reference
Detailed runnable implementation is in [Assignment3_g25ait1136.ipynb](Assignment3_g25ait1136.ipynb).
