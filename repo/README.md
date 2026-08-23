# Countability in Old English: Corpus Extraction and Analysis Pipeline

Code accompanying the MA thesis *[Thesis Title]* (Poppy, [Year]),
Saarland University, supervised by Prof. Annemarie Verkerk and
Dr. Kurt Erbach.

This repository contains the full extraction, lemmatization, feature-
construction, statistical testing, and clustering pipeline used to
study noun countability marking in Old English, based on ten texts
from the York-Toronto-Helsinki Parsed Corpus of Old English Prose
(YCOE).

## Pipeline order

Run in this order; each script's output is the next script's input.

1. **`extract.py`** — parses the `.psd` (bracketed constituency-tree)
   YCOE files with a custom recursive-descent parser and extracts
   every noun-phrase instance, recording the head noun, its case, and
   any co-occurring determiner, quantifier, or numeral.
   Input: `*.psd` files. Output: `*_nouns.csv` (per text), `all_texts_nouns.csv`.

2. **`lemmatize.py`** — decodes the ASCII special-character escape
   convention used in the YCOE source files, and matches each noun
   token to a lemma against an Old English word-form-to-lemma resource.
   Output: `all_texts_nouns_lemmatized.csv`.

3. **`build_features.py`** — aggregates token-level co-occurrence data
   to one proportional feature vector per lemma (bare, determiner,
   quantifier, numeral, genitive proportions), applying a minimum-
   frequency threshold.
   Output: `lemma_feature_vectors.csv`.

4. **`build_lemma_by_text.py`** — the same feature vectors broken out
   per lemma per text, rather than pooled, for cross-genre comparison.

5. **`contingency_tables.py`** — token-level 2×2 contingency tables and
   chi-square tests of independence for the co-occurrence relationships
   tested in the thesis (numeral/other-modifier; genitive/numeral;
   genitive/determiner), used in place of Pearson correlation on
   lemma-aggregated proportions.

6. **`cluster.py`** — the original DBSCAN clustering procedure
   (density-based, parameters selected via grid search).

7. **`fixed_k_clustering.py`** — the fixed-*k* agglomerative (Ward)
   clustering robustness check, run on arcsine-square-root-transformed,
   standardized feature vectors, at *k*=8 (matching Allan 1980's class
   count) and *k*=12 (matching Grimm & Wahlang 2021's coarsest PDE
   model). Includes the *k*-sweep used to select these values and the
   raw-vs-transformed robustness comparison.

8. **`build_figures.py`** — cherry-picked lemma-level visualizations
   (scatterplot, heatmap, contrast bar chart) used in the Discussion
   chapter.

9. **`concordance.py`** — a lookup tool for retrieving and reading the
   original sentence context for any lemma (optionally filtered by
   text), used throughout for manual validation of automatically
   extracted patterns against the underlying YCOE sentences.

## Requirements

See `requirements.txt`. Python 3.10+.

```
pip install -r requirements.txt
```

## Data

The YCOE `.psd` and `.pos` source files are not redistributed here;
they are available from the
[YCOE website](https://www-users.york.ac.uk/~lang22/YCOE/) under its
own license. Place the relevant `.psd`/`.pos` files in the working
directory before running `extract.py`.

## Example output

`example_output/` contains sample figures and a sample clustered
feature-vector CSV from `fixed_k_clustering.py`, included so the
pipeline's output format is visible without needing to first obtain
the YCOE source files.

## Citation

If you use this code, please cite the accompanying thesis:

> [Poppy]. ([Year]). *[Thesis title]*. MA thesis, Saarland University.

## License

MIT License (see `LICENSE`), unless otherwise required by the terms
of the YCOE data license for any redistributed derived data.
