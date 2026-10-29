# Data

## Published here

| File | What it is | Used for |
|---|---|---|
| `derived/sinhala_char_counts.tsv` | Count and MLE of each of the 59 basic Sinhala characters across 2,268,895 characters (white space excluded) | Experiment 1 |
| `derived/sinhala_word_frequencies.tsv` | Frequency and rank of all 74,369 word types (681,233 tokens) | Experiment 3 (Zipf) |
| `derived/news_vs_sports.svmlight` | 1,001 headlines as binary bag-of-words vectors, label +1 news / −1 sports | Experiment 4 |
| `derived/news_2009_vs_2010.svmlight` | 1,000 headlines as binary bag-of-words vectors, label +1 2009 / −1 2010 | Experiment 4 |
| `results/*_2010.csv` | Tables exactly as reported in the 2010 paper | Comparison |
| `results/reproduction_2026.md` | Output of `reproduce/reproduce_2026.py` | Comparison |

The bag-of-words files contain feature indices only. Word order is discarded and **no index-to-word vocabulary is published**, so the original headlines cannot be reconstructed from them. The news-vs-sports file has 501 sports headlines; the 2010 paper reports 500.

## Not published, and why

The corpus was assembled in 2010 from:

- online archives of *Divaina*, *Silumina* and *Lankadeepa*, and defence.lk
- stories by Martin Wickramasinghe
- the UCSC Language Technology Research Laboratory (LTRL) Sinhala corpus
- English, Spanish and Japanese language-ID samples from the UW–Madison CS 769 course dataset

That text belongs to its publishers and authors, and the LTRL corpus and course dataset have their own terms. Redistributing them is not mine to do, so the following are **withheld**:

- the raw corpus and its integer-encoded form (the encoding plus its vocabulary reconstructs the text)
- the raw headline collections and their vocabularies
- the language-ID training and test documents

As a result, experiments 2 (Naive Bayes language ID) and 5 (n-gram language models) cannot be re-run from this repository; their 2010 results are preserved in `results/`. Researchers who need Sinhala text today should start from openly licensed sources, such as the LTRL resources (on their terms) or Wikipedia and Common Crawl derived corpora.
