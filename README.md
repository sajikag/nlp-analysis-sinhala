# Analysis of Sinhala using NLP techniques (2010), with a 2026 reproduction

Sajika Gallege · M.S. Computer Sciences, University of Wisconsin–Madison · CS 769 (Advanced Natural Language Processing), Spring 2010

📄 **[Read the original report (PDF)](paper/Gallege_2010_NLP_Analysis_of_Sinhala.pdf)** · [2026 reproduction results](data/results/reproduction_2026.md) · [Data and what's withheld](data/README.md)

---

In 2010 there was very little machine-readable Sinhala, the native language of about 15 million people in Sri Lanka. Text online used inconsistent legacy fonts and encodings, and most NLP tools of the time accepted ASCII only. This project built a **681,233-token Unicode Sinhala corpus** (74,369 word types, 2.27 million characters) from newspaper archives, literature and the UCSC LTRL corpus, then applied five classical NLP techniques to characterise the language.

In 2026 I re-ran the experiments that the publishable data allows, with modern tooling and a baseline the original lacked.

## Results

| # | Experiment | 2010 result | 2026 reproduction |
|---|---|---|---|
| 1 | Character MLE | White space ~3× more frequent than the top letter ‘න’; first independent vowel only at rank 16 | ✅ Identical (same counts) |
| 2 | Naive Bayes language ID: Sinhala (romanised) vs English, Spanish, Japanese | 40/40 test documents correct | Not re-run: needs raw text |
| 3 | Zipf's law | Visually Zipfian on a log-log plot | ✅ Confirmed and quantified: near-linear (R² 0.99), but head slope **−0.75**, shallower than the classic −1 |
| 4a | SVM topic classification: news vs sports headlines | 90.1 % (linear), 89.9 % (poly) | ✅ 91.9 % / 91.9 % vs a 46 % majority baseline |
| 4b | SVM: 2009 vs 2010 news headlines | 90.5 % (linear), 89.7 % (poly) | ✅ 91.7 % / 91.6 % vs a 47 % majority baseline |
| 5 | n-gram language models (1–4) | Test perplexity 4,634 → 1,612; large train/test gap | Not re-run: needs raw text |

In both runs the kernel difference is not significant (paired t-test). Full tables: [`data/results/reproduction_2026.md`](data/results/reproduction_2026.md).

![Zipf plot](reproduce/figures/zipf.png)

## What the reproduction adds

- **A baseline.** The 2010 report gave SVM accuracies without saying what chance looked like. A majority-class baseline at ~47 % shows the ~91 % is real signal.
- **A fitted Zipf exponent.** "Roughly a straight line" becomes a measured slope, and the slope says something the eyeball didn't: the head of the Sinhala distribution is flatter than English-style Zipf. That is consistent with rich inflection spreading each lemma across many surface forms.
- **An honest gap.** 54 % of word types occur only once. Together with the 2010 train/test perplexity gap, that says the corpus was too small for higher-order language models, as the original report suspected.

## Looking back from 2026

The 2010 conclusions hold up, and the 2009-vs-2010 result is the most interesting: headlines from the last year of the civil war and from an election year are about 91 % separable by vocabulary alone. That is also a reminder that topic classifiers easily learn *when* text was written, not just what it is about, which matters for anyone training models on news.

Sinhala remains a low-resource language for modern language models. The lessons here still apply: get the encoding right first, measure the data before modelling it, and check every result against a baseline.

## Repository layout

```
paper/        original 2010 report (unchanged)
data/
  derived/    non-reconstructable data: character counts, word frequencies, bag-of-words vectors
  results/    2010 tables as reported, plus the 2026 reproduction output
  README.md   what is published, what is withheld, and why
reproduce/
  reproduce_2026.py   re-runs experiments 1, 3 and 4
  figures/            generated plots
```

## Run it

```bash
pip install -r reproduce/requirements.txt
python reproduce/reproduce_2026.py
```

## Licence

Code: MIT. Derived data, results and documentation: CC BY 4.0. The original report is © 2010 Sajika Gallege. See [LICENSE](LICENSE) and [`data/README.md`](data/README.md) for third-party text that is not included.
