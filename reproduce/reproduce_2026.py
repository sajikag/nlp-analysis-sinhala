"""Re-run the 2010 Sinhala NLP experiments from the published derived data.

Reproduces three of the five original experiments with modern tooling:
  1. Character maximum-likelihood estimates   (data/derived/sinhala_char_counts.tsv)
  2. Zipf's law, now with a fitted exponent   (data/derived/sinhala_word_frequencies.tsv)
  3. SVM topic classification, 10 random 90/10 splits, linear vs polynomial kernel,
     plus a majority-class baseline the 2010 report did not include
     (data/derived/*.svmlight)

Naive Bayes language ID and the n-gram language models need the raw corpus,
which is not redistributed (see data/README.md), so they are not re-run here.

Usage:  python reproduce/reproduce_2026.py
Writes: data/results/reproduction_2026.md and reproduce/figures/*.png
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.datasets import load_svmlight_file
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.svm import SVC, LinearSVC

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
RESULTS = ROOT / "data" / "results"
FIGURES = ROOT / "reproduce" / "figures"
SEED = 769  # the course number; any fixed seed works

# Accuracies reported in the 2010 report (mean over 10 folds, %)
REPORTED_2010 = {
    "news_vs_sports": {"linear": 90.1, "poly": 89.9},
    "news_2009_vs_2010": {"linear": 90.506, "poly": 89.698},
}


def read_tsv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def character_mle() -> list[str]:
    rows = read_tsv(DERIVED / "sinhala_char_counts.tsv")
    total = sum(int(r["count"]) for r in rows)
    lines = [
        "## 1. Character MLE (white space excluded)",
        "",
        f"Total basic characters: {total:,}",
        "",
        "| Rank | Char | Code point | Count | MLE |",
        "|---|---|---|---|---|",
    ]
    for i, r in enumerate(rows[:10], 1):
        lines.append(f"| {i} | {r['char']} | {r['codepoint']} | {int(r['count']):,} | {int(r['count']) / total:.4f} |")
    vowels = {chr(c) for c in range(0x0D85, 0x0D97)}
    first_vowel = next(i for i, r in enumerate(rows, 1) if r["char"] in vowels)
    lines += [
        "",
        f"No independent vowel appears in the top ten; the first is at rank {first_vowel}. "
        "Vowel sounds are mostly written as dependent signs attached to consonants, "
        "which this count (basic characters only) excludes.",
        "",
    ]
    return lines


def zipf() -> list[str]:
    rows = read_tsv(DERIVED / "sinhala_word_frequencies.tsv")
    f = np.array([int(r["frequency"]) for r in rows], dtype=float)
    r = np.array([int(r["rank"]) for r in rows], dtype=float)
    tokens, types = int(f.sum()), len(f)

    # Fit log f = a + b log r on ranks 1..1000, the usual range for Zipf estimates,
    # and over all ranks for comparison.
    def fit(mask):
        res = stats.linregress(np.log10(r[mask]), np.log10(f[mask]))
        return res.slope, res.rvalue**2, res.intercept

    head_slope, head_r2, head_icpt = fit(r <= 1000)
    all_slope, all_r2, _ = fit(r >= 1)
    hapax = int((f == 1).sum())

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=150)
    ax.loglog(r, f, ".", ms=2, alpha=0.5, label="Sinhala corpus")
    ref = f[0] * r ** -1.0
    ax.loglog(r, ref, "--", lw=1, label="ideal Zipf (slope −1)")
    head = r <= 1000
    fitted = 10 ** (head_icpt + head_slope * np.log10(r[head]))
    ax.loglog(r[head], fitted, "-", lw=1.5, label=f"least-squares fit, ranks ≤1000 (slope {head_slope:.2f})")
    ax.set_xlabel("rank")
    ax.set_ylabel("frequency")
    ax.set_title("Zipf's law: 681k-token Sinhala corpus")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "zipf.png")
    plt.close(fig)

    return [
        "## 2. Zipf's law",
        "",
        f"Tokens: {tokens:,} · Types: {types:,} · Hapax legomena: {hapax:,} ({hapax / types:.0%} of types)",
        "",
        "| Fit range | Slope | R² |",
        "|---|---|---|",
        f"| Ranks 1–1000 | {head_slope:.3f} | {head_r2:.3f} |",
        f"| All ranks | {all_slope:.3f} | {all_r2:.3f} |",
        "",
        "The 2010 report judged Zipf behaviour visually from the log-log plot. Fitting the exponent "
        "sharpens that conclusion: the distribution is close to linear in log-log space (high R²), so "
        "Sinhala is Zipfian in form, but the head is "
        + ("shallower" if head_slope > -1 else "steeper")
        + f" than the classic −1 (slope {head_slope:.2f}). A shallower head is commonly reported for "
        "morphologically rich languages: one lemma is spread across many inflected surface forms, so "
        "the most frequent word types are individually less dominant than in English. "
        f"Across all ranks the slope is {all_slope:.2f}, pulled "
        + ("down" if all_slope < head_slope else "up")
        + f" by the long tail of hapax legomena ({hapax / types:.0%} of types occur once).",
        "",
        "![Zipf plot](../../reproduce/figures/zipf.png)",
        "",
    ]


def svm_task(name: str) -> list[str]:
    X, y = load_svmlight_file(str(DERIVED / f"{name}.svmlight"))
    models = {
        "linear": lambda: LinearSVC(C=1.0, max_iter=10000),
        # SVMlight -t 1 -s 1 -r 1 -d 1: (s·<x,x'> + r)^d with s=1, r=1, d=1
        "poly": lambda: SVC(kernel="poly", degree=1, gamma=1.0, coef0=1.0, C=1.0),
        "majority": lambda: DummyClassifier(strategy="most_frequent"),
    }
    splits = ShuffleSplit(n_splits=10, test_size=0.1, random_state=SEED)
    acc = {k: [] for k in models}
    for train, test in splits.split(X):
        for k, make in models.items():
            m = make().fit(X[train], y[train])
            acc[k].append(100 * (m.predict(X[test]) == y[test]).mean())
    acc = {k: np.array(v) for k, v in acc.items()}
    t, p = stats.ttest_rel(acc["linear"], acc["poly"])
    rep = REPORTED_2010[name]
    title = {"news_vs_sports": "News vs sports headlines", "news_2009_vs_2010": "2009 vs 2010 news headlines"}[name]
    return [
        f"### {title} (n = {X.shape[0]}, vocabulary = {X.shape[1]:,})",
        "",
        "| Model | 2010 mean (%) | 2026 mean (%) | 2026 st. dev |",
        "|---|---|---|---|",
        f"| Linear SVM | {rep['linear']:.1f} | {acc['linear'].mean():.1f} | {acc['linear'].std(ddof=1):.2f} |",
        f"| Polynomial SVM (d=1) | {rep['poly']:.1f} | {acc['poly'].mean():.1f} | {acc['poly'].std(ddof=1):.2f} |",
        f"| Majority-class baseline | — | {acc['majority'].mean():.1f} | {acc['majority'].std(ddof=1):.2f} |",
        "",
        f"Linear vs polynomial, two-tailed paired t-test: t = {t:.3f}, p = {p:.3f}.",
        "",
    ]


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = [
        "# 2026 reproduction",
        "",
        "Generated by `reproduce/reproduce_2026.py` from the derived data in `data/derived/`. "
        "Library versions differ from 2010 (scikit-learn instead of SVMlight; random splits differ), "
        "so small differences from the original report are expected.",
        "",
    ]
    out += character_mle()
    out += zipf()
    out += ["## 3. SVM topic classification", ""]
    for name in ("news_vs_sports", "news_2009_vs_2010"):
        out += svm_task(name)
    out += [
        "The majority-class baseline, absent from the 2010 report, shows the classifiers are doing real work: "
        "both tasks are balanced, so chance is about 50 %.",
        "",
    ]
    (RESULTS / "reproduction_2026.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
