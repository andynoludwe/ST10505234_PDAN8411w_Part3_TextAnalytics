# ST1# PDAN8411w — Part 3: Pipelines and Text Data
### Insurance Customer Reviews — Topic Modelling & Sentiment Classification

| Field | Detail |
|---|---|
| **Student** | Andisiwe (ST10505234) |
| **Module** | PDAN8411 – Programming for Data Analytics |
| **Assessment** | POE Part 3 – Pipelines and Text Data |
| **Language** | Python 3.13 |
| **Dataset** | `insurance_customer_reviews_gemini_enhanced.csv` (1,020 reviews) |

---

## Project Overview

This project applies two complementary NLP techniques to a corpus of 1,020 insurance customer reviews to answer a real client brief: *what are customers complaining about, and how do they feel overall?*

- **LDA topic modelling** (scikit-learn) identifies five broad areas of concern from the review text without any pre-defined labels
- **Multinomial Naive Bayes sentiment classification** (TextBlob + scikit-learn pipeline) classifies each review as Positive, Neutral, or Negative
- A **cross-analysis** combines both outputs to identify which specific concern areas carry the highest negative sentiment load

---

## Repository Structure

```
├── PDAN8411w_Part3_TextAnalytics.ipynb   # Main notebook — full analysis pipeline
├── insurance_customer_reviews_gemini_enhanced.csv  # Source dataset (place in same folder as notebook)
├── PDAN8411_Part3_TextAnalytics_Report.pdf         # Submitted PDF report
└── README.md
```

---

## Dataset

**File:** `insurance_customer_reviews_gemini_enhanced.csv`

| Property | Value |
|---|---|
| Rows | 1,020 |
| Columns | 8 (`ReviewID`, `CustomerID`, `ReviewDate`, `Rating`, `ReviewText`, `Sentiment`, `GeneratedTask`, `AIGenerated`) |
| Missing values | 0 |
| Duplicate rows | 0 |
| Average review length | ~156 words |
| Sentiment split | 53.7% Positive / 34.5% Negative / 11.8% Neutral |

The dataset is a locally prepared, AI-enhanced collection of insurance customer reviews designed to simulate realistic complaint language typical of platforms such as Hello Peter (South Africa) and Trustpilot. The `ReviewText` column is the primary input; `Sentiment` is used as a secondary validation label, not as training ground truth.

---

## Notebook Structure

| Section | Content |
|---|---|
| **1. Introduction** | Model choice justification, dataset quality assessment, analysis plan |
| **2. Imports & Data Loading** | Library imports, NLTK downloads, CSV load |
| **3. EDA** | Shape/dtypes, missing values, sentiment distribution, review length, word frequencies, word clouds |
| **4. Text Preprocessing** | Lowercase, punctuation removal, stopword removal (NLTK + domain noise words), lemmatisation, bigram chart, cleaned word clouds (corpus-level + by sentiment class) |
| **5. Vectorisation** | CountVectorizer (for LDA) and TF-IDF (for sentiment, leak-free — fit on train split only) |
| **6. Topic Modelling (LDA)** | Perplexity curve (3–11 topics), final 5-topic model, top words per topic, word distribution charts, pyLDAvis interactive panel, dominant topic assignment, gensim c_v coherence (5 vs 8 topics) |
| **7. Sentiment Analysis** | TextBlob polarity scoring, Multinomial NB baseline, GridSearchCV tuning, confusion matrix, per-class metrics, ROC curves, TextBlob vs original label validity check (Cohen's Kappa) |
| **8. Cross-Analysis** | Per-topic sentiment breakdown — identifies which concern areas carry the most negative sentiment |
| **9. Model Justification** | Printed model summaries, written justification of all algorithm choices |
| **10. Conclusion** | Key findings, limitations, suggested next steps |
| **11. References** | Harvard-style citations |

---

## Key Findings

### Topic Modelling
Five topics were identified, but the distribution was skewed rather than even:

| Topic | Label | Reviews | Neg % |
|---|---|---|---|
| 1 | Claims Communication & Delays | 75 (7.4%) | **20.0%** |
| 2 | Mixed / Hedged Service Commentary | 97 (9.5%) | 2.1% |
| 3 | Digital & App Experience | 195 (19.1%) | 1.5% |
| 4 | General Customer Service & Process Satisfaction | 591 (57.9%) | 6.8% |
| 5 | Claims & Damage Incidents | 62 (6.1%) | 25.8% |

Topic 4 absorbed 57.9% of reviews — acting as a broad catch-all — while the four smaller topics captured more specific, actionable concern areas. This is reported transparently in the notebook and report rather than reframed as a balanced outcome.

### Sentiment Classification

| Model | Accuracy | Weighted F1 | Negative Recall |
|---|---|---|---|
| Baseline MNB (alpha=1.0) | 0.843 | 0.848 | 0.40 |
| Tuned MNB (GridSearchCV) | 0.853 | 0.854 | 0.533 |

ROC-AUC (one-vs-rest): Negative 0.918 / Neutral 0.926 / Positive 0.968. The model's main weakness is Negative-class recall, traced to weak agreement (Cohen's Kappa = 0.326) between TextBlob's labels and the dataset's original `Sentiment` column — a labelling issue rather than a classifier failure.

---

## Setup & Running

### Requirements

- Python **3.13** (recommended — 3.14 lacks pre-built wheels for several dependencies)
- See install cell in the notebook (Cell 1) for the full package list

### Quickstart

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

Place `insurance_customer_reviews_gemini_enhanced.csv` in the same directory as the notebook, then open `PDAN8411w_Part3_TextAnalytics.ipynb` in Jupyter or Google Colab.

**Important — run in this order:**

1. Run **Cell 1** (installs wordcloud, pyLDAvis, funcy, gensim)
2. **Restart the kernel** (Kernel → Restart, or Runtime → Restart runtime in Colab)
3. Run all remaining cells from the top

> pyLDAvis requires a kernel restart after installation to import correctly. Cell 63 has a `try/except` fallback that prints a clear message if this step is skipped — it will not crash the rest of the notebook.

### Google Colab

Upload both the notebook and the CSV to Colab (or mount your Google Drive), then follow the same three steps above.

---

## Dependencies

| Package | Purpose |
|---|---|
| `pandas`, `numpy` | Data manipulation |
| `matplotlib`, `seaborn` | Visualisation |
| `nltk` | Tokenisation, stopwords, lemmatisation |
| `textblob` | Lexicon-based sentiment scoring |
| `wordcloud` | Word cloud generation |
| `scikit-learn` | CountVectorizer, TF-IDF, LDA, MNB, GridSearchCV, metrics |
| `gensim` | Topic coherence (c_v score) |
| `pyLDAvis` | Interactive LDA topic visualisation |

---

## Notes on Reproducibility

- `random_state=42` is set on all stochastic components (LDA, train/test split, GridSearchCV)
- The TF-IDF vectoriser is fitted **on the training split only** to prevent data leakage (fixed from an earlier version)
- The duplicate cleaning pipeline present in older notebook versions has been removed — `df['clean_text']` is produced by a single authoritative pipeline in Section 4

---

## References

Blei, D.M., Ng, A.Y. and Jordan, M.I. (2003) 'Latent Dirichlet Allocation', *Journal of Machine Learning Research*, 3, pp. 993–1022.

Loria, S. (2020) *TextBlob Documentation*. Available at: https://textblob.readthedocs.io

Pedregosa, F. et al. (2011) 'Scikit-learn: Machine Learning in Python', *Journal of Machine Learning Research*, 12, pp. 2825–2830.

Bird, S., Klein, E. and Loper, E. (2009) *Natural Language Processing with Python*. O'Reilly Media. Available at: https://www.nltk.org

Pang, B. and Lee, L. (2008) 'Opinion Mining and Sentiment Analysis', *Foundations and Trends in Information Retrieval*, 2(1–2), pp. 1–135.0505234_PDAN8411w_Part3_TextAnalytics
