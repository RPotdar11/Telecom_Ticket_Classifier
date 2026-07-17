# Telecom Ticket Auto-Categorization (NLP + Machine Learning)

An AI model that automatically categorizes real telecom customer complaints into actionable categories — billing, outage, connectivity issues, slow speed, plan/data cap, and customer service — using NLP and a Naive Bayes classifier.

This mirrors real-world "ticket triage" systems used by telecom companies to route customer complaints to the right team faster, reducing response time for urgent issues like outages.

## Dataset

Real-world dataset: **Comcast Telecom Complaints** (2,224 customer complaints), sourced from Kaggle.
Dataset link: https://www.kaggle.com/datasets/abdulmuqtadir1908/comcast-telecom-consumer-complaints

The raw dataset has no category labels — only free-text complaints. This project generates labels itself using a keyword-based weak supervision approach (see below).

## How it works

1. **Weak supervision labeling** — since the raw data has no categories, a shared keyword dictionary (`CATEGORY_KEYWORDS`) scans each complaint and assigns it to one of 6 categories, or "other" if no keywords match.
2. **TF-IDF vectorization** — converts complaint text into numerical features, weighing words by how distinctive they are to a given complaint.
3. **Naive Bayes classification** — trained on the labeled subset (1,361 of 2,224 tickets), with a balanced class prior to prevent the model from being biased toward the largest category (billing).
4. **Compulsory keyword gate** — at prediction time, if a new ticket contains none of the known category keywords (e.g. "Hello"), it's automatically labeled "other" without the ML model being consulted at all — preventing the classic classifier problem of always forcing a guess.

## Results

- **85% accuracy** on a held-out 20% test split
- Per-category recall ranges from 47% (outage) to 98% (slow_speed) — smaller categories perform worse due to fewer training examples (class imbalance), a known and explainable limitation discussed in the code comments
- Verified the model correctly rejects irrelevant input (e.g., "Hello," "thanks") as "other" instead of forcing a false category

## Tools & Libraries

- Python
- pandas — data loading and manipulation
- scikit-learn — TF-IDF vectorization, Naive Bayes classifier, evaluation metrics
- Google Colab — development environment

## How to run this project

1. Clone this repository
2. Download the dataset from the Kaggle link above
3. Create a folder named `data` inside the project directory, and place the CSV inside it, so the path looks like:
   ```
   telecom-ticket-classification/
   └── data/
       └── Comcast_telecom_complaints_data.csv
   ```
4. Install dependencies:
   ```
   pip install pandas scikit-learn
   ```
5. Run the script:
   ```
   python ticket_classifier.py
   ```

The script automatically checks the `data/` folder (or the same folder as the script) for the dataset, and will show a clear message telling you exactly what to do if it can't find the file.

## Project structure

```
├── ticket_classifier.py     # Main script: labeling, training, evaluation, prediction
├── data/                     # Place the downloaded Kaggle CSV here (not included in repo)
├── README.md                 # This file
└── requirements.txt          # Python dependencies
```

## Known limitations & future improvements

- Keyword-based labeling only confidently labels 60% of the dataset; the remaining 40% ("other") are too vague for rule-based labeling and were excluded from training
- Class imbalance affects smaller categories (outage, connectivity_issues) — more real labeled examples would likely improve their accuracy
- Future improvement: replace keyword-based labeling with a small manually-labeled sample plus a more advanced transformer-based model (e.g. BERT) for higher coverage and accuracy

## Author

Rutuja Potdar — MS Telecommunications, RIT