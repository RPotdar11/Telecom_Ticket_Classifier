# PROJECT 1B (REAL DATA, 6 CATEGORIES, BALANCED, KEYWORD-GATED): 
# Auto-Categorize Network Trouble Tickets
# Dataset: Comcast Telecom Complaints (Kaggle)
# Run this in Google Colab: https://colab.research.google.com
# First cell: !pip install scikit-learn pandas --quiet
#
# BEFORE RUNNING: Upload "Comcast_telecom_complaints_data.csv" to your Colab session
# (click the folder icon on the left sidebar -> upload button)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# STEP 1: Load the real dataset
df = pd.read_csv("Comcast_telecom_complaints_data.csv")
print(f"Loaded {len(df)} real complaints\n")

# STEP 2: All category keywords now live in ONE shared dictionary.
# This is used twice in this project: once to label training data (Step 3),
# and again later to gate new predictions (Step 9) -- so both steps always
# agree on exactly which words belong to which category.
# IMPORTANT: order matters -- when labeling, we check dictionary entries top to
# bottom and stop at the first match, so put more specific categories first.
CATEGORY_KEYWORDS = {
    "billing": ["bill", "charge", "fee", "payment", "refund", "price",
                "pricing", "overcharg", "invoice", "cost", "money"],
    "outage": ["down", "outage", "no service", "no internet",
               "availability", "dead"],
    "connectivity_issues": ["intermittent", "unstable", "spotty", "keeps dropping",
                             "drops", "reconnect", "unable to connect",
                             "connection issue", "connection problem", "disconnect",
                             "wifi", "wi-fi", "wireless", "signal", "modem",
                             "unreliab", "connectivity", "dropped connection",
                             "losing signal", "loses signal", "poor connection"],
    "slow_speed": ["slow connection", " slow speed", "throttl", "buffer", "lag", "mbps", "bandwidth"],
    "plan_data_cap": ["cap", "plan", "package", "data usage", "limit",
                       "upgrade", "contract"],
    "customer_service": ["customer service", "poor service", "rude", "unhelpful",
                          "representative", "horrible", "terrible", "bad experience",
                          "lack of service", "no help", "poor support"],
}

def label_ticket(text):
    text = str(text).lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(w in text for w in keywords):
            return category
    return "other"

df["category"] = df["Customer Complaint"].apply(label_ticket)
print("Category distribution (including unmatched 'other'):")
print(df["category"].value_counts(), "\n")

# STEP 3: Keep only labeled rows for training
labeled_df = df[df["category"] != "other"].copy()
print(f"Training on {len(labeled_df)} labeled tickets out of {len(df)} total\n")

# STEP 4: TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
X = vectorizer.fit_transform(labeled_df["Customer Complaint"])
y = labeled_df["category"]

# STEP 5: Train/test split (stratify keeps category proportions balanced in both sets)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# STEP 6: Train
# By default, Naive Bayes assumes a category is more likely just because it saw
# MORE examples of it during training (billing had 664 vs connectivity_issues' 61).
# This biases predictions toward the biggest category. Setting class_prior to be
# EQUAL across all categories removes that bias and lets the model judge each
# ticket purely on its words, not on how common that category was in training.
n_classes = y.nunique()
model = MultinomialNB(class_prior=[1 / n_classes] * n_classes)
model.fit(X_train, y_train)

# STEP 7: Evaluate
predictions = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}\n")
print(classification_report(y_test, predictions, zero_division=0))

# STEP 8: Keyword gate -- checks if ANY category keyword is present at all
def contains_any_keyword(text):
    text = str(text).lower()
    for keywords in CATEGORY_KEYWORDS.values():
        if any(w in text for w in keywords):
            return True
    return False

# STEP 9: Final prediction function combining the keyword gate + the ML model
# Logic: if the ticket contains NONE of our known keywords, we compulsorily
# label it "other" -- we never let the ML model guess blindly on text it has
# no real signal about. Only if at least one keyword is present do we let the
# trained model decide WHICH of the 6 categories fits best.
def predict_ticket(text):
    if not contains_any_keyword(text):
        return "other", None
    X_new = vectorizer.transform([text])
    probs = model.predict_proba(X_new)[0]
    best_idx = probs.argmax()
    best_class = model.classes_[best_idx]
    confidence = probs[best_idx]
    return best_class, confidence

new_tickets = [
    "My connection has been dead all day",
    "Why was I billed extra this cycle",
    "Internet speed is terrible in the evenings",
    "I want to upgrade my current package",
    "The connection keeps dropping every few minutes",
    "The support representative was extremely rude to me",
    "Hello",
    "thanks"
]

print("\nPredictions with compulsory keyword gate:")
for text in new_tickets:
    category, confidence = predict_ticket(text)
    if confidence is None:
        print(f"  '{text}'  ->  {category}  (no keywords found -- skipped model)")
    else:
        print(f"  '{text}'  ->  {category}  (confidence: {confidence:.2f})")
