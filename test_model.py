import pickle
import re
from scipy.sparse import hstack, csr_matrix

model = pickle.load(open("model/fraud_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))
FRAUD_KEYWORDS = pickle.load(open("model/fraud_keywords.pkl", "rb"))

try:
    GENUINE_KEYWORDS = pickle.load(open("model/genuine_keywords.pkl", "rb"))
except:
    GENUINE_KEYWORDS = []

def transform(text):
    clean = text.lower()
    tfidf = vectorizer.transform([clean])
    extra = [[
        sum(1 for kw in FRAUD_KEYWORDS if kw in clean),
        sum(1 for kw in GENUINE_KEYWORDS if kw in clean),
        sum(1 for w in text.split() if w.isupper()),
        text.count("!"),
        len(text.split()),
        int("whatsapp" in clean),
        int(bool(re.search(r"(earn|salary|stipend)\s*[\$Rs.]+\s*\d{4,}", clean))),
        int(bool(re.search(r"(pay|payment|fee|deposit|charges|transfer)\s*(rs|usd|inr)?\s*\d+", clean))),
        int(bool(re.search(r"(technical interview|hr round|interview scheduled)", clean)))
    ]]
    return hstack([tfidf, csr_matrix(extra)])

texts = [
    "submit registration charges 155 USD along with this document offer last for 72 hours factory worker no interview direct selection",
    "Urgent hiring! Earn 50000 per month from home. No experience needed. Registration fee 500 required. Send resume on WhatsApp immediately!",
    "We are hiring a Python Developer. 2 years experience. Salary 6LPA. Technical interview and HR round. Send CV to hr@techcorp.in"
]

for text in texts:
    X = transform(text)
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    print("Text:", text[:60])
    print("Prediction:", "FRAUD" if pred==1 else "LEGIT")
    print("Fraud:", round(proba[1]*100, 1), "%")
    print()
