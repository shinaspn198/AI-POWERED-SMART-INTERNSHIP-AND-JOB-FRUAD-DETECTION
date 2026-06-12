import os
import re
import pickle
import pandas as pd

from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Datasets:", os.listdir("dataset"))

FRAUD_KEYWORDS = [
    "registration fee", "registration charges", "joining fee", "training fee",
    "deposit required", "security deposit", "processing fee", "verification fee",
    "activation fee", "software fee", "payment required", "pay now",
    "pay before joining", "transfer amount", "send payment",
    "pay to receive certificate", "certificate fee", "certificate after payment",
    "offer letter processing fee", "internship registration fee",

    "no interview", "direct selection", "direct joining", "immediate joining",
    "urgent confirmation", "limited time offer", "offer valid for 72 hours",
    "confirm within 24 hours", "limited seats", "apply immediately",

    "whatsapp only", "contact on whatsapp", "send resume on whatsapp",
    "call immediately",

    "guaranteed job", "guaranteed salary", "guaranteed income",
    "earn from home", "earn 50000", "earn 30000", "earn lakhs",
    "easy money", "work from home guaranteed",

    "mlm", "multi level", "network marketing",
    "data entry", "copy paste", "form filling",
    "free laptop", "free training", "referral bonus"
]

GENUINE_KEYWORDS = [
    "certificate of completion", "successfully completed", "completion date",
    "certificate id", "cin", "official email", "official website",
    "program coordinator", "co-founder", "signature",

    "internship offer letter", "department", "stipend", "mentor",
    "supervisor", "work schedule", "responsibilities",
    "confidentiality", "hr discussion", "document verification",
    "start date", "end date", "joining date", "official onboarding",
    "technical interview", "hr round", "selected after interview",
    "project work", "weekly reviews", "learning opportunity"
]

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s%₹$]", "", text)
    return re.sub(r"\s+", " ", text).strip()

def prepare_text_label_dataset(df, dataset_name):
    df.columns = [str(c).strip() for c in df.columns]

    if "text" in df.columns and "fraudulent" in df.columns:
        return df[["text", "fraudulent"]]

    if "text" in df.columns and "label" in df.columns:
        df = df.rename(columns={"label": "fraudulent"})
        return df[["text", "fraudulent"]]

    if "content" in df.columns and "fraudulent" in df.columns:
        df = df.rename(columns={"content": "text"})
        return df[["text", "fraudulent"]]

    if "content" in df.columns and "label" in df.columns:
        df = df.rename(columns={"content": "text", "label": "fraudulent"})
        return df[["text", "fraudulent"]]

    print(f"⚠️ {dataset_name} format not recognized. Columns found: {list(df.columns)}")
    return None

data_list = []

data1 = pd.read_csv("dataset/fake_job_postings.csv")
data1 = data1[["title", "company_profile", "description", "requirements", "fraudulent"]].fillna("")
data1["text"] = data1["title"] + " " + data1["company_profile"] + " " + data1["description"] + " " + data1["requirements"]
data1 = data1[["text", "fraudulent"]]
data_list.append(data1)

data2 = pd.read_csv("dataset/job_posts.csv")
data2["text"] = (
    data2["Title"].fillna("") + " " +
    data2["Company"].fillna("") + " " +
    data2["JobDescription"].fillna("") + " " +
    data2["JobRequirment"].fillna("")
)
data2["fraudulent"] = 0
data2 = data2[["text", "fraudulent"]]
data_list.append(data2)

data3 = pd.read_csv("dataset/internship_fraud_dataset.csv")
data3.columns = ["text", "fraudulent"]
data_list.append(data3)

data4 = pd.read_csv("dataset/fake_internship_keywords_dataset.csv")
data4.columns = ["text", "fraudulent"]
data_list.append(data4)

data5 = pd.read_excel("dataset/dataset_new.xlsx")
data5 = prepare_text_label_dataset(data5, "dataset_new.xlsx")
if data5 is not None:
    data_list.append(data5)

data6 = pd.read_excel("dataset/dataset_500.xlsx")
data6 = prepare_text_label_dataset(data6, "dataset_500.xlsx")
if data6 is not None:
    data_list.append(data6)

data7 = pd.read_excel("dataset/fraud_dataset_5000_final.xlsx")
data7 = prepare_text_label_dataset(data7, "fraud_dataset_5000_final.xlsx")
if data7 is not None:
    data_list.append(data7)


data = pd.concat(data_list, ignore_index=True)

data = data.dropna(subset=["text", "fraudulent"])
data = data.drop_duplicates(subset=["text"])
data["fraudulent"] = data["fraudulent"].astype(int)

print(f"\nTotal samples: {len(data)}")
print(f"Fraud: {data['fraudulent'].sum()} | Legit: {(data['fraudulent'] == 0).sum()}")

data["text_clean"] = data["text"].apply(clean_text)

def fraud_kw_count(text):
    return sum(1 for kw in FRAUD_KEYWORDS if kw in str(text).lower())

def genuine_kw_count(text):
    return sum(1 for kw in GENUINE_KEYWORDS if kw in str(text).lower())

def caps(text):
    return sum(1 for w in str(text).split() if w.isupper())

def ex(text):
    return str(text).count("!")

def length(text):
    return len(str(text).split())

def wa(text):
    return int("whatsapp" in str(text).lower())

def sal(text):
    return int(bool(re.search(r"(earn|salary|stipend)\s*[\$₹]?\s*\d{4,}", str(text).lower())))

def payment_pattern(text):
    return int(bool(re.search(r"(pay|payment|fee|deposit|charges|transfer)\s*(₹|\$|inr|usd)?\s*\d+", str(text).lower())))

def interview_pattern(text):
    return int(bool(re.search(r"(technical interview|hr round|interview scheduled|selected after interview)", str(text).lower())))

data["fraud_kw"] = data["text"].apply(fraud_kw_count)
data["genuine_kw"] = data["text"].apply(genuine_kw_count)
data["caps"] = data["text"].apply(caps)
data["ex"] = data["text"].apply(ex)
data["len"] = data["text"].apply(length)
data["wa"] = data["text"].apply(wa)
data["sal"] = data["text"].apply(sal)
data["payment"] = data["text"].apply(payment_pattern)
data["interview"] = data["text"].apply(interview_pattern)

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_features=20000
)

X_tfidf = vectorizer.fit_transform(data["text_clean"])

extra = csr_matrix(
    data[
        [
            "fraud_kw",
            "genuine_kw",
            "caps",
            "ex",
            "len",
            "wa",
            "sal",
            "payment",
            "interview"
        ]
    ].values
)

X = hstack([X_tfidf, extra])
y = data["fraudulent"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n--- Model Evaluation ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

os.makedirs("model", exist_ok=True)

pickle.dump(model, open("model/fraud_model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))
pickle.dump(FRAUD_KEYWORDS, open("model/fraud_keywords.pkl", "wb"))
pickle.dump(GENUINE_KEYWORDS, open("model/genuine_keywords.pkl", "wb"))

print("\n✅ Model saved successfully")