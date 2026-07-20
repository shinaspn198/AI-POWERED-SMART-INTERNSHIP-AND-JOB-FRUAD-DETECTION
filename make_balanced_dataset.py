import pandas as pd
import re
import os

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
FILES = {
    "dataset_500": "dataset_500.xlsx",
    "dataset_new": "dataset_new.xlsx",
    "fake_keywords": "fake_internship_keywords_dataset.csv",
    "fake_jobs": "fake_job_postings.csv",
    "fraud_5000": "fraud_dataset_5000_final(2).xlsx",
    "fraud_120": "fraud_internship_email_dataset_120.xlsx",
    "fraud_520": "fraud_internship_email_dataset_520.xlsx",
    "internship": "internship_fraud_dataset.csv",
    "job_posts": "job_posts.csv",
}

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\S+@\S+", " email ", text)
    text = re.sub(r"\d+", " number ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def normalize_label(x):
    x = str(x).lower().strip()
    if x in ["1", "1.0", "fraud", "fraudulent", "fake", "true", "yes"]:
        return 1
    return 0

frames = []

# 1 dataset_500
df = pd.read_excel(os.path.join(BASE_DIR, FILES["dataset_500"]))
frames.append(pd.DataFrame({
    "text": df["text"],
    "label": df["fraudulent"]
}))

# 2 dataset_new
df = pd.read_excel(os.path.join(BASE_DIR, FILES["dataset_new"]))
frames.append(pd.DataFrame({
    "text": df["text"],
    "label": df["label"]
}))

# 3 fake internship keywords
df = pd.read_csv(os.path.join(BASE_DIR, FILES["fake_keywords"]))
frames.append(pd.DataFrame({
    "text": df["text"],
    "label": 1
}))

# 4 fake_job_postings
df = pd.read_csv(os.path.join(BASE_DIR, FILES["fake_jobs"]))
text_cols = ["title", "company_profile", "description", "requirements", "benefits"]
text = df[text_cols].fillna("").agg(" ".join, axis=1)
frames.append(pd.DataFrame({
    "text": text,
    "label": df["fraudulent"]
}))

# 5 fraud 5000
df = pd.read_excel(os.path.join(BASE_DIR, FILES["fraud_5000"]))
frames.append(pd.DataFrame({
    "text": df["text"],
    "label": df["fraudulent"]
}))

# 6 fraud email 120
df = pd.read_excel(os.path.join(BASE_DIR, FILES["fraud_120"]))
frames.append(pd.DataFrame({
    "text": df["Email_Text"] if "Email_Text" in df.columns else df.iloc[:, 0],
    "label": 1
}))

# 7 fraud email 520
df = pd.read_excel(os.path.join(BASE_DIR, FILES["fraud_520"]))
frames.append(pd.DataFrame({
    "text": df["Email_Text"] if "Email_Text" in df.columns else df.iloc[:, 0],
    "label": 1
}))

# 8 internship fraud dataset
df = pd.read_csv(os.path.join(BASE_DIR, FILES["internship"]))
frames.append(pd.DataFrame({
    "text": df["job_text"],
    "label": df["label"]
}))

# 9 job_posts as genuine samples
df = pd.read_csv(os.path.join(BASE_DIR, FILES["job_posts"]))
job_cols = ["Title", "Company", "JobDescription", "JobRequirment", "RequiredQual"]
job_cols = [c for c in job_cols if c in df.columns]
text = df[job_cols].fillna("").agg(" ".join, axis=1)
frames.append(pd.DataFrame({
    "text": text,
    "label": 0
}))

# Combine
data = pd.concat(frames, ignore_index=True)

# Clean
data["text"] = data["text"].apply(clean_text)
data["label"] = data["label"].apply(normalize_label)

# Remove empty, short, duplicates
data = data.dropna()
data = data[data["text"].str.split().str.len() >= 10]
data = data.drop_duplicates(subset=["text"])

# Balance
fraud = data[data["label"] == 1]
legit = data[data["label"] == 0]

size = min(len(fraud), len(legit), 4000)

fraud_balanced = fraud.sample(size, random_state=42)
legit_balanced = legit.sample(size, random_state=42)

balanced = pd.concat([fraud_balanced, legit_balanced])
balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
balanced.to_csv(os.path.join(BASE_DIR, "balanced_fraud_dataset.csv"), index=False)
balanced.to_excel(os.path.join(BASE_DIR, "balanced_fraud_dataset.xlsx"), index=False)

print("✅ Balanced dataset created successfully!")
print("Fraud samples:", len(fraud_balanced))
print("Legit samples:", len(legit_balanced))
print("Total samples:", len(balanced))
print("Saved as:")
print("balanced_fraud_dataset.csv")
print("balanced_fraud_dataset.xlsx")