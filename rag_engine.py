import os
import re
import pandas as pd
import streamlit as st

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import google.generativeai as genai


# ==========================
# GEMINI CONFIG
# ==========================

try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=API_KEY)

    print("===== AVAILABLE MODELS =====")
    for model in genai.list_models():
        if "generateContent" in model.supported_generation_methods:
            print(model.name)
    print("============================")

    MODEL_NAME = "gemini-flash-latest"
    gemini_model = genai.GenerativeModel(MODEL_NAME)

except Exception as e:
    gemini_model = None
    print("Gemini initialization error:", e)


# ==========================
# PATH
# ==========================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

CSV_PATH = os.path.join(
    BASE_DIR,
    "rag_data.csv"
)



# ==========================
# EMBEDDING MODEL
# ==========================

embedding_model = None


def get_embedding_model():

    global embedding_model

    if embedding_model is None:

        embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

    return embedding_model



# ==========================
# LOAD KNOWLEDGE BASE
# ==========================


chunks = None
embeddings = None


def load_rag_data():

    global chunks
    global embeddings


    if chunks is None:

        if os.path.exists(CSV_PATH):

            df = pd.read_csv(
                CSV_PATH
            )

            chunks = df["chunk_text"].tolist()

            embeddings = (
                get_embedding_model()
                .encode(chunks)
            )

        else:

            chunks=[]
            embeddings=None


    return chunks, embeddings




# ==========================
# DOCUMENT CHUNKING
# ==========================


def chunk_document(text):

    sentences = re.split(
        r'(?<=[.!?])\s+',
        text
    )

    result=[]

    temp=""

    for s in sentences:

        temp += " "+s

        if len(temp)>300:

            result.append(
                temp.strip()
            )

            temp=""


    if temp:

        result.append(
            temp.strip()
        )

    return result




# ==========================
# SIMPLE EXTRACTION
# ==========================


def extract_answer(text,question):

    q=question.lower()


    patterns={

    "company":
    r"(?:company|organization|from)[:\s]+([A-Za-z0-9 .&]+)",


    "email":
    r"[\w\.-]+@[\w\.-]+\.\w+",


    "website":
    r"(?:https?://|www\.)\S+",


    "certificate":
    r"(?:certificate id|id)[:\s]+([A-Za-z0-9-]+)",


    "salary":
    r"(?:₹|\$)\s?\d+[\d,]*",


    }


    for key,pattern in patterns.items():

        if key in q:

            match=re.search(
                pattern,
                text,
                re.I
            )

            if match:

                return match.group(0)


    return None




# ==========================
# SEMANTIC SEARCH
# ==========================


def search_context(question,doc_chunks):


    model=get_embedding_model()


    doc_vectors=model.encode(
        doc_chunks
    )


    q_vector=model.encode(
        [question]
    )


    scores=cosine_similarity(
        q_vector,
        doc_vectors
    )[0]


    index=scores.argmax()


    return doc_chunks[index]





# ==========================
# GEMINI ANSWER
# ==========================


def generate_answer(question,context):


    if gemini_model is None:

        return context



    prompt=f"""

You are a document verification assistant.

Answer only from the given document.

Document:
{context}


Question:
{question}


Give a short clear answer.
"""


    response=gemini_model.generate_content(
        prompt
    )


    return response.text




# ==========================
# MAIN FUNCTION
# ==========================


def rag_answer(question,document_text=None):


    if document_text:


        direct=extract_answer(
            document_text,
            question
        )


        if direct:

            return direct



        chunks=chunk_document(
            document_text
        )


        context=search_context(
            question,
            chunks
        )


        return generate_answer(
            question,
            context
        )



    base_chunks,base_embeddings=load_rag_data()


    if base_chunks:

        context=search_context(
            question,
            base_chunks
        )

        return generate_answer(
            question,
            context
        )


    return "No information found."
