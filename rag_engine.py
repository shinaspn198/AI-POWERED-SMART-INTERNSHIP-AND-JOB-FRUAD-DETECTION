import os
import re
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "rag_data.csv")

# -------------------------------------------------------
# LOAD MODEL ONCE — cached at module level
# -------------------------------------------------------
_model           = None
_base_chunks     = None
_base_embeddings = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def get_base_knowledge():
    global _base_chunks, _base_embeddings
    if _base_chunks is None:
        if os.path.exists(CSV_PATH):
            df           = pd.read_csv(CSV_PATH)
            _base_chunks = df["chunk_text"].tolist()
            _base_embeddings = get_model().encode(_base_chunks)
        else:
            _base_chunks     = []
            _base_embeddings = None
    return _base_chunks, _base_embeddings

# -------------------------------------------------------
# CHUNK DOCUMENT INTO OVERLAPPING SENTENCE GROUPS
# -------------------------------------------------------
def chunk_document(text, chunk_size=3):
    sentences = re.split(r'(?<=[.!?\n])\s+', text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
    if not sentences:
        return [text[:1000]]
    chunks = []
    step = max(1, chunk_size - 1)
    for i in range(0, len(sentences), step):
        chunk = " ".join(sentences[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
    return chunks

# -------------------------------------------------------
# REGEX FIELD EXTRACTOR — works on any document
# -------------------------------------------------------
def extract_field(text, question):
    q = question.lower()

    patterns = {
        ("company", "organization", "employer", "firm", "issued by", "from"): [
            r"(?:company|organization|employer|firm|from|by)[:\s]+([A-Za-z0-9 &.,'-]{3,60})",
            r"([A-Z][A-Za-z0-9 &.]{2,40}(?:Pvt|Ltd|Inc|Corp|Technologies|Solutions|Services|Interns)[A-Za-z. ]*)"
        ],
        ("name", "candidate", "awarded to", "issued to", "intern", "student"): [
            r"(?:awarded to|issued to|this is to certify that|candidate|intern|student)[:\s]+([A-Z][A-Za-z ]{2,40})",
            r"(?:Mr\.|Ms\.|Mrs\.)\s+([A-Z][A-Za-z ]{2,40})"
        ],
        ("duration", "how long", "period", "tenure"): [
            r"(\d+\s*(?:month|week|day)s?(?:\s+and\s+\d+\s*(?:month|week|day)s?)?)",
            r"(?:duration|period|tenure)[:\s]+([^\n.]{3,50})"
        ],
        ("date", "issued on", "when", "issued"): [
            r"(?:date|issued on|dated)[:\s]+(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})",
            r"(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})"
        ],
        ("email", "mail", "contact email"): [
            r"[\w\.\-]+@[\w\.\-]+\.\w{2,}"
        ],
        ("website", "url", "web", "link"): [
            r"(?:www\.[\w\.\-]+\.\w{2,}|https?://[\w\.\-/]+)"
        ],
        ("salary", "stipend", "pay", "compensation", "ctc"): [
            r"(?:salary|stipend|compensation|ctc|pay)[:\s]*(?:₹|\$|INR|USD|Rs\.?)?\s*[\d,]+(?:\s*(?:per|/)?\s*(?:month|year|annum))?",
            r"(?:₹|\$|Rs\.?)\s*[\d,]+(?:\s*(?:per|/)?\s*(?:month|year))?"
        ],
        ("certificate id", "cert id", "cin", "reference", "ref no", "id"): [
            r"(?:certificate\s*id|cert\s*id|cin|ref(?:erence)?\s*(?:no|number|#)?)[:\s#]*([A-Z0-9\/\-]{4,30})"
        ],
        ("signed", "signature", "authorized", "signatory"): [
            r"(?:signed\s*by|authorized\s*by|signature\s*of|signatory)[:\s]+([A-Za-z ,.\-]{3,60})",
            r"([A-Z][A-Za-z ]{2,30})\s*\n\s*(?:Co-?[Ff]ounder|[Dd]irector|[Mm]anager|[Cc]oordinator)"
        ],
        ("program", "domain", "role", "position", "internship in", "worked as"): [
            r"(?:program|domain|role|position|internship in|as a|as an)[:\s]+([A-Za-z0-9 &,]+)",
            r"(?:Data Science|Machine Learning|Web Development|Software|Marketing|Finance|HR|Design)[A-Za-z &]*"
        ]
    }

    for keywords, pats in patterns.items():
        if any(kw in q for kw in keywords):
            for pat in pats:
                match = re.search(pat, text, re.IGNORECASE)
                if match:
                    result = match.group(1).strip() if match.lastindex else match.group(0).strip()
                    if result:
                        return result
    return None

# -------------------------------------------------------
# SEMANTIC SEARCH
# -------------------------------------------------------
def semantic_search(question, chunks, embeddings, top_k=3):
    m      = get_model()
    q_emb  = m.encode([question])
    scores = cosine_similarity(q_emb, embeddings)[0]
    top_idx = scores.argsort()[-top_k:][::-1]
    return [(chunks[i], scores[i]) for i in top_idx if scores[i] > 0.15]

# -------------------------------------------------------
# MAIN RAG ANSWER
# -------------------------------------------------------
def rag_answer(question, document_text=None):
    question = question.strip()
    if not question:
        return "Please ask a question."

    # 1 — Try regex extraction from document
    if document_text and len(document_text.strip()) > 30:
        direct = extract_field(document_text, question)
        if direct:
            return direct

        # 2 — Semantic search on the actual document
        doc_chunks = chunk_document(document_text)
        if doc_chunks:
            m          = get_model()
            doc_embs   = m.encode(doc_chunks)
            results    = semantic_search(question, doc_chunks, doc_embs, top_k=2)
            if results:
                answer = " ... ".join([r[0] for r in results])
                return answer[:800]

    # 3 — Fall back to base CSV knowledge
    base_chunks, base_embs = get_base_knowledge()
    if base_chunks and base_embs is not None:
        results = semantic_search(question, base_chunks, base_embs, top_k=1)
        if results:
            return results[0][0]

    return "I couldn't find a specific answer in this document. Try asking about the company name, duration, certificate ID, salary, or who signed it."