import os
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# -------------------------------------------------------
# GROQ CLIENT — no torch, no sentence_transformers needed
# -------------------------------------------------------
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -------------------------------------------------------
# GROQ LLM ANSWER
# -------------------------------------------------------
def ask_groq(question, context):
    prompt = f"""You are an AI assistant helping students in India verify job and internship documents for fraud.

IMPORTANT: The document text below was extracted using OCR (optical character recognition) from a scanned PDF or image. It may contain OCR errors like garbled characters, wrong letters, or spacing issues. Please interpret it intelligently despite these errors.

Document content (may contain OCR errors):
---
{context[:3000]}
---

Question: {question}

Instructions:
- Answer based on the document content above
- If text looks garbled, try to interpret what it likely says
- Extract exact values (names, dates, amounts, IDs) even if slightly garbled
- For company name, look for letterhead, logo text, or "For [Company Name]" at the bottom
- If truly cannot find the answer say: "This information is not found in the document."
- Never say the document is fraudulent just because OCR text looks garbled
"""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"AI assistant error: {str(e)}"

# -------------------------------------------------------
# MAIN RAG ANSWER
# -------------------------------------------------------
def rag_answer(question, document_text=None):
    question = question.strip()
    if not question:
        return "Please ask a question."

    if document_text and len(document_text.strip()) > 30:
        return ask_groq(question, document_text)

    return "Please upload or paste a document first so I can answer your questions about it."