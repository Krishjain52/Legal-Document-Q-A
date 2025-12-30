def detect_clause_types(question):
    q = question.lower()
    types = []

    if "terminate" in q:
        types.append("termination")
    if "one-sided" in q or "liability" in q:
        types.append("liability")
    if "confidential" in q:
        types.append("confidentiality")

    return types

from vectorstore import collection, embed_text

def retrieve_chunks(question):
    query_embedding = embed_text(question)
    clause_types = detect_clause_types(question)

    where_filter = (
        {"clause_type": {"$in": clause_types}}
        if clause_types else None
    )

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        where=where_filter
    )

    docs = []
    for doc, meta in zip(
        results["documents"][0],
        results["metadatas"][0]
    ):
        docs.append({
            "text": doc,
            "metadata": meta
        })

    return docs

from gemini_client import get_text_model

def generate_answer(question):
    docs = retrieve_chunks(question)

    if not docs:
        return {
            "answer": "Not found in document.",
            "confidence": 0.0,
            "sources": []
        }

    context = "\n\n".join(
        f"(Page {d['metadata']['page']}): {d['text']}"
        for d in docs
    )

    prompt = f"""
You are a legal document analysis assistant.
Answer ONLY using the content below.
Do NOT use external knowledge.
If the answer is missing, say "Not found in document."

Content:
{context}

Question: {question}
"""

    model = get_text_model()
    response = model.generate_content(prompt)

    confidence = min(1.0, 0.25 * len(docs))

    return {
        "answer": response.text,
        "confidence": round(confidence, 2),
        "sources": [
            {
                "page": d["metadata"]["page"],
                "clause_type": d["metadata"]["clause_type"]
            }
            for d in docs
        ]
    }
