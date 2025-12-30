from langchain_text_splitters import RecursiveCharacterTextSplitter


def classify_clause(text):
    t = text.lower()
    if "terminate" in t:
        return "termination"
    if "confidential" in t:
        return "confidentiality"
    if "liability" in t or "liable" in t:
        return "liability"
    if "indemn" in t:
        return "indemnity"
    return "general"


def chunk_pages(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = []

    for page in pages:
        splits = splitter.split_text(page["text"])
        for s in splits:
            chunks.append({
                "text": s,
                "metadata": {
                    "page": page["page"],
                    "clause_type": classify_clause(s)
                }
            })

    return chunks


