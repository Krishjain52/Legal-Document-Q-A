import streamlit as st
from ingest import extract_text
from langchain_text_splitters import RecursiveCharacterTextSplitter
from gemini_client import generate_answer

# Title
st.title("Legal Document Q&A (RAG + Gemini)")

# Upload PDF
uploaded_file = st.file_uploader("Upload NDA / Contract PDF", type=["pdf"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF uploaded successfully!")

    # Extract text
    pages = extract_text("temp.pdf")
    st.write(f"PDF has {len(pages)} pages.")

    # Chunk text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = []
    for page in pages:
        for chunk in splitter.split_text(page["text"]):
            chunks.append({
                "text": chunk,
                "page": page["page"]
            })

    st.write(f"Total chunks created: {len(chunks)}")

    # Ask question
    question = st.text_input("Ask a question about the document:")

    if question:
        # Combine all chunks for simple prompt
        context = "\n\n".join([f"Page {c['page']}: {c['text']}" for c in chunks])

        prompt = f"""
You are a legal assistant. Answer ONLY using the context below.

Context:
{context}

Question:
{question}
"""
        answer = generate_answer(prompt)
        st.subheader("Answer:")
        st.write(answer)



