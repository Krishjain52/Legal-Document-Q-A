from google import genai
from chromadb import Client
from chromadb.config import Settings
import uuid

client = Client(
    Settings(
        persist_directory="chroma_db",
        anonymized_telemetry=False
    )
)

collection = client.get_or_create_collection("legal_docs")


def embed_text(text):
    result = genai.embed_content(
        model="models/embedding-001",
        content=text
    )
    return result["embedding"]


def store_chunks(chunks):
    for chunk in chunks:
        collection.add(
            ids=[str(uuid.uuid4())],
            embeddings=[embed_text(chunk["text"])],
            documents=[chunk["text"]],
            metadatas=[chunk["metadata"]]
        )

    client.persist()