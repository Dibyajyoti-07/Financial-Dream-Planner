from pathlib import Path

from rag.embeddings import LocalEmbeddings  # import before chromadb - avoids a Windows torch/onnxruntime DLL clash

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

KNOWLEDGE_BASE_DIR = Path(__file__).parent / "knowledge_base"
PERSIST_DIR = str(Path(__file__).parent / "chroma_db")
COLLECTION_NAME = "financial_planner_kb"


def load_documents():
    docs = []
    for path in sorted(KNOWLEDGE_BASE_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        docs.append(Document(page_content=text, metadata={"source": path.name}))
    return docs


def build_index():
    docs = load_documents()
    if not docs:
        raise FileNotFoundError(f"No .txt files found in {KNOWLEDGE_BASE_DIR}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    Chroma.from_documents(
        documents=chunks,
        embedding=LocalEmbeddings(),
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
    )
    print(f"Indexed {len(chunks)} chunks from {len(docs)} files into {PERSIST_DIR}")


if __name__ == "__main__":
    build_index()
