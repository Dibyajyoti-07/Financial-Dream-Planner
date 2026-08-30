from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


class LocalEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return _get_model().encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text):
        return _get_model().encode([text], normalize_embeddings=True)[0].tolist()
