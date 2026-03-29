import chromadb
from chromadb.utils import embedding_functions

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "1c_knowledge_base"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

_client = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=emb_fn,
        )
    return _collection


def search(query: str, k: int = 3) -> list[str]:
    collection = _get_collection()
    count = collection.count()
    if count == 0:
        return []
    results = collection.query(
        query_texts=[query],
        n_results=min(k, count),
    )
    documents = results.get("documents", [[]])[0]
    return documents
