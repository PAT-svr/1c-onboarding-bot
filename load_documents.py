import glob
import os

import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader

KNOWLEDGE_BASE_PATH = "./knowledge_base"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "1c_knowledge_base"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def load_documents():
    documents = []

    pdf_files = glob.glob(os.path.join(KNOWLEDGE_BASE_PATH, "**/*.pdf"), recursive=True)
    docx_files = glob.glob(os.path.join(KNOWLEDGE_BASE_PATH, "**/*.docx"), recursive=True)

    for path in pdf_files:
        print(f"Загружаю PDF: {path}")
        loader = PyPDFLoader(path)
        documents.extend(loader.load())

    for path in docx_files:
        print(f"Загружаю DOCX: {path}")
        loader = Docx2txtLoader(path)
        documents.extend(loader.load())

    return documents


def main():
    print("Загрузка документов из knowledge_base/...")
    documents = load_documents()

    if not documents:
        print("Документы не найдены. Добавьте PDF или .docx файлы в папку knowledge_base/ и запустите снова.")
        return

    print(f"Загружено страниц: {len(documents)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    print(f"Разбито на фрагменты: {len(chunks)}")

    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Очищаем коллекцию перед повторной загрузкой
    try:
        client.delete_collection(COLLECTION_NAME)
        print("Старая коллекция удалена.")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=emb_fn,
    )

    texts = [chunk.page_content for chunk in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {"source": chunk.metadata.get("source", "unknown")}
        for chunk in chunks
    ]

    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        batch_meta = metadatas[i:i + batch_size]
        collection.add(documents=batch_texts, ids=batch_ids, metadatas=batch_meta)
        print(f"  Добавлено фрагментов {i} — {i + len(batch_texts) - 1}")

    print(f"\nГотово. Сохранено {len(texts)} фрагментов в ChromaDB ({CHROMA_PATH}/).")


if __name__ == "__main__":
    main()
