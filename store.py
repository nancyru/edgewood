from langchain_core.documents import Document
from langchain_community.document_loaders import JSONLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.text_splitter import CharacterTextSplitter
from qdrant_client import models, QdrantClient
from qdrant_client.http.models import Distance, VectorParams


PATH_TO_JSON_DATA = "data/firecrawl_result.json"


def metadata_func(record: dict, metadata: dict) -> dict:
    """
    Define the metadata extraction function.
    """
    metadata["url"] = record.get("metadata").get("url")
    return metadata


def load_docs(path=PATH_TO_JSON_DATA):
    """
    Load json data into LangChain documents.
    """
    loader = JSONLoader(
        file_path=path,
        jq_schema=".data[]",
        content_key="markdown",
        metadata_func=metadata_func,
    )

    documents = loader.load()
    return documents


def chunk_docs(documents):
    """
    Chunk documents.
    """
    text_splitter = CharacterTextSplitter(chunk_size=7000, chunk_overlap=500)
    docs = text_splitter.split_documents(documents)
    return docs


def initialize_vector_store():
    """
    Initialize Qdrant vector store with embeddings.
    """

    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    dimension = len(
        embeddings.embed_query("I just want to know the embedding space dimension")
    )

    client = QdrantClient(":memory:")

    client.create_collection(
        collection_name="edgewood",
        vectors_config=models.VectorParams(
            size=dimension,
            distance=models.Distance.COSINE,
        ),
    )
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="edgewood",
        embedding=embeddings,
    )
    return vector_store


def create_vector_store(test=False):
    """
    Initialize vector store and add documents.
    """
    documents = load_docs()
    if test:
        documents = documents[:10]
    chunked_docs = chunk_docs(documents)
    vector_store = initialize_vector_store()
    vector_store.add_documents(documents=chunked_docs)
    print(f"{len(chunked_docs)} documents loaded")
    return vector_store
