import faiss
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore, create_kv_docstore
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config


def make_embeddings():
    """Creates OpenAI embeddings based on config

    Returns:
        OpenAIEmbeddings: Embedding model configured via config.EMBED_MODEL.
    """
    return OpenAIEmbeddings(model=config.EMBED_MODEL)


def make_empty_vectorstore():
    """Creates an empty vectorstore using FAISS.
    
    Uses the embedding dimension specified in config for index dimension.

    Returns:
        FAISS: The FAISS vectorstore where embeddings and document vectors are stored.
    """
    embeddings = make_embeddings()
    vs_index = faiss.IndexFlatL2(config.EMBED_DIM)
    vectorstore = FAISS(
        embedding_function=embeddings,
        index=vs_index,
        docstore=InMemoryDocstore({}),
        index_to_docstore_id={},
        normalize_L2=True
    )
    return vectorstore


def make_child_splitter():
    """Creates a child_splitter to be used for ParentDocumentRetriever.
    
    chunk size and chunk overlap are specified in config

    Returns:
        RecursiveCharacterTextSplitter: Configured text splitter.
    """
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHILD_CHUNK_SIZE,
        chunk_overlap=config.CHILD_CHUNK_OVERLAP
    )
    return child_splitter


def make_parent_store(parent_path=config.PARENT_PATH):
    """Creates or loads the parent store locally.

    Args:
        parent_path (str, optional): Path for the parent store to be created or loaded. Defaults to config.PARENT_PATH.

    Returns:
        DocStore: A key-value store for the parent documents.
    """
    return create_kv_docstore(LocalFileStore(parent_path))


def make_parent_retriever(vectorstore, parent_store, child_splitter, search_type=config.SEARCH_TYPE):
    """Creates the parent retriever.
    
    The parent retriever is used for further chunking of the sub-rules,
    for better matching on the embeddings, and then saving a reference
    to the original sub rule so it can be returned.

    Args:
        vectorstore (FAISS): The FAISS vectorstore where embeddings and document vectors are stored.
        parent_store (DocStore): A key-value store for parent documents
        child_splitter (RecursiveCharacterTextSplitter): The textsplitter used for further chunking.
        search_type (str, optional): Which search type to be used. Defaults to config.SEARCH_TYPE.

    Returns:
        ParentDocumentRetriever: a retriever
    """
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=parent_store,
        child_splitter=child_splitter,
        search_type=search_type,
        search_kwargs={
            "k": config.RETRIEVER_K,
            "fetch_k": config.RETRIEVER_FETCH_K,
            "lambda_mult": config.RETRIEVER_LAMBDA
        }
    )
    return retriever
