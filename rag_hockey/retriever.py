
from pathlib import Path

from langchain_community.vectorstores import FAISS

from . import config
from .helpers import make_child_splitter, make_embeddings, make_parent_retriever, make_parent_store
from .ingest import build_index


def load_retriever(vs_path=config.VS_PATH, parent_path=config.PARENT_PATH):
    """Load a retriever from saved vectorstore and parent store.

    Args:
        vs_path (str, optional): Path to the saved vectorstore. Defaults to ``config.VS_PATH``.
        parent_path (str, optional): Path to the saved parent store. Defaults to ``config.PARENT_PATH``.

    Returns:
        ParentDocumentRetriever: Loaded retriever.
    """
    embeddings = make_embeddings()
    vectorstore = FAISS.load_local(
        folder_path=vs_path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    
    parent_store = make_parent_store(parent_path)
    child_splitter = make_child_splitter()
    retriever = make_parent_retriever(vectorstore, parent_store, child_splitter)
    
    return retriever


def get_retriever(rebuild=False, vs_path=config.VS_PATH, parent_path=config.PARENT_PATH):
    """Get an existing retriever or rebuild it if missing.

    Args:
        rebuild (bool, optional): Force rebuild even if data exists. Defaults to False.
        vs_path (str, optional): Path to the vectorstore. Defaults to ``config.VS_PATH``.
        parent_path (str, optional): Path to the parent store. Defaults to ``config.PARENT_PATH``.

    Returns:
        ParentDocumentRetriever: Ready-to-use retriever.
    """
    vs_exists = Path(vs_path).exists()
    parent_exists = Path(parent_path).exists()
    
    # check if rebuild = True or if vector store and parent store both exists.
    # if not, build/rebuild index
    if rebuild or not (vs_exists and parent_exists):
        build_index(pdf_path=config.DOC_PATH, vs_path=vs_path, parent_path=parent_path)
    
    # loads the retriever
    retriever = load_retriever(
        vs_path=vs_path,
        parent_path=parent_path
    )
    return retriever