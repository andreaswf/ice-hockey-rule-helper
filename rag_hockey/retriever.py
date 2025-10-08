
from pathlib import Path

from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore, create_kv_docstore
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config
from .ingest import build_index


def load_retriever(vs_path=config.VS_PATH, parent_path=config.PARENT_PATH):
    embeddings = OpenAIEmbeddings(model=config.EMBED_MODEL)
    vectorstore = FAISS.load_local(
        folder_path=vs_path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    
    # laod parrent stores
    parent_store = create_kv_docstore(LocalFileStore(parent_path))
    
    # recreate child splitter. Purely as ParentDocumentRetriever needs it as parameter
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHILD_CHUNK_SIZE,
        chunk_overlap=config.CHILD_CHUNK_OVERLAP
    )
    
    # retriever
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=parent_store,
        child_splitter=child_splitter,
        search_type="mmr",
        search_kwargs={
            "k": config.RETRIEVER_K,
            "fetch_k": config.RETRIEVER_FETCH_K,
            "lambda_mult": config.RETRIEVER_LAMBDA
        }
    )
    
    
    return retriever


def get_retriever(rebuild=False, vs_path=config.VS_PATH, parent_path=config.PARENT_PATH):
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