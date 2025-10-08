import config
import faiss
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore, create_kv_docstore
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .parsing import MAIN_RE, SUB_RE, merge_docs, slice_on_regex


def build_index(pdf_path=config.DOC_PATH, vs_path=config.VS_PATH, parent_path=config.PARENT_PATH):
    
    #load documents, crop, and merge back together
    docs = PyPDFLoader(str(pdf_path)).load()
    docs_cropped = docs[config.CROP_START:config.CROP_END]
    merged = merge_docs(docs_cropped, source=config.SOURCE_LABEL)
    
    # do the splits on first main rules then sub rules
    split_on_main_rules = slice_on_regex(merged, MAIN_RE, prefix='main')
    split_on_sub_rules = slice_on_regex(split_on_main_rules, SUB_RE, prefix="sub")
    
    # use ParentStore and child splitter
    parent_store = create_kv_docstore(LocalFileStore(parent_path))
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHILD_CHUNK_SIZE,
        chunk_overlap=config.CHILD_CHUNK_OVERLAP
    )
    
    # embeddings and vectorstore
    embeddings = OpenAIEmbeddings(model=config.EMBED_MODEL)
    vs_index = faiss.IndexFlatL2(config.EMBED_DIM)
    vectorstore = FAISS(
        embedding_function=embeddings,
        index=vs_index,
        docstore=InMemoryDocstore({}),
        index_to_docstore_id={},
        normalize_L2=True
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
    
    retriever.add_documents(split_on_sub_rules)
    vectorstore.save_local(vs_path)
    
    return retriever
    
