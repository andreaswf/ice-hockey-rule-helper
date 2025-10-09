from langchain_community.document_loaders import PyPDFLoader

from . import config
from .parsing import MAIN_RE, SUB_RE, merge_docs, slice_on_regex
from .retriever_helpers import (
    make_child_splitter,
    make_empty_vectorstore,
    make_parent_retriever,
    make_parent_store,
)


def build_index(pdf_path=config.DOC_PATH, vs_path=config.VS_PATH, parent_path=config.PARENT_PATH):
    """Build the parent/child retrieval index from a PDF.
    
    Loads the PDF, crops the pages per config, merges them, slices into main and sub rules,
    then creates the parent store, child splitter, and an empty FAISS vector store and
    indexes the chunks. Persists the vector store to ``vs_path`` and the parent store to
    ``parent_path``.

    Args:
        pdf_path (str, optional): Path to rulebook PDF. Defaults to config.DOC_PATH.
        vs_path (str, optional): Path where the FAISS vector store is saved. Defaults to config.VS_PATH.
        parent_path (str, optional): Path where the parent store is saved/loaded. Defaults to config.PARENT_PATH.

    Returns:
        ParentDocumentRetriever: A fully configured retriever
    """
    #load documents, crop, and merge back together
    docs = PyPDFLoader(str(pdf_path)).load()
    docs_cropped = docs[config.CROP_START:config.CROP_END]
    merged = merge_docs(docs_cropped, source=config.SOURCE_LABEL)
    
    # do the splits on first main rules then sub rules
    split_on_main_rules = slice_on_regex(merged, MAIN_RE, prefix='main')
    split_on_sub_rules = slice_on_regex(split_on_main_rules, SUB_RE, prefix="sub")
    
    # create parentstore, childsplitter, empty vectorstore and retriever
    parent_store = make_parent_store(parent_path)
    child_splitter = make_child_splitter()
    vectorstore = make_empty_vectorstore()
    retriever = make_parent_retriever(vectorstore, parent_store, child_splitter)
    
    # add documents to retriever and save to vs
    retriever.add_documents(split_on_sub_rules)
    vectorstore.save_local(vs_path)
    
    return retriever
    
