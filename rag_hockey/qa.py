
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable

from . import config
from .prompts import get_main_prompt_template, get_multiquery_prompt_template
from .retriever import get_retriever

load_dotenv()


def get_llm():
    """Create and initialize the LLM using parameters from config.

    Returns:
        BaseChatModel: Configured chat model.
    """
    llm = init_chat_model(
        model=config.CHAT_MODEL, 
        model_provider=config.MODEL_PROVIDER, 
        temperature=config.TEMPERATURE, 
        timeout=config.TIMEOUT_S, 
        max_retries=config.MAX_RETRIES
    )
    return llm

def get_base_retriever():
    """Return the base retriever.

    Returns:
        ParentDocumentRetriever: Fully configured retriever.
    """
    return get_retriever(rebuild=False)

def get_multi_query_retriever(retriever, llm, multi_prompt) -> MultiQueryRetriever:
    """Create a multi-query retriever for generating reformulated questions.

    Args:
        retriever (ParentDocumentRetriever): The base retriever to wrap.
        llm (BaseChatModel): LLM used to generate query variations.
        multi_prompt (ChatPromptTemplate): Prompt template for multi-query generation.

    Returns:
        MultiQueryRetriever: Configured multi-query retriever.
    """
    return MultiQueryRetriever.from_llm(retriever=retriever, llm=llm, include_original=True, prompt=multi_prompt)

def format_docs(docs: list[Document]) -> str:
    """Format retrieved documents into readable text with citations.

    Args:
        docs (list[Document]): Retrieved documents.

    Returns:
        str: Formatted document text with metadata.
    """
    parts = []
    for doc in docs:
        meta = {
            "main_rule_id": doc.metadata.get("main_rule_id", "N/A"),
            "main_rule_name": doc.metadata.get("main_rule_name", "N/A"),
            "sub_rule_id": doc.metadata.get("sub_rule_id", "N/A"),
            "sub_rule_name": doc.metadata.get("sub_rule_name", "N/A"),
            "source": doc.metadata.get("source", "N/A")
        }
        citation = f"[{meta['sub_rule_id']} {meta['main_rule_name']}, {meta['sub_rule_name']}]"
        parts.append(f"Citation: {citation}\nContent:{doc.page_content.strip()}")
    return "\n\n".join(parts)


def retrieve_docs(question: str, llm, use_multiretriever=config.USE_MULTI_QUERY) -> list[Document]:
    """Retrieve relevant documents for a question.

    Args:
        question (str): User question.
        llm (BaseChatModel): LLM used for optional multi-query retrieval.
        use_multiretriever (bool, optional): Whether to use multi-query retrieval. Defaults to config.USE_MULTI_QUERY.

    Returns:
        list[Document]: Retrieved documents.
    """
    retriever = get_base_retriever()
    if use_multiretriever:
        multi_prompt = get_multiquery_prompt_template()
        multi_retriever = get_multi_query_retriever(retriever, llm, multi_prompt)
        docs = multi_retriever.invoke(question)
    else:
        docs = retriever.invoke(question)
    return docs
    
    
def build_rag_chain(prompt_template: ChatPromptTemplate, llm):
    """Build a simple RAG chain from prompt, LLM, and parser.

    Args:
        prompt_template (ChatPromptTemplate): Prompt template for the chain.
        llm (BaseChatModel): LLM to use in the chain.

    Returns:
        Runnable: Composed RAG chain.
    """
    rag_chain = (
        prompt_template
        | llm
        | StrOutputParser()
    )
    return rag_chain


def answer_question(question: str, llm) -> dict:
    """Answer a question using RAG with document retrieval.

    Args:
        question (str): The user question.
        llm (BaseChatModel): LLM used for generation.

    Returns:
        dict: Contains the final answer and retrieved documents.
    """
    docs = retrieve_docs(question, llm)
    context = format_docs(docs)
    
    rag_chain = build_rag_chain(prompt_template=get_main_prompt_template(), llm=llm)
    answer = rag_chain.invoke({"question": question, "context": context})
    return {"answer": answer, "documents": docs}
    
    

@traceable
def rag_bot(question: str) -> dict:
    """Main entry point for the RAG pipeline.

    Args:
        question (str): User question.

    Returns:
        dict: RAG-generated answer and supporting documents.
    """
    llm = get_llm()
    return answer_question(question, llm)