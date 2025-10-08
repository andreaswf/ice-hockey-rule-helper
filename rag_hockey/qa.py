
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
    llm = init_chat_model(
        model=config.CHAT_MODEL, 
        model_provider=config.MODEL_PROVIDER, 
        temperature=config.TEMPERATURE, 
        timeout=config.TIMEOUT_S, 
        max_retries=config.MAX_RETRIES
    )
    return llm

def get_base_retriever():
    return get_retriever(rebuild=False)

def get_multi_query_retriever(retriever, llm, multi_prompt):
    return MultiQueryRetriever.from_llm(retriever=retriever, llm=llm, include_original=True, prompt=multi_prompt)

def format_docs(docs: list[Document]):
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


def retrieve_docs(question: str, llm, use_multiretriever=config.USE_MULTI_QUERY):
    retriever = get_base_retriever()
    if use_multiretriever:
        multi_prompt = get_multiquery_prompt_template()
        multi_retriever = get_multi_query_retriever(retriever, llm, multi_prompt)
        docs = multi_retriever.invoke(question)
    else:
        docs = retriever.invoke(question)
    return docs
    
    
def build_rag_chain(prompt_template: ChatPromptTemplate, llm):
    rag_chain = (
        prompt_template
        | llm
        | StrOutputParser()
    )
    return rag_chain


def answer_question(question: str, llm) -> dict:
    docs = retrieve_docs(question, llm)
    context = format_docs(docs)
    
    rag_chain = build_rag_chain(prompt_template=get_main_prompt_template(), llm=llm)
    answer = rag_chain.invoke({"question": question, "context": context})
    return {"answer": answer, "documents": docs}
    
    

@traceable
def rag_bot(question: str):
    llm = get_llm()
    return answer_question(question, llm)