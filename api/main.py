import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Query, Request
from pydantic import BaseModel

from rag_hockey.qa import answer_question, get_llm
from rag_hockey.retriever import get_retriever


class QuestionInput(BaseModel):
    question: str
    
    
class DocumentOut(BaseModel):
    page_content: str
    metadata: dict[str, Any] = {}
    
class QAResponse(BaseModel):
    answer: str
    documents: list[DocumentOut] | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("DO THIS AT LOAD")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY")
    
    get_retriever()
    llm = get_llm()
    
    # store llm in app.state for later use
    app.state.llm = llm
    
    yield
    print("DO THIS AT SHUTDOWN")



app = FastAPI(lifespan=lifespan)

@app.get("/")
def root():
    return {"status": "ok", "message": "Ice Hockey Rule Assistant API is running"}


@app.post("/predict", response_model=QAResponse)
def qa(
    request: Request, 
    question: QuestionInput, 
    include_docs: bool = Query(False, description="Include retrieved Documents")
):
    llm = request.app.state.llm
    result = answer_question(
        question=question.question, 
        llm=llm, 
        include_docs=include_docs
    )
    
    if include_docs:
        raw_docs = result.get('docs', [])
        docs_out = [DocumentOut(page_content=d.page_content, metadata=d.metadata or {}) for d in raw_docs]
    else:
        docs_out = None
    
    return QAResponse(answer=result["answer"], documents=docs_out)
    