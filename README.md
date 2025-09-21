# Ice hockey rule assistant
## This is still under development

## The project
This project is for familarising myself with the LandChain architecture. The goal of the project is to create a working prototype of an ice hockey rule assistant LLM using RAG and the official IIHF rulebook. The makes use of:
- LLM prompt templates
- Text splitting
- OpenAI embeddings
- OpenAI API (there is also a local version running Llama)
- FAISS or Chroma for vector store
- Retriever for single or batches
- A RAG pipeline

Will futher add the use of agentic architecture for the:
- selecting the correct rulebook (if both IIHF and NHL would be available)
- verifying citations to the rulebook are correct 
- rewrite the prompt into multiple examples for better lookup
- if the input prompt is a game situation, rewrite it to a better prompt for the model


Currently the project is being tested in notebooks. The goal is to add an API using FastAPI for single and batch predictions.