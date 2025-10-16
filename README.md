# Ice hockey rule assistant
## This is still under development

## The project
This project is for familarising myself with the LandChain architecture. The goal of the project is to create a working prototype of an ice hockey rule assistant using RAG and the official IIHF rulebook. The makes use of:
- LLM via OpenAI
- API via fastAPI
- Text splitting for rules and subrules, and further splitting subrules using ParentDocumentRetriever
- FAISS for vectorstore
- Both singlequery and multiquery retriever
- Citations to the rule used
- Evaluation using a testset of rules and ground truth.

Further looking to add:
- Something to verify citations to the rulebook are correct
- If the question is a game situation, then rewrite into a better question/prompt.
