from langchain_core.prompts import ChatPromptTemplate


def create_prompt_template(system_template: str) -> ChatPromptTemplate:
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("user", "{question}")
    ])
    return prompt_template


def get_main_prompt_template() -> ChatPromptTemplate:
    system_template = """You are an ice hockey rule assistant.

    Follow these rules:
    - Answer ONLY using the provided context below. If no context contains a rule identifier, respond "I don't know". Do not infer from gneeral hockey knowledge.
    - Use bulletpoints. 
    - After each bullet, append the Citation for the sentence(s) you used.
    - Citations example: [78.5 DISALLOWED GOALS, GOALS]


    Context (use only what is inside the markers):
    ---
    {context}
    ---"""
    return create_prompt_template(system_template)


def get_multiquery_prompt_template() -> ChatPromptTemplate:
    system_template = """You are an AI language model assistant. Your task is
    to generate 3 different versions of the given user
    question to retrieve relevant documents from a vector database.
    By generating multiple perspectives on the user question,
    your goal is to help the user overcome some of the limitations
    of distance-based similarity search. Provide these alternative
    questions separated by newlines. The questions are all about ice hockey.
    """
    return create_prompt_template(system_template)


