from langchain.schema import Document

from rag_hockey.qa import format_docs


def test_format_docs():
    docs = [
        Document(
            page_content="Test content", 
            metadata={
                "main_rule_name": "SLASHING",
                "sub_rule_id": "61.2",
                "sub_rule_name": "MINOR PENALTY"
            }
        )
    ]
    output = format_docs(docs)
    assert "Citation: [61.2 SLASHING, MINOR PENALTY]" in output
    assert "Test content" in output
    