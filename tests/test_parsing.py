from langchain.schema import Document

from rag_hockey.parsing import merge_docs, normalize_ocr


# test common splits
def test_normalize_ocr():
    s = "PENAL TY and AL TERNATE"
    out = normalize_ocr(s)
    assert "PENALTY" in out
    assert "ALTERNATE" in out
    

# test correct merging of pages (Documents)
def test_merge_docs():
    d1 = Document(page_content="Hello ", metadata={'source': 'd1_source'})
    d2 = Document(page_content="World", metadata={'source': 'd2_source'})
    
    merged = merge_docs([d1, d2], source="merged_source")
    
    assert merged[0].page_content == "Hello World"
    assert merged[0].metadata['source'] == "merged_source"
    
    
