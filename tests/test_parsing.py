from rag_hockey.parsing import normalize_ocr, merge_docs
from langchain.schema import Document




def test_normalize_ocr_fixes_common_splits():
    s = "PENAL TY and AL TERNATE"
    out = normalize_ocr(s)
    assert "PENALTY" in out
    assert "ALTERNATE" in out
    
    
def test_merge_docs():
    d1 = Document(page_content="Hello", metadata={'source': 'd1_source'})
    d2 = Document(page_content="World", metadata={'source': 'd2_source'})
    
    merged = merge_docs([d1, d2], source="merged_source")
    
    assert merged.page_content == "HelloWorld"
    assert merged.metadata == "merged_source"