# tests/test_import_sanity.py
def test_import_sanity():
    import rag_hockey
    import rag_hockey.parsing as p
    assert hasattr(p, "normalize_ocr")