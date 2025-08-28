# tests/test_unit_cases.py

import pytest
from fastapi.testclient import TestClient
from api.main import app   # or your FastAPI entrypoint

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "Document Portal" in response.text
    
def test_health_ok():
    '''test_health_ok() - Tests the /health endpoint returns correct status'''
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["service"] == "document-portal"
    
def test_analyze_success(monkeypatch):
    """test_analyze_success() - Tests the /analyze endpoint with a mock PDF file"""
    # Mock dependencies inside api.main
    import api.main as main

    class DummyDocHandler:
        def save_pdf(self, file_adapter):
            return "dummy/path.pdf"

    def dummy_read_pdf_via_handler(handler, path: str) -> str:
        assert isinstance(handler, DummyDocHandler)
        assert path == "dummy/path.pdf"
        return "Sample PDF text"

    class DummyAnalyzer:
        def analyze_document(self, text: str):
            assert text == "Sample PDF text"
            return {"summary": "ok", "length": len(text)}

    monkeypatch.setattr(main, "DocHandler", lambda: DummyDocHandler())
    monkeypatch.setattr(main, "read_pdf_via_handler", dummy_read_pdf_via_handler)
    monkeypatch.setattr(main, "DocumentAnalyzer", lambda: DummyAnalyzer())

    files = {"file": ("test.pdf", b"%PDF-1.4 ...", "application/pdf")}
    resp = client.post("/analyze", files=files)
    assert resp.status_code == 200
    assert resp.json() == {"summary": "ok", "length": len("Sample PDF text")}

def test_analyze_failure(monkeypatch):
    """ test_analyze_failure() - Tests error handling in document analysis """
    import api.main as main

    class DummyDocHandler:
        def save_pdf(self, file_adapter):
            return "dummy/path.pdf"

    def dummy_read_pdf_via_handler(handler, path: str) -> str:
        return "text"

    class FailingAnalyzer:
        def analyze_document(self, text: str):
            raise ValueError("boom")

    monkeypatch.setattr(main, "DocHandler", lambda: DummyDocHandler())
    monkeypatch.setattr(main, "read_pdf_via_handler", dummy_read_pdf_via_handler)
    monkeypatch.setattr(main, "DocumentAnalyzer", lambda: FailingAnalyzer())

    files = {"file": ("test.pdf", b"%PDF-1.4 ...", "application/pdf")}
    resp = client.post("/analyze", files=files)
    assert resp.status_code == 500
    assert "Analysis failed" in resp.json()["detail"]
    
def test_compare_success(monkeypatch):
    """ test_compare_success() - Tests successful document comparison with mocked dependencies """
    import api.main as main

    class DummyComparator:
        def __init__(self):
            self.session_id = "abc123"
        def save_uploaded_files(self, reference, actual):
            # Ensure FastAPIFileAdapter-like interface was passed
            assert hasattr(reference, "name") and hasattr(actual, "name")
            return ("/tmp/ref.pdf", "/tmp/act.pdf")
        def combine_documents(self):
            return "REF...\nACT..."

    class DummyDF:
        def __init__(self, rows):
            self._rows = rows
        def to_dict(self, orient="records"):
            assert orient == "records"
            return self._rows

    class DummyComparatorLLM:
        def compare_documents(self, combined_text: str):
            assert "REF" in combined_text and "ACT" in combined_text
            return DummyDF([{ "page": 1, "change": "Added clause" }])

    monkeypatch.setattr(main, "DocumentComparator", lambda: DummyComparator())
    monkeypatch.setattr(main, "DocumentComparatorLLM", lambda: DummyComparatorLLM())

    files = {
        "reference": ("ref.pdf", b"ref", "application/pdf"),
        "actual": ("act.pdf", b"act", "application/pdf"),
    }
    resp = client.post("/compare", files=files)
    assert resp.status_code == 200
    body = resp.json()
    assert body["rows"] == [{"page": 1, "change": "Added clause"}]
    assert body["session_id"] == "abc123"
    
def test_compare_failure(monkeypatch):
    """ test_compare_failure() - Tests error handling in document comparison """
    import api.main as main

    class FailingComparator:
        def __init__(self):
            self.session_id = "x"
        def save_uploaded_files(self, reference, actual):
            return ("/tmp/ref.pdf", "/tmp/act.pdf")
        def combine_documents(self):
            raise RuntimeError("nope")

    monkeypatch.setattr(main, "DocumentComparator", lambda: FailingComparator())
    # LLM comparator won't be reached but mock anyway
    monkeypatch.setattr(main, "DocumentComparatorLLM", lambda: object())

    files = {
        "reference": ("ref.pdf", b"ref", "application/pdf"),
        "actual": ("act.pdf", b"act", "application/pdf"),
    }
    resp = client.post("/compare", files=files)
    assert resp.status_code == 500
    assert "Comparison failed" in resp.json()["detail"]
    
