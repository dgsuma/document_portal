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