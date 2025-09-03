# # tests/test_unit_cases.py

# import io
# import os
# import sys
# import json
# import tempfile
# import shutil
# from pathlib import Path
# from unittest.mock import Mock, patch, MagicMock
# import pytest
# import pandas as pd
# from fastapi.testclient import TestClient
# from langchain.schema import Document
# from api.main import app

# client = TestClient(app)

# def test_home():
#     response = client.get("/")
#     assert response.status_code == 200
#     assert "Document Portal" in response.text


# def test_health_ok():
#     '''test_health_ok() - Tests the /health endpoint returns correct status'''
#     resp = client.get("/health")
#     assert resp.status_code == 200
#     data = resp.json()
#     assert data["status"] == "ok"
#     assert data["service"] == "document-portal"


# def test_analyze_success(monkeypatch):
#     """test_analyze_success() - Tests the /analyze endpoint with a mock PDF file"""
#     # Mock dependencies inside api.main
#     import api.main as main

#     class DummyDocHandler:
#         def save_pdf(self, file_adapter):
#             return "dummy/path.pdf"

#     def dummy_read_pdf_via_handler(handler, path: str) -> str:
#         assert isinstance(handler, DummyDocHandler)
#         assert path == "dummy/path.pdf"
#         return "Sample PDF text"

#     class DummyAnalyzer:
#         def analyze_document(self, text: str):
#             assert text == "Sample PDF text"
#             return {"summary": "ok", "length": len(text)}

#     monkeypatch.setattr(main, "DocHandler", lambda: DummyDocHandler())
#     monkeypatch.setattr(main, "read_pdf_via_handler", dummy_read_pdf_via_handler)
#     monkeypatch.setattr(main, "DocumentAnalyzer", lambda: DummyAnalyzer())

#     files = {"file": ("test.pdf", b"%PDF-1.4 ...", "application/pdf")}
#     resp = client.post("/analyze", files=files)
#     assert resp.status_code == 200
#     assert resp.json() == {"summary": "ok", "length": len("Sample PDF text")}


# def test_analyze_failure(monkeypatch):
#     """ test_analyze_failure() - Tests error handling in document analysis """
#     import api.main as main

#     class DummyDocHandler:
#         def save_pdf(self, file_adapter):
#             return "dummy/path.pdf"

#     def dummy_read_pdf_via_handler(handler, path: str) -> str:
#         return "text"

#     class FailingAnalyzer:
#         def analyze_document(self, text: str):
#             raise ValueError("boom")

#     monkeypatch.setattr(main, "DocHandler", lambda: DummyDocHandler())
#     monkeypatch.setattr(main, "read_pdf_via_handler", dummy_read_pdf_via_handler)
#     monkeypatch.setattr(main, "DocumentAnalyzer", lambda: FailingAnalyzer())

#     files = {"file": ("test.pdf", b"%PDF-1.4 ...", "application/pdf")}
#     resp = client.post("/analyze", files=files)
#     assert resp.status_code == 500
#     assert "Analysis failed" in resp.json()["detail"]


# def test_compare_success(monkeypatch):
#     """ test_compare_success() - Tests successful document comparison with mocked dependencies """
#     import api.main as main

#     class DummyComparator:
#         def __init__(self):
#             self.session_id = "abc123"
#         def save_uploaded_files(self, reference, actual):
#             # Ensure FastAPIFileAdapter-like interface was passed
#             assert hasattr(reference, "name") and hasattr(actual, "name")
#             return ("/tmp/ref.pdf", "/tmp/act.pdf")
#         def combine_documents(self):
#             return "REF...\nACT..."

#     class DummyDF:
#         def __init__(self, rows):
#             self._rows = rows
#         def to_dict(self, orient="records"):
#             assert orient == "records"
#             return self._rows

#     class DummyComparatorLLM:
#         def compare_documents(self, combined_text: str):
#             assert "REF" in combined_text and "ACT" in combined_text
#             return DummyDF([{ "page": 1, "change": "Added clause" }])

#     monkeypatch.setattr(main, "DocumentComparator", lambda: DummyComparator())
#     monkeypatch.setattr(main, "DocumentComparatorLLM", lambda: DummyComparatorLLM())

#     files = {
#         "reference": ("ref.pdf", b"ref", "application/pdf"),
#         "actual": ("act.pdf", b"act", "application/pdf"),
#     }
#     resp = client.post("/compare", files=files)
#     assert resp.status_code == 200
#     body = resp.json()
#     assert body["rows"] == [{"page": 1, "change": "Added clause"}]
#     assert body["session_id"] == "abc123"


# def test_compare_failure(monkeypatch):
#     """ test_compare_failure() - Tests error handling in document comparison """
#     import api.main as main

#     class FailingComparator:
#         def __init__(self):
#             self.session_id = "x"
#         def save_uploaded_files(self, reference, actual):
#             return ("/tmp/ref.pdf", "/tmp/act.pdf")
#         def combine_documents(self):
#             raise RuntimeError("nope")

#     monkeypatch.setattr(main, "DocumentComparator", lambda: FailingComparator())
#     # LLM comparator won't be reached but mock anyway
#     monkeypatch.setattr(main, "DocumentComparatorLLM", lambda: object())

#     files = {
#         "reference": ("ref.pdf", b"ref", "application/pdf"),
#         "actual": ("act.pdf", b"act", "application/pdf"),
#     }
#     resp = client.post("/compare", files=files)
#     assert resp.status_code == 500
#     assert "Comparison failed" in resp.json()["detail"]


# def test_chat_index_success(monkeypatch):
#     """ test_chat_index_success() - Tests successful chat index creation with multiple files """
#     import api.main as main

#     class DummyCI:
#         def __init__(self, temp_base, faiss_base, use_session_dirs, session_id):
#             self.session_id = session_id or "sess-001"
#         def built_retriver(self, wrapped, chunk_size, chunk_overlap, k):
#             # wrapped is a list of adapters
#             assert isinstance(wrapped, list)
#             return None

#     monkeypatch.setattr(main, "ChatIngestor", lambda **kwargs: DummyCI(**kwargs))

#     # two files
#     files = [
#         ("files", ("a.pdf", b"a", "application/pdf")),
#         ("files", ("b.txt", b"b", "text/plain")),
#     ]
#     data = {"use_session_dirs": "true", "chunk_size": "500", "chunk_overlap": "50", "k": "3"}
#     resp = client.post("/chat/index", files=files, data=data)
#     assert resp.status_code == 200
#     body = resp.json()
#     assert body["session_id"] == "sess-001"
#     assert body["k"] == 3
#     assert body["use_session_dirs"] is True


# def test_chat_index_failure(monkeypatch):
#     """ test_chat_index_failure() - Tests error handling in chat index creation """
#     import api.main as main

#     class FailingCI:
#         def __init__(self, **kwargs):
#             self.session_id = "s"
#         def built_retriver(self, *a, **k):
#             raise RuntimeError("index fail")

#     monkeypatch.setattr(main, "ChatIngestor", lambda **kwargs: FailingCI(**kwargs))

#     files = [("files", ("a.pdf", b"a", "application/pdf"))]
#     resp = client.post("/chat/index", files=files, data={"use_session_dirs": "true", "k": "5"})
#     assert resp.status_code == 500
#     assert "Indexing failed" in resp.json()["detail"]


# def test_chat_query_requires_session_id_when_use_session_dirs_true():
#     """ test_chat_query_requires_session_id_when_use_session_dirs_true() - Tests validation logic for session_id requirement """
#     resp = client.post("/chat/query", data={"question": "Hi", "use_session_dirs": "true", "k": "5"})
#     assert resp.status_code == 400
#     assert "session_id is required" in resp.json()["detail"]


# def test_chat_query_index_not_found(monkeypatch):
#     """ test_chat_query_index_not_found() - Tests error handling when FAISS index directory doesn't exist """
#     import api.main as main
#     # Force directory to be missing
#     monkeypatch.setattr(main.os.path, "isdir", lambda p: False)

#     resp = client.post(
#         "/chat/query",
#         data={"question": "Hi", "session_id": "sess", "use_session_dirs": "true", "k": "2"},
#     )
#     assert resp.status_code == 404
#     assert "FAISS index not found" in resp.json()["detail"]


# def test_chat_query_success(monkeypatch, tmp_path):
#     """ test_chat_query_success() - Tests successful chat query with mocked RAG system """
#     import api.main as main

#     # Pretend index dir exists
#     monkeypatch.setattr(main.os.path, "isdir", lambda p: True)

#     class DummyRAG:
#         def __init__(self, session_id=None):
#             self.session_id = session_id
#         def load_retriever_from_faiss(self, index_dir, k, index_name):
#             # Validate parameters passed from endpoint
#             assert isinstance(index_dir, str)
#             assert isinstance(k, int)
#             assert isinstance(index_name, str)
#         def invoke(self, question, chat_history=None):
#             assert question == "What?"
#             return "Because..."

#     monkeypatch.setattr(main, "ConversationalRAG", lambda session_id=None: DummyRAG(session_id=session_id))

#     resp = client.post(
#         "/chat/query",
#         data={
#             "question": "What?",
#             "session_id": "sess-42",
#             "use_session_dirs": "true",
#             "k": "4",
#         },
#     )
#     assert resp.status_code == 200
#     body = resp.json()
#     assert body["answer"] == "Because..."
#     assert body["session_id"] == "sess-42"
#     assert body["k"] == 4
#     assert body["engine"] == "LCEL-RAG"


# # ============================================================================
# # UNIT TESTS FOR CORE COMPONENTS
# # ============================================================================

# class TestModelLoader:
#     """Test cases for ModelLoader class"""
    
#     @patch.dict(os.environ, {
#         'GROQ_API_KEY': 'test_groq_key',
#         'GOOGLE_API_KEY': 'test_google_key',
#         'ENV': 'test'
#     })
#     @patch('utils.model_loader.load_config')
#     def test_model_loader_initialization(self, mock_load_config):
#         """Test ModelLoader initializes correctly with API keys"""
#         mock_load_config.return_value = {
#             'embedding_model': {'model_name': 'models/embedding-001'},
#             'llm': {'google': {'provider': 'google', 'model_name': 'gemini-pro'}}
#         }
        
#         from utils.model_loader import ModelLoader
#         loader = ModelLoader()
        
#         assert loader.api_key_mgr.get('GROQ_API_KEY') == 'test_groq_key'
#         assert loader.api_key_mgr.get('GOOGLE_API_KEY') == 'test_google_key'
#         mock_load_config.assert_called_once()
    
#     @patch.dict(os.environ, {}, clear=True)
#     def test_model_loader_missing_api_keys(self):
#         """Test ModelLoader raises exception when API keys are missing"""
#         from utils.model_loader import ModelLoader
#         from exception.custom_exception import DocumentPortalException
        
#         with pytest.raises(DocumentPortalException):
#             ModelLoader()
    
#     @patch.dict(os.environ, {
#         'GROQ_API_KEY': 'test_groq_key',
#         'GOOGLE_API_KEY': 'test_google_key'
#     })
#     @patch('utils.model_loader.load_config')
#     @patch('utils.model_loader.GoogleGenerativeAIEmbeddings')
#     def test_load_embeddings_success(self, mock_embeddings, mock_load_config):
#         """Test successful embedding model loading"""
#         mock_load_config.return_value = {
#             'embedding_model': {'model_name': 'models/embedding-001'},
#             'llm': {'google': {'provider': 'google', 'model_name': 'gemini-pro'}}
#         }
#         mock_embeddings.return_value = Mock()
        
#         from utils.model_loader import ModelLoader
#         loader = ModelLoader()
#         embeddings = loader.load_embeddings()
        
#         mock_embeddings.assert_called_once_with(
#             model='models/embedding-001',
#             google_api_key='test_google_key'
#         )
#         assert embeddings is not None


# class TestFaissManager:
#     """Test cases for FaissManager class"""
    
#     def test_faiss_manager_initialization(self, tmp_path):
#         """Test FaissManager initializes correctly"""
#         from src.document_ingestion.data_ingestion import FaissManager
        
#         with patch('src.document_ingestion.data_ingestion.ModelLoader') as mock_loader:
#             mock_model_loader = Mock()
#             mock_model_loader.load_embeddings.return_value = Mock()
#             mock_loader.return_value = mock_model_loader
            
#             fm = FaissManager(tmp_path / "test_index")
            
#             assert fm.index_dir.exists()
#             assert fm.meta_path.exists() or not fm.meta_path.exists()  # Can be either
#             assert fm._meta == {"rows": {}}
    
#     def test_fingerprint_generation(self):
#         """Test fingerprint generation for deduplication"""
#         from src.document_ingestion.data_ingestion import FaissManager
        
#         # Test with source metadata
#         metadata1 = {"source": "test.pdf", "row_id": "1"}
#         fingerprint1 = FaissManager._fingerprint("test text", metadata1)
#         assert fingerprint1 == "test.pdf::1"
        
#         # Test without source metadata (should use hash)
#         metadata2 = {"other": "data"}
#         fingerprint2 = FaissManager._fingerprint("test text", metadata2)
#         assert len(fingerprint2) == 64  # SHA256 hash length
    
#     def test_add_documents_deduplication(self, tmp_path):
#         """Test document deduplication in add_documents"""
#         from src.document_ingestion.data_ingestion import FaissManager
#         from langchain.schema import Document
        
#         with patch('src.document_ingestion.data_ingestion.ModelLoader') as mock_loader:
#             mock_model_loader = Mock()
#             mock_embeddings = Mock()
#             mock_model_loader.load_embeddings.return_value = mock_embeddings
#             mock_loader.return_value = mock_model_loader
            
#             fm = FaissManager(tmp_path / "test_index")
            
#             # Mock FAISS vectorstore
#             mock_vs = Mock()
#             mock_vs.add_documents = Mock()
#             mock_vs.save_local = Mock()
#             fm.vs = mock_vs
            
#             # Add same document twice
#             doc = Document(page_content="test", metadata={"source": "test.pdf"})
            
#             # First addition should add document
#             added1 = fm.add_documents([doc])
#             assert added1 == 1
#             mock_vs.add_documents.assert_called_once()
            
#             # Second addition should skip (deduplication)
#             mock_vs.add_documents.reset_mock()
#             added2 = fm.add_documents([doc])
#             assert added2 == 0
#             mock_vs.add_documents.assert_not_called()


# class TestChatIngestor:
#     """Test cases for ChatIngestor class"""
    
#     @patch('src.document_ingestion.data_ingestion.ModelLoader')
#     @patch('src.document_ingestion.data_ingestion.CustomLogger')
#     def test_chat_ingestor_initialization(self, mock_logger, mock_model_loader, tmp_path):
#         """Test ChatIngestor initializes correctly"""
#         from src.document_ingestion.data_ingestion import ChatIngestor
        
#         mock_logger.return_value.get_logger.return_value = Mock()
#         mock_model_loader.return_value = Mock()
        
#         ci = ChatIngestor(
#             temp_base=str(tmp_path / "data"),
#             faiss_base=str(tmp_path / "faiss"),
#             session_id="test_session"
#         )
        
#         assert ci.session_id == "test_session"
#         assert ci.temp_dir.exists()
#         assert ci.faiss_dir.exists()
    
#     @patch('src.document_ingestion.data_ingestion.ModelLoader')
#     @patch('src.document_ingestion.data_ingestion.CustomLogger')
#     def test_document_splitting(self, mock_logger, mock_model_loader):
#         """Test document splitting functionality"""
#         from src.document_ingestion.data_ingestion import ChatIngestor
#         from langchain.schema import Document
        
#         mock_logger.return_value.get_logger.return_value = Mock()
#         mock_model_loader.return_value = Mock()
        
#         ci = ChatIngestor(session_id="test")
        
#         # Create a long document for splitting
#         long_text = "This is a test document. " * 200  # Should trigger splitting
#         docs = [Document(page_content=long_text, metadata={"source": "test.pdf"})]
        
#         chunks = ci._split(docs, chunk_size=100, chunk_overlap=20)
        
#         assert len(chunks) > 1  # Should be split into multiple chunks
#         assert all(isinstance(chunk, Document) for chunk in chunks)


# class TestDocumentAnalyzer:
#     """Test cases for DocumentAnalyzer class"""
    
#     @patch('src.document_analyzer.data_analysis.ModelLoader')
#     @patch('src.document_analyzer.data_analysis.CustomLogger')
#     @patch('src.document_analyzer.data_analysis.PROMPT_REGISTRY')
#     def test_document_analyzer_initialization(self, mock_prompt_registry, mock_logger, mock_model_loader):
#         """Test DocumentAnalyzer initializes correctly"""
#         from src.document_analyzer.data_analysis import DocumentAnalyzer
        
#         mock_logger.return_value.get_logger.return_value = Mock()
#         mock_llm = Mock()
#         mock_model_loader.return_value.load_llm.return_value = mock_llm
#         mock_prompt_registry.__getitem__.return_value = Mock()
        
#         analyzer = DocumentAnalyzer()
        
#         assert analyzer.llm == mock_llm
#         assert analyzer.parser is not None
#         assert analyzer.fixing_parser is not None
    
#     @patch('src.document_analyzer.data_analysis.ModelLoader')
#     @patch('src.document_analyzer.data_analysis.CustomLogger')
#     @patch('src.document_analyzer.data_analysis.PROMPT_REGISTRY')
#     def test_analyze_document_success(self, mock_prompt_registry, mock_logger, mock_model_loader):
#         """Test successful document analysis"""
#         from src.document_analyzer.data_analysis import DocumentAnalyzer
        
#         mock_logger.return_value.get_logger.return_value = Mock()
#         mock_llm = Mock()
#         mock_model_loader.return_value.load_llm.return_value = mock_llm
        
#         # Mock the chain execution
#         mock_chain = Mock()
#         expected_result = {"Summary": ["Test summary"], "Title": "Test Title"}
#         mock_chain.invoke.return_value = expected_result
        
#         # Mock prompt and chain building
#         mock_prompt = Mock()
#         mock_prompt.__or__ = Mock(return_value=mock_chain)
#         mock_prompt_registry.__getitem__.return_value = mock_prompt
        
#         analyzer = DocumentAnalyzer()
#         analyzer.fixing_parser = Mock()
#         analyzer.fixing_parser.__ror__ = Mock(return_value=mock_chain)
        
#         result = analyzer.analyze_document("Test document text")
        
#         assert result == expected_result
#         mock_chain.invoke.assert_called_once()


# class TestConversationalRAG:
#     """Test cases for ConversationalRAG class"""
    
#     @patch('src.document_chat.retrieval.ModelLoader')
#     @patch('src.document_chat.retrieval.CustomLogger')
#     @patch('src.document_chat.retrieval.PROMPT_REGISTRY')
#     def test_conversational_rag_initialization(self, mock_prompt_registry, mock_logger, mock_model_loader):
#         """Test ConversationalRAG initializes correctly"""
#         from src.document_chat.retrieval import ConversationalRAG
        
#         mock_logger.return_value.get_logger.return_value = Mock()
#         mock_llm = Mock()
#         mock_model_loader.return_value.load_llm.return_value = mock_llm
#         mock_prompt_registry.__getitem__.return_value = Mock()
        
#         rag = ConversationalRAG(session_id="test_session")
        
#         assert rag.session_id == "test_session"
#         assert rag.llm == mock_llm
#         assert rag.retriever is None  # Should be None initially
#         assert rag.chain is None  # Should be None initially
    
#     @patch('src.document_chat.retrieval.ModelLoader')
#     @patch('src.document_chat.retrieval.CustomLogger')
#     @patch('src.document_chat.retrieval.PROMPT_REGISTRY')
#     @patch('src.document_chat.retrieval.FAISS')
#     def test_load_retriever_from_faiss_success(self, mock_faiss, mock_prompt_registry, mock_logger, mock_model_loader, tmp_path):
#         """Test successful FAISS retriever loading"""
#         from src.document_chat.retrieval import ConversationalRAG
        
#         mock_logger.return_value.get_logger.return_value = Mock()
#         mock_llm = Mock()
#         mock_model_loader_instance = Mock()
#         mock_model_loader_instance.load_llm.return_value = mock_llm
#         mock_model_loader_instance.load_embeddings.return_value = Mock()
#         mock_model_loader.return_value = mock_model_loader_instance
#         mock_prompt_registry.__getitem__.return_value = Mock()
        
#         # Mock FAISS loading
#         mock_vectorstore = Mock()
#         mock_retriever = Mock()
#         mock_vectorstore.as_retriever.return_value = mock_retriever
#         mock_faiss.load_local.return_value = mock_vectorstore
        
#         # Create dummy index directory
#         index_dir = tmp_path / "faiss_index"
#         index_dir.mkdir()
        
#         rag = ConversationalRAG(session_id="test")
#         result = rag.load_retriever_from_faiss(str(index_dir), k=5)
        
#         assert result == mock_retriever
#         assert rag.retriever == mock_retriever
#         assert rag.chain is not None  # Should build chain after loading retriever
    
#     @patch('src.document_chat.retrieval.ModelLoader')
#     @patch('src.document_chat.retrieval.CustomLogger')
#     @patch('src.document_chat.retrieval.PROMPT_REGISTRY')
#     def test_invoke_without_retriever_fails(self, mock_prompt_registry, mock_logger, mock_model_loader):
#         """Test that invoke fails when retriever is not loaded"""
#         from src.document_chat.retrieval import ConversationalRAG
#         from exception.custom_exception import DocumentPortalException
        
#         mock_logger.return_value.get_logger.return_value = Mock()
#         mock_llm = Mock()
#         mock_model_loader.return_value.load_llm.return_value = mock_llm
#         mock_prompt_registry.__getitem__.return_value = Mock()
        
#         rag = ConversationalRAG(session_id="test")
        
#         with pytest.raises(DocumentPortalException):
#             rag.invoke("What is this document about?")


# # ============================================================================
# # DATA MODEL VALIDATION TESTS
# # ============================================================================

# class TestDataModels:
#     """Test cases for Pydantic data models"""
    
#     def test_metadata_model_validation(self):
#         """Test Metadata model validation"""
#         from model.models import Metadata
        
#         valid_data = {
#             "Summary": ["Test summary"],
#             "Title": "Test Document",
#             "Author": ["John Doe"],
#             "DateCreated": "2024-01-01",
#             "LastModifiedDate": "2024-01-02",
#             "Publisher": "Test Publisher",
#             "Language": "English",
#             "PageCount": 10,
#             "SentimentTone": "Neutral"
#         }
        
#         metadata = Metadata(**valid_data)
#         assert metadata.Title == "Test Document"
#         assert metadata.PageCount == 10
#         assert len(metadata.Author) == 1
    
#     def test_change_format_model_validation(self):
#         """Test ChangeFormat model validation"""
#         from model.models import ChangeFormat
        
#         valid_data = {
#             "Page": "1",
#             "Changes": "Added new paragraph"
#         }
        
#         change = ChangeFormat(**valid_data)
#         assert change.Page == "1"
#         assert change.Changes == "Added new paragraph"
    
#     def test_summary_response_model_validation(self):
#         """Test SummaryResponse model validation"""
#         from model.models import SummaryResponse, ChangeFormat
        
#         valid_data = [
#             {"Page": "1", "Changes": "Added introduction"},
#             {"Page": "2", "Changes": "NO CHANGE"}
#         ]
        
#         summary = SummaryResponse(valid_data)
#         assert len(summary.root) == 2
#         assert summary.root[0].Page == "1"


# class TestConfigLoader:
#     """Test cases for configuration loading"""
    
#     def test_load_config_default_path(self, tmp_path):
#         """Test loading config from default path"""
#         from utils.config_loader import load_config
        
#         # Create a temporary config file
#         config_dir = tmp_path / "config"
#         config_dir.mkdir()
#         config_file = config_dir / "config.yaml"
        
#         config_content = """
# embedding_model:
#   model_name: "models/embedding-001"
  
# llm:
#   google:
#     provider: "google"
#     model_name: "gemini-pro"
#     temperature: 0.2
#         """
#         config_file.write_text(config_content)
        
#         # Mock project root to return our tmp_path
#         with patch('utils.config_loader._project_root', return_value=tmp_path):
#             config = load_config()
            
#             assert "embedding_model" in config
#             assert config["embedding_model"]["model_name"] == "models/embedding-001"
#             assert config["llm"]["google"]["provider"] == "google"
    
#     def test_load_config_custom_path(self, tmp_path):
#         """Test loading config from custom path"""
#         from utils.config_loader import load_config
        
#         # Create a custom config file
#         config_file = tmp_path / "custom_config.yaml"
#         config_content = """
# test_setting: "custom_value"
#         """
#         config_file.write_text(config_content)
        
#         config = load_config(str(config_file))
#         assert config["test_setting"] == "custom_value"
    
#     def test_load_config_missing_file(self, tmp_path):
#         """Test loading config with missing file raises error"""
#         from utils.config_loader import load_config
        
#         non_existent_file = tmp_path / "missing_config.yaml"
        
#         with pytest.raises(FileNotFoundError):
#             load_config(str(non_existent_file))


# # ============================================================================
# # INTEGRATION TESTS
# # ============================================================================

# class TestDocumentProcessingIntegration:
#     """Integration tests for document processing workflow"""
    
#     def create_mock_pdf_file(self, content="Test PDF content"):
#         """Helper to create mock PDF file-like object"""
#         file_obj = io.BytesIO()
#         file_obj.write(f"%PDF-1.4\n{content}".encode())
#         file_obj.seek(0)
#         file_obj.name = "test.pdf"
#         return file_obj
    
#     @patch('src.document_ingestion.data_ingestion.fitz')
#     def test_doc_handler_save_and_read_pdf(self, mock_fitz, tmp_path):
#         """Test DocHandler can save and read PDF files"""
#         from src.document_ingestion.data_ingestion import DocHandler
        
#         # Mock PyMuPDF
#         mock_doc = Mock()
#         mock_doc.page_count = 2
#         mock_page1 = Mock()
#         mock_page1.get_text.return_value = "Page 1 content"
#         mock_page2 = Mock()
#         mock_page2.get_text.return_value = "Page 2 content"
#         mock_doc.load_page.side_effect = [mock_page1, mock_page2]
#         mock_fitz.open.return_value.__enter__.return_value = mock_doc
        
#         dh = DocHandler(data_dir=str(tmp_path), session_id="test_session")
        
#         # Test save PDF
#         mock_file = self.create_mock_pdf_file()
#         saved_path = dh.save_pdf(mock_file)
        
#         assert os.path.exists(saved_path)
#         assert saved_path.endswith(".pdf")
        
#         # Test read PDF
#         text = dh.read_pdf(saved_path)
        
#         assert "Page 1 content" in text
#         assert "Page 2 content" in text
#         assert "--- Page 1 ---" in text
    
#     def test_document_comparator_workflow(self, tmp_path):
#         """Test DocumentComparator full workflow"""
#         from src.document_ingestion.data_ingestion import DocumentComparator
        
#         dc = DocumentComparator(base_dir=str(tmp_path), session_id="test_session")
        
#         # Create mock files
#         ref_file = self.create_mock_pdf_file("Reference content")
#         ref_file.name = "reference.pdf"
#         act_file = self.create_mock_pdf_file("Actual content")
#         act_file.name = "actual.pdf"
        
#         # Test save files
#         with patch('src.document_ingestion.data_ingestion.fitz'):
#             ref_path, act_path = dc.save_uploaded_files(ref_file, act_file)
            
#             assert os.path.exists(ref_path)
#             assert os.path.exists(act_path)
#             assert ref_path.name == "reference.pdf"
#             assert act_path.name == "actual.pdf"


# # ============================================================================
# # ERROR HANDLING AND EDGE CASE TESTS
# # ============================================================================

# class TestErrorHandling:
#     """Test error handling and edge cases"""
    
#     def test_invalid_file_types(self):
#         """Test handling of invalid file types"""
#         files = {"file": ("test.txt", b"text content", "text/plain")}
        
#         # This should work for document analysis (depends on implementation)
#         # But let's test with an unsupported type
#         files_invalid = {"file": ("test.exe", b"binary content", "application/octet-stream")}
        
#         with patch('api.main.DocHandler') as mock_handler:
#             mock_handler.return_value.save_pdf.side_effect = ValueError("Invalid file type")
            
#             resp = client.post("/analyze", files=files_invalid)
#             assert resp.status_code == 500
    
#     def test_empty_file_handling(self):
#         """Test handling of empty files"""
#         files = {"file": ("empty.pdf", b"", "application/pdf")}
        
#         with patch('api.main.DocHandler') as mock_handler:
#             mock_handler.return_value.save_pdf.return_value = "empty.pdf"
            
#             with patch('api.main.read_pdf_via_handler') as mock_reader:
#                 mock_reader.return_value = ""
                
#                 with patch('api.main.DocumentAnalyzer') as mock_analyzer:
#                     mock_analyzer.return_value.analyze_document.return_value = {"error": "Empty document"}
                    
#                     resp = client.post("/analyze", files=files)
#                     assert resp.status_code == 200
#                     assert "error" in resp.json()
    
#     def test_large_file_handling(self):
#         """Test handling of large files (mock scenario)"""
#         # Create a "large" file scenario
#         large_content = "x" * 10000  # 10KB mock large file
#         files = {"file": ("large.pdf", large_content.encode(), "application/pdf")}
        
#         with patch('api.main.DocHandler') as mock_handler:
#             mock_handler.return_value.save_pdf.return_value = "large.pdf"
            
#             with patch('api.main.read_pdf_via_handler') as mock_reader:
#                 mock_reader.return_value = large_content
                
#                 with patch('api.main.DocumentAnalyzer') as mock_analyzer:
#                     mock_analyzer.return_value.analyze_document.return_value = {
#                         "Summary": ["Large document processed"],
#                         "length": len(large_content)
#                     }
                    
#                     resp = client.post("/analyze", files=files)
#                     assert resp.status_code == 200
#                     result = resp.json()
#                     assert result["length"] == len(large_content)
    
#     def test_concurrent_session_handling(self, tmp_path):
#         """Test handling of concurrent sessions"""
#         from src.document_ingestion.data_ingestion import ChatIngestor
        
#         with patch('src.document_ingestion.data_ingestion.ModelLoader') as mock_loader:
#             mock_loader.return_value = Mock()
            
#             # Create multiple ingestors with different sessions
#             ci1 = ChatIngestor(
#                 temp_base=str(tmp_path / "data"),
#                 faiss_base=str(tmp_path / "faiss"),
#                 session_id="session_1"
#             )
            
#             ci2 = ChatIngestor(
#                 temp_base=str(tmp_path / "data"),
#                 faiss_base=str(tmp_path / "faiss"),
#                 session_id="session_2"
#             )
            
#             # Verify sessions are isolated
#             assert ci1.session_id != ci2.session_id
#             assert ci1.temp_dir != ci2.temp_dir
#             assert ci1.faiss_dir != ci2.faiss_dir


# # ============================================================================
# # PERFORMANCE TESTS
# # ============================================================================

# class TestPerformance:
#     """Basic performance validation tests"""
    
#     def test_health_endpoint_response_time(self):
#         """Test health endpoint responds quickly"""
#         import time
        
#         start_time = time.time()
#         resp = client.get("/health")
#         end_time = time.time()
        
#         response_time = end_time - start_time
        
#         assert resp.status_code == 200
#         assert response_time < 1.0  # Should respond within 1 second
    
#     def test_multiple_concurrent_health_checks(self):
#         """Test multiple concurrent health check requests"""
#         import concurrent.futures
#         import time
        
#         def make_health_request():
#             start = time.time()
#             resp = client.get("/health")
#             end = time.time()
#             return resp.status_code, end - start
        
#         # Make 5 concurrent requests
#         with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
#             futures = [executor.submit(make_health_request) for _ in range(5)]
#             results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
#         # All should succeed
#         assert all(status == 200 for status, _ in results)
#         # All should be reasonably fast
#         assert all(duration < 2.0 for _, duration in results)



# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# """ tests/test_unit_cases.py """

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
    
def test_chat_index_success(monkeypatch):
    """ test_chat_index_success() - Tests successful chat index creation with multiple files """
    import api.main as main

    class DummyCI:
        def __init__(self, temp_base, faiss_base, use_session_dirs, session_id):
            self.session_id = session_id or "sess-001"
        def built_retriver(self, wrapped, chunk_size, chunk_overlap, k):
            # wrapped is a list of adapters
            assert isinstance(wrapped, list)
            return None

    monkeypatch.setattr(main, "ChatIngestor", lambda **kwargs: DummyCI(**kwargs))

    # two files
    files = [
        ("files", ("a.pdf", b"a", "application/pdf")),
        ("files", ("b.txt", b"b", "text/plain")),
    ]
    data = {"use_session_dirs": "true", "chunk_size": "500", "chunk_overlap": "50", "k": "3"}
    resp = client.post("/chat/index", files=files, data=data)
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == "sess-001"
    assert body["k"] == 3
    assert body["use_session_dirs"] is True
    
def test_chat_index_failure(monkeypatch):
    """ test_chat_index_failure() - Tests error handling in chat index creation """
    import api.main as main

    class FailingCI:
        def __init__(self, **kwargs):
            self.session_id = "s"
        def built_retriver(self, *a, **k):
            raise RuntimeError("index fail")

    monkeypatch.setattr(main, "ChatIngestor", lambda **kwargs: FailingCI(**kwargs))

    files = [("files", ("a.pdf", b"a", "application/pdf"))]
    resp = client.post("/chat/index", files=files, data={"use_session_dirs": "true", "k": "5"})
    assert resp.status_code == 500
    assert "Indexing failed" in resp.json()["detail"]
    
def test_chat_query_requires_session_id_when_use_session_dirs_true():
    """ test_chat_query_requires_session_id_when_use_session_dirs_true() - Tests validation logic for session_id requirement """
    resp = client.post("/chat/query", data={"question": "Hi", "use_session_dirs": "true", "k": "5"})
    assert resp.status_code == 400
    assert "session_id is required" in resp.json()["detail"]
    
def test_chat_query_index_not_found(monkeypatch):
    """ test_chat_query_index_not_found() - Tests error handling when FAISS index directory doesn't exist """
    import api.main as main
    # Force directory to be missing
    monkeypatch.setattr(main.os.path, "isdir", lambda p: False)

    resp = client.post(
        "/chat/query",
        data={"question": "Hi", "session_id": "sess", "use_session_dirs": "true", "k": "2"},
    )
    assert resp.status_code == 404
    assert "FAISS index not found" in resp.json()["detail"]
    
def test_chat_query_success(monkeypatch, tmp_path):
    """ test_chat_query_success() - Tests successful chat query with mocked RAG system """
    import api.main as main

    # Pretend index dir exists
    monkeypatch.setattr(main.os.path, "isdir", lambda p: True)

    class DummyRAG:
        def __init__(self, session_id=None):
            self.session_id = session_id
        def load_retriever_from_faiss(self, index_dir, k, index_name):
            # Validate parameters passed from endpoint
            assert isinstance(index_dir, str)
            assert isinstance(k, int)
            assert isinstance(index_name, str)
        def invoke(self, question, chat_history=None):
            assert question == "What?"
            return "Because..."

    monkeypatch.setattr(main, "ConversationalRAG", lambda session_id=None: DummyRAG(session_id=session_id))

    resp = client.post(
        "/chat/query",
        data={
            "question": "What?",
            "session_id": "sess-42",
            "use_session_dirs": "true",
            "k": "4",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Because..."
    assert body["session_id"] == "sess-42"
    assert body["k"] == 4
    assert body["engine"] == "LCEL-RAG"