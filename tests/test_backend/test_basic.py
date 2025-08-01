from unittest.mock import patch, Mock

def test_temp_dir_fixture(temp_dir):
    """Test that the temp directory fixture works"""

    test_file = temp_dir / "hello.txt"
    test_file.write_text("Hello from test!")

    assert test_file.exists()
    assert test_file.read_text() == "Hello from test!"


def test_pdf_service_imports():
    """Test that we can import PDFService"""
    with patch.dict('sys.modules', {
        # Core llama_index modules
        'llama_index': Mock(),
        'llama_index.core': Mock(),
        'llama_index.core.base': Mock(),
        'llama_index.core.base.response': Mock(),
        'llama_index.core.base.response.schema': Mock(),
        'llama_index.core.base.llms': Mock(),
        'llama_index.core.base.llms.types': Mock(),
        'llama_index.core.llms': Mock(),
        'llama_index.core.llms.callbacks': Mock(),
        'llama_index.core.llms.custom': Mock(),
        'llama_index.core.bridge': Mock(),
        'llama_index.core.bridge.pydantic': Mock(),
        'llama_index.core.embeddings': Mock(),
        'llama_index.llms': Mock(),
        'llama_index.llms.llama_cpp': Mock(),
        'llama_index.llms.ollama': Mock(),
        # External dependencies
        'llama_cpp': Mock(),
        'pymupdf': Mock(),
        'requests': Mock(),
    }):
        from src.backend.service import PDFService

        service = PDFService()
        assert service is not None


def test_import_paths():
    try:
        from backend.agent import PDFAgent
        print("backend.agent works!")
    except ImportError as e:
        print("backend.agent failed: ", e)
    
    try:
        from src.backend.agent import PDFAgent
        print("src.backend.agent works!")
    except ImportError as e:
        print("src.backend.agent failed: ", e)
