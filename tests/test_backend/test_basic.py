from unittest.mock import Mock

def test_temp_dir_fixture(temp_dir):
    """Test that the temp directory fixture works"""

    test_file = temp_dir / "hello.txt"
    test_file.write_text("Hello from test!")

    assert test_file.exists()
    assert test_file.read_text() == "Hello from test!"


def test_pdf_service_with_mock_agent():
    """Test PDFService with injected mock agent"""
    mock_agent = Mock()
    from src.backend.service import PDFService
    service = PDFService(agent=mock_agent)
    
    assert service is not None
    assert service.agent == mock_agent


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
