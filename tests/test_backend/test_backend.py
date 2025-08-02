from unittest.mock import patch, Mock

def test_temp_dir_fixture(temp_dir):
    """Test that the temp directory fixture works"""

    test_file = temp_dir / "hello.txt"
    test_file.write_text("Hello from test!")

    assert test_file.exists()
    assert test_file.read_text() == "Hello from test!"

def test_import_paths():
    try:
        from src.backend.agent import PDFAgent
        print("backend.agent works!")
    except ImportError as e:
        print("backend.agent failed: ", e)
    
    try:
        from src.backend.agent import PDFAgent
        print("src.backend.agent works!")
    except ImportError as e:
        print("src.backend.agent failed: ", e)
        
def test_service_init_with_mock_agent():
    """Test PDFService with injected mock agent"""
    mock_agent = Mock()
    from src.backend.service import PDFService
    service = PDFService(agent=mock_agent)
    
    assert service is not None
    assert service.agent == mock_agent

@patch('pymupdf.open')
def test_service_load_pdf(mock_pdf_open, temp_dir):
    """Test loading a PDF document"""
    mock_pdf_path = temp_dir / "test.pdf"
    mock_pdf_path.touch()

    mock_agent = Mock()
    from src.backend.service import PDFService

    with patch.object(PDFService, '_get_image_paths') as mock_get_paths, \
        patch.object(PDFService, '_convert_pages_to_images') as mock_convert:

        mock_document = Mock()
        mock_pdf_open.return_value = mock_document
        mock_get_paths.return_value = ['page_001.png', 'page_002.png']

        service = PDFService(agent=mock_agent)
        result = service.load_pdf(str(mock_pdf_path))

        # 1: PDF was loaded
        mock_pdf_open.assert_called_once_with(str(mock_pdf_path))

        # 2: Index was created via agent
        mock_agent.create_index.assert_called_once_with(str(mock_pdf_path))

        # 3: Image paths were returned
        assert result == ['page_001.png', 'page_002.png']


        




