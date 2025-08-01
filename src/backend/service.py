from typing import List, Optional, TYPE_CHECKING
from dotenv import load_dotenv
import time, os, shutil

if TYPE_CHECKING:
    from src.backend.agent import PDFAgent
    import pymupdf as pd

load_dotenv(verbose=True)

class PDFService:
    """
    Service class for handling all PDF operations including loading, parsing, and querying.
    """
    def __init__(self, agent: "PDFAgent" =None):
        self.pdf: Optional["pd.Document"] = None  # raw document handle
        self.agent = agent or self._create_default_agent()
        # make sure storage/ui exists and clear it
        os.makedirs("storage/ui", exist_ok=True)
        self._clear_ui_folder()
        # make sure storage/data exists and clear it
        self._clear_data_folder()

    def load_pdf(self, file_path: str) -> List[str]:
        """
        Discards old PDF, loads a new one from the given path.
        Returns a list of image paths for each page in the PDF to be rendered in the UI.
        """
        import pymupdf as pd
        start = time.time()
        self._discard_pdf()

        assert os.path.exists(file_path), f"File {file_path} does not exist on the disk."
        self.pdf = pd.open(file_path)
        assert self.pdf is not None, "PyMuPDF failed to load the document."

        self._convert_pages_to_images(os.path.basename(file_path))
        print(f"-*-File {os.path.basename(file_path)} loaded successfully in {round(time.time()-start, 2)}s!-*-")

        self.agent.create_index(file_path)

        return self._get_image_paths()

    def _discard_pdf(self) -> None:
        """
        Discards the currently loaded file and clears all related data to prepare for a new file.
        """
        if self.pdf is not None:
            self.pdf.close()
            print("--Old file closed!--")
            self._clear_ui_folder()
            self.parser.clear_parsed_content()
            self._clear_data_folder()

    def _convert_pages_to_images(self, file_name: str) -> None:
        """
        Converts each page of the loaded PDF into a PNG image and saves them to ~/storage/ui for rendering.
        """
        assert os.path.exists("storage/ui"), "UI storage folder does not exist. Please create it first."
        for i, page in enumerate(self.pdf):
            page_png = page.get_pixmap(dpi=150)
            page_png.save(f"storage/ui/{file_name[:9]}_{i:04d}.png")
        print("--UI images created!--")

    def _get_image_paths(self) -> List[str]:
        """
        Returns a list of image paths, each of which represents a page from the loaded PDF file. The images live in ~/storage/ui/.
        """
        assert os.path.exists("storage/ui"), "UI storage folder does not exist. Please create it first."
        assert os.listdir("storage/ui"), "UI storage folder is empty. Please load a PDF file first."
        paths = sorted([os.path.abspath(os.path.join("storage/ui", fname)) for fname in os.listdir("storage/ui")])
        return paths

    def _create_default_agent(self):
        """Create default agent for production use"""
        from src.backend.agent import PDFAgent
        return PDFAgent(llm_backend="docker")

    @staticmethod
    def _clear_ui_folder() -> None:
        """
        Deletes all images (pdf pages) from the UI folder and makes sure we get an empty folder.
        """
        if os.path.exists(os.getenv('UI_PATH')):
            shutil.rmtree(os.getenv('UI_PATH'))
        os.makedirs(os.getenv('UI_PATH'), exist_ok=True)
        print(f"--UI folder cleared--")

    @staticmethod
    def _clear_data_folder() -> None:
        """
        Clears the data folder where indexed documents are stored and makes sure we get an empty folder.
        """
        if os.path.exists(os.getenv('DATA_PATH')):
            shutil.rmtree(os.getenv('DATA_PATH'))
        os.makedirs(os.getenv('DATA_PATH'), exist_ok=True)
        print(f"--Data folder cleared--")

        

    

