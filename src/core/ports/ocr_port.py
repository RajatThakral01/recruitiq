from abc import ABC, abstractmethod

class OCRPort(ABC):
    """
    Abstract port for text extraction (OCR).
    """
    @abstractmethod
    def extract_text(self, file_path: str) -> str:
        """
        Extracts raw text from a PDF file.
        """
        pass

    @abstractmethod
    def validate_file(self, file_path: str) -> bool:
        """
        Validates the file before processing (type/size).
        """
        pass
