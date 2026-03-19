import os
from pathlib import Path
import pdfplumber
import re
from src.core.ports.ocr_port import OCRPort
from src.infrastructure.config import settings
from src.infrastructure.logger import logger
from src.infrastructure.exceptions import OCRFailedException, ScoringException

class PDFPlumberAdapter(OCRPort):
    """Concrete implementation of OCRPort using pdfplumber."""

    def validate_file(self, file_path: str) -> bool:
        """Validate existence, extension, and size of the PDF file.

        Returns True if all checks pass, otherwise logs the reason and returns False.
        """
        try:
            p = Path(file_path)
            if not p.is_file():
                logger.error(f"OCR validation failed: file does not exist -> {file_path}")
                return False
            if p.suffix.lower() != ".pdf":
                logger.error(f"OCR validation failed: unsupported extension {p.suffix} for {file_path}")
                return False
            max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
            if p.stat().st_size > max_bytes:
                logger.error(f"OCR validation failed: file size exceeds limit ({p.stat().st_size} > {max_bytes})")
                return False
            return True
        except Exception as e:
            logger.error(f"OCR validation exception: {e}")
            return False

    def extract_text(self, file_path: str) -> str:
        """Extract and clean text from a PDF using pdfplumber.

        Raises:
            OCRFailedException: if pdfplumber cannot read the file.
        """
        if not self.validate_file(file_path):
            raise OCRFailedException("File validation failed for OCR extraction.")
        try:
            with pdfplumber.open(file_path) as pdf:
                pages_text = []
                for i, page in enumerate(pdf.pages, start=1):
                    txt = page.extract_text() or ""
                    pages_text.append(txt)
                raw_text = "\n".join(pages_text)
                logger.info(f"Extracted {len(pdf.pages)} pages, {len(raw_text)} characters from {file_path}")
        except Exception as e:
            logger.error(f"PDFPlumber extraction error: {e}")
            raise OCRFailedException("Failed to extract text with pdfplumber.", detail=str(e))

        # Clean up whitespace
        # Collapse 3+ spaces into a single space
        cleaned = re.sub(r" {3,}", " ", raw_text)
        # Collapse 3+ newlines into two newlines
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = cleaned.strip()

        if len(cleaned) < 50:
            logger.warning("Low text extraction, possible scanned PDF or empty content.")

        return cleaned
