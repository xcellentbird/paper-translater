"""
utils/pdf_loader.py
"""

from pathlib import Path

import httpx

from config import DATA_ROOT

ARXIV_URL_PREFIX = "https://arxiv.org/abs/"
ARXIV_PDF_URL_PREFIX = "https://arxiv.org/pdf/"


def load_url(url: str) -> bytes:
    """주어진 URL 에서 바이너리 데이터를 받아온다."""
    response = httpx.get(url)
    response.raise_for_status()
    return response.content


class ArxivPDFLoader:
    """arXiv 논문 PDF 를 다운로드하는 책임을 가진 로더."""

    def __init__(self) -> None:
        pass

    def _normalize_to_pdf_url(self, paper_url: str) -> str:
        """abs / pdf URL 을 모두 PDF 다운로드용 URL 로 정규화."""
        if paper_url.startswith(ARXIV_URL_PREFIX):
            return paper_url.replace(ARXIV_URL_PREFIX, ARXIV_PDF_URL_PREFIX)
        if paper_url.startswith(ARXIV_PDF_URL_PREFIX):
            return paper_url
        raise ValueError(f"Invalid paper URL: {paper_url}")

    def download_pdf(self, paper_url: str, save_root: Path = DATA_ROOT) -> Path:
        """arXiv URL(Absolute 또는 PDF)을 받아 PDF 파일을 로컬에 저장한다."""
        pdf_url = self._normalize_to_pdf_url(paper_url)
        content = load_url(pdf_url)

        filename = pdf_url.split("/")[-1]
        save_path = save_root / f"{filename}.pdf"

        with open(save_path, "wb") as file:
            file.write(content)

        return save_path
