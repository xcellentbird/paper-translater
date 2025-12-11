import pathlib

import pytest

import pymupdf
from utils.pdf_loader import ArxivPDFLoader


TEST_PDF_URL = "https://arxiv.org/pdf/2405.07437"
TEST_SAVE_PATH = pathlib.Path("tests/data/")


@pytest.fixture
def pdf_loader():
    return ArxivPDFLoader()


def test_download_pdf(pdf_loader):
    # when: PDF 를 다운로드한다
    pdf_path = pdf_loader.download_pdf(TEST_PDF_URL, save_root=TEST_SAVE_PATH)

    # then: 지정한 경로에 파일이 생성되어 있어야 한다
    assert pdf_path.exists()
    assert pdf_path.is_file()
    assert pdf_path.name == "2405.07437.pdf"
    assert pymupdf.open(pdf_path, filetype="pdf").page_count > 0
