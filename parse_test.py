import fitz
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

PDF_PATH = "chemistry-atoms.pdf"   # <-- point at any PDF you have (an OpenStax chapter is ideal)

pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = False             # <-- kills the 56-minute waste
pipeline_options.do_table_structure = True  # keep tables as real tables, not text

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

# --- current approach: PyMuPDF (exactly what main.py does today) ---
def pymupdf_extract(path):
    doc = fitz.open(path)
    return "".join(page.get_text() for page in doc)

pymupdf_text = pymupdf_extract(PDF_PATH)
open("out_pymupdf.txt", "w").write(pymupdf_text)

# --- new approach: Docling ---
result = converter.convert(PDF_PATH)          # <-- use the configured converter, NOT a fresh one
docling_md = result.document.export_to_markdown()
open("out_docling.md", "w").write(docling_md)

# --- quick comparison ---
print("PyMuPDF chars :", len(pymupdf_text))
print("Docling  chars:", len(docling_md))
print("Docling headings (# lines):", sum(1 for l in docling_md.splitlines() if l.startswith("#")))
print("Docling table rows (| lines):", sum(1 for l in docling_md.splitlines() if l.strip().startswith("|")))
print("\nNow open out_pymupdf.txt and out_docling.md side by side.")