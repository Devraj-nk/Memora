# Turns a raw source (file path or URL) into plain text + metadata
# (source, mime type, timestamps). Plain text/markdown/code first;
# PDF/docx/OCR come later via the optional `parsing` dependency group.


def parse(source_path: str) -> str:
    raise NotImplementedError
