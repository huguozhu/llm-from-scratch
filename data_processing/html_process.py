from fastwarc.warc import ArchiveIterator, WarcRecordType
from resiliparse.parse.encoding import detect_encoding
from resiliparse.extract.html2text import extract_plain_text


def extract_text_from_html(html_content: bytes):
    coding = detect_encoding(html_content)
    content = html_content.decode(coding)
    plain_text = extract_plain_text(content)
    return plain_text
