# ==============================================================================
# HTML 文本提取模块
# ==============================================================================
# 功能概述：
#   从 Common Crawl WARC 网页存档中提取纯文本内容。
#   流程：detect_encoding() 自动检测编码 -> 解码 -> extract_plain_text() 剥离 HTML 标签。
# 依赖：fastwarc（WARC 解析）、resiliparse（HTML 文本提取，Chatnoir 团队开发）
# ==============================================================================
from fastwarc.warc import ArchiveIterator, WarcRecordType
from resiliparse.parse.encoding import detect_encoding
from resiliparse.extract.html2text import extract_plain_text


def extract_text_from_html(html_content: bytes):
    coding = detect_encoding(html_content)
    content = html_content.decode(coding)
    plain_text = extract_plain_text(content)
    return plain_text
