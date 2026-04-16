# ==============================================================================
# data_processing 包初始化模块
# ==============================================================================
# 功能概述：
#   预训练数据处理管线工具包，涵盖从原始网页数据到高质量训练语料的完整流程：
#   - html_process             : 从 WARC 网页存档中提取纯文本
#   - language_identification  : 基于 fastText 的语种识别（176 种语言）
#   - mask_pii                 : 个人身份信息（PII）脱敏（邮箱、电话、IP 地址）
#   - harmful_detect           : NSFW / 仇恨言论检测（fastText 分类器）
#   - quality_filter           : 基于规则的文本质量过滤（词数、词长、字母比、省略号比）
#   - quality_classfier        : 基于 fastText 的文本质量分类器（高质量/低质量）
#   - deduplicate              : 数据去重（精确行去重 + MinHash 近似去重）
# ==============================================================================
from .html_process import *
from .language_identification import *
from .mask_pii import *
from .harmful_detect import *
from .quality_filter import *
from .quality_classfier import *
from .deduplicate import *
