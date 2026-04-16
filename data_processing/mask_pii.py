# ==============================================================================
# PII 个人身份信息脱敏模块
# ==============================================================================
# 功能概述：
#   使用正则表达式检测和遮蔽文本中的个人身份信息（PII），防止敏感信息泄露到训练数据。
#   支持：邮箱(|||EMAIL_ADDRESS|||)、电话(|||PHONE_NUMBER|||)、IPv4(|||IP_ADDRESS|||)
#   每个方法返回 (masked_content, match_count) 元组。
# ==============================================================================
import re


class PIIMasker:
    def __init__(self):
        pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        self.email_pattern = re.compile(pattern=pattern)
        # self.phone_pattern = re.compile(pattern=r"\b\d{3}-\d{3}-\d{4}\b")

        self.phone_pattern = re.compile(
            r"""
            (?:
                \(?\d{3}\)?[-.\s]?   
                \d{3}[-.\s]?        
                \d{4}              
            )
            """,
            re.VERBOSE,
        )

        self.ipv4_pattern = re.compile(
            r"""
            \b
            (?:
                25[0-5]         
                |
                2[0-4][0-9]    
                |
                1[0-9]{2}     
                |
                [1-9]?[0-9]  
            )
            (?:
                \.
                (?:
                    25[0-5]
                    |
                    2[0-4][0-9]
                    |
                    1[0-9]{2}
                    |
                    [1-9]?[0-9]
                )
            ){3}
            \b
            """,
            re.VERBOSE,
        )

    def mask_emails(self, content: str) -> tuple[str, int]:
        """
        Mask emails in the content.
        """

        matchs = self.email_pattern.findall(content)

        masked_content = self.email_pattern.sub("|||EMAIL_ADDRESS|||", content)
        return masked_content, len(matchs)

    def mask_phone_numbers(self, content: str) -> tuple[str, int]:
        matchs = self.phone_pattern.findall(content)
        masked_content = self.phone_pattern.sub("|||PHONE_NUMBER|||", content)
        return masked_content, len(matchs)

    def mask_ipv4(self, content: str) -> tuple[str, int]:
        matchs = self.ipv4_pattern.findall(content)
        masked_content = self.ipv4_pattern.sub("|||IP_ADDRESS|||", content)
        return masked_content, len(matchs)
