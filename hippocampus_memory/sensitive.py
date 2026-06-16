from __future__ import annotations

import re

SENSITIVE_PATTERNS = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"\b\d{15}(\d{2}[0-9Xx])?\b"),
    re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    re.compile(r"\b(?:api[_-]?key|secret|password|token)\s*[:=]\s*\S+", re.IGNORECASE),
]

SENSITIVE_KEYWORDS = ["身份证", "手机号", "家庭住址", "病历", "诊断", "密码", "密钥"]


def is_sensitive_text(text: str) -> bool:
    if any(keyword in text for keyword in SENSITIVE_KEYWORDS):
        return True
    return any(pattern.search(text) for pattern in SENSITIVE_PATTERNS)
