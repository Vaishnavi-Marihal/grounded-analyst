import re

NUMBER = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")

DATE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}:\d{2}\b|"
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|"
    r"Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\.?\s+\d{1,2}(?:,?\s+\d{4})?\b",
    re.IGNORECASE,
)


def extract_numbers(text: str) -> list[str]:
    text = DATE.sub(" ", text)
    return [m.replace(",", "") for m in NUMBER.findall(text)]


def is_grounded(token: str, tool_numbers: list[float]) -> bool:
    parts = token.split(".")
    decimals = len(parts[1]) if len(parts) == 2 else 0
    value = float(token)
    return any(round(t, decimals) == value for t in tool_numbers)


def find_ungrounded(answer: str, tool_numbers: list[float]) -> list[str]:
    return [t for t in extract_numbers(answer) if not is_grounded(t, tool_numbers)]

def collect_numbers(obj) -> list[float]:
    if isinstance(obj, bool):
        return []
    if isinstance(obj, (int, float)):
        return [float(obj)]
    if isinstance(obj, dict):
        return [n for v in obj.values() for n in collect_numbers(v)]
    if isinstance(obj, list):
        return [n for v in obj for n in collect_numbers(v)]
    return []