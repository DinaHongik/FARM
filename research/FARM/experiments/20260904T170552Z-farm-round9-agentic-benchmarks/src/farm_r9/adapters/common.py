from __future__ import annotations

import re
import unicodedata


_TOKENS = re.compile(r"[a-z0-9]+")


def normalize_label(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).replace("_", " ").casefold()
    return " ".join(_TOKENS.findall(value))


def split_fields(value: str) -> list[str]:
    return [field.strip() for field in value.split("###") if field.strip()]
