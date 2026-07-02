import re
import string


_punct_table = str.maketrans("", "", string.punctuation)


def normalize_librispeech_text(text: str) -> str:
    """Normalize text for LibriSpeech-style WER/CER evaluation."""
    text = text.upper()
    text = text.translate(_punct_table)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def chars_for_cer(text: str) -> str:
    """CER over non-space characters avoids whitespace tokenization artifacts."""
    return normalize_librispeech_text(text).replace(" ", "")

