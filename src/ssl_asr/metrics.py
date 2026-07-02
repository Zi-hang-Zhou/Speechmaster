from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import jiwer

from .text import chars_for_cer, normalize_librispeech_text


@dataclass
class ASRMetrics:
    wer: float
    cer: float
    n: int
    substitutions: int
    deletions: int
    insertions: int
    hits: int


def compute_asr_metrics(references: Iterable[str], hypotheses: Iterable[str]) -> ASRMetrics:
    refs = [normalize_librispeech_text(x) for x in references]
    hyps = [normalize_librispeech_text(x) for x in hypotheses]
    word_out = jiwer.process_words(refs, hyps)
    cer = jiwer.cer([chars_for_cer(x) for x in refs], [chars_for_cer(x) for x in hyps])
    return ASRMetrics(
        wer=float(word_out.wer),
        cer=float(cer),
        n=len(refs),
        substitutions=int(word_out.substitutions),
        deletions=int(word_out.deletions),
        insertions=int(word_out.insertions),
        hits=int(word_out.hits),
    )

