from __future__ import annotations

from itertools import islice
from typing import Iterable

import numpy as np
from datasets import Audio, load_dataset


def load_librispeech_split(
    split: str,
    subset: str = "clean",
    limit: int | None = None,
    streaming: bool = True,
    cache_dir: str | None = None,
) -> list[dict]:
    """Load a bounded LibriSpeech split from Hugging Face."""
    ds = load_dataset(
        "openslr/librispeech_asr",
        subset,
        split=split,
        streaming=streaming,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    items: Iterable[dict] = ds
    if limit is not None:
        items = islice(items, limit)
    out = []
    for item in items:
        audio = item["audio"]
        text = item.get("text") or item.get("sentence") or item.get("transcript")
        out.append(
            {
                "id": item.get("id") or item.get("file") or str(len(out)),
                "array": np.asarray(audio["array"], dtype=np.float32),
                "sampling_rate": int(audio["sampling_rate"]),
                "text": text,
            }
        )
    return out


def duration_seconds(item: dict) -> float:
    return float(len(item["array"]) / item["sampling_rate"])

