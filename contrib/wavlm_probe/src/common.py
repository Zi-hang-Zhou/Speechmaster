"""Shared utilities: manifest IO, char vocab, LibriSpeech parsing."""
import os
import json
import glob

# Char vocab: CTC blank(0) + space + A-Z + apostrophe
CHARS = [" "] + [chr(c) for c in range(ord("A"), ord("Z") + 1)] + ["'"]
BLANK = 0
CHAR2ID = {c: i + 1 for i, c in enumerate(CHARS)}  # 0 reserved for blank
ID2CHAR = {i + 1: c for i, c in enumerate(CHARS)}
VOCAB_SIZE = len(CHARS) + 1  # + blank


def text_to_ids(text):
    text = text.upper()
    return [CHAR2ID[c] for c in text if c in CHAR2ID]


def ids_to_text(ids):
    return "".join(ID2CHAR.get(i, "") for i in ids)


def parse_librispeech(root):
    """Return list of dicts {uttid, flac, text} for a LibriSpeech split dir."""
    items = []
    for trans in glob.glob(os.path.join(root, "**", "*.trans.txt"), recursive=True):
        d = os.path.dirname(trans)
        with open(trans) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                uttid, text = line.split(" ", 1)
                flac = os.path.join(d, uttid + ".flac")
                items.append({"uttid": uttid, "flac": flac, "text": text})
    items.sort(key=lambda x: x["uttid"])
    return items


def load_manifest(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def save_manifest(items, path):
    with open(path, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
