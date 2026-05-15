DEFAULT_LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]


def normalize_label(label: str) -> str:
    return label.lower().replace(" ", "_").replace("-", "_")
