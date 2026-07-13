"""Minimal TextImputer example for an encoder sentiment classifier.

Run from the project root with:

    uv run python examples/language/imputer_example.py

The first run downloads the Hugging Face model. The word-level player strategy
uses NLTK tokenization; if needed, install the resource once with:

    uv run python -m nltk.downloader punkt_tab
"""

from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from shapiq.imputer.text.imputer import TextImputer

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
TEXT = "The movie was surprisingly good and very entertaining."


def select_device() -> str:
    """Select the best available torch device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    """Create a word-level encoder TextImputer and print its final score."""
    device = select_device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device).eval()

    imputer = TextImputer(
        model=model,
        tokenizer=tokenizer,
        text=TEXT,
        model_type="encoder_classifier",
        player_level="word",
        perturbation_type="removal",
        class_idx=1,
        output_type="probability",
        device=device,
    )

    normalized_score = float(imputer(imputer.grand_coalition)[0])

    print(f"Input text      : {TEXT}")
    print(f"Word players    : {imputer.player_strategy.get_players()}")
    print(f"Full prediction : {imputer.full_prediction:.4f}")
    print(f"Empty prediction: {imputer.empty_prediction:.4f}")
    print(f"Final score     : {normalized_score:.4f}")
    print("\nFinal score = full prediction - empty prediction")


if __name__ == "__main__":
    main()
