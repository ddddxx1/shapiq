"""Minimal TextImputer example for an encoder sentiment classifier.

Run from the project root with:

    uv run python examples/language/imputer_example.py

The first run downloads the Hugging Face model. The word-level player strategy
uses NLTK tokenization; if needed, install the resource once with:

    uv run python -m nltk.downloader punkt_tab
"""

from __future__ import annotations

from itertools import combinations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from shapiq.approximator import KernelSHAP, KernelSHAPIQ
from shapiq.imputer.text.imputer import TextImputer

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
TEXT = "The movie is not bad."


def select_device() -> str:
    """Select the best available torch device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    """Create a word-level encoder TextImputer and explain its score."""
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
    words = imputer.player_strategy.get_players()
    exact_budget = 2**imputer.n_players

    print(f"Input text      : {TEXT}")
    print(f"Word players    : {words}")
    print(f"Final score     : {normalized_score:.4f}")
    print("\nFinal score = full prediction - empty prediction")

    shapley_values = KernelSHAP(n=imputer.n_players, random_state=42).approximate(
        budget=exact_budget,
        game=imputer,
    )

    print("\nFirst-order Shapley values")
    print(f"{'idx':>3}  {'word':<18}  {'value':>12}")
    print("-" * 40)
    for idx, word in enumerate(words):
        print(f"{idx:>3}  {word:<18}  {shapley_values[(idx,)]:>+12.4f}")

    shapley_interactions = KernelSHAPIQ(
        n=imputer.n_players,
        index="k-SII",
        max_order=2,
        random_state=42,
    ).approximate(
        budget=exact_budget,
        game=imputer,
    )

    print("\nSecond-order Shapley interactions (k-SII)")
    print(f"{'pair':<31}  {'value':>12}")
    print("-" * 47)
    for i, j in combinations(range(imputer.n_players), 2):
        pair_name = f"{words[i]} x {words[j]}"
        print(f"{pair_name:<31}  {shapley_interactions[(i, j)]:>+12.4f}")

    print("\nShowing force plot for first-order Shapley values...")
    shapley_values.plot_force(
        feature_names=words, show=True, abbreviate=False, contribution_threshold=0.01
    )
    print("Showing force plot for second-order Shapley interactions...")
    shapley_interactions.plot_force(feature_names=words, show=True)


if __name__ == "__main__":
    main()
