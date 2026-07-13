"""Use TextImputer to score text with three model families.

Run from the project root with:

    uv run python examples/language/text_callables_usage.py

The first run downloads the Hugging Face models. Install text dependencies with
``uv sync --extra text`` or ``pip install "shapiq[text]"`` if needed.
"""

from __future__ import annotations

import argparse
from itertools import combinations

import numpy as np
import torch
from matplotlib import pyplot as plt
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from shapiq.approximator import KernelSHAP, KernelSHAPIQ
from shapiq.imputer.text.imputer import TextImputer
from shapiq.interaction_values import InteractionValues
from shapiq.plot import bar_plot

INPUT_TEXTS = [
    "The movie was surprisingly good and very entertaining.",
    "The movie was surprisingly something and very entertaining.",
    "The movie was surprisingly bad and very boring.",
]

CAUSAL_LM_INPUT_TEXTS = [
    "The sky is clear and the sun is shining",
    "Dark clouds cover the sky and heavy rain is falling",
    "There is not enough information to predict the weather",
]

ENCODER_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
CAUSAL_LM_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  # openai-community/gpt2 sshleifer/tiny-gpt2
SEQ2SEQ_MODEL_NAME = "google/flan-t5-small"


def select_device() -> str:
    """Select the best available torch device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def print_scores(
    title: str,
    texts: list[str],
    normalized_scores: np.ndarray,
    full_scores: np.ndarray,
    empty_scores: np.ndarray,
) -> None:
    """Pretty-print normalized, full, and empty scores per input text."""
    print(f"\n{title}")
    print("-" * len(title))
    print(f"{'normalized':>12}  {'full':>12}  {'empty':>12}  text")
    print("-" * 72)
    for text, normalized, full, empty in zip(
        texts,
        normalized_scores,
        full_scores,
        empty_scores,
        strict=True,
    ):
        print(f"{float(normalized):>+12.4f}  {float(full):>12.4f}  {float(empty):>12.4f}  {text}")


def print_score_delta_explanation(
    texts: list[str],
    scores_by_model_type: dict[str, np.ndarray],
    empty_scores_by_model_type: dict[str, np.ndarray],
) -> None:
    """Print the table represented by the normalized-score plot."""
    print("\nHow to read the plot")
    print("--------------------")
    print("Each score is computed through TextImputer.__call__(grand_coalition).")
    print("That means shapiq applies Game normalization:")
    print("    normalized_score = full_prediction - empty_prediction")
    print("Here, empty_prediction is the score when all word players are removed.")
    print("Positive values mean the original text scores above its empty baseline.")
    print("Negative values mean the original text scores below its empty baseline.\n")

    for model_type, scores in scores_by_model_type.items():
        empty_scores = empty_scores_by_model_type[model_type]
        print(f"{model_type}")
        print(f"{'input':>5}  {'normalized':>12}  {'empty':>12}  meaning")
        print("-" * 72)

        for idx, (text, score, empty_score) in enumerate(
            zip(texts, scores, empty_scores, strict=True)
        ):
            normalized_score = float(score)
            if normalized_score > 0:
                meaning = "full text scores above empty baseline"
            elif normalized_score < 0:
                meaning = "full text scores below empty baseline"
            else:
                meaning = "same as empty baseline"

            print(f"{idx:>5}  {normalized_score:>+12.4f}  {float(empty_score):>12.4f}  {meaning}")
            print(f"       {text}")
        print()


def score_with_normalized_imputer(
    *,
    model,
    tokenizer,
    texts: list[str],
    device: str,
    model_type: str,
    target_label: str = "positive",
    prompt_template: str = "{text}",
    class_idx: int = 1,
    output_type: str = "logit",
    normalize_target_logprob: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score texts through built-in word-level TextImputer game calls."""
    normalized_scores = []
    full_scores = []
    empty_scores = []

    for text in texts:
        imputer = TextImputer(
            model=model,
            tokenizer=tokenizer,
            text=text,
            batch_size=16,
            device=device,
            class_idx=class_idx,
            output_type=output_type,
            target_label=target_label,
            prompt_template=prompt_template,
            normalize_target_logprob=normalize_target_logprob,
            model_type=model_type,
            player_level="word",
            perturbation_type="removal",
        )
        normalized_score = imputer(imputer.grand_coalition)[0]  # Game.__call__() normalized value
        normalized_scores.append(normalized_score)
        full_scores.append(imputer.full_prediction)
        empty_scores.append(imputer.empty_prediction)

    return (
        np.asarray(normalized_scores, dtype=float),
        np.asarray(full_scores, dtype=float),
        np.asarray(empty_scores, dtype=float),
    )


def make_word_level_imputer(
    *,
    model,
    tokenizer,
    text: str,
    device: str,
    model_type: str,
    target_label: str = "positive",
    prompt_template: str = "{text}",
    class_idx: int = 1,
    output_type: str = "logit",
    normalize_target_logprob: bool = True,
) -> TextImputer:
    """Create TextImputer through imputer.py's built-in word/removal strategies."""
    return TextImputer(
        model=model,
        tokenizer=tokenizer,
        text=text,
        batch_size=16,
        device=device,
        class_idx=class_idx,
        output_type=output_type,
        target_label=target_label,
        prompt_template=prompt_template,
        normalize_target_logprob=normalize_target_logprob,
        model_type=model_type,
        player_level="word",
        perturbation_type="removal",
    )


def print_word_level_explanations(
    *,
    label: str,
    imputer: TextImputer,
    sv_budget: int,
    sii_budget: int,
    show_plots: bool,
) -> None:
    """Compute and print word Shapley values and pairwise interactions."""
    words = imputer.player_strategy.get_players()
    print(f"\nWord-level explanation for {label}")
    print("=" * (27 + len(label)))
    print(f"Explained text: {imputer.text}")
    print(f"Players ({imputer.n_players} words): {words}")
    print(f"Empty prediction: {imputer.empty_prediction:.4f}")
    print(f"Full prediction : {imputer.full_prediction:.4f}")
    print(f"Normalized full : {float(imputer(imputer.grand_coalition)[0]):+.4f}")

    # ------------------------------------------------------------------
    # 计算一阶 Shapley Value
    # ------------------------------------------------------------------

    sv = KernelSHAP(n=imputer.n_players, random_state=42).approximate(
        budget=sv_budget,
        game=imputer,
    )
    print("\nFirst-order Shapley values")
    print(f"{'idx':>3}  {'word':<18}  {'SV':>12}")
    print("-" * 40)
    for idx, word in enumerate(words):
        print(f"{idx:>3}  {word:<18}  {sv[(idx,)]:>+12.4f}")

    # ------------------------------------------------------------------
    # 计算二阶 Shapley Interaction
    # ------------------------------------------------------------------

    sii = KernelSHAPIQ(
        n=imputer.n_players,
        index="k-SII",
        max_order=2,
        random_state=42,
    ).approximate(
        budget=sii_budget,
        game=imputer,
    )
    print("\nSecond-order Shapley interactions (k-SII)")
    print(f"{'pair':<31}  {'interaction':>12}")
    print("-" * 47)

    # 枚举所有不重复的单词对

    for i, j in combinations(range(imputer.n_players), 2):
        print(f"{words[i]} x {words[j]:<20}  {sii[(i, j)]:>+12.4f}")

    if show_plots:
        sv.plot_sentence(words=words)
        sii.plot_network(feature_names=words)


def run_encoder_classifier(
    texts: list[str],
    device: str,
    *,
    explain_text: str,
    sv_budget: int,
    sii_budget: int,
    show_explanation_plots: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use TextImputer with an encoder classifier for sentiment probability."""
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(ENCODER_MODEL_NAME)
    model.to(device).eval()

    scores, full_scores, empty_scores = score_with_normalized_imputer(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        device=device,
        model_type="encoder_classifier",
        class_idx=1,
        output_type="probability",
    )
    print_scores(
        "TextImputer encoder_classifier: normalized P(positive sentiment)",
        texts,
        scores,
        full_scores,
        empty_scores,
    )
    # 为指定文本创建单词级 TextImputer
    word_imputer = make_word_level_imputer(
        model=model,
        tokenizer=tokenizer,
        text=explain_text,
        device=device,
        model_type="encoder_classifier",
        class_idx=1,
        output_type="probability",
    )
    # 计算单词级 Shapley Value 和单词对交互
    print_word_level_explanations(
        label="encoder_classifier",
        imputer=word_imputer,
        sv_budget=sv_budget,
        sii_budget=sii_budget,
        show_plots=show_explanation_plots,
    )
    return scores, full_scores, empty_scores


def run_causal_lm_sentiment(
    texts: list[str],
    device: str,
    *,
    explain_text: str,
    sv_budget: int,
    sii_budget: int,
    show_explanation_plots: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use TextImputer with a causal LM for target-label log-probability."""
    tokenizer = AutoTokenizer.from_pretrained(CAUSAL_LM_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(CAUSAL_LM_MODEL_NAME)
    model.to(device).eval()

    scores, full_scores, empty_scores = score_with_normalized_imputer(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        device=device,
        model_type="causal_lm",
        target_label=" positive",
        prompt_template="Review: {text}\nSentiment:",
    )
    print_scores(
        "TextImputer causal_lm: normalized log P(' positive' | prompt)",
        texts,
        scores,
        full_scores,
        empty_scores,
    )
    word_imputer = make_word_level_imputer(
        model=model,
        tokenizer=tokenizer,
        text=explain_text,
        device=device,
        model_type="causal_lm",
        target_label=" positive",
        prompt_template="Review: {text}\nSentiment:",
    )
    print_word_level_explanations(
        label="causal_lm",
        imputer=word_imputer,
        sv_budget=sv_budget,
        sii_budget=sii_budget,
        show_plots=show_explanation_plots,
    )
    return scores, full_scores, empty_scores


def run_causal_lm_weather(
    texts: list[str],
    device: str,
    *,
    explain_text: str,
    sv_budget: int,
    sii_budget: int,
    show_explanation_plots: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use a causal LM to score a natural continuation.

    The model receives a weather description followed by:

        ", so the weather is"

    It then scores the natural continuation:

        " good"

    This is closer to the original next-token prediction objective of a
    causal language model than instruction-style sentiment classification.
    """
    tokenizer = AutoTokenizer.from_pretrained(CAUSAL_LM_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(CAUSAL_LM_MODEL_NAME)
    model.to(device).eval()

    # GPT-2 does not define a dedicated padding token.
    # Use EOS as padding when TextImputer batches perturbed inputs.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.pad_token_id = tokenizer.pad_token_id

    scores, full_scores, empty_scores = score_with_normalized_imputer(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        device=device,
        model_type="causal_lm",
        # The leading whitespace is important for GPT-2 tokenization:
        # this represents the word as a continuation of the prompt.
        target_label=" good",
        # Keep the task structure fixed while TextImputer perturbs only
        # the weather-description words represented by {text}.
        prompt_template="{text}, so the weather is",
    )

    print_scores(
        "TextImputer causal_lm: normalized log P(' good' | weather context)",
        texts,
        scores,
        full_scores,
        empty_scores,
    )

    word_imputer = make_word_level_imputer(
        model=model,
        tokenizer=tokenizer,
        text=explain_text,
        device=device,
        model_type="causal_lm",
        target_label=" good",
        prompt_template="{text}, so the weather is",
    )

    print_word_level_explanations(
        label="causal_lm",
        imputer=word_imputer,
        sv_budget=sv_budget,
        sii_budget=sii_budget,
        show_plots=show_explanation_plots,
    )

    return scores, full_scores, empty_scores


def run_seq2seq(
    texts: list[str],
    device: str,
    *,
    explain_text: str,
    sv_budget: int,
    sii_budget: int,
    show_explanation_plots: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use TextImputer with an encoder-decoder model for target-label log-probability."""
    tokenizer = AutoTokenizer.from_pretrained(SEQ2SEQ_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(SEQ2SEQ_MODEL_NAME)
    model.to(device).eval()

    scores, full_scores, empty_scores = score_with_normalized_imputer(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        device=device,
        model_type="seq2seq",
        target_label="positive",
        prompt_template="sst2 sentence: {text}",
        normalize_target_logprob=True,
    )
    print_scores(
        "TextImputer seq2seq: normalized mean log P('positive' | prompt)",
        texts,
        scores,
        full_scores,
        empty_scores,
    )
    word_imputer = make_word_level_imputer(
        model=model,
        tokenizer=tokenizer,
        text=explain_text,
        device=device,
        model_type="seq2seq",
        target_label="positive",
        prompt_template="sst2 sentence: {text}",
        normalize_target_logprob=True,
    )
    print_word_level_explanations(
        label="seq2seq",
        imputer=word_imputer,
        sv_budget=sv_budget,
        sii_budget=sii_budget,
        show_plots=show_explanation_plots,
    )
    return scores, full_scores, empty_scores


def show_score_delta_plot(
    texts: list[str],
    scores_by_model_type: dict[str, np.ndarray],
) -> None:
    """Show a shapiq bar plot of normalized TextImputer scores.

    This is a diagnostic visualization of normalized full-text scores, not a
    token-level Shapley explanation. Each bar is ``full_prediction - empty_prediction``.
    """
    feature_names = [f"input {idx}: {text}" for idx, text in enumerate(texts)]
    normalized_values = []

    for model_type, scores in scores_by_model_type.items():
        interaction_values = InteractionValues.from_first_order_array(
            np.asarray(scores, dtype=float),
            index="SV",
            baseline_value=0.0,
        )
        interaction_values.estimation_budget = model_type
        normalized_values.append(interaction_values)

    axis = bar_plot(
        normalized_values,
        feature_names=feature_names,
        show=False,
        abbreviate=True,
        max_display=len(texts),
        global_plot=False,
        plot_base_value=False,
    )
    axis.set_title("Normalized TextImputer scores")
    axis.set_xlabel("full_prediction - empty_prediction")
    handles, _ = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles, list(scores_by_model_type), title="TextImputer model_type")
    axis.figure.tight_layout()
    plt.show(block=True)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run shapiq TextImputer scoring on example input texts.",
    )
    parser.add_argument(
        "--callable",
        choices=["all", "encoder", "causal", "seq2seq"],
        default="all",
        help="Which TextImputer model_type example to run.",
    )
    parser.add_argument(
        "--text",
        action="append",
        dest="texts",
        help=(
            "Input text to score. Can be passed multiple times. "
            "Defaults to three sentiment variants."
        ),
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not show the score-delta plot after running the TextImputer example.",
    )
    parser.add_argument(
        "--explain-text-index",
        type=int,
        default=0,
        help="Which input text to explain at word level.",
    )
    parser.add_argument(
        "--sv-budget",
        type=int,
        default=64,
        help="Approximation budget for word-level Shapley values.",
    )
    parser.add_argument(
        "--sii-budget",
        type=int,
        default=128,
        help="Approximation budget for second-order Shapley interactions.",
    )
    parser.add_argument(
        "--no-explanation-plots",
        action="store_true",
        help="Print word-level values but do not show sentence/network explanation plots.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the selected TextImputer example."""
    args = parse_args()
    device = select_device()

    print(f"Using device: {device}")

    runners_sentiment = {
        "encoder": run_encoder_classifier,
        "causal": run_causal_lm_sentiment,
        "seq2seq": run_seq2seq,
    }

    scores_by_model_type: dict[str, np.ndarray] = {}
    empty_scores_by_model_type: dict[str, np.ndarray] = {}

    if args.callable == "all":
        # Use the shared sentiment texts when all model families run together.
        texts = args.texts or INPUT_TEXTS

        if not 0 <= args.explain_text_index < len(texts):
            msg = f"--explain-text-index must be between 0 and {len(texts) - 1}."
            raise ValueError(msg)

        explain_text = texts[args.explain_text_index]

        print(f"Word-level explanation target: input {args.explain_text_index}")

        for model_type, runner in runners_sentiment.items():
            scores, _, empty_scores = runner(
                texts,
                device,
                explain_text=explain_text,
                sv_budget=args.sv_budget,
                sii_budget=args.sii_budget,
                show_explanation_plots=(not args.no_plot and not args.no_explanation_plots),
            )

            scores_by_model_type[model_type] = scores
            empty_scores_by_model_type[model_type] = empty_scores

    elif args.callable == "causal":
        # Use continuation-style weather texts for the causal LM example.
        texts = args.texts or CAUSAL_LM_INPUT_TEXTS

        if not 0 <= args.explain_text_index < len(texts):
            msg = f"--explain-text-index must be between 0 and {len(texts) - 1}."
            raise ValueError(msg)

        explain_text = texts[args.explain_text_index]

        print(f"Word-level explanation target: input {args.explain_text_index}")

        scores, _, empty_scores = run_causal_lm_weather(
            texts,
            device,
            explain_text=explain_text,
            sv_budget=args.sv_budget,
            sii_budget=args.sii_budget,
            show_explanation_plots=(not args.no_plot and not args.no_explanation_plots),
        )

        scores_by_model_type["causal"] = scores
        empty_scores_by_model_type["causal"] = empty_scores

    else:
        # Run either the encoder or seq2seq sentiment example on sentiment texts.
        texts = args.texts or INPUT_TEXTS

        if not 0 <= args.explain_text_index < len(texts):
            msg = f"--explain-text-index must be between 0 and {len(texts) - 1}."
            raise ValueError(msg)

        explain_text = texts[args.explain_text_index]

        print(f"Word-level explanation target: input {args.explain_text_index}")

        scores, _, empty_scores = runners_sentiment[args.callable](
            texts,
            device,
            explain_text=explain_text,
            sv_budget=args.sv_budget,
            sii_budget=args.sii_budget,
            show_explanation_plots=(not args.no_plot and not args.no_explanation_plots),
        )

        scores_by_model_type[args.callable] = scores
        empty_scores_by_model_type[args.callable] = empty_scores

    print_score_delta_explanation(
        texts,
        scores_by_model_type,
        empty_scores_by_model_type,
    )

    if not args.no_plot:
        show_score_delta_plot(
            texts,
            scores_by_model_type,
        )


if __name__ == "__main__":
    main()
