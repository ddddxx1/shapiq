# ruff: noqa: T201
"""Multilingual Explanation Comparison Demo.

This demo investigates whether SHAPIQ explanations remain consistent across
semantically equivalent texts (translations or paraphrases).

The demo supports two comparison modes:

1. User-defined Comparison
2. Model-generated Comparison

The explanation pipeline always relies on the public SHAPIQ API:

TextImputer
    ↓
SHAPIQ
    ↓
InteractionValues
    ↓
Force Plot

The demo visualizes explanations only.

It does NOT automatically compare or evaluate explanation consistency.
"""

from __future__ import annotations

import sys 
import os  # new
src_path = os.path.join(os.path.dirname(__file__), '..', '..', 'src')
sys.path.insert(0, src_path)

import types
fake_cext = types.ModuleType('shapiq.tree.conversion.cext')
fake_cext.create_edge_tree_arrays = lambda *args, **kwargs: None
sys.modules['shapiq.tree.conversion.cext'] = fake_cext

import shapiq.tree.conversion.cext as cext
cext.create_edge_tree_arrays = lambda *args, **kwargs: None
import json  # new

from datetime import datetime  # new

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from shapiq.approximator import SHAPIQ
from shapiq.imputer.text_imputer import TextImputer
from shapiq.plot import force_plot

# ============================================================
# Configuration
# ============================================================

PARAPHRASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

TRANSLATION_MODEL_EN_DE = "Helsinki-NLP/opus-mt-en-de"
TRANSLATION_MODEL_DE_EN = "Helsinki-NLP/opus-mt-de-en"

MLM_MODEL = "bert-base-uncased"


EXPLANATION_MODEL = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"

DEBUG = False  # change false

MLM_NUM_SAMPLES = 3 if DEBUG else 10

DEFAULT_PLAYER_LEVEL = "word"

DEFAULT_INTERACTION_INDEX = "k-SII"

DEFAULT_MAX_ORDER = 2

EXPLANATION_BUDGET = 2048

SUPPORTED_INTERACTION_INDICES = [
    "k-SII",
    "STII",
]

"""DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)"""

DEVICE = "cpu"

# Load explanation model (shared across all explanations)
tokenizer = AutoTokenizer.from_pretrained(EXPLANATION_MODEL)

model = AutoModelForSequenceClassification.from_pretrained(EXPLANATION_MODEL).to(DEVICE)

model.eval()

# ============================================================
# CLI Helper Functions
# ============================================================


def ask_choice(
    prompt: str,
    valid_choices: list[str],
) -> str:
    """Ask the user to choose one option."""
    while True:
        value = input(prompt).strip()

        if value in valid_choices:
            return value

        print("\nInvalid input. Please try again.\n")


def ask_integer(
    prompt: str,
    minimum: int,
    maximum: int,
    default: int,
) -> int:
    """Read an integer within a given range."""
    while True:
        value = input(prompt).strip()

        if value == "":
            return default
        try:
            number = int(value)
            if minimum <= number <= maximum:
                return number
        except ValueError:
            pass
        print(f"Please enter an integer between {minimum} and {maximum}.")


# ============================================================
# Menu
# ============================================================


def print_header() -> None:
    """Print the demo header."""
    print()

    print("=" * 60)
    print("SHAPIQ Multilingual Explanation Demo")
    print("=" * 60)

    print()

    print("Research Question")

    print("Are SHAPIQ explanations stable across semantically equivalent texts?")

    print()


# ============================================================
# Text Generation
# ============================================================


def translate_sentence(
    sentence: str,
    source_language: str,
    target_language: str,
) -> str:
    """Translate a sentence using OPUS-MT."""
    if source_language == "English" and target_language == "German":
        model_name = TRANSLATION_MODEL_EN_DE

    elif source_language == "German" and target_language == "English":
        model_name = TRANSLATION_MODEL_DE_EN

    else:
        msg = f"Unsupported translation direction: {source_language} -> {target_language}"
        raise ValueError(msg)

    print(f"\nLoading translation model: {model_name}")

    translation_tokenizer = AutoTokenizer.from_pretrained(model_name)

    translation_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)

    translation_model.eval()

    inputs = translation_tokenizer(
        sentence,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        outputs = translation_model.generate(
            **inputs,
            max_new_tokens=64,
        )

    generated = translation_tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    ).strip()

    del translation_model
    del translation_tokenizer

    return generated


def paraphrase_sentence(
    sentence: str,
    style: str,
) -> str:
    """Generate a paraphrase with a selected rewriting strength."""
    print(f"\nLoading paraphrase model: {PARAPHRASE_MODEL}")

    paraphrase_tokenizer = AutoTokenizer.from_pretrained(PARAPHRASE_MODEL)

    paraphrase_model = AutoModelForCausalLM.from_pretrained(PARAPHRASE_MODEL).to(DEVICE)

    paraphrase_model.eval()

    instructions = {
        "1": (
            "Paraphrase the sentence with minimal changes. "
            "Replace only a few words with synonyms. "
            "Preserve the original sentence structure and all meaning."
        ),
        "2": (
            "Paraphrase the sentence using clearly different wording "
            "and a noticeably different sentence structure. "
            "Preserve exactly the same meaning and all important information."
        ),
        "3": (
            "Paraphrase the sentence by changing its grammatical structure. "
            "Keep every original entity, fact, sentiment, and relationship unchanged. "
            "Use only information explicitly present in the original sentence. "
            "Do not infer, generalize, or introduce any new details. "
            "You must produce a visibly different sentence structure. "
            "Do not copy the original sentence verbatim."
        ),
    }

    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise paraphrasing assistant. "
                "Preserve the exact meaning and factual content of the original sentence. "
                "Never introduce new people, objects, events, or facts. "
                "Return only one paraphrased sentence. "
                "Do not explain your answer. "
                "Do not add quotation marks."
            ),
        },
        {
            "role": "user",
            "content": (f"{instructions[style]}\n\nSentence:\n{sentence}"),
        },
    ]

    text = paraphrase_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = paraphrase_tokenizer(
        text,
        return_tensors="pt",
    ).to(DEVICE)

    generation_settings = {
        "1": {
            "max_new_tokens": 128,
            "do_sample": False,
        },
        "2": {
            "max_new_tokens": 128,
            "do_sample": True,
            "top_p": 0.85,
            "temperature": 1.1,
        },
        "3": {
            "max_new_tokens": 128,
            "do_sample": True,
            "top_p": 0.9,
            "temperature": 0.5,
            "repetition_penalty": 1.1,
        },
    }

    with torch.no_grad():
        outputs = paraphrase_model.generate(
            **inputs,
            **generation_settings[style],
        )

    generated_tokens = outputs[
        0,
        inputs["input_ids"].shape[1] :,
    ]

    generated = paraphrase_tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    del paraphrase_model
    del paraphrase_tokenizer

    return generated


# ============================================================
# Comparison Modes
# ============================================================


def user_defined_comparison() -> tuple[str, str]:
    """Read two user-defined sentences for comparison."""
    print("\nSentence 1")

    sentence_1 = input("> ").strip()

    print("\nSentence 2")

    sentence_2 = input("> ").strip()

    return sentence_1, sentence_2


def model_generated_translation(
    sentence: str,
) -> str:
    """Generate a translated sentence for comparison."""
    print()

    print("Source Language")

    print("1. English")
    print("2. German")

    language = ask_choice(
        "\nChoose: ",
        ["1", "2"],
    )

    if language == "1":
        generated = translate_sentence(
            sentence,
            "English",
            "German",
        )

    else:
        generated = translate_sentence(
            sentence,
            "German",
            "English",
        )

    return sentence, generated


def paraphrase_mode(
    sentence: str,
) -> tuple[str, str]:
    """Select paraphrase strength and generate Sentence 2."""
    print()
    print("Paraphrase Style")
    print("1. Conservative")
    print("2. Moderate")
    print("3. Structural")

    style = ask_choice(
        "\nChoose: ",
        ["1", "2", "3"],
    )

    generated = paraphrase_sentence(
        sentence,
        style,
    )

    return sentence, generated


def model_generated_comparison() -> tuple[str, str]:
    """Generate a translated or paraphrased sentence for comparison."""
    print()
    print("Original Sentence")
    sentence = input("> ").strip()
    print()
    print("Generation Mode")
    print("1. Translation")
    print("2. Paraphrase")
    mode = ask_choice(
        "\nChoose: ",
        ["1", "2"],
    )
    if mode == "1":
        return model_generated_translation(
            sentence,
        )
    return paraphrase_mode(
        sentence,
    )


def choose_comparison_mode() -> tuple[str, str]:
    """Choose the sentence comparison mode."""
    print()
    print()
    print("Step 1 / 5\n")
    print("Comparison Mode")
    print("1. User-defined Comparison")
    print("2. Model-generated Comparison")
    mode = ask_choice(
        "\nChoose: ",
        ["1", "2"],
    )
    if mode == "1":
        return user_defined_comparison()
    return model_generated_comparison()


# ============================================================
# Explanation Settings
# ============================================================


def choose_explanation_settings() -> tuple[str, int]:
    """Choose the interaction index and maximum interaction order."""
    print()
    print("Step 2 / 5\n")
    print("Interaction Index")
    print("1. k-SII")
    print("2. STII")
    index = ask_choice(
        "\nChoose: ",
        ["1", "2"],
    )
    interaction_index = SUPPORTED_INTERACTION_INDICES[int(index) - 1]
    print()
    max_order = ask_integer(
        f"Maximum Order [Default {DEFAULT_MAX_ORDER}]: ",
        minimum=1,
        maximum=5,
        default=DEFAULT_MAX_ORDER,
    )
    return (
        interaction_index,
        max_order,
    )


# ========================================================================================
# Build TextImputer:Construct a TextImputer using the public API implemented in this PR.
# ========================================================================================


def build_text_imputer(
    text: str,
    perturbation_type: str,
) -> TextImputer:
    kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "text": text,
        "player_level": DEFAULT_PLAYER_LEVEL,
        "perturbation_type": perturbation_type,
        "model_type": "encoder_classifier",
        "class_idx": 0,
        "output_type": "probability",  # probability改成logit
    }

    # Only MLM infilling requires additional arguments.
    if perturbation_type == "mlm_infilling":
        kwargs["mlm_model_name"] = MLM_MODEL
        kwargs["mlm_num_samples"] = MLM_NUM_SAMPLES

    return TextImputer(**kwargs)


# =================================================================
# Explain One Sentence:Compute interaction values for one sentence.
# =================================================================


def explain_sentence(
    text: str,
    perturbation_type: str,
    interaction_index: str,
    max_order: int,
):
    print()
    print("=" * 80)
    print(f"Perturbation : {perturbation_type}")
    print("=" * 80)

    imputer = build_text_imputer(
        text=text,
        perturbation_type=perturbation_type,
    )

    # --------------------------------------------------------
    # SHAPIQ approximator
    # Uses the official shapiq API.
    # --------------------------------------------------------

    approximator = SHAPIQ(
        n=imputer.n_features,
        index=interaction_index,
        max_order=max_order,
    )

    interaction_values = approximator.approximate(
        budget=EXPLANATION_BUDGET,
        game=imputer,
    )

    feature_names = imputer.player_strategy.get_players()
    return interaction_values, feature_names


# ============================================================
# Explain with Both Perturbation Strategies
# ============================================================


def explain_both_perturbations(
    text: str,
    interaction_index: str,
    max_order: int,
):
    """Generate explanations using both perturbation strategies."""
    removal_values = explain_sentence(
        text=text,
        perturbation_type="removal",
        interaction_index=interaction_index,
        max_order=max_order,
    )

    mlm_values = explain_sentence(
        text=text,
        perturbation_type="mlm_infilling",
        interaction_index=interaction_index,
        max_order=max_order,
    )

    return {
        "removal": removal_values,
        "mlm_infilling": mlm_values,
    }


# ============================================================
# Full Explanation Pipeline
# ============================================================


def generate_explanations(
    sentence_1: str,
    sentence_2: str,
    interaction_index: str,
    max_order: int,
):
    """Explain both translated sentences."""
    print("\nGenerating interaction explanations...\n")

    results = {
        "sentence_1": explain_both_perturbations(
            text=sentence_1,
            interaction_index=interaction_index,
            max_order=max_order,
        ),
        "sentence_2": explain_both_perturbations(
            text=sentence_2,
            interaction_index=interaction_index,
            max_order=max_order,
        ),
    }

    return results


# ============================================================
# Visualization
# ============================================================


def visualize_results(results, sentence_1, sentence_2, interaction_index, max_order):
    """Print and visualize the computed interaction values."""
    save_results_to_file(results, sentence_1, sentence_2, interaction_index, max_order)
    save_force_plots(results, sentence_1, sentence_2)

    for sentence_name, sentence_results in results.items():
        print("\n" + "=" * 80)
        print(sentence_name.replace("_", " ").title())
        print("=" * 80)

        for perturbation_name, result in sentence_results.items():
            interaction_values, feature_names = result

            print(f"\nPerturbation: {perturbation_name}")

            # --------------------------------------------------------------
            # print first-order interaction values
            # --------------------------------------------------------------

            print("\nFirst-order Interaction Values")
            print("-" * 80)
            for player_idx, feature_name in enumerate(feature_names):
                value = interaction_values[(player_idx,)]
                print(f"{feature_name:<25} {value:+.6f}")

            # --------------------------------------------------------------
            # print second-order interaction values
            # --------------------------------------------------------------

            print("\nSecond-order Interaction Values")
            print("-" * 80)
            for player_idx_1 in range(len(feature_names)):
                for player_idx_2 in range(
                    player_idx_1 + 1,
                    len(feature_names),
                ):
                    value = interaction_values[(player_idx_1, player_idx_2)]

                    interaction_name = (
                        f"{feature_names[player_idx_1]} x {feature_names[player_idx_2]}"
                    )

                    print(f"{interaction_name:<40} {value:+.6f}")

            force_plot(
                interaction_values,
                feature_names=feature_names,
                show=False,
            )


# ============================================================
# Save Results to file
# ============================================================


def save_results_to_file(results, sentence_1, sentence_2, interaction_index, max_order):
    """Save the explanation results to a JSON file."""
    save_dir = "demo/multilingual/Multilingual_Expanation_Demo_Results"
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    data = {
        "timestamp": timestamp,
        "sentence_1": sentence_1,
        "sentence_2": sentence_2,
        "interaction_index": interaction_index,
        "max_order": max_order,
        "results": {},
    }

    for sentence_name, sentence_results in results.items():
        data["results"][sentence_name] = {}
        for perturbation_name, (interaction_values, feature_names) in sentence_results.items():
            first_order = {}
            second_order = {}

            for player_idx, feature_name in enumerate(feature_names):
                first_order[feature_name] = float(interaction_values[(player_idx,)])

            for player_idx_1 in range(len(feature_names)):
                for player_idx_2 in range(player_idx_1 + 1, len(feature_names)):
                    key = f"{feature_names[player_idx_1]} x {feature_names[player_idx_2]}"
                    second_order[key] = float(interaction_values[(player_idx_1, player_idx_2)])

            data["results"][sentence_name][perturbation_name] = {
                "first_order": first_order,
                "second_order": second_order,
                "feature_names": feature_names,
            }

    json_path = os.path.join(save_dir, f"results_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Results saved to: {json_path}")
    return json_path


def save_force_plots(results, sentence_1, sentence_2):
    """Save force plots as images."""
    save_dir = "demo/multilingual/Multilingual_Expanation_Demo_Results"
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    import matplotlib.pyplot as plt

    for sentence_name, sentence_results in results.items():
        for perturbation_name, (interaction_values, feature_names) in sentence_results.items():
            try:
                filename = f"{sentence_name}_{perturbation_name}_{timestamp}.png"
                filepath = os.path.join(save_dir, filename)

                fig = force_plot(
                    interaction_values,
                    feature_names=feature_names,
                    show=False,
                )

                if hasattr(fig, "savefig"):
                    fig.savefig(filepath, dpi=150, bbox_inches="tight")
                else:
                    plt.savefig(filepath, dpi=150, bbox_inches="tight")

                print(f"✅ Force plot saved: {filepath}")

            except Exception as e:
                print(f"⚠️ Could not save force plot for {sentence_name} {perturbation_name}: {e}")
                print("   Try saving manually...")
                try:
                    import matplotlib.pyplot as plt

                    plt.savefig(filepath, dpi=150, bbox_inches="tight")
                    print(f"✅ Force plot saved (alternative method): {filepath}")
                except:
                    pass


# ============================================================
# Main
# ============================================================


def main() -> None:
    """Run the multilingual explanation demo."""
    print_header()

    sentence_1, sentence_2 = choose_comparison_mode()

    interaction_index, max_order = choose_explanation_settings()

    print("\nSentence 1")

    print(sentence_1)

    print("\nSentence 2")

    print(sentence_2)

    results = generate_explanations(
        sentence_1=sentence_1,
        sentence_2=sentence_2,
        interaction_index=interaction_index,
        max_order=max_order,
    )

    visualize_results(results, sentence_1, sentence_2, interaction_index, max_order)

    print("\nDemo finished.")


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()
