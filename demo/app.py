"""Use Streamlit UI for Multilingual Explanation Demo.
This module provides a web-based user interface for the SHAPIQ multilingual explanation comparison demo.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from datetime import datetime

import matplotlib.pyplot as plt
import streamlit as st
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

MLM_NUM_SAMPLES = 10
DEFAULT_PLAYER_LEVEL = "word"
EXPLANATION_BUDGET = 2048
SUPPORTED_INTERACTION_INDICES = ["k-SII", "STII"]
DEVICE = "cpu"


# ============================================================
# Load Models (cached)
# ============================================================


@st.cache_resource
def load_explanation_model():
    """Load the explanation model with caching for Streamlit performance."""
    tokenizer = AutoTokenizer.from_pretrained(EXPLANATION_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(EXPLANATION_MODEL).to(DEVICE)
    model.eval()
    return tokenizer, model


@st.cache_resource
def load_translation_model(direction):
    """Load the explanation model with caching for Streamlit performance."""
    if direction == "en-de":
        model_name = TRANSLATION_MODEL_EN_DE
    else:
        model_name = TRANSLATION_MODEL_DE_EN
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)
    model.eval()
    return tokenizer, model


@st.cache_resource
def load_paraphrase_model():
    """Load the paraphrase model with caching."""
    tokenizer = AutoTokenizer.from_pretrained(PARAPHRASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(PARAPHRASE_MODEL).to(DEVICE)
    model.eval()
    return tokenizer, model


# ============================================================
# Translation & Paraphrase
# ============================================================


def translate_sentence(sentence, direction):
    tokenizer, model = load_translation_model(direction)
    inputs = tokenizer(sentence, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=64)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def paraphrase_sentence(sentence, style):
    tokenizer, model = load_paraphrase_model()
    instructions = {
        "Conservative": "Paraphrase with minimal changes. Replace only a few words with synonyms. Preserve structure and meaning.",
        "Moderate": "Paraphrase using clearly different wording and noticeably different structure. Preserve exact meaning.",
        "Structural": "Paraphrase by changing grammatical structure. Keep entities, facts, sentiment unchanged.",
    }
    messages = [
        {
            "role": "system",
            "content": "You are a precise paraphrasing assistant. Return only one paraphrased sentence.",
        },
        {"role": "user", "content": f"{instructions[style]}\n\nSentence: {sentence}"},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    if "assistant" in generated:
        generated = generated.split("assistant")[-1].strip()
    return generated


# ============================================================
# Explanation
# ============================================================


def get_explanation(text, tokenizer, model, interaction_index, max_order, perturbation_type):
    """Compute interaction values for a single sentence."""
    imputer = TextImputer(
        model=model,
        tokenizer=tokenizer,
        text=text,
        player_level=DEFAULT_PLAYER_LEVEL,
        perturbation_type=perturbation_type,
        model_type="encoder_classifier",
        class_idx=0,
        output_type="probability",
    )
    if perturbation_type == "mlm_infilling":
        imputer.mlm_model_name = MLM_MODEL
        imputer.mlm_num_samples = MLM_NUM_SAMPLES

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
# Initialize session_state
# ============================================================

if "generated_sentence" not in st.session_state:
    st.session_state.generated_sentence = ""
if "results" not in st.session_state:
    st.session_state.results = None
if "sentence_2_value" not in st.session_state:
    st.session_state.sentence_2_value = ""
if "pending_compute" not in st.session_state:
    st.session_state.pending_compute = False
if "sentence_1_cache" not in st.session_state:
    st.session_state.sentence_1_cache = ""


# ============================================================
# clear the old results
# ============================================================


def clear_results():
    st.session_state.results = None
    st.session_state.generated_sentence = ""
    st.session_state.pending_compute = False


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(page_title="SHAPIQ Multilingual Demo", layout="wide")

st.title("🌐 SHAPIQ Multilingual Explanation Demo")
st.markdown("---")

# Sidebar with the user settings
with st.sidebar:
    st.header("⚙️ Settings")

    comparison_mode = st.radio(
        "Comparison Mode",
        ["User-defined", "Model-generated"],
        help="User-defined: enter two sentences manually. Model-generated: enter one, translate/paraphrase the second.",
        on_change=clear_results,  # ..
        key="comparison_mode_radio",  # ..
    )

    if comparison_mode == "Model-generated":
        generation_mode = st.radio(
            "Generation Mode",
            ["Translation", "Paraphrase"],
            on_change=clear_results,  # ..
            key="generation_mode_radio",  # ..
        )
        if generation_mode == "Translation":
            direction = st.selectbox(
                "Translation Direction",
                ["English → German", "German → English"],
                on_change=clear_results,  # ..
                key="direction_select",  # ..
            )
            direction_map = {"English → German": "en-de", "German → English": "de-en"}
        else:
            paraphrase_style = st.selectbox(
                "Paraphrase Style",
                ["Conservative", "Moderate", "Structural"],
                on_change=clear_results,  # ..
                key="paraphrase_style_select",  # ..
            )

    interaction_index = st.selectbox(
        "Interaction Index",
        SUPPORTED_INTERACTION_INDICES,
        index=0,
        on_change=clear_results,  # ..
        key="interaction_index_select",  # ..
    )

    max_order = st.slider(
        "Maximum Interaction Order",
        1,
        5,
        2,
        on_change=clear_results,  # ..
        key="max_order_slider",  # ..
    )

    run_button = st.button("🚀 Run Explanation", type="primary", use_container_width=True)

# Research question based on mode
if comparison_mode == "User-defined":
    st.markdown("""
    What would you like to compare? Enter two sentences below.
    """)
else:
    st.markdown("""
    Are SHAPIQ explanations stable across semantically equivalent texts?
    """)

# Mode-specific instruction
if comparison_mode == "Model-generated":
    if generation_mode == "Translation":
        st.info(
            "📝 **Translation Mode**: Enter a sentence below. We'll translate it and compare both explanations."
        )
    else:
        st.info(
            "📝 **Paraphrase Mode**: Enter a sentence below. We'll generate a paraphrase and compare both explanations."
        )

# Main input area
col1, col2 = st.columns(2)

with col1:
    st.subheader("Sentence 1")
    if comparison_mode == "User-defined":
        sentence_1 = st.text_area("Enter first sentence:", value="The movie is good!", height=100)
    else:
        sentence_1 = st.text_area(
            "Enter original sentence:", value="The movie is good!", height=100
        )

with col2:
    st.subheader("Sentence 2")
    if comparison_mode == "User-defined":
        sentence_2 = st.text_area("Enter second sentence:", value="Der Film ist gut!", height=100)
    else:
        # load result from session_state
        if st.session_state.generated_sentence:
            display_value = st.session_state.generated_sentence
        else:
            display_value = ""

        sentence_2 = st.text_area(
            "Generated:",
            value=display_value,
            height=100,
            disabled=True,
            placeholder="Enter sentence 1 and click 'Run Explanation'",
        )
        if st.session_state.generated_sentence:
            st.caption(f"✅ Translated from: {sentence_1}")


# ============================================================
# Run
# ============================================================

# Check if computation is pending (translation done, ready to compute)
if st.session_state.get("pending_compute", False):
    st.session_state.pending_compute = False
    sentence_1 = st.session_state.get("sentence_1_cache", "")
    sentence_2 = st.session_state.get("generated_sentence", "")

    if sentence_1 and sentence_2:
        tokenizer, model = load_explanation_model()

        with st.status("**🔄 Computing explanations...**", expanded=True) as status:
            results = {}
            for i, (sentence, label) in enumerate(
                [(sentence_1, "Sentence 1"), (sentence_2, "Sentence 2")]
            ):
                status.update(label=f"**Step 4 / 5: Computing {label}...**")
                removal_iv, removal_fn = get_explanation(
                    sentence, tokenizer, model, interaction_index, max_order, "removal"
                )
                mlm_iv, mlm_fn = get_explanation(
                    sentence, tokenizer, model, interaction_index, max_order, "mlm_infilling"
                )
                results[label] = {
                    "removal": (removal_iv, removal_fn),
                    "mlm_infilling": (mlm_iv, mlm_fn),
                    "text": sentence,
                }
            status.update(label="**Step 5 / 5: Done!✅**", state="complete")
            st.session_state.results = results
            st.session_state.sentence_2_value = sentence_2

elif run_button:
    if not sentence_1:
        st.error("Please enter sentence 1.")
        st.stop()

    if comparison_mode == "Model-generated":
        # Firstly: generate sentence 2, then to compute
        with st.spinner("Generating sentence 2..."):
            if generation_mode == "Translation":
                direction_short = direction_map[direction]
                st.session_state.generated_sentence = translate_sentence(
                    sentence_1, direction_short
                )
            else:
                st.session_state.generated_sentence = paraphrase_sentence(
                    sentence_1, paraphrase_style
                )

        st.session_state.sentence_1_cache = sentence_1
        st.session_state.pending_compute = True
        st.rerun()
    else:
        # User-defined mode
        if not sentence_2:
            st.error("Please enter sentence 2.")
            st.stop()

        tokenizer, model = load_explanation_model()

        with st.status("**🔄 Computing explanations...**", expanded=True) as status:
            results = {}

            for i, (sentence, label) in enumerate(
                [(sentence_1, "Sentence 1"), (sentence_2, "Sentence 2")]
            ):
                status.update(label=f"**Step 4 / 5: Computing {label}...**")
                removal_iv, removal_fn = get_explanation(
                    sentence, tokenizer, model, interaction_index, max_order, "removal"
                )
                mlm_iv, mlm_fn = get_explanation(
                    sentence, tokenizer, model, interaction_index, max_order, "mlm_infilling"
                )
                results[label] = {
                    "removal": (removal_iv, removal_fn),
                    "mlm_infilling": (mlm_iv, mlm_fn),
                    "text": sentence,
                }
            status.update(label="**Step 5 / 5: Done!✅**", state="complete")
            st.session_state.results = results
            st.session_state.sentence_2_value = sentence_2


# ============================================================
# Display Results
# ============================================================

if st.session_state.results is not None:
    results = st.session_state.results
    st.markdown("---")
    st.header("📊 Results")
    st.caption("Step 5 / 5: Visualization")

    col1, col2 = st.columns(2)
    with col1:
        st.caption(f"**Sentence 1:** {results['Sentence 1']['text']}")
    with col2:
        st.caption(f"**Sentence 2:** {results['Sentence 2']['text']}")

    st.markdown("---")

    # First-order interaction values (bar charts)
    st.subheader("First-Order Interaction Values")
    tabs = st.tabs(["Sentence 1", "Sentence 2"])

    for idx, (label, tab) in enumerate(zip(["Sentence 1", "Sentence 2"], tabs, strict=False)):
        with tab:
            cols = st.columns(2)
            for j, (pert_type, display_name) in enumerate(
                [("removal", "Removal"), ("mlm_infilling", "MLM Infilling")]
            ):
                with cols[j]:
                    st.caption(f"{display_name}")
                    iv, fn = results[label][pert_type]
                    data = {fn[i]: float(iv[(i,)]) for i in range(len(fn))}
                    st.bar_chart(data)

    # Force plots, show and save
    st.subheader("Force Plots")

    save_dir = "demo/Multilingual_Expanation_Demo_Results"
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for label in ["Sentence 1", "Sentence 2"]:
        st.markdown(f"**{label}:** {results[label]['text']}")
        cols = st.columns(2)
        for j, (pert_type, display_name) in enumerate(
            [("removal", "Removal"), ("mlm_infilling", "MLM Infilling")]
        ):
            with cols[j]:
                st.caption(display_name)
                iv, fn = results[label][pert_type]
                fig = force_plot(iv, feature_names=fn, show=False)

                filename = f"{label.lower().replace(' ', '_')}_{pert_type}_{timestamp}.png"
                filepath = os.path.join(save_dir, filename)
                fig.savefig(filepath, dpi=150, bbox_inches="tight")
                print(f"✅ Force plot saved: {filepath}")

                if hasattr(fig, "set_size_inches"):
                    fig.set_size_inches(8, 5)
                st.pyplot(fig, use_container_width=True)
                plt.close()

    st.success("✅ Done!")

st.markdown("---")
st.caption("Built with Streamlit + SHAPIQ")
