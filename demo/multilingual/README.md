# 🌐 Multilingual Explanation Comparison Demo

## Overview

This demo investigates whether **SHAPIQ explanations**, especially interaction values, remain consistent across **semantically equivalent texts**, such as translations and paraphrases.

Rather than evaluating translation or paraphrase quality, the demo focuses on the robustness of SHAPIQ explanations across different linguistic realizations of the same meaning.

The demo showcases the newly implemented **TextImputer** together with SHAPIQ interaction explanations.

---

# Requirements

- Python **3.12** (recommended)
- Git
- Internet connection (required for downloading Hugging Face models on first use)

> **Note:** The demo has been developed and tested with **Python 3.12**. Other Python versions may require different dependency versions.

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/ddddxx1/shapiq.git
cd shapiq
git checkout Multilingualdemo-Liang
```

## 2. Install Dependencies

This project uses **uv** for dependency management.

Install all required dependencies with:

```bash
uv sync --all-extras
```

The Streamlit interface is **not included** in the default project dependencies and should be installed separately:

```bash
uv pip install streamlit
```

---

# Running the Demo

The demo provides both a web interface and a command-line interface.

## Streamlit Interface (Recommended)

```bash
uv run streamlit run demo/multilingual/app.py
```


---

## Command-Line Interface

```bash
uv run python demo/multilingual/Multilingual_Expanation_Compasion.py
```

Follow the terminal prompts to compare explanations.

---

# Demo Workflow

## Step 1 — Choose a Comparison Mode

Two comparison modes are available.

### User-defined Comparison

Enter two sentences manually.

Example:

```text
Sentence 1:
The movie was fantastic.

Sentence 2:
Der Film war fantastisch.
```

No language generation model is used.

---

### Model-generated Comparison

Enter one original sentence.

The second sentence is automatically generated using one of the following modes:

- Translation
- Paraphrase

---

## Step 2 — Generate the Comparison Sentence

### Translation Mode

Supported language pairs:

| Direction | Model |

|-----------|-------|

| English → German | Helsinki... |

| German → English | Helsinki... |

The translation models are loaded only when Translation mode is selected.

![Translation Mode](images_UI/translation_Mode.png)

---

### Paraphrase Mode

Three paraphrasing styles are supported:

- Conservative

- Moderate

- Structural

![Paraphrase Mode](images_UI/paraphrase_Mode.png)
Model:

```
Qwen/Qwen2.5-0.5B-Instruct
```

The model is prompted to generate exactly one paraphrased sentence without explanations or quotation marks.

---

## Step 3 — Configure SHAPIQ

### Interaction Index

Supported indices:

- k-SII
- STII

### Maximum Interaction Order

- Range: **1–5**
- Default: **2**

---

## Step 4 — Run the Explanation

For each sentence, the demo computes explanations using both perturbation strategies:

- Removal
- MLM Infilling

---

## Output & Visualization

After clicking **Run Explanation**, the following outputs are displayed.

### Main Interface

![Main Page](images_UI/main_page.png)

---

### First-order Interaction Values

The explanation results can be inspected in two complementary views.

#### Bar Chart

![Bar Chart](images_UI/bar_view.png)

#### Table View

![Table View](images_UI/table_view.png)

---

### SHAPIQ Force Plots

Interactive force plots are generated for each sentence under both perturbation strategies.

![Force Plot](images_UI/force_plots.png)

---

For two sentences and two perturbation strategies, a complete comparison produces four explanation results:

- Sentence 1 + Removal

- Sentence 1 + MLM Infilling

- Sentence 2 + Removal

- Sentence 2 + MLM Infilling

---

# Models

| Purpose | Model |
|---------|-------|
| Explanation Model | `lxyuan/distilbert-base-multilingual-cased-sentiments-student` |
| English → German Translation | `Helsinki-NLP/opus-mt-en-de` |
| German → English Translation | `Helsinki-NLP/opus-mt-de-en` |
| Paraphrase | `Qwen/Qwen2.5-0.5B-Instruct` |
| MLM Infilling | `bert-base-uncased` |

---

# Default Configuration

| Setting | Value |
|---------|-------|
| Device | CPU |
| Player Level | Word |
| Perturbation Strategies | Removal + MLM Infilling |
| MLM Model | `bert-base-uncased` |
| MLM Samples | `3` (Debug) / `100` (Full) |
| Default Interaction Index | k-SII |
| Default Maximum Order | `2` |
| Explanation Budget | `2048` |
| Class Index | `0` |

---

# Known Issues

### First Run

The required Hugging Face models are downloaded automatically the first time they are used.

This includes:

- Explanation model
- Translation models
- Paraphrase model
- MLM model

The initial download may take several minutes depending on your network connection.

---

### Internet Connection

An internet connection is required the first time the models are downloaded.

Afterwards, all models are loaded from the local Hugging Face cache.

---

### Streamlit

The Streamlit package is **not installed by default** with SHAPIQ.

If the web interface cannot be launched, install it manually:

```bash
uv pip install streamlit
```

---

### Performance

The **MLM Infilling** perturbation strategy performs multiple forward passes and is therefore considerably slower than the **Removal** strategy.

---

### Device

The demo runs on **CPU by default** to maximize compatibility across different operating systems and hardware platforms.