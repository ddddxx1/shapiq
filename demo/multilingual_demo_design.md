# 🌐 Multilingual Explanation Comparison Demo

## Design Specification

------------------------------------------------------------------------

## 1. Motivation

This demo is designed for the **Multilinguality & Robustness** topic
proposed in the SHAPIQ project.

The goal is not to evaluate machine translation or paraphrase quality
itself, but to investigate:

> Are SHAPIQ explanations, especially interaction values, stable across
> semantically equivalent texts such as translations or paraphrases?

The demo showcases the newly implemented `TextImputer` together with
SHAPIQ interaction explanations.
插图
------------------------------------------------------------------------

## 2. Research Question

Given two semantically equivalent sentences,

-   original sentence
-   translated sentence (English ↔ German)
-   paraphrased sentence (English)

run the same explanation pipeline and compare:

- **First-order interaction values** — individual feature contributions and displayed as bar charts
- **Second-order interaction values** — pairwise feature interactions
- **Force plots** — SHAPIQ's interactive visualizations


The demo does not automatically conclude whether explanations are
stable. Instead, it visualizes and prints the explanation evidence for the
presenter or user to compare.

------------------------------------------------------------------------

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Two Comparison Modes** | User-defined (any two sentences) or Model-generated (translation/paraphrase) |
| **Multiple Interaction Indices** | k-SII, STII |
| **Two Perturbation Strategies** | Removal (baseline) and MLM Infilling (context-aware) |
| **Multi-language Support** | English ↔ German (translation); English paraphrasing |
| **Result Export** | JSON data + PNG force plots saved automatically |
| **Two Interfaces** | Web UI (Streamlit) + CLI (terminal) |

------------------------------------------------------------------------

## 3. Overall Pipeline

The demo consists of two input branches.

``` text
                    Comparison Mode
                           |
        +------------------+------------------+
        |                                     |
        v                                     v
User-defined Comparison           Model-generated Comparison
        |                                     |
Sentence 1                     Original Sentence
Sentence 2                              |
                                        v
                          Translation / Paraphrase
                                        |
                       +----------------+----------------+
                       |                                 |
                       v                                 v
                    OPUS-MT                    Qwen2.5-0.5B-Instruct
                       |                                 |
                       +----------------+----------------+
                                        |
                                        v
                                   Sentence 2
        +---------------------------+---------------------------+
                                    |
                                    v
                           Explanation Pipeline
                                    |
                               TextImputer
                                    |
                           SHAPIQ Approximator
                                    |
                           InteractionValues
                                    |
                  Terminal Values + SHAPIQ Force Plot
```

------------------------------------------------------------------------

## 4. Comparison Modes

### 4.1 User-defined Comparison

The user directly provides two sentences.

Example:

**Sentence 1**

``` text
The movie was fantastic.
```

**Sentence 2**

``` text
Der Film war fantastisch.
```

No text generation model is used.

------------------------------------------------------------------------

### 4.2 Model-generated Comparison

The user provides one original sentence.

The second sentence is automatically generated.

Two generation modes are supported:

-   Translation
-   Paraphrase

------------------------------------------------------------------------

## 5. Translation

### Supported Languages

The current implementation supports:

-   English → German
-   German → English

### Translation Models

English to German:

``` text
Helsinki-NLP/opus-mt-en-de
```

German to English:

``` text
Helsinki-NLP/opus-mt-de-en
```

Translation is performed with `AutoModelForSeq2SeqLM`.

The translation models are loaded only when translation mode is
selected.

------------------------------------------------------------------------

## 6. Paraphrase

The current implementation provides three paraphrase styles.

### Conservative

Minimal changes.

The model is instructed to replace only a few words with synonyms while
preserving sentence structure and meaning.

Generation is deterministic.

### Moderate

Clearly different wording and a noticeably different sentence structure.

The exact meaning and important information should remain unchanged.

Sampling is enabled.

### Structural

The grammatical structure should be visibly changed.

The model is explicitly instructed to preserve:

-   entities
-   facts
-   sentiment
-   relationships

and not introduce new information.

### Paraphrase Model

``` text
Qwen/Qwen2.5-0.5B-Instruct
```

The model is loaded with `AutoModelForCausalLM`.

The prompt uses the tokenizer chat template and requests exactly one
paraphrased sentence without explanations or quotation marks.

Generated text cannot be manually edited inside the demo workflow.

------------------------------------------------------------------------

## 7. User Interface

### Web Interface (Streamlit) — Recommended

The recommended way to explore the demo. Provides an interactive dashboard with real-time visualization.

use command line: streamlit run demo/app.py
Then open your browser to http://localhost:8501


### Interactive Workflow

#### Step 1: Choose Comparison Mode (Sidebar)

| Mode | Description |
|------|-------------|
| **User-defined** | Enter two sentences manually for comparison |
| **Model-generated** | Enter one sentence, auto-generate the second via translation or paraphrase |

---

#### Step 2: Select Generation Mode (Model-generated only)

| Mode | Description |
|------|-------------|
| **Translation** | Translate between English and German using OPUS-MT |
| **Paraphrase** | Generate a paraphrase using Qwen2.5-0.5B-Instruct |

**Three paraphrase styles are available:**

| Style | Description |
|-------|-------------|
| **Conservative** | Minimal changes, synonym replacement only |
| **Moderate** | Clearly different wording, different sentence structure |
| **Structural** | Changed grammatical structure, preserves entities and facts |

---

#### Step 3: Choose Explanation Settings (Sidebar)

| Setting | Options | Description |
|---------|---------|-------------|
| **Interaction Index** | k-SII, STII | Shapley interaction indices |
| **Maximum Order** | 1–5 (default 2) | Maximum interaction order to compute |

---

#### Step 4: Enter Sentences (Main Area)

- **Sentence 1**: The reference sentence (or original sentence in Model-generated mode)
- **Sentence 2**: The comparison sentence (auto-generated in Model-generated mode)

---

#### Step 5: View Results

After clicking **"Run Explanation"**, the following outputs are displayed:

**1. First-Order Interaction Values (Bar Charts)**

- Individual contribution of each word
- Shown separately for Sentence 1 and Sentence 2
- Separate tabs for Removal and MLM Infilling

**2. Force Plots (Interactive Visualizations)**

- SHAPIQ's signature explanation plots
- 4 plots total: Sentence 1/2 × Removal/MLM Infilling
- Saved as PNG files automatically

**[插图位置: 结果展示截图 - 显示一阶值柱状图和 Force Plot]**


The demo also uses a CLI terminal interface.

For terminal-based interaction: PYTHONPATH=src python demo/Multilingual_Expanation_Compasion.py

No Gradio.

No Streamlit.

Menus use numbered choices.

Example:

``` text
============================================================
SHAPIQ Multilingual Explanation Demo
============================================================

Research Question
Are SHAPIQ explanations stable across semantically equivalent texts?

Step 1 / 5

Comparison Mode
1. User-defined Comparison
2. Model-generated Comparison

Choose: 2

Original Sentence
> The movie is good!

Generation Mode
1. Translation
2. Paraphrase

Choose: 1

Source Language
1. English
2. German

Choose: 1

Step 2 / 5

Interaction Index
1. k-SII
2. STII

Choose: 1

Maximum Order [Default 2]: 2

[... results displayed ...]
```

Invalid menu inputs are detected and the user is asked to enter a valid
option again.

The current visualization function explicitly prints first-order and
second-order values.Therefore, the intended demo configuration is maximum order 2.

------------------------------------------------------------------------

## 8. Explanation Pipeline

The demo always uses the implemented public `TextImputer` API.

The demo does not reimplement imputation logic.

Pipeline:

``` text
Sentence
   |
   v
TextImputer
   |
   v
SHAPIQ Approximator
   |
   v
InteractionValues
   |
   +----------------------+
   |                      |
   v                      v
Terminal Output       Force Plot
```

------------------------------------------------------------------------

## 9. Fixed Explanation Settings

### Player Strategy

``` text
word
```

The user cannot modify the player strategy.

Word-level players are retrieved from:

``` python
imputer.player_strategy.get_players()
```

These player names are passed to the force plot as `feature_names`.

### Explanation Model

``` text
lxyuan/distilbert-base-multilingual-cased-sentiments-student
```

The model is loaded with `AutoModelForSequenceClassification`.

The explanation target is configured as:

``` text
class_idx = 0
output_type = probability
model_type = encoder_classifier
```

The same explanation model and target configuration are used for both
sentences.

------------------------------------------------------------------------


## 10. Perturbation Strategies

The user does not choose a perturbation strategy.

The demo automatically runs both:

### Experiment 1: Removal

``` text
perturbation_type = removal
```

This provides a fast baseline.

### Experiment 2: MLM Infilling

``` text
perturbation_type = mlm_infilling
```

This provides context-aware perturbation.

Both perturbation strategies use the same:

-   sentence
-   explanation model
-   interaction index
-   maximum order
-   SHAPIQ approximator configuration

The purpose is to compare a fast removal baseline with context-aware MLM
infilling.

------------------------------------------------------------------------

## 11. MLM Configuration

### MLM Model

``` text
bert-base-uncased
```

### Number of Samples

The demo currently contains a debug switch:

``` python
DEBUG = True
```

When debug mode is enabled:

``` text
MLM_NUM_SAMPLES = 3
```

When debug mode is disabled:

``` text
MLM_NUM_SAMPLES = 100
```

This allows fast development and debugging while preserving the intended
higher-sample configuration for final experiments.

------------------------------------------------------------------------

## 12. SHAPIQ Workflow

The demo does not compute interaction values manually.

For every sentence and perturbation strategy:

``` text
Text
 |
 v
TextImputer
 |
 v
SHAPIQ(
    n = imputer.n_features,
    index = selected interaction index,
    max_order = selected maximum order
)
 |
 v
approximator.approximate(
    budget = 2048,
    game = imputer
)
 |
 v
InteractionValues
```

The explanation budget is currently fixed to:

``` text
2048
```

The resulting `InteractionValues` object is used directly by the output
and visualization pipeline.

------------------------------------------------------------------------

## 13. Terminal Explanation Output

For every sentence and perturbation strategy, the demo prints the
interaction values.

### First-order Interaction Values

For every player index `i`, the demo retrieves:

``` python
interaction_values[(i,)]
```

and aligns the value with:

``` python
feature_names[i]
```

Example:

``` text
First-order Interaction Values
--------------------------------------------------------------------------------
The                       -0.000037
movie                     +0.057387
was                       -0.076947
fantastic                 +0.421160
.                         -0.049468
```

### Second-order Interaction Values

For every player pair `(i, j)` with `i < j`, the demo retrieves:

``` python
interaction_values[(i, j)]
```

The corresponding feature names are printed together.

Example:

``` text
Second-order Interaction Values
--------------------------------------------------------------------------------
The x movie                              +0.012345
The x fantastic                          -0.023456
movie x fantastic                        +0.134567
was x fantastic                          -0.098765
```

The terminal currently prints all first-order and all second-order
values.

No Top-k filtering is applied in the current implementation.

------------------------------------------------------------------------

## 14. Visualization

The demo uses the official SHAPIQ force plot:

``` python
force_plot(
    interaction_values,
    feature_names=feature_names,
    show=True,
)
```

A force plot is produced for every sentence and every perturbation
strategy.

For two sentences and two perturbation strategies, the current workflow
produces four force plots:

``` text
Sentence 1 + Removal
Sentence 1 + MLM Infilling
Sentence 2 + Removal
Sentence 2 + MLM Infilling
```

The force plots receive the word-level player names as `feature_names`.

The demo does not reimplement SHAPIQ visualization.

Plots are currently displayed interactively and are automatically saved.

------------------------------------------------------------------------

## 15. Result Structure

The explanation pipeline stores results in a nested dictionary.

Conceptually:

``` text
results
 |
 +-- sentence_1
 |      |
 |      +-- removal
 |      |      +-- InteractionValues
 |      |      +-- feature_names
 |      |
 |      +-- mlm_infilling
 |             +-- InteractionValues
 |             +-- feature_names
 |
 +-- sentence_2
        |
        +-- removal
        |      +-- InteractionValues
        |      +-- feature_names
        |
        +-- mlm_infilling
               +-- InteractionValues
               +-- feature_names
```

This keeps explanation computation separate from terminal output and
visualization.

------------------------------------------------------------------------

## 16. Design Principles

### Principle 1

The demo visualizes evidence.

It does not automatically determine whether explanations are stable.

### Principle 2

Translation and paraphrasing are only methods for generating Sentence 2.

The real purpose of the demo is explanation comparison.

### Principle 3

The demo never reimplements `TextImputer`.

Everything goes through the public API.

### Principle 4

The demo never reimplements SHAPIQ interaction computation.

Interaction values are produced by the SHAPIQ approximator.

### Principle 5

The official SHAPIQ force plot is used for visualization.

### Principle 6

The same explanation target is used for both semantically equivalent
sentences so that their explanation structures can be compared.

### Principle 7

Readability is preferred over unnecessary software engineering
complexity.

The demo is intended for presentation and experimentation.

------------------------------------------------------------------------

## 17. Project Structure

The demo is implemented as one Python file.

``` text
shapiq/
├── demo/
│   ├── app.py                                    # Streamlit UI (main entry)
│   ├── Multilingual_Expanation_Compasion.py      # CLI version
│   ├── multilingual_demo_design.md               # Design specification
│   ├── Multilingual_Expanation_Demo_Results/     # Generated outputs (ignored)
│   └── __init__.py
├── src/
│   └── shapiq/                                    # Core library
│       ├── approximator/                          # SHAPIQ approximators
│       ├── imputer/                               # TextImputer implementation
│       ├── plot/                                  # Visualization tools
│       └── tree/                                  # Tree-based explanations
├── tests/                                         # Unit tests
├── .gitignore
├── README.md
├── LICENSE
└── pyproject.toml
``` 

------------------------------------------------------------------------

## 18. Current Debug Configuration
The demo supports a debug mode for faster testing.
The current development version uses:

``` text
DEBUG = True
MLM_NUM_SAMPLES = 3
EXPLANATION_BUDGET = 2048
DEFAULT_PLAYER_LEVEL = word
DEFAULT_INTERACTION_INDEX = k-SII
DEFAULT_MAX_ORDER = 2
```

Before a final experiment, debug mode can be disabled to restore:

``` text
MLM_NUM_SAMPLES = 100
```

------------------------------------------------------------------------

## 19. Device Configuration

DEVICE = "cpu"

------------------------------------------------------------------------

## 20. Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| **shapiq** | Latest | Core explanation library |
| **torch** | 2.0+ | Deep learning framework |
| **transformers** | 4.30+ | Hugging Face models |
| **streamlit** | 1.59+ | Web UI framework |
| **matplotlib** | 3.7+ | Visualization |
| **numpy** | 1.24+ | Numerical operations |
| **scipy** | 1.10+ | Scientific computing |
| **nltk** | 3.9+ | Natural language processing |

------------------------------------------------------------------------

## 21. Expected Outcome

The demo enables users to compare explanation consistency between
semantically equivalent texts while showcasing the modular `TextImputer`
implementation.

It highlights:

-   translation robustness
-   paraphrase robustness
-   multilingual sentiment explanations
-   word-level players
-   first-order interaction values
-   second-order interaction values
-   Removal vs MLM Infilling
-   official SHAPIQ force plots

The final interpretation remains with the user or presenter.

The demo provides explanation evidence rather than an automatic
stability verdict.
