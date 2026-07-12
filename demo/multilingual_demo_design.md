# Multilingual Explanation Comparison Demo

## Design Specification (Current Implementation Version)

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

------------------------------------------------------------------------

## 2. Research Question

Given two semantically equivalent sentences,

-   original sentence
-   translated sentence
-   paraphrased sentence

run the same explanation pipeline and compare:

-   first-order interaction values
-   second-order interaction values
-   force plots

The demo does not automatically conclude whether explanations are
stable.

Instead, it visualizes and prints the explanation evidence for the
presenter or user to compare.

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

Chinese is not included in the current version because the explanation
pipeline uses word-level players and multilingual tokenization
compatibility has not yet been validated for Chinese.

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

The demo uses a CLI terminal interface.

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

Choose:
```

Invalid menu inputs are detected and the user is asked to enter a valid
option again.

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

## 10. User-selectable Explanation Settings

### Interaction Index

Supported indices:

-   k-SII
-   STII
-   FSII

Default workflow selection is available through the CLI.

### Maximum Order

The user selects the maximum interaction order.

Allowed range:

``` text
1 to 5
```

Default:

``` text
2
```

The current visualization function explicitly prints first-order and
second-order values.

Therefore, the intended demo configuration is maximum order 2.

------------------------------------------------------------------------

## 11. Perturbation Strategies

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

## 12. MLM Configuration

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

## 13. SHAPIQ Workflow

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

## 14. Terminal Explanation Output

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

## 15. Visualization

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

Plots are currently displayed interactively and are not automatically
saved.

------------------------------------------------------------------------

## 16. Result Structure

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

## 17. Design Principles

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

## 18. Implementation Constraints

The demo is implemented as one Python file.

Current demo implementation:

``` text
demo/
    Multilingual Expanation Compasion.py
```

No additional helper modules are required.

The implementation directly orchestrates:

-   translation
-   paraphrasing
-   TextImputer construction
-   SHAPIQ approximation
-   terminal interaction output
-   force plot visualization

------------------------------------------------------------------------

## 19. Current Debug Configuration

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

## 20. Expected Outcome

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
