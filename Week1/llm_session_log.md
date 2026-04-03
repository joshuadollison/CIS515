# CIS515 - Week 1 Assignment LLM Session Log

**Date:** April 3, 2026
**Platform:** Claude Code (Claude Opus 4.6)
**Purpose:** Assignment assistance for CIS515 AI and Data Analytics Strategy — Assignment 1: Databricks ML Management: Automated Apparel Classification Monitoring

---

## Session Overview

This session covered the following topics:
1. Assignment planning and task breakdown
2. Part 1 written answers (Q1.1 and Q1.3)
3. Part 2 CNN model architecture (all 6 code tasks)
4. Part 3 hyperparameter search strategy and Optuna code
5. Part 5 written answers (Q5.1 and Q5.2)
6. Part 6 drift analysis and production recommendations

---

## 1. Assignment Planning

**User:** Asked for a complete task list of everything needed to submit the assignment.

**Claude:** Analyzed all files in the Week1 folder (A1 Framework notebook, Solution NN MLFlow Demo notebook, Assignment1_Starter.docx, Lab 1-2.docx) and produced a comprehensive task list of ~25 items covering:
- Setup (Databricks account, workspace, notebook import, LLM session)
- Part 1: 2 written answers about data exploration and drift
- Part 2: 6 code tasks building the CNN architecture
- Part 3: 1 written answer + 5 code tasks for Optuna hyperparameter search
- Part 4: 4 MLflow screenshots + 1 written interpretation
- Part 5: 2 written answers about model registry
- Part 6: 1 written answer about drift evaluation
- Submission: 3 deliverables (.ipynb, answer sheet .pdf, LLM session log .pdf)

---

## 2. Part 1 — Question 1.1: CNN Preprocessing vs NN Feature Engineering

**User:** Drafted an answer comparing CNN preprocessing to the NN demo's feature engineering and asked for help articulating why there was less work in the CNN.

**Claude:** Suggested adding the *why*: CNNs learn their own features from raw pixel data through convolutional filters, eliminating the need for manual feature extraction. The NN demo required EDA, manual feature engineering (job groupings, AI flags), one-hot encoding, and standard scaling.

**Final Answer:**
> In the NN demo we had to perform exploratory data analysis to understand the data, manually engineer features (e.g., job groupings, AI flags), and apply one-hot encoding and standard scaling before the model could use the data. In contrast, the CNN required only normalizing pixel values to 0–1 and adding a channel dimension (28×28×1), because convolutional layers automatically learn spatial features directly from raw images, eliminating the need for manual feature extraction.

---

## 3. Part 1 — Question 1.3: Drift Simulation Effects

**User:** Drafted an answer about drift effects and identified occlusion as the most impactful type.

**Claude:** Suggested two additions:
1. Contrast with other drift types — noise preserves overall shape, darkening retains edges, but occlusion completely removes spatial information in that region.
2. Connect to how CNNs work — Conv2D filters learn to detect features in specific spatial locations, so occlusion is especially harmful.

**Final Answer:**
> These changes will absolutely affect prediction quality. The model was trained on clean, studio-quality images, so these distortions will degrade its ability to find the same patterns it learned during training. The most impactful will likely be occlusion, since it places a random black rectangle over ~25% of the image, completely destroying spatial features in that region. Unlike noise (which adds fuzz but preserves overall shape) or darkening (which reduces intensity but retains edges), occlusion removes local information entirely — which is especially harmful for a CNN that relies on convolutional filters to detect features in specific spatial locations.

---

## 4. Part 2 — CNN Model Architecture (build_keras function)

**User:** Built the model step by step, asking for validation at each stage.

### Step 1: Sequential model with InputLayer
```python
model = models.Sequential([
    models.layers.InputLayer(input_shape=(28, 28, 1))
])
```
**Claude:** Confirmed correct. Noted the explicit InputLayer approach is clean since it separates input definition from the first operation.

### Step 2: First Conv2D + MaxPool pair
```python
model.add(layers.Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding="valid", activation="relu"))
model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
```
**Claude:** Confirmed matches assignment spec exactly.

### Step 3: Second Conv2D + MaxPool pair
Same pattern with 16 filters. **Claude:** Confirmed correct.

### Step 4: Flatten
```python
model.add(layers.Flatten())
```
**Claude:** Confirmed correct.

### Step 5: Dense layer
```python
model.add(layers.Dense(128, activation="relu"))
```
**Claude:** Confirmed correct.

### Step 6: Dropout, Output, Compile
**Claude** identified two issues in the user's initial code:
1. `Dropout()` needs a rate argument (e.g., 0.5) — without it, no dropout occurs
2. Final `Dense` layer needs the number of output units (10 for Fashion-MNIST)

**User** decided to:
- Add a `dropout` parameter to the function (for Optuna tuning)
- Use the existing `conv1_filters`, `conv2_filters`, `learning_rate` args instead of hardcoded values
- Use the pre-defined `num_classes` variable for the output layer

### Final build_keras function:
```python
def build_keras(conv1_filters, conv2_filters, learning_rate, dropout=0.5):
    model = models.Sequential([
        models.layers.InputLayer(input_shape=(28, 28, 1))
    ])
    model.add(layers.Conv2D(filters=conv1_filters, kernel_size=(3, 3), strides=(1, 1), padding="valid", activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(filters=conv2_filters, kernel_size=(3, 3), strides=(1, 1), padding="valid", activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(num_classes, activation="softmax"))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
```

---

## 5. Part 3 — Hyperparameter Search Strategy

**User:** Asked for help choosing hyperparameter ranges.

**Claude:** Provided ranges with detailed justification:

- **conv1_filters [8, 16, 32]** — Early layers detect simple features; powers of 2 align with GPU memory architecture
- **conv2_filters [16, 32, 64]** — Deeper layers combine features into complex patterns, benefiting from more filters
- **learning_rate [1e-4 to 1e-2, log scale]** — Effect is multiplicative; log scale samples proportionally across magnitudes
- **epochs [5 to 20]** — Fashion-MNIST is small enough to converge quickly; beyond 20 risks overfitting
- **batch_size [32, 64, 128]** — Standard trade-off between gradient noise and training speed
- **N_TRIALS = 15** — Enough for Optuna's Bayesian sampler to explore with 6 hyperparameters
- **dropout [0.2 to 0.5]** — Below 0.2 provides minimal regularization; above 0.5 starves the network

### Optuna Code Review

**Claude** found 3 bugs in the user's Optuna code:
1. **Typo:** `batch` should be `batch_size` in `mlflow.log_param("batch_size", batch)`
2. **Duplicate logging:** `mlflow.log_params(trial.params)` already logs all params — the individual `log_param` calls for epochs and batch_size would error
3. **Wrong direction:** `direction="minimize"` should be `direction="maximize"` since we're optimizing accuracy

---

## 6. Part 5 — Model Registry Questions

### Question 5.1: Model Signatures

**User's Answer:**
> A model signature records the schema of the model's expected inputs and outputs—specifically, the shape, data types, and dimensionality of the input tensors (e.g., float32 arrays of shape [28, 28, 1]) and output tensors (e.g., 10-class probability vectors). This information is stored so that MLflow can validate inference requests at serving time, preventing mismatched data from reaching the model and causing silent errors or crashes.

**Claude:** Confirmed strong and complete.

### Question 5.2: Model Registration, Versioning, Aliasing, Model URL

**User's Answer:**
> - **(a) Registering the model and tracking versions** creates a centralized catalog of all model iterations with their metrics, parameters, and artifacts, enabling reproducibility and easy rollback if a newer version underperforms in production.
> - **(b) Assigning an alias (e.g., "staging" or "champion")** to a registered model version decouples deployment references from specific version numbers, so downstream services always point to the correct model without code changes when a new version is promoted.
> - **(c) Creating a model URL** provides a stable, shareable endpoint that team members and automated pipelines can use to load or serve the model directly, streamlining collaboration and CI/CD integration.

**Claude:** Confirmed strong, concise, nothing missing.

---

## 7. Part 6 — Drift Evaluation and Production Recommendations

**User:** Shared accuracy drop chart showing:
- Noisy: ~22% drop
- Occluded: ~39% drop
- Dark: ~11% drop
- All three exceed the 5% warning threshold

**Claude** brainstormed production actions:

**Detection/Monitoring:**
- Automated drift detection (KS test on pixel distributions)
- Accuracy monitoring with alerts at the 5% threshold

**Prevention/Hardening:**
- Data augmentation (random noise, cutout, brightness variation in training pipeline)
- Preprocessing pipeline (contrast adjustment before inference for dark images)

**Response/Recovery:**
- Automated retraining triggered by drift detection
- Fallback model using MLflow aliases for automatic rollback
- Human-in-the-loop for low-confidence predictions

**Operational:**
- Input quality gates to reject/flag degraded images
- Segment-specific models for unavoidable conditions

---

## End of Session
