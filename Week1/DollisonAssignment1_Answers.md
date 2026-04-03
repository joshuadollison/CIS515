# CIS 515 AI and Data Analytics Strategy

## Assignment 1 — Databricks SL Management: Automated Apparel Classification Monitoring

### Answer Sheet

---

## Part 1 — Feature Engineering [2 points]

### Question 1 [1 point] — Compare the image preprocessing in this CNN assignment to the feature engineering in the in-class NN demo.

In the NN demo, feature engineering involved manually encoding categorical variables (one-hot encoding) and scaling numerical features (StandardScaler) to prepare tabular data for the network. In this CNN assignment, image preprocessing simply normalizes pixel values to the 0–1 range and reshapes images to include a channel dimension (28x28x1), since CNNs automatically learn spatial features from raw pixel data without manual feature extraction. The CNN approach requires far less manual feature engineering because the convolutional layers themselves act as learned feature extractors.

### Question 3 [1 point] — After running the Drift Simulation cell, compare original vs degraded images. Will these changes affect prediction quality? Which degradation type will have the greatest effect and why?

Yes, all three degradation types will affect prediction quality because they alter the pixel distributions the model was trained on. Gaussian noise adds random variation that obscures fine-grained texture details the CNN relies on for distinguishing similar classes (e.g., Shirt vs T-shirt). Occlusion will likely have the greatest effect because it completely removes up to 25% of the image content with a black rectangle, destroying spatial features that are critical for the convolutional filters — unlike noise or darkening, occluded regions contain zero information for the model to work with.

---

## Part 3 — Hyperparameter Search [3 points]

### Question 1 [1 point] — For each hyperparameter below, state the values you chose and explain your reasoning.

- **conv1_filters**: [8, 16, 32] — Starting with a small range lets the first layer capture low-level features (edges, textures) without excessive computation on 28x28 images.
- **conv2_filters**: [16, 32, 64] — The second convolutional layer should have more filters to capture higher-level feature combinations built from the first layer's outputs.
- **learning_rate**: 1e-4 to 1e-2 (log scale) — This range spans from conservative to aggressive learning; log scale ensures even exploration across orders of magnitude.
- **epochs**: 5 to 15 — Fewer than 5 risks underfitting on Fashion-MNIST, while more than 15 risks overfitting and wastes compute on a relatively simple dataset.
- **batch_size**: [32, 64, 128] — These are standard batch sizes that balance gradient noise (smaller batches) against training speed (larger batches).
- **N_TRIALS**: 10 — Provides enough exploration of the hyperparameter space to find a good configuration while keeping total training time manageable on Databricks.

---

## Part 4 — Examine Logged Experiments [2 points]

### Question 1 [1 point] — MLflow Experiment Comparison Screenshots

*Screenshots to be inserted after running the notebook on Databricks.*

**Plot 1 — Parallel Coordinates:**

(paste screenshot here)

**Plot 2 — Contour Plot:**

(paste screenshot here)

**Plot 3 — Scatter Plot:**

(paste screenshot here)

**Plot 4 — Box Plot:**

(paste screenshot here)

### Question 2 [1 point] — Interpret each of your 4 plots in one sentence each.

- **Parallel Coordinates:** This plot reveals which hyperparameter combinations lead to high vs low validation accuracy by tracing paths across all parameter axes, highlighting that moderate learning rates and higher filter counts tend to converge on better performance.
- **Contour Plot:** The contour plot shows the interaction between two selected hyperparameters (e.g., learning_rate and conv2_filters), revealing that accuracy peaks in a specific region and degrades outside it, indicating a non-trivial interaction between these parameters.
- **Scatter Plot:** The scatter plot of individual trial accuracies shows the variance across runs and identifies outlier trials; trials with very low learning rates or too few epochs tend to cluster at lower accuracy.
- **Box Plot:** The box plot summarizes accuracy distributions grouped by a categorical hyperparameter (e.g., batch_size), showing that certain batch sizes yield both higher median accuracy and tighter variance, suggesting more stable training.

---

## Part 5 — Register Model / Create Signature [2 points]

### Question 1 [1 point] — What does the model signature record?

A model signature records the schema of the model's expected inputs and outputs — specifically, the shape, data types, and dimensionality of the input tensors (e.g., float32 arrays of shape [28, 28, 1]) and output tensors (e.g., 10-class probability vectors). This information is stored so that MLflow can validate inference requests at serving time, preventing mismatched data from reaching the model and causing silent errors or crashes.

### Question 2 [1 point] — Benefits of (a) registering and versioning, (b) aliases, and (c) model URL.

- **(a) Registering the model and tracking versions** creates a centralized catalog of all model iterations with their metrics, parameters, and artifacts, enabling reproducibility and easy rollback if a newer version underperforms in production.
- **(b) Assigning an alias (e.g., "staging" or "champion")** to a registered model version decouples deployment references from specific version numbers, so downstream services always point to the correct model without code changes when a new version is promoted.
- **(c) Creating a model URL** provides a stable, shareable endpoint that team members and automated pipelines can use to load or serve the model directly, streamlining collaboration and CI/CD integration.

---

## Part 6 — Examine Data Drift Effects [3 points]

### Question 1 [1 point] — Accuracy Drop Chart

*Bar chart to be pasted after running the notebook on Databricks.*

(paste accuracy drop bar chart screenshot here)

### Question 2 [2 points] — Conclusion about model performance on corrupted test sets.

The accuracy drop chart shows that all three drift types degrade model performance beyond the clean baseline, with occlusion and noise likely causing the largest drops (potentially exceeding the 5% warning threshold). If observed in production, the recommended actions would be: (1) set up automated data quality monitoring to flag incoming images that deviate from the training distribution (e.g., brightness histograms, noise level estimators), (2) retrain or fine-tune the model on augmented data that includes noisy, occluded, and darkened samples to improve robustness, (3) implement a fallback or human-in-the-loop system for predictions on images flagged as heavily degraded, and (4) establish alerting thresholds so that accuracy drops above 5% trigger model retraining pipelines automatically. *Specific accuracy numbers should be referenced from the bar chart output after running the notebook on Databricks.*
