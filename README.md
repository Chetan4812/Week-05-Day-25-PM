# Week-05-Day-25-PM

---

# Part A — Concept Application (40%)

### Problem Identification
*   Classify 5 given scenarios as Supervised / Unsupervised / Reinforcement Learning
*   Justify each classification <br>


### Regression vs Classification
*   Identify problem type based on target variable
*   Justify using target column data type and range <br>

| Scenario | ML Type | Justification |
| :--- | :--- | :--- |
| Spam detection using labelled emails | Supervised → Classification | Uses **labeled examples** to categorize data into distinct classes (Spam vs. Not Spam). |
| Grouping customers by purchasing behaviour (no labels) | Unsupervised → Clustering | Groups data points based on **inherent similarities** without pre-defined category labels. |
| Robot learns to walk via rewards | Reinforcement Learning | Learns optimal behavior through **trial and error** to maximize a reward signal. |
| Predicting house prices from features | Supervised → Regression | Predicts a **continuous numerical value** based on historical input-output pairs. |
| Recommending products from browsing history | Unsupervised → Collaborative Filtering | Identifies **patterns and associations** between different users or items based on shared behavior. |


### Dataset Understanding
*   Create a small Pandas DataFrame for supervised ML (Student Exam Performance)
*   Separate features (X) and target (y) <br>
[Solution](dataset_features_target.py)

---

## Part B — Stretch Problem (30%)

*   One complete Regression example — House Price Prediction (input, output, use case, model, RMSE)
*   One complete Classification example — Email Spam Detection (input, output, use case, model, accuracy) <br>
[Solution](regression_classification_examples.py)

## Example Summary

**Regression — House Price Prediction**
*   **Input:** Area (sqft), Bedrooms, Age of house
*   **Output:** Price (₹) — continuous numeric value
*   **Use Case:** Real estate portals (e.g., 99acres, MagicBricks)
*   **Model:** Linear Regression
*   **Metric:** RMSE (Root Mean Squared Error)

**Classification — Email Spam Detection**
*   **Input:** Word count, Link count, Caps ratio, Sender score
*   **Output:** Spam label (0 = Not Spam, 1 = Spam) — discrete binary
*   **Use Case:** Gmail / Outlook spam filters
*   **Model:** Logistic Regression
*   **Metric:** Accuracy

---

## Part C — Interview Ready (20%)

**Q1 — What are the types of machine learning?**

### Supervised Learning
Model learns from labelled input-output pairs. Goal: learn a mapping `f(X) → y`.
*   **Regression** — target is continuous: house price, temperature, salary
*   **Classification** — target is discrete: spam/not spam, disease yes/no, fraud/not fraud

### Unsupervised Learning
Model finds patterns in unlabelled data. No ground truth is provided.
*   **Clustering** — K-Means groups similar items: customer segmentation, document grouping
*   **Dimensionality Reduction** — PCA compresses features: visualisation, noise removal
*   **Anomaly Detection** — identifies outliers: fraud detection, network intrusion

### Reinforcement Learning
An agent learns by taking actions in an environment and receiving rewards or penalties.
*   No dataset needed — learns from experience
*   Examples: AlphaGo, self-driving car control, robot arm manipulation

| Type | Labels? | Goal | Example |
| :--- | :--- | :--- | :--- |
| Supervised (Regression) | ✅ Continuous | Predict a number | House price prediction |
| Supervised (Classification) | ✅ Discrete | Predict a class | Spam detection |
| Unsupervised | ❌ None | Discover structure | Customer segmentation |
| Reinforcement | 🏆 Rewards | Maximise reward | AlphaGo, robot navigation |

**Q2 (Coding) — Separate features and target from dataset** <br>
[Solution](separate_features_target.py)

**Q3 — Difference between regression and classification?**

Both are supervised learning tasks but differ in the type of output they produce.

**Regression** predicts a **continuous numeric value** — the output can be any real number.
*   Question answered: *"How much?"* or *"How many?"*
*   Examples: house price, stock price, temperature
*   Metrics: MSE, RMSE, MAE, R²

**Classification** predicts a **discrete class label** — the output belongs to a fixed set of categories.
*   Question answered: *"Which class?"* or *"Is it X?"*
*   Examples: spam/not spam, survived/not survived, digit recognition
*   Metrics: Accuracy, Precision, Recall, F1-score, AUC-ROC

| Aspect | Regression | Classification |
| :--- | :--- | :--- |
| Output | Continuous float | Discrete label |
| Loss function | MSE / MAE | Cross-Entropy |
| Example target | ₹4,500,000 | 1 (Spam) |
| Algorithm | Linear Regression | Logistic Regression |

---

## Part D — AI-Augmented Task (10%)

### 1. Prompt AI:
*"Explain types of machine learning with real-world examples."*

### 2. Document output

[AI Output](AI_output.md) for the above prompt

### 3. Evaluate Correctness

*   **Supervised Learning:** Correctly defined as learning from labelled data. Regression and Classification sub-types are correctly distinguished by output type (continuous vs discrete). Code examples are idiomatic and runnable.
*   **Unsupervised Learning:** Correctly defined as finding structure in unlabelled data. K-Means clustering example is appropriate and the code is correct — `n_init=10` is good practice to avoid local minima warnings.
*   **Reinforcement Learning:** Correctly defined using agent/environment/reward terminology. Real-world examples (AlphaGo, self-driving cars) are accurate and well-chosen.
*   **Summary table:** All four rows (Regression, Classification, Unsupervised, Reinforcement) are correctly categorised.

> **One improvement made:**
> The AI gave a generic K-Means example with arbitrary coordinates. In our [regression_classification_examples.py](regression_classification_examples.py) we built a meaningful synthetic dataset with interpretable features (Area, Bedrooms, Age for house price; WordCount, LinkCount, CapsRatio for spam) — making the examples domain-grounded rather than just numerical.

### Runnability

All AI-provided code runs without modification given `pandas`, `numpy`, and `scikit-learn` are installed (`pip install pandas scikit-learn`).
