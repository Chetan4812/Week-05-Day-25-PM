### Types of Machine Learning with Real-World Examples

Machine learning algorithms are broadly categorised into three types based on how they learn from data.

---

#### 1. Supervised Learning

The model is trained on **labelled data** — each input has a corresponding known output. The algorithm learns to map inputs to outputs and generalises to unseen examples.

**Sub-types:**
- **Regression** — output is a continuous value
- **Classification** — output is a discrete class label

```python
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

# Regression example — predicting house price
data_reg = {
    'Area': [800, 1200, 950, 1500, 700],
    'Rooms': [2, 3, 2, 4, 1],
    'Price': [4500000, 7200000, 5100000, 9800000, 3200000],
}
df_r = pd.DataFrame(data_reg)
X_r, y_r = df_r[['Area', 'Rooms']], df_r['Price']
X_tr, X_te, y_tr, y_te = train_test_split(X_r, y_r, test_size=0.2, random_state=42)
reg = LinearRegression().fit(X_tr, y_tr)
print(f"Regression RMSE: {mean_squared_error(y_te, reg.predict(X_te))**0.5:.0f}")

# Classification example — loan approval
data_clf = {
    'Income': [35000, 80000, 52000, 21000, 67000],
    'Credit': [700, 820, 650, 550, 780],
    'Approved': [1, 1, 0, 0, 1],
}
df_c = pd.DataFrame(data_clf)
X_c, y_c = df_c[['Income', 'Credit']], df_c['Approved']
X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X_c, y_c, test_size=0.2, random_state=42)
clf = LogisticRegression().fit(X_tr2, y_tr2)
print(f"Classification Accuracy: {accuracy_score(y_te2, clf.predict(X_te2))*100:.1f}%")
```

**Real-world examples:**
- Email spam detection (classification)
- Stock price prediction (regression)
- Medical diagnosis — cancer malignant/benign (classification)
- Weather temperature forecasting (regression)

---

#### 2. Unsupervised Learning

The model is given **unlabelled data** and must discover hidden patterns or structure on its own. No ground truth is provided.

```python
from sklearn.cluster import KMeans
import numpy as np

# Customer segmentation example
customer_data = np.array([
    [25, 30000], [45, 80000], [35, 55000],
    [22, 25000], [50, 90000], [30, 45000],
])  # [Age, Annual Income]

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
labels = kmeans.fit_predict(customer_data)
print(f"Cluster labels: {labels}")
# Groups customers into 'low income' and 'high income' segments
```

**Real-world examples:**
- Customer segmentation for marketing campaigns
- Anomaly detection in network traffic
- Topic modelling in news articles
- Dimensionality reduction with PCA

---

#### 3. Reinforcement Learning

An **agent** learns by interacting with an **environment**. It takes actions and receives rewards (positive) or penalties (negative). The goal is to maximise cumulative reward over time.

No dataset is needed — the agent learns from experience.

**Real-world examples:**
- AlphaGo / AlphaZero (board game AI)
- Self-driving car lane keeping
- Robot arm learning to pick objects
- Personalised recommendation systems that update in real time

> *Source: Scikit-learn documentation — scikit-learn.org*

---

### Summary Table

| Type | Labels? | Goal | Algorithm Examples |
| :--- | :--- | :--- | :--- |
| Supervised — Regression | ✅ Continuous | Predict a number | Linear Regression, SVR |
| Supervised — Classification | ✅ Discrete | Predict a class | Logistic Regression, SVM, Decision Tree |
| Unsupervised | ❌ None | Find structure | K-Means, PCA, DBSCAN |
| Reinforcement | 🏆 Rewards | Maximise reward | Q-Learning, PPO, A3C |
