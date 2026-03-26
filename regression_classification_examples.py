import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt

np.random.seed(42)

# ══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 1 — REGRESSION: Predicting House Price
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("REGRESSION EXAMPLE — House Price Prediction")
print("=" * 60)
print("  Input  : Area (sqft), Bedrooms, Age of house")
print("  Output : Price (₹) — continuous numeric value")
print("  Use case: Real estate portals (e.g., 99acres, Zillow)")
print()

# Synthetic dataset
n = 200
area      = np.random.randint(500, 3000, n)
bedrooms  = np.random.randint(1, 6, n)
age       = np.random.randint(1, 40, n)
price     = (area * 3000) + (bedrooms * 200000) - (age * 15000) + \
            np.random.normal(0, 100000, n)

df_reg = pd.DataFrame({'Area': area, 'Bedrooms': bedrooms, 'Age': age, 'Price': price})

X_r = df_reg[['Area', 'Bedrooms', 'Age']]
y_r = df_reg['Price']

X_tr, X_te, y_tr, y_te = train_test_split(X_r, y_r, test_size=0.2, random_state=42)

reg = LinearRegression()
reg.fit(X_tr, y_tr)
y_pred_r = reg.predict(X_te)

mse  = mean_squared_error(y_te, y_pred_r)
rmse = np.sqrt(mse)

print(f"  Model      : Linear Regression")
print(f"  Train size : {len(X_tr)} | Test size : {len(X_te)}")
print(f"  MSE        : {mse:,.0f}")
print(f"  RMSE       : ₹{rmse:,.0f}")

# Plot
plt.figure(figsize=(7, 4))
plt.scatter(y_te, y_pred_r, alpha=0.6, color='steelblue', edgecolors='white', s=40)
plt.plot([y_te.min(), y_te.max()], [y_te.min(), y_te.max()], 'r--', linewidth=1.5)
plt.title('Regression — Actual vs Predicted House Price', fontweight='bold')
plt.xlabel('Actual Price (₹)')
plt.ylabel('Predicted Price (₹)')
plt.tight_layout()
plt.savefig('regression_example.png', dpi=150)
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 2 — CLASSIFICATION: Email Spam Detection
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("CLASSIFICATION EXAMPLE — Email Spam Detection")
print("=" * 60)
print("  Input  : Word count, Link count, Caps ratio, Sender score")
print("  Output : Spam label (0 = Not Spam, 1 = Spam) — discrete")
print("  Use case: Gmail, Outlook spam filters")
print()

# Synthetic dataset
word_count  = np.random.randint(10, 500, n)
link_count  = np.random.randint(0, 20, n)
caps_ratio  = np.random.uniform(0, 1, n)
sender_score = np.random.uniform(0, 1, n)

# Spam if many links, high caps, or low sender score
spam = ((link_count > 8) | (caps_ratio > 0.7) | (sender_score < 0.3)).astype(int)

df_clf = pd.DataFrame({
    'WordCount': word_count, 'LinkCount': link_count,
    'CapsRatio': caps_ratio, 'SenderScore': sender_score,
    'Spam': spam
})

X_c = df_clf[['WordCount', 'LinkCount', 'CapsRatio', 'SenderScore']]
y_c = df_clf['Spam']

X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X_c, y_c, test_size=0.2, random_state=42)

clf = LogisticRegression(max_iter=500)
clf.fit(X_tr2, y_tr2)
y_pred_c = clf.predict(X_te2)

acc = accuracy_score(y_te2, y_pred_c)

print(f"  Model      : Logistic Regression")
print(f"  Train size : {len(X_tr2)} | Test size : {len(X_te2)}")
print(f"  Accuracy   : {acc * 100:.2f}%")
print(f"  Spam emails in test: {y_te2.sum()} / {len(y_te2)}")
