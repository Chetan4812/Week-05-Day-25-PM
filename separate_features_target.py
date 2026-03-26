import pandas as pd

# Sample dataset — Diabetes risk prediction
data = {
    'Age':       [25, 45, 34, 52, 28, 61, 38, 47],
    'BMI':       [22.5, 30.1, 27.8, 35.2, 21.0, 38.6, 24.3, 31.5],
    'Glucose':   [85, 120, 105, 145, 78, 160, 92, 130],
    'BloodPres': [70, 85, 78, 92, 68, 95, 74, 88],
    'Diabetic':  [0,  1,   0,  1,  0,  1,   0,  1],  # target
}

df = pd.DataFrame(data)
print("Full Dataset:")
print(df)

# Separate features (X) and target (y)
X = df.drop(columns=['Diabetic'])   # features: everything except target
y = df['Diabetic']                  # target column

print(f"\nFeatures X  ({X.shape[0]} rows × {X.shape[1]} cols):")
print(X)

print(f"\nTarget y  ({y.shape[0]} values):")
print(y.values)
print(f"\nClasses : {y.unique().tolist()}  (0 = Not Diabetic, 1 = Diabetic)")
