import pandas as pd

# Create a small supervised ML dataset — Student Exam Performance
data = {
    'StudyHours':    [1.5, 3.0, 4.5, 2.0, 5.0, 6.5, 3.5, 7.0, 2.5, 8.0],
    'AttendancePct': [60,  75,  85,  50,  90,  95,  70,  98,  55,  100],
    'PrevScore':     [45,  60,  72,  38,  80,  88,  65,  92,  42,  97 ],
    'Passed':        [0,   0,   1,   0,   1,   1,   1,   1,   0,   1  ],   # target
}

df = pd.DataFrame(data)

print("Supervised ML Dataset — Student Exam Performance:")
print(df)
print(f"\nShape: {df.shape}  ({df.shape[0]} samples, {df.shape[1]} columns)")

# Separate features (X) and target (y)
X = df.drop(columns=['Passed'])   # all columns except target
y = df['Passed']                  # target column only

print("\nFeatures X:")
print(X)
print(f"\nFeature columns : {X.columns.tolist()}")
print(f"X shape         : {X.shape}")

print("\nTarget y (Passed):")
print(y.values)
print(f"y shape         : {y.shape}")
print(f"Classes         : {y.unique().tolist()}  (0 = Failed, 1 = Passed)")
