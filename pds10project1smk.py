import pandas as pd
import numpy as np
from faker import Faker
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Initialize Faker for generating random names
fake = Faker()

# Generate dataset
num_students = 1000
data = {
    "Student Name": [fake.name() for _ in range(num_students)],
    "Bangla": np.random.randint(30, 101, num_students),  # Marks between 30-100
    "English": np.random.randint(30, 101, num_students),
    "Math": np.random.randint(30, 101, num_students),
    "ICT": np.random.randint(30, 101, num_students)
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Define a target variable: Pass (1) or Fail (0) based on an average threshold (50%)
df["Pass"] = (df[["Bangla", "English", "Math", "ICT"]].mean(axis=1) >= 50).astype(int)

# Split dataset into training (700) and testing (300)
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["Pass"])

# Prepare features and labels
X_train = train_df[["Bangla", "English", "Math", "ICT"]]
y_train = train_df["Pass"]
X_test = test_df[["Bangla", "English", "Math", "ICT"]]
y_test = test_df["Pass"]

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Save dataset to CSV
csv_path = r"student_dataset.csv"
df.to_csv(csv_path, index=False)

print(accuracy*100,"%")