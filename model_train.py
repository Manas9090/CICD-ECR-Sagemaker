import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

# Load your processed dataset
df = pd.read_csv('processed_data.csv')

# Split into X and y
X = df.drop(columns=['loan_default'])
y = df['loan_default']

# Train-test split (optional but good practice)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(clf, "model/model.pkl")
print("Model trained and saved to model/model.pkl")
