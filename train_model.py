import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Create model folder
os.makedirs("model", exist_ok=True)

# Load datasets
fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

# Labels
fake["label"] = 0
true["label"] = 1

# Merge and shuffle
data = pd.concat([fake, true], ignore_index=True)
data = data.sample(frac=1, random_state=42)

# Features and labels
X = data["text"]
y = data["label"]

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X = vectorizer.fit_transform(X)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = PassiveAggressiveClassifier(max_iter=50, random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("=" * 60)
print("Accuracy :", accuracy_score(y_test, y_pred))
print("=" * 60)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# Save Model
joblib.dump(model, "model/fake_news_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("\n✅ Model trained and saved successfully!")

# -----------------------------
# DEBUG TEST
# -----------------------------
print("\n========== MODEL TEST ==========")

real_article = true.iloc[0]["text"]
fake_article = fake.iloc[0]["text"]

real_pred = model.predict(vectorizer.transform([real_article]))
fake_pred = model.predict(vectorizer.transform([fake_article]))

print("Real Article Prediction :", real_pred)
print("Fake Article Prediction :", fake_pred)