import pandas as pd
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# -----------------------------
# 1. Load Dataset
# -----------------------------
# CSV file must contain:
# text,label
# "Your account is blocked click here",phishing
# "Meeting today at 10 AM",safe

data = pd.read_csv("emails.csv")

print("Dataset Loaded Successfully")
print(data.head())


# -----------------------------
# 2. Clean Email Text
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " URL ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


data["clean_text"] = data["text"].apply(clean_text)


# -----------------------------
# 3. Split Data
# -----------------------------
X = data["clean_text"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -----------------------------
# 4. Convert Text to Numbers
# -----------------------------
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)

X_train_vector = vectorizer.fit_transform(X_train)
X_test_vector = vectorizer.transform(X_test)


# -----------------------------
# 5. Train Model
# -----------------------------
model = LogisticRegression()
model.fit(X_train_vector, y_train)


# -----------------------------
# 6. Test Model
# -----------------------------
y_pred = model.predict(X_test_vector)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# -----------------------------
# 7. Save Model
# -----------------------------
joblib.dump(model, "phishing_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\nModel saved successfully!")


# -----------------------------
# 8. Predict New Email
# -----------------------------
def predict_email(email_text):
    cleaned = clean_text(email_text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    return prediction


while True:
    email = input("\nEnter email text to check or type exit: ")

    if email.lower() == "exit":
        break

    result = predict_email(email)

    if result.lower() == "phishing":
        print("Result: ⚠️ Phishing Email")
    else:
        print("Result: ✅ Safe Email")
