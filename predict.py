import re
import joblib

# Load model and vectorizer
model = joblib.load("model/fake_news_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\\S+', '', text)
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

print("=" * 60)
print("📰 Fake News Detection System")
print("=" * 60)

news = input("\nPaste the complete news article:\n\n")

news = clean_text(news)

news_vector = vectorizer.transform([news])

prediction = model.predict(news_vector)

print("\nRaw Prediction:", prediction)

if prediction[0] == 0:
    print("\n❌ Prediction : FAKE NEWS")
else:
    print("\n✅ Prediction : REAL NEWS")