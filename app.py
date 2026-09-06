import re
import joblib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from google import genai

from sklearn.metrics import ConfusionMatrixDisplay


# ===============================
# Page Configuration
# ===============================

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide"
)


# ===============================
# Load Model
# ===============================

@st.cache_resource
def load_model():
    model = joblib.load("model/fake_news_model.pkl")
    vectorizer = joblib.load("model/vectorizer.pkl")
    return model, vectorizer


model, vectorizer = load_model()


# ===============================
# Gemini AI
# ===============================

gemini_api_key = os.environ.get("GEMINI_API_KEY")

if gemini_api_key:
    gemini_client = genai.Client(api_key=gemini_api_key)
else:
    gemini_client = None


def get_gemini_explanation(news, prediction_label):

    if gemini_client is None:
        return "Gemini AI is not configured."

    prompt = f"""
You are an AI assistant helping with a Fake News Detection project.

The Machine Learning model predicted this article as: {prediction_label}

News article:
{news}

Explain the prediction in simple English.

Give:
1. A short explanation of why the article may look {prediction_label}.
2. Important points or warning signs in the article.
3. A clear reminder that AI analysis is not a substitute for checking reliable sources.

Do not claim that the article is definitely true or definitely false.
Keep the answer concise and easy to understand.
"""

    try:

        response = gemini_client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )

        return response.text

    except Exception as e:

        return f"Gemini AI error: {e}"


# ===============================
# Text Cleaning
# ===============================

def clean_text(text):

    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z ]', '', text)

    return text


# ===============================
# Custom CSS
# ===============================

st.markdown("""
<style>

.stApp {
    background-color: #F5F7FA;
}

[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
}

[data-testid="stSidebar"] * {
    color: #222222 !important;
}

/* Main page text color */

[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3,
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] label {
    color: #222222 !important;
}

.main-title {
    text-align: center;
    font-size: 52px;
    font-weight: 800;
    color: #0E76A8 !important;
    margin-bottom: 5px;
}

.sub-title {
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    color: #222222 !important;
}

.desc {
    text-align: center;
    font-size: 19px;
    color: #444444 !important;
    margin-bottom: 25px;
}

.stButton > button {
    width: 100%;
    height: 55px;
    border-radius: 12px;
    font-size: 18px;
    font-weight: bold;
}

textarea {
    font-size: 18px !important;
}

.result-box {
    padding: 20px;
    border-radius: 15px;
    margin-top: 15px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)


# ===============================
# Header
# ===============================

st.markdown(
    '<div class="main-title">📰 Fake News Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">🤖 AI & Machine Learning Project</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="desc">
    Detect whether a news article is <b>REAL</b> or <b>FAKE</b>
    using Machine Learning.
    </div>
    """,
    unsafe_allow_html=True
)


# ===============================
# Sidebar
# ===============================

st.sidebar.title("📌 Project Information")

st.sidebar.success("Machine Learning Project")

st.sidebar.markdown("""
### 📄 Project

**Fake News Detection Using Machine Learning**

---

### 🤖 Algorithm

**Passive Aggressive Classifier**

---

### 🔤 Feature Extraction

**TF-IDF Vectorization**

---

### 🎯 Model Performance

**Accuracy: 99.58%**

**Precision: 99.57%**

**Recall: 99.58%**

**F1-Score: 99.58%**

---

### 🛠 Technologies

✅ Python  
✅ Pandas  
✅ Scikit-Learn  
✅ Streamlit  
✅ Joblib  
✅ Gemini AI

---

### 👨‍💻 Developer

**Akash Kumar Mandal**
""")


# ===============================
# News Input
# ===============================

st.subheader("📝 Enter News Article")

news = st.text_area(
    "Paste the complete news article below:",
    height=250,
    placeholder="Paste complete news article here...",
    key="news_text"
)


# ===============================
# Buttons
# ===============================

def clear_news():
    st.session_state.news_text = ""


col1, col2 = st.columns(2)

with col1:
    check = st.button("🔍 Check News")

with col2:
    st.button("🗑️ Clear", on_click=clear_news)


# ===============================
# Prediction
# ===============================

if check:

    if news.strip() == "":

        st.warning("⚠️ Please enter a news article.")

    else:

        with st.spinner("🔎 Analyzing News..."):

            cleaned_news = clean_text(news)

        vector = vectorizer.transform([cleaned_news])

        prediction = model.predict(vector)

        st.divider()

        st.subheader("📊 Prediction Result")


        # ===============================
        # REAL NEWS
        # ===============================

        if prediction[0] == 1:

            st.success("✅ REAL NEWS")

            st.markdown(
                '<div class="result-box">🟢 The article is predicted as REAL.</div>',
                unsafe_allow_html=True
            )

            prediction_label = "REAL"


        # ===============================
        # FAKE NEWS
        # ===============================

        else:

            st.error("❌ FAKE NEWS")

            st.markdown(
                '<div class="result-box">🔴 The article is predicted as FAKE.</div>',
                unsafe_allow_html=True
            )

            prediction_label = "FAKE"


        # ===============================
        # Prediction Strength
        # ===============================

        try:

            score = model.decision_function(vector)[0]

            import math

            strength = 1 / (1 + math.exp(-abs(score)))

            strength = strength * 100

            st.subheader("📈 Prediction Strength")

            st.progress(int(strength))

            st.write(
                f"**Prediction Strength: {strength:.2f}%**"
            )

            st.caption(
                "Note: This is prediction strength, not a true probability."
            )

        except Exception:

            st.info("Prediction strength is not available.")


        # ===============================
        # Gemini AI Analysis
        # ===============================

        st.divider()

        st.subheader("🤖 Gemini AI Analysis")

        with st.spinner("🤖 Gemini is analyzing the article..."):

            explanation = get_gemini_explanation(
                news,
                prediction_label
            )

        st.write(explanation)


# ==========================================================
# CSV BATCH PREDICTION
# ==========================================================

st.divider()

st.subheader("📂 Batch News Prediction")

st.write(
    "Upload a CSV file containing news articles and predict "
    "multiple articles at once."
)

uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    try:

        df = pd.read_csv(uploaded_file)

        st.write("### 📄 Uploaded Data")

        st.dataframe(
            df.head(),
            use_container_width=True
        )


        # Check for required column

        if "news" not in df.columns:

            st.error(
                "❌ CSV must contain a column named 'news'."
            )

        else:

            # Remove empty values

            df["news"] = df["news"].fillna("")


            # Clean text

            cleaned_batch_news = df["news"].astype(str).apply(clean_text)


            # Convert news into TF-IDF vectors

            news_vectors = vectorizer.transform(
                cleaned_batch_news
            )


            # Make predictions

            predictions = model.predict(news_vectors)


            # Convert prediction to labels

            df["Prediction"] = [
                "REAL" if prediction == 1 else "FAKE"
                for prediction in predictions
            ]


            st.success(
                f"✅ {len(df)} news articles analyzed successfully!"
            )

            st.write("### 📊 Prediction Results")

            st.dataframe(
                df,
                use_container_width=True
            )


            # Convert dataframe to CSV

            csv_data = df.to_csv(
                index=False
            ).encode("utf-8")


            # Download button

            st.download_button(
                label="⬇️ Download Results CSV",
                data=csv_data,
                file_name="fake_news_predictions.csv",
                mime="text/csv"
            )

    except Exception as e:

        st.error(
            f"❌ Error reading CSV file: {e}"
        )


# ==========================================================
# MODEL PERFORMANCE DASHBOARD
# ==========================================================

st.divider()

st.subheader("📊 Model Performance Dashboard")

st.write(
    "Performance of the Passive Aggressive Classifier "
    "on the test dataset."
)


# ===============================
# Metrics
# ===============================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Accuracy",
        "99.58%"
    )

with col2:

    st.metric(
        "Precision",
        "99.57%"
    )

with col3:

    st.metric(
        "Recall",
        "99.58%"
    )

with col4:

    st.metric(
        "F1-Score",
        "99.58%"
    )


# ===============================
# Confusion Matrix
# ===============================

st.subheader("🔲 Confusion Matrix")

cm = np.array([
    [4670, 26],
    [12, 4272]
])

fig, ax = plt.subplots()

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["FAKE", "REAL"]
)

disp.plot(
    ax=ax,
    values_format="d"
)

ax.set_title("Fake News Detection - Confusion Matrix")

st.pyplot(fig)


# ===============================
# Explanation
# ===============================

st.info(
    """
    **Confusion Matrix Explanation**

    • 4670 FAKE news correctly classified as FAKE

    • 4272 REAL news correctly classified as REAL

    • 26 FAKE news incorrectly classified as REAL

    • 12 REAL news incorrectly classified as FAKE
    """
)


# ===============================
# Model Comparison
# ===============================

st.subheader("🏆 Model Comparison")

comparison_data = pd.DataFrame({

    "Model": [
        "Logistic Regression",
        "Naive Bayes",
        "Passive Aggressive"
    ],

    "Accuracy": [
        "98.57%",
        "93.72%",
        "99.58%"
    ],

    "Precision": [
        "98.19%",
        "93.68%",
        "99.57%"
    ],

    "Recall": [
        "98.83%",
        "93.11%",
        "99.58%"
    ],

    "F1-Score": [
        "98.51%",
        "93.40%",
        "99.58%"
    ]
})

st.dataframe(
    comparison_data,
    use_container_width=True,
    hide_index=True
)


# ===============================
# Footer
# ===============================

st.divider()

st.markdown(
    """
    <center>
    <h4>🚀 Fake News Detection System</h4>
    <p>Developed by Akash Kumar Mandal</p>
    <p>AI & Machine Learning Project</p>
    </center>
    """,
    unsafe_allow_html=True
)
