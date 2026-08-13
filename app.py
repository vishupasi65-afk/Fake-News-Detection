import joblib
import streamlit as st

@st.cache_resource
def load_model():
    model = joblib.load("model/fake_news_model.pkl")
    vectorizer = joblib.load("model/vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# ===============================
# Page Configuration
# ===============================

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide"
)

# ===============================
# Custom CSS
# ===============================

st.markdown("""
<style>

.stApp{
    background:#F5F7FA;
}

.main-title{
    text-align:center;
    font-size:52px;
    font-weight:800;
    color:#0E76A8;
}

.sub-title{
    text-align:center;
    font-size:30px;
    font-weight:bold;
    color:#222;
}

.desc{
    text-align:center;
    font-size:20px;
    color:#444;
}

.stButton>button{
    width:100%;
    height:55px;
    border-radius:12px;
    font-size:18px;
    font-weight:bold;
}

textarea{
    font-size:18px !important;
}

</style>
""", unsafe_allow_html=True)

# ===============================
# Header
# ===============================

st.markdown('<div class="main-title">📰 Fake News Detection System</div>', unsafe_allow_html=True)

st.markdown('<div class="sub-title">🤖 AI & Machine Learning Project</div>', unsafe_allow_html=True)

st.markdown("""
<div class="desc">
Detect whether a news article is <b>REAL</b> or <b>FAKE</b> using Machine Learning.
</div>
""", unsafe_allow_html=True)

st.write("")

# ===============================
# Sidebar
# ===============================

st.sidebar.title("📌 Project Information")

st.sidebar.success("Machine Learning Project")

st.sidebar.markdown("""

### 📄 Project

Fake News Detection Using Machine Learning

---

### 🤖 Algorithm

Passive Aggressive Classifier

---

### 🎯 Accuracy

99%

---

### 🛠 Technologies

✅ Python

✅ Pandas

✅ Scikit-Learn

✅ Streamlit

---

### 👨‍💻 Developer

Akash Kumar Mandal

Data Science Intern

Pinnacle Labs

""")

# ===============================
# Text Area
# ===============================

if "news_text" not in st.session_state:
    st.session_state.news_text = ""

news = st.text_area(
    "📝 Enter News Article",
    key="news_text",
    height=250,
    placeholder="Paste complete news article here..."
)

# ===============================
# Buttons
# ===============================

col1, col2 = st.columns(2)

with col1:

    check = st.button("🔍 Check News")


# ===============================
# Prediction
# ===============================

if check:

    if news.strip() == "":

        st.warning("⚠ Please enter a news article.")

    else:

        with st.spinner("Analyzing News..."):

            vector = vectorizer.transform([news.strip()])

            prediction = model.predict(vector)

        st.divider()

        st.subheader("📊 Prediction Result")

        if prediction[0] == 1:

            st.success("✅ REAL NEWS")

        else:

            st.error("❌ FAKE NEWS")

        try:

            probability = model.predict_proba(vector)

            confidence = max(probability[0]) * 100

            st.subheader("📈 Confidence Score")

            st.progress(int(confidence))

            st.success(f"{confidence:.2f}%")

        except:

            st.info("Confidence score is not available for Passive Aggressive Classifier.")

# ===============================
# Footer
# ===============================

st.divider()

st.markdown("""
<center>

### 🚀 Developed by Akash Kumar Mandal

Data Science Internship Project

Pinnacle Labs • 2026

</center>
""", unsafe_allow_html=True)
